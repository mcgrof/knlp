"""Micro-benchmark: solve_triangular vs Neumann for the smd input-conditioned
affine chunk kernel (run_trellis_memory_chunked, input_gate path).

Answers one question: on a real GPU, is the all-slots Neumann iteration (matmul,
no cuSOLVER, no [B,H,M,C,C] fold) faster than torch.linalg.solve_triangular for
the smd write-solve -- and is it bit-exact (forward AND gradients)?

Both solvers are exact in principle (the per-slot unit-lower-triangular system is
nilpotent, so C-1 Neumann sweeps reproduce the solve). This checks that and times
forward and forward+backward for: solve-eager, neumann-eager, neumann-compiled.

    python3 scripts/bench_ic_solver.py --device cuda --dtype fp32 --compile 1
"""

import argparse
import time

import torch

from trellis_lm.activations import ln_silu
from trellis_lm.trellis_memory import run_trellis_memory_chunked


def make_inputs(B, H, T, D, M, device, dt, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dt)

    write = r(B, H, T, D)
    read = r(B, H, T, D)  # M_q read is [B,H,T,D]
    alpha = r(B, H, T, M)
    beta = torch.sigmoid(r(B, H, T, 1)) * 0.5 + 0.5  # (0.5,1) decay, per head
    gamma = torch.full((H,), 0.05, device=device, dtype=dt)
    input_gate = torch.sigmoid(r(B, H, T, M)) * 2.0  # (0,2), a~1 init
    return write, read, alpha, beta, gamma, input_gate


def run(fn, write, read, alpha, beta, gamma, input_gate, chunk, solver):
    return fn(
        write,
        read,
        alpha,
        beta,
        gamma,
        ln_silu,
        "M_q",
        chunk,
        input_gate=input_gate,
        ic_solver=solver,
    )


def timeit(callable_, n, warmup, device):
    for _ in range(warmup):
        callable_()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        callable_()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp64", "bf16"])
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--H", type=int, default=8)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--D", type=int, default=64)
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--compile", type=int, default=1)
    args = p.parse_args()

    dt = {"fp32": torch.float32, "fp64": torch.float64, "bf16": torch.bfloat16}[
        args.dtype
    ]
    dev = args.device
    print(
        f"device={dev} dtype={args.dtype} B={args.B} H={args.H} T={args.T} "
        f"D={args.D} M={args.M} chunk={args.chunk} "
        f"({args.T // args.chunk} chunks)",
        flush=True,
    )
    if dev == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    inp = make_inputs(args.B, args.H, args.T, args.D, args.M, dev, dt)

    # ---- exactness: forward + gradients (alpha as the leaf) ----
    def fwd(solver, requires_grad=False):
        w, r, a, b, g, ig = inp
        a = a.clone().requires_grad_(requires_grad)
        y = run(run_trellis_memory_chunked, w, r, a, b, g, ig, args.chunk, solver)
        return y, a

    y_s, _ = fwd("solve")
    y_n, _ = fwd("neumann")
    ferr = (y_s - y_n).abs().max().item()
    frel = ferr / y_s.abs().max().item()
    print(f"[exact] forward max_abs_err={ferr:.3e} rel={frel:.3e}", flush=True)

    ys, a_s = fwd("solve", True)
    ys.sum().backward()
    gs = a_s.grad.clone()
    yn, a_n = fwd("neumann", True)
    yn.sum().backward()
    gn = a_n.grad.clone()
    gerr = (gs - gn).abs().max().item()
    grel = gerr / gs.abs().max().item()
    print(f"[exact] d/dalpha  max_abs_err={gerr:.3e} rel={grel:.3e}", flush=True)

    # ---- timing: forward, and forward+backward ----
    w, r, a, b, g, ig = inp

    def fwd_only(solver, f):
        return lambda: run(f, w, r, a.detach(), b, g, ig, args.chunk, solver)

    def fwd_bwd(solver, f):
        def step():
            aa = a.detach().clone().requires_grad_(True)
            y = run(f, w, r, aa, b, g, ig, args.chunk, solver)
            y.sum().backward()

        return step

    res = {}
    res["solve fwd"] = timeit(fwd_only("solve", run_trellis_memory_chunked),
                              args.iters, args.warmup, dev)
    res["neumann fwd"] = timeit(fwd_only("neumann", run_trellis_memory_chunked),
                                args.iters, args.warmup, dev)
    res["solve fwd+bwd"] = timeit(fwd_bwd("solve", run_trellis_memory_chunked),
                                  args.iters, args.warmup, dev)
    res["neumann fwd+bwd"] = timeit(fwd_bwd("neumann", run_trellis_memory_chunked),
                                    args.iters, args.warmup, dev)

    if args.compile:
        try:
            cfn = torch.compile(run_trellis_memory_chunked, dynamic=False)
            # exactness of the compiled neumann path
            yc = run(cfn, w, r, a.detach(), b, g, ig, args.chunk, "neumann")
            cerr = (yc - y_n).abs().max().item()
            print(f"[exact] compiled-neumann forward max_abs_err={cerr:.3e}",
                  flush=True)
            res["neumann fwd (compiled)"] = timeit(
                fwd_only("neumann", cfn), args.iters, args.warmup, dev)
            res["neumann fwd+bwd (compiled)"] = timeit(
                fwd_bwd("neumann", cfn), args.iters, args.warmup, dev)
        except Exception as e:  # ROCm/triton may not compile; report + continue
            print(f"[compile] skipped: {e}", flush=True)

    print("\n=== per-call latency (ms), lower is better ===", flush=True)
    base_f = res["solve fwd"]
    base_fb = res["solve fwd+bwd"]
    for k, v in res.items():
        base = base_fb if "+bwd" in k else base_f
        print(f"  {k:32s} {v * 1e3:8.2f} ms   ({base / v:5.2f}x vs solve)",
              flush=True)


if __name__ == "__main__":
    main()
