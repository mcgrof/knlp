"""End-to-end tok/s for the smd model: does the Neumann kernel's win survive in
the full TrellisLM forward+backward (not just the isolated chunk kernel)?

Builds the science-run smd_diag config (input-conditioned per-slot gate) and
times train steps (forward + backward + optimizer) for ic_solver in {solve,
neumann} with and without torch.compile. Reports tok/s so the kernel-level
speedup can be translated into a real training-throughput number.

    python3 scripts/bench_smd_e2e.py --device cuda --n_slots 32 --steps 40
"""

import argparse
import time

import torch

from trellis_lm.config import TrellisConfig
from trellis_lm.model import TrellisLM


def build(n_slots, ic_solver, dtype):
    return TrellisConfig(
        vocab_size=50257,
        d_model=512,
        n_layers=10,
        n_heads=8,
        d_head=64,
        n_slots=n_slots,
        max_seq_len=2048,
        dtype=dtype,
        chunk_size=16,
        trellis_write_mode="input_conditioned",
        trellis_input_gate_act="sigmoid",
        trellis_input_gate_scope="per_slot",
        write_l2norm=True,
        activation="silu",
        alpha_mode="linear",
        beta_mode="scalar_per_head",
        beta_init=0.5,
        value_readout_act="none",
        output_path="current",
        gamma_init=0.05,
        trellis_layer0_gamma_mult=0.5,
        trellis_ic_solver=ic_solver,
    )


def bench(n_slots, ic_solver, compile_model, device, dt, B, T, steps, warmup):
    torch.manual_seed(0)
    cfg = build(n_slots, ic_solver, {torch.float32: "fp32", torch.bfloat16: "bf16"}[dt])
    model = TrellisLM(cfg).to(device)
    model.train()
    if compile_model:
        model = torch.compile(model, dynamic=False)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    tok = B * T

    def step():
        idx = torch.randint(0, 50257, (B, T), device=device)
        with torch.autocast(device_type=device.type, dtype=dt, enabled=dt != torch.float32):
            _, loss = model(idx, labels=idx, training=True)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_s = time.time() - t0
    return tok * steps / dt_s, dt_s / steps * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--n_slots", type=int, default=32)
    p.add_argument("--B", type=int, default=8)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()
    dev = torch.device(args.device)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    print(f"device={dev} dtype={args.dtype} n_slots={args.n_slots} "
          f"B={args.B} T={args.T} steps={args.steps}", flush=True)
    if dev.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0), flush=True)

    configs = [
        ("solve", False),
        ("neumann", False),
        ("neumann", True),
    ]
    base = None
    print("\n=== end-to-end train step (fwd+bwd+opt) ===", flush=True)
    for solver, comp in configs:
        try:
            tps, ms = bench(args.n_slots, solver, comp, dev, dt,
                            args.B, args.T, args.steps, args.warmup)
        except Exception as e:
            print(f"  {solver:8s} compile={comp}: FAILED {e}", flush=True)
            continue
        if base is None:
            base = tps
        tag = f"{solver}{' +compile' if comp else ''}"
        print(f"  {tag:20s} {tps:9.0f} tok/s  {ms:8.1f} ms/step  "
              f"({tps / base:5.2f}x vs solve)", flush=True)


if __name__ == "__main__":
    main()
