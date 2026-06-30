"""Triton fused state-evolution: forward A/B + fwd+bwd grad-check against the
fp64-validated PyTorch path. CUDA + triton only; skips elsewhere (monster has
no GPU). fp32 tolerances (the kernel and torch.bmm differ at the fp32 floor).
"""

import torch


def _have():
    try:
        import triton  # noqa: F401

        from trellis_lm.trellis_triton import HAS_TRITON
    except Exception as exc:
        print(f"SKIP (triton import failed: {exc!r})")
        return False
    if not HAS_TRITON:
        print("SKIP (trellis_triton.HAS_TRITON is false)")
        return False
    if not torch.cuda.is_available():
        print("SKIP (torch.cuda.is_available is false)")
        return False
    if getattr(torch.version, "hip", None) is not None:
        print(f"SKIP (HIP build: {torch.version.hip})")
        return False
    return True


def run():
    if not _have():
        return True
    from trellis_lm.trellis_memory import (
        run_trellis_memory_chunked_state_evolution as evo,
        trellis_chunk_decay,
    )
    from trellis_lm.trellis_triton import (
        trellis_state_evolution_triton,
        TrellisStateEvolutionTriton,
    )
    from trellis_lm.activations import identity, ln_silu, silu

    dev = "cuda"
    torch.manual_seed(0)
    ok = True

    # T not a multiple of C exercises time padding. M=48 exercises the repaired
    # pointwise-activation slot padding path used by the Stage-2 H100 rows.
    cases = (
        ("ln_silu", "ln_silu", ln_silu, 64, None, 0.0),
        ("silu", "silu", silu, 48, None, 0.0),
        ("identity", "identity", identity, 48, None, 0.0),
        ("silu_residual", "silu", silu, 48, None, 0.25),
        ("silu_scalar_gate", "silu", silu, 48, "scalar", 0.0),
        ("silu_channel_gate_residual", "silu", silu, 48, "channel", 0.10),
    )
    for label, act_name, phi, M, gate_mode, residual_mix in cases:
        B, H, T, D, C = 2, 4, 67, 64, 16
        write = torch.randn(B, H, T, D, device=dev)
        alpha = torch.randn(B, H, T, M, device=dev)
        beta = torch.sigmoid(torch.randn(B, H, T, 1, device=dev))
        gamma = torch.rand(H, device=dev) + 0.1
        update_gate = None
        if gate_mode == "scalar":
            update_gate = torch.sigmoid(torch.randn(B, H, T, 1, device=dev))
        elif gate_mode == "channel":
            update_gate = torch.sigmoid(torch.randn(B, H, T, M, device=dev))
        P, rmat, _ = trellis_chunk_decay(beta, C)
        nC = P.shape[2]

        # --- forward A/B
        M0_ref, u_ref, _, _, _ = evo(
            write,
            alpha,
            beta,
            gamma,
            phi,
            C,
            P=P,
            rmat=rmat,
            update_gate=update_gate,
            residual_update_mix=residual_mix,
        )
        M0_t, u_t = trellis_state_evolution_triton(
            write,
            alpha,
            P,
            rmat,
            gamma,
            C,
            act_name,
            "none",
            0.0,
            update_gate,
            residual_mix,
        )
        for nm, r, f in (("M0", M0_ref, M0_t), ("u", u_ref, u_t)):
            rel = (r - f).abs().max().item() / (r.abs().max().item() + 1e-12)
            ok &= rel < 1e-4
            result = "PASS" if rel < 1e-4 else "FAIL"
            print(f"{label:26s} fwd {nm:3s} rel {rel:.2e} {result}")

        # --- fwd+bwd grad-check vs the PyTorch state-evolution reference.
        c1 = torch.randn(B, H, nC, M, D, device=dev)
        c2 = torch.randn(B, H, nC, C, M, device=dev)

        def grads(use_triton):
            w = write.clone().requires_grad_()
            a = alpha.clone().requires_grad_()
            Pp = P.clone().requires_grad_()
            rm = rmat.clone().requires_grad_()
            g = gamma.clone().requires_grad_()
            gate = (
                update_gate.clone().requires_grad_()
                if update_gate is not None
                else None
            )
            if use_triton:
                m0, us = TrellisStateEvolutionTriton.apply(
                    w,
                    a,
                    Pp,
                    rm,
                    g,
                    C,
                    act_name,
                    "none",
                    0.0,
                    gate,
                    residual_mix,
                )
            else:
                m0, us, _, _, _ = evo(
                    w,
                    a,
                    None,
                    g,
                    phi,
                    C,
                    P=Pp,
                    rmat=rm,
                    update_gate=gate,
                    residual_update_mix=residual_mix,
                )
            ((m0 * c1).sum() + (us * c2).sum()).backward()
            out = [w.grad, a.grad, Pp.grad, rm.grad, g.grad]
            if gate is not None:
                out.append(gate.grad)
            return out

        ref = grads(False)
        tr = grads(True)
        names = ["write", "alpha", "P", "rmat", "gamma"]
        if update_gate is not None:
            names.append("gate")
        for nm, r, f in zip(names, ref, tr):
            rel = (r - f).abs().max().item() / (r.abs().max().item() + 1e-12)
            ok &= rel < 1e-3
            result = "PASS" if rel < 1e-3 else "FAIL"
            print(f"{label:26s} grad {nm:6s} rel {rel:.2e} {result}")

    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
