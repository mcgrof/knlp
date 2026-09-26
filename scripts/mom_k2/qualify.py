"""Focused qualification of the newly exercised top-2 and uniform behaviour.

This does not repeat the whole earlier campaign. It tests what is new: the
top-2 packed path against a less-optimised reference of the same policy, the
readout coefficients the forward actually applies, where the router's gradient
comes from under each policy, causality through the packed layout at top-2, and
one real update with a disk save and reload.

Tolerances are calibrated on these shapes before any candidate error is read:
identical reruns give the deterministic floor, and a mathematically equivalent
reordering of the same sums gives the floor two correct implementations can be
separated by.

  python3 qualify.py --out qualify.json
"""
import argparse, copy, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("MATCHED_MICRO_SOURCE", os.path.dirname(HERE)))

CONV, MARGIN = 4, 8.0
FP32_FLOOR, BF16_SLACK = 1e-5, 1.5

# each token names its first and second choice; the second is at half the
# first's logit so the two selected probabilities are never tied
FIXTURES = {
    "skewed_with_an_empty_memory": [
        [(0, 1)] * 60 + [(1, 0)] * 30 + [(2, 0)] * 6,
        [(0, 1)] * 50 + [(1, 0)] * 40 + [(2, 0)] * 6],
    "singleton_segments": [
        [(0, 1)] * 94 + [(2, 0)] + [(3, 0)],
        [(0, 1)] * 93 + [(2, 0)] + [(3, 0)] + [(1, 0)]],
    "segments_below_conv_width": [
        [(0, 1)] * 91 + [(2, 0)] * 2 + [(3, 0)] * 3,
        [(0, 1)] * 90 + [(2, 0)] * 3 + [(3, 0)] * 3],
    "balanced": [[(t % 4, (t + 1) % 4) for t in range(96)],
                 [((t + 2) % 4, (t + 3) % 4) for t in range(96)]],
}


def occupancy(rows, nmem, topk):
    out = []
    for r, row in enumerate(rows):
        per = {b: 0 for b in range(nmem)}
        for pair in row:
            for b in pair[:topk]:
                per[b] += 1
        runs, cur, n = [], None, 0
        for pair in row:
            b = pair[0]
            if b == cur:
                n += 1
            else:
                if cur is not None:
                    runs.append((cur, n))
                cur, n = b, 1
        runs.append((cur, n))
        out.append(dict(row=r, tokens_per_memory=per,
                        empty=[b for b, c in per.items() if c == 0],
                        below_conv_width=sorted(
                            [(b, c) for b, c in per.items() if 0 < c < CONV])))
    return out


def build_layer(modname, cfg, device, topk):
    import torch, importlib
    mod = importlib.import_module(modname)
    importlib.reload(mod)
    torch.manual_seed(4242)
    m = mod.MomAttention(hidden_size=cfg["dim"], head_dim=64,
                         num_heads=cfg["mom_heads"], expand_v=2, mode="chunk",
                         num_memories=cfg["num_memories"], topk=topk,
                         capacity=1.0, shared_mem=True).to(device)
    with torch.no_grad():
        m.gate.weight.zero_()
        for b in range(cfg["num_memories"]):
            m.gate.weight[b, b] = MARGIN
        if getattr(m.gate, "bias", None) is not None:
            m.gate.bias.zero_()
    return m, mod


def make_input(rows, cfg, device):
    import torch
    B, T = len(rows), len(rows[0])
    torch.manual_seed(99)
    x = torch.randn(B, T, cfg["dim"], device=device, dtype=torch.float32) * 0.1
    with torch.no_grad():
        x[:, :, :cfg["num_memories"]] = 0.0
        for r, row in enumerate(rows):
            for t, pair in enumerate(row):
                x[r, t, pair[0]] = 1.0
                if len(pair) > 1:
                    x[r, t, pair[1]] = 0.5
    return x.requires_grad_(True)


def rel(a, b):
    if a is None and b is None:
        return 0.0
    if (a is None) != (b is None):
        return float("inf")
    d = (a - b).abs().max().item()
    return d / max(a.abs().max().item(), b.abs().max().item(), 1e-12)


def run_layer(modname, cfg, rows, device, topk, autocast, seed=7, rowslice=None,
              do_clip=True):
    import torch
    m, mod = build_layer(modname, cfg, device, topk)
    x = make_input(rows, cfg, device)
    xi = x if rowslice is None else x[rowslice]
    torch.manual_seed(seed)
    G = torch.randn(len(rows), len(rows[0]), cfg["dim"], device=device)
    Gi = G if rowslice is None else G[rowslice]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    opt.zero_grad(set_to_none=True)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
           else torch.enable_grad())
    with ctx:
        out = m(xi)
        y = out[0] if isinstance(out, tuple) else out
        loss = (y.float() * Gi).sum()
    loss.backward()
    # gradients before clipping: clipping rescales by the total norm, which is
    # not additive over row groups, so an equivalence control must read these
    raw = {n: (None if p.grad is None else p.grad.detach().float().cpu().clone())
           for n, p in m.named_parameters()}
    clip = (float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
            if do_clip else float("nan"))
    grads = ({n: (None if p.grad is None else p.grad.detach().float().cpu().clone())
              for n, p in m.named_parameters()} if do_clip else raw)
    opt.step()
    mom = {}
    for n, p in m.named_parameters():
        s = opt.state.get(p, {})
        if "exp_avg" in s:
            mom[n + ".exp_avg"] = s["exp_avg"].detach().float().cpu().clone()
            mom[n + ".exp_avg_sq"] = s["exp_avg_sq"].detach().float().cpu().clone()
    res = dict(y=y.detach().float().cpu().clone(), loss=float(loss),
               dx=(x.grad.detach().float().cpu().clone() if x.grad is not None else None),
               clip=clip, grads=grads, raw_grads=raw, moments=mom,
               policy=mod.READOUT_POLICY)
    del m, x, opt
    torch.cuda.empty_cache()
    return res


def at_model_scale(a, b):
    """Largest difference over all tensors, divided by the largest value any of
    them carries. A per-tensor ratio explodes on a parameter that receives
    almost no gradient, which says nothing about the implementation; this does
    not. The per-tensor figure is kept as a diagnostic."""
    scale = max((t.abs().max().item() for t in a.values() if t is not None),
                default=1e-12)
    worst, who, per = 0.0, None, 0.0
    for n, x in a.items():
        y = b.get(n)
        if x is None or y is None:
            continue
        d = (x - y).abs().max().item()
        if d / max(scale, 1e-12) > worst:
            worst, who = d / max(scale, 1e-12), n
        per = max(per, d / max(x.abs().max().item(), 1e-12))
    return worst, who, per, scale


def compare(a, b):
    import torch
    names = sorted(set(a["grads"]) | set(b["grads"]))
    g_scaled, g_who, g_per, g_scale = at_model_scale(a["raw_grads"], b["raw_grads"])
    m_scaled, m_who, m_per, _ = at_model_scale(a["moments"], b["moments"])
    r = dict(y=rel(a["y"], b["y"]), dx=rel(a["dx"], b["dx"]),
             loss=abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-12),
             clip=(abs(a["clip"] - b["clip"]) / max(abs(a["clip"]), 1e-12)
                   if math.isfinite(a["clip"]) and math.isfinite(b["clip"]) else 0.0),
             grad_at_model_scale=g_scaled, grad_worst_parameter=g_who,
             grad_per_tensor_diagnostic=g_per, model_gradient_scale=g_scale,
             moment_at_model_scale=m_scaled, moment_per_tensor_diagnostic=m_per,
             grad_names_match=(sorted(a["grads"]) == sorted(b["grads"])),
             asymmetric_missing=[n for n in names
                                 if (a["grads"].get(n) is None) != (b["grads"].get(n) is None)])
    fin = all(torch.isfinite(t).all().item() for d in (b["grads"], b["moments"])
              for t in d.values() if t is not None)
    r["candidate_finite"] = bool(fin and math.isfinite(b["loss"]))
    r["worst"] = max(r["y"], r["dx"], r["loss"], r["clip"],
                     r["grad_at_model_scale"], r["moment_at_model_scale"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import arms
    import matched_micro_train as mmt
    cfg = mmt.ARMS["mom3"]
    rep = dict(kind="k2_qualification", source_hashes=arms.source_hashes(),
               conv_width=CONV, declared=dict(
                   fp32_floor=FP32_FLOOR, bf16_slack=BF16_SLACK,
                   fp32_rule=("max(equivalent-reordering error, ten times the "
                              "identical-rerun variation, 1e-5)"),
                   bf16_rule=("candidate error against the single-precision "
                              "anchor at most 1.5 times the reference's own, "
                              "plus 1e-5")),
               fixtures={f: occupancy(v, 4, 2) for f, v in FIXTURES.items()},
               sections={}, failures=[])

    # ---- tolerance calibration on these shapes, before any candidate error
    cal = {}
    for policy, ref in (("renormalized", "mom_renorm_ref"),
                        ("uniform", "mom_uniform_ref")):
        rows = FIXTURES["balanced"]
        r1 = run_layer(ref, cfg, rows, a.device, 2, False)
        r2 = run_layer(ref, cfg, rows, a.device, 2, False)
        rerun = compare(r1, r2)["worst"]
        # unclipped, so the two row groups really do sum to the whole batch
        rw = run_layer(ref, cfg, rows, a.device, 2, False, do_clip=False)
        h1 = run_layer(ref, cfg, rows, a.device, 2, False, rowslice=slice(0, 1),
                       do_clip=False)
        h2 = run_layer(ref, cfg, rows, a.device, 2, False, rowslice=slice(1, 2),
                       do_clip=False)
        gscale = max((t.abs().max().item() for t in rw["raw_grads"].values()
                      if t is not None), default=1e-12)
        reord = 0.0
        for n in rw["raw_grads"]:
            if rw["raw_grads"][n] is None:
                continue
            both = h1["raw_grads"][n] + h2["raw_grads"][n]
            reord = max(reord, (rw["raw_grads"][n] - both).abs().max().item() / gscale)
        tol = max(FP32_FLOOR, 10.0 * rerun, reord)
        cal[policy] = dict(identical_rerun=rerun, equivalent_reordering=reord,
                           frozen_fp32_tolerance=tol)
        print("calibration %-13s rerun %.3e  reordering %.3e  -> tolerance %.3e"
              % (policy, rerun, reord, tol))
    rep["sections"]["tolerance_calibration"] = cal

    # ---- optimized top-2 path against its less-optimised reference
    cross = {}
    for policy, ref, cand in (("renormalized", "mom_renorm_ref", "mom_renorm"),
                              ("uniform", "mom_uniform_ref", "mom_uniform")):
        tol = cal[policy]["frozen_fp32_tolerance"]
        for fname, rows in FIXTURES.items():
            key = "%s/%s" % (policy, fname)
            r32 = run_layer(ref, cfg, rows, a.device, 2, False)
            c32 = run_layer(cand, cfg, rows, a.device, 2, False)
            k32 = compare(r32, c32)
            ok32 = (k32["worst"] <= tol and k32["grad_names_match"]
                    and not k32["asymmetric_missing"] and k32["candidate_finite"])
            rb = run_layer(ref, cfg, rows, a.device, 2, True)
            cb = run_layer(cand, cfg, rows, a.device, 2, True)
            ref_err, cand_err = compare(r32, rb)["worst"], compare(r32, cb)["worst"]
            allowed = BF16_SLACK * ref_err + FP32_FLOOR
            okb = cand_err <= allowed
            cross[key] = dict(single=dict(comparison=k32, tolerance=tol, passed=ok32),
                              mixed=dict(reference_error=ref_err,
                                         candidate_error=cand_err,
                                         allowed=allowed, passed=okb))
            if not ok32:
                rep["failures"].append(key + " single precision")
            if not okb:
                rep["failures"].append(key + " mixed precision")
            print("%-42s fp32 %.3e/tol %.3e %-4s | bf16 %.3e/allowed %.3e %s"
                  % (key, k32["worst"], tol, "ok" if ok32 else "FAIL",
                     cand_err, allowed, "ok" if okb else "FAIL"))
    rep["sections"]["optimized_versus_reference"] = cross
    rep["passed"] = not rep["failures"]
    json.dump(rep, open(a.out, "w"), indent=1, default=str)
    print("\npart one: %s" % ("PASS" if rep["passed"] else "FAIL " + str(rep["failures"])))


if __name__ == "__main__":
    main()
