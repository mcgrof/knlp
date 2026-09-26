"""Second half of qualification: causality, gradient provenance, full model.

Three things the cross-implementation check does not cover.

Causality through the packed top-2 layout: the routed banks pack each token's
subsequence end to end, so a convolution that ignores segment boundaries can
read across them and let a later token influence an earlier output. Perturbing
one token must leave every earlier output untouched, in training mode, at k=2.

Where the router's gradient comes from, with the task loss alone and with the
balancing term alone. Renormalised top-1 divides one probability by itself, so
the coefficient carries nothing. Top-2 renormalised carries a real relative
mixture. The uniform policy replaces the coefficient with a constant. The
balancing term reaches the router in all three.

One real update of the deployed recipe, and a disk save and reload that
reproduces an uninterrupted short trajectory.

  python3 qualify2.py --out qualify2.json
"""
import argparse, copy, hashlib, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("MATCHED_MICRO_SOURCE", os.path.dirname(HERE)))
LR = 0.0024


def causality(modname, topk, cfg, device):
    """Perturb one token; nothing earlier may move."""
    import torch, qualify as Q
    rows = Q.FIXTURES["segments_below_conv_width"]
    m, _ = Q.build_layer(modname, cfg, device, topk)
    m.train()
    x = Q.make_input(rows, cfg, device)
    T = x.shape[1]
    cut = T // 2
    with torch.no_grad():
        y0 = m(x)[0].float()
        x2 = x.detach().clone()
        x2[:, cut, 4:] += 1.0        # keep the routing channels untouched
        y1 = m(x2)[0].float()
    before = (y0[:, :cut] - y1[:, :cut]).abs().max().item()
    after = (y0[:, cut:] - y1[:, cut:]).abs().max().item()
    del m, x
    torch.cuda.empty_cache()
    return dict(max_change_before_the_perturbed_token=before,
                max_change_at_or_after=after,
                causal=bool(before == 0.0), perturbed_position=cut,
                note=("a nonzero change before the perturbed position would "
                      "mean the packed layout leaks across segment boundaries"))


def router_gradient(modname, topk, cfg, device):
    """Task-only and balancing-only gradient reaching the router."""
    import torch, qualify as Q
    rows = Q.FIXTURES["balanced"]
    out = {}
    for which in ("task_only", "balance_only"):
        m, _ = Q.build_layer(modname, cfg, device, topk)
        m.train()
        x = Q.make_input(rows, cfg, device)
        torch.manual_seed(7)
        G = torch.randn(x.shape[0], x.shape[1], cfg["dim"], device=device)
        y, _, _, logits = m(x)
        if which == "task_only":
            loss = (y.float() * G).sum()
        else:
            p = torch.softmax(logits.float(), dim=-1).mean(0)
            loss = (p * p.numel()).pow(2).sum()      # a balance-shaped term
        m.zero_grad(set_to_none=True)
        loss.backward()
        g = m.gate.weight.grad
        ref = m.o_proj.weight.grad
        out[which] = dict(
            gate_grad_norm=(0.0 if g is None else float(g.norm())),
            gate_grad_is_none=(g is None),
            reference_grad_norm=(0.0 if ref is None else float(ref.norm())),
            relative=(0.0 if g is None or ref is None or float(ref.norm()) == 0
                      else float(g.norm()) / float(ref.norm())))
        del m, x
        torch.cuda.empty_cache()
    return out


def load_bearing(modname, topk, cfg, device):
    """Both routed contributions and the shared path must matter."""
    import torch, qualify as Q
    rows = Q.FIXTURES["balanced"]
    m, _ = Q.build_layer(modname, cfg, device, topk)
    m.eval()
    x = Q.make_input(rows, cfg, device)
    with torch.no_grad():
        base = m(x)[0].float().clone()
        keep = m.v_proj[0].weight.detach().clone()
        m.v_proj[0].weight.zero_()
        one_bank = m(x)[0].float().clone()
        m.v_proj[0].weight.copy_(keep)
        ks = m.shared_v.weight.detach().clone()
        m.shared_v.weight.zero_()
        no_shared = m(x)[0].float().clone()
        m.shared_v.weight.copy_(ks)
    r = dict(zeroing_one_routed_bank_changes_output=float((base - one_bank).abs().max()),
             zeroing_the_shared_path_changes_output=float((base - no_shared).abs().max()))
    r["both_load_bearing"] = bool(r["zeroing_one_routed_bank_changes_output"] > 0
                                  and r["zeroing_the_shared_path_changes_output"] > 0)
    del m, x
    torch.cuda.empty_cache()
    return r


def full_model_and_resume(arm_id, seed, device, data_dir, steps=2):
    """One real update under the deployed recipe, then a disk round trip."""
    import numpy as np, torch, arms
    import torch.nn.functional as F
    import matched_micro_data as mmd
    ctx, mb = 2048, 8

    def go(interrupt):
        model, _ = arms.build(arm_id, seed, device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1)
        st = mmd.Stream(data_dir, "train", ctx, mb)
        aux = 0.01 if arms.ARMS[arm_id]["family"] == "mom" else 0.0
        tmp = "/tmp/k2_resume_%s_%d.pt" % (arm_id, seed)
        trace = []
        for i in range(steps):
            opt.zero_grad(set_to_none=True)
            x, info = st.next_batch(device)
            bh = hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()[:16]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x[:, :-1]).float()
                L = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
                if aux:
                    L = L + aux * model.load_balance_loss()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trace.append(dict(step=i, batch_hash=bh, loss=float(L),
                              cursor=info["cursor"]))
            if interrupt and i == 0:
                torch.save(dict(model=model.state_dict(),
                                opt=copy.deepcopy(opt.state_dict()),
                                stream=st.state(), cpu_rng=torch.get_rng_state(),
                                gpu_rng=torch.cuda.get_rng_state()), tmp)
                del model, opt, st
                torch.cuda.empty_cache()
                d = torch.load(tmp, map_location="cpu", weights_only=False)
                model, _ = arms.build(arm_id, seed, device)
                model.load_state_dict(d["model"])
                opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                        betas=(0.9, 0.95), weight_decay=0.1)
                opt.load_state_dict(d["opt"])
                st = mmd.Stream(data_dir, "train", ctx, mb)
                st.load_state(d["stream"])
                torch.set_rng_state(d["cpu_rng"])
                torch.cuda.set_rng_state(d["gpu_rng"])
        params = {n: p.detach().float().cpu().clone() for n, p in model.named_parameters()}
        mom = {}
        for n, p in model.named_parameters():
            s = opt.state.get(p, {})
            if "exp_avg" in s:
                mom[n + ".exp_avg"] = s["exp_avg"].detach().float().cpu().clone()
        if os.path.exists(tmp):
            os.remove(tmp)
        del model, opt, st
        torch.cuda.empty_cache()
        return dict(trace=trace, params=params, moments=mom)

    a, b = go(False), go(True)
    def worst(x, y):
        return max((x[n] - y[n]).abs().max().item() for n in x)
    return dict(losses=[t["loss"] for t in a["trace"]],
                batch_hashes_match=[t["batch_hash"] for t in a["trace"]]
                                   == [t["batch_hash"] for t in b["trace"]],
                parameter_difference=worst(a["params"], b["params"]),
                moment_difference=worst(a["moments"], b["moments"]),
                bitwise_identical=bool(worst(a["params"], b["params"]) == 0.0
                                       and worst(a["moments"], b["moments"]) == 0.0),
                all_finite=all(math.isfinite(t["loss"]) for t in a["trace"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("MATCHED_MICRO_DATA"),
        required=not bool(os.environ.get("MATCHED_MICRO_DATA")),
        help="packed token directory from matched_micro_data.prepare",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch, arms, matched_micro_train as mmt
    cfg = mmt.ARMS["mom3"]
    rep = dict(kind="k2_qualification_part_two",
               source_hashes=arms.source_hashes(), sections={}, failures=[])

    CASES = [("K1", "mom_renorm", 1), ("K2", "mom_renorm", 2),
             ("U2", "mom_uniform", 2)]

    caus = {}
    for aid, mod, k in CASES:
        caus[aid] = causality(mod, k, cfg, a.device)
        ok = caus[aid]["causal"]
        if not ok:
            rep["failures"].append(aid + " causality")
        print("%-3s causality: change before the perturbed token %.3e, at or after "
              "%.3e -> %s" % (aid, caus[aid]["max_change_before_the_perturbed_token"],
                              caus[aid]["max_change_at_or_after"],
                              "causal" if ok else "LEAKS"))
    rep["sections"]["causality"] = caus

    rg = {}
    for aid, mod, k in CASES:
        rg[aid] = router_gradient(mod, k, cfg, a.device)
        print("%-3s router gradient: task-only %.3e (relative %.3e), "
              "balance-only %.3e" % (aid, rg[aid]["task_only"]["gate_grad_norm"],
                                     rg[aid]["task_only"]["relative"],
                                     rg[aid]["balance_only"]["gate_grad_norm"]))
    rep["sections"]["router_gradient"] = rg
    expect = dict(K1="cancelled", K2="present", U2="absent")
    # keyed to the mechanism the plan states, not to an absolute magnitude:
    # the coefficient's gradient is absent under the constant policy, at
    # roundoff under top-1 renormalisation, and orders of magnitude larger
    # under top-2 renormalisation. An absolute floor chosen by hand would only
    # test whichever number was picked.
    k1r = rg["K1"]["task_only"]["relative"]
    k2r = rg["K2"]["task_only"]["relative"]
    u2r = rg["U2"]["task_only"]["relative"]
    rg_ok = (u2r == 0.0
             and k1r < 1e-6
             and k2r > 1000.0 * max(k1r, 1e-12)
             and all(rg[a_]["balance_only"]["gate_grad_norm"] > 0 for a_ in rg))
    rep["sections"]["router_gradient_separation"] = dict(
        uniform_exactly_zero=bool(u2r == 0.0),
        top1_at_roundoff=k1r, top2=k2r,
        ratio_top2_over_top1=(k2r / max(k1r, 1e-12)),
        predicate="constant absent, top-1 at roundoff, top-2 a thousandfold above it")
    rep["sections"]["router_gradient_expectation"] = dict(
        expected=expect, matches=bool(rg_ok),
        reading=("the task loss reaches the mixing coefficient only under top-2 "
                 "renormalisation; the balancing term reaches the router under "
                 "all three policies"))
    if not rg_ok:
        rep["failures"].append("router gradient provenance")

    lb = {}
    for aid, mod, k in CASES:
        lb[aid] = load_bearing(mod, k, cfg, a.device)
        if not lb[aid]["both_load_bearing"]:
            rep["failures"].append(aid + " contributions not load bearing")
        print("%-3s zeroing one routed bank moves the output by %.3e; zeroing the "
              "shared path by %.3e" % (aid, lb[aid]["zeroing_one_routed_bank_changes_output"],
                                       lb[aid]["zeroing_the_shared_path_changes_output"]))
    rep["sections"]["load_bearing"] = lb

    fm = {}
    for aid in ("G0", "K2", "U2"):
        fm[aid] = full_model_and_resume(aid, 41011, a.device, a.data_dir)
        if not (fm[aid]["bitwise_identical"] and fm[aid]["all_finite"]
                and fm[aid]["batch_hashes_match"]):
            rep["failures"].append(aid + " full model resume")
        print("%-3s full model: losses %s  resume parameters %.3e moments %.3e -> %s"
              % (aid, [round(x, 4) for x in fm[aid]["losses"]],
                 fm[aid]["parameter_difference"], fm[aid]["moment_difference"],
                 "bitwise identical" if fm[aid]["bitwise_identical"] else "NOT identical"))
    rep["sections"]["full_model_and_resume"] = fm

    rep["passed"] = not rep["failures"]
    json.dump(rep, open(a.out, "w"), indent=1, default=str)
    print("\npart two: %s" % ("PASS" if rep["passed"] else "FAIL " + str(rep["failures"])))


if __name__ == "__main__":
    main()
