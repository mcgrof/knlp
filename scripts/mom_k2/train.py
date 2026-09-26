"""Train the eight-job cohort: four arms at two development seeds, to 80M.

Replaces the four assumptions of the earlier runner: the arms are not hard
coded, the corpus is allowed to wrap and the wrap is logged, seeds are checked
by content hashes of every parameter rather than a weight sum, and no
equal-time number is interpolated. Full checkpoints are written on a declared
grid and on both sides of a declared time budget.

  python3 train.py --out cohort.json
"""
import argparse, copy, hashlib, json, math, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("MATCHED_MICRO_SOURCE", os.path.dirname(HERE)))

LR, HORIZON, WARM = 0.0024, 4882, 488
CTX, MB = 2048, 8
PER = MB * CTX
GRID = [0, 610, 1220, 2441, 3661, 4882]
CUTOFF_S = 1503.1
DIAG_AT = [0, 2441, 4882]
OFFSET_ROWS = {41011: 0, 41013: 9768}
ORDER = [("G0", 41011), ("K1", 41011), ("K2", 41011), ("U2", 41011),
         ("U2", 41013), ("K2", 41013), ("K1", 41013), ("G0", 41013)]


def lr_at(s):
    if s < WARM:
        return LR * (s + 1) / WARM
    return LR * (0.1 + 0.45 * (1 + math.cos(math.pi * (s - WARM) / (HORIZON - WARM))))


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
    ap.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("MOM_K2_CHECKPOINT_DIR"),
        required=not bool(os.environ.get("MOM_K2_CHECKPOINT_DIR")),
        help="external directory for training checkpoints",
    )
    ap.add_argument("--only", default="", help="comma separated arm/seed to run")
    a = ap.parse_args()
    import numpy as np, torch
    import torch.nn.functional as F
    import arms as A, matched_micro_data as mmd
    os.makedirs(a.checkpoint_dir, exist_ok=True)

    meta = json.load(open(a.data_dir + "/meta.json"))
    dev = np.asarray(np.load(a.data_dir + "/dev_selection.npy", mmap_mode="r"))
    vb, s = [], 0
    while len(vb) < 8 and s + (CTX + 1) * MB <= len(dev):
        vb.append(torch.from_numpy(np.stack(
            [dev[s + j * CTX: s + j * CTX + CTX + 1].astype(np.int64)
             for j in range(MB)])).to(a.device))
        s += MB * CTX
    eval_hash = hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes()
                                        for x in vb)).hexdigest()
    torch.manual_seed(1234)
    diag_ids = torch.randint(0, 50257, (2, 512), device=a.device)

    out_path = a.out
    rep = (
        json.load(open(out_path))
        if os.path.exists(out_path)
        else dict(
            kind="k2_cohort",
            source_hashes=A.source_hashes(),
            schedule=dict(
                peak_lr=LR,
                warmup_updates=WARM,
                horizon_updates=HORIZON,
                final_lr=lr_at(HORIZON - 1),
                formula="linear warmup then cosine to a tenth of peak",
            ),
            recipe=dict(
                context=CTX,
                microbatch=MB,
                targets_per_update=PER,
                updates=HORIZON,
                targets=HORIZON * PER,
                optimizer="AdamW betas (0.9, 0.95) weight decay 0.1",
                clipping=1.0,
                aux_coefficient=0.01,
                precision="bfloat16 autocast, fp32 master parameters",
            ),
            data=dict(
                dir=a.data_dir,
                train_tokens=meta["packed"]["train"]["tokens"],
                token_sha256=meta["token_bytes_sha256"],
                evaluation_subset_sha256=eval_hash,
                evaluation_targets=len(vb) * PER,
                offsets=OFFSET_ROWS,
            ),
            equal_time=dict(
                cutoff_seconds=CUTOFF_S,
                declared="before any loss of this cohort existed",
            ),
            grid=GRID,
            runs={},
            seed_fingerprints={},
            batch_hash_log={},
        )
    )

    want = None
    if a.only:
        want = set(tuple(x.split("/")) for x in a.only.split(","))

    shared = {}
    for aid, seed in ORDER:
        key = "%s/seed%d" % (aid, seed)
        if want is not None and (aid, str(seed)) not in want:
            continue
        if key in rep["runs"] and rep["runs"][key].get("complete"):
            print("%-14s already complete, skipping" % key)
            continue
        fam = A.ARMS[aid]["family"]
        if fam == "mom" and seed not in shared:
            base, _ = A.build("K1", seed, a.device)
            shared[seed] = {k: v.detach().clone() for k, v in base.state_dict().items()}
            del base
            torch.cuda.empty_cache()
        state = shared.get(seed) if fam == "mom" else None
        t_alloc = time.time()
        model, checks = A.build(aid, seed, a.device, state=state)
        fp, _ = A.state_fingerprint(model)
        rep["seed_fingerprints"].setdefault(str(seed), {})[aid] = fp
        assert checks["passed"], (key, checks)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1)
        st = mmd.Stream(a.data_dir, "train", CTX, MB)
        st.cursor = OFFSET_ROWS[seed]
        aux = 0.01 if fam == "mom" else 0.0

        def validate():
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                v = [float(F.cross_entropy(
                    model(x[:, :-1]).float().reshape(-1, 50257),
                    x[:, 1:].reshape(-1))) for x in vb]
            model.train()
            return statistics.mean(v)

        def diagnose():
            """Routing loads, selected-weight spread and the router's gradient."""
            if fam != "mom":
                return None
            mix = [b.mixer for b, k in zip(model.blocks, model.kinds) if k == "M"]
            for m in mix:
                m._capture_routing = True
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(diag_ids)
            model.train()
            loads, ent, wmean, wstd = [], [], [], []
            for m in mix:
                w = getattr(m, "_last_routing_weights", None)
                sel = getattr(m, "_last_selected_memories", None)
                m._capture_routing = False
                if w is None:
                    continue
                f = w.float().reshape(-1, w.shape[-1])
                wmean.append(float(f.mean())); wstd.append(float(f.std()))
                p = f / f.sum(-1, keepdim=True).clamp_min(1e-9)
                ent.append(float(-(p * p.clamp_min(1e-9).log()).sum(-1).mean()))
                cnt = torch.bincount(sel.reshape(-1), minlength=4).float()
                loads.append((cnt / cnt.sum()).tolist())
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(diag_ids[:, :-1]).float()
                ce = F.cross_entropy(lg.reshape(-1, 50257), diag_ids[:, 1:].reshape(-1))
            ce.backward()
            gg = [float(m.gate.weight.grad.norm()) for m in mix
                  if m.gate.weight.grad is not None]
            ref = [float(m.o_proj.weight.grad.norm()) for m in mix
                   if m.o_proj.weight.grad is not None]
            model.zero_grad(set_to_none=True)
            return dict(per_bank_load=loads, selected_weight_mean=wmean,
                        selected_weight_std=wstd, selected_entropy=ent,
                        task_only_gate_grad_norm=gg,
                        reference_grad_norm=ref,
                        relative=[g / r for g, r in zip(gg, ref) if r > 0])

        def snapshot(tag, done, clock):
            p = "%s/%s_seed%d_%s.pt" % (a.checkpoint_dir, aid, seed, tag)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            stream=st.state(), updates_done=done,
                            horizon_updates=HORIZON, peak_lr=LR, arm=aid,
                            seed=seed, module=A.ARMS[aid]["module"],
                            training_clock_s=clock, fingerprint=fp,
                            source_hashes=A.source_hashes(),
                            cpu_rng=torch.get_rng_state(),
                            gpu_rng=torch.cuda.get_rng_state()), p)
            return p

        hist = [dict(updates=0, targets=0, training_clock_s=0.0,
                     val_loss=validate())]
        diag = {"0": diagnose()}
        snapshot("u0", 0, 0.0)
        clock, steps = 0.0, []
        wrapped_at, epochs = None, 0
        bhl = {}
        before_ck = after_ck = None
        crossed = finished_early = False
        guard = None
        for i in range(HORIZON):
            for g in opt.param_groups:
                g["lr"] = lr_at(i)
            # a short guard so the pair straddling the budget is adjacent
            if (not crossed) and guard is not None and clock >= guard:
                pend = snapshot("pending_before", i, clock)
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            x, info = st.next_batch(a.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x[:, :-1]).float()
                L = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
                if aux:
                    L = L + aux * model.load_balance_loss()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            clock += dt
            steps.append(dt)
            if not math.isfinite(float(L)):
                rep["runs"][key] = dict(arm=aid, seed=seed, complete=False,
                                        failure="nonfinite loss at update %d" % i)
                json.dump(rep, open(out_path, "w"), indent=1, default=str)
                raise SystemExit("nonfinite loss in %s at update %d" % (key, i))
            if info["wrapped"]:
                wrapped_at = i
                epochs = info["epochs"]
            if i in (0, 1, HORIZON - 1) or (wrapped_at is not None
                                            and abs(i - wrapped_at) <= 1):
                bhl[str(i)] = hashlib.sha256(
                    x.detach().cpu().numpy().tobytes()).hexdigest()[:16]
            if guard is None and len(steps) >= 20:
                guard = CUTOFF_S - 5 * statistics.median(steps[-20:])
            if (not crossed) and clock > CUTOFF_S:
                before_ck = dict(
                    path="%s/%s_seed%d_before_cutoff.pt"
                    % (a.checkpoint_dir, aid, seed),
                    updates=i,
                    adjacent=True,
                )
                if os.path.exists(
                    "%s/%s_seed%d_pending_before.pt" % (a.checkpoint_dir, aid, seed)
                ):
                    os.replace(
                        "%s/%s_seed%d_pending_before.pt"
                        % (a.checkpoint_dir, aid, seed),
                        before_ck["path"],
                    )
                else:
                    before_ck = dict(path=None, updates=i, adjacent=False)
                after_ck = dict(path=snapshot("after_cutoff", i + 1, clock),
                                updates=i + 1, training_clock_s=clock)
                crossed = True
            if (i + 1) in GRID:
                hist.append(dict(updates=i + 1, targets=(i + 1) * PER,
                                 training_clock_s=clock, val_loss=validate()))
                snapshot("u%d" % (i + 1), i + 1, clock)
                print("  %-14s +%4d  %8.1f s  val %.4f"
                      % (key, i + 1, clock, hist[-1]["val_loss"]))
            if (i + 1) in DIAG_AT:
                diag[str(i + 1)] = diagnose()
        finished_early = not crossed
        pend = "%s/%s_seed%d_pending_before.pt" % (a.checkpoint_dir, aid, seed)
        if os.path.exists(pend):
            os.remove(pend)
        rep["runs"][key] = dict(
            arm=aid,
            seed=seed,
            module=A.ARMS[aid]["module"],
            checks=checks,
            fingerprint=fp,
            complete=True,
            final_val_loss=hist[-1]["val_loss"],
            history=hist,
            training_clock_s=clock,
            all_in_wall_s=time.time() - t_alloc,
            median_update_s=statistics.median(steps),
            p90_update_s=sorted(steps)[int(0.9 * len(steps)) - 1],
            targets=HORIZON * PER,
            stream=st.state(),
            wrapped_at_update=wrapped_at,
            epochs=epochs,
            finished_before_cutoff=finished_early,
            unused_budget_s=(CUTOFF_S - clock) if finished_early else None,
            checkpoints=dict(
                final="%s/%s_seed%d_u%d.pt" % (a.checkpoint_dir, aid, seed, HORIZON),
                before_cutoff=before_ck,
                after_cutoff=after_ck,
            ),
            routing_diagnostics=diag,
        )
        rep["batch_hash_log"][key] = bhl
        print("  %-14s DONE %d targets  clock %.1f s  wall %.1f s  val %.4f  "
              "%s" % (key, HORIZON * PER, clock, time.time() - t_alloc,
                      hist[-1]["val_loss"],
                      "finished inside the budget" if finished_early else "crossed the budget"))
        del model, opt, st
        torch.cuda.empty_cache()
        json.dump(rep, open(out_path, "w"), indent=1, default=str)

    # the arms of a seed must have read identical batches
    rep["batch_hash_agreement"] = {}
    for seed in (41011, 41013):
        keys = [k for k in rep["batch_hash_log"] if k.endswith("seed%d" % seed)]
        common = set.intersection(*[set(rep["batch_hash_log"][k]) for k in keys]) if keys else set()
        agree = all(rep["batch_hash_log"][k][i] == rep["batch_hash_log"][keys[0]][i]
                    for k in keys for i in common) if keys else None
        rep["batch_hash_agreement"]["seed%d" % seed] = dict(
            arms=keys, positions=sorted(common), identical=agree)
    json.dump(rep, open(out_path, "w"), indent=1, default=str)
    print("\nbatch hash agreement:", {k: v["identical"] for k, v
                                      in rep["batch_hash_agreement"].items()})


if __name__ == "__main__":
    main()
