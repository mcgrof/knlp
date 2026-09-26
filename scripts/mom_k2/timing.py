"""Measure what the cohort will cost, then price the whole bundle.

Timing runs carry no profiler: profiler overhead must not reach a throughput or
equal-time scientific number. Each arm is measured alone, after its own warmup,
on the exact training shape, and the slower of its median and ninetieth
percentile update is what the admission arithmetic uses.

  python3 timing.py --out timing.json
"""
import argparse, hashlib, json, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get("MATCHED_MICRO_SOURCE", os.path.dirname(HERE)))
LR, UPDATES = 0.0024, 4882
JOBS = 8
CLOSEOUT_S = 1800.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--data-dir",
        default=os.environ.get("MATCHED_MICRO_DATA"),
        required=not bool(os.environ.get("MATCHED_MICRO_DATA")),
        help="packed token directory from matched_micro_data.prepare",
    )
    ap.add_argument("--measure", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--allocation-hours", type=float, default=12.0)
    ap.add_argument("--spent-hours", type=float, required=True,
                    help="device time already charged to this study")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import numpy as np, torch
    import torch.nn.functional as F
    import arms, matched_micro_data as mmd
    ctx, mb = 2048, 8
    per = mb * ctx

    rep = dict(kind="k2_timing_and_admission",
               source_hashes=arms.source_hashes(),
               shape=dict(context=ctx, microbatch=mb, targets_per_update=per),
               measured_updates=a.measure, warmup_updates=a.warmup,
               profiler="not attached during timing", arms={}, admission={})

    for aid in ("G0", "K1", "K2", "U2"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, checks = arms.build(aid, 41011, a.device)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                                weight_decay=0.1)
        st = mmd.Stream(a.data_dir, "train", ctx, mb)
        aux = 0.01 if arms.ARMS[aid]["family"] == "mom" else 0.0
        pbytes = sum(p.numel() * p.element_size() for p in model.parameters())

        def one():
            opt.zero_grad(set_to_none=True)
            x, _ = st.next_batch(a.device)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x[:, :-1]).float()
                L = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
                if aux:
                    L = L + aux * model.load_balance_loss()
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        for _ in range(a.warmup):
            one()
        torch.cuda.reset_peak_memory_stats()
        ts = [one() for _ in range(a.measure)]
        med = statistics.median(ts)
        p90 = sorted(ts)[int(0.9 * len(ts)) - 1]

        # evaluation and checkpoint cost, both charged to the wall ledger
        dev = np.asarray(np.load(a.data_dir + "/dev_selection.npy", mmap_mode="r"))
        vb, s = [], 0
        while len(vb) < 8 and s + (ctx + 1) * mb <= len(dev):
            vb.append(torch.from_numpy(np.stack(
                [dev[s + j * ctx: s + j * ctx + ctx + 1].astype(np.int64)
                 for j in range(mb)])).to(a.device))
            s += mb * ctx
        torch.cuda.synchronize(); t0 = time.perf_counter()
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x in vb:
                lg = model(x[:, :-1]).float()
                F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1))
        model.train()
        torch.cuda.synchronize()
        eval_s = time.perf_counter() - t0
        p = "/tmp/k2_ckpt_probe.pt"
        t0 = time.perf_counter()
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict()), p)
        ck_s = time.perf_counter() - t0
        ck_bytes = os.path.getsize(p)
        os.remove(p)

        rep["arms"][aid] = dict(
            checks=checks, median_update_s=med, p90_update_s=p90,
            mean_update_s=statistics.mean(ts),
            targets_per_s_median=per / med, targets_per_s_p90=per / p90,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            parameter_bytes=pbytes,
            evaluation_seconds=eval_s, checkpoint_seconds=ck_s,
            checkpoint_bytes=ck_bytes,
            raw_update_seconds=ts)
        print("%-3s median %.4f s  p90 %.4f s  %6.0f tgt/s (p90)  peak %.2f/%.2f GiB  "
              "eval %.1f s  ckpt %.1f s"
              % (aid, med, p90, per / p90,
                 torch.cuda.max_memory_allocated() / 2**30,
                 torch.cuda.max_memory_reserved() / 2**30, eval_s, ck_s))
        del model, opt, st, vb
        torch.cuda.empty_cache()

    # ---- admission arithmetic for the complete bundle
    ck_per_job = 6          # updates 0, 610, 1220, 2441, 3661, 4882
    ev_per_job = 6
    cutoff_ck = 2           # the pair either side of the time budget
    train = 0.0
    overhead = 0.0
    for aid in ("G0", "K1", "K2", "U2"):
        r = rep["arms"][aid]
        train += 2 * UPDATES * r["p90_update_s"]
        overhead += 2 * (ev_per_job * r["evaluation_seconds"]
                         + (ck_per_job + cutoff_ck) * r["checkpoint_seconds"])
    need = 1.3 * train + overhead + CLOSEOUT_S
    have = a.allocation_hours * 3600.0 - a.spent_hours * 3600.0
    rep["admission"] = dict(
        jobs=JOBS, updates_per_job=UPDATES,
        training_seconds_at_p90=train,
        training_with_30_percent_headroom=1.3 * train,
        explicit_overhead_seconds=overhead,
        overhead_detail=dict(evaluations_per_job=ev_per_job,
                             checkpoints_per_job=ck_per_job + cutoff_ck),
        protected_closeout_seconds=CLOSEOUT_S,
        required_seconds=need, required_hours=need / 3600.0,
        allocation_hours=a.allocation_hours, already_spent_hours=a.spent_hours,
        remaining_seconds=have, remaining_hours=have / 3600.0,
        fits=bool(need <= have),
        margin_hours=(have - need) / 3600.0,
        rule=("1.3 times the ninetieth-percentile training time for all eight "
              "jobs, plus measured evaluation and checkpoint cost, plus half an "
              "hour of protected closeout"))
    json.dump(rep, open(a.out, "w"), indent=1, default=str)
    ad = rep["admission"]
    print("\nbundle: training at p90 %.3f h, with headroom %.3f h, overhead %.3f h, "
          "closeout %.3f h" % (train / 3600, 1.3 * train / 3600,
                               overhead / 3600, CLOSEOUT_S / 3600))
    print("required %.3f h against %.3f h remaining (of %.1f, %.3f already spent) -> fits %s"
          % (ad["required_hours"], ad["remaining_hours"], a.allocation_hours,
             a.spent_hours, ad["fits"]))
    if ad["fits"]:
        print("margin %.3f h" % ad["margin_hours"])


if __name__ == "__main__":
    main()
