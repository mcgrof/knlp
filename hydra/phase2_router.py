# SPDX-License-Identifier: MIT
"""hydra: capability predictor + shortfall router (phase 2).

Consumes trace.parquet from phase2_bench.py and reproduces the
capability-routing architecture at small scale: per-dimension
requirement labels from correctness deltas (no LLM judge), an
encoder predictor with K sigmoid heads, benchmark-anchored model
capability profiles with pool-relative normalization, shortfall
matching with a tau sweep, and the cost-quality frontier against
random / always-strong / always-cheap / oracle baselines.

Sub-commands:
  labels  Build per-query requirement labels r in [0,1] per
          dimension: r = normalized rank of the cheapest model that
          answers correctly (0 = cheapest solves it, 1 = only the
          strongest or none). Group-aware train/test split.
  train   Fine-tune the encoder (default ModernBERT-base) with K
          masked sigmoid heads on the train split.
  route   Derive capability profiles from train-split accuracies
          (affine map into the predictor's empirical score band),
          run shortfall matching over the test split across a tau
          sweep, emit frontier.json + GATE.json with paired
          bootstrap statistics.
"""

import argparse
import json

import numpy as np
import pandas as pd

DIM_ORDER = ["math_reasoning", "knowledge", "code"]


def model_cols(df):
    return [c.split("::", 1)[1] for c in df.columns if c.startswith("correct::")]


def ladder_by_cost(df, models):
    costs = {m: df[f"gpu_s::{m}"].mean() for m in models}
    return sorted(models, key=lambda m: costs[m]), costs


def cmd_labels(args):
    df = pd.read_parquet(args.trace)
    models = model_cols(df)
    ladder, costs = ladder_by_cost(df, models)
    n = len(ladder)

    def req(row):
        for i, m in enumerate(ladder):
            if row[f"correct::{m}"] == 1:
                return i / (n - 1)
        return 1.0

    df["requirement"] = df.apply(req, axis=1)
    rng = np.random.RandomState(args.seed)
    groups = sorted(df["group"].unique())
    test_groups = set(
        rng.choice(groups, size=max(1, int(0.2 * len(groups))), replace=False)
    )
    df["split"] = np.where(df["group"].isin(test_groups), "test", "train")
    df.to_parquet(args.out)
    print(
        json.dumps(
            {
                "ladder_cheap_to_strong": ladder,
                "mean_gpu_s": {m: round(costs[m], 4) for m in ladder},
                "split": df["split"].value_counts().to_dict(),
                "mean_requirement": round(float(df["requirement"].mean()), 4),
            }
        )
    )


def _tokenize(tok, texts, max_len=512):
    return tok(
        list(texts),
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )


def cmd_train(args):
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModel,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    df = pd.read_parquet(args.trace)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = _tokenize(tok, tr["prompt"], args.max_len)
    dim_idx = torch.tensor([DIM_ORDER.index(d) for d in tr["dim"]])
    target = torch.tensor(tr["requirement"].values, dtype=torch.float32)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], dim_idx, target)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = AutoModel.from_pretrained(args.encoder).to(device)
    head = torch.nn.Linear(backbone.config.hidden_size, len(DIM_ORDER)).to(device)
    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(head.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    steps = len(dl) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(device)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    backbone.train()
    for epoch in range(args.epochs):
        tot, seen = 0.0, 0
        for ids, mask, didx, y in dl:
            ids, mask = ids.to(device), mask.to(device)
            didx, y = didx.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device, dtype=torch.float16):
                h = backbone(input_ids=ids, attention_mask=mask)
                cls = h.last_hidden_state[:, 0]
                logits = head(cls)
                per = lossf(logits.gather(1, didx.unsqueeze(1)).squeeze(1), y)
                loss = per.mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(y)
            seen += len(y)
        print(json.dumps({"epoch": epoch, "loss": round(tot / seen, 4)}))

    import os  # noqa: PLC0415

    os.makedirs(args.out, exist_ok=True)
    backbone.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    import torch as _t  # noqa: PLC0415

    _t.save(head.state_dict(), f"{args.out}/head.pt")
    print(json.dumps({"saved": args.out}))


def _predict(df, ckpt, max_len=512, batch=64):
    import torch  # noqa: PLC0415
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(ckpt)
    backbone = AutoModel.from_pretrained(ckpt).to(device).eval()
    head = torch.nn.Linear(backbone.config.hidden_size, len(DIM_ORDER)).to(device)
    head.load_state_dict(torch.load(f"{ckpt}/head.pt", map_location=device))
    head.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(df), batch):
            enc = _tokenize(tok, df["prompt"].iloc[i : i + batch], max_len)
            with torch.autocast(device, dtype=torch.float16):
                h = backbone(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                )
                logits = head(h.last_hidden_state[:, 0])
            preds.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.vstack(preds)


def cmd_route(args):
    df = pd.read_parquet(args.trace)
    models = model_cols(df)
    ladder, costs = ladder_by_cost(df, models)
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"].reset_index(drop=True)

    scores_tr = _predict(tr.reset_index(drop=True), args.ckpt)
    scores_te = _predict(te, args.ckpt)

    beta = {}
    for k, dim in enumerate(DIM_ORDER):
        m = (tr["dim"] == dim).values
        s = scores_tr[m, k]
        beta[dim] = (float(np.percentile(s, 5)), float(np.percentile(s, 95)))

    profiles = {}
    for mdl in ladder:
        profiles[mdl] = {}
        for dim in DIM_ORDER:
            sub = tr[tr["dim"] == dim]
            profiles[mdl][dim] = float(sub[f"correct::{mdl}"].mean())
    for dim in DIM_ORDER:
        raw = np.array([profiles[m][dim] for m in ladder])
        lo, hi = beta[dim]
        span = raw.max() - raw.min()
        norm = (
            (raw - raw.min()) / span * (hi - lo) + lo
            if span > 0
            else np.full_like(raw, (lo + hi) / 2)
        )
        for m, v in zip(ladder, norm):
            profiles[m][dim] = float(v)

    dim_idx_te = np.array([DIM_ORDER.index(d) for d in te["dim"]])
    r_hat = scores_te[np.arange(len(te)), dim_idx_te]

    correct = {m: te[f"correct::{m}"].values for m in ladder}
    cost = {m: te[f"gpu_s::{m}"].values for m in ladder}
    strong = max(ladder, key=lambda m: np.mean([profiles[m][d] for d in DIM_ORDER]))
    cheap = ladder[0]

    def route(tau):
        picked = []
        for i in range(len(te)):
            dim = te["dim"].iloc[i]
            shortfalls = {m: max(0.0, r_hat[i] - profiles[m][dim]) for m in ladder}
            eligible = [m for m in ladder if shortfalls[m] <= tau]
            picked.append(
                min(eligible, key=lambda m: costs[m])
                if eligible
                else min(ladder, key=lambda m: shortfalls[m])
            )
        acc = np.array([correct[m][i] for i, m in enumerate(picked)])
        c = np.array([cost[m][i] for i, m in enumerate(picked)])
        return picked, acc, c

    always_strong = correct[strong]
    strong_cost = cost[strong]
    rng = np.random.RandomState(0)
    rand_pick = rng.choice(ladder, size=len(te))
    frontier = []
    for tau in np.linspace(0, 1.0, 41):
        _, acc, c = route(float(tau))
        frontier.append(
            {
                "tau": round(float(tau), 3),
                "accuracy": float(acc.mean()),
                "gpu_s": float(c.mean()),
                "cost_saving_vs_strong": float(1 - c.mean() / strong_cost.mean()),
            }
        )

    oracle_acc, oracle_cost = [], []
    for i in range(len(te)):
        solvers = [m for m in ladder if correct[m][i] == 1]
        m = min(solvers, key=lambda x: costs[x]) if solvers else cheap
        oracle_acc.append(correct[m][i])
        oracle_cost.append(cost[m][i])

    base = {
        "always_strong": {
            "model": strong,
            "accuracy": float(always_strong.mean()),
            "gpu_s": float(strong_cost.mean()),
        },
        "always_cheap": {
            "model": cheap,
            "accuracy": float(correct[cheap].mean()),
            "gpu_s": float(cost[cheap].mean()),
        },
        "random": {
            "accuracy": float(
                np.mean([correct[m][i] for i, m in enumerate(rand_pick)])
            ),
            "gpu_s": float(np.mean([cost[m][i] for i, m in enumerate(rand_pick)])),
        },
        "oracle": {
            "accuracy": float(np.mean(oracle_acc)),
            "gpu_s": float(np.mean(oracle_cost)),
        },
    }

    margin = args.margin
    ok = [
        f
        for f in frontier
        if f["accuracy"] >= base["always_strong"]["accuracy"] - margin
        and f["cost_saving_vs_strong"] >= 0.25
    ]
    best = max(ok, key=lambda f: f["cost_saving_vs_strong"]) if ok else None

    gate = {"margin": margin, "pass": best is not None, "best_point": best}
    if best is not None:
        picked, acc, _ = route(best["tau"])
        deltas = acc.astype(float) - always_strong.astype(float)
        boots = []
        rs = np.random.RandomState(1)
        for _ in range(2000):
            idx = rs.randint(0, len(deltas), len(deltas))
            boots.append(deltas[idx].mean())
        gate["paired_delta_mean"] = float(deltas.mean())
        gate["paired_delta_ci95"] = [
            float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)),
        ]
        gate["noninferior_at_margin"] = gate["paired_delta_ci95"][0] > -margin
        gate["pass"] = bool(gate["noninferior_at_margin"])
        gate["route_mix"] = pd.Series(picked).value_counts().to_dict()
    rand_dom = all(
        not (
            f["accuracy"] <= base["random"]["accuracy"]
            and f["gpu_s"] >= base["random"]["gpu_s"]
        )
        for f in frontier
    )
    gate["dominates_random"] = bool(rand_dom)

    out = {
        "ladder": ladder,
        "beta_bands": beta,
        "profiles": profiles,
        "baselines": base,
        "frontier": frontier,
        "gate": gate,
        "n_test": len(te),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"gate": gate, "baselines": base}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("labels")
    a.add_argument("--trace", default="trace.parquet")
    a.add_argument("--seed", type=int, default=7)
    a.add_argument("--out", default="trace_labeled.parquet")
    a.set_defaults(fn=cmd_labels)

    t = sub.add_parser("train")
    t.add_argument("--trace", default="trace_labeled.parquet")
    t.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--max-len", type=int, default=512)
    t.add_argument("--out", default="predictor_ckpt")
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("route")
    r.add_argument("--trace", default="trace_labeled.parquet")
    r.add_argument("--ckpt", default="predictor_ckpt")
    r.add_argument("--margin", type=float, default=0.05)
    r.add_argument("--out", default="frontier.json")
    r.set_defaults(fn=cmd_route)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
