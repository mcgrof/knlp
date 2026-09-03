#!/usr/bin/env python3
"""Option-ranking evaluation of matched-micro checkpoints.

Scores accountability-benchmark queries by log-probability ranking:
for each query, every option is scored as log P(option | context +
question) under the trained model, and the model answers with the
best-scoring option. No generation, no parsing. Predictions go to a
JSONL the benchmark's own scorer consumes, so accuracy is computed by
the certified path.

Per-arm scoring honors each candidate's audited semantics:

  stack (attn/gdn)  one causal forward; logits[t] predict token t+1.
  titans            same, via the library's raw-logit forward; the
                    checkpoint loads with assign=True because the
                    library double-registers memory-MLP views.
  hope              the repository's two-pass prequential protocol,
                    re-run per sequence from a fresh fast state: each
                    chunk is scored before its own targets feed the
                    memory, exactly as trained (train mode, conv
                    disabled, boundary-target supervision). No state
                    is cached or branched across options — the
                    fidelity audit showed mid-chunk restore computes
                    a different operator.

Options are ranked by mean per-token log-probability by default
(length-robust); the raw dump records sums as well. Queries whose
prompt exceeds the trained context length are skipped identically for
every arm (the scorer counts them wrong for everyone), keeping the
comparison fair; the skip count is reported.

--self-check validates the extraction: the per-position log-probs
must reproduce the harness's own evaluate()/hope_two_pass() losses on
a random batch to ~1e-4 before any real scoring runs.
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import matched_micro_train as mmt  # noqa: E402


def load_checkpoint(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arm = ck["arm"]
    model = mmt.build_arm(arm, torch.device(device))
    model.load_state_dict(ck["model"], assign=(arm == "titans"))
    model = model.to(device)
    return arm, model, ck


def stack_position_logprobs(model, idx):
    """log P(idx[t] | idx[<t]) for t >= 1, shape [T-1]."""
    logits = model(idx)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1).squeeze(0)


def hope_position_logprobs(model, idx, train_chunk):
    """Prequential per-position log-probs mirroring hope_two_pass:
    within a chunk [start, end), logits[i] predict idx[start+i+1],
    the last one predicting the next chunk's first token."""
    from nested_learning.training import compute_teach_signal

    fs = model.init_fast_state()
    _, total_len = idx.shape
    out = torch.empty(total_len - 1)
    for start in range(0, total_len, train_chunk):
        end = min(start + train_chunk, total_len)
        chunk = idx[:, start:end]
        next_tok = idx[:, end : end + 1] if end < total_len else None
        logits = model(chunk, fast_state=fs)
        if next_tok is not None:
            targets = torch.cat([chunk[:, 1:], next_tok], dim=1)
            score_logits = logits
        else:
            targets = chunk[:, 1:]
            score_logits = logits[:, :-1]
        if targets.numel():
            with torch.no_grad():
                logp = F.log_softmax(score_logits.float(), dim=-1)
                vals = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
            out[start : start + vals.numel()] = vals.cpu()
        teach = compute_teach_signal(model, logits, chunk, next_tokens=next_tok)
        with torch.no_grad():
            model(
                chunk,
                teach_signal=teach,
                fast_state=fs,
                finalize_updates=(end >= total_len),
            )
    return out


def position_logprobs(arm, model, cfg, idx):
    if cfg["kind"] == "hope":
        return hope_position_logprobs(model, idx, cfg["train_chunk"])
    return stack_position_logprobs(model, idx).cpu()


def self_check(arm, model, cfg, device, seq_len=256, batch=1, tol=2e-4):
    """The extraction must reproduce the harness's own loss."""
    torch.manual_seed(97)
    idx = torch.randint(0, mmt.CONTRACT["vocab_size"], (batch, seq_len)).to(device)
    lps = position_logprobs(arm, model, cfg, idx)
    mine = -lps.mean().item()
    if cfg["kind"] == "hope":
        ref = mmt.hope_two_pass(model, idx, cfg["train_chunk"], backward=False)
    elif cfg["kind"] == "titans":
        with torch.no_grad():
            ref = model(idx, return_loss=True).item()
    else:
        with torch.no_grad():
            ref = mmt.ce_loss(model(idx), idx).item()
    drift = abs(mine - ref)
    print(f"  self-check {arm}: mine {mine:.6f} ref {ref:.6f} drift {drift:.2e}")
    if drift > tol:
        raise SystemExit(f"self-check FAILED for {arm}: drift {drift} > {tol}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", required=True, help="episodes JSONL")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--norm", choices=("mean", "sum"), default="mean")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split the query set across this many processes; every query "
        "is scored independently from a fresh state, so a sharded run and "
        "a serial one produce the same predictions",
    )
    ap.add_argument("--shard", type=int, default=0, help="this process's shard index")
    a = ap.parse_args()
    if not 0 <= a.shard < a.num_shards:
        raise SystemExit(f"shard {a.shard} outside 0..{a.num_shards - 1}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mmt.CONTRACT["tokenizer"])
    arm, model, ck = load_checkpoint(a.checkpoint, a.device)
    cfg = mmt.ARMS[arm]
    if cfg["kind"] == "hope":
        model.train()  # the CMS delta clip is train-mode-only (audited)
    else:
        model.eval()

    if a.self_check:
        self_check(arm, model, cfg, a.device)

    episodes = [json.loads(l) for l in open(a.episodes)]
    os.makedirs(a.out_dir, exist_ok=True)
    preds, raw = [], []
    skipped = 0
    done = 0
    seen = -1
    for ep in episodes:
        for q in ep["queries"]:
            seen += 1
            if seen % a.num_shards != a.shard:
                continue
            prefix = f"{ep['context']}\nQ: {q['question']}\nA:"
            prefix_ids = tok(prefix).input_ids
            per_option = []
            for opt in q["options"]:
                opt_ids = tok(" " + str(opt)).input_ids
                if len(prefix_ids) + len(opt_ids) > a.max_tokens:
                    per_option = None
                    break
                idx = torch.tensor([prefix_ids + opt_ids]).to(a.device)
                with torch.no_grad() if cfg["kind"] != "hope" else torch.enable_grad():
                    lps = position_logprobs(arm, model, cfg, idx)
                span = lps[len(prefix_ids) - 1 :]
                per_option.append(
                    dict(
                        option=opt,
                        sum=float(span.sum()),
                        mean=float(span.mean()),
                        tokens=len(opt_ids),
                    )
                )
            if per_option is None:
                skipped += 1
                continue
            best = max(per_option, key=lambda o: o[a.norm])
            preds.append(dict(query_id=q["query_id"], prediction=best["option"]))
            raw.append(dict(query_id=q["query_id"], options=per_option))
            done += 1
            if done % 50 == 0:
                print(f"  {arm}: {done} queries scored", flush=True)

    with open(os.path.join(a.out_dir, "predictions.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    with open(os.path.join(a.out_dir, "raw_logprobs.jsonl"), "w") as f:
        for r in raw:
            f.write(json.dumps(r) + "\n")
    manifest = dict(
        arm=arm,
        checkpoint=os.path.abspath(a.checkpoint),
        contract=ck.get("contract"),
        norm=a.norm,
        max_tokens=a.max_tokens,
        shard=a.shard,
        num_shards=a.num_shards,
        queries_scored=done,
        queries_skipped_overlength=skipped,
        env=mmt.environment_manifest(),
    )
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{arm}: {done} scored, {skipped} skipped -> {a.out_dir}", flush=True)


if __name__ == "__main__":
    main()
