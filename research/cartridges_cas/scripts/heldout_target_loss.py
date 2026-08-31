"""How well does a cartridge model the teacher's real distribution?

Every number in this lane so far comes from one instrument: forced-choice
letter margins over twenty questions.  That instrument says cartridges
trained on damaged targets and on clean ones are indistinguishable, which
is worth believing only if a measurement of a different kind agrees.

This is that measurement, and it is deliberately the objective's own
terms rather than the evaluation's.  Score a cartridge by the
distillation cross-entropy it achieves against *clean* deduplicated
targets, on elements its training schedule never drew.  Held out matters:
on trained elements a cartridge that memorized its own damaged targets
would be flattered by them.  Clean targets matter too -- both arms are
graded against the teacher's true distribution, so the arm trained on
near-hard-labels is asked something it was never trained on, which is
exactly the question.

A dissociation is the interesting outcome and it should be reported
plainly if it appears: worse fidelity here with equal document
specificity would say the defect really did cost teacher fidelity, and
that teacher fidelity is not what document specificity is made of.

**Read the default score for what it is.** 89.5% of clean target mass
sits on the top-1 token, so grading the full row is mostly a test of
agreement with the teacher's argmax, and it duly ranks the damaged arm
-- which trains almost only on the argmax, at roughly 1.8x weight --
above the clean one.  That is close to circular and must not be
reported as the defect improving teacher fidelity.  Set TAIL_ONLY=1 to
drop each row's argmax and grade only the remaining tenth, which is
where the teacher's uncertainty actually lives and which no arm was
directly optimized for.

Env: MODEL, CARTS (comma list name=path), DATA_PARQUET (the CLEAN one),
     SCHEDULE_JSON (to exclude trained elements), OUT_JSON,
     N_HELDOUT (default 200), SEED (default 0, must match training),
     TAIL_ONLY=1 to grade only the non-argmax mass.
"""

import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/data/mcgrof/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/heldout")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import TrainDataset, DataSource
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

import sys

# runs both from the repo (control_aware/ one level up) and from the
# staged tree on a GPU host (control_aware/ alongside this file)
for _d in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    sys.path.insert(0, str(_d))
from control_aware.targets import parse_element, build_target_set  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CARTS = os.environ["CARTS"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
SCHEDULE_JSON = os.environ["SCHEDULE_JSON"]
OUT_JSON = os.environ["OUT_JSON"]
N_HELDOUT = int(os.environ.get("N_HELDOUT", "200"))
TAIL_ONLY = os.environ.get("TAIL_ONLY", "0") == "1"
SEED = int(os.environ.get("SEED", "0"))
DEVICE = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)
VOCAB = int(model.config.vocab_size)

dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=SEED)

schedule = json.loads(Path(SCHEDULE_JSON).read_text())
trained = {i for step in schedule["schedule"] for i in step}
held = [i for i in range(len(dataset.elements)) if i not in trained][:N_HELDOUT]
print(
    f"[heldout] {len(dataset.elements)} elements, {len(trained)} ever trained, "
    f"scoring {len(held)} held out",
    flush=True,
)


def load_cart(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    tk = [t(p) for p in ck["trainable_keys"]]
    tv = [t(p) for p in ck["trainable_values"]]
    fk = ck.get("frozen_keys") or []
    fv = ck.get("frozen_values") or []
    if fk:
        ik = [torch.cat([t(fk[i]), tk[i]], dim=2) for i in range(len(tk))]
        iv = [torch.cat([t(fv[i]), tv[i]], dim=2) for i in range(len(tv))]
        nfrozen = t(fk[0]).shape[2]
    else:
        ik, iv, nfrozen = tk, tv, 0
    return TrainableCache(
        config=ac, init_keys=ik, init_values=iv, num_frozen_tokens=nfrozen
    ).to(DEVICE)


def drop_argmax(ts):
    """Same target set with each row's most likely token removed.

    89.5% of clean target mass sits on the top-1 token, so grading
    against the full row mostly measures agreement with the teacher's
    argmax and rewards whichever objective over-weighted it -- which is
    the objective the defect produced.  The teacher's actual uncertainty
    lives in the remaining tenth.  This keeps only that, so the score
    asks about the distribution rather than the choice.
    """
    best = {}
    for j, (r, p) in enumerate(zip(ts.row_idxs, ts.probs)):
        if r not in best or p > ts.probs[best[r]]:
            best[r] = j
    keep = [j for j in range(len(ts.row_idxs)) if best.get(ts.row_idxs[j]) != j]
    return type(ts)(
        [ts.row_idxs[j] for j in keep],
        [ts.token_ids[j] for j in keep],
        [ts.probs[j] for j in keep],
        ts.denom,
    )


# targets are built once: every cartridge is graded against the same
# clean rows, so any difference is the cartridge and not the grading
EOT = tok.convert_tokens_to_ids("<|im_end|>")
TARGETS = {}
for i in held:
    el = dataset.elements[i]
    et = parse_element(el.topk_token_idxs, el.topk_token_ids, el.topk_logprobs, EOT)
    ts = build_target_set(et, "dedup_legacy_support")
    TARGETS[i] = drop_argmax(ts) if TAIL_ONLY else ts
if TAIL_ONLY:
    kept = sum(len(t.row_idxs) for t in TARGETS.values())
    print(f"[heldout] TAIL_ONLY: grading {kept} non-argmax entries", flush=True)


@torch.no_grad()
def heldout_loss(cache):
    losses = []
    for i in held:
        el = dataset.elements[i]
        ids = el.input_ids.to(DEVICE)
        cache.clear()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=ids,
                seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
                position_ids=torch.arange(ids.shape[0], device=DEVICE),
                use_cache=True,
                past_key_values=cache,
            )
        ts = TARGETS[i]
        gi, tids, probs = ts.tensors(device=DEVICE)
        gi = gi - 1
        valid = (gi >= 0) & (gi < ids.shape[0]) & (tids >= 0) & (tids < VOCAB)
        if int(valid.sum()) == 0:
            continue
        lp = F.log_softmax(out.logits.float(), dim=-1)[0, gi[valid], tids[valid]]
        # normalized by the kept mass, so elements with more surviving
        # target mass do not simply dominate the mean
        losses.append(
            float(-(probs[valid] * lp).sum() / probs[valid].sum().clamp(min=1e-9))
        )
    return losses


def main():
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    report = dict(
        model=MODEL,
        parquet=DATA_PARQUET,
        n_heldout=len(held),
        held_ids=held,
        conditions={},
    )
    per_element = {}
    for spec in CARTS.split(","):
        name, path = spec.split("=", 1)
        cache = load_cart(path)
        L = heldout_loss(cache)
        per_element[name] = L
        # per-element losses are kept so any pair of arms can be compared
        # after the fact; with more than two arms the built-in pairing
        # below stays silent and this is the only way back to a difference
        report["conditions"][name] = dict(
            n=len(L),
            mean=statistics.fmean(L),
            median=statistics.median(L),
            per_element=L,
        )
        print(
            f"[heldout] {name}: mean clean-target CE {statistics.fmean(L):.4f} "
            f"over {len(L)} elements",
            flush=True,
        )
        with open(OUT_JSON, "w") as f:
            json.dump(report, f, indent=1)
        del cache
        torch.cuda.empty_cache()

    names = list(per_element)
    if len(names) == 2:
        a, b = names
        d = [x - y for x, y in zip(per_element[a], per_element[b])]
        m = statistics.fmean(d)
        se = statistics.stdev(d) / (len(d) ** 0.5) if len(d) > 1 else 0.0
        report["paired"] = dict(
            a=a, b=b, mean_diff=m, se=se, lo=m - 1.96 * se, hi=m + 1.96 * se
        )
        with open(OUT_JSON, "w") as f:
            json.dump(report, f, indent=1)
        print(
            f"[heldout] {a} minus {b}: {m:+.4f} "
            f"[{m - 1.96 * se:+.4f}, {m + 1.96 * se:+.4f}] "
            f"(negative favours {a})",
            flush=True,
        )
    print(f"HELDOUT_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()
