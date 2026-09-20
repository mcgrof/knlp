# SPDX-License-Identifier: GPL-2.0
"""A0: does the measurement have room to show a translation result?

Before fitting any mapper, establish three things.

That the coordinates are understood -- the stored keys really are the content
keys rotated by position, on both models, with a positional off-by-one visibly
failing the same check.

That the two geometries are compatible enough to map between, and exactly which
parts do not line up.

And, most importantly, the dynamic range of the evaluation. A translated cache
will be scored against target-native prefill. That comparison is only
informative if the gap between target-native and *no* cache is large, and if
caches that carry the wrong content score badly. If a model does nearly as well
with no context, or with somebody else's context, then a mapper can look
excellent while transmitting nothing, and every later number is unreadable.

So A0 runs the controls the roadmap requires -- no cache, wrong prompt,
shuffled positions, shuffled heads, and same-model identity -- and reports what
each costs in held-out continuation divergence. A1 is only worth running if
these separate.

Env: HIP_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
Run:
    python -m research.kv_translate.run_a0 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from research.jacobian_kv import artifacts as art  # noqa: E402
from research.jacobian_kv.evaluate import (  # noqa: E402
    argmax_flip_rate,
    delta_nll,
    teacher_forced_kl,
)
from research.kv_translate.pairs import (  # noqa: E402
    check_pair,
    describe,
    prefill,
    to_content_keys,
    to_stored_keys,
    verify_rope,
)

SRC = "Qwen/Qwen2.5-0.5B"
TGT = "Qwen/Qwen2.5-1.5B"


def make_cache(keys, values):
    from transformers import DynamicCache

    c = DynamicCache()
    for i, (k, v) in enumerate(zip(keys, values)):
        c.update(k, v, i)
    return c


def continuation_logits_grad(model, keys, values, cont_ids, prompt_len):
    """Teacher-forced continuation on top of a supplied cache, differentiably.

    ``keys``/``values`` may be ``None`` for the no-cache control, in which case
    the continuation is scored unconditioned and its positions start at zero --
    otherwise the model would be told a prefix exists that it cannot see.

    Gradients flow back into the supplied cache tensors, which is what lets a
    translator be trained against the target's behaviour rather than against
    its tensors. Callers that only need a number should use
    :func:`continuation_logits`, which is this wrapped in ``no_grad``; building
    a graph for an evaluation would cost memory for nothing.
    """
    if keys is None:
        n = cont_ids.shape[1]
        attn = torch.ones(1, n, dtype=torch.long, device=cont_ids.device)
        pos = torch.arange(n, device=cont_ids.device).unsqueeze(0)
        return model(input_ids=cont_ids, attention_mask=attn, position_ids=pos).logits
    total = prompt_len + cont_ids.shape[1]
    attn = torch.ones(1, total, dtype=torch.long, device=cont_ids.device)
    pos = torch.arange(prompt_len, total, device=cont_ids.device).unsqueeze(0)
    return model(
        input_ids=cont_ids,
        attention_mask=attn,
        position_ids=pos,
        past_key_values=make_cache(keys, values),
        use_cache=True,
    ).logits


@torch.no_grad()
def continuation_logits(model, keys, values, cont_ids, prompt_len):
    """:func:`continuation_logits_grad` for evaluation: no graph, no gradient."""
    return continuation_logits_grad(model, keys, values, cont_ids, prompt_len)


def score(ref_logits, logits, cont_ids):
    return {
        "kl": float(teacher_forced_kl(ref_logits, logits).mean()),
        "delta_nll": float(delta_nll(ref_logits, logits, cont_ids).mean()),
        "argmax_flip": argmax_flip_rate(ref_logits, logits),
        "nll": float(
            -torch.log_softmax(logits[:, :-1].float(), -1)
            .gather(-1, cont_ids[:, 1:].unsqueeze(-1))
            .mean()
        ),
    }


def wikitext_chunks(tok, n_chunks, ctx, cont_len, seed):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    span = ctx + cont_len
    n_avail = ids.shape[0] // span
    rng = np.random.default_rng(seed)
    picks = rng.choice(n_avail, size=n_chunks, replace=False)
    return [(int(p), ids[p * span : (p + 1) * span].clone()) for p in sorted(picks)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SRC)
    ap.add_argument("--target", default=TGT)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--cont-len", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", default="a0")
    args = ap.parse_args()

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.target)
    tok_src = AutoTokenizer.from_pretrained(args.source)
    same_tok = tok.get_vocab() == tok_src.get_vocab()
    log(f"tokenizers identical: {same_tok}")

    models = {}
    for role, mid in (("source", args.source), ("target", args.target)):
        m = AutoModelForCausalLM.from_pretrained(
            mid, dtype=dtype, attn_implementation="eager"
        ).to(dev)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        models[role] = m
    geom = {
        r: describe(models[r], mid)
        for r, mid in (("source", args.source), ("target", args.target))
    }
    for r, g in geom.items():
        log(
            f"{r}: {g.n_layers}L n_kv={g.n_kv_heads} D={g.head_dim} theta={g.rope_theta:g}"
        )
    compat = check_pair(geom["source"], geom["target"])
    log(f"pair: {json.dumps(compat, default=str)}")

    chunks = wikitext_chunks(tok, args.n_prompts, args.ctx, args.cont_len, args.seed)

    # ---- RoPE gate, on one prompt, both models -------------------------
    probe_ids = chunks[0][1][: args.ctx].unsqueeze(0).to(dev)
    rope = {}
    for r in ("source", "target"):
        p = prefill(models[r], probe_ids, geom[r])
        rope[r] = verify_rope(p)
        log(
            f"rope[{r}] {'PASS' if rope[r]['passed'] else 'FAIL'} "
            f"derot_cos_min={rope[r]['derot_cos_min']:.6f} "
            f"offbyone_cos_max={rope[r]['offbyone_cos_max']:.4f} "
            f"roundtrip={rope[r]['roundtrip_rel_max']:.2e}"
        )
        # round-tripping through the content frame must be a no-op end to end
        rt = to_stored_keys(to_content_keys(p), geom[r], p.n_tokens)
        rel = max(
            float((a.float() - b.float()).norm() / b.float().norm())
            for a, b in zip(rt, p.keys_post)
        )
        rope[r]["frame_roundtrip_rel_max"] = rel
        log(f"rope[{r}] content-frame round trip rel {rel:.2e}")
        del p

    # ---- dynamic range: what do the controls cost ----------------------
    rows = []
    rng = np.random.default_rng(args.seed + 77)
    for ci, (cid, ids) in enumerate(chunks):
        ids = ids.to(dev)
        prompt = ids[: args.ctx].unsqueeze(0)
        cont = ids[args.ctx : args.ctx + args.cont_len].unsqueeze(0)

        tp = prefill(models["target"], prompt, geom["target"])
        ref = continuation_logits(
            models["target"], tp.keys_post, tp.values, cont, args.ctx
        ).float()

        arms = {}
        arms["target_native"] = (tp.keys_post, tp.values)
        arms["no_cache"] = (None, None)

        # wrong prompt: a different chunk's cache, same shape
        other = chunks[(ci + 1) % len(chunks)][1][: args.ctx].unsqueeze(0).to(dev)
        wp = prefill(models["target"], other, geom["target"])
        arms["wrong_prompt"] = (wp.keys_post, wp.values)

        # Two different things get called "shuffled positions", and only one of
        # them is a control.
        #
        # Permuting the (key, value) pairs together is a no-op by construction:
        # attention over a cache is a sum over cached slots, so reordering the
        # slots cannot change the result. Position lives in the rotation already
        # baked into the stored keys, not in the slot index. This arm is kept
        # because it must score zero -- if it ever does not, the harness is
        # reading the cache in a way that depends on storage order, which would
        # be a bug worth catching.
        perm = torch.from_numpy(rng.permutation(args.ctx)).to(dev)
        arms["permuted_kv_slots_noop"] = (
            [k[:, :, perm, :] for k in tp.keys_post],
            [v[:, :, perm, :] for v in tp.values],
        )

        # The real position control: keep each key's content, but re-rotate it
        # at the wrong position. That breaks the binding between what a token
        # says and where it sits, which is exactly the failure a mapper fitted
        # without position awareness would produce.
        content = to_content_keys(tp)
        wrong_pos = perm.to(content[0].device)
        theta = geom["target"].rope_theta
        from research.cartridges_cas.scripts.cas_kv_rope import rerot

        arms["scrambled_rope_positions"] = (
            [rerot(c.float(), wrong_pos, theta) for c in content],
            tp.values,
        )

        # shuffled kv heads
        hperm = torch.from_numpy(rng.permutation(geom["target"].n_kv_heads)).to(dev)
        arms["shuffled_heads"] = (
            [k[:, hperm, :, :] for k in tp.keys_post],
            [v[:, hperm, :, :] for v in tp.values],
        )

        # content-frame round trip: de-RoPE then re-RoPE, must be a no-op
        arms["rope_roundtrip"] = (
            to_stored_keys(to_content_keys(tp), geom["target"], tp.n_tokens),
            tp.values,
        )

        for name, (k, v) in arms.items():
            lg = continuation_logits(models["target"], k, v, cont, args.ctx).float()
            rows.append({"context": cid, "arm": name, **score(ref, lg, cont)})
            del lg
        del tp, wp, arms
        if dev == "cuda":
            torch.cuda.empty_cache()
        log(f"  context {ci + 1}/{len(chunks)} done")

    # ---- summarise ------------------------------------------------------
    arms = sorted({r["arm"] for r in rows})
    summary = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        summary[a] = {
            m: float(np.mean([r[m] for r in rs]))
            for m in ("kl", "delta_nll", "argmax_flip", "nll")
        }
    log("")
    log(f"{'arm':22s} {'KL':>10s} {'dNLL':>9s} {'flip':>7s} {'NLL':>8s}")
    for a in sorted(summary, key=lambda x: summary[x]["kl"]):
        s = summary[a]
        log(
            f"{a:22s} {s['kl']:10.4f} {s['delta_nll']:9.4f} {s['argmax_flip']:7.3f} {s['nll']:8.4f}"
        )

    noop = summary["permuted_kv_slots_noop"]["kl"]
    floor = summary["no_cache"]["kl"]
    native = summary["target_native"]["kl"]
    rt = summary["rope_roundtrip"]["kl"]
    verdict = {
        "dynamic_range_kl": floor - native,
        "no_cache_kl": floor,
        "wrong_prompt_kl": summary["wrong_prompt"]["kl"],
        "rope_roundtrip_kl": rt,
        "rope_gate_passed": all(rope[r]["passed"] for r in rope),
        "scrambled_positions_kl": summary["scrambled_rope_positions"]["kl"],
        "permuted_slots_is_a_noop": abs(noop) < 1e-3,
        "controls_separate": floor > 0.5
        and summary["wrong_prompt"]["kl"] > 0.25 * floor,
        "roundtrip_is_a_noop": rt < 1e-3,
    }
    verdict["A0_PASS"] = bool(
        verdict["rope_gate_passed"]
        and verdict["controls_separate"]
        and verdict["roundtrip_is_a_noop"]
        and verdict["permuted_slots_is_a_noop"]
        and verdict["scrambled_positions_kl"] > 0.1
        and not compat["blockers"]
    )
    log("")
    log(f"A0 {'PASS' if verdict['A0_PASS'] else 'FAIL'}: {json.dumps(verdict)}")

    out = art.ensure_run_dir(args.out_dir, args.run_id)
    art.write_json(
        os.path.join(out, "a0.json"),
        {
            "source": geom["source"].to_dict(),
            "target": geom["target"].to_dict(),
            "tokenizers_identical": same_tok,
            "pair": compat,
            "rope": rope,
            "arms": summary,
            "verdict": verdict,
            "ctx": args.ctx,
            "cont_len": args.cont_len,
            "n_prompts": args.n_prompts,
            "seed": args.seed,
            "dtype": args.dtype,
            "wall_s": time.time() - t0,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "versions": art.versions(),
            "code_commit": art.git_state()[0],
            "code_dirty": art.git_state()[1],
        },
    )
    with art.JsonlWriter(os.path.join(out, "rows.jsonl")) as w:
        for r in rows:
            w.write(r)
    log(f"wrote {out}")
    return 0 if verdict["A0_PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
