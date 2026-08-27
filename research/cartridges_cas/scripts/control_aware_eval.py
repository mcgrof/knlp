#!/usr/bin/env python3
"""Gate 0 evaluator: strict generation, forced-choice content, and
control-state diagnostics for control-aware screen checkpoints.

Four sections per cartridge checkpoint (plus a no-cartridge control for
the generation and forced-choice sections):

1. PRIMARY generation eval: thinking disabled, cap 32, strict
   standalone-letter parsing, unclosed thinking counted invalid, every
   raw response persisted.  Reports strict accuracy, parser-invalid,
   cap-hit, mean/median length, standalone-letter rate, and
   end-of-turn occurrence/position.
2. STRESS generation eval: thinking enabled, cap 256 (not the primary
   endpoint), with unclosed-thinking rate reported explicitly.
3. FORCED-CHOICE content eval: score the five letter candidates at the
   first answer position under an identical prompt; report A-E argmax
   accuracy, correct-letter log-probability, correct-minus-best-wrong
   margin, and entropy over the five (tokenizer representations of the
   letters are verified and recorded).
4. CONTROL-STATE probe on a fixed held-out element set (never in the
   frozen training schedule): unique/anchor/legacy losses on the same
   probe, first-row loss, chosen-token probability at first versus
   ordinary rows, end-of-turn probability at natural stored endings and
   after a one-letter answer, unique/anchor gradient norms and cosine,
   and K-versus-V gradient energy by layer.

Env: MODEL, CARTS (comma list of name=path), PATIENT, DATA_PARQUET,
SCHEDULE_JSON, LONGHEALTH_JSON (optional offline benchmark), OUT_JSON,
MAX_Q, PROBE_N.
"""

import json
import os
import re
import statistics
import sys
from pathlib import Path

os.environ.setdefault("CARTRIDGES_DIR", "/root/cartridges")
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/ca_eval")
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.data.longhealth.utils import load_longhealth_dataset
from cartridges.datasets import TrainDataset, DataSource
from cartridges.generation import flex_generate
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

for _d in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    sys.path.insert(0, str(_d))
from control_aware.targets import build_target_set, parse_element  # noqa: E402

EVALUATOR_VERSION = "control-aware-eval-v1"

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
CARTS = os.environ["CARTS"]
PATIENT = os.environ["PATIENT"]
DATA_PARQUET = os.environ["DATA_PARQUET"]
SCHEDULE_JSON = os.environ["SCHEDULE_JSON"]
LONGHEALTH_JSON = os.environ.get("LONGHEALTH_JSON", "")
OUT_JSON = os.environ.get("OUT_JSON", "/tmp/ca_eval/eval.json")
MAX_Q = int(os.environ.get("MAX_Q", "20"))
PROBE_N = int(os.environ.get("PROBE_N", "16"))
SINK_MAX = 4
DEVICE = "cuda"

STANDALONE = re.compile(r"^[\s\*\_\#\-]*\(?([A-Ea-e])\)?[\.\:\)\s\*\_]*$")
STATED = re.compile(
    r"(?:answer|option|choice|letter)\s*(?:is|:|=)?\s*\**\(?([A-Ea-e])\)?\b",
    re.IGNORECASE,
)


def strict_parse(resp):
    if "<think>" in resp and "</think>" not in resp:
        return "", "unclosed_think"
    body = resp.split("</think>")[-1] if "</think>" in resp else resp
    body = body.strip()
    if not body:
        return "", "empty"
    first_line = next((ln for ln in body.splitlines() if ln.strip()), "")
    m = STANDALONE.match(first_line.strip())
    if m:
        return m.group(1).upper(), "standalone"
    m = STATED.search(body)
    if m:
        return m.group(1).upper(), "stated"
    return "", "invalid"


def load_patients(patient_ids):
    if not LONGHEALTH_JSON:
        return load_longhealth_dataset(patient_ids)
    from cartridges.data.longhealth.utils import LongHealthPatient

    data = json.loads(Path(LONGHEALTH_JSON).read_text())
    for pid, row in data.items():
        for qq in row["questions"]:
            qq["question_id"] = pid + "_" + str(qq["No"])
    return [
        LongHealthPatient(patient_id=pid, **row)
        for pid, row in data.items()
        if pid in patient_ids
    ]


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
EOT = tok.convert_tokens_to_ids("<|im_end|>")
model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
model.eval()
model.requires_grad_(False)
ac = AttnConfig(
    n_layers=model.config.num_hidden_layers,
    n_heads=model.config.num_key_value_heads,
    head_dim=model.config.head_dim,
)

LETTERS = "ABCDE"
LETTER_IDS = {}
for L in LETTERS:
    enc = tok(L, add_special_tokens=False).input_ids
    assert len(enc) == 1, f"letter {L} is not a single token: {enc}"
    LETTER_IDS[L] = enc[0]


def load_cart(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p).to(torch.bfloat16)

    fk = ck.get("frozen_keys") or []
    nfrozen = t(fk[0]).shape[2] if fk else 0
    use_frozen = 0 < nfrozen <= SINK_MAX

    def cat(fro, tra):
        tt = [t(p) for p in tra]
        if fro and use_frozen:
            ff = [t(p) for p in fro]
            return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
        return tt

    ik = cat(ck.get("frozen_keys"), ck["trainable_keys"])
    iv = cat(ck.get("frozen_values"), ck["trainable_values"])
    return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)


def question_prompt(q):
    return (
        f"Question: {q.question}\nA) {q.answer_a}\nB) {q.answer_b}\n"
        f"C) {q.answer_c}\nD) {q.answer_d}\nE) {q.answer_e}\n\n"
        "Answer with ONLY the letter (A, B, C, D, or E). Do not explain."
    )


def chat_ids(prompt, thinking):
    return (
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=thinking,
        )
        .to(DEVICE)
        .flatten()
    )


def questions():
    for patient in load_patients([PATIENT]):
        for q in patient.questions[:MAX_Q]:
            yield q


@torch.no_grad()
def generation_eval(cache, thinking, cap):
    rows = []
    for q in questions():
        ids = chat_ids(question_prompt(q), thinking)
        if cache is not None:
            cache.clear()
        out = flex_generate(
            model,
            tok,
            ids,
            seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
            position_ids=torch.arange(ids.shape[0], device=DEVICE),
            max_new_tokens=cap,
            cache=cache,
            temperature=0.0,
        )
        out_ids = list(out.get(0, []))
        resp = tok.decode(out_ids, skip_special_tokens=True)
        letter, reason = strict_parse(resp)
        amap = dict(
            A=q.answer_a, B=q.answer_b, C=q.answer_c, D=q.answer_d, E=q.answer_e
        )
        eot_pos = out_ids.index(EOT) if EOT in out_ids else -1
        rows.append(
            dict(
                parsed=letter,
                parse_reason=reason,
                strict_correct=bool(letter) and amap.get(letter, "") == q.correct,
                n_tokens=len(out_ids),
                cap_hit=len(out_ids) >= cap,
                eot_pos=eot_pos,
                raw=resp,
            )
        )
    n = max(len(rows), 1)
    lengths = [r["n_tokens"] for r in rows]
    return (
        dict(
            thinking=thinking,
            cap=cap,
            n=len(rows),
            strict_acc=sum(r["strict_correct"] for r in rows) / n,
            parser_invalid=sum(r["parse_reason"] in ("invalid", "empty") for r in rows)
            / n,
            unclosed_think=sum(r["parse_reason"] == "unclosed_think" for r in rows) / n,
            standalone_rate=sum(r["parse_reason"] == "standalone" for r in rows) / n,
            cap_hit=sum(r["cap_hit"] for r in rows) / n,
            mean_len=statistics.fmean(lengths) if lengths else 0,
            median_len=statistics.median(lengths) if lengths else 0,
            eot_rate=sum(r["eot_pos"] >= 0 for r in rows) / n,
            eot_mean_pos=(
                statistics.fmean([r["eot_pos"] for r in rows if r["eot_pos"] >= 0])
                if any(r["eot_pos"] >= 0 for r in rows)
                else None
            ),
        ),
        rows,
    )


@torch.no_grad()
def cached_forward_logits(ids, cache):
    if cache is None:
        # FlexQwen3 forward expects a cache object; an empty
        # TrainableCache reduces the mask to plain causal attention.
        cache = TrainableCache(config=ac).to(DEVICE)
    cache.clear()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=ids,
            seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
            position_ids=torch.arange(ids.shape[0], device=DEVICE),
            use_cache=True,
            past_key_values=cache,
        )
    return out.logits[0].float()


@torch.no_grad()
def forced_choice_eval(cache):
    rows = []
    for q in questions():
        ids = chat_ids(question_prompt(q), thinking=False)
        lp = F.log_softmax(cached_forward_logits(ids, cache)[-1], dim=-1)
        letter_lps = {L: float(lp[LETTER_IDS[L]]) for L in LETTERS}
        amap = dict(
            A=q.answer_a, B=q.answer_b, C=q.answer_c, D=q.answer_d, E=q.answer_e
        )
        correct_letter = next((k for k, v in amap.items() if v == q.correct), "")
        pred = max(letter_lps, key=letter_lps.get)
        vals = torch.tensor(list(letter_lps.values()))
        p5 = F.softmax(vals, dim=-1)
        entropy = float(-(p5 * p5.log()).sum())
        wrong_best = max(v for k, v in letter_lps.items() if k != correct_letter)
        eot_after = None
        if correct_letter:
            ids2 = torch.cat(
                [ids, torch.tensor([LETTER_IDS[correct_letter]], device=DEVICE)]
            )
            lp2 = F.log_softmax(cached_forward_logits(ids2, cache)[-1], dim=-1)
            eot_after = float(lp2[EOT].exp())
        rows.append(
            dict(
                correct_letter=correct_letter,
                pred=pred,
                fc_correct=pred == correct_letter,
                correct_lp=letter_lps.get(correct_letter),
                margin=(letter_lps.get(correct_letter, 0.0) - wrong_best),
                entropy=entropy,
                eot_after_letter=eot_after,
            )
        )
    n = max(len(rows), 1)
    return dict(
        n=len(rows),
        fc_acc=sum(r["fc_correct"] for r in rows) / n,
        correct_lp_mean=statistics.fmean(
            [r["correct_lp"] for r in rows if r["correct_lp"] is not None]
        ),
        margin_mean=statistics.fmean([r["margin"] for r in rows]),
        entropy_mean=statistics.fmean([r["entropy"] for r in rows]),
        eot_after_letter_mean=statistics.fmean(
            [r["eot_after_letter"] for r in rows if r["eot_after_letter"] is not None]
        ),
        letter_token_ids=LETTER_IDS,
    )


# ---------------------------------------------------------------------------
# control-state probe on held-out elements
# ---------------------------------------------------------------------------

dataset = TrainDataset.Config(
    data_sources=[DataSource(path=DATA_PARQUET, type="local")],
    top_k_logits=20,
    packed_seq_length=2048,
    packing_mode="truncate",
).instantiate(tokenizer=tok, seed=0)
schedule = json.loads(Path(SCHEDULE_JSON).read_text())
sched_ids = {i for step in schedule["schedule"] for i in step}
PROBE_IDS = [i for i in range(len(dataset.elements)) if i not in sched_ids][:PROBE_N]


def entry_loss(logits, entries, denom):
    gi, tids, probs = entries
    seq_len = logits.shape[0]
    vocab = logits.shape[-1]
    gi = gi - 1
    valid = (gi >= 0) & (gi < seq_len) & (tids >= 0) & (tids < vocab)
    if int(valid.sum()) == 0:
        return None
    lp = F.log_softmax(logits.float(), dim=-1)[gi[valid], tids[valid]]
    return -(probs[valid] * lp).sum() / denom


def probe_eval(cache):
    """Losses, control-position probabilities, and branch-gradient
    geometry on the fixed held-out probe."""
    tot = dict(legacy=0.0, unique=0.0, anchor=0.0, first_row=0.0)
    chosen_first, chosen_content, eot_natural = [], [], []
    g_unique, g_anchor = None, None
    kv_energy = dict(k=0.0, v=0.0)
    n_ok = 0
    for i in PROBE_IDS:
        el = dataset.elements[i]
        et = parse_element(el.topk_token_idxs, el.topk_token_ids, el.topk_logprobs, EOT)
        ids = el.input_ids.to(DEVICE)
        if cache is not None:
            cache.clear()
            for p in cache.parameters():
                p.grad = None
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                input_ids=ids,
                seq_ids=torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE),
                position_ids=torch.arange(ids.shape[0], device=DEVICE),
                use_cache=True,
                past_key_values=cache,
            )
        logits = out.logits[0]
        ts_grp = build_target_set(et, "legacy_grouped_replay")
        ts_uni = build_target_set(et, "dedup_legacy_support")
        n_uni = len(ts_uni.token_ids)
        denom = et.n_serialized
        uni = entry_loss(logits, ts_uni.tensors(device=DEVICE), denom)
        anc_entries = (
            torch.tensor(ts_grp.row_idxs[n_uni:], dtype=torch.long, device=DEVICE),
            torch.tensor(ts_grp.token_ids[n_uni:], dtype=torch.long, device=DEVICE),
            torch.tensor(ts_grp.probs[n_uni:], dtype=torch.float32, device=DEVICE),
        )
        anc = (
            entry_loss(logits, anc_entries, denom)
            if len(ts_grp.token_ids) > n_uni
            else None
        )
        if uni is None:
            continue
        n_ok += 1
        tot["unique"] += float(uni.detach())
        if anc is not None:
            tot["anchor"] += float(anc.detach())
        tot["legacy"] += float(uni.detach()) + (
            float(anc.detach()) if anc is not None else 0.0
        )
        first_entries = [
            (ri, tid, p)
            for ri, tid, p in zip(ts_grp.row_idxs, ts_grp.token_ids, ts_grp.probs)
            if ri == et.first_row_idx
        ]
        fr = entry_loss(
            logits,
            (
                torch.tensor(
                    [e[0] for e in first_entries], dtype=torch.long, device=DEVICE
                ),
                torch.tensor(
                    [e[1] for e in first_entries], dtype=torch.long, device=DEVICE
                ),
                torch.tensor(
                    [e[2] for e in first_entries], dtype=torch.float32, device=DEVICE
                ),
            ),
            max(len(first_entries), 1),
        )
        if fr is not None:
            tot["first_row"] += float(fr.detach())
        with torch.no_grad():
            lp_full = F.log_softmax(logits.float(), dim=-1)
            for r in et.rows:
                gi = r.idx - 1
                if gi < 0 or gi >= logits.shape[0]:
                    continue
                p_chosen = float(lp_full[gi, r.chosen_id].exp())
                if r.idx == et.first_row_idx:
                    chosen_first.append(p_chosen)
                elif r.idx not in et.control_rows():
                    chosen_content.append(p_chosen)
                if et.eot_row_idx is not None and r.idx == et.eot_row_idx:
                    eot_natural.append(float(lp_full[gi, EOT].exp()))
        if cache is not None:

            def grab():
                ks = [
                    (
                        p.grad.detach().float().clone()
                        if p.grad is not None
                        else torch.zeros_like(p, dtype=torch.float32)
                    )
                    for p in cache.trainable_keys
                ]
                vs = [
                    (
                        p.grad.detach().float().clone()
                        if p.grad is not None
                        else torch.zeros_like(p, dtype=torch.float32)
                    )
                    for p in cache.trainable_values
                ]
                return ks, vs

            def flat(ks, vs):
                return torch.cat([t.flatten() for t in ks + vs])

            if anc is not None:
                uni.backward(retain_graph=True)
                ku, vu = grab()
                for p in cache.parameters():
                    p.grad = None
                anc.backward()
                ka, va = grab()
                gu, ga = flat(ku, vu), flat(ka, va)
                g_unique = gu if g_unique is None else g_unique + gu
                g_anchor = ga if g_anchor is None else g_anchor + ga
                # the legacy gradient is the branch sum
                kv_energy["k"] += sum(
                    float((a + b).pow(2).sum()) for a, b in zip(ku, ka)
                )
                kv_energy["v"] += sum(
                    float((a + b).pow(2).sum()) for a, b in zip(vu, va)
                )
            else:
                uni.backward()
                ku, vu = grab()
                gu = flat(ku, vu)
                g_unique = gu if g_unique is None else g_unique + gu
                kv_energy["k"] += sum(float(t.pow(2).sum()) for t in ku)
                kv_energy["v"] += sum(float(t.pow(2).sum()) for t in vu)

    report = dict(
        probe_n=n_ok,
        loss_legacy=tot["legacy"] / max(n_ok, 1),
        loss_unique=tot["unique"] / max(n_ok, 1),
        loss_anchor=tot["anchor"] / max(n_ok, 1),
        loss_first_row=tot["first_row"] / max(n_ok, 1),
        chosen_p_first_mean=statistics.fmean(chosen_first) if chosen_first else None,
        chosen_p_content_mean=(
            statistics.fmean(chosen_content) if chosen_content else None
        ),
        eot_p_natural_mean=statistics.fmean(eot_natural) if eot_natural else None,
    )
    if g_unique is not None:
        report["grad_norm_unique"] = float(g_unique.norm())
        if g_anchor is not None:
            report["grad_norm_anchor"] = float(g_anchor.norm())
            report["grad_cosine"] = float(
                (g_unique @ g_anchor)
                / (g_unique.norm() * g_anchor.norm()).clamp_min(1e-12)
            )
    if cache is not None:
        report["kv_grad_energy_ratio"] = kv_energy["k"] / max(kv_energy["v"], 1e-12)
        for p in cache.parameters():
            p.grad = None
    return report


def main():
    Path(os.path.dirname(OUT_JSON)).mkdir(parents=True, exist_ok=True)
    conditions = [("no_cartridge", None)]
    for spec in CARTS.split(","):
        name, path = spec.split("=", 1)
        conditions.append((name, path))

    report = dict(
        evaluator=EVALUATOR_VERSION,
        model=MODEL,
        patient=PATIENT,
        max_q=MAX_Q,
        probe_ids=PROBE_IDS,
        results={},
    )
    raws = {}
    for name, path in conditions:
        cache = load_cart(path) if path else None
        primary, rows_p = generation_eval(cache, thinking=False, cap=32)
        stress, rows_s = generation_eval(cache, thinking=True, cap=256)
        fc = forced_choice_eval(cache)
        probe = probe_eval(cache) if cache is not None else None
        report["results"][name] = dict(
            primary=primary, stress=stress, forced_choice=fc, probe=probe
        )
        raws[name] = dict(primary=rows_p, stress=rows_s)
        print(
            f"[ctrl-eval] {name}: strict={primary['strict_acc']:.3f} "
            f"invalid={primary['parser_invalid']:.2f} len={primary['mean_len']:.1f} "
            f"| fc={fc['fc_acc']:.3f} margin={fc['margin_mean']:.3f} "
            f"| stress={stress['strict_acc']:.3f}",
            flush=True,
        )
        if cache is not None:
            del cache
            torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    with open(OUT_JSON.replace(".json", "_raw.json"), "w") as f:
        json.dump(raws, f, indent=1)
    print(f"CTRL_EVAL_DONE {OUT_JSON}")


if __name__ == "__main__":
    main()
