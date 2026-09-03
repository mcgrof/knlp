#!/usr/bin/env python3
"""CAS-faithful LongHealth evaluator (the paper's Table 15 protocol).

Implements the Cartridges-at-Scale (arXiv:2606.04557) evaluation exactly:
  - System prompt + user prompt verbatim from Appendix H Table 15 (LongHealth
    row): the model answers with the exact OPTION TEXT wrapped in <answer> tags,
    not a letter.
  - Sampling per Appendix A: thinking enabled, temperature 0.6, max 2048
    completion tokens. (Top-p/top-k are unspecified in the paper; the vLLM path
    inherits Qwen3's generation_config defaults, the flex path uses plain
    temperature sampling — both noted in the output.)
  - Scoring per Appendix D: extract the <answer> tag contents, compare against
    all five options by string similarity (difflib ratio), closest match wins.
  - Aggregate accuracy over questions, averaged over RUNS runs (paper: >=3),
    mean +/- std reported.

Modes (MODE env):
  nocontext  system + question only, via a vLLM OpenAI endpoint. Paper: 37.5.
  fulldoc    the gold patient's FULL record text prepended to the user message
             (Appendix H; Table 3 "Oracle (1 doc)" -- the single gold document,
             not the 20-doc corpus). Paper: 87.4. Via vLLM.
  cart       a trained cartridge as TrainableCache prefix, no document text in
             the prompt (Appendix H cartridge setting). Local flex path with a
             plain sampling loop that terminates on the model's EOS set or the
             2048-token cap — the same termination protocol as the vLLM arms
             (no answer-tag early stop: it could fire on a rehearsed tag inside
             the think block and truncate reasoning).

Env: MODE, PATIENTS (space-sep; default all 20), MAX_Q (default 20), RUNS
(default 3), VLLM_URL (baselines), CART_DIR (cart mode), COLLAPSE=1 (cart mode:
co-load ALL patients' carts in one prefix and answer every patient's questions
against it -- the interference measurement), CONCURRENCY (baselines, default
32), OUT_JSON, DEVICE (cart mode), SINK_MAX (cart reconstruction),
MAX_COMPLETION (completion cap in tokens, thinking included; default 2048,
recorded in the output JSON -- every published number was scored at 2048),
SAVE_RAW=1 (cart mode: dump every generation plus a tagged-vs-untagged
accuracy breakdown to OUT_JSON.raw.json -- needed because a missing
answer tag is NOT scored as wrong, it falls back to the last 300 chars,
so the aggregate counts cannot tell format failure from knowledge
failure).
"""

import os
import json
import re
import sys
import time
import difflib
import statistics
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("CARTRIDGES_DIR", os.path.expanduser("~/cartridges"))
os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", os.path.expanduser("~/cas_out"))
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

MODE = os.environ.get("MODE", "nocontext")
PATIENTS = os.environ.get(
    "PATIENTS", " ".join(f"patient_{i:02d}" for i in range(1, 21))
).split()
MAX_Q = int(os.environ.get("MAX_Q", "20"))
RUNS = int(os.environ.get("RUNS", "3"))
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8005/v1")
CART_DIR = os.environ.get("CART_DIR", "")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "32"))
OUT_JSON = os.environ.get(
    "OUT_JSON", os.path.expanduser(f"~/cas_out/eval_t15_{MODE}.json")
)
DEVICE = os.environ.get("DEVICE", "cuda:0")
SINK_MAX = int(os.environ.get("SINK_MAX", "4"))
# Per-question outcomes and raw generations. Off by default (they are
# large), but without them a tag-missing rate is undiagnosable: the
# scorer FALLS BACK to the last 300 chars when the tag is absent, so a
# missing tag is not the same as a wrong answer and the aggregate
# counts cannot separate the two.
SAVE_RAW = os.environ.get("SAVE_RAW", "0") == "1"
RAW_ROWS = []
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
# Completion cap in tokens (thinking + answer). 2048 is the value every
# published number here was scored at; raise it to see how many of the
# tag-missing answers are the cap and not the model
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "2048"))
TEMPERATURE = 0.6

# --- Table 15 (Appendix H), LongHealth row, verbatim -------------------------
SYSTEM_PROMPT = (
    "Please reference the patient medical records to answer the user's "
    "questions. Choose the single best option and provide your answer exactly "
    "as it appears in the options.\n\n"
    "Wrap your answer in: <answer> The correct option text here </answer>"
)


def user_prompt(q, doc_text=None):
    """Table 15 user turn; fulldoc mode prepends the gold record (Appendix H)."""
    body = (
        f"{q.question}\n\n"
        f"A. {q.answer_a}  B. {q.answer_b}  C. {q.answer_c}  "
        f"D. {q.answer_d}  E. {q.answer_e}"
    )
    if doc_text is not None:
        return f"{doc_text}\n\n{body}"
    return body


# --- Appendix D scoring: fuzzy option matching -------------------------------
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _norm(s):
    return " ".join(s.lower().split())


def correct_index(q):
    """Index of the correct option, matched once by normalized text (guards
    against whitespace/case drift between q.correct and answer_a..e)."""
    options = [q.answer_a, q.answer_b, q.answer_c, q.answer_d, q.answer_e]
    hits = [i for i, o in enumerate(options) if _norm(o) == _norm(q.correct)]
    assert len(hits) == 1, f"{q.question_id}: correct matches {len(hits)} options"
    return hits[0]


def score_response(text, q):
    """Returns (correct: 0/1, tag_found: bool). Strips the thinking block,
    extracts the last <answer> tag (fallback: the post-think tail), and picks
    the closest of the five options by difflib ratio. An empty/garbage
    prediction (no positive similarity) scores incorrect rather than
    tie-breaking to option A."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    matches = ANSWER_RE.findall(text)
    tag_found = bool(matches)
    pred = matches[-1].strip() if matches else text.strip()[-300:]
    options = [q.answer_a, q.answer_b, q.answer_c, q.answer_d, q.answer_e]

    def sim(a, b):
        return difflib.SequenceMatcher(
            None, a.lower().strip(), b.lower().strip()
        ).ratio()

    if not pred:
        return 0, tag_found
    ratios = [sim(pred, opt) for opt in options]
    if max(ratios) <= 0.0:
        return 0, tag_found
    return int(ratios.index(max(ratios)) == correct_index(q)), tag_found


def aggregate(per_run):
    """per_run: list of (correct, total, tags_missing) -> summary dict."""
    accs = [c / max(t, 1) for c, t, _ in per_run]
    return {
        "runs": len(per_run),
        "acc_mean": round(statistics.mean(accs), 4),
        "acc_std": round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
        "per_run_acc": [round(a, 4) for a in accs],
        "per_run_counts": [
            {"correct": c, "total": t, "answer_tag_missing": m} for c, t, m in per_run
        ],
    }


# --- baseline modes (vLLM OpenAI endpoint) -----------------------------------
def run_vllm_mode(patients_data):
    from openai import OpenAI

    client = OpenAI(base_url=VLLM_URL, api_key="EMPTY", timeout=600)
    jobs = []  # (run, patient, q, doc_text_or_None)
    for run in range(RUNS):
        for patient in patients_data:
            doc = "\n\n".join(patient.texts.values()) if MODE == "fulldoc" else None
            for q in patient.questions[:MAX_Q]:
                jobs.append((run, patient.patient_id, q, doc))

    import zlib

    def one(job):
        run, pid, q, doc = job
        last_err = None
        for attempt in range(2):  # one retry on transient errors
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(q, doc)},
                    ],
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION,
                    seed=run * 100003
                    + zlib.crc32(q.question_id.encode()) % 100003
                    + attempt,
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                )
                text = resp.choices[0].message.content or ""
                # vLLM may surface thinking via a separate field; fold it in so
                # the </think> strip in score_response stays a no-op if absent
                reasoning = getattr(resp.choices[0].message, "reasoning_content", None)
                if reasoning and "<think>" not in text:
                    text = f"<think>{reasoning}</think>{text}"
                corr, tag = score_response(text, q)
                return (run, pid, corr, tag, None)
            except Exception as e:  # noqa: BLE001 - retry once, then record
                last_err = str(e)[:200]
        return (run, pid, 0, False, last_err)

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for i, r in enumerate(ex.map(one, jobs)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(
                    f"  [{MODE}] {i + 1}/{len(jobs)} t={time.time() - t0:.0f}s",
                    flush=True,
                )
    errors = [r[4] for r in results if r[4]]
    # A biased summary is worse than no summary: refuse to report when more
    # than 2% of requests failed even after a retry (e.g. server died mid-run).
    if len(errors) > 0.02 * len(jobs):
        raise SystemExit(
            f"[{MODE}] {len(errors)}/{len(jobs)} requests errored (>2%); "
            f"refusing to write a biased summary. First: {errors[0]}"
        )
    ok = [r for r in results if not r[4]]
    assert ok, "no successful requests"
    per_run = []
    for run in range(RUNS):
        rr = [r for r in ok if r[0] == run]
        per_run.append((sum(r[2] for r in rr), len(rr), sum(1 for r in rr if not r[3])))
    per_patient = {}
    for pid in {r[1] for r in ok}:
        pr = [r for r in ok if r[1] == pid]
        per_patient[pid] = round(sum(r[2] for r in pr) / max(len(pr), 1), 4)
    if errors:
        print(
            f"  [{MODE}] WARNING: {len(errors)} errored requests excluded", flush=True
        )
    return per_run, per_patient, errors[:10]


# --- cart mode (flex path, stop-on-</answer> loop) ---------------------------
def run_cart_mode(patients_data):
    import torch
    from transformers import AutoTokenizer
    from cartridges.cache import AttnConfig, TrainableCache
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    di = int(DEVICE.split(":")[1]) if ":" in DEVICE else 0
    torch.cuda.set_device(di)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = FlexQwen3ForCausalLM.from_pretrained(MODEL).to(DEVICE).to(torch.bfloat16)
    model.eval()
    ac = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )

    def load_cart_kv(path):
        """Per-layer (K, V) lists: [frozen sink | trainable] when frozen is a
        sink; trainable-only when frozen is a distractor block (same rule as
        cas_combine_eval)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)

        def t(p):
            return torch.as_tensor(p.data if hasattr(p, "data") else p).to(
                torch.bfloat16
            )

        fk = ck.get("frozen_keys") or []
        nfrozen = t(fk[0]).shape[2] if fk else 0
        use_frozen = 0 < nfrozen <= SINK_MAX

        def cat(fro, tra):
            tt = [t(p) for p in tra]
            if fro and use_frozen:
                ff = [t(p) for p in fro]
                return [torch.cat([ff[i], tt[i]], dim=2) for i in range(len(tt))]
            return tt

        return (
            cat(ck.get("frozen_keys"), ck["trainable_keys"]),
            cat(ck.get("frozen_values"), ck["trainable_values"]),
        )

    def make_cache(paths):
        """One TrainableCache holding the given carts concatenated in order —
        a single cart for oracle mode, all carts for collapse mode."""
        per = [load_cart_kv(p) for p in paths]
        nl = ac.n_layers
        ik = [
            torch.cat([per[c][0][li] for c in range(len(per))], dim=2)
            for li in range(nl)
        ]
        iv = [
            torch.cat([per[c][1][li] for c in range(len(per))], dim=2)
            for li in range(nl)
        ]
        return TrainableCache(config=ac, init_keys=ik, init_values=iv).to(DEVICE)

    # EOS set from the model's generation config (Qwen3 has two EOS ids); the
    # vLLM arm stops on the same set, keeping the termination protocol identical
    eos_cfg = getattr(model.generation_config, "eos_token_id", None)
    if eos_cfg is None:
        eos_cfg = [tok.eos_token_id]
    stop_ids = set(eos_cfg if isinstance(eos_cfg, (list, tuple)) else [eos_cfg])

    @torch.no_grad()
    def generate(cache, ids, seed):
        """Single-sequence sampling loop mirroring flex_generate (temperature-
        scaled multinomial, fp32 softmax); terminates on the model EOS set or
        the MAX_COMPLETION cap — same protocol as the vLLM arms."""
        torch.manual_seed(seed)
        cache.clear()
        cur_ids = ids
        pos = torch.arange(ids.shape[0], device=DEVICE)
        sids = torch.zeros(ids.shape[0], dtype=torch.long, device=DEVICE)
        out = []
        for _ in range(MAX_COMPLETION):
            o = model(
                input_ids=cur_ids,
                seq_ids=sids,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                mode="generate",
            )
            logits = o.logits[0, -1].float() / TEMPERATURE
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
            if nxt in stop_ids:
                break
            out.append(nxt)
            cur_ids = torch.tensor([nxt], device=DEVICE)
            pos = pos[-1:] + 1
            sids = sids[-1:]
        cache.clear()
        return tok.decode(out, skip_special_tokens=True)

    per_run = []
    per_patient_hits = {}
    skipped = sorted(
        {
            p.patient_id
            for p in patients_data
            if not os.path.exists(os.path.join(CART_DIR, f"{p.patient_id}.pt"))
        }
    )
    if skipped:
        print(
            f"  [cart] WARNING: no cartridge for {skipped}; aggregate covers "
            f"{len(patients_data) - len(skipped)} patients",
            flush=True,
        )
    collapse = os.environ.get("COLLAPSE", "0") == "1"
    present = [p for p in patients_data if p.patient_id not in skipped]
    shared_cache = None
    if collapse:
        # All carts co-loaded in one prefix (concatenated in patient order);
        # every patient's questions are answered against the combined cache.
        shared_cache = make_cache(
            [os.path.join(CART_DIR, f"{p.patient_id}.pt") for p in present]
        )
        print(
            f"  [cart] COLLAPSE mode: {len(present)} carts co-loaded, "
            f"{shared_cache.num_cartridge_tokens()} prefix tokens",
            flush=True,
        )
    t0 = time.time()
    for run in range(RUNS):
        c = t = miss = 0
        for patient in patients_data:
            cart = os.path.join(CART_DIR, f"{patient.patient_id}.pt")
            if not os.path.exists(cart):
                continue
            cache = shared_cache if collapse else make_cache([cart])
            for q in patient.questions[:MAX_Q]:
                ids = (
                    tok.apply_chat_template(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt(q)},
                        ],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        enable_thinking=True,
                    )
                    .to(DEVICE)
                    .flatten()
                )
                text = generate(cache, ids, seed=run * 7919 + t)
                corr, tag = score_response(text, q)
                if SAVE_RAW:
                    RAW_ROWS.append(
                        dict(
                            run=run,
                            patient=patient.patient_id,
                            qid=q.question_id,
                            correct=int(corr),
                            tag_found=bool(tag),
                            n_chars=len(text),
                            truncated="</think>" not in text,
                            text=text,
                        )
                    )
                c += corr
                t += 1
                miss += 0 if tag else 1
                key = (patient.patient_id, run)
                per_patient_hits[key] = per_patient_hits.get(key, [0, 0])
                per_patient_hits[key][0] += corr
                per_patient_hits[key][1] += 1
            if not collapse:
                del cache
                torch.cuda.empty_cache()
            print(
                f"  [cart] run{run} {patient.patient_id} done "
                f"t={time.time() - t0:.0f}s",
                flush=True,
            )
        per_run.append((c, t, miss))
    assert any(
        t > 0 for _, t, _ in per_run
    ), f"no cartridges evaluated (CART_DIR={CART_DIR}, skipped={skipped})"
    per_patient = {}
    for (pid, _run), (hc, ht) in per_patient_hits.items():
        a, b = per_patient.get(pid, (0, 0))
        per_patient[pid] = (a + hc, b + ht)
    per_patient = {pid: round(a / max(b, 1), 4) for pid, (a, b) in per_patient.items()}
    return per_run, per_patient, [f"skipped_no_cart: {skipped}"] if skipped else []


def main():
    from cartridges.data.longhealth.utils import load_longhealth_dataset

    patients_data = load_longhealth_dataset(PATIENTS)
    print(
        f"[eval-t15] mode={MODE} patients={len(patients_data)} max_q={MAX_Q} "
        f"runs={RUNS}",
        flush=True,
    )
    if MODE in ("nocontext", "fulldoc"):
        per_run, per_patient, errors = run_vllm_mode(patients_data)
    elif MODE == "cart":
        assert CART_DIR, "cart mode needs CART_DIR"
        per_run, per_patient, errors = run_cart_mode(patients_data)
    else:
        raise SystemExit(f"unknown MODE {MODE}")

    res = {
        "protocol": "CAS Table 15 (system+user verbatim), temp 0.6, thinking "
        f"on, max {MAX_COMPLETION}, fuzzy option match (difflib), "
        f"runs={RUNS}; top-p/k: vLLM=model generation_config "
        "defaults, flex=plain temperature sampling (paper leaves "
        "them unspecified)",
        "mode": MODE,
        "model": MODEL,
        "patients": [p.patient_id for p in patients_data],
        "max_q": MAX_Q,
        "max_completion": MAX_COMPLETION,
        "cart_dir": CART_DIR or None,
        "cart_collapse": os.environ.get("COLLAPSE", "0") == "1",
        "summary": aggregate(per_run),
        "per_patient_acc": dict(sorted(per_patient.items())),
        "errors_sample": errors,
        "paper_reference": {
            "nocontext": 37.5,
            "fulldoc_oracle_1doc": 87.4,
            "isolated_cart_aggregate": 73.6,
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    if SAVE_RAW and RAW_ROWS:
        # separate file: the raws are large and the summary should stay
        # cheap to read
        raw_path = OUT_JSON.replace(".json", ".raw.json")
        json.dump(RAW_ROWS, open(raw_path, "w"), indent=1)
        res["raw_path"] = raw_path
        tagged = [r for r in RAW_ROWS if r["tag_found"]]
        untagged = [r for r in RAW_ROWS if not r["tag_found"]]
        res["tag_breakdown"] = dict(
            tagged_n=len(tagged),
            tagged_acc=(
                round(sum(r["correct"] for r in tagged) / len(tagged), 4)
                if tagged
                else None
            ),
            untagged_n=len(untagged),
            untagged_acc=(
                round(sum(r["correct"] for r in untagged) / len(untagged), 4)
                if untagged
                else None
            ),
            untagged_truncated=sum(1 for r in untagged if r["truncated"]),
        )
    json.dump(res, open(OUT_JSON, "w"), indent=2)
    print(json.dumps(res["summary"], indent=2))
    print(
        f"CAS_EVAL_T15_DONE mode={MODE} acc={res['summary']['acc_mean']} "
        f"+/-{res['summary']['acc_std']} -> {OUT_JSON}",
        flush=True,
    )


if __name__ == "__main__":
    main()
