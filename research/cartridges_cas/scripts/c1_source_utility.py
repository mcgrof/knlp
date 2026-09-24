#!/usr/bin/env python3
"""Does the saved 4B cartridge carry anything useful about its patients?

A screen, not a verdict. Three arms answer the same twenty questions through
the same model and the same generation path, differing only in what the model
is given: the saved cartridge as a key/value prefix, nothing at all, or the
gold patient's record in the prompt. The middle arm is the floor and the last
is the ceiling; if the cartridge does not clear the floor by a useful margin
there is nothing to build a receiver pilot on.

Three things this is careful about, each because of a specific defect already
found in this lane.

The protocol is the one the cartridge was trained and evaluated under, which
turns out to hinge on the capitalisation of the model name: the library keys
``MODELS_WITH_THINKING`` on ``tokenizer.name_or_path`` by exact string, the
training config used ``Qwen/Qwen3-4b``, and loading the canonical
``Qwen/Qwen3-4B`` silently swaps in a different instruction that asks for a
``<thinking>`` block. Both the model and the tokenizer are therefore pinned to
the lowercase spelling, and the run refuses to start if the match fails.

Scoring never converts an unparseable answer into a point. The library's own
scorer falls back to option A when it finds no ``<answer>`` tag -- its comment
says "random guess" and the code always picks A -- so an arm that never
terminates properly collects whatever fraction of the time A happens to be
right. Both scores are reported: the library's, so this is comparable with the
number recorded beside the checkpoint, and a strict one that counts an
unparseable answer as wrong. The gate reads the strict one.

The answer is taken from after the reasoning block and from the last tag, not
the first. A thinking model rehearses the answer format inside its reasoning,
and scoring the rehearsal rather than the conclusion is a way to be wrong in
both directions at once.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import sys
import time

MODEL_NAME = "Qwen/Qwen3-4b"  # lowercase: see the module docstring
PATIENTS = [f"patient_{i:02d}" for i in range(1, 11)]
ARMS = ("nocart", "source", "fulldoc")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from research.cartridges_cas.scripts.score_longhealth import (  # noqa: E402
    UNRESOLVED,
    norm as normalise,
    score as strict_score,
    summarise,
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def faithful_score(response, options, gold):
    """The library's own scorer, reproduced so the two can be compared.

    Deliberately identical to ``LongHealthMultipleChoiceGenerateDataset.score``,
    fallback included, because its purpose here is comparability with the
    number recorded beside the checkpoint -- not correctness. It searches the
    first tag in the raw text, does not strip the reasoning block, and when it
    finds no tag it scores the first option.
    """
    norm_opts = [normalise(o) for o in options]
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        pred = m.group(1).strip().lower()
        best = max(
            norm_opts, key=lambda x: difflib.SequenceMatcher(None, pred, x).ratio()
        )
        return float(best == normalise(gold)), False
    return float(norm_opts[0] == normalise(gold)), True  # falls back to option A


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------


def build_record_text(patient):
    """The gold patient's record, as the ceiling arm sees it."""
    from cartridges.data.longhealth.resources import FULL_STRING_TEMPLATE

    notes = "\n".join(f"<{nid}>\n{t}\n</{nid}>" for nid, t in patient.texts.items())
    return FULL_STRING_TEMPLATE.format(
        patient_id=patient.patient_id,
        name=patient.name,
        birthday=patient.birthday,
        diagnosis=patient.diagnosis,
        num_notes=len(patient.texts),
        notes=notes,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartridge", required=True, help="the saved source cartridge .pt")
    ap.add_argument(
        "--cartridge-sha256",
        default="",
        help="refuse to run unless the artifact hashes to this",
    )
    ap.add_argument(
        "--dataset-json",
        required=True,
        help="the staged benchmark. The library fetches it from a moving "
        "branch ref over the network on every call, which is neither an input "
        "nor available on a machine without egress.",
    )
    ap.add_argument("--dataset-sha256", default="")
    ap.add_argument("--per-patient", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--total-token-budget",
        type=int,
        default=30720,
        help="the plan's ceiling on generated tokens across every arm",
    )
    ap.add_argument("--num-frozen-tokens", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--prefill-chunk",
        type=int,
        default=0,
        help="fill the cache this many prompt tokens at a time instead of in "
        "one forward. Attention's working set scales with the query positions "
        "in a single call, so a record-length prompt needs this on a card "
        "running the uncompiled kernel. 0 disables it.",
    )
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--experiment-id", default="cartridge-source-utility")
    args = ap.parse_args()

    os.environ.setdefault("CARTRIDGES_DIR", "/data/cartridges")
    os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cartridges_out")
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    sys.path.insert(0, os.environ["CARTRIDGES_DIR"])
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from transformers import AutoTokenizer

    from cartridges.cache import AttnConfig, TrainableCache
    from cartridges.data.longhealth.utils import load_longhealth_dataset
    from cartridges.generation import flex_generate
    from cartridges.initialization.tokenization_utils import MODELS_WITH_THINKING
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"INVALID: unknown arm {a!r}")

    if args.cartridge_sha256:
        h = hashlib.sha256()
        with open(args.cartridge, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        if h.hexdigest() != args.cartridge_sha256:
            raise SystemExit(
                f"INVALID: cartridge hashes to {h.hexdigest()}, expected "
                f"{args.cartridge_sha256}"
            )
        log(f"cartridge verified {h.hexdigest()[:16]}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # The instruction the model receives depends on this exact string. If the
    # match fails the run would quietly measure a different protocol from the
    # one the cartridge was built for.
    if tok.name_or_path not in MODELS_WITH_THINKING:
        raise SystemExit(
            f"INVALID: tokenizer identifies as {tok.name_or_path!r}, which the "
            "library does not list as a thinking model, so the question would "
            "carry a different instruction than the one this cartridge was "
            "trained and evaluated under"
        )
    log(f"tokenizer {tok.name_or_path} matches the training protocol")

    # The checkout this runs from carries a local edit that replaces the
    # compiled flex kernel with the eager one, for a card that cannot compile
    # it. On this card the eager path materialises a full float32 score matrix
    # -- about twelve bytes per head-query-key element, some tens of gigabytes
    # for a record-length prompt -- and its numerics differ from the compiled
    # kernel's, so a result produced under it is neither affordable nor
    # comparable with one that is not. Checked here rather than discovered as
    # an out-of-memory error partway through the third arm.
    from cartridges.models import attention as _att

    compiled = hasattr(_att.flex_attention_generate, "_torchdynamo_orig_callable")
    # A ROCm device reports itself as cuda, and is the one card that genuinely
    # cannot compile the kernel, so it is exempt -- with the consequence
    # recorded in the result rather than waved through: its numerics are not
    # the compiled kernel's, so a score from it is not comparable with one
    # from a machine that compiled.
    is_rocm = torch.version.hip is not None
    if dev == "cuda" and not is_rocm and not compiled:
        raise SystemExit(
            "INVALID: flex attention is not compiled on this device. The "
            "eager path materialises the full score matrix and changes the "
            "numerics; set CARTRIDGES_FLEX_COMPILE=1 or restore the compiled "
            "aliases in cartridges/models/attention.py before running."
        )
    log(f"flex attention compiled: {compiled} (device {dev}, rocm={is_rocm})")
    if not compiled:
        log(
            "  NOTE: the eager attention path is in use; its numerics differ "
            "from the compiled kernel, so these scores are not comparable "
            "with a run that compiled"
        )

    bench_sha, patients = load_patients_pinned(
        args.dataset_json, PATIENTS, args.dataset_sha256
    )
    log(f"benchmark {bench_sha[:16]} pinned, {len(patients)} patients")
    by_id = {p.patient_id: p for p in patients}
    questions = select_two_per_patient(patients, per_patient=args.per_patient)
    log(f"{len(questions)} questions: {', '.join(q.question_id for q in questions)}")

    ds_cfg = {"wrap": make_wrapper(tok)}

    model = FlexQwen3ForCausalLM.from_pretrained(MODEL_NAME).to(dev).to(torch.bfloat16)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    log(f"model {model.config._name_or_path}, {model.config.num_hidden_layers} layers")

    cartridge, n_frozen = None, None
    if "source" in arms:
        cartridge, n_frozen = load_cartridge(args.cartridge, model, dev, torch.bfloat16)
        log(
            f"cartridge {cartridge.num_cartridge_tokens()} prefix tokens "
            f"({n_frozen} frozen)"
        )

    rows_by_arm, spent_total, t0 = {}, 0, time.time()
    for arm in arms:
        log(f"--- {arm} ---")
        rows, spent = run_arm(
            arm,
            questions,
            by_id,
            model=model,
            tok=tok,
            ds_cfg=ds_cfg,
            cartridge=cartridge,
            max_new_tokens=args.max_new_tokens,
            device=dev,
            budget=args.total_token_budget - spent_total,
            prefill_chunk=args.prefill_chunk,
        )
        rows_by_arm[arm] = rows
        spent_total += spent
        # Persist as each arm finishes. The last arm is the one that puts a
        # record-length prompt through the model, so it is the one most likely
        # to run out of memory, and losing the two cheap arms with it would
        # mean paying for them twice.
        with open(os.path.join(args.out_dir, f"rows.{arm}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        log(f"  {arm} persisted, {spent_total} generated tokens so far")

    report = finish(
        args,
        rows_by_arm,
        {
            "model": MODEL_NAME,
            "cartridge": os.path.abspath(args.cartridge),
            "cartridge_sha256": args.cartridge_sha256 or None,
            "cartridge_prefix_tokens": (
                int(cartridge.num_cartridge_tokens()) if cartridge is not None else None
            ),
            "benchmark_sha256": bench_sha,
            "benchmark_path": os.path.abspath(args.dataset_json),
            "question_ids": [q.question_id for q in questions],
            "per_patient": args.per_patient,
            "arms": arms,
            "decoding": "greedy (temperature 0.0)",
            "decoding_note": (
                "the checkpoint's own recorded evaluation sampled at 0.3; a "
                "single sample at that temperature would put several questions "
                "of noise on a twenty-question screen, so every arm here is "
                "greedy and identical"
            ),
            "max_new_tokens": args.max_new_tokens,
            "generated_tokens_total": spent_total,
            "generated_token_budget": args.total_token_budget,
            "flex_attention_compiled": compiled,
            "rocm": is_rocm,
            "comparability_note": (
                None
                if compiled
                else "eager attention path: numerics differ from the compiled "
                "kernel, not comparable across devices"
            ),
            "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
            "torch": torch.__version__,
            "elapsed_s": round(time.time() - t0, 1),
        },
    )
    log(f"rows and verdict in {args.out_dir}")
    return 0 if report["gate"]["advances"] else 3


def make_wrapper(tok):
    """The exact question text the checkpoint was trained and evaluated with.

    Taken from the library's own eval dataset rather than paraphrased, because
    the instruction it appends depends on whether the tokenizer's name is in
    the library's thinking-model set, and getting that wrong measures a
    different protocol.
    """
    from cartridges.initialization.tokenization_utils import MODELS_WITH_THINKING

    def wrap(question, patient):
        options = "\n".join(getattr(question, f"answer_{c}") for c in "abcde")
        if tok.name_or_path not in MODELS_WITH_THINKING:
            cot = (
                "You should first think step by step. Then give your final "
                "answer exactly as it appears in the options. Your output "
                "should be in the following format: \n<thinking> "
                "{YOUR_THOUGHT_PROCESS} </thinking> "
            )
        else:
            cot = (
                "Please provide your answer exactly as it appears in the "
                "options with the following format:"
            )
        info = (
            f"ID {patient.patient_id}, Name: {patient.name}, "
            f"Birthday: {patient.birthday}, Diagnosis: {patient.diagnosis}"
        )
        return (
            f"Please answer the question below about the following patient: {info}"
            f"\n\n<question>\n{question.question}\n</question>"
            f"\n\n<options>\n{options}\n</options>\n{cot}"
            "\n\n<answer>\n{YOUR_ANSWER}\n</answer>"
        )

    return wrap


def load_patients_pinned(path, patient_ids, want_sha256=""):
    """Read the benchmark from a file instead of from a branch.

    The library fetches the dataset over the network from
    ``refs/heads/main`` on every call. A moving ref is not an input: an
    upstream edit would renumber the questions under a recorded id list with
    no error anywhere, and it also makes the evaluation depend on the rented
    machine reaching GitHub. The file is hashed and the hash is recorded with
    the result.
    """
    import json as _json

    # Only the record type, not the loader: importing the loader's module
    # drags in the library's dataset and plotting stack, none of which this
    # needs, and one of which fetches the benchmark over the network.
    from cartridges.data.longhealth.utils import LongHealthPatient

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    got = h.hexdigest()
    if want_sha256 and got != want_sha256:
        raise SystemExit(f"INVALID: benchmark hashes to {got}, expected {want_sha256}")
    data = _json.loads(open(path, encoding="utf-8").read())
    for pid, row in data.items():
        for q in row["questions"]:
            q["question_id"] = f"{pid}_{q['No']}"
    return got, [
        LongHealthPatient(patient_id=pid, **row)
        for pid, row in data.items()
        if patient_ids is None or pid in patient_ids
    ]


def select_two_per_patient(patients, tag="c1-source-utility-v1", per_patient=2):
    """Two questions per patient, ranked by a hash of the question id.

    Independent of patient order, of list order within a patient, and of any
    language's random number generator: the same twenty ids fall out of the
    ids alone. Questions whose five options are not all distinct are dropped
    because no scorer can resolve them -- one question in this pool repeats an
    option across two slots -- and that is a validity filter, not a filter on
    which answer is correct. Nothing here is conditioned on the answer, so the
    draw stays uniform over answer slots.
    """
    picked = []
    for p in sorted(patients, key=lambda x: x.patient_id):
        eligible = [
            q
            for q in p.questions
            if len({normalise(getattr(q, f"answer_{s}")) for s in "abcde"}) == 5
        ]
        if len(eligible) < per_patient:
            raise SystemExit(
                f"INVALID: {p.patient_id} has {len(eligible)} scorable questions"
            )
        ranked = sorted(
            eligible,
            key=lambda q: hashlib.sha256(f"{tag}|{q.question_id}".encode()).hexdigest(),
        )[:per_patient]
        picked.extend(
            sorted(ranked, key=lambda q: int(q.question_id.rsplit("_", 1)[1]))
        )
    return picked


def load_cartridge(path, model, device, dtype):
    """Rebuild the saved cartridge as a key/value prefix.

    Two shape traps sit here. The head width is ``config.head_dim`` and not
    ``hidden_size // num_attention_heads``; for this pair those are 128 and 80
    and the wrong one aborts. And the right one does not identify the model:
    Qwen3-4b and Qwen3-8B both present as thirty-six layers, eight key/value
    heads and a head width of 128, so a 4B cartridge loaded against an 8B
    model passes every shape check and answers with nonsense. The model name
    is therefore asserted too.
    """
    import torch

    from cartridges.cache import AttnConfig, TrainableCache

    ck = torch.load(path, map_location="cpu", weights_only=False)
    fk, fv = ck.get("frozen_keys") or [], ck.get("frozen_values") or []
    tk, tv = ck["trainable_keys"], ck["trainable_values"]
    n_frozen = int(fk[0].shape[2]) if len(fk) else 0

    def merge(fro, tra):  # frozen first, on the token dim
        if not len(fro):
            return [p.detach().to(dtype).contiguous() for p in tra]
        return [
            torch.cat([f.detach(), t.detach()], dim=2).to(dtype).contiguous()
            for f, t in zip(fro, tra)
        ]

    ac = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )
    got = (len(tk), int(tk[0].shape[1]), int(tk[0].shape[3]))
    want = (ac.n_layers, ac.n_heads, ac.head_dim)
    if got != want:
        raise SystemExit(f"INVALID: cartridge geometry {got} != model {want}")
    name = str(getattr(model.config, "_name_or_path", "")).lower()
    if "qwen3-4b" not in name:
        raise SystemExit(
            f"INVALID: this cartridge was trained on Qwen/Qwen3-4b but the "
            f"model identifies as {name!r}; the shapes match for Qwen3-8B too, "
            "so shape agreement does not establish the pair"
        )
    cache = (
        TrainableCache(
            config=ac,
            init_keys=merge(fk, tk),
            init_values=merge(fv, tv),
            num_frozen_tokens=n_frozen,
        )
        .to(device)
        .to(dtype)
    )
    for p in cache.parameters():
        p.requires_grad_(False)
    del ck
    return cache, n_frozen


def prefill_in_chunks(model, ids, cache, chunk, device):
    """Fill the cache with all but the last token, a slice at a time.

    Attention on this path materialises a score matrix proportional to the
    number of query positions in the forward, so putting a whole record-length
    prompt through at once needs tens of gigabytes while putting it through in
    slices needs a fraction of that. It is a memory choice and nothing else:
    the keys and values a token produces depend on the tokens before it and
    not on how many came with it in the same call, so the cache this leaves is
    the cache one forward would have left.

    Positions are absolute across slices. A slice numbered from zero would
    rotate its keys as if the record started there.
    """
    import torch

    n = int(ids.shape[0])
    for a in range(0, n - 1, chunk):
        b = min(a + chunk, n - 1)
        with torch.no_grad():
            model(
                input_ids=ids[a:b],
                seq_ids=torch.zeros(b - a, dtype=torch.long, device=device),
                position_ids=torch.arange(a, b, dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                mode="generate",
            )
    return n - 1


def run_arm(
    arm,
    questions,
    patients_by_id,
    *,
    model,
    tok,
    ds_cfg,
    cartridge,
    max_new_tokens,
    device,
    budget,
    prefill_chunk=0,
):
    """Answer every question once, under one arm."""
    import torch

    from cartridges.generation import flex_generate

    rows, spent = [], 0
    stop_ids = sorted(
        {
            i
            for i in (
                tok.eos_token_id,
                tok.convert_tokens_to_ids("<|im_end|>"),
                tok.convert_tokens_to_ids("<|endoftext|>"),
            )
            if i is not None and i >= 0
        }
    )
    for n, q in enumerate(questions, 1):
        pid = "_".join(q.question_id.split("_")[:2])
        prompt = ds_cfg["wrap"](q, patients_by_id[pid])
        if arm == "fulldoc":
            prompt = f"{build_record_text(patients_by_id[pid])}\n\n{prompt}"
        ids = (
            tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=True,
            )
            .to(device)
            .flatten()
        )
        cache = cartridge if arm == "source" else None
        if cache is not None:
            cache.clear()
        t0 = time.time()
        n_prompt = int(ids.shape[0])
        if prefill_chunk and n_prompt > prefill_chunk:
            from cartridges.cache import AttnConfig, TrainableCache

            if cache is None:
                cache = TrainableCache(
                    config=AttnConfig(
                        n_layers=model.config.num_hidden_layers,
                        n_heads=model.config.num_key_value_heads,
                        head_dim=getattr(
                            model.config,
                            "head_dim",
                            model.config.hidden_size
                            // model.config.num_attention_heads,
                        ),
                    )
                ).to(device)
            last = prefill_in_chunks(model, ids, cache, prefill_chunk, device)
            gen_ids, gen_pos = ids[last:], torch.arange(last, n_prompt, device=device)
        else:
            gen_ids, gen_pos = ids, torch.arange(n_prompt, device=device)
        with torch.no_grad():
            out = flex_generate(
                model,
                tok,
                gen_ids,
                seq_ids=torch.zeros(gen_ids.shape[0], dtype=torch.long, device=device),
                position_ids=gen_pos,
                cache=cache,
                stop_token_ids=stop_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        gen = out.get(0, [])
        spent += len(gen)
        if spent > budget:
            raise SystemExit(
                f"INVALID: generated tokens exceeded the declared budget "
                f"({spent} > {budget}) partway through arm {arm}"
            )
        text = tok.decode(gen, skip_special_tokens=True)
        opts = [getattr(q, f"answer_{c}") for c in "abcde"]
        hits = [i for i, o in enumerate(opts) if normalise(o) == normalise(q.correct)]
        if len(hits) != 1:
            raise SystemExit(
                f"INVALID: {q.question_id} has {len(hits)} options matching its "
                "recorded answer, so it cannot be scored"
            )
        sc = strict_score(
            text, opts, hits[0], n_new_tokens=len(gen), max_new_tokens=max_new_tokens
        )
        faith, fell_back = faithful_score(text, opts, q.correct)
        rows.append(
            {
                "arm": arm,
                "question_id": q.question_id,
                "patient": pid,
                "prompt_tokens": int(ids.shape[0]),
                "generated_tokens": len(gen),
                "cap_hit": bool(sc["cap_hit"]),
                "seconds": round(time.time() - t0, 2),
                "correct": sc["correct"],
                "outcome": sc["outcome"],
                "picked": sc["picked"],
                "gold_index": hits[0],
                "answer_tag": sc["answer_tag"],
                "think_closed": sc["think_closed"],
                "pred": sc["pred"],
                "top_ratio": sc["top_ratio"],
                "match_margin": sc["margin"],
                "faithful_correct": faith,
                "faithful_fell_back_to_option_a": bool(fell_back),
                "gold": q.correct,
                "response": text,
            }
        )
        if n % 5 == 0 or n == len(questions):
            acc = sum(r["correct"] for r in rows) / len(rows)
            log(f"  {arm}: {n}/{len(questions)}, running {acc:.3f}, {spent} tokens")
    return rows, spent


# ---------------------------------------------------------------------------
# the screen's gate
# ---------------------------------------------------------------------------

MIN_NET_GAIN = 2  # documents of exact answer, over the no-cartridge floor
MAX_PARSER_EXCESS = 0


def gate(by_arm):
    """The plan's release criteria, computed on the strict score.

    A screen, not a significance test and not a deployment decision. Twenty
    questions put the binomial standard error near eleven points, so the
    two-question bar this reads is well inside the noise of a single draw;
    it is a rule for releasing the next stage's resources and it is reported
    as one.
    """
    floor = by_arm.get("nocart")
    out = {
        "rule": (
            f"each arm must answer at least {MIN_NET_GAIN} more questions "
            "correctly than the no-cartridge floor, and must not produce more "
            "unparseable or non-terminating answers than it does"
        ),
        "n_questions": len(floor) if floor else 0,
        "caveat": (
            "exploratory patient-clustered screen over previously exposed "
            "questions; at twenty questions this cannot resolve a difference "
            "of a few points and is not offered as one"
        ),
        "arms": {},
    }
    if not floor:
        out["verdict"] = "NO_FLOOR_ARM"
        out["advances"] = False
        return out

    def bad(rows):
        """Answers the instrument could not read, plus ones that never ended.

        Counted per row rather than summed over two independent tallies: a
        generation that hit the cap and also produced nothing readable is one
        failure, not two.
        """
        return sum(1 for r in rows if r["outcome"] in UNRESOLVED or r["cap_hit"])

    base_correct = sum(r["correct"] for r in floor)
    base_bad = bad(floor)
    out["floor"] = {
        "correct": base_correct,
        "accuracy": base_correct / len(floor),
        "unparseable_or_capped": base_bad,
    }
    for arm, rows in sorted(by_arm.items()):
        if arm == "nocart":
            continue
        c = sum(r["correct"] for r in rows)
        b = bad(rows)
        rec = {
            "correct": c,
            "accuracy": c / len(rows),
            "net_gain_over_floor": c - base_correct,
            "meets_net_gain": bool(c - base_correct >= MIN_NET_GAIN),
            "unparseable_or_capped": b,
            "parser_excess_over_floor": b - base_bad,
            "meets_parser_excess": bool(b - base_bad <= MAX_PARSER_EXCESS),
            "faithful_correct": sum(r["faithful_correct"] for r in rows),
            "faithful_fell_back": sum(
                r["faithful_fell_back_to_option_a"] for r in rows
            ),
        }
        rec["passes"] = bool(rec["meets_net_gain"] and rec["meets_parser_excess"])
        out["arms"][arm] = rec

    src = out["arms"].get("source")
    if src is None:
        out["verdict"] = "SOURCE_ARM_ABSENT"
        out["advances"] = False
    elif src["passes"]:
        out["verdict"] = "SOURCE_UTILITY_ESTABLISHED"
        out["advances"] = True
    else:
        out["verdict"] = "SOURCE_UTILITY_UNESTABLISHED"
        out["advances"] = False
    ceiling = out["arms"].get("fulldoc")
    if ceiling is not None and not ceiling["passes"]:
        out["benchmark_uninformative"] = True
        out["note"] = (
            "the ceiling arm, which is given the record outright, did not "
            "clear the floor either; on this evidence the questions do not "
            "separate having the document from not having it, and no verdict "
            "about the cartridge can be read from them"
        )
    return out


def finish(args, rows_by_arm, extra):
    """Write the rows, the per-arm summary and the verdict."""
    rows_path = os.path.join(args.out_dir, "rows.jsonl")
    with open(rows_path, "w") as fh:
        for arm in sorted(rows_by_arm):
            for r in rows_by_arm[arm]:
                fh.write(json.dumps(r) + "\n")
    report = {
        "contract": "cartridge_c1_source_utility_v1",
        "experiment_id": args.experiment_id,
        **extra,
        "per_arm": {a: summarise(r) for a, r in sorted(rows_by_arm.items())},
        "faithful_per_arm": {
            a: {
                "correct": sum(x["faithful_correct"] for x in r),
                "accuracy": round(sum(x["faithful_correct"] for x in r) / len(r), 4),
                "fell_back_to_option_a": sum(
                    x["faithful_fell_back_to_option_a"] for x in r
                ),
                "note": (
                    "the library's own scorer, reproduced for comparability with "
                    "the number recorded beside the checkpoint; it scores the "
                    "first option when it finds no tag, so its accuracy is not "
                    "a measure of answering"
                ),
            }
            for a, r in sorted(rows_by_arm.items())
        },
        "gate": gate(rows_by_arm),
    }
    json.dump(report, open(os.path.join(args.out_dir, "C1_RESULT.json"), "w"), indent=2)
    log("")
    log(f"{'arm':10s} {'correct':>8s} {'acc':>7s} {'unresolved':>11s} {'capped':>7s}")
    for a, s in sorted(report["per_arm"].items()):
        log(
            f"{a:10s} {s['n_correct']:>8d} {s['accuracy']:>7.3f} "
            f"{s['unresolved']:>11d} {s['cap_hit']:>7d}"
        )
    g = report["gate"]
    log("")
    log(f"verdict: {g['verdict']} (advances={g['advances']})")
    for a, rec in sorted(g.get("arms", {}).items()):
        log(
            f"  {a:10s} net {rec['net_gain_over_floor']:+d} "
            f"(needs +{MIN_NET_GAIN}), parser excess "
            f"{rec['parser_excess_over_floor']:+d} (needs <={MAX_PARSER_EXCESS}) "
            f"-> {'pass' if rec['passes'] else 'FAIL'}"
        )
    if g.get("benchmark_uninformative"):
        log(f"  NOTE: {g['note']}")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
