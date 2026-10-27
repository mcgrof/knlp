# SPDX-License-Identifier: MIT
"""LongHealth multiple-choice protocol for the frozen reader.

Prompt, decoding and scoring follow the Cartridges at Scale Table 15
setting for LongHealth exactly: the system prompt below, the question with
its five options in the user turn, the chat template with thinking
enabled, sampling at temperature 0.6 with the model's shipped top-k 20 and
top-p 0.95, a 2,048-token completion cap, and fuzzy option matching on the
text inside the last ``<answer>`` tag. The benchmark file is the public
LongHealth ``benchmark_v5.json``; its digest is recorded by every run.
"""

from __future__ import annotations

import dataclasses
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from . import model as M

SYSTEM_PROMPT = (
    "Please reference the patient medical records to answer the user's "
    "questions. Choose the single best option and provide your answer exactly "
    "as it appears in the options.\n\n"
    "Wrap your answer in: <answer> The correct option text here </answer>"
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
OPTION_KEYS = ("answer_a", "answer_b", "answer_c", "answer_d", "answer_e")
DECODER = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "max_completion": 2048}


@dataclasses.dataclass
class Question:
    patient_id: str
    question_id: str
    question: str
    options: tuple[str, str, str, str, str]
    correct: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Question":
        return Question(
            patient_id=row["patient_id"],
            question_id=row["question_id"],
            question=row["question"],
            options=tuple(row["options"]),
            correct=row["correct"],
        )


def load_benchmark(path: Path) -> tuple[dict[str, Any], str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patient_questions(
    data: dict[str, Any], patient_id: str, max_q: int = 20
) -> list[Question]:
    row = data[patient_id]
    out = []
    for q in row["questions"][:max_q]:
        out.append(
            Question(
                patient_id=patient_id,
                question_id=f"{patient_id}_{q['No']}",
                question=q["question"],
                options=tuple(q[k] for k in OPTION_KEYS),
                correct=q["correct"],
            )
        )
    return out


def record_text(data: dict[str, Any], patient_id: str) -> str:
    """The patient's full record as the native evaluator joins it."""
    return "\n\n".join(data[patient_id]["texts"].values())


def user_prompt(q: Question, doc_text: str | None = None) -> str:
    a, b, c, d, e = q.options
    body = f"{q.question}\n\nA. {a}  B. {b}  C. {c}  D. {d}  E. {e}"
    if doc_text is not None:
        return f"{doc_text}\n\n{body}"
    return body


def prompt_ids(
    tokenizer: Any, q: Question, doc_text: str | None = None, thinking: bool = True
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(q, doc_text)},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def correct_index(q: Question) -> int:
    hits = [i for i, o in enumerate(q.options) if _norm(o) == _norm(q.correct)]
    if len(hits) != 1:
        raise ValueError(f"{q.question_id}: correct matches {len(hits)} options")
    return hits[0]


def score_response(text: str, q: Question) -> tuple[int, bool]:
    """``(correct, tag_found)``: strip the thinking block, take the last
    answer tag (fallback: the last 300 characters), pick the closest option
    by difflib ratio; an empty prediction scores wrong."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    matches = ANSWER_RE.findall(text)
    tag_found = bool(matches)
    pred = matches[-1].strip() if matches else text.strip()[-300:]
    if not pred:
        return 0, tag_found

    def sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(
            None, a.lower().strip(), b.lower().strip()
        ).ratio()

    ratios = [sim(pred, opt) for opt in q.options]
    if max(ratios) <= 0.0:
        return 0, tag_found
    return int(ratios.index(max(ratios)) == correct_index(q)), tag_found


def make_sampler(
    generator: torch.Generator,
    temperature: float = DECODER["temperature"],
    top_k: int = DECODER["top_k"],
    top_p: float = DECODER["top_p"],
):
    """Top-k then nucleus sampling on FP32 logits ``[B, V]``, the model's
    shipped decoder settings; at least one token survives both filters."""

    def sample(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_k > 0:
            vals, idx = torch.topk(probs, min(top_k, probs.shape[-1]), dim=-1)
        else:
            vals, idx = torch.sort(probs, descending=True, dim=-1)
        if top_p < 1.0:
            keep = (torch.cumsum(vals, dim=-1) - vals) < top_p
            keep[:, 0] = True
            vals = vals * keep
        draw = torch.multinomial(
            vals / vals.sum(-1, keepdim=True),
            1,
            generator=generator,
        )
        return idx.gather(1, draw).squeeze(1)

    return sample


def questions_jsonl_rows(questions: list[Question]) -> list[dict[str, Any]]:
    """Rows a reader-only consumer may receive: the question and options,
    never the record text."""
    return [
        {
            "world_id": q.patient_id,
            "qid": q.question_id,
            "text": q.question,
            "options": list(q.options),
            "patient_id": q.patient_id,
            "question_id": q.question_id,
            "question": q.question,
            "style": "longhealth",
        }
        for q in questions
    ]


def question_from_row(row: dict[str, Any]) -> Question:
    return Question(
        patient_id=row["patient_id"],
        question_id=row["question_id"],
        question=row["question"],
        options=tuple(row["options"]),
        correct=row.get("correct", ""),
    )


@torch.no_grad()
def answer_panel(
    reader: M.Reader,
    tokenizer: Any,
    prefix: list[tuple[torch.Tensor, torch.Tensor]] | None,
    first_position: int,
    questions: list[Question],
    runs: int,
    seed: int,
    batch: int,
    max_completion: int = DECODER["max_completion"],
    doc_text: str | None = None,
    prefix_mask: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Sampled answers for every question, ``runs`` times, from one prefix
    (a cartridge) or from text in the prompt (``doc_text``). One generator
    per batch; its seed is recorded on every row."""
    rows: list[dict[str, Any]] = []
    device = reader.device
    for run in range(runs):
        for start in range(0, len(questions), batch):
            group = questions[start : start + batch]
            prompts = [prompt_ids(tokenizer, q, doc_text) for q in group]
            ids, mask = M.left_pad(prompts, reader.pad_id, device)
            batch_seed = seed + run * 100003 + start
            generator = torch.Generator(device=device).manual_seed(batch_seed)
            sampler = make_sampler(generator)
            group_prefix = None
            group_mask = None
            if prefix is not None:
                group_prefix = [
                    (k.expand(len(group), -1, -1, -1), v.expand(len(group), -1, -1, -1))
                    for k, v in prefix
                ]
                if prefix_mask is not None:
                    group_mask = prefix_mask.expand(len(group), -1)
            out = reader.decode(
                group_prefix,
                ids,
                mask,
                first_position,
                max_completion,
                sampler=sampler,
                prefix_mask=group_mask,
            )
            for row, q in enumerate(group):
                gen = out["ids"][row].tolist()
                kept = M.strip_generation(gen, reader.eos_ids, reader.pad_id)
                text = tokenizer.decode(kept, skip_special_tokens=True)
                stopped = len(kept) < len(gen) or any(t in reader.eos_ids for t in gen)
                correct, tag = score_response(text, q)
                rows.append(
                    {
                        "run": run,
                        "patient_id": q.patient_id,
                        "question_id": q.question_id,
                        "correct": int(correct),
                        "tag_found": bool(tag),
                        "generated_tokens": len(kept),
                        "stop_reason": "eos" if stopped else "cap",
                        "truncated_thinking": "</think>" not in text,
                        "prompt_tokens": len(prompts[row]),
                        "batch_seed": batch_seed,
                        "text": text,
                        "pred_ids": kept,
                    }
                )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import statistics

    runs = sorted({r["run"] for r in rows})
    per_run = []
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        per_run.append(
            {
                "correct": sum(r["correct"] for r in rr),
                "total": len(rr),
                "answer_tag_missing": sum(1 for r in rr if not r["tag_found"]),
                "cap_hits": sum(1 for r in rr if r["stop_reason"] == "cap"),
            }
        )
    accs = [p["correct"] / max(p["total"], 1) for p in per_run]
    return {
        "runs": len(runs),
        "acc_mean": statistics.mean(accs) if accs else float("nan"),
        "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "per_run_acc": accs,
        "per_run_counts": per_run,
        "questions": len({r["question_id"] for r in rows}),
        "decoder": dict(DECODER),
    }
