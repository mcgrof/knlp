"""Prompt rendering for every model role.

The chat format reproduces the upstream GSM8K recipe for
Qwen2.5-0.5B-Instruct (the ``qwen_instruct_user_boxed_math`` template of
the Never Give Up reference launcher): no system turn, and the solve
instruction appended to the user message.  All roles are single user
turns; earlier attempts are quoted inside the user message, never
replayed as assistant turns, so every role sees a bounded, logged input.

Model inputs contain only the question and the model's own generated
history.  Gold answers, reference solutions and grader internals are
never passed to any function in this module; ``check_no_gold`` is the
harness-level assertion that enforces it.
"""

from __future__ import annotations

SOLVE_SUFFIX = (
    "\n\nPlease reason step by step, and put your final answer within \\boxed{}"
)

TRACE_HEAD = 768
TRACE_TAIL = 256
NOTE_TOKENS = 256
INPUT_CEILING = 8192
TRUNC_MARK = "\n[... middle of attempt omitted ...]\n"


def chat(user: str) -> str:
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def solve_prompt(question: str) -> str:
    return chat(question + SOLVE_SUFFIX)


def revised_prompt(question: str, note: str) -> str:
    """Solve prompt with a strategy note (shared by the summary and plan arms)."""
    return chat(
        question
        + "\n\nA note on how to approach this problem:\n"
        + note.strip()
        + "\n\nUse the note if it helps, but work the problem out yourself."
        + SOLVE_SUFFIX
    )


def summary_prompt(question: str, traces: list[str], prev_note: str | None) -> str:
    parts = [
        f"Problem:\n{question}\n",
        "Below are earlier attempts at this problem. None of them passed the "
        "final-answer check.\n",
    ]
    for k, t in enumerate(traces, 1):
        parts.append(f"[Attempt {k}]\n{t.strip()}\n")
    if prev_note:
        parts.append(f"[Earlier note]\n{prev_note.strip()}\n")
    parts.append(
        "These attempts did not pass the final-answer check. Describe their "
        "shared approach, identify one assumption worth changing, and propose "
        "an alternative. A failed answer does not prove every step or the "
        "whole approach wrong. Do not infer the reference answer. Do not solve "
        "the problem. Answer in under 150 words."
    )
    return chat("\n".join(parts))


def plan_prompt(question: str, n_failed: int, prev_note: str | None) -> str:
    parts = [
        f"Problem:\n{question}\n",
        f"{n_failed} earlier attempts at this problem did not pass the "
        "final-answer check. Their contents are not shown.\n",
    ]
    if prev_note:
        parts.append(f"[Earlier note]\n{prev_note.strip()}\n")
    parts.append(
        "Before anyone solves it again, write a strategy: state what the "
        "problem asks, list the quantities involved and how they relate, and "
        "propose a careful plan to reach the answer. If an earlier note is "
        "shown, propose a different approach from it. Do not solve the "
        "problem. Answer in under 150 words."
    )
    return chat("\n".join(parts))


def critique_prompt(question: str, trace: str) -> str:
    return chat(
        f"Problem:\n{question}\n\n[Attempt]\n{trace.strip()}\n\n"
        "This attempt did not pass the final-answer check. A failed answer "
        "does not prove every step wrong. Identify the step most likely to be "
        "wrong and say what should change. Do not write a full solution. "
        "Answer in under 150 words."
    )


def correction_prompt(question: str, trace: str, critique: str) -> str:
    return chat(
        f"{question}\n\n[Earlier attempt]\n{trace.strip()}\n\n"
        f"[Critique of that attempt]\n{critique.strip()}\n\n"
        "Write a corrected, complete solution." + SOLVE_SUFFIX
    )


def bound_trace(tok, text: str) -> tuple[str, bool]:
    """Keep the first TRACE_HEAD and last TRACE_TAIL tokens of a trace."""
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= TRACE_HEAD + TRACE_TAIL:
        return text, False
    head = tok.decode(ids[:TRACE_HEAD])
    tail = tok.decode(ids[-TRACE_TAIL:])
    return head + TRUNC_MARK + tail, True


def check_no_gold(prompt: str, item_id: str, refs: dict | None) -> None:
    """Refuse any prompt that carries a reference solution field.

    The gold *number* can legitimately appear in a question or in a
    model's own wrong-but-close work, so the check targets the text only
    the dataset holds: every line of the reference solution longer than
    40 characters.  ``refs`` maps item id to that reference text and is
    loaded only for this check.
    """
    ref = refs.get(item_id) if refs else None
    if not ref:
        return
    for line in ref.splitlines():
        line = line.strip()
        if len(line) > 40 and line in prompt:
            raise AssertionError(f"reference solution leaked for {item_id}")
