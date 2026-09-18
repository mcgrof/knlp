"""The five solve policies and their resource accounting.

Every policy starts from the same screened item and its screening
history, and stops at the first verified success.  Two budget modes:

* ``time``: a wall-clock deadline ``C`` seconds after the run starts.
  All model calls (solve, summary, planning, critique, correction),
  prompt processing and grading consume the clock.  A success counts
  only if its generation *and* grading finished by the deadline.
* ``tokens``: a cap on generated tokens across every role.  Input
  processing and wall time are still charged in the ledger.

Policies, after the common 64-attempt screen:

A  more independent attempts with the original prompt and settings.
B  a same-model summary of this item's latest failures, then revised
   attempts; every four failed revised attempts are summarized again.
C  independent attempts at the development-selected temperature.
D  a same-model strategy written from the question and the failure
   count only (no failure contents), refreshed every four failures.
R  independent starts, each failed start critiqued and corrected once.

Only solve and correction completions are graded; notes and critiques
never count as candidate answers.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from fractions import Fraction

from . import prompts as P
from .grader import grade

ARMS = ("A", "B", "C", "D", "R")
SOLVE_TOKENS = 4096
AUX_TEMPERATURE = 0.7
REVISE_EVERY = 4
HISTORY_FOR_SUMMARY = 4
# Safety stop against a degenerate model that emits zero tokens forever
# under a token budget; far above any budget in use.
MAX_CALLS = 2000


def derive_seed(*parts) -> int:
    h = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


@dataclass
class Budget:
    mode: str
    seconds: float | None = None
    tokens: int | None = None

    def tag(self) -> str:
        return f"time{self.seconds:.3f}" if self.mode == "time" else f"tok{self.tokens}"


class Ledger:
    """Append-only JSONL files: per-call rows, prompts by hash, run records."""

    def __init__(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        self.calls = open(os.path.join(outdir, "calls.jsonl"), "a")
        self.runs_path = os.path.join(outdir, "runs.jsonl")
        self.runs = open(self.runs_path, "a")
        self.prompts = open(os.path.join(outdir, "prompts.jsonl"), "a")
        self.seen_prompts: set[str] = set()

    def done_runs(self) -> set[str]:
        done = set()
        with open(self.runs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["run_key"])
        return done

    def prompt(self, h: str, text: str):
        if h not in self.seen_prompts:
            self.seen_prompts.add(h)
            self.prompts.write(json.dumps({"sha": h, "prompt": text}) + "\n")

    def call(self, row: dict):
        self.calls.write(json.dumps(row) + "\n")
        self.calls.flush()

    def run(self, rec: dict):
        for fh in (self.prompts, self.calls):
            fh.flush()
            os.fsync(fh.fileno())
        self.runs.write(json.dumps(rec) + "\n")
        self.runs.flush()
        os.fsync(self.runs.fileno())

    def close(self):
        for fh in (self.calls, self.runs, self.prompts):
            fh.close()


@dataclass
class RunState:
    t0: float
    deadline: float | None
    tokens_left: int | None
    call_idx: int = 0
    gen_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    n_graded: int = 0
    n_aux: int = 0
    n_truncated_out: int = 0
    n_trace_bounded: int = 0
    solved: bool = False
    t_success: float | None = None
    tokens_at_success: int | None = None
    overrun: float = 0.0
    exhausted: bool = False
    failures: list = field(default_factory=list)


class Runner:
    def __init__(self, backend, ledger: Ledger, refs: dict | None, run_tag: str):
        self.b = backend
        self.ledger = ledger
        self.refs = refs
        self.run_tag = run_tag

    async def _call(
        self,
        st: RunState,
        ctx: dict,
        role: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        gold: Fraction | None,
        extra: dict | None = None,
    ) -> dict | None:
        """One model call.  Returns the result, or None if budget is spent."""
        if st.exhausted:
            return None
        if st.call_idx >= MAX_CALLS:
            st.exhausted = True
            return None
        if st.deadline is not None and time.monotonic() >= st.deadline:
            st.exhausted = True
            return None
        if st.tokens_left is not None:
            if st.tokens_left <= 0:
                st.exhausted = True
                return None
            max_tokens = min(max_tokens, st.tokens_left)
        P.check_no_gold(prompt, ctx["item"], self.refs)
        seed = derive_seed(ctx["seed_base"], st.call_idx)
        ph = sha(prompt)
        self.ledger.prompt(ph, prompt)
        res = await self.b.generate(
            prompt, max_tokens, temperature, 1.0, seed, deadline=st.deadline
        )
        g = None
        if gold is not None and res["finish"] != "deadline":
            g = grade(res["text"], gold)
        t_done = time.monotonic()
        st.gen_tokens += res["gen_tokens"]
        st.prompt_tokens += res["prompt_tokens"]
        st.cached_tokens += res["cached_tokens"]
        if st.tokens_left is not None:
            st.tokens_left -= res["gen_tokens"]
        if res["finish"] == "length":
            st.n_truncated_out += 1
        if gold is None:
            st.n_aux += 1
        else:
            st.n_graded += 1
        counted = False
        if g is not None and g["correct"]:
            in_time = st.deadline is None or t_done <= st.deadline
            if in_time:
                counted = True
                st.solved = True
                st.t_success = t_done - st.t0
                st.tokens_at_success = st.gen_tokens
            else:
                st.overrun = max(st.overrun, t_done - st.deadline)
        if res["finish"] == "deadline":
            st.exhausted = True
            st.overrun = max(st.overrun, res["t_end"] - st.deadline)
        row = {
            "run_key": ctx["run_key"],
            "item": ctx["item"],
            "arm": ctx["arm"],
            "rep": ctx["rep"],
            "mode": ctx["mode"],
            "role": role,
            "call_idx": st.call_idx,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_sha": ph,
            "prompt_tokens": res["prompt_tokens"],
            "cached_tokens": res["cached_tokens"],
            "gen_tokens": res["gen_tokens"],
            "finish": res["finish"],
            "t_start": res["t_start"] - st.t0,
            "t_end": res["t_end"] - st.t0,
            "t_graded": t_done - st.t0,
            "graded": g is not None,
            "correct": bool(g and g["correct"]),
            "counted_success": counted,
            "answer": g["answer"] if g else None,
            "rule": g["rule"] if g else None,
            "status": g["status"] if g else None,
            "text": res["text"],
        }
        if extra:
            row.update(extra)
        self.ledger.call(row)
        st.call_idx += 1
        if gold is not None and not counted and res["finish"] != "deadline":
            st.failures.append(res["text"])
        return res

    def _bounded(self, st: RunState, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            b, cut = P.bound_trace(self.b.tok, t)
            st.n_trace_bounded += int(cut)
            out.append(b)
        return out

    def _fit(self, build, traces: list[str]) -> tuple[str, int]:
        """Drop oldest traces until the prompt fits the input ceiling."""
        k = 0
        while True:
            prompt = build(traces[k:])
            if self.b.n_tokens(prompt) + P.NOTE_TOKENS <= P.INPUT_CEILING or k >= len(
                traces
            ):
                return prompt, k
            k += 1

    async def run(
        self,
        item: dict,
        screen_fail: list[str],
        arm: str,
        rep: int,
        budget: Budget,
        temp_c: float,
    ) -> dict:
        q = item["question"]
        gold = Fraction(item["gold"])
        run_key = f"{self.run_tag}|{item['id']}|{arm}|{rep}|{budget.tag()}"
        ctx = {
            "run_key": run_key,
            "item": item["id"],
            "arm": arm,
            "rep": rep,
            "mode": budget.mode,
            "seed_base": derive_seed(self.run_tag, item["id"], arm, rep, budget.mode),
        }
        t0 = time.monotonic()
        st = RunState(
            t0=t0,
            deadline=t0 + budget.seconds if budget.mode == "time" else None,
            tokens_left=budget.tokens if budget.mode == "tokens" else None,
        )
        n_screen_fail = len(screen_fail)
        dropped_history = 0

        if arm in ("A", "C"):
            temp = 1.0 if arm == "A" else temp_c
            prompt = P.solve_prompt(q)
            while not st.solved and not st.exhausted:
                await self._call(st, ctx, "solve", prompt, SOLVE_TOKENS, temp, gold)

        elif arm == "B":
            recent = screen_fail[-HISTORY_FOR_SUMMARY:]
            note = None
            while not st.solved and not st.exhausted:
                traces = self._bounded(st, recent)
                sp, k = self._fit(lambda ts: P.summary_prompt(q, ts, note), traces)
                dropped_history += k
                r = await self._call(
                    st,
                    ctx,
                    "summary",
                    sp,
                    P.NOTE_TOKENS,
                    AUX_TEMPERATURE,
                    None,
                    {"history_dropped": k, "n_history": len(traces) - k},
                )
                if r is None or r["finish"] == "deadline":
                    break
                note = r["text"]
                before = len(st.failures)
                for _ in range(REVISE_EVERY):
                    if st.solved or st.exhausted:
                        break
                    await self._call(
                        st,
                        ctx,
                        "revised",
                        P.revised_prompt(q, note),
                        SOLVE_TOKENS,
                        1.0,
                        gold,
                    )
                recent = st.failures[before:][-HISTORY_FOR_SUMMARY:] or recent

        elif arm == "D":
            note = None
            while not st.solved and not st.exhausted:
                n_failed = n_screen_fail + len(st.failures)
                r = await self._call(
                    st,
                    ctx,
                    "plan",
                    P.plan_prompt(q, n_failed, note),
                    P.NOTE_TOKENS,
                    AUX_TEMPERATURE,
                    None,
                    {"n_failed_shown": n_failed},
                )
                if r is None or r["finish"] == "deadline":
                    break
                note = r["text"]
                for _ in range(REVISE_EVERY):
                    if st.solved or st.exhausted:
                        break
                    await self._call(
                        st,
                        ctx,
                        "revised",
                        P.revised_prompt(q, note),
                        SOLVE_TOKENS,
                        1.0,
                        gold,
                    )

        elif arm == "R":
            while not st.solved and not st.exhausted:
                r = await self._call(
                    st, ctx, "solve", P.solve_prompt(q), SOLVE_TOKENS, 1.0, gold
                )
                if r is None or st.solved or r["finish"] == "deadline":
                    continue
                (trace,) = self._bounded(st, [r["text"]])
                c = await self._call(
                    st,
                    ctx,
                    "critique",
                    P.critique_prompt(q, trace),
                    P.NOTE_TOKENS,
                    AUX_TEMPERATURE,
                    None,
                )
                if c is None or c["finish"] == "deadline":
                    continue
                await self._call(
                    st,
                    ctx,
                    "correction",
                    P.correction_prompt(q, trace, c["text"]),
                    SOLVE_TOKENS,
                    1.0,
                    gold,
                )
        else:
            raise ValueError(arm)

        rec = {
            "run_key": run_key,
            "item": item["id"],
            "arm": arm,
            "rep": rep,
            "mode": budget.mode,
            "budget_seconds": budget.seconds,
            "budget_tokens": budget.tokens,
            "solved": st.solved,
            "t_success": st.t_success,
            "tokens_at_success": st.tokens_at_success,
            "elapsed": time.monotonic() - t0,
            "gen_tokens": st.gen_tokens,
            "prompt_tokens": st.prompt_tokens,
            "cached_tokens": st.cached_tokens,
            "n_calls": st.call_idx,
            "n_graded": st.n_graded,
            "n_aux": st.n_aux,
            "n_truncated_out": st.n_truncated_out,
            "n_trace_bounded": st.n_trace_bounded,
            "history_dropped": dropped_history,
            "overrun": st.overrun,
            "temp_c": temp_c if arm == "C" else None,
        }
        self.ledger.run(rec)
        return rec
