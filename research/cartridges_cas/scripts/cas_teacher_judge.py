#!/usr/bin/env python3
"""Judge self-study answers for support by their source note.

Reads items written by cas_teacher_items.py (or candidate items from
cas_teacher_resample.py, same fields) and asks a judge model, through
an OpenAI-compatible chat endpoint, whether each final answer is a
faithful answer to its question from the note the teacher answered
from. The note is the system message, so a server with prefix caching
reuses it across the items that share it. The judge returns a verdict
in the categories used for the manual audit (SUPPORTED, MINOR_ISSUES,
CONTRADICTED, UNSUPPORTED, NO_ANSWER, DEGENERATE), whether the question
rests on a false premise, and a short reason. exclude is true for the
last four verdicts and when an answer accepts a false premise.

Items marked for regeneration are skipped. Results are appended to a
JSONL file keyed by item, so a rerun resumes; a reply that does not
parse is recorded with verdict PARSE_ERROR and retried on the next run.

calibrate mode compares judgments with labeled rows (a JSON list with
key and exclude) and reports agreement and Cohen's kappa.

JUDGE_MODE=claims asks instead for a list of the answer's factual
claims, each SUPPORTED, CONTRADICTED or NOT_IN_NOTE with the exact note
text that decides it, and derives the verdict mechanically: a quote
that is not in the note makes its claim UNVERIFIED, and any central
claim that is not SUPPORTED excludes the answer.

merge mode combines the judgments of several judges so that an item
passes only when every judge passed it. JUDGMENTS lists the judges'
files in order; the first judge that failed the item, or that has no
parsed reply for it, decides the merged record. A passing item is
SUPPORTED only when every judge said SUPPORTED, else MINOR_ISSUES.
Each merged record keeps every judge's verdict.

Env: ITEMS (comma list of items JSONL), NOTES (comma list of notes
JSON), OUT (judgments JSONL), URL (http://localhost:8000/v1), MODEL,
CONCURRENCY (64), MAX_TOKENS (4096), REASONING_EFFORT (medium, sent
when set), LIMIT (optional). calibrate: OUT, LABELS. merge: ITEMS,
JUDGMENTS (comma list of judgments JSONL), OUT.
"""

import asyncio
import json
import os
import re
import sys

SYSTEM = """You check answers in a clinical question-answering dataset against their only source, the clinical note below. Use no outside knowledge about the patient.

Judge the FINAL ANSWER to the QUESTION:
- SUPPORTED: every factual claim that bears on the question is stated in or directly derivable from the note; requested formats are honored reasonably.
- MINOR_ISSUES: substantially correct, with a small unsupported embellishment, an imprecise detail, or a small omission that does not change the main answer. Clearly general medical knowledge that does not contradict the note is at most MINOR_ISSUES.
- CONTRADICTED: a central claim conflicts with the note (wrong value, date, drug, dose, frequency, laterality, diagnosis, sequence of events, study or specimen a finding belongs to).
- UNSUPPORTED: a central claim is absent from the note and presented as if from it (invented dates, events, results, drugs, codes, or wrong general pharmacology presented as fact).
- NO_ANSWER: empty, cut off, or does not attempt the question.
- DEGENERATE: repetition loops or garbled text.
Check long structured answers (tables, lists, timelines, JSON) item by item: one clearly wrong central item makes it CONTRADICTED; one minor wrong peripheral item makes it MINOR_ISSUES. Read dates and dosing schemes carefully: notes may write day/month or month/year ("08/19" can mean August 2019), and "1-1-1-1" or "1-0-1-0" give doses per time of day.
Also decide whether the QUESTION rests on a false premise (asserts something the note contradicts or does not contain) and whether the answer accepted it.

Reply with only one JSON object:
{"verdict": "...", "false_premise": true|false, "accepted_false_premise": true|false, "issues": "one or two sentences, empty if SUPPORTED"}

CLINICAL NOTE:
"""

EXCLUDE = {"CONTRADICTED", "UNSUPPORTED", "NO_ANSWER", "DEGENERATE"}

CLAIMS_SYSTEM = """You verify answers in a clinical question-answering dataset against their only source, the clinical note below. Use no outside knowledge about the patient.

List the factual claims in the FINAL ANSWER that bear on the QUESTION (at most 30; merge trivial ones). Include every date, value, dose, frequency, drug, finding, diagnosis, procedure and event the answer states. For each claim give:
- "claim": the claim in a few words;
- "status": "SUPPORTED" (stated in or directly derivable from the note), "CONTRADICTED" (the note says otherwise), or "NOT_IN_NOTE" (the note does not contain it);
- "quote": for SUPPORTED or CONTRADICTED, the exact text from the note that decides it, copied character for character (10 to 200 characters); empty for NOT_IN_NOTE;
- "central": true if the claim is part of what the question asks for, false if it is peripheral.
Read dates and dosing schemes carefully: notes may write day/month or month/year ("08/19" can mean August 2019), and "1-1-1-1" or "1-0-1-0" give doses per time of day. Also report whether the QUESTION rests on a false premise and whether the answer accepted it, and whether the answer is empty, cut off, or a repetition loop.

Reply with only one JSON object:
{"claims": [{"claim": "...", "status": "...", "quote": "...", "central": true}], "false_premise": false, "accepted_false_premise": false, "no_answer_or_degenerate": false}

CLINICAL NOTE:
"""


DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
QUOTES = {
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u201c"): '"',
    ord("\u201d"): '"',
}


def norm_ws(s):
    """Compare quotes loosely on typography only: unicode dashes and
    quotes to ASCII, Markdown escapes and table pipes removed,
    whitespace collapsed, lowercase."""
    s = (s or "").translate(DASHES).translate(QUOTES).replace("\u00a0", " ")
    s = s.replace("\\", "").replace("|", " ").replace("**", "")
    return " ".join(s.split()).lower()


def quote_in_note(quote, note_norm, note_words):
    """A quote is verified when every fragment (split at ellipses) of at
    least two words has its words, in order, inside a span of the note no
    longer than twice the fragment: punctuation, symbols and table layout
    may differ, the words may not."""
    frags = re.split(r"\.\.\.|\u2026", quote or "")
    ok = False
    for f in frags:
        words = re.findall(r"[a-z0-9]+", norm_ws(f).replace("\u00ae", ""))
        if len(words) < 2:
            continue
        if not in_order(words, note_words):
            return False
        ok = True
    return ok


def in_order(words, note_words):
    n, m = len(note_words), len(words)
    limit = 2 * m + 4
    for start in (i for i in range(n) if note_words[i] == words[0]):
        j, k = start, 0
        while j < n and j - start < limit and k < m:
            if note_words[j] == words[k]:
                k += 1
            j += 1
        if k == m:
            return True
    return False


def claims_verdict(d, note):
    """Mechanical verdict from a claim list: a SUPPORTED or CONTRADICTED
    claim whose quote is not in the note counts as UNVERIFIED. Exclude
    when any central claim is CONTRADICTED, NOT_IN_NOTE or UNVERIFIED,
    when the answer accepted a false premise, or when it has no answer."""
    nn = norm_ws(note)
    nw = re.findall(r"[a-z0-9]+", nn)
    bad, minor = [], 0
    for c in d.get("claims") or []:
        st = c.get("status")
        if st in ("SUPPORTED", "CONTRADICTED") and not quote_in_note(
            c.get("quote"), nn, nw
        ):
            st = "UNVERIFIED"
        if st != "SUPPORTED":
            if c.get("central", True):
                bad.append(f"{st}: {c.get('claim', '')}")
            else:
                minor += 1
    if d.get("no_answer_or_degenerate"):
        verdict = "NO_ANSWER"
    elif bad:
        verdict = (
            "CONTRADICTED"
            if any(b.startswith("CONTRA") for b in bad)
            else "UNSUPPORTED"
        )
    elif minor:
        verdict = "MINOR_ISSUES"
    else:
        verdict = "SUPPORTED"
    return verdict, bad


def parse_claims(text):
    m = re.findall(r"\{.*\}", text or "", re.DOTALL)
    for cand in reversed(m):
        try:
            d = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(d.get("claims"), list):
            return d
    return None


def parse(text):
    m = re.findall(r"\{.*\}", text or "", re.DOTALL)
    for cand in reversed(m):
        try:
            d = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if d.get("verdict") in EXCLUDE | {"SUPPORTED", "MINOR_ISSUES"}:
            return d
    return None


async def judge_all(items, notes, out_path):
    import httpx

    url = os.environ.get("URL", "http://localhost:8000/v1")
    model = os.environ["MODEL"]
    conc = int(os.environ.get("CONCURRENCY", "64"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "4096"))
    effort = os.environ.get("REASONING_EFFORT", "medium")
    claims = os.environ.get("JUDGE_MODE", "verdict") == "claims"
    sem = asyncio.Semaphore(conc)
    done = [0]
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    async with httpx.AsyncClient(base_url=url, timeout=1800, limits=limits) as client:

        async def one(it, f):
            body = dict(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (CLAIMS_SYSTEM if claims else SYSTEM)
                        + notes[it["note"]],
                    },
                    {
                        "role": "user",
                        "content": f"QUESTION:\n{it['question']}\n\n"
                        f"FINAL ANSWER:\n{it['answer']}",
                    },
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if effort:
                body["reasoning_effort"] = effort
            async with sem:
                try:
                    r = await client.post("/chat/completions", json=body)
                    r.raise_for_status()
                    text = r.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    text = f"ERROR {type(e).__name__}: {e}"
            d = parse_claims(text) if claims else parse(text)
            rec = dict(
                key=it["key"], raw=(text if claims else text[-2000:]) if text else ""
            )
            if d is not None and claims:
                verdict, bad = claims_verdict(d, notes[it["note"]])
                d = dict(
                    verdict=verdict,
                    false_premise=d.get("false_premise"),
                    accepted_false_premise=d.get("accepted_false_premise"),
                    issues="; ".join(bad)[:1000],
                    n_claims=len(d.get("claims") or []),
                )
            if d is None:
                rec.update(verdict="PARSE_ERROR", exclude=None)
            else:
                acc = bool(d.get("accepted_false_premise"))
                rec.update(
                    verdict=d["verdict"],
                    false_premise=bool(d.get("false_premise")),
                    accepted_false_premise=acc,
                    issues=d.get("issues", ""),
                    exclude=d["verdict"] in EXCLUDE or acc,
                )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[0] += 1
            if done[0] % 200 == 0:
                print(f"[judge] {done[0]}/{len(items)}", flush=True)

        with open(out_path, "a") as f:
            await asyncio.gather(*(one(it, f) for it in items))


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r["verdict"] != "PARSE_ERROR":
                done[r["key"]] = r
    return done


def judge():
    notes = {}
    for p in os.environ["NOTES"].split(","):
        notes.update(json.load(open(p)))
    items = [
        json.loads(line) for p in os.environ["ITEMS"].split(",") for line in open(p)
    ]
    items = [it for it in items if it["status"] == "judge"]
    out = os.environ["OUT"]
    done = load_done(out)
    todo = [it for it in items if it["key"] not in done]
    if os.environ.get("LIMIT"):
        todo = todo[: int(os.environ["LIMIT"])]
    print(
        f"[judge] {len(items)} items, {len(done)} done, {len(todo)} to do", flush=True
    )
    asyncio.run(judge_all(todo, notes, out))
    done = load_done(out)
    bad = sum(1 for it in items if it["key"] not in done)
    ex = sum(1 for r in done.values() if r["exclude"])
    print(f"JUDGE_DONE judged={len(done)} unparsed_or_missing={bad} exclude={ex}")


def calibrate():
    done = load_done(os.environ["OUT"])
    labels = json.load(open(os.environ["LABELS"]))
    pairs = [
        (l["exclude"], done[l["key"]]["exclude"]) for l in labels if l["key"] in done
    ]
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    tp = sum(a and b for a, b in pairs)
    fn = sum(a and not b for a, b in pairs)
    fp = sum(b and not a for a, b in pairs)
    print(
        "CALIBRATE "
        + json.dumps(
            dict(
                n=n,
                missing=len(labels) - n,
                agreement=round(agree, 4),
                kappa=round(kappa, 4),
                label_exclude_rate=round(pa, 4),
                judge_exclude_rate=round(pb, 4),
                judge_recall=round(tp / (tp + fn), 4) if tp + fn else None,
                judge_precision=round(tp / (tp + fp), 4) if tp + fp else None,
            )
        )
    )


def merge():
    items = [
        json.loads(line) for p in os.environ["ITEMS"].split(",") for line in open(p)
    ]
    items = [it for it in items if it["status"] == "judge"]
    judges = [load_done(p) for p in os.environ["JUDGMENTS"].split(",")]
    counts = {}
    with open(os.environ["OUT"], "w") as f:
        for it in items:
            recs = [j.get(it["key"]) for j in judges]
            rec = dict(
                key=it["key"], judges=[r["verdict"] if r else None for r in recs]
            )
            for n, r in enumerate(recs):
                if r is None:
                    rec.update(verdict="UNJUDGED", exclude=True, decided_by=n)
                    break
                if r["exclude"]:
                    rec.update(
                        verdict=r["verdict"],
                        exclude=True,
                        decided_by=n,
                        accepted_false_premise=r.get("accepted_false_premise"),
                        issues=r.get("issues", ""),
                    )
                    break
            else:
                both = all(r["verdict"] == "SUPPORTED" for r in recs)
                rec.update(
                    verdict="SUPPORTED" if both else "MINOR_ISSUES", exclude=False
                )
            counts[rec["verdict"]] = counts.get(rec["verdict"], 0) + 1
            f.write(json.dumps(rec) + "\n")
    print(f"MERGE_DONE items={len(items)} {json.dumps(counts, sort_keys=True)}")


if __name__ == "__main__":
    {"judge": judge, "calibrate": calibrate, "merge": merge}[sys.argv[1]]()
