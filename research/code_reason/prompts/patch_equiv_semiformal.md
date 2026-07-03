# Task: Patch equivalence verification (semi-formal certificate)

You are given two patches to the same repository. Decide whether they are
**behaviorally equivalent**, and record your reasoning as a certificate.

Repository root: {repo_root}
Patch A:
{patch_a}
Patch B:
{patch_b}

Build the argument evidence-first: define equivalence, state premises about
each patch's behavior with file/line evidence, search for a distinguishing
input (counterexample), then conclude. Output ONLY this JSON certificate
(schema: certificate.schema.json):

```json
{
  "task_type": "patch_equiv",
  "answer": {"equivalent": true_or_false},
  "definitions": ["equivalence = identical observable behavior on all inputs"],
  "premises": [
    {"id": "P1", "claim": "...",
     "evidence": [{"file": "...", "line_start": 0, "line_end": 0,
                   "source": "repo_read"}]}
  ],
  "traces": ["per-input or per-test behavior comparisons"],
  "counter_hypotheses": ["a distinguishing input, or why none exists"],
  "formal_conclusion": "equivalent | not equivalent, because ...",
  "confidence": "low | medium | high",
  "known_gaps": ["what you could not verify statically"]
}
```
