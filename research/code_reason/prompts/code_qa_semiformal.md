# Task: Code question answering (semi-formal certificate)

Answer the question about the repository, grounding every claim in evidence.
Repository root: {repo_root}
Question:
{question}

Output ONLY this JSON certificate (schema: certificate.schema.json):

```json
{
  "task_type": "code_qa",
  "answer": {"text": "your answer"},
  "definitions": [],
  "premises": [
    {"id": "P1", "claim": "supporting fact",
     "evidence": [{"file": "...", "line_start": 0, "line_end": 0,
                   "source": "repo_read"}]}
  ],
  "traces": [],
  "counter_hypotheses": ["alternative answers considered and why rejected"],
  "formal_conclusion": "the answer, justified by the premises above",
  "confidence": "low | medium | high",
  "known_gaps": []
}
```
