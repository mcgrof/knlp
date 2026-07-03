# Task: Fault localization (semi-formal certificate)

A bug is described below. Localize the fault and justify each candidate with
static evidence. Repository root: {repo_root}
Bug report / failing behavior:
{bug_report}

Output ONLY this JSON certificate (schema: certificate.schema.json):

```json
{
  "task_type": "fault_localization",
  "answer": {"ranked_locations": [{"file": "...", "line_start": 0,
                                    "line_end": 0}]},
  "definitions": [],
  "premises": [
    {"id": "P1", "claim": "why this location is implicated",
     "evidence": [{"file": "...", "line_start": 0, "line_end": 0,
                   "source": "ast"}]}
  ],
  "traces": ["control/data-flow path from symptom to candidate"],
  "counter_hypotheses": ["locations considered and ruled out, with reason"],
  "formal_conclusion": "ranked candidates, most-likely first, because ...",
  "confidence": "low | medium | high",
  "known_gaps": []
}
```
