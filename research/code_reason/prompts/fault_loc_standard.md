# Task: Fault localization (standard agentic reasoning)

A bug is described below. Identify the source locations most likely to
contain the fault, ranked most-likely first.

Repository root: {repo_root}
Bug report / failing behavior:
{bug_report}

Inspect the code statically and output a ranked list:

```json
{"ranked_locations": [{"file": "...", "line_start": 0, "line_end": 0}]}
```
