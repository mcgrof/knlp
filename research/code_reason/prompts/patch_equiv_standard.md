# Task: Patch equivalence verification (standard agentic reasoning)

You are given two patches to the same repository. Decide whether they are
**behaviorally equivalent**: do they produce the same observable behavior on
all inputs and tests?

Repository root: {repo_root}
Patch A:
{patch_a}
Patch B:
{patch_b}

Explore the affected files and reason directly. Then output a final answer:

```json
{"equivalent": true_or_false, "rationale": "one paragraph"}
```
