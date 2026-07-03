<!-- Shared execution-free constraints, prepended by the runner in paper mode. -->
## Constraints (paper baseline, execution-free)
You may INSPECT the repository but you MUST NOT:
- execute target repository code or import its modules,
- run target repository tests (pytest, mvn/gradle test, npm test, cargo test),
- install target dependencies,
- read git history (paper reproduction disables it).

You reason from static evidence only. Available read-only tools:
`repo_read(path, line_start, line_end)`, `list_files(glob)`,
`grep(pattern)`, `ast(path)` (parser output, not execution).
Every claim you make must cite concrete evidence (file + line range).
