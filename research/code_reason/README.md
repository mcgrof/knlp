# Code Reasoning R&D harness

Reproduce and extend "Agentic Code Reasoning" (arXiv:2603.01896) inside
knlp's Kconfig workflow. Boring plumbing first: config, boundaries, and a
paper-faithful reproduction path before any addendums.

## Defconfigs
- `make defconfig-code-reason-paper` — paper reproduction only
  (static/execution-free verifier, standard + semi-formal, no addendums).
- `make defconfig-code-reason` — paper harness + knlp augmented certificate
  addendums and A-vs-blB artifacts, each independently ablatable.

Every toggle flows Kconfig -> `.config` -> `build/code_reason/config.json`
(via `scripts/gen_config_json.py`) -> the runner. No experiment policy lives
in Python constants.

## Ticket status
- [x] Ticket 1: Kconfig + defconfig plumbing + config.json + smoke test
- [x] Ticket 2: artifact schemas + writer
- [x] Ticket 3: paper prompt templates
- [x] Ticket 4: safe repo-exploration tools (execution-free guard)
- [x] Ticket 5: dataset manifest builders
- [x] Ticket 6: agent runner (model_client, agent_loop, tool_router)
- [x] Ticket 7: paper metrics
- [x] Ticket 8: augmented addendum registry
- [ ] Ticket 9: report generator
