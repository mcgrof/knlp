# Routing thread context — 2026-03-31

Target thread:
- `routing` (`<#1488645192012529896>`)

Goal of the thread handoff:
- continue routing analysis/reporting
- use the current practical-prior findings to write a report in the style of `~/devel/cartridges-engineering-guide/`
- intended report location: `~/devel/routing-analysis`

## Read first
1. `/data/knlp/docs/routing_triton_kernel_status_20260331.md`
2. `/data/knlp/docs/routing_idea_provenance_20260331.md`
3. `/home/mcgrof/.openclaw/workspace/results/routing_practical_prior_results_20260331.md`
4. `/home/mcgrof/.openclaw/workspace/results/routing_vllm_gap_audit_20260331.md`
5. `/home/mcgrof/.openclaw/workspace/results/routing_prior_provenance_20260331.md`
6. `/home/mcgrof/.openclaw/workspace/logs/routing-accuracy-prior-execution-20260331.log`
7. `/home/mcgrof/.openclaw/workspace/logs/triton-routing-kernel-vllm-retry-20260331.log`
8. `/home/mcgrof/.openclaw/workspace/plans/routing-accuracy-prior-execution-plan-20260331.md`

## Durable results roots
- kernel + earlier routed inference root:
  - `/data/knlp-key-results/paper-router/triton-fused-routing-kernel-20260331T191500Z/`
- practical-prior phase root:
  - `/data/knlp-key-results/paper-router/routing-practical-priors-20260331T214200Z/`

## Repo-local results/code/docs
- routing code tree:
  - `/data/knlp/routing/`
- practical-prior analysis bundle:
  - `/data/knlp/results/routing-practical-priors-20260331T214200Z/`
- helper code copy:
  - `/data/knlp/tools/routing_analysis/routing_prior_extraction_sweep.py`
- docs:
  - `/data/knlp/docs/routing_triton_kernel_status_20260331.md`
  - `/data/knlp/docs/routing_idea_provenance_20260331.md`
  - `/data/knlp/docs/routing_thread_context_20260331.md`

## Main findings to remember
### Kernel phase
- fused Triton kernel solved the Python-loop overhead problem
- large microbench wins vs Python-loop reference were real
- but TTFT / serving-path claims are still not proven

### Practical-prior phase
- provenance-check showed the older ~0.914 cosine result came from a synthetic benchmark path, not a ready-made serving truth
- practical prefill-derived priors **did** improve routing substantially
- first credible operating points now exist
- next rational step is downstream validation + minimal vLLM integration, not more vague debate about whether routing can ever work

## Provenance summary
The routing idea line came from:
- FIM-guided reciprocal-attention / structural approximation work
- later realization that attention probability traces likely give better **functional routing priors**
- eigenvalues/eigenvectors as a way to think about allocation strength and direction across heads
- later kernel work proving routing speed was worth taking seriously

See:
- `/data/knlp/docs/routing_idea_provenance_20260331.md`

## Repos / hosts
- main repo with copied routing code/docs/results:
  - `prune:/data/knlp`
- initial draft/prototype area to build up over time:
  - `monster:/home/mcgrof/devel/paper-router`
- intended report destination for this topic:
  - `monster:/home/mcgrof/devel/routing-analysis`
- style inspiration for report structure:
  - `monster:/home/mcgrof/devel/cartridges-engineering-guide/`

## Best starting ask for the thread
Ask it to:
1. read the files above
2. verify the copied practical-prior bundle in `prune:/data/knlp*`
3. write a routing report outline in the style of `cartridges-engineering-guide`
4. place/report that work under `~/devel/routing-analysis`
5. preserve the provenance story: FIM-guided approximation → attention-probability functional priors → eigenstructure → routing
