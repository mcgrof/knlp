# Low-rank-V kernel (preserved, parked)

Exploration code for a low-rank value-cache decode kernel from the pos-quant-v
line. Kept here for provenance only — it did not produce a deployable win, so it
is intentionally **not** written up. The mechanism and a ~4× traffic reduction
were real, but the rank-32 fidelity turned out to be downstream robustness rather
than genuine per-layer low rank (some layers' V is near full rank), so the line
was parked.

`vlowrank_kernel.py` is the kernel; `vlowrank_bench.py` / `vlowrank_latency.py`
and the `*_pareto.py` scripts are its timing/quality harnesses;
`gen_v_artifact.py` builds the offline tensors (regenerated, not committed).
