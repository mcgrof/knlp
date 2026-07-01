# KRI-TierKV

KRI-TierKV is a prototype for KRI-guided tiered KV-cache eviction and sparse
retrieval. The KV cache is split into fixed-size token blocks; recent blocks stay
in a fast tier, old blocks move to a slow tier, and KRI-D-sum decides which old
blocks are worth keeping and fetching. Sparse attention then reads the recent
fast window plus the top-K selected slow blocks. Asymmetric quantization is
layered on only after that selection works.

The first version is deliberately boring: it runs normal dense inference and
records what each policy *would* keep, fetch, and skip, so we can answer one
question before building anything expensive — does KRI-D-sum pick the old blocks
that actually carry attention mass, better than FIFO or a recent-only window? If
it does, a real offload path is worth building. If it does not, we learned that
cheaply.

## Attribution

The temporal-tiering framing is inspired by *TTKV: Temporal-Tiered KV Cache for
Long-Context LLM Inference* (given as arXiv:2604.19769), which partitions the KV
cache into a fast HBM tier and a slow DRAM tier with heterogeneous precision and
overlaps slow-tier access with attention work. KRI-TierKV is **not** a TTKV
clone: it keeps the tiering idea and replaces the tiering policy with KRI-D-sum
block selection plus asymmetric fake quantization.

Two caveats are load-bearing. First, the TTKV arXiv id (2604.19769, dated April
2026) could not be verified from the authoring assistant's training data; confirm
the id, title, and tiering claim against arXiv before publishing, and keep the
attribution generic if it does not resolve. Second, on the A100 (Ampere), FP8 is
treated as fake/storage quantization only — quantize-to-int storage plus dequant
before attention. Native FP8 Transformer Engine throughput is a Hopper/H100
feature, so nothing here claims an FP8 speedup; the quantization work is a
memory-traffic and quality study.

## How the pieces fit

The cache is a sequence of blocks (default 128 tokens). A `BlockIndex` tracks each
block's tier, its KRI-D-sum score, when it was evicted, and how often it was
fetched back, and it knows the protected set that eviction must never touch: the
system/prefix region, the recent fast window, and the neighborhood of the current
decode point.

`KRI-D-sum` scores a block as the norm of its summed keys plus the norm of its
summed values, aggregated over layers and heads — the same quantity computed in
`gpt2-kri-ft/src/canonical_kri.py`. It is query-independent: computed once from
the prefix, so a block's tier and score do not depend on the decode query. That
property is what makes it a candidate for a prefix-stable tiering policy rather
than a per-query router.

Eviction keeps the highest-scoring evictable blocks up to the slow-tier budget
and drops the rest; `fifo` and `recency` are the positional baselines,
`kri_d_sum` is the score-driven policy. Retrieval chooses what a decode step
reads: `dense_reference` (all blocks), `recent_only` (fast tier only), `kri_topk`
(fast plus the top-K slow blocks by KRI-D-sum), and `oracle_topk` (top-K slow by
measured attention mass, for offline comparison only). The oracle is the ground
truth KRI-D-sum is trying to approximate.

## Configuration

The Kconfig symbols live in `Kconfig.kri_tierkv`:

- `CONFIG_KNLP_KRI_TIERKV` — enable the feature (default off).
- `CONFIG_KNLP_KRI_TIERKV_EMU` — emulation mode (record decisions, do not alter
  attention).
- `CONFIG_KNLP_KRI_TIERKV_KRI_D_SUM` — use KRI-D-sum as the block selector.
- `CONFIG_KNLP_KRI_TIERKV_ASYM_QUANT` — enable asymmetric fake quantization.
- `CONFIG_KNLP_KRI_TIERKV_V_ONLY` — quantize V only, keep K at 16 bits (the safe
  default when quant is on).
- `CONFIG_KNLP_KRI_TIERKV_HTML_REPORT` — emit the HTML report.

The safe quantization default is K16/V8; any sub-16-bit key policy (`k8v4`) is
opt-in and flagged unsafe, because on fragile-key models the keys collapse under
low precision. This follows the KV-protection Pareto finding that K precision is
the dominant lever.

## Layout note

The code lives at top-level `kri_tierkv/` (with the experiment under
`experiments/`), matching the repo's existing top-level packages `routing/` and
`gpt2/`. The queued spec referred to a `knlp/kri_tierkv/` path, but this repo has
no `knlp/` namespace package — the repository itself is knlp — so the top-level
placement is the faithful mapping.

## Status and next steps

Milestone 1 (emulation) is the current landed core: block bookkeeping, the
scoring and policy functions, fake quantization, the JSONL trace, and the
attention-mass recall / high-mass false-negative / bytes-moved metrics, with unit
tests. The experiment runner drives a real model on an A100 to compute per-block
KRI-D-sum and the attention-mass oracle, then reports how each policy scores.

Milestone 2 adds an experimental sparse-attention mask so the selected blocks
actually drive a forward pass, compared against the dense reference on needle and
QA tasks. Milestone 3 adds the asymmetric quant runs (K16/V8, K16/V4, and the
opt-in K8/V4). Real DRAM/SSD paging is deliberately out of scope until the
emulation recall numbers justify building it.
