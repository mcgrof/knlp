"""
agent_task_rwr_attention.py
=====================================================================
Task spec for integrating Random-Walk-with-Restart (RWR) into attention,
with (a) reciprocal/reversible Markov splitting, (b) FlashAttention-style
SRAM tiling, and (c) tensor-core-friendly compute layouts.

This doc is the "source of truth" for what to build, why, and how we’ll
judge success. It’s designed to slot into the same project where
Lens-Gated / RA / MLA work lives.

---------------------------------------------------------------------
GOALS
---------------------------------------------------------------------
G1) Add an RWR-augmented attention path that factorizes attention into:
      A(q_i, :)  ≈  LOCAL(i)  +  RWR(i)
    where:
      - LOCAL(i): standard short-range or block window attention
      - RWR(i):   long-range structure captured via a token-graph RWR
    This yields fewer full QK^T matmuls, better cache behavior, and a
    natural “importance” signal from hitting probabilities.

G2) Support reciprocal / reversible Markov variants:
      - Reciprocal lens: couple S and S^T views (consistent with RA lens)
      - Reversible chain: enforce detailed balance (symmetrized transition)

G3) FlashAttention-style tiling for SRAM:
      - Tile KV into on-chip SRAM blocks
      - Stream Q blocks; fuse softmax with accumulation when applicable
      - Keep tensor core utilization high (multiples of 16 for fp16/bf16)

G4) Expose knobs to choose:
      - Window size for LOCAL
      - RWR restart α, walk steps T (or power series truncation)
      - Graph sparsification policy (top-k, threshold, banded + jumps)
      - Reversible vs plain RWR
      - Reciprocal coupling strength with S^T lens

G5) Clean APIs, AMP-safe, and drop-in compatible with our patcher.

---------------------------------------------------------------------
INTUITION / MATH SKETCH
---------------------------------------------------------------------
Let tokens be nodes 1..n with similarity edges W (e.g., from cosine(Q,K),
MI, or learned sparsifier). Row-normalize to P = D^{-1} W (D = diag(row sums)).
RWR vector for query i:
    r_i = α * e_i + (1-α) * r_i P
Fixed-point solution:
    r_i = α * e_i (I - (1-α) P)^{-1}
Use truncated Neumann series:
    r_i ≈ α * sum_{t=0..T} ((1-α) P)^t e_i
We never form the dense inverse. We apply T steps of sparse matvecs with P.
Top entries of r_i indicate long-range “visitation” importance.

Reversible (detailed-balance) variant:
    P_rev = 1/2 ( P + D^{-1} P^T D )
This guarantees a reversible chain w.r.t. stationary π ∝ D•1, improving
stability and smoothing asymmetries (pairs nicely with RA’s S vs S^T lens).

Reciprocal coupling (lens):
We mix forward and backward walk saliency:
    r_i^recip = β * r_i   + (1-β) * ( “backward saliency” from P^T )
Optionally use the discoverability column bias as in the lens.

Final attention weights per query i:
    A(i,:) = softmax(  LOCAL(i,:)  ⊕  γ * PROJECT( r_i^recip )  )
where PROJECT maps RWR scores onto value indices (use the same K index space).
⊕ denotes (masked) addition over indices participating in each term.

---------------------------------------------------------------------
HIGH-LEVEL IMPLEMENTATION PLAN
---------------------------------------------------------------------
T1. Graph builder (per batch)
    - Build W sparsely:
        * candidate: top-k over cosine(Q,K) in tiled blocks (no full QK)
        * optionally add local band edges (±w window) to ensure connectivity
    - Row-normalize to P; store in block-sparse CSC/CSR (torch.sparse_csr).

T2. RWR engine
    - For each query block (size Bq):
        * Seed vectors E (Bq x n) as one-hot rows for token indices in the block
          (or batched identity-gather trick).
        * Iteratively: R_{t+1} = αE + (1-α) R_t P   for t=0..T  (sparse mm)
        * Option: reversible chain → use P_rev
        * Option: reciprocal → mix forward with backward transport
    - Keep everything batched & sparse; use bf16/fp16 math but accumulate in fp32 if needed.

T3. Attention factorization (compute split)
    - LOCAL(i): standard windowed (or block) attention computed with
      FlashAttention-like SRAM tiling:
        * Tile sizes tuned to (128, 256, …); head_dim padded to multiples of 16/32
        * Use fused softmax for numeric stability
    - RWR(i): form logits only on the sparse support selected by r_i (e.g., top-k
      hits or thresholded). Do a sparse gather of V on those indices and compute
      the weighted sum. No dense QK for these pairs.

T4. SRAM tiling & tensor-core fitting
    - Tile K,V into blocks that fit in SRAM (e.g., 128xhead_dim per block)
    - Stream Q blocks (Bq x head_dim)
    - Ensure operands are shaped to use tensor cores: (Bq, head_dim) x (head_dim, Bk)
      with head_dim multiple of 16 and Bq/Bk chosen {64, 128, 256} depending on GPU.
    - For RWR matvecs, operate on block-sparse structures that align with the same
      K/V tiling when possible to minimize extra movement.

T5. API wiring
    - New module: RWRKernelAttention
        forward(q, k, v, *, rwr_cfg, local_cfg, lens_cfg) -> y
      rwr_cfg: {alpha, steps, reversible, reciprocal_beta, topk, threshold}
      local_cfg: {window, block_size, causal_mask}
      lens_cfg: {use_discoverability, lens_strength γ, column_bias u_h}
    - Patcher: patch_gpt2_with_rwr_attention(model, ...)
      toggles replacing standard attention with RWRKernelAttention.

T6. Logging & metrics
    - Log % of pairs handled by LOCAL vs RWR
    - RWR entropy / concentration stats
    - Hit-rate vs true softmax baseline on a probe batch (optional)
    - Throughput (tokens/s) and SRAM tile occupancy

---------------------------------------------------------------------
FILES TO ADD / TOUCH
---------------------------------------------------------------------
ADD:
  - ./rwr_attention.py                (RWRKernelAttention + helpers)
  - ./lib/graph_builder.py            (W/P construction, reversible symmetrization)
  - ./docs/rwr_attention.md           (design, tips, pitfalls)

TOUCH:
  - ./train_ra_mla.py                 (CLI flags, registry)
  - ./ra_lens_gpt2.py                 (optional: lens coupling hooks/shared bias)
  - ./lib/optimizers.py               (no changes required; ensure AMP interactions ok)

---------------------------------------------------------------------
CLI (examples; add via argparse)
---------------------------------------------------------------------
  --attn-kind rwr
  --rwr-alpha 0.2
  --rwr-steps 4
  --rwr-topk 32
  --rwr-threshold 0.0
  --rwr-reversible            # enable P_rev
  --rwr-reciprocal-beta 0.5   # mix forward/backward saliency
  --rwr-window 128            # LOCAL window half-width
  --rwr-block-size 128        # SRAM tile size
  --rwr-lens-strength 0.3     # γ
  --rwr-use-discoverability   # enable lens column bias
  --rwr-head-dim-pad 64       # round up head_dim to multiple for tensor cores

---------------------------------------------------------------------
ACCEPTANCE CRITERIA
---------------------------------------------------------------------
A1) Dry run:
    python train_ra_mla.py --dry-run --attn-kind rwr --rwr-steps 1
    (CPU ok; builds sparse graph, 1 step RWR, forward pass succeeds)

A2) Numerical sanity vs dense baseline on tiny seq:
    - For seq_len<=128, compare RWR+LOCAL vs dense softmax attention outputs
    - L2 diff below a loose tolerance when rwr_topk≈full and window≈full

A3) Throughput:
    - With seq_len=4k, heads=16, dim_head padded to 64 or 128, bf16:
      RWR attention should show ≥20–35% lower HBM traffic than dense attention
      at same batch/head settings (report nvtx or timing + memory stats)

A4) Tensor core utilization:
    - Nsight / torch compile logs show GEMMs on tensor cores for local tiles
    - Head dim multiples satisfied; no silent fp32 fallbacks

A5) Stability:
    - Reversible chain toggling does not blow up (loss finite for 1k iters)
    - Reciprocal beta in [0,1] smoothly interpolates behavior

---------------------------------------------------------------------
IMPLEMENTATION DETAILS (CONCRETE)
---------------------------------------------------------------------
[lib/graph_builder.py]
  - build_sparse_W(Q, K, topk, window, thresh):
      * Tiled cosine(Q,K) search: for each (Q_block x K_block), compute scores,
        keep local window + topk extras, drop others
      * Return indices + values per head (batch-aware)

  - normalize_to_P(W):
      * Row-sum per node, divide (avoid zero by epsilon), return CSR sparse

  - reversible(P):
      * with D = diag(row_sums_of_W), compute P_rev = 0.5*(P + D^{-1} P^T D)
      * Implement with sparse ops (coalesce + index tricks)

[rwr_attention.py]
  - class RWRKernelAttention(nn.Module):
        def __init__(..., cfgs):
            # store cfg, alloc temp buffers, padding config
        def forward(self, q, k, v, **cfg):
            # 1) SRAM tiling params (block_size, head_dim_pad)
            # 2) Build or reuse sparse P (cache if keys unchanged in kv cache scenario)
            # 3) Compute LOCAL logits/y using FA-style tiled kernel
            # 4) Compute RWR saliency:
            #      R0 = E (one-hots for queries in this block)
            #      for t in 1..T: R = alpha*E + (1-alpha)*R @ P_or_Prev
            #      if reciprocal: R = beta*R + (1-beta)*R_backward
            #    Select topk or threshold indices per row; gather V; do weighted sum
            # 5) Fuse outputs: y = y_local + γ * y_rwr
            #    If lens/discoverability enabled, bias logits before softmax
            # 6) Return y (and optional debug stats)

  - FlashAttention-style local path:
        * Implement tiled QK^T, online softmax, and PV accumulation
        * Or call an existing FA kernel if available; keep fallback PyTorch path

  - Sparse utils:
        * batched_sparse_mm(R, P) using torch.sparse.mm or custom block-sparse
        * Keep accumulation in fp32; cast back to bf16/fp16

[docs/rwr_attention.md]
  - RWR math and truncation trade-offs
  - Reversible vs reciprocal differences
  - Tile size recipes per GPU (A100/H100 vs MI300): head_dim 64/128, block 128/256
  - Debug playbook (how to verify sparsity, topk, correctness)

---------------------------------------------------------------------
DEFAULTS / STARTING HYPERPARAMS
---------------------------------------------------------------------
- α (restart): 0.2
- T (steps):   4
- topk:        32 (per query row after LOCAL window)
- window:      128
- reversible:  off (start), then on for ablation
- reciprocal β: 0.5
- lens γ:      0.3
- head_dim_pad: 64  (pad d_head up to multiple of 64)

---------------------------------------------------------------------
RISKS & GUARDRAILS
---------------------------------------------------------------------
- Graph build cost: amortize by reusing P when KV cache is reused
- Numerical issues: clamp row-sums, use fp32 accumulations in matvec chain
- Don’t explode memory: keep W/P as block-sparse; avoid converting to dense
- Keep a “degenerate” mode: set T=0 or γ=0 to validate LOCAL path alone
"""

# Optional reference stubs the agent can copy and fill:

RWR_ATTENTION_REFERENCE_STUB = r'''
import torch
import torch.nn as nn
from typing import Optional

class RWRKernelAttention(nn.Module):
    def __init__(self, dim_head: int, rwr_alpha=0.2, rwr_steps=4,
                 rwr_topk=32, rwr_threshold=0.0, reversible=False,
                 reciprocal_beta=0.5, lens_strength=0.3,
                 window=128, block_size=128, head_dim_pad=64,
                 use_discoverability=False):
        super().__init__()
        self.dim_head = dim_head
        self.pad = head_dim_pad
        self.rwr_alpha = rwr_alpha
        self.rwr_steps = rwr_steps
        self.rwr_topk = rwr_topk
        self.rwr_threshold = rwr_threshold
        self.reversible = reversible
        self.reciprocal_beta = reciprocal_beta
        self.lens_strength = lens_strength
        self.window = window
        self.block_size = block_size
        self.use_discoverability = use_discoverability
        # TODO: buffers/temps as needed

    def forward(self, q, k, v, *, P_sparse=None, P_rev_sparse=None):
        """
        q,k,v: [B, H, N, Dh] with Dh padded to multiple of self.pad if needed.

        P_sparse/P_rev_sparse: optional prebuilt sparse transition matrices
        (per batch/head or shared layout). If None, build via graph_builder.
        """
        # 1) Local attention via tiled SRAM kernel (fallback: naive)
        y_local = self._local_flash_like(q, k, v)

        # 2) Build P (or use cached)
        if P_sparse is None:
            P_sparse, P_rev_sparse = self._build_sparse_P(q, k)

        # 3) RWR saliency for current query block(s)
        y_rwr = self._rwr_long_range(q, v, P_sparse, P_rev_sparse)

        # 4) Fuse
        y = y_local + self.lens_strength * y_rwr
        return y

    def _local_flash_like(self, q, k, v):
        # TODO: implement tiled QK^T, online softmax, PV
        # For now: simple fallback (ensure masking/causal if needed)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = attn.softmax(dim=-1)
        return torch.matmul(attn, v)

    def _build_sparse_P(self, q, k):
        # TODO: call into lib/graph_builder.build_sparse_W -> normalize_to_P
        # TODO: reversible symmetrization if self.reversible
        raise NotImplementedError

    def _rwr_long_range(self, q, v, P_sparse, P_rev_sparse):
        # TODO: implement batched truncated RWR with sparse matvecs
        # - build E for query rows (block-wise)
        # - R_{t+1} = αE + (1-α) R_t @ P
        # - reciprocal / reversible hooks
        # - select topk/threshold indices, gather V, weighted sum
        raise NotImplementedError
'''

GRAPH_BUILDER_REFERENCE_STUB = r'''
import torch

def build_sparse_W(Q, K, topk=32, window=128, threshold=0.0):
    """
    Return indices+values for a sparse similarity graph W.
    Tiled search: keep local window ±window plus topk global neighbors.
    """
    raise NotImplementedError

def normalize_to_P(W_sparse):
    """
    Row-normalize W to P = D^{-1} W. Return CSR/COO sparse tensor.
    """
    raise NotImplementedError

def reversible(P_sparse, D_row_sums):
    """
    Compute P_rev = 0.5*(P + D^{-1} P^T D) in sparse form.
    """
    raise NotImplementedError
'''
