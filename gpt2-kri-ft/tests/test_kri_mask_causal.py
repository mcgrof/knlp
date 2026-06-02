"""Tests for the KRI attention mask.

Covers:
  - Strict causality: mask[..., t, k] is False for any k > t.
  - Model-level causality: changing tokens *after* position t never
    moves the logits at position t (under any KRI configuration).
  - Mask budget: the number of selected GLOBAL prefix blocks per query
    position is <= configured `global_topk_blocks`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kri_mask import KRIConfig, build_kri_mask, num_blocks  # noqa: E402
from src.model_gpt2_kri import GPT2KRI  # noqa: E402
from src.utils import pick_device, set_seed  # noqa: E402


def test_mask_is_strictly_causal() -> None:
    cfg = KRIConfig(
        block_size=16,
        local_window_tokens=64,
        global_topk_blocks=4,
        prefill_split=128,
        use_novelty=False,
    )
    T = 256
    mask = build_kri_mask(cfg, seq_len=T, batch_size=2, n_head=4, device=torch.device("cpu"))
    # The strict-upper-triangle (k > t) must be all False.
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    violations = (mask & upper.view(1, 1, T, T)).any().item()
    assert not violations, "mask attends to future tokens"
    print("PASS: mask is strictly causal")


def test_mask_budget() -> None:
    """The total kept positions per decode-region query is bounded:

        local_window_tokens
      + (block_size * |protected_blocks|)
      + (block_size * (1 + global_topk_blocks))   # current block + KRI

    We assert the *number of kept positions* per query t in the decode
    region is at most that bound. This is a soft upper bound — actual
    can be less when blocks overlap or KRI picks <topk.
    """
    T = 384
    cfg = KRIConfig(
        block_size=16,
        local_window_tokens=64,
        global_topk_blocks=4,
        prefill_split=128,
        use_novelty=False,
        protected_blocks=(0,),
    )
    mask = build_kri_mask(cfg, seq_len=T, batch_size=1, n_head=2, device=torch.device("cpu"))
    bound = (
        cfg.local_window_tokens
        + cfg.block_size * len(cfg.protected_blocks)
        + cfg.block_size * (1 + cfg.global_topk_blocks)
    )
    kept_per_query = mask[0, 0].sum(-1)  # [T]
    decode_kept = kept_per_query[cfg.prefill_split + 1 :]
    worst = int(decode_kept.max().item())
    assert worst <= bound, (
        f"some decode-region query attends to {worst} positions, "
        f"budget bound is {bound}"
    )
    print(f"PASS: max kept per decode query = {worst} <= bound {bound}")


def test_kri_topk_block_count() -> None:
    """Using K/V from a real model with `per_head=True`, verify that
    for each (batch, head, decode-query) the number of distinct
    *global prefix* blocks visible — excluding local window, protected
    sink, and current block — is <= global_topk_blocks.

    With `per_head=False` (the training-efficient union mode), the mask
    is shared across heads and may show *more* prefix blocks than
    top-k. That is by design; the strict budget claim is per-head.
    """
    cfg = KRIConfig(
        block_size=16,
        local_window_tokens=64,
        global_topk_blocks=4,
        prefill_split=128,
        use_novelty=False,
        protected_blocks=(0,),
        score_layer_index=6,
        per_head=True,
    )
    T = 256
    set_seed(0)
    device = pick_device()
    model = GPT2KRI.from_hf_gpt2().to(device).eval()
    H = model.cfg.n_head
    ids = torch.randint(0, model.cfg.vocab_size, (2, T), device=device)

    kvs = model.collect_kv(ids)
    k_per_layer = [kv[0] for kv in kvs]
    v_per_layer = [kv[1] for kv in kvs]
    # q at score layer is what attention computed from h's ln_1 output;
    # we approximate by using k since both come from the same projection
    # space. For this test we only care about *budget*, not selection
    # quality, so use k itself as the "query bank" — it makes the cosine
    # well-defined.
    q_per_layer = [kv[0] for kv in kvs]

    mask = build_kri_mask(
        cfg, seq_len=T, batch_size=ids.shape[0], n_head=H,
        k_per_layer=k_per_layer, v_per_layer=v_per_layer, q_per_layer=q_per_layer,
        device=device,
    )

    bs = cfg.block_size
    NB = num_blocks(T, bs)
    # For each (b, h, t) in decode region, count distinct prefix blocks
    # that lie strictly before the local window and aren't protected.
    bad = []
    for t in range(cfg.prefill_split + 1, T):
        local_start_block = max(0, t - cfg.local_window_tokens + 1) // bs
        for b in range(ids.shape[0]):
            for h in range(H):
                cols = mask[b, h, t].nonzero(as_tuple=True)[0]
                cols_blocks = (cols // bs).unique().tolist()
                # prefix blocks: < local_start_block, not protected
                prefix_blocks = [
                    bk for bk in cols_blocks
                    if bk < local_start_block and bk not in cfg.protected_blocks
                ]
                if len(prefix_blocks) > cfg.global_topk_blocks:
                    bad.append((b, h, t, len(prefix_blocks)))
    assert not bad, f"top-k violations: {bad[:5]}..."
    print(f"PASS: global prefix blocks per query <= {cfg.global_topk_blocks}")


def test_model_level_causality() -> None:
    """Take two sequences identical up to position t, different after.

    Then logits at position t (and everything before) must agree.
    We exercise the KRI mask path; this guards against accidental
    information flow from the future, e.g., a bug where block selection
    or the KRI score reads from k>t.
    """
    set_seed(7)
    device = pick_device()
    model = GPT2KRI.from_hf_gpt2().to(device).eval()
    H = model.cfg.n_head
    T = 256
    vocab = model.cfg.vocab_size

    base = torch.randint(0, vocab, (1, T), device=device)
    alt = base.clone()
    t_change = 130
    # Change everything after t_change to random different tokens
    alt[:, t_change + 1 :] = torch.randint(0, vocab, (1, T - t_change - 1), device=device)

    cfg = KRIConfig(
        block_size=16,
        local_window_tokens=64,
        global_topk_blocks=4,
        prefill_split=120,
        use_novelty=False,
        protected_blocks=(0,),
        score_layer_index=6,
    )

    # Build masks from each sequence's K/V independently. Causality must
    # hold even when the *mask* is built from each sequence's own K/V.
    with torch.no_grad():
        kvs_base = model.collect_kv(base)
        kvs_alt = model.collect_kv(alt)

    def fwd_with_kri(ids, kvs):
        k_per_layer = [kv[0] for kv in kvs]
        v_per_layer = [kv[1] for kv in kvs]
        q_per_layer = [kv[0] for kv in kvs]
        mask = build_kri_mask(
            cfg, seq_len=T, batch_size=ids.shape[0], n_head=H,
            k_per_layer=k_per_layer, v_per_layer=v_per_layer, q_per_layer=q_per_layer,
            device=device,
        )
        logits, _ = model(ids, attn_mask=mask)
        return logits.float()

    with torch.no_grad():
        l_base = fwd_with_kri(base, kvs_base)
        l_alt = fwd_with_kri(alt, kvs_alt)

    diff = (l_base[:, : t_change + 1, :] - l_alt[:, : t_change + 1, :]).abs()
    max_abs = diff.max().item()
    print(f"max_abs(logit_diff up to t={t_change}) = {max_abs:.3e}")
    assert max_abs < 1e-3, f"future tokens influenced logits at t<=t_change ({max_abs:.3e})"
    print("PASS: model-level causality under KRI mask")


def main() -> int:
    test_mask_is_strictly_causal()
    test_mask_budget()
    test_kri_topk_block_count()
    test_model_level_causality()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
