"""Tiny offline model fixtures for the FP8-atlas unit tests. Every model here is built from a
config with random weights -- no download, no GPU, constructed in milliseconds. They exercise the
four structural axes the atlas cares about: GQA grouping, K-bias presence, partial RoPE, and fused
QKV.
"""

import torch
import torch.nn as nn

_COMMON = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    attn_implementation="sdpa",
    max_position_embeddings=64,
    tie_word_embeddings=False,
)


def tiny_llama(n_q=4, n_kv=2, head_dim=8, seed=0):
    """GQA, biasless, full RoPE (the common modern-small-model shape)."""
    torch.manual_seed(seed)
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        rope_theta=10000.0,
        **_COMMON,
    )
    return LlamaForCausalLM(cfg).eval()


def tiny_qwen2(n_q=4, n_kv=2, head_dim=8, seed=0):
    """GQA, K/Q/V bias present, full RoPE (the Qwen FP8-K-fragility shape)."""
    torch.manual_seed(seed)
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        rope_theta=10000.0,
        **_COMMON,
    )
    return Qwen2ForCausalLM(cfg).eval()


def tiny_phi(n_q=4, head_dim=8, partial_rotary_factor=0.5, seed=0):
    """MHA, partial RoPE (rotary_dim = head_dim*factor), the Phi-2 mixed-subspace shape."""
    torch.manual_seed(seed)
    from transformers import PhiConfig, PhiForCausalLM

    cfg = PhiConfig(
        num_attention_heads=n_q,
        head_dim=head_dim,
        partial_rotary_factor=partial_rotary_factor,
        **_COMMON,
    )
    return PhiForCausalLM(cfg).eval()


def synthetic_fused_info(n_q=4, n_kv=2, head_dim=8, bias=True, seed=0):
    """A minimal `info` dict (as architecture_discovery.discover would emit) backed by a REAL fused
    qkv nn.Linear, for testing fold_qk_gauge_weights on the fused path without a full model.
    """
    torch.manual_seed(seed)
    dim = (n_q + 2 * n_kv) * head_dim

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = nn.Linear(head_dim * n_q, dim, bias=bias)
            self.num_key_value_groups = n_q // n_kv

    attn = _Attn()
    return dict(
        layer_idx=0,
        attn_module=attn,
        n_q_heads=n_q,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        fused=True,
        qkv_proj=attn.qkv_proj,
        k_proj=None,
    )
