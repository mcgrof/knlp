"""The mechanism discriminators must address the right channels. The interleaved pre-bias for
GPT-NeoX is the riskiest: it must modify ONLY the K slab of the per-head [Q K V] packing, never Q
or V. The subspace harness must quantize only its named subspace. Both are GPU-free verifiable.
"""

import torch

from tools.kv.fp8_failure import architecture_discovery as AD
from tools.kv.fp8_failure import mechanism_probe as MP


def _tiny_neox():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            rotary_pct=0.25,
            attn_implementation="sdpa",
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).eval()


def test_interleaved_prebias_touches_only_k_slab():
    m = _tiny_neox()
    infos = AD.discover(m)
    nh, hd = infos[0]["n_q_heads"], infos[0]["head_dim"]
    qkv = infos[0]["qkv_proj"]
    ids = torch.randint(0, 64, (1, 6))

    cap = {}
    h0 = qkv.register_forward_hook(
        lambda mod, i, o: cap.__setitem__("native", o.detach().clone())
    )
    with torch.no_grad():
        m(ids)
    h0.remove()

    harness = MP.InterleavedPrebiasHarness(m, infos).install()
    cap2 = {}
    h1 = qkv.register_forward_hook(
        lambda mod, i, o: cap2.__setitem__("preb", o.detach().clone())
    )
    with torch.no_grad():
        m(ids)
    h1.remove()
    harness.remove()

    nat = cap["native"].view(1, -1, nh, 3 * hd)
    preb = cap2["preb"].view(1, -1, nh, 3 * hd)
    # Q slab [0:hd] and V slab [2hd:3hd] must be byte-identical; only the K slab [hd:2hd] changes.
    assert torch.equal(nat[..., :hd], preb[..., :hd]), "Q slab must be untouched"
    assert torch.equal(
        nat[..., 2 * hd : 3 * hd], preb[..., 2 * hd : 3 * hd]
    ), "V slab untouched"
    assert not torch.equal(
        nat[..., hd : 2 * hd], preb[..., hd : 2 * hd]
    ), "K slab must change"


def test_subspace_full_rope_passthrough_is_noop():
    # On a full-RoPE model rotary_dim == head_dim, so passthrough_only quantizes nothing -> identical
    # logits, while rotary_only quantizes everything (== full).
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    m = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            attn_implementation="sdpa",
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).eval()
    infos = AD.discover(m)
    hd = infos[0]["head_dim"]
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        base = m(ids).logits
        with MP.SubspaceKHarness(m, infos, hd, "passthrough_only"):
            pas = m(ids).logits
        with MP.SubspaceKHarness(m, infos, hd, "rotary_only"):
            rot = m(ids).logits
        with MP.SubspaceKHarness(m, infos, hd, "full"):
            full = m(ids).logits
    assert torch.equal(base, pas), "full-RoPE passthrough_only must be a no-op"
    assert torch.equal(rot, full), "full-RoPE rotary_only must equal full-K quant"
    assert not torch.equal(base, full), "full-K quant must actually change logits"
