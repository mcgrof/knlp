"""Discovery must read structure from the live module graph, never from names: GQA grouping, bias
presence, fused vs separate QKV, and the partial-RoPE subspace partition. Wrong discovery =
silently quantizing the wrong tensor on a GPU, the exact waste the brief warns about."""

import _fixtures as fx

from tools.kv.fp8_failure import architecture_discovery as AD


def test_gqa_biasless_fullrope_llama():
    infos = AD.discover(fx.tiny_llama(n_q=4, n_kv=2, head_dim=8))
    assert len(infos) == 2
    a = infos[0]
    assert a["n_q_heads"] == 4 and a["n_kv_heads"] == 2 and a["head_dim"] == 8
    assert a["fused"] is False
    assert a["has_k_bias"] is False and a["k_bias"] is None
    assert a["rotary_dim"] == 8 and a["is_partial_rope"] is False
    s = AD.summarize(infos)
    assert s["gqa_groups"] == 2 and s["has_k_bias"] is False


def test_biased_gqa_qwen2():
    infos = AD.discover(fx.tiny_qwen2(n_q=4, n_kv=2, head_dim=8))
    a = infos[0]
    assert (
        a["has_k_bias"] is True and a["has_q_bias"] is True and a["has_v_bias"] is True
    )
    assert a["k_bias"] is not None and tuple(a["k_bias"].shape) == (2 * 8,)
    assert AD.summarize(infos)["has_k_bias"] is True


def test_partial_rope_phi():
    infos = AD.discover(fx.tiny_phi(n_q=4, head_dim=8, partial_rotary_factor=0.5))
    a = infos[0]
    assert a["is_partial_rope"] is True
    assert a["rotary_dim"] == 4 and a["head_dim"] == 8
    rot, pas = a["rotary_mask"], a["passthrough_mask"]
    # exact partition of the head: OR=all, AND=none
    assert bool((rot | pas).all()) and not bool((rot & pas).any())
    assert int(rot.sum()) == 4 and int(pas.sum()) == 4
    assert bool(rot[:4].all()) and bool(pas[4:].all())


def test_gpt_neox_interleaved_partial_rope():
    # GPT-NeoX/Pythia: layers at gpt_neox.layers, fused interleaved QKV with a K-bias, and the
    # partial-rotary fraction lives in the rope dict (transformers 5.x), not a top-level attr.
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    m = GPTNeoXForCausalLM(
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
    infos = AD.discover(m)
    a = infos[0]
    assert a["fused"] is True and a["fused_interleaved"] is True
    assert a["has_k_bias"] is True  # NeoX query_key_value carries a bias
    assert a["head_dim"] == 8 and a["rotary_dim"] == 2  # 0.25 * 8
    assert a["is_partial_rope"] is True
    s = AD.summarize(infos)
    assert s["fused_interleaved"] is True and s["is_partial_rope"] is True
