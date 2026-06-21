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


def test_passthrough_prebias_beats_passthrough_only_when_bias_in_tail():
    # The confound-breaker: with a large K-bias injected on the un-rotated PASS-THROUGH channels,
    # passthrough_only (quantizes the biased tail) is much worse than passthrough_prebias (subtracts
    # that bias first). This is exactly the partial-RoPE-legacy case where bias lives in the tail.
    from transformers import PhiConfig, PhiForCausalLM

    torch.manual_seed(0)
    m = PhiForCausalLM(
        PhiConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=8,
            partial_rotary_factor=0.5,  # rotary_dim=4, pass-through channels [4:8]
            attn_implementation="sdpa",
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).eval()
    rd, hd = 4, 8
    # inject a big DC offset on the pass-through K channels of every layer
    for layer in m.model.layers:
        b = layer.self_attn.k_proj.bias.detach().view(4, hd)
        b[:, rd:] = 30.0
        layer.self_attn.k_proj.bias.data.copy_(b.view(-1))

    infos = AD.discover(m)
    pass_bias = {
        i["layer_idx"]: MP.kbias_per_head(i)[:, rd:] for i in infos
    }  # [n_kv, n_pass]
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        base = [m(ids).logits[0].float()]

        def err(mode, pb=None):
            with MP.SubspaceKHarness(m, infos, rd, mode, pass_bias_by_layer=pb):
                lg = m(ids).logits[0].float()
            return (base[0] - lg).abs().mean().item()

        e_only = err("passthrough_only")
        e_pb = err("passthrough_prebias", pass_bias)
    assert (
        e_pb < e_only
    ), f"prebias must beat passthrough_only with bias in tail ({e_pb} vs {e_only})"


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
