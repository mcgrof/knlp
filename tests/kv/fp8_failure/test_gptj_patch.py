"""GPT-J is eager-only in transformers 5.x and bypasses ALL_ATTENTION_FUNCTIONS, so the normal
harness silently no-ops it. The _attn patch must (a) actually intercept (K-nuke changes logits),
(b) move K under quant (audit delta > 0), (c) restore cleanly, and (d) FAIL CLOSED on a non-GPT-J
model rather than silently do nothing -- the whole point is to never publish a false 'tolerant'.
"""

import torch

from tools.kv.fp8_failure import architecture_discovery as AD
from tools.kv.fp8_failure import gptj_patch as GJ


def _tiny_gptj():
    from transformers import GPTJConfig, GPTJForCausalLM

    torch.manual_seed(0)
    return GPTJForCausalLM(
        GPTJConfig(
            vocab_size=64,
            n_embd=32,
            n_layer=2,
            n_head=4,
            rotary_dim=4,
            attn_implementation="eager",
            n_positions=128,
            tie_word_embeddings=False,
        )
    ).eval()


def test_gptj_patch_intercepts_and_is_causal():
    m = _tiny_gptj()
    infos = AD.discover(m)
    assert (
        infos[0]["has_k_bias"] is False
    )  # GPT-J attention is biasless -> the clean control
    info = GJ.verify_intercepts(m, infos, device="cpu")
    assert info["calls"] >= len(infos)  # one _attn call per layer
    assert info["max_k_delta"] > 0.0  # quant actually moved K
    assert (
        info["knuke_delta"] > 1e-4
    )  # K-nuke is causal in the logits (no-op would be exactly 0)


def test_gptj_patch_clean_restore():
    m = _tiny_gptj()
    infos = AD.discover(m)
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        base = m(ids).logits.clone()
        with GJ.GPTJKVAttnPatch(
            m,
            infos,
            __import__("k_bias_common").parse_spec("fp8:per_tensor"),
            __import__("k_bias_common").parse_spec("bf16"),
        ):
            quant = m(ids).logits.clone()
        restored = m(ids).logits.clone()
    assert not torch.equal(base, quant), "quant must change logits"
    assert torch.equal(base, restored), "restore must be exact"


def test_gptj_patch_fails_closed_on_non_gptj():
    # Pointing the GPT-J patch at a non-GPT-J model must raise, not silently no-op.
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
            attn_implementation="eager",
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).eval()
    infos = AD.discover(m)
    import k_bias_common as kbc

    try:
        GJ.GPTJKVAttnPatch(
            m, infos, kbc.parse_spec("fp8:per_tensor"), kbc.parse_spec("bf16")
        ).install()
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "GPT-J patch must fail closed on a non-GPTJAttention model"
