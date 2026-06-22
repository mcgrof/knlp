"""The 72B device_map path must (a) parse k_proj.bias layer indices from real safetensors key names,
(b) cap memory across all GPUs with no cpu key (fail-fast, not meta-offload), and (c) FAIL CLOSED if
any parameter is on meta -- a meta param means the model didn't fit and the forward would die mid-
graph. All GPU-free verifiable."""

import re

import torch

from tools.kv.fp8_failure import run_smoke as RS


def test_kbias_safetensors_key_regex():
    # the regex run_smoke uses to map a safetensors key -> layer index
    rx = re.compile(r"layers\.(\d+)\..*k_proj\.bias$")
    assert rx.search("model.layers.0.self_attn.k_proj.bias").group(1) == "0"
    assert rx.search("model.layers.41.self_attn.k_proj.bias").group(1) == "41"
    assert (
        rx.search("model.layers.5.self_attn.k_proj.weight") is None
    )  # weight, not bias
    assert rx.search("model.layers.5.self_attn.q_proj.bias") is None  # q, not k


def test_multi_gpu_max_memory_no_cpu_key():
    mm = RS._multi_gpu_max_memory()  # CPU box: 0 GPUs -> empty dict, never a 'cpu' key
    assert "cpu" not in mm
    assert all(isinstance(k, int) for k in mm)  # only integer GPU ids


def test_repair_and_guard_passes_clean_model():
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
    from tools.kv.fp8_failure import architecture_discovery as AD

    infos = AD.discover(m)
    RS._repair_and_guard(
        m, infos, "tiny/nonexistent"
    )  # no meta params -> must not raise


def test_repair_and_guard_fails_closed_on_meta_param():
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
    from tools.kv.fp8_failure import architecture_discovery as AD

    infos = AD.discover(m)
    # simulate an offloaded weight landing on meta (accelerate disk/meta offload)
    import torch.nn as nn

    w = m.model.layers[0].self_attn.v_proj.weight
    m.model.layers[0].self_attn.v_proj.weight = nn.Parameter(
        torch.empty(w.shape, device="meta"), requires_grad=False
    )
    raised = False
    try:
        RS._repair_and_guard(m, infos, "tiny/nonexistent")
    except RuntimeError as e:
        raised = "meta" in str(e)
    assert raised, "a param on meta must raise a clear error, not pass silently"
