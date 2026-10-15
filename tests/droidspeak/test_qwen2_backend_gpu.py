import pytest

from research.droidspeak.profile import RecomputeInterval
from research.droidspeak.qwen2_backend import Qwen2DroidSpeak, tokenize_hotpot_prompt


class TinyTokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {str(index): index for index in range(64)}


class RecordingTokenizer:
    def encode(self, text, add_special_tokens):
        offset = 1 if add_special_tokens else 100
        return [offset + index for index, _ in enumerate(text.split())]


def test_hotpot_cache_covers_whole_prompt_except_final_input_token() -> None:
    tokenizer = RecordingTokenizer()
    context_ids, final_input_ids, original_cache_tokens = tokenize_hotpot_prompt(
        tokenizer,
        context="passage words",
        question="which answer",
        max_context_tokens=1_000,
    )

    passage_ids = tokenizer.encode(
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words. The following are given "
        "passages. passage words",
        add_special_tokens=True,
    )
    suffix_ids = tokenizer.encode(
        "Answer the question based on the given passages. Only give me the "
        "answer and do not output any other words. Question: which answer "
        "Answer:",
        add_special_tokens=False,
    )
    full_prompt = passage_ids + suffix_ids

    assert context_ids == full_prompt[:-1]
    assert final_input_ids == full_prompt[-1:]
    assert original_cache_tokens == len(full_prompt) - 1


def test_full_recompute_parity_and_hybrid_layer_sources() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(1)
    sender = transformers.Qwen2ForCausalLM(config).cuda().eval()
    torch.manual_seed(2)
    receiver = transformers.Qwen2ForCausalLM(config).cuda().eval()
    pair = Qwen2DroidSpeak(sender, receiver, TinyTokenizer())
    context_ids = torch.tensor([[4, 5, 6, 7, 8, 9]], device="cuda")
    prompt = type("Prompt", (), {"context_ids": context_ids})()

    sender_state = pair.capture_sender(context_ids)
    audit = pair.audit_full_recompute(prompt, sender_state)

    assert audit == {"max_key_abs_error": 0.0, "max_value_abs_error": 0.0}

    interval = RecomputeInterval(2, 4)
    hybrid = pair._recompute_context(context_ids, sender_state, interval)
    for layer_idx in range(pair.num_layers):
        hybrid_key, hybrid_value = hybrid[layer_idx]
        sender_key, sender_value = sender_state.kv_cache[layer_idx]
        if interval.contains(layer_idx):
            assert not torch.equal(hybrid_key, sender_key)
            assert not torch.equal(hybrid_value, sender_value)
        else:
            assert torch.equal(hybrid_key, sender_key)
            assert torch.equal(hybrid_value, sender_value)
