"""Check replay diagnostic instrumentation against a tiny CPU model."""

import unittest


class ReplayTraceTests(unittest.TestCase):
    def test_layer_trace_and_absolute_attention_reference(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from tools.kv.prequantized.drift import Trace

        config = Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
        )
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval()
        with torch.inference_mode(), Trace(model) as trace:
            model(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
        self.assertIn("0.q", trace.values)
        self.assertIn("1.residual", trace.values)
        reference = trace.attention_reference(3)
        self.assertLess(reference["fp32_reference_max_error"], 1e-5)
        self.assertEqual(reference["key_positions"], [0, 3])


if __name__ == "__main__":
    unittest.main()
