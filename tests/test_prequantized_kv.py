"""Numerical and state-contract checks before renting a GPU."""

import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config

from tools.kv.prequantized.cache import (
    AttentionReadProof,
    Observer,
    make_cache,
    qdq,
    reference_e4m3,
)
from tools.kv.prequantized.contract import (
    append_result,
    completed_rows,
    digest,
    paired_nll,
    paired_retrieval,
    pinned,
    validate_manifest,
    validate_matrix,
    write_once,
)


class QuantizerTests(unittest.TestCase):
    def test_write_before_attention_rounds_current_and_chunk_once(self):
        from tools.kv.prequantized.recipes import WRITE_FIRST

        config = Qwen3Config(num_hidden_layers=1)
        obs = Observer("k16_v8", {"0.k": 1, "0.v": 1}, cache_recipe=WRITE_FIRST)
        cache = make_cache(config, obs)
        initial = torch.tensor([[[[1.07], [2.13]]]])
        key, value = cache.update(initial, initial, 0)
        self.assertTrue(torch.equal(key, initial))
        self.assertTrue(torch.equal(value, qdq(initial, 1)))
        saved = value.clone()
        chunk = torch.tensor([[[[1.19], [2.29]]]])
        _, value = cache.update(chunk, chunk, 0)
        self.assertTrue(torch.equal(value[..., :2, :], saved))
        self.assertTrue(torch.equal(value[..., 2:, :], qdq(chunk, 1)))
        self.assertEqual(obs.diagnostics["0.v"]["elements"], 4)

    def test_actual_attention_read_proof_on_tiny_model(self):
        from transformers import Qwen3ForCausalLM

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
        observer = Observer()
        cache = make_cache(config, observer)
        with torch.inference_mode(), AttentionReadProof(model, observer):
            model(torch.tensor([[1, 2, 3]]), past_key_values=cache, use_cache=True)
            model(torch.tensor([[4]]), past_key_values=cache, use_cache=True)
        self.assertEqual(observer.attention_reads, {0: 3, 1: 3})

    def test_scalar_reference_at_every_tie(self):
        values = [0, 2**-10, 3 * 2**-10, -500, 500]
        for exponent in range(-6, 9):
            for mantissa in range(8):
                x = (1 + (mantissa + 0.5) / 8) * 2**exponent
                values.extend([x, -x])
        actual = qdq(torch.tensor(values), 1).tolist()
        self.assertEqual(actual, [reference_e4m3(x) for x in values])

    def test_quantized_writes_leave_fresh_prefill_logits_unchanged(self):
        from transformers import Qwen3ForCausalLM

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
        observer = Observer(
            "k8_v8", {f"{layer}.{kind}": 1 for layer in (0, 1) for kind in ("k", "v")}
        )
        cache = make_cache(config, observer)
        ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            baseline = model(ids, use_cache=True)
            with AttentionReadProof(model, observer):
                rounded = model(ids, past_key_values=cache, use_cache=True)
        self.assertTrue(torch.equal(baseline.logits, rounded.logits))
        self.assertTrue(
            torch.equal(
                cache.layers[0].keys, qdq(baseline.past_key_values.layers[0].keys, 1)
            )
        )

    def test_independent_components_and_scale(self):
        x = torch.tensor([[[[1.06, 0.004]]]], dtype=torch.float16)
        observer = Observer("k16_v8", {"0.k": 0.02, "0.v": 0.01})
        self.assertTrue(torch.equal(observer.transform(0, "k", x), x))
        self.assertTrue(torch.equal(observer.transform(0, "v", x), qdq(x, 0.01)))
        self.assertRaises(
            ValueError, Observer("k8_v8", {"0.k": 0}).transform, 0, "k", x
        )

    def test_calibration_scale_is_computed_in_float32(self):
        observer = Observer(calibrate=True)
        value = torch.tensor([[[[3.875]]]], dtype=torch.float16)
        observer.transform(0, "k", value)
        self.assertEqual(
            observer.frozen_scales()["0.k"],
            float((value.float().abs().max() / 448).item()),
        )

    def test_cache_preserves_old_entries_and_fresh_token(self):
        config = Qwen3Config(num_hidden_layers=1)
        observer = Observer("k16_v8", {"0.k": 1, "0.v": 1})
        cache = make_cache(config, observer)
        k = torch.tensor([[[[1.07], [2.13]]]])
        v = k + 0.05
        returned_k, returned_v = cache.update(k, v, 0)
        self.assertTrue(torch.equal(returned_k, k))
        self.assertTrue(torch.equal(returned_v, v))
        saved = cache.layers[0].values.clone()
        new = torch.tensor([[[[1.19]]]])
        _, values = cache.update(new, new, 0)
        self.assertTrue(torch.equal(values[..., :-1, :], saved))
        self.assertTrue(torch.equal(values[..., -1:, :], new))
        self.assertTrue(torch.equal(cache.layers[0].values[..., :-1, :], saved))
        self.assertEqual(observer.counts[0]["written_tokens"], 3)
        self.assertEqual(observer.diagnostics["0.v"]["elements"], 3)

    def test_sliding_positions_and_eviction(self):
        from transformers import GptOssConfig

        config = GptOssConfig(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=128,
        )
        observer = Observer(
            "k8_v8", {f"{layer}.{kind}": 1 for layer in (0, 1) for kind in ("k", "v")}
        )
        cache = make_cache(config, observer)
        for layer in (0, 1):
            initial = torch.arange(129.0).view(1, 1, 129, 1)
            cache.update(initial, initial, layer)
            for step in range(130, 260):
                entry = torch.tensor([[[[float(step)]]]])
                cache.update(entry, entry, layer)
        self.assertEqual(cache.get_seq_length(), 259)
        self.assertEqual(cache.layers[0].keys.shape[-2], 127)
        self.assertEqual(cache.layers[1].keys.shape[-2], 259)
        self.assertEqual(observer.diagnostics["0.k"]["elements"], 259)


class ArtifactTests(unittest.TestCase):
    def test_reject_generic_gemm_as_packed_dispatch(self):
        from tools.kv.prequantized.runner import require_packed_dispatch

        for model in ("Qwen/Qwen3-8B-AWQ", "openai/gpt-oss-20b"):
            self.assertRaises(
                RuntimeError, require_packed_dispatch, model, ["aten::matmul", "gemm"]
            )
        require_packed_dispatch("Qwen/Qwen3-8B-AWQ", ["awq_gemm_kernel"])
        require_packed_dispatch(
            "openai/gpt-oss-20b", ["_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256"]
        )

    def test_result_validation_rejects_foreign_missing_and_failed_rows(self):
        import copy
        from tools.kv.prequantized.validation import row_provenance, validate_results
        from tools.kv.prequantized.contract import ARMS

        sample = {
            "id": "nll-one",
            "kind": "nll",
            "length": 2048,
            "prompt_hash": "p",
            "article_id": "a",
        }
        manifest = {"samples": [sample], "activation_dtype": "float16"}
        scales = {"scale_hash": "s"}
        rows = []
        for arm in ARMS:
            rows.append(
                {
                    "sample_id": "nll-one",
                    "kind": "nll",
                    "context_tokens": 2048,
                    "prompt_hash": "p",
                    "article_id": "a",
                    "status": "completed",
                    "scored_tokens": 128,
                    "mean_nll": 1.0,
                    "token_nll": [1.0] * 128,
                    "provenance": row_provenance(
                        {"source_sha": "a" * 40}, manifest, arm, "s"
                    ),
                    "cache_receipt": {
                        "attention_reads": {"0": 2048},
                        "quantization": {},
                    },
                }
            )
        base = {"source_sha": "a" * 40}
        validate_results(rows, manifest, base, scales)
        for mutate in (
            lambda r: r.pop(),
            lambda r: r.append(r[0]),
            lambda r: r[0].update(status="failed"),
            lambda r: r[0]["provenance"].update(source_sha="b" * 40),
            lambda r: r[0].update(prompt_hash="foreign"),
            lambda r: r[0].update(token_nll=[float("nan")] * 128),
        ):
            bad = copy.deepcopy(rows)
            mutate(bad)
            self.assertRaises(ValueError, validate_results, bad, manifest, base, scales)

    def test_matrix_rejects_unsupported_sampling_and_recipe(self):
        import copy
        import json

        path = (
            Path(__file__).resolve().parents[1]
            / "tools/kv/prequantized/matrix.example.json"
        )
        matrix = json.loads(path.read_text())
        validate_matrix(matrix)
        for section, key, value in (
            ("quality", "bootstrap_resamples", 100),
            ("quantizer", "format", "fp8_e4m3fnuz"),
            ("quantizer", "calibration_decode_tokens", 16),
        ):
            changed = copy.deepcopy(matrix)
            changed[section][key] = value
            self.assertRaises(ValueError, validate_matrix, changed)

    def test_pins_hashes_and_split_overlap(self):
        self.assertRaises(ValueError, pinned, "main")
        manifest = {
            name: "a" * 40
            for name in (
                "model_revision",
                "tokenizer_revision",
                "dataset_revision",
                "source_sha",
            )
        }
        manifest.update(
            samples=[{"id": "one"}],
            sample_hash=digest([{"id": "one"}]),
            calibration_article_ids=["c"],
            evaluation_article_ids=["e"],
        )
        validate_manifest(manifest)
        manifest["evaluation_article_ids"] = ["c"]
        self.assertRaises(ValueError, validate_manifest, manifest)

    def test_append_only_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            provenance = {"source": "a" * 40}
            row = {"sample_id": "one", "provenance": provenance}
            completed = completed_rows(path, provenance)
            append_result(path, row, completed)
            self.assertEqual(completed_rows(path, provenance)["one"], row)
            self.assertRaises(ValueError, append_result, path, row, completed)
            self.assertRaises(ValueError, completed_rows, path, {"source": "b" * 40})
            frozen = Path(directory) / "frozen.json.gz"
            write_once(frozen, row)
            write_once(frozen, row)
            self.assertRaises(ValueError, write_once, frozen, {"different": True})

    def test_document_bootstrap_does_not_collapse_at_power_of_two(self):
        baseline = {str(i): 1.0 for i in range(32)}
        arm = {str(i): 1 + (i - 15.5) / 100 for i in range(32)}
        result = paired_nll(arm, baseline)
        self.assertLess(result["lower_95_one_sided"], result["upper_95_one_sided"])
        self.assertEqual(result, paired_nll(arm, baseline))
        self.assertRaises(ValueError, paired_nll, {"one": 0}, baseline)

    def test_zero_errors_retain_positive_exact_uncertainty(self):
        cases = {str(i): True for i in range(100)}
        result = paired_retrieval(cases, cases)
        self.assertAlmostEqual(result["loss_upper_95_exact"], 1 - 0.05**0.01)
        self.assertTrue(result["supported_margin"])
        low_baseline = {str(i): i < 70 for i in range(100)}
        self.assertFalse(
            paired_retrieval(low_baseline, low_baseline)["supported_margin"]
        )
        improved = {str(i): True for i in range(100)}
        comparison = paired_retrieval(improved, low_baseline)
        self.assertLess(comparison["accuracy_loss_upper_95_one_sided"], 0)
        self.assertLess(comparison["accuracy_loss_lower_95_one_sided"], -0.3)


if __name__ == "__main__":
    unittest.main()
