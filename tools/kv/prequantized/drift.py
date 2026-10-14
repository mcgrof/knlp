"""Measure packed-backend replay drift at identical logical positions."""

import contextlib
import importlib
import json
from pathlib import Path

import torch

from tools.kv.prequantized.contract import digest, write_once


def last_token(tensor):
    return tensor.reshape(-1, tensor.shape[-1])[-1].detach().float().cpu()


class Trace:
    def __init__(self, model):
        self.model, self.values, self.handles = model, {}, []

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):

            def residual(module, args, kwargs, i=index):
                value = kwargs.get("hidden_states", args[0] if args else None)
                self.values[f"{i}.residual"] = last_token(value)

            self.handles.append(
                layer.register_forward_pre_hook(residual, with_kwargs=True)
            )
            for kind in ("q", "k", "v"):

                def projection(module, args, output, i=index, kind=kind):
                    self.values[f"{i}.{kind}"] = last_token(output)

                self.handles.append(
                    getattr(layer.self_attn, f"{kind}_proj").register_forward_hook(
                        projection
                    )
                )
            if hasattr(layer.mlp, "router"):

                def route(module, args, output, i=index):
                    for n, tensor in enumerate(
                        output if isinstance(output, tuple) else (output,)
                    ):
                        self.values[f"{i}.router.{n}"] = last_token(tensor)

                self.handles.append(layer.mlp.router.register_forward_hook(route))
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        backend = self.model.config._attn_implementation
        if backend == "eager":
            self.target = importlib.import_module(
                type(self.model.model.layers[0].self_attn).__module__
            )
            self.original = self.target.eager_attention_forward
        else:
            self.target = ALL_ATTENTION_FUNCTIONS
            self.original = self.target[backend]

        def checked(module, query, key, value, *args, **kwargs):
            output = self.original(module, query, key, value, *args, **kwargs)
            if module.layer_idx == 0:
                self.values["attention.q"] = query[0, 0, -1].detach().float().cpu()
                self.values["attention.k"] = key[0, 0].detach().float().cpu()
                self.values["attention.v"] = value[0, 0].detach().float().cpu()
                self.values["attention.output"] = (
                    output[0][0, -1, 0].detach().float().cpu()
                )
                mask = args[0] if args else kwargs.get("attention_mask")
                if mask is not None:
                    self.values["attention.mask"] = (
                        mask[0, 0, -1, : key.shape[-2]].detach().float().cpu()
                    )
                self.scale = module.scaling
                self.sink = float(module.sinks[0]) if hasattr(module, "sinks") else None
                self.window = getattr(module, "sliding_window", None)
            return output

        if backend == "eager":
            self.target.eager_attention_forward = checked
        else:
            self.target[backend] = checked
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        if self.model.config._attn_implementation == "eager":
            self.target.eager_attention_forward = self.original
        else:
            self.target[self.model.config._attn_implementation] = self.original

    def attention_reference(self, absolute_position):
        q, k, v = (self.values[f"attention.{kind}"] for kind in ("q", "k", "v"))
        positions = torch.arange(absolute_position + 1 - len(k), absolute_position + 1)
        allowed = positions <= absolute_position
        if self.window:
            allowed &= positions > absolute_position - self.window
        score = k @ q * self.scale
        score[~allowed] = -torch.inf
        if self.sink is not None:
            score = torch.cat([score, torch.tensor([self.sink])])
        probability = score.softmax(-1)[: len(k)]
        reference = probability @ v
        actual = self.values["attention.output"]
        mask = self.values.get("attention.mask")
        return {
            "layer": 0,
            "query_head": 0,
            "kv_head": 0,
            "gqa_mapping": "query_head // num_key_value_groups",
            "qkv": "captured after RoPE and Cache.update",
            "absolute_query_position": absolute_position,
            "key_positions": [int(positions[0]), int(positions[-1])],
            "window": self.window,
            "sink_logit": self.sink,
            "explicit_mask_matches_backend": (
                None if mask is None else bool(torch.equal(mask > -1e10, allowed))
            ),
            "fp32_reference_max_error": float((reference - actual).abs().max()),
            "fp32_reference_mean_error": float((reference - actual).abs().mean()),
        }


def compare(a, b, target, runner):
    aa, bb = a.float().log_softmax(-1), b.float().log_softmax(-1)
    return {
        **runner.error(a, b),
        "target_token": target,
        "target_nll_a": -float(aa[target]),
        "target_nll_b": -float(bb[target]),
        "target_delta_nll": float(bb[target] - aa[target]),
        "top1_changed": int(a.argmax()) != int(b.argmax()),
        "kl_a_to_b": float((aa.exp() * (aa - bb)).sum()),
        "total_variation": float((aa.exp() - bb.exp()).abs().sum() / 2),
    }


def trace_comparison(a, b):
    rows = []
    for name in a.values:
        if name.startswith("attention."):
            continue
        x, y = a.values[name], b.values[name]
        rows.append(
            {
                "name": name,
                "max_error": float((x - y).abs().max()),
                "mean_error": float((x - y).abs().mean()),
                "changed_elements": int((x != y).sum()),
            }
        )
    return {
        "first_divergence": next((r for r in rows if r["changed_elements"]), None),
        "rows": rows,
    }


@torch.inference_mode()
def diagnose(args, manifest, runner):
    from transformers.cache_utils import DynamicCache

    model, _ = runner.load(manifest)
    prov = runner.provenance(manifest)
    output = Path(args.output)
    path = output / "drift-rows.jsonl"
    if path.exists():
        raise ValueError("refusing to overwrite a diagnostic")
    chosen = []
    for length in (2048, 8192):
        chosen.extend(
            [
                s
                for s in manifest["samples"]
                if s["kind"] == "nll" and s["length"] == length and not s.get("reserve")
            ][:8]
        )
    if manifest["model"].startswith("openai/"):
        example = chosen[0]
        for length in (127, 128, 129, 255, 256, 257):
            chosen.append(
                {
                    **example,
                    "id": f"boundary-{length}",
                    "length": length,
                    "diagnostic_steps": 3,
                }
            )
    rows = []
    for sample in chosen:
        length, steps, ids = (
            sample["length"],
            sample.get("diagnostic_steps", 16),
            sample["token_ids"],
        )
        single = DynamicCache(config=model.config)
        runner.forward(model, ids[:length], single)
        chunks = DynamicCache(config=model.config)
        runner.forward(model, ids[:length], chunks)
        chunk_logits = []
        for start in range(0, steps, 4):
            count = min(4, steps - start)
            result = model(
                input_ids=torch.tensor(
                    [ids[length + start : length + start + count]], device="cuda"
                ),
                past_key_values=chunks,
                use_cache=True,
                logits_to_keep=0,
            ).logits
            if not torch.isfinite(result).all():
                raise RuntimeError("nonfinite chunked diagnostic logits")
            chunk_logits.extend(result[0].detach().float().cpu().unbind(0))
        for step in range(steps):
            tracing = step == 0 and (
                sample in (chosen[0], chosen[8]) or sample["id"] == "boundary-127"
            )
            with Trace(model) if tracing else contextlib.nullcontext() as cached_trace:
                a = (
                    runner.forward(model, [ids[length + step]], single)
                    .logits[0, -1]
                    .detach()
                    .float()
                    .cpu()
                )
            with Trace(model) if tracing else contextlib.nullcontext() as full_trace:
                b = (
                    runner.forward(model, ids[: length + step + 1])
                    .logits[0, -1]
                    .detach()
                    .float()
                    .cpu()
                )
            target = ids[length + step + 1]
            row = {
                "sample_id": sample["id"],
                "prefix_tokens": length,
                "append": step + 1,
                "absolute_position": length + step,
                "provenance": prov,
                "cached_minus_full": compare(a, b, target, runner),
                "chunked_minus_full": compare(chunk_logits[step], b, target, runner),
                "cached_minus_chunked": compare(a, chunk_logits[step], target, runner),
            }
            if tracing:
                row["layer_comparison"] = trace_comparison(cached_trace, full_trace)
                row["attention_reference"] = {
                    "cached": cached_trace.attention_reference(length + step),
                    "full": full_trace.attention_reference(length + step),
                }
                name = f"drift-trace-{sample['id']}.pt"
                with (output / name).open("xb") as stream:
                    torch.save(
                        {
                            "cached": cached_trace.values,
                            "full": full_trace.values,
                            "provenance": prov,
                        },
                        stream,
                    )
                row["trace_file"] = name
            with path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
        print(json.dumps({"diagnostic": sample["id"], "steps": steps}), flush=True)
    write_once(
        output / "drift.json",
        {
            "provenance": prov,
            "row_count": len(rows),
            "rows_hash": digest(rows),
            "status": "measured",
            "diagnostic_inputs_reused": True,
            "interpretation": "layer and attention receipts require inspection; no automatic tolerance change",
        },
    )
