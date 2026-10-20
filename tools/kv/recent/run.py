"""Run a frozen physical reference-cache quality comparison on one GPU."""

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoTokenizer
from tools.kv.native.gsm8k import number
from tools.kv.prequantized.fixtures import parse_final
from tools.kv.recent.cache import MODES, inventory, make_cache, native_library_cache


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def flush_row(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sync():
    torch.cuda.synchronize()


def nll_position_kwargs(model_name, input_ids, start):
    """Give independent text requests positions unrelated to prior batches.

    The Qwen multimodal wrapper retains generation's rope_deltas on the
    model object. A new KV cache alone does not clear those offsets before
    a cached direct forward. Explicit text positions bypass that stale
    request state without modifying the official model implementation.
    """
    if not model_name.startswith("Qwen/"):
        return {}
    positions = torch.arange(
        start, start + input_ids.shape[-1], device=input_ids.device
    )
    return {"position_ids": positions.unsqueeze(0).expand(input_ids.shape[0], -1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--stage", choices=("development", "quality"), required=True)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(gzip.decompress(args.inputs.read_bytes()))
    source = json.loads(Path("SOURCE.json").read_text())
    for name, expected in source["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Runtime source changed: {name}")
    torch.set_num_threads(min(16, max(1, int(os.environ.get("OMP_NUM_THREADS", "8")))))
    torch.manual_seed(20261019)
    save(
        args.out / "START.json",
        {
            "at": time.time(),
            "model": data["model"],
            "revision": data["revision"],
            "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
            "args": vars(args) | {"inputs": str(args.inputs), "out": str(args.out)},
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
            "environment": subprocess.check_output(["nvidia-smi"], text=True),
            "source": source,
        },
    )
    path = args.model_path or snapshot_download(
        data["model"],
        revision=data["revision"],
        ignore_patterns=["*.msgpack", "*.h5", "*.bin", "*.gguf"],
    )
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    load_start = time.monotonic()
    # Keep the official model implementation and weight mapping. All arms have
    # identical BF16 weights and SDPA attention; only stored attention KV changes.
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    sync()
    save(
        args.out / "MODEL.json",
        {
            "load_seconds": time.monotonic() - load_start,
            "class": type(model).__name__,
            "parameter_dtypes": sorted({str(x.dtype) for x in model.parameters()}),
            "parameter_bytes": sum(
                x.numel() * x.element_size() for x in model.parameters()
            ),
            "generation_config": model.generation_config.to_dict(),
            "config": model.config.to_dict(),
        },
    )
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])

    def generate(rows, mode, max_tokens, library=False):
        width = max(len(r["token_ids"]) for r in rows)
        ids = torch.full(
            (len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda"
        )
        mask = torch.zeros_like(ids)
        for i, r in enumerate(rows):
            ids[i, -len(r["token_ids"]) :] = torch.tensor(r["token_ids"], device="cuda")
            mask[i, -len(r["token_ids"]) :] = 1
        cache = (
            native_library_cache(model.config)
            if library
            else make_cache(model.config, mode)
        )
        torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.monotonic()
        result = model.generate(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=cache,
            do_sample=False,
            max_new_tokens=max_tokens,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        sync()
        elapsed = time.monotonic() - start
        audit = inventory(cache)
        audit.update(
            seconds=elapsed,
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        outputs = []
        for r, tokens in zip(rows, result.sequences[:, width:].cpu().tolist()):
            stop = next((i for i, t in enumerate(tokens) if t in eos), len(tokens))
            ended = stop < len(tokens)
            tokens = tokens[:stop]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            outputs.append(
                {
                    "id": r["id"],
                    "tokens": tokens,
                    "text": text,
                    "output_tokens": len(tokens),
                    "eos": ended,
                    "truncated": not ended and len(tokens) == max_tokens,
                }
            )
        del result, cache, ids, mask
        gc.collect()
        return outputs, audit

    with torch.inference_mode():
        if args.stage == "development":
            # Independent native cache parity precedes observing candidate task scores.
            rows = data["gsm8k"]["development"][:4]
            library, la = generate(rows, "native", 64, True)
            native, na = generate(rows, "native", 64)
            same = [r["tokens"] for r in library] == [r["tokens"] for r in native]
            save(
                args.out / "NATIVE-PARITY.json",
                {
                    "pass": same,
                    "library": library,
                    "reference": native,
                    "library_cache": la,
                    "reference_cache": na,
                },
            )
            if not same:
                raise RuntimeError("Native cache parity failed")
            for mode in MODES:
                outputs, audit = generate(data["gsm8k"]["development"], mode, 2048)
                for r in outputs:
                    gold = next(
                        x["gold"]
                        for x in data["gsm8k"]["development"]
                        if x["id"] == r["id"]
                    )
                    values = re.findall(data["gsm8k"]["primary_regex"], r["text"])
                    r.update(gold=gold, answer=number(values[-1]) if values else None)
                    r["correct"] = r["answer"] == gold
                save(
                    args.out / (mode + "-development.json"),
                    {"outputs": outputs, "cache": audit},
                )
            save(args.out / "DONE.json", {"at": time.time(), "stage": args.stage})
            return

        # Fixed model-specific order reverses the two quantized arms between models.
        modes = (
            MODES if data["model"].startswith("Qwen/") else ("native", "k8v8", "k16v8")
        )
        for mode in modes:
            target = args.out / mode
            target.mkdir()
            for length in (1024, 8192):
                samples = [
                    r
                    for r in data["samples"]
                    if r["kind"] == "nll" and r["length"] == length
                ]
                for row in samples:
                    cache = make_cache(model.config, mode)
                    ids = torch.tensor([row["token_ids"]], device="cuda")
                    sync()
                    start = time.monotonic()
                    model(
                        input_ids=ids[:, :length],
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                        **nll_position_kwargs(data["model"], ids[:, :length], 0),
                    )
                    logits = model(
                        input_ids=ids[:, length : length + 128],
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=128,
                        **nll_position_kwargs(
                            data["model"], ids[:, length : length + 128], length
                        ),
                    ).logits.float()
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Nonfinite raw NLL logits")
                    nll = torch.nn.functional.cross_entropy(
                        logits[0], ids[0, length + 1 : length + 129], reduction="none"
                    )
                    sync()
                    flush_row(
                        target / "nll.jsonl",
                        {
                            "id": row["id"],
                            "length": length,
                            "article_id": row["article_id"],
                            "nll": nll.cpu().tolist(),
                            "mean_nll": nll.mean().item(),
                            "seconds": time.monotonic() - start,
                            "cache": inventory(cache),
                        },
                    )
                    del cache, ids, logits, nll
                retrieval = [
                    r
                    for r in data["samples"]
                    if r["kind"] == "retrieval" and r["length"] == length
                ]
                retrieval_batch = min(args.batch, 2) if length >= 8192 else args.batch
                for index in range(0, len(retrieval), retrieval_batch):
                    batch = retrieval[index : index + retrieval_batch]
                    outputs, audit = generate(batch, mode, 128)
                    for row, result in zip(batch, outputs):
                        answer, _ = parse_final(
                            tokenizer, result["tokens"], data["model"]
                        )
                        result.update(
                            length=length,
                            gold=row["answer"],
                            answer=answer,
                            correct=answer.casefold() == row["answer"].casefold(),
                        )
                        flush_row(target / "retrieval.jsonl", result)
                    flush_row(
                        target / "batches.jsonl",
                        {
                            "kind": "retrieval",
                            "length": length,
                            "index": index,
                            "cache": audit,
                        },
                    )
                    print(mode, "retrieval", length, index + len(batch), flush=True)
            samples = data["gsm8k"]["samples"]
            for index in range(0, len(samples), args.batch):
                batch = samples[index : index + args.batch]
                outputs, audit = generate(batch, mode, 2048)
                for row, result in zip(batch, outputs):
                    values = re.findall(data["gsm8k"]["primary_regex"], result["text"])
                    answer = number(values[-1]) if values else None
                    result.update(
                        gold=row["gold"], answer=answer, correct=answer == row["gold"]
                    )
                    flush_row(target / "gsm8k.jsonl", result)
                flush_row(
                    target / "batches.jsonl",
                    {"kind": "gsm8k", "index": index, "cache": audit},
                )
                print(mode, "gsm8k", index + len(batch), flush=True)
        save(args.out / "DONE.json", {"at": time.time(), "stage": args.stage})


if __name__ == "__main__":
    main()
