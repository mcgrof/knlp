# SPDX-License-Identifier: MIT
"""Run one immutable stage of the seven-arm physical cache reference."""
import argparse
import gc
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from tools.kv.native.gsm8k import number
from tools.kv.prequantized.fixtures import parse_final
from .cache import ARMS, inventory, make_cache
from .calibration import Observer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sync():
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--stage", choices=["calibration", "development", "confirmation"], required=True
    )
    p.add_argument("--scales", type=Path)
    p.add_argument("--development", type=Path)
    p.add_argument("--batch", type=int, default=8)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    data = json.loads(gzip.decompress(a.inputs.read_bytes()))
    source = json.loads(Path("SOURCE.json").read_text())
    for rel, h in source["files"].items():
        if sha(rel) != h:
            raise ValueError(f"Runtime source changed: {rel}")
    if (
        torch.__version__.split("+")[0] != "2.11.0"
        or transformers.__version__ != "5.10.4"
    ):
        raise ValueError("Unfrozen Torch/Transformers version")
    torch.set_num_threads(min(16, max(1, int(os.environ.get("OMP_NUM_THREADS", "8")))))
    torch.manual_seed(20261020)
    # No attention monkeypatches, Q casts, or model arithmetic changes.
    dtype = getattr(torch, data["dtype"])
    tokenizer = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    sync()
    attention_file = inspect.getsourcefile(type(model.model.layers[0].self_attn))
    attention_source = Path(attention_file).read_text()
    (a.out / "model-attention-source.py.txt").write_text(attention_source)
    weight_files = {
        str(p.relative_to(a.model_path)): sha(p)
        for p in a.model_path.rglob("*.safetensors")
    }
    if not weight_files:
        raise ValueError("Missing pinned weight files")
    official = json.loads(a.weights.read_text())
    if official["model"] != data["model"] or official["revision"] != data["revision"]:
        raise ValueError("Official checkpoint manifest identity mismatch")
    if weight_files != {k: v["sha256"] for k, v in official["files"].items()}:
        raise ValueError("Checkpoint bytes differ from official pinned weights")

    meta = dict(
        at=time.time(),
        model=data["model"],
        revision=data["revision"],
        dtype=data["dtype"],
        input_sha256=sha(a.inputs),
        source=source,
        torch=torch.__version__,
        transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(),
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        model_class=type(model).__name__,
        parameter_dtypes=sorted({str(x.dtype) for x in model.parameters()}),
        parameter_bytes=sum(x.numel() * x.element_size() for x in model.parameters()),
        model_load_seconds=time.monotonic() - start,
        weight_sha256=weight_files,
        attention_source_sha256=hashlib.sha256(attention_source.encode()).hexdigest(),
        generation_config=model.generation_config.to_dict(),
        config=model.config.to_dict(),
        nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True),
    )
    if meta["parameter_dtypes"] != ["torch." + data["dtype"]]:
        raise ValueError("Non-native weights")
    save(a.out / "START.json", meta)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    fixed = None
    if a.scales:
        sc = json.loads(a.scales.read_text())
        if (
            sc["inputs_sha256"] != sha(a.inputs)
            or sc["model"] != data["model"]
            or sc["revision"] != data["revision"]
        ):
            raise ValueError("Calibration identity mismatch")
        if sc["source"] != source:
            raise ValueError("Calibration source changed")
        fixed = sc["scales"]
        save(
            a.out / "SCALES-IDENTITY.json",
            dict(sha256=sha(a.scales), source=sc["source"]),
        )
    if a.stage != "calibration" and fixed is None:
        raise ValueError("Frozen calibration required")
    if a.stage == "confirmation":
        if a.development is None:
            raise ValueError("Development receipts required")
        done = json.loads((a.development / "DONE.json").read_text())
        dev = json.loads((a.development / "START.json").read_text())
        if not done["native_parity_pass"] or done["native_gsm_correct"] < 32:
            raise ValueError("Development/native floor failed")
        if (
            dev["source"] != source
            or dev["input_sha256"] != sha(a.inputs)
            or dev["args"]["batch"] != a.batch
        ):
            raise ValueError("Development identity or batch mismatch")
        if json.loads((a.development / "SCALES-IDENTITY.json").read_text())[
            "sha256"
        ] != sha(a.scales):
            raise ValueError("Development scales changed")

    def cache(arm="N", library=False, observer=None):
        return make_cache(model.config, arm, fixed, observer, library)

    def positions(ids, start):
        return {
            "position_ids": torch.arange(
                start, start + ids.shape[-1], device=ids.device
            )[None, :].expand(ids.shape[0], -1)
        }

    def generate(rows, arm, cap, library=False):
        width = max(len(r["token_ids"]) for r in rows)
        ids = torch.full(
            (len(rows), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda"
        )
        mask = torch.zeros_like(ids)
        for j, r in enumerate(rows):
            ids[j, -len(r["token_ids"]) :] = torch.tensor(r["token_ids"], device="cuda")
            mask[j, -len(r["token_ids"]) :] = 1
        c = cache(arm, library)
        torch.cuda.reset_peak_memory_stats()
        sync()
        t = time.monotonic()
        result = model.generate(
            input_ids=ids,
            attention_mask=mask,
            past_key_values=c,
            do_sample=False,
            max_new_tokens=cap,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            use_cache=True,
        )
        sync()
        elapsed = time.monotonic() - t
        audit = inventory(c) | dict(
            seconds=elapsed,
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        outputs = []
        for row, tokens in zip(rows, result.sequences[:, width:].cpu().tolist()):
            stop = next((i for i, x in enumerate(tokens) if x in eos), len(tokens))
            ended = stop < len(tokens)
            tokens = tokens[:stop]
            outputs.append(
                dict(
                    id=row["id"],
                    tokens=tokens,
                    text=tokenizer.decode(tokens, skip_special_tokens=True),
                    output_tokens=len(tokens),
                    eos=ended,
                    truncated=not ended and len(tokens) == cap,
                    error=None,
                )
            )
        del result, c, ids, mask
        gc.collect()
        return outputs, audit

    def parity():
        # Exact logits for matched prefill and fresh multi-token continuation,
        # plus greedy IDs on the actual checkpoint against the native library.
        rows = [r for r in data["samples"]["development"] if r["kind"] == "gsm8k"][:4]
        lib, la = generate(rows, "N", 64, True)
        ref, ra = generate(rows, "N", 64)
        passed = [r["tokens"] for r in lib] == [r["tokens"] for r in ref]
        logit_checks = []
        for length in [1024, 8192]:
            row = next(
                r
                for r in data["samples"]["development"]
                if r["kind"] == "nll" and r["length"] == length
            )
            ids = torch.tensor([row["token_ids"]], device="cuda")
            c = cache()
            d = cache(library=True)
            for offset, count in [(0, length), (length, 3), (length + 3, 1)]:
                x = ids[:, offset : offset + count]
                l = model(
                    input_ids=x,
                    past_key_values=c,
                    use_cache=True,
                    logits_to_keep=1,
                    **positions(x, offset),
                ).logits
                r = model(
                    input_ids=x,
                    past_key_values=d,
                    use_cache=True,
                    logits_to_keep=1,
                    **positions(x, offset),
                ).logits
                same = torch.equal(l, r)
                passed = passed and same
                logit_checks.append(
                    dict(
                        length=length,
                        offset=offset,
                        count=count,
                        exact=same,
                        max_abs=float((l - r).abs().max()),
                    )
                )
            del c, d, ids, l, r
        receipt = dict(
            pass_native=passed,
            generated_library=lib,
            generated_reference=ref,
            library_cache=la,
            reference_cache=ra,
            logits=logit_checks,
        )
        save(a.out / "NATIVE-PARITY.json", receipt)
        if not passed:
            raise RuntimeError("Exact native-cache parity failed; stop model")
        return passed

    def calibrate():
        parity()
        collected = Observer(model.config.num_hidden_layers, capture=True)

        def run(observer, label):
            for row in data["samples"]["calibration"]:
                observer.begin(row)
                c = cache(observer=observer)
                length = row["length"]
                x = torch.tensor([row["token_ids"][:length]], device="cuda")
                sync()
                t = time.monotonic()
                logits = model(
                    input_ids=x,
                    past_key_values=c,
                    use_cache=True,
                    logits_to_keep=1,
                    **positions(x, 0),
                ).logits
                for step in range(128):
                    x = logits[:, -1, :].argmax(-1, keepdim=True)
                    logits = model(
                        input_ids=x,
                        past_key_values=c,
                        use_cache=True,
                        logits_to_keep=1,
                        **positions(x, length + step),
                    ).logits
                sync()
                append(
                    a.out / (label + "-documents.jsonl"),
                    dict(
                        id=row["id"],
                        length=length,
                        seconds=time.monotonic() - t,
                        consumed=c.get_seq_length(),
                    ),
                )
                del c, x, logits
                print("calibration", label, row["id"], flush=True)

        run(collected, "collect")
        scales = collected.scales()
        receipt = dict(
            model=data["model"],
            revision=data["revision"],
            inputs_sha256=sha(a.inputs),
            source=source,
            scales=scales,
            convention="FP32 clamp_min(native maxabs across all calibration prefill and 128 decode operands,1e-12)/448",
            rounding="E4M3FN nearest ties to even, clip [-448,448], decode FP32 multiply then native cast",
            observer_site="Native model cache update, after native key normalization/RoPE",
            calibration_document_ids=data["document_ids"]["calibration"],
        )
        save(a.out / "scales.json", receipt)
        torch.save(collected.fixtures, a.out / "native-activation-fixtures.pt")
        audit = Observer(model.config.num_hidden_layers, fixed=scales)
        run(audit, "audit")
        save(
            a.out / "QUANTIZATION-AUDIT.json",
            dict(
                scales_sha256=sha(a.out / "scales.json"),
                metrics=audit.summary(),
                pass2_maxima_equal=audit.scales() == scales,
                full_calibration_operands=True,
            ),
        )
        if audit.scales() != scales:
            raise ValueError("Native calibration replay changed")
        save(
            a.out / "DONE.json",
            dict(at=time.time(), stage=a.stage, native_parity_pass=True),
        )

    with torch.inference_mode():
        if a.stage == "calibration":
            calibrate()
            return
        native_parity = parity()
        samples = data["samples"][a.stage]
        native_correct = 0
        # Frozen order reverses all compressed arms on the second checkpoint.
        arms = ARMS if data["dtype"] == "float16" else ("N", *reversed(ARMS[1:]))
        for arm in arms:
            target = a.out / arm
            target.mkdir()
            for row in (r for r in samples if r["kind"] == "nll"):
                length = row["length"]
                c = cache(arm)
                ids = torch.tensor([row["token_ids"]], device="cuda")
                torch.cuda.reset_peak_memory_stats()
                sync()
                t = time.monotonic()
                model(
                    input_ids=ids[:, :length],
                    past_key_values=c,
                    use_cache=True,
                    logits_to_keep=1,
                    **positions(ids[:, :length], 0),
                )
                logits = model(
                    input_ids=ids[:, length : length + 128],
                    past_key_values=c,
                    use_cache=True,
                    logits_to_keep=128,
                    **positions(ids[:, length : length + 128], length),
                ).logits.float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Nonfinite NLL")
                nll = torch.nn.functional.cross_entropy(
                    logits[0], ids[0, length + 1 : length + 129], reduction="none"
                )
                sync()
                audit = inventory(c) | dict(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
                append(
                    target / "nll.jsonl",
                    dict(
                        id=row["id"],
                        article_id=row["article_id"],
                        length=length,
                        nll=nll.cpu().tolist(),
                        mean_nll=float(nll.mean()),
                        seconds=time.monotonic() - t,
                        cache=audit,
                    ),
                )
                del c, ids, logits, nll
                print(a.stage, arm, "nll", row["id"], flush=True)
            for kind in ["retrieval", "gsm8k"]:
                groups = [
                    (
                        length,
                        [
                            r
                            for r in samples
                            if r["kind"] == kind
                            and (kind == "gsm8k" or r["length"] == length)
                        ],
                    )
                    for length in ([1024, 8192] if kind == "retrieval" else [None])
                ]
                for length, group in groups:
                    batch = min(a.batch, 2) if length == 8192 else a.batch
                    for offset in range(0, len(group), batch):
                        part = group[offset : offset + batch]
                        outs, audit = generate(
                            part, arm, 2048 if kind == "gsm8k" else 128
                        )
                        for row, r in zip(part, outs):
                            if kind == "gsm8k":
                                matches = re.findall(data["gsm_regex"], r["text"])
                                answer = number(matches[-1]) if matches else None
                                gold = row["gold"]
                                correct = answer == gold
                            else:
                                answer, _ = parse_final(
                                    tokenizer, r["tokens"], data["model"]
                                )
                                gold = row["answer"]
                                correct = answer.casefold() == gold.casefold()
                            r.update(
                                gold=gold,
                                answer=answer,
                                correct=correct,
                                length=length,
                                parser_failure=answer is None or answer == "",
                            )
                            if arm == "N" and kind == "gsm8k":
                                native_correct += int(correct)
                            append(target / (kind + ".jsonl"), r)
                        append(
                            target / "batches.jsonl",
                            dict(
                                kind=kind,
                                length=length,
                                offset=offset,
                                count=len(part),
                                cache=audit,
                            ),
                        )
                        print(
                            a.stage, arm, kind, length, offset + len(part), flush=True
                        )
        save(
            a.out / "DONE.json",
            dict(
                at=time.time(),
                stage=a.stage,
                native_parity_pass=native_parity,
                native_gsm_correct=native_correct,
            ),
        )


if __name__ == "__main__":
    main()
