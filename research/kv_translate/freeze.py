# SPDX-License-Identifier: GPL-2.0
"""One mapper, frozen, with everything needed to replay it exactly.

Every result in this lane so far refitted the map at whatever context it was
about to measure. That is fine for a latency curve, where the timing does not
depend on the fitted values, and wrong for a quality curve, where it means the
"context generalisation" observed belongs to a *family* of mappers rather than
to any artifact that could be deployed. A deployed translator is one set of
weights applied at every length.

So the map is fitted once, at one length, on a pinned set of calibration
documents, and locked before any held-out result is read. The lock is the point:
choosing the fitting length after seeing which length evaluates best would make
the comparison meaningless, and nothing but discipline prevents that, so the
artifact carries the evidence of when it was written.

What travels with the weights is what a later reader needs to disbelieve the
result: resolved model revisions rather than names, tokenizer and chat-template
hashes rather than an assumption of compatibility, the document identifiers of
every split, hashes of the weights themselves, the support each block selected,
and every dtype in the chain from solve to storage to multiply. An earlier
archived run in this lane left its revision fields empty and could not be
replayed from its own bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

import torch


def tensor_hash(t: torch.Tensor) -> str:
    x = t.detach().to("cpu").contiguous()
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()[:32]


def model_identity(model_id: str, model, tok) -> dict:
    """Identity a reader can check, not a name they have to trust."""
    try:
        from huggingface_hub import snapshot_download

        rev = os.path.basename(
            snapshot_download(model_id, local_files_only=True).rstrip("/")
        )
    except Exception:  # noqa: BLE001
        rev = "unresolved"
    vocab = tok.get_vocab()
    tmpl = getattr(tok, "chat_template", None) or ""
    cfg = model.config
    return {
        "model_id": model_id,
        "revision": rev,
        "tokenizer_sha256": hashlib.sha256(
            json.dumps(sorted(vocab.items()), separators=(",", ":")).encode()
        ).hexdigest()[:32],
        "tokenizer_len": len(vocab),
        "chat_template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()[:32],
        "n_layers": int(cfg.num_hidden_layers),
        "n_kv_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        "head_dim": int(
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        ),
        "rope_theta": float(getattr(cfg, "rope_theta", 0.0) or 0.0),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "model_dtype": str(next(model.parameters()).dtype),
    }


def save(
    path: str,
    mapper_k,
    mapper_v,
    *,
    source: dict,
    target: dict,
    fit_context: int,
    calib_doc_ids,
    dev_doc_ids,
    eval_doc_ids,
    config: dict,
    cache_layout: str = "B,n_kv_heads,T,head_dim",
    key_frame: str = "content (de-rotated); re-rotated at the target position on apply",
) -> dict:
    """Write the artifact and return its manifest.

    Weights are stored at the precision they were solved in. The serving cast
    is a property of a deployment, not of the artifact, so it is recorded as a
    separate decision rather than baked in here.
    """
    blocks = {}
    for kind, mp in (("k", mapper_k), ("v", mapper_v)):
        for (li, h), m in mp.maps.items():
            blocks[f"{kind}.{li}.{h}"] = {
                "M": m.M.detach().cpu(),
                "b": m.b.detach().cpu(),
                "layers": list(m.layers),
                "head_local": bool(m.head_local),
                "ridge": float(m.ridge),
                "n_calib_tokens": int(m.n_calib_tokens),
            }
    weight_hashes = {
        k: {"M": tensor_hash(v["M"]), "b": tensor_hash(v["b"])}
        for k, v in blocks.items()
    }
    joint = hashlib.sha256(
        json.dumps(
            {k: v for k, v in sorted(weight_hashes.items())}, separators=(",", ":")
        ).encode()
    ).hexdigest()
    manifest = {
        "artifact_version": 1,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fit_context": int(fit_context),
        "source": source,
        "target": target,
        "calib_doc_ids": list(calib_doc_ids),
        "dev_doc_ids": list(dev_doc_ids),
        "eval_doc_ids": list(eval_doc_ids),
        "n_blocks": len(blocks),
        "weight_hashes": weight_hashes,
        "joint_weight_sha256": joint,
        "dtypes": {
            "fit": "float64",
            "storage": str(next(iter(blocks.values()))["M"].dtype),
            "multiply": "set by the deployment, not by this artifact",
            "accumulation": "torch default for the multiply dtype",
        },
        "cache_layout": cache_layout,
        "key_frame": key_frame,
        "config": config,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"blocks": blocks, "manifest": manifest}, path)
    manifest["serialized_bytes"] = os.path.getsize(path)
    with open(path + ".manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True, default=str)
    return manifest


def load(path: str, layout, geom, device="cpu", verify: bool = True):
    """Rebuild the two mappers, checking the weights are the ones recorded.

    The hash check is not ceremony. A frozen artifact exists so that later
    results can be attributed to one set of weights, and an artifact that has
    silently changed defeats the only thing it is for.
    """
    from research.kv_translate.fit import AffineMap
    from research.kv_translate.run_a1 import Mapper

    blob = torch.load(path, map_location=device, weights_only=False)
    blocks, manifest = blob["blocks"], blob["manifest"]
    # measured from the file rather than stored in it, since it is the size of
    # the thing that contains it
    manifest["serialized_bytes"] = os.path.getsize(path)
    if verify:
        for name, rec in blocks.items():
            want = manifest["weight_hashes"][name]
            if tensor_hash(rec["M"]) != want["M"] or tensor_hash(rec["b"]) != want["b"]:
                raise RuntimeError(
                    f"frozen artifact {path} block {name} does not match its hash"
                )
    maps = {"k": {}, "v": {}}
    for name, rec in blocks.items():
        kind, li, h = name.split(".")
        li, h = int(li), int(h)
        maps[kind][(li, h)] = AffineMap(
            M=rec["M"].to(device),
            b=rec["b"].to(device),
            layers=tuple(rec["layers"]),
            head=h,
            kind=kind,
            target_layer=li,
            ridge=rec["ridge"],
            head_local=rec["head_local"],
            n_calib_tokens=rec["n_calib_tokens"],
        )
    return (
        Mapper(maps["k"], layout, geom, "k"),
        Mapper(maps["v"], layout, geom, "v"),
        manifest,
    )
