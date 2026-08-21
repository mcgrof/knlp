# SPDX-License-Identifier: GPL-2.0
"""Capture per-layer K/V/Q for KV-House temporal-structure analysis.

For each text sample this records, per layer:

- post-RoPE K and V exactly as cached, [n_kv, T, hd];
- post-RoPE Q, [n_q, T, hd], optionally strided along T to bound
  disk (queries are only needed as attention probes);
- pre-RoPE (post-bias) K from the k_proj output, [n_kv, T, hd] —
  the information-theoretic diagnostic for whether RoPE destroys
  temporal locality.

Reuses tools/kv/k_bias_common.py runtime attention discovery (no
name-based model assumptions) and its ALL_ATTENTION_FUNCTIONS hook
pattern, which requires attn_implementation="sdpa". Vendor-neutral:
torch.cuda works on both CUDA and ROCm.

Text classes: "wikitext" (coherent prose), "code" (structured text:
this repository's own sources), "random" (uniform random token ids
— the no-structure control).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

import tools.kv.k_bias_common as kbc  # noqa: E402


def code_text(min_chars):
    parts = []
    globs = (
        "kv_house/*.py",
        "routing/prefix_integrity/*.py",
        "backends/*.py",
        "tools/kv/*.py",
        "experiments/kv_house/*.py",
    )
    for pat in globs:
        for p in sorted(Path(REPO).glob(pat)):
            parts.append(p.read_text())
            if sum(len(s) for s in parts) > min_chars:
                return "\n".join(parts)
    return "\n".join(parts)


def build_streams(tok, classes, total, seed, vocab_size):
    """One long token stream per text class. Every sample across
    every context length is a disjoint slice of its class stream
    (global cursor), so no capture file's tokens are a subset of
    another's — the calibration/eval file split stays leak-free
    across context lengths."""
    streams = {}
    for cls in classes:
        if cls == "random":
            gen = torch.Generator().manual_seed(seed)
            streams[cls] = torch.randint(0, vocab_size, (total,), generator=gen)
        elif cls == "code":
            ids = tok(code_text(total * 8), return_tensors="pt").input_ids[0]
            assert ids.numel() >= total, "not enough code tokens"
            streams[cls] = ids
        else:
            chunks = kbc.calib_prompts(tok, n=1, seq_len=total + 1024)
            assert chunks, "not enough wikitext tokens"
            streams[cls] = torch.tensor(chunks[0], dtype=torch.long)
    return streams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--contexts", default="512,1024,2048")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--classes", default="wikitext,code,random")
    ap.add_argument("--q-stride", type=int, default=4)
    ap.add_argument("--layers", default="", help="csv subset, empty = all")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = kbc.load_model(args.model, args.dtype, args.device, False)
    infos = kbc.discover_attention(model)
    info_by_mod = {id(i["attn_module"]): i for i in infos}
    keep_layers = (
        {int(x) for x in args.layers.split(",") if x != ""}
        if args.layers
        else {i["layer_idx"] for i in infos}
    )

    cap = {}

    def mk_kproj_hook(idx, info):
        def hook(mod, inp, out):
            o = out
            if info["fused"]:
                s0, s1 = info["k_slice"]
                o = out[..., s0:s1]
            t = o.shape[1]
            hd = info["head_dim"]
            cap.setdefault(idx, {})["k_pre"] = (
                o.detach().reshape(1, t, -1, hd).transpose(1, 2).to(torch.float16)
            )

        return hook

    handles = []
    for info in infos:
        if info["layer_idx"] not in keep_layers:
            continue
        proj = info["k_proj"] if not info["fused"] else info["qkv_proj"]
        if proj is not None:
            handles.append(
                proj.register_forward_hook(mk_kproj_hook(info["layer_idx"], info))
            )

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    impl = model.config._attn_implementation
    orig = ALL_ATTENTION_FUNCTIONS[impl]

    def attn_hook(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **kw):
        info = info_by_mod.get(id(module))
        if info is not None and info["layer_idx"] in keep_layers:
            idx = info["layer_idx"]
            d = cap.setdefault(idx, {})
            d["k_post"] = k.detach().to(torch.float16)
            d["v"] = v.detach().to(torch.float16)
            d["q_post"] = q.detach()[:, :, :: args.q_stride, :].to(torch.float16)
        return orig(
            module, q, k, v, attention_mask, dropout=dropout, scaling=scaling, **kw
        )

    ALL_ATTENTION_FUNCTIONS[impl] = attn_hook

    env = {
        "model": args.model,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "transformers": __import__("transformers").__version__,
        "q_stride": args.q_stride,
        "seed": args.seed,
        "n_layers": len(infos),
        "layers": sorted(keep_layers),
        "heads": {
            "n_q": infos[0]["n_q_heads"],
            "n_kv": infos[0]["n_kv_heads"],
            "head_dim": infos[0]["head_dim"],
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(env, indent=2))
    print(json.dumps(env, indent=2))

    ctxs = [int(c) for c in args.contexts.split(",")]
    classes = args.classes.split(",")
    total = sum(c * args.num_samples for c in ctxs)
    streams = build_streams(tok, classes, total, args.seed, model.config.vocab_size)
    cursor = {cls: 0 for cls in classes}

    manifest = []
    for ctx in ctxs:
        for cls in classes:
            ids_list = []
            for _ in range(args.num_samples):
                ids_list.append(streams[cls][cursor[cls] : cursor[cls] + ctx])
                cursor[cls] += ctx
            for si, ids in enumerate(ids_list):
                cap.clear()
                t0 = time.time()
                with torch.no_grad():
                    model(ids.unsqueeze(0).to(args.device), use_cache=False)
                rec = {}
                for idx, d in cap.items():
                    rec[idx] = {k: t.squeeze(0).cpu() for k, t in d.items()}
                name = f"{cls}_ctx{ctx}_s{si}.pt"
                torch.save(rec, out_dir / name)
                manifest.append({"file": name, "class": cls, "ctx": ctx, "sample": si})
                print(f"captured {name} in {time.time() - t0:.1f}s")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for h in handles:
        h.remove()
    ALL_ATTENTION_FUNCTIONS[impl] = orig


if __name__ == "__main__":
    main()
