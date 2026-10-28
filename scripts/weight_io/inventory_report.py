#!/usr/bin/env python3
"""Render Markdown tables from Stage A inventory artifacts.

Inputs are the outputs of hf_inventory.py (classes.json, selection.jsonl,
inventory.receipts.json, compact pins) and projections.py. The report
separates header-confirmed byte layout (what the shard headers say) from
config-derived or calculated values, and never reads weight payloads.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def gib(n):
    return f"{n / 2**30:.3f}"


def mib(n):
    return f"{n / 2**20:.3f}"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage-a",
        required=True,
        help="directory with classes.json, selection.jsonl, projections.json and pins/ (or pins_compact/)",
    )
    ap.add_argument(
        "--fetch-manifest",
        help="fetch_manifest.jsonl from hf_inventory.py fetch, if available",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.stage_a)
    classes = json.loads((d / "classes.json").read_text())
    sel = load_jsonl(d / "selection.jsonl")
    proj = json.loads((d / "projections.json").read_text())
    receipts = json.loads((d / "inventory.receipts.json").read_text())
    pins = {}
    pin_dir = d / "pins_compact" if (d / "pins_compact").is_dir() else d / "pins"
    for p in sorted(pin_dir.glob("*.pin.json")):
        j = json.loads(p.read_text())
        pins[j["repo"]] = j
    fetched = {}
    if args.fetch_manifest and Path(args.fetch_manifest).exists():
        for m in load_jsonl(args.fetch_manifest):
            fetched[(m["repo"], m["file"], m["abs_start"], m["abs_end"])] = m

    L = []
    L.append("## Repositories and shard totals (source facts)\n")
    L.append(
        "| Model | Revision | Shards | Shard bytes (API) | Tensor bytes (headers) | Tensors | Layers | Classes | Gated | License |"
    )
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for repo, c in classes.items():
        pin = pins.get(repo, {})
        shards = [
            f for f in pin.get("files", []) if f["rfilename"].endswith(".safetensors")
        ]
        api_bytes = sum((f.get("lfs_size") or f.get("size") or 0) for f in shards)
        L.append(
            f"| {repo} | `{c['revision'][:12]}` | {len(shards)} | {api_bytes:,} ({gib(api_bytes)} GiB) | {c['total_tensor_bytes']:,} | {c['n_tensors']:,} | {c['n_layers']} | {len(c['classes'])} | {pin.get('gated')} | {pin.get('license')} |"
        )
    hdr_bytes = sum(r["header_len"] for r in receipts)
    L.append(
        f"\n{len(receipts)} shard headers were read by range request ({hdr_bytes:,} header bytes in total); no payload was read for this table.\n"
    )

    L.append("## Layer classes (header-confirmed byte layout)\n")
    L.append(
        "A class is a set of layers whose tensor roles, dtypes and shapes are identical. Bytes per layer come from the shard headers.\n"
    )
    L.append(
        "| Model | Class | Layers | Range | Roles | Routed experts | Bytes per layer | GiB per layer |"
    )
    L.append("|---|---|---:|---|---:|---:|---:|---:|")
    # expert counts from selection (fused tensors carry fused_experts)
    fused_by = {}
    for s in sel:
        if s.get("fused_experts"):
            fused_by[(s["repo"], s["class"])] = s["fused_experts"]
    for repo, c in classes.items():
        for h, v in c["classes"].items():
            ls = v["layers"]
            ne = v["n_experts"] or fused_by.get((repo, h), 0)
            rng = f"{ls[0]}..{ls[-1]}" if len(ls) > 1 else str(ls[0])
            bpl = v["bytes_per_layer"]
            bpl_s = ", ".join(f"{b:,}" for b in bpl)
            L.append(
                f"| {repo} | `{h}` | {len(ls)} | {rng} | {v['n_roles']} | {ne} | {bpl_s} | {', '.join(gib(b) for b in bpl)} |"
            )

    L.append("\n## Routed expert units (header-confirmed)\n")
    L.append(
        "Bytes for one routed expert, split into weight codes and dependent metadata (scales, zero points). The bank is the expert count times one expert; it excludes shared experts, routers, attention and norms.\n"
    )
    L.append(
        "| Model | Class | Expert tensors | Code bytes | Metadata bytes | One expert (MiB) | Experts | Bank (GiB) | Top-k from config | Top-k bytes (MiB) |"
    )
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    seen = set()
    for s in sel:
        if s.get("expert") != 0 or not s["unit"].startswith("layer"):
            continue
        key = (s["repo"], s["class"])
        if key in seen:
            continue
        rows = [
            x
            for x in sel
            if x["repo"] == s["repo"]
            and x["unit"] == s["unit"]
            and x.get("expert") == 0
        ]
        code = sum(
            x["nbytes"]
            for x in rows
            if "scale" not in x["name"] and "zeros" not in x["name"]
        )
        meta = sum(
            x["nbytes"] for x in rows if "scale" in x["name"] or "zeros" in x["name"]
        )
        cls = classes[s["repo"]]["classes"][s["class"]]
        ne = cls["n_experts"] or fused_by.get(key, 0)
        cfg = pins.get(s["repo"], {}).get("config.json", {})
        topk = ""
        stack = [cfg]
        while stack and topk == "":
            o = stack.pop()
            if not isinstance(o, dict):
                continue
            for ck in (
                "num_experts_per_tok",
                "num_experts_per_token",
                "n_experts_per_tok",
                "moe_topk",
            ):
                if ck in o:
                    topk = o[ck]
                    break
            stack.extend(v for v in o.values() if isinstance(v, dict))
        names = sorted(
            {
                (
                    x["name"].split(".experts.")[-1].split(".", 1)[-1]
                    if ".experts." in x["name"]
                    else x["name"].rsplit(".", 1)[-1]
                )
                for x in rows
            }
        )
        one = code + meta
        L.append(
            f"| {s['repo']} | `{s['class']}` | {', '.join(names)} | {code:,} | {meta:,} | {mib(one)} | {ne} | {gib(one * ne)} | {topk} | {mib(one * topk) if topk else ''} |"
        )
        seen.add(key)

    L.append("\n## Smallest independent useful rows (header-confirmed)\n")
    L.append(
        "| Model | Tensor | dtype | Row bytes | Start mod 4 KiB (row 1000) | Rows per 4 KiB page |"
    )
    L.append("|---|---|---|---:|---:|---:|")
    for s in sel:
        if s["unit"] == "embedding_row1000" or s["unit"] == "lm_head_row1000":
            L.append(
                f"| {s['repo']} | {s['name']} | {s['dtype']} | {s['nbytes']:,} | {s['start_mod_4k']} | {4096 / s['nbytes']:.2f} |"
            )

    L.append(
        "\n## Selected units and page footprints (calculated from header offsets)\n"
    )
    L.append(
        "Footprint ratio is the union of fetch granules touched, times the granule size, divided by the selected bytes. Granule sizes are labeled hypothetical fetch units, not an SSD or HBF property. Commands assume contiguous extents and a 128 KiB maximum command.\n"
    )
    L.append(
        "| Model | Unit | Ranges | Selected bytes | x4 KiB | x64 KiB | x2 MiB | Min. commands | Fetched |"
    )
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for u in proj["units"]:
        rows = [s for s in sel if s["repo"] == u["repo"] and s["unit"] == u["unit"]]
        got = sum(
            1
            for s in rows
            if (s["repo"], s["file"], s["abs_start"], s["abs_end"]) in fetched
        )
        fr = u["footprint_ratio"]
        L.append(
            f"| {u['repo']} | {u['unit']} | {u['n_ranges']} | {u['W_required']:,} | {fr['4096']:.4f} | {fr['65536']:.3f} | {fr[str(2 << 20)]:.3f} | {u['direct_commands_contiguous']:,} | {got}/{len(rows)} |"
        )

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
