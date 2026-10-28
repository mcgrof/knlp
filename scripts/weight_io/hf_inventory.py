#!/usr/bin/env python3
"""Inventory real open-weight checkpoints from safetensors shard headers.

This tool answers one question without a GPU and without downloading a
whole model: where does each tensor of a Hugging Face checkpoint live on
disk, in bytes? It resolves a repository to an immutable revision, reads
each safetensors shard header through HTTP range requests, records every
tensor's absolute byte range, classifies layers by the set of tensor roles
they contain, selects a bounded set of layers and experts, and fetches only
those byte ranges with hashes.

Subcommands:

  resolve   pin a repository revision; save config, index and file sizes
  headers   read every shard header by range request; write inventory.jsonl
  classify  group layers into classes by tensor-role signature; summarize
  select    choose first/middle/last layer per class and emit selection.jsonl
  fetch     download the selected byte ranges to local files with SHA-256
  verify    check fetched ranges against full shards when both exist

All byte offsets are absolute file offsets. For safetensors the data
section starts at 8 + header_length, and tensor data_offsets are relative
to that section. The tool rejects header lengths above a bound and ranges
that overlap or leave the file. Range responses must return HTTP 206 with a
matching Content-Range, or the tool aborts rather than reading a full shard.

Credentials come from the Hugging Face token file or HF_TOKEN. They are
sent only to the hub host; redirect targets receive the Range header only.
No token is written to any output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

HUB = "https://huggingface.co"
MAX_HEADER_BYTES = 256 * 1024 * 1024
MAX_HEADER_JSON = 64 * 1024 * 1024
USER_AGENT = "knlp-weight-io-inventory/1"


# ----------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------


def _token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    p = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
    if p.exists():
        return p.read_text().strip()
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_opener_noredirect = urllib.request.build_opener(_NoRedirect)
_opener_plain = urllib.request.build_opener()


def _hub_request(url: str, method: str = "GET", extra: dict | None = None):
    """Request a hub URL with auth; follow at most one redirect manually.

    Returns (response, final_url_host). Authorization is sent only to the
    hub host. The Range header, if any, is forwarded to the redirect target.
    """
    headers = {"User-Agent": USER_AGENT}
    tok = _token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    if extra:
        headers.update(extra)
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        resp = _opener_noredirect.open(req, timeout=120)
        return resp, urllib.parse.urlsplit(url).netloc
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            loc = e.headers.get("Location")
            if not loc:
                raise
            if loc.startswith("/"):
                loc = HUB + loc
            h2 = {"User-Agent": USER_AGENT}
            if extra:
                h2.update(extra)
            req2 = urllib.request.Request(loc, method=method, headers=h2)
            resp = _opener_plain.open(req2, timeout=300)
            return resp, urllib.parse.urlsplit(loc).netloc
        raise


def api_json(path: str) -> dict:
    resp, _ = _hub_request(f"{HUB}{path}")
    return json.loads(resp.read().decode())


def resolve_url(repo: str, revision: str, filename: str) -> str:
    return f"{HUB}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{filename}"


def range_get(
    repo: str, revision: str, filename: str, start: int, length: int
) -> tuple[bytes, dict]:
    """Fetch exactly [start, start+length) and verify the 206 contract."""
    assert length > 0
    end = start + length - 1
    url = resolve_url(repo, revision, filename)
    t0 = time.monotonic()
    resp, host = _hub_request(url, extra={"Range": f"bytes={start}-{end}"})
    status = resp.status
    cr = resp.headers.get("Content-Range", "")
    cl = resp.headers.get("Content-Length")
    if status != 206:
        resp.close()
        raise RuntimeError(
            f"{filename}: expected 206 for range, got {status}; refusing full-body read"
        )
    m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", cr)
    if not m or int(m.group(1)) != start or int(m.group(2)) != end:
        resp.close()
        raise RuntimeError(
            f"{filename}: Content-Range mismatch: {cr!r} for {start}-{end}"
        )
    if cl is not None and int(cl) != length:
        resp.close()
        raise RuntimeError(f"{filename}: Content-Length {cl} != {length}")
    buf = bytearray()
    while len(buf) < length:
        chunk = resp.read(min(8 << 20, length - len(buf)))
        if not chunk:
            break
        buf += chunk
    resp.close()
    if len(buf) != length:
        raise RuntimeError(f"{filename}: short range body {len(buf)} != {length}")
    dt = time.monotonic() - t0
    meta = {
        "status": status,
        "content_range": cr,
        "host": host,
        "total_size": None if m.group(3) == "*" else int(m.group(3)),
        "seconds": round(dt, 4),
    }
    return bytes(buf), meta


# ----------------------------------------------------------------------
# resolve
# ----------------------------------------------------------------------


def cmd_resolve(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.repos:
        repo, _, rev = spec.partition("@")
        rev = rev or "main"
        info = api_json(
            f"/api/models/{repo}/revision/{urllib.parse.quote(rev, safe='')}?blobs=true"
        )
        sha = info["sha"]
        files = []
        for s in info.get("siblings", []):
            files.append(
                {
                    "rfilename": s["rfilename"],
                    "size": s.get("size"),
                    "lfs_sha256": (s.get("lfs") or {}).get("sha256"),
                    "lfs_size": (s.get("lfs") or {}).get("size"),
                }
            )
        pin = {
            "repo": repo,
            "requested_revision": rev,
            "sha": sha,
            "gated": info.get("gated"),
            "license": (info.get("cardData") or {}).get("license"),
            "last_modified": info.get("lastModified"),
            "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": files,
        }
        for name in ("config.json", "model.safetensors.index.json"):
            if any(f["rfilename"] == name for f in files):
                resp, _ = _hub_request(resolve_url(repo, sha, name))
                pin[name] = json.loads(resp.read().decode())
        stem = repo.replace("/", "__")
        (out / f"{stem}.pin.json").write_text(json.dumps(pin, indent=1, sort_keys=True))
        st = [f for f in files if f["rfilename"].endswith(".safetensors")]
        tot = sum(f.get("lfs_size") or f.get("size") or 0 for f in st)
        print(f"{repo} @ {sha} gated={pin['gated']} safetensors={len(st)} bytes={tot}")


# ----------------------------------------------------------------------
# headers -> inventory.jsonl
# ----------------------------------------------------------------------

_LAYER_RE = re.compile(
    r"^(?:model\.|language_model\.|transformer\.)?(?:model\.)?layers\.(\d+)\.(.*)$"
)
_EXPERT_RE = re.compile(r"^(.*experts)\.(\d+)\.(.*)$")


def classify_name(name: str) -> dict:
    """Return layer_index, expert, role and top-level kind for a tensor name."""
    m = _LAYER_RE.match(name)
    if not m:
        kind = "other"
        if "embed" in name:
            kind = "embedding"
        elif "lm_head" in name:
            kind = "lm_head"
        elif name.endswith("norm.weight") or ".norm." in name:
            kind = "final_norm"
        return {"layer_index": None, "expert": None, "role": name, "kind": kind}
    li = int(m.group(1))
    rest = m.group(2)
    em = _EXPERT_RE.match(rest)
    if em:
        return {
            "layer_index": li,
            "expert": int(em.group(2)),
            "role": f"{em.group(1)}.E.{em.group(3)}",
            "kind": "routed_expert",
        }
    kind = "layer_other"
    if "shared_expert" in rest:
        kind = "shared_expert"
    elif ".mlp.gate." in f".{rest}" and not rest.endswith("proj.weight"):
        kind = "router"
    elif "mlp" in rest:
        kind = "mlp"
    elif "attn" in rest or "attention" in rest:
        kind = "attention"
    elif "norm" in rest:
        kind = "norm"
    return {"layer_index": li, "expert": None, "role": rest, "kind": kind}


def parse_header(blob: bytes, filename: str, file_size: int | None):
    hl = struct.unpack("<Q", blob[:8])[0]
    if hl > MAX_HEADER_JSON or hl == 0:
        raise RuntimeError(f"{filename}: unreasonable header length {hl}")
    return hl


def cmd_headers(args):
    pins = sorted(Path(args.pins).glob("*.pin.json"))
    if args.repos:
        want = set(args.repos)
        pins = [p for p in pins if json.loads(p.read_text())["repo"] in want]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    receipts = []
    with open(outp, mode) as fo:
        for p in pins:
            pin = json.loads(p.read_text())
            repo, sha = pin["repo"], pin["sha"]
            shards = [
                f for f in pin["files"] if f["rfilename"].endswith(".safetensors")
            ]
            print(f"{repo}: {len(shards)} shards", flush=True)
            for f in shards:
                fn = f["rfilename"]
                fsize = f.get("lfs_size") or f.get("size")
                # two bounded range requests: 8-byte length, then the JSON header
                b8, m8 = range_get(repo, sha, fn, 0, 8)
                hl = parse_header(b8, fn, fsize)
                hdr, mh = range_get(repo, sha, fn, 8, hl)
                if (
                    mh["total_size"] is not None
                    and fsize is not None
                    and mh["total_size"] != fsize
                ):
                    raise RuntimeError(
                        f"{fn}: API size {fsize} != Content-Range total {mh['total_size']}"
                    )
                fsize = fsize or mh["total_size"]
                j = json.loads(hdr.decode())
                meta = j.pop("__metadata__", None)
                data_base = 8 + hl
                ranges = []
                for name, t in j.items():
                    s, e = t["data_offsets"]
                    a0, a1 = data_base + s, data_base + e
                    if a1 > fsize or a0 < data_base or e < s:
                        raise RuntimeError(
                            f"{fn}:{name}: range {a0}-{a1} outside file {fsize}"
                        )
                    ranges.append((a0, a1, name))
                    c = classify_name(name)
                    row = {
                        "repo": repo,
                        "revision": sha,
                        "file": fn,
                        "file_size": fsize,
                        "header_len": hl,
                        "name": name,
                        "dtype": t["dtype"],
                        "shape": t["shape"],
                        "abs_start": a0,
                        "abs_end": a1,
                        "nbytes": a1 - a0,
                        "start_mod_4k": a0 % 4096,
                        "start_mod_2m": a0 % (2 << 20),
                        **c,
                    }
                    fo.write(json.dumps(row) + "\n")
                ranges.sort()
                for (a, b, n1), (c_, d, n2) in zip(ranges, ranges[1:]):
                    if c_ < b:
                        raise RuntimeError(f"{fn}: overlapping tensors {n1} and {n2}")
                receipts.append(
                    {
                        "repo": repo,
                        "revision": sha,
                        "file": fn,
                        "file_size": fsize,
                        "header_len": hl,
                        "n_tensors": len(j),
                        "header_sha256": hashlib.sha256(hdr).hexdigest(),
                        "metadata": meta,
                        "host": mh["host"],
                        "status": mh["status"],
                        "lfs_sha256": f.get("lfs_sha256"),
                    }
                )
                print(
                    f"  {fn}: {len(j)} tensors, header {hl} B, file {fsize} B",
                    flush=True,
                )
    rp = outp.with_suffix(".receipts.json")
    old = json.loads(rp.read_text()) if (args.append and rp.exists()) else []
    rp.write_text(json.dumps(old + receipts, indent=1))


# ----------------------------------------------------------------------
# classify
# ----------------------------------------------------------------------


def load_inventory(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def layer_signatures(rows):
    """Map (repo, layer_index) -> sorted tuple of (role, dtype, shape) with expert collapsed."""
    sig = {}
    nexp = {}
    for r in rows:
        if r["layer_index"] is None:
            continue
        key = (r["repo"], r["layer_index"])
        sig.setdefault(key, set()).add((r["role"], r["dtype"], tuple(r["shape"])))
        if r["expert"] is not None:
            nexp.setdefault(key, set()).add(r["expert"])
    out = {}
    for key, s in sig.items():
        out[key] = (tuple(sorted(s)), len(nexp.get(key, ())))
    return out


def cmd_classify(args):
    rows = load_inventory(args.inventory)
    sigs = layer_signatures(rows)
    bytes_by = {}
    for r in rows:
        if r["layer_index"] is None:
            continue
        bytes_by[(r["repo"], r["layer_index"])] = (
            bytes_by.get((r["repo"], r["layer_index"]), 0) + r["nbytes"]
        )
    repos = sorted({r["repo"] for r in rows})
    report = {}
    for repo in repos:
        keys = sorted(k for k in sigs if k[0] == repo)
        classes = {}
        for k in keys:
            s, ne = sigs[k]
            h = hashlib.sha256(json.dumps(s).encode()).hexdigest()[:10]
            cl = classes.setdefault(
                h,
                {
                    "layers": [],
                    "n_roles": len(s),
                    "n_experts": ne,
                    "bytes_per_layer": set(),
                },
            )
            cl["layers"].append(k[1])
            cl["bytes_per_layer"].add(bytes_by[k])
        nonlayer = {}
        for r in rows:
            if r["repo"] == repo and r["layer_index"] is None:
                nonlayer[r["name"]] = {
                    "dtype": r["dtype"],
                    "shape": r["shape"],
                    "nbytes": r["nbytes"],
                    "kind": r["kind"],
                }
        total = sum(r["nbytes"] for r in rows if r["repo"] == repo)
        report[repo] = {
            "revision": next(r["revision"] for r in rows if r["repo"] == repo),
            "total_tensor_bytes": total,
            "n_tensors": sum(1 for r in rows if r["repo"] == repo),
            "n_layers": len(keys),
            "classes": {
                h: {
                    "layers": v["layers"],
                    "n_roles": v["n_roles"],
                    "n_experts": v["n_experts"],
                    "bytes_per_layer": sorted(v["bytes_per_layer"]),
                }
                for h, v in classes.items()
            },
            "non_layer_tensors": nonlayer,
        }
        print(
            f"{repo}: {total} B in {report[repo]['n_tensors']} tensors, {len(keys)} layers, {len(classes)} classes"
        )
        for h, v in classes.items():
            ls = v["layers"]
            span = f"{ls[0]}..{ls[-1]}" if len(ls) > 1 else str(ls[0])
            print(
                f"  class {h}: {len(ls)} layers [{span}], {v['n_roles']} roles, {v['n_experts']} experts, bytes/layer {sorted(v['bytes_per_layer'])}"
            )
    Path(args.out).write_text(json.dumps(report, indent=1))


# ----------------------------------------------------------------------
# select
# ----------------------------------------------------------------------


def _fused_expert_count(r: dict) -> int:
    """Return the expert count when a tensor stacks all routed experts.

    Some checkpoints store `mlp.experts.gate_up_proj` as one 3-D tensor
    whose leading axis is the expert index, instead of one tensor per
    expert. Treat any 3-D tensor whose role names experts without an
    index as fused; the leading dimension is the expert count.
    """
    if r.get("expert") is not None:
        return 0
    if "experts" in r["role"] and len(r["shape"]) == 3:
        return int(r["shape"][0])
    return 0


def cmd_select(args):
    rows = load_inventory(args.inventory)
    sigs = layer_signatures(rows)
    by_layer = {}
    for r in rows:
        by_layer.setdefault((r["repo"], r["layer_index"]), []).append(r)
    sel = []
    for repo in sorted({r["repo"] for r in rows}):
        keys = sorted(k for k in sigs if k[0] == repo)
        classes = {}
        for k in keys:
            h = hashlib.sha256(json.dumps(sigs[k][0]).encode()).hexdigest()[:10]
            classes.setdefault(h, []).append(k[1])
        for h, layers in classes.items():

            def single_shard(li):
                return len({r["file"] for r in by_layer[(repo, li)]}) == 1

            def nearest_single(li):
                # a layer that spans two shards cannot be replayed from one
                # file; step outward within the class to the nearest one
                if single_shard(li):
                    return li, False
                for d in range(1, len(layers)):
                    for cand in (li - d, li + d):
                        if cand in layers and single_shard(cand):
                            return cand, True
                return li, False

            picks = {}
            for li, pos in (
                (layers[0], "first"),
                (layers[len(layers) // 2], "middle"),
                (layers[-1], "last"),
            ):
                adj, moved = nearest_single(li)
                picks.setdefault(adj, pos + ("(adjusted)" if moved else ""))
            for li, pos in sorted(picks.items()):
                for r in by_layer[(repo, li)]:
                    if r["expert"] is not None and r["expert"] not in args.experts:
                        continue
                    fused = _fused_expert_count(r)
                    if fused:
                        # one 3-D tensor holds every expert; select contiguous
                        # per-expert slabs along the leading axis
                        slab = r["nbytes"] // fused
                        for e in args.experts:
                            if e >= fused:
                                continue
                            a0 = r["abs_start"] + e * slab
                            sel.append(
                                {
                                    **r,
                                    "abs_start": a0,
                                    "abs_end": a0 + slab,
                                    "nbytes": slab,
                                    "start_mod_4k": a0 % 4096,
                                    "start_mod_2m": a0 % (2 << 20),
                                    "expert": e,
                                    "kind": "routed_expert",
                                    "fused_experts": fused,
                                    "sub_range": True,
                                    "unit": f"layer{li}",
                                    "class": h,
                                    "position": pos,
                                }
                            )
                        continue
                    sel.append({**r, "unit": f"layer{li}", "class": h, "position": pos})
        # embedding and head: whole tensors are large; select a few rows instead
        for r in rows:
            if r["repo"] != repo or r["layer_index"] is not None:
                continue
            if r["kind"] in ("embedding", "lm_head") and len(r["shape"]) == 2:
                rowsz = r["nbytes"] // r["shape"][0]
                for tid in args.token_ids:
                    sel.append(
                        {
                            **r,
                            "abs_start": r["abs_start"] + tid * rowsz,
                            "abs_end": r["abs_start"] + (tid + 1) * rowsz,
                            "nbytes": rowsz,
                            "start_mod_4k": (r["abs_start"] + tid * rowsz) % 4096,
                            "start_mod_2m": (r["abs_start"] + tid * rowsz) % (2 << 20),
                            "unit": f"{r['kind']}_row{tid}",
                            "class": "rows",
                            "position": "row",
                            "sub_range": True,
                        }
                    )
            elif r["kind"] == "final_norm":
                sel.append(
                    {**r, "unit": "final_norm", "class": "nonlayer", "position": "n/a"}
                )
    with open(args.out, "w") as f:
        for s in sel:
            f.write(json.dumps(s) + "\n")
    tot = sum(s["nbytes"] for s in sel)
    per = {}
    for s in sel:
        per[s["repo"]] = per.get(s["repo"], 0) + s["nbytes"]
    print(f"selected {len(sel)} ranges, {tot} B")
    for k, v in per.items():
        print(f"  {k}: {v} B ({v / 2**30:.3f} GiB)")


# ----------------------------------------------------------------------
# fetch
# ----------------------------------------------------------------------


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cmd_fetch(args):
    sel = load_inventory(args.selection)
    if args.repos:
        sel = [s for s in sel if s["repo"] in set(args.repos)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "fetch_manifest.jsonl"
    done = set()
    if manifest_path.exists():
        for m in load_inventory(str(manifest_path)):
            done.add((m["repo"], m["file"], m["abs_start"], m["abs_end"]))
    with open(manifest_path, "a") as mf:
        for s in sel:
            key = (s["repo"], s["file"], s["abs_start"], s["abs_end"])
            if key in done:
                continue
            if s["nbytes"] > args.max_range_bytes:
                print(f"skip {s['name']} ({s['nbytes']} B > max)", flush=True)
                continue
            stem = s["repo"].replace("/", "__")
            d = out / stem / "compact"
            d.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", s["name"])
            suffix = f"_{s['abs_start']}" if s.get("sub_range") else ""
            fpath = d / f"{safe}{suffix}.bin"
            t0 = time.monotonic()
            blob, meta = range_get(
                s["repo"], s["revision"], s["file"], s["abs_start"], s["nbytes"]
            )
            fpath.write_bytes(blob)
            rec = {
                **{
                    k: s[k]
                    for k in (
                        "repo",
                        "revision",
                        "file",
                        "name",
                        "dtype",
                        "shape",
                        "abs_start",
                        "abs_end",
                        "nbytes",
                        "unit",
                        "kind",
                        "layer_index",
                        "expert",
                    )
                },
                "compact_path": str(fpath.relative_to(out)),
                "sha256": _sha(blob),
                "http": meta,
                "wall_seconds": round(time.monotonic() - t0, 3),
            }
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            print(
                f"{s['repo']} {s['name']} {s['nbytes']} B in {rec['wall_seconds']} s",
                flush=True,
            )


def cmd_fetch_full(args):
    """Download whole shards by range in chunks and verify the LFS sha256."""
    pin = json.loads(Path(args.pin).read_text())
    repo, sha = pin["repo"], pin["sha"]
    out = Path(args.out) / repo.replace("/", "__") / "full"
    out.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_bytes
    results = []
    for f in pin["files"]:
        fn = f["rfilename"]
        if not fn.endswith(".safetensors") and fn not in (
            "config.json",
            "model.safetensors.index.json",
        ):
            continue
        size = f.get("lfs_size") or f.get("size")
        dst = out / fn
        if dst.exists() and dst.stat().st_size == size and not args.force:
            print(f"have {fn}", flush=True)
            continue
        h = hashlib.sha256()
        t0 = time.monotonic()
        with open(dst, "wb") as fo:
            pos = 0
            while pos < size:
                n = min(chunk, size - pos)
                blob, _ = range_get(repo, sha, fn, pos, n)
                fo.write(blob)
                h.update(blob)
                pos += n
            fo.flush()
            os.fsync(fo.fileno())
        dt = time.monotonic() - t0
        ok = (f.get("lfs_sha256") is None) or (h.hexdigest() == f["lfs_sha256"])
        results.append(
            {
                "file": fn,
                "size": size,
                "sha256": h.hexdigest(),
                "lfs_sha256": f.get("lfs_sha256"),
                "verified": ok,
                "seconds": round(dt, 2),
            }
        )
        print(
            f"{fn}: {size} B in {dt:.1f} s ({size / dt / 2**20:.1f} MiB/s) sha_ok={ok}",
            flush=True,
        )
        if not ok:
            raise RuntimeError(f"{fn}: sha256 mismatch against LFS pointer")
    (out / "full_manifest.json").write_text(
        json.dumps({"repo": repo, "revision": sha, "files": results}, indent=1)
    )


def cmd_verify(args):
    """Compare fetched compact ranges against the same bytes inside full shards."""
    man = load_inventory(args.manifest)
    base = Path(args.out)
    n_ok = n_bad = n_skip = 0
    for m in man:
        full = base / m["repo"].replace("/", "__") / "full" / m["file"]
        if not full.exists():
            n_skip += 1
            continue
        with open(full, "rb") as f:
            f.seek(m["abs_start"])
            b = f.read(m["nbytes"])
        if _sha(b) == m["sha256"]:
            n_ok += 1
        else:
            n_bad += 1
            print(f"MISMATCH {m['repo']} {m['name']}")
    print(f"verified ok={n_ok} bad={n_bad} skipped(no full shard)={n_skip}")
    if n_bad:
        sys.exit(1)


# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("resolve")
    p.add_argument("repos", nargs="+", help="org/name or org/name@revision")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_resolve)

    p = sp.add_parser("headers")
    p.add_argument("--pins", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repos", nargs="*")
    p.add_argument("--append", action="store_true")
    p.set_defaults(fn=cmd_headers)

    p = sp.add_parser("classify")
    p.add_argument("--inventory", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_classify)

    p = sp.add_parser("select")
    p.add_argument("--inventory", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--experts", type=int, nargs="*", default=[0, 1, 7, 31, 127])
    p.add_argument("--token-ids", type=int, nargs="*", default=[0, 1000, 151643])
    p.set_defaults(fn=cmd_select)

    p = sp.add_parser("fetch")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repos", nargs="*")
    p.add_argument("--max-range-bytes", type=int, default=8 << 30)
    p.set_defaults(fn=cmd_fetch)

    p = sp.add_parser("fetch-full")
    p.add_argument("--pin", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-bytes", type=int, default=256 << 20)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_fetch_full)

    p = sp.add_parser("verify")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
