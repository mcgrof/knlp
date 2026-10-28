#!/usr/bin/env python3
"""Stream one real decoder layer from a checkpoint file to a GPU and run it.

This measures the path: file read -> pinned host buffer -> host-to-device
copy -> decoder layer forward on synthetic activations -> completion. It
instantiates only one decoder layer of the architecture (no full model
allocation), binds the admitted tensors, and records each phase with
monotonic nanosecond fences so an NVMe tracer can be joined by window.

Modes:
  resident  load the layer once, then run the forward N times from GPU
            memory. Expected storage reads during the timed trials: zero.
  stream    for each trial, evict the file from the page cache (verified
            with mincore), read the layer's tensors again, copy them to
            the GPU into a fresh buffer, run the forward.
  prefetch  two GPU weight buffers; the read+copy of trial t+1 overlaps
            the compute of trial t on separate streams.

The activations are synthetic (seeded normal). The output of the streamed
layer is compared to the resident layer's output, which must match
exactly for the same bytes. For an AWQ checkpoint the INT4 tensors are
dequantized on the GPU with a reference unpack (no vendor kernel) and the
output is compared against the BF16 layer with a reported relative error.

The replay JSON written here has the same window fields as
replay_ranges.py so traced_replay.py can attribute device commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
from replay_ranges import fiemap, device_lbas, resident_pages  # noqa: E402

PAGE = 4096


def read_index(model_dir: Path):
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    return None


def shard_headers(model_dir: Path):
    """Map tensor name -> (file, abs_start, nbytes, dtype, shape)."""
    import struct

    out = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            hl = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hl))
        base = 8 + hl
        for name, t in hdr.items():
            if name == "__metadata__":
                continue
            s, e = t["data_offsets"]
            out[name] = (str(f), base + s, e - s, t["dtype"], t["shape"])
    return out


DT = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I32": torch.int32,
    "F8_E4M3": torch.float8_e4m3fn,
}


def pread_into(
    fd: int, buf: torch.Tensor, offset: int, nbytes: int, direct_align: int = 0
):
    mv = memoryview(buf.numpy().view("u1") if buf.dtype != torch.uint8 else buf.numpy())
    got = 0
    while got < nbytes:
        k = os.preadv(fd, [mv[got : got + min(nbytes - got, 64 << 20)]], offset + got)
        if k <= 0:
            raise IOError("short read")
        got += k
    return got


def layer_tensor_names(headers, prefix):
    return sorted(n for n in headers if n.startswith(prefix))


def build_layer(config, layer_idx, device):
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer,
        Qwen2RotaryEmbedding,
    )

    with torch.device("meta"):
        layer = Qwen2DecoderLayer(config, layer_idx)
    rot = Qwen2RotaryEmbedding(config, device=device)
    return layer, rot


def awq_dequant(
    qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Reference AutoAWQ GEMM unpack: qweight [K, N/8] int32 -> [K, N] fp16.

    AutoAWQ packs column 8w+j of a word into nibble position
    [0,2,4,6,1,3,5,7][j]; unpacking therefore reads column j from nibble
    [0,4,1,5,2,6,3,7][j] (the inverse permutation). Groups span K.
    """
    nibble_of_column = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=qweight.device)
    shifts = (nibble_of_column * 4).to(torch.int32)
    K, N8 = qweight.shape
    q = (qweight.unsqueeze(-1) >> shifts) & 0xF  # [K, N/8, 8]
    q = q.reshape(K, N8 * 8)
    z = (qzeros.unsqueeze(-1) >> shifts) & 0xF
    z = z.reshape(qzeros.shape[0], N8 * 8)
    G = K // scales.shape[0]
    assert G == group_size, (G, group_size)
    z = z.repeat_interleave(G, dim=0)
    s = scales.repeat_interleave(G, dim=0)
    return ((q - z).to(scales.dtype) * s).contiguous()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model-dir",
        required=True,
        help="directory with config.json and safetensors shards",
    )
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--mode", choices=["resident", "stream", "prefetch"], required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--read", choices=["pread", "direct"], default="pread")
    ap.add_argument("--cache", choices=["cold", "warm"], default="cold")
    ap.add_argument(
        "--awq-reference-dir", help="BF16 model dir to compare an AWQ layer against"
    )
    ap.add_argument("--part-start-sectors", type=int, required=True)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoConfig

    mdir = Path(args.model_dir)
    config = AutoConfig.from_pretrained(mdir)
    headers = shard_headers(mdir)
    prefix = f"model.layers.{args.layer}."
    names = layer_tensor_names(headers, prefix)
    if not names:
        sys.exit("no tensors for layer")
    files = sorted({headers[n][0] for n in names})
    if len(files) != 1:
        sys.exit(f"layer spans {len(files)} shards; this harness expects one")
    fpath = files[0]
    st = os.stat(fpath)
    is_awq = getattr(config, "quantization_config", None) is not None
    qc = getattr(config, "quantization_config", None) or {}
    group_size = (
        qc.get("group_size")
        if isinstance(qc, dict)
        else getattr(qc, "group_size", None)
    )

    dev = torch.device("cuda")
    torch.manual_seed(0)
    layer, rot = build_layer(config, args.layer, dev)
    x = torch.randn(
        args.batch, args.seq, config.hidden_size, device=dev, dtype=torch.bfloat16
    )
    pos = torch.arange(args.seq, device=dev).unsqueeze(0)
    cos, sin = rot(x, pos)

    ranges = [(headers[n][1], headers[n][2]) for n in names]
    W_required = sum(n for _, n in ranges)
    fd0 = os.open(fpath, os.O_RDONLY)
    extents = fiemap(fd0, st.st_size)
    lbas = device_lbas(extents, ranges, args.part_start_sectors)

    # pinned host staging: one buffer per tensor
    host = {
        n: torch.empty(headers[n][2], dtype=torch.uint8, pin_memory=True) for n in names
    }
    rfd = os.open(fpath, os.O_RDONLY | (os.O_DIRECT if args.read == "direct" else 0))

    def read_all():
        got = 0
        for n in names:
            _, off, nb, _, _ = headers[n]
            if args.read == "direct":
                # read the aligned superset into a page-aligned scratch, then slice
                s = off - off % PAGE
                e = off + nb + (-(off + nb) % PAGE)
                scratch = torch.empty(e - s, dtype=torch.uint8, pin_memory=True)
                got += pread_into(rfd, scratch, s, e - s)
                host[n].copy_(scratch[off - s : off - s + nb])
            else:
                got += pread_into(rfd, host[n], off, nb)
        return got

    def to_gpu(stream=None):
        gpu = {}
        with torch.cuda.stream(stream) if stream is not None else torch.no_grad():
            for n in names:
                _, _, nb, dt, shape = headers[n]
                g = torch.empty(nb, dtype=torch.uint8, device=dev)
                g.copy_(host[n], non_blocking=True)
                gpu[n] = g.view(DT[dt]).view(*shape)
        return gpu

    def bind(gpu):
        sd = {}
        for n in names:
            short = n[len(prefix) :]
            t = gpu[n]
            if is_awq and short.endswith(".qweight"):
                base = short[: -len(".qweight")]
                w = awq_dequant(
                    t,
                    gpu[prefix + base + ".qzeros"],
                    gpu[prefix + base + ".scales"],
                    group_size,
                )
                sd[base + ".weight"] = w.t().contiguous().to(torch.bfloat16)
            elif is_awq and (short.endswith(".qzeros") or short.endswith(".scales")):
                continue
            else:
                sd[short] = (
                    t.to(torch.bfloat16)
                    if t.dtype in (torch.float16, torch.float32)
                    else t
                )
        layer.load_state_dict(sd, strict=True, assign=True)

    def forward():
        with torch.no_grad():
            return layer(x, position_embeddings=(cos, sin), position_ids=pos)

    trials = []
    ref_out = None
    t_setup0 = time.monotonic_ns()
    if args.mode == "resident":
        read_all()
        gpu = to_gpu()
        torch.cuda.synchronize()
        bind(gpu)
        ref_out = forward()
        torch.cuda.synchronize()
    t_setup1 = time.monotonic_ns()

    peak0 = torch.cuda.max_memory_allocated()
    for t in range(args.trials):
        rec = {"trial": t}
        if args.mode != "resident":
            if args.cache == "cold":
                for _ in range(4):
                    os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
                    r = resident_pages(fd0, ranges, st.st_size)
                    if r[0] == 0:
                        break
                    time.sleep(0.2)
            else:
                read_all()
                r = resident_pages(fd0, ranges, st.st_size)
            rec["resident_pages_before"], rec["target_pages"] = r
        else:
            rec["resident_pages_before"], rec["target_pages"] = None, None
        time.sleep(args.settle)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic_ns()
        if args.mode == "resident":
            t_read = t_copy = t0
            got = 0
        else:
            got = read_all()
            t_read = time.monotonic_ns()
            gpu = to_gpu()
            torch.cuda.synchronize()
            t_copy = time.monotonic_ns()
            bind(gpu)
        out = forward()
        torch.cuda.synchronize()
        t1 = time.monotonic_ns()
        time.sleep(args.settle)
        t2 = time.monotonic_ns()
        if ref_out is None:
            ref_out = out
            rec["max_abs_diff_vs_first"] = 0.0
        else:
            rec["max_abs_diff_vs_first"] = (
                (out.float() - ref_out.float()).abs().max().item()
            )
        rec.update(
            {
                "t_start_ns": t0,
                "t_read_end_ns": t_read,
                "t_copy_end_ns": t_copy,
                "t_end_ns": t1,
                "t_drain_end_ns": t2,
                "elapsed_ns": t1 - t0,
                "read_ns": t_read - t0,
                "copy_ns": t_copy - t_read,
                "compute_ns": t1 - t_copy,
                "F_bytes": got,
                "F_kind": "read_return" if args.read == "pread" else "padded_direct",
                "H2D_bytes": W_required if args.mode != "resident" else 0,
                "gpu_peak_alloc_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        trials.append(rec)
        print(
            f"trial {t}: read {rec['read_ns'] / 1e6:.2f} ms copy {rec['copy_ns'] / 1e6:.2f} ms compute {rec['compute_ns'] / 1e6:.2f} ms "
            f"F={got} diff={rec['max_abs_diff_vs_first']:.3g} peak={rec['gpu_peak_alloc_bytes']}",
            flush=True,
        )
        if args.mode != "resident":
            del gpu
            torch.cuda.empty_cache()

    awq_cmp = None
    if args.awq_reference_dir:
        # compare this (AWQ) layer's output against the BF16 reference layer
        rdir = Path(args.awq_reference_dir)
        rcfg = AutoConfig.from_pretrained(rdir)
        rh = shard_headers(rdir)
        rnames = layer_tensor_names(rh, prefix)
        rlayer, _ = build_layer(rcfg, args.layer, dev)
        sd = {}
        with open(sorted({rh[n][0] for n in rnames})[0], "rb") as fh:
            for n in rnames:
                _, off, nb, dt, shape = rh[n]
                fh.seek(off)
                b = torch.frombuffer(bytearray(fh.read(nb)), dtype=torch.uint8)
                sd[n[len(prefix) :]] = b.view(DT[dt]).view(*shape).to(dev)
        rlayer.load_state_dict(sd, strict=True, assign=True)
        with torch.no_grad():
            r_out = rlayer(x, position_embeddings=(cos, sin), position_ids=pos)
        d = ref_out.float() - r_out.float()
        awq_cmp = {
            "max_abs_diff": d.abs().max().item(),
            "rel_fro": (d.norm() / r_out.float().norm()).item(),
            "reference_dir": str(rdir),
        }
        print(
            f"AWQ vs BF16 reference: max_abs={awq_cmp['max_abs_diff']:.4g} rel_fro={awq_cmp['rel_fro']:.4g}"
        )

    doc = {
        "file": fpath,
        "file_size": st.st_size,
        "inode": st.st_ino,
        "label": f"gpu-{args.mode}-{args.read}-{args.cache}",
        "repo": str(mdir),
        "unit": f"layer{args.layer}",
        "arm": args.read,
        "cache": args.cache,
        "qd": 1,
        "chunk": 0,
        "align": PAGE,
        "mode": args.mode,
        "is_awq": is_awq,
        "n_ranges": len(ranges),
        "W_required_bytes": W_required,
        "ranges": ranges,
        "tensor_names": names,
        "extents": extents,
        "device_lbas": lbas,
        "part_start_sectors": args.part_start_sectors,
        "setup_ns": t_setup1 - t_setup0,
        "pinned_host_bytes": sum(v.numel() for v in host.values()),
        "gpu_peak_alloc_before_trials": peak0,
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0),
        "synthetic_activations": {
            "batch": args.batch,
            "seq": args.seq,
            "dtype": "bf16",
            "seed": 0,
        },
        "awq_vs_reference": awq_cmp,
        "trials": trials,
    }
    Path(args.out).write_text(json.dumps(doc))


if __name__ == "__main__":
    main()
