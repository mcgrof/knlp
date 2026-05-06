"""Stage: storage benchmark on the multi-output serde branch.

Runs the same Llama-3.1-8B-Instruct-shaped (n_layers=32, n_kv_heads=8,
head_dim=128) storage round-trip used to validate Eqs. 3-4 in the
paper.  Three encodings are compared on the same KV chunks:

  bf16_baseline   raw bf16 K + V bytes (no quantization, reference)
  sym_fp8         upstream-style symmetric FP8: cast K and V each
                  to torch.float8_e4m3fn, store as uint8 bytes.
                  Mirrors LMCache's Fp8QuantizationSerializer
                  semantics from the upstream serde wrapper PR.
  asym_k16_v8     AsymK16V8MultiSerializer / Deserializer from the
                  serde-multi-output-extensions branch
                  (storage-only-dequant mode, group_size=2).

Per cell the stage records: encoded bytes, write+fsync time, cold
read time (best-effort drop_caches before each read), decode time,
recovery quality (V relative error).  All three encodings are run
on every cell so per-cell comparisons share the same chunk content.

Storage targets:

  vm_disk   the user-supplied CONFIG_KNLP_NVME_PATH.  If the config
            is empty, falls back to a tmpdir under the run dir and
            emits a warning that absolute throughput is not portable.

  tmpfs     /dev/shm/<run_id>/serdes_bench.  RAM-backed; characterizes
            the encode/decode cost without storage cost.  Useful for
            sanity-checking that the codec is not the bottleneck on
            slow disks.

The pass/fail criterion is layout-invariant: byte ratios must match
the paper claims (sym 0.500x FP16, asym 0.750x FP16) within 1% on
every cell, and V relative error must stay below 0.075 (FP8 noise
threshold).  Throughput numbers are informational and recorded with
full hardware context for offline comparison.

The stage uses an import shim that bypasses lmcache's package init
(which transitively requires the C extension lmcache.c_ops, not
required for the pure-Python codec).  This lets the stage run on
hosts without a CUDA toolkit installed.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ..stages import StageContext, StageResult


# ---------------------------------------------------------------------------
# Llama-3.1-8B-Instruct KV shape per chunk per layer.
# ---------------------------------------------------------------------------

_LLAMA_8B = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "n_layers": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
}


# ---------------------------------------------------------------------------
# Pass criteria for layout-invariant claims.
# ---------------------------------------------------------------------------

_BYTE_RATIO_TOL = 0.01
_V_REL_ERR_TOL = 0.075
_EXPECTED_RATIOS = {
    "bf16_baseline": 1.0,
    "sym_fp8": 0.5,
    "asym_k16_v8": 0.75,
    # Mode 2 / split-tier: V_8 only on disk; K stays in host RAM.
    # Layout invariant: V_8 / (K_16 + V_16) = 1/4 (paper Eq. 4).
    "asym_v_only": 0.25,
}


# ---------------------------------------------------------------------------
# Stub-injection helper for lmcache's broken-without-C-ext init.
# ---------------------------------------------------------------------------


def _install_serde_shim(lmc_path: Path):
    """Make pure-Python multi serde imports work without c_ops.

    Returns the module handle for
    ``lmcache.v1.distributed.serde.asym_k16_v8`` (and as a side
    effect, the dependent modules are also loaded into sys.modules).
    """
    import importlib.util
    import logging
    import types

    import torch as _t

    sys.modules.setdefault("lmcache", types.ModuleType("lmcache"))

    v1 = types.ModuleType("lmcache.v1")
    v1.__path__ = [str(lmc_path / "lmcache/v1")]
    sys.modules["lmcache.v1"] = v1
    sys.modules.setdefault(
        "lmcache.v1.distributed", types.ModuleType("lmcache.v1.distributed")
    )
    sys.modules.setdefault(
        "lmcache.v1.multiprocess", types.ModuleType("lmcache.v1.multiprocess")
    )

    class _StubMemoryObj:
        pass

    mm = types.ModuleType("lmcache.v1.memory_management")
    mm.MemoryObj = _StubMemoryObj
    sys.modules["lmcache.v1.memory_management"] = mm

    logmod = types.ModuleType("lmcache.logging")
    logmod.init_logger = lambda name=None: logging.getLogger(name or "lmcache.stub")
    sys.modules["lmcache.logging"] = logmod

    ct = types.ModuleType("lmcache.v1.multiprocess.custom_types")
    ct.IPCCacheEngineKey = type("IPCCacheEngineKey", (), {})
    sys.modules["lmcache.v1.multiprocess.custom_types"] = ct

    proto = types.ModuleType("lmcache.v1.protocol")
    DTYPE_TO_INT = {
        None: 0, _t.half: 1, _t.float16: 2, _t.bfloat16: 3,
        _t.float: 4, _t.float32: 4, _t.float64: 5, _t.double: 5,
        _t.uint8: 6, _t.float8_e4m3fn: 7, _t.float8_e5m2: 8,
    }
    proto.DTYPE_TO_INT = DTYPE_TO_INT
    proto.INT_TO_DTYPE = {v: k for k, v in DTYPE_TO_INT.items()}
    sys.modules["lmcache.v1.protocol"] = proto

    serde_pkg = types.ModuleType("lmcache.v1.distributed.serde")
    serde_pkg.__path__ = [str(lmc_path / "lmcache/v1/distributed/serde")]
    sys.modules["lmcache.v1.distributed.serde"] = serde_pkg

    kv_pkg = types.ModuleType("lmcache.v1.kv_codec")
    kv_pkg.__path__ = [str(lmc_path / "lmcache/v1/kv_codec")]
    sys.modules["lmcache.v1.kv_codec"] = kv_pkg

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    _load("lmcache.v1.distributed.api", lmc_path / "lmcache/v1/distributed/api.py")
    _load("lmcache.v1.kv_codec.errors", lmc_path / "lmcache/v1/kv_codec/errors.py")
    _load("lmcache.v1.kv_codec.encoded_kv", lmc_path / "lmcache/v1/kv_codec/encoded_kv.py")
    asym_codec = _load(
        "lmcache.v1.kv_codec.asym_k16_v8",
        lmc_path / "lmcache/v1/kv_codec/asym_k16_v8.py",
    )
    kv_pkg.AsymK16V8Codec = asym_codec.AsymK16V8Codec
    kv_pkg.ScaleScope = sys.modules["lmcache.v1.kv_codec.encoded_kv"].ScaleScope

    _load(
        "lmcache.v1.distributed.serde.base",
        lmc_path / "lmcache/v1/distributed/serde/base.py",
    )
    _load(
        "lmcache.v1.distributed.serde.multi",
        lmc_path / "lmcache/v1/distributed/serde/multi.py",
    )
    return _load(
        "lmcache.v1.distributed.serde.asym_k16_v8",
        lmc_path / "lmcache/v1/distributed/serde/asym_k16_v8.py",
    )


# ---------------------------------------------------------------------------
# Encodings.
# ---------------------------------------------------------------------------


@dataclass
class _FakeMemoryObj:
    """Minimal stand-in mirroring the test_*.py pattern.

    The multi serde uses MemoryObj only as a typing reference; a
    lightweight dataclass with a ``.tensor`` attribute is enough.
    """

    tensor: object = None  # torch.Tensor


def _byte_buffer(num_bytes: int):
    import torch
    return _FakeMemoryObj(tensor=torch.zeros(num_bytes, dtype=torch.uint8))


def _encode_bf16(k, v):
    import torch
    kb = k.contiguous().view(torch.int16).numpy().tobytes()
    vb = v.contiguous().view(torch.int16).numpy().tobytes()
    return struct.pack("<II", len(kb), len(vb)) + kb + vb


def _decode_bf16(blob, k_shape, v_shape):
    import torch
    klen, vlen = struct.unpack_from("<II", blob, 0)
    head = 8
    kb = blob[head : head + klen]
    vb = blob[head + klen : head + klen + vlen]
    k = torch.frombuffer(bytearray(kb), dtype=torch.int16).view(torch.bfloat16).reshape(k_shape).clone()
    v = torch.frombuffer(bytearray(vb), dtype=torch.int16).view(torch.bfloat16).reshape(v_shape).clone()
    return k, v


def _encode_sym_fp8(k, v):
    import torch
    k_fp8 = k.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    v_fp8 = v.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
    kb = k_fp8.numpy().tobytes()
    vb = v_fp8.numpy().tobytes()
    return struct.pack("<II", len(kb), len(vb)) + kb + vb


def _decode_sym_fp8(blob, k_shape, v_shape, dtype):
    import torch
    klen, vlen = struct.unpack_from("<II", blob, 0)
    head = 8
    kb = blob[head : head + klen]
    vb = blob[head + klen : head + klen + vlen]
    k_fp8 = torch.frombuffer(bytearray(kb), dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(k_shape)
    v_fp8 = torch.frombuffer(bytearray(vb), dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(v_shape)
    return k_fp8.to(dtype).clone(), v_fp8.to(dtype).clone()


# ---------------------------------------------------------------------------
# Storage primitives.
# ---------------------------------------------------------------------------


def _drop_caches() -> bool:
    try:
        subprocess.run(
            ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
            input=b"3\n", check=True, capture_output=True, timeout=5,
        )
        return True
    except Exception:
        return False


def _fsync_dir(path: Path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_blob(blob: bytes, path: Path) -> float:
    t0 = time.perf_counter_ns()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, blob)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)
    return (time.perf_counter_ns() - t0) / 1e9


def _read_blob(path: Path) -> Tuple[bytes, float]:
    t0 = time.perf_counter_ns()
    with open(path, "rb") as f:
        data = f.read()
    return data, (time.perf_counter_ns() - t0) / 1e9


# ---------------------------------------------------------------------------
# Per-cell measurement.
# ---------------------------------------------------------------------------


def _measure_cell(name, encode_fn, decode_fn, k, v, target_dir: Path, seed: int):
    target_dir.mkdir(parents=True, exist_ok=True)
    blob_path = target_dir / f"chunk_{seed}_{name}.bin"

    t0 = time.perf_counter_ns()
    blob = encode_fn(k, v)
    t_encode = (time.perf_counter_ns() - t0) / 1e9
    n_bytes = len(blob)

    t_write = _write_blob(blob, blob_path)
    _drop_caches()
    blob_back, t_read = _read_blob(blob_path)
    if blob_back != blob:
        raise RuntimeError(f"{name}: bytes mismatch on round-trip")

    t0 = time.perf_counter_ns()
    k_out, v_out = decode_fn(blob_back)
    t_decode = (time.perf_counter_ns() - t0) / 1e9

    v_diff = (v_out.float() - v.float()).abs()
    v_rel = (v_diff / (v.float().abs() + 1e-6)).median().item()
    v_rel_max = (v_diff / (v.float().abs() + 1e-6)).max().item()
    # Mode 2 / split-tier returns ``k_out is None`` because K is not
    # in the blob.  Record None instead of failing — the pass-criteria
    # check below skips the bit-exact-K assertion for that lane.
    if k_out is None:
        k_diff = None
    else:
        k_diff = (k_out.float() - k.float()).abs().max().item()

    return {
        "encoding": name,
        "n_bytes": n_bytes,
        "encode_s": t_encode,
        "write_s": t_write,
        "read_s": t_read,
        "decode_s": t_decode,
        "write_MBps": (n_bytes / 1e6) / max(t_write, 1e-9),
        "read_MBps": (n_bytes / 1e6) / max(t_read, 1e-9),
        "k_max_abs_err": k_diff,
        "v_rel_err_median": v_rel,
        "v_rel_err_max": v_rel_max,
    }


# ---------------------------------------------------------------------------
# Hardware probe.
# ---------------------------------------------------------------------------


def _safe_run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"ERR: {e}"


def _probe_hardware() -> dict:
    import torch
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "drop_caches_supported": _drop_caches(),
        "lsblk": _safe_run(["lsblk", "-o", "NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL,VENDOR,ROTA"]),
        "df": _safe_run(["df", "-hT"]),
        "mount": _safe_run(["mount"]),
        "mdstat": _safe_run(["cat", "/proc/mdstat"]),
        "fstab": _safe_run(["cat", "/etc/fstab"]),
        "virt_what": _safe_run(["sudo", "-n", "virt-what"]),
    }


# ---------------------------------------------------------------------------
# Stage entry.
# ---------------------------------------------------------------------------


def run(ctx: StageContext) -> StageResult:
    cfg = ctx.cfg
    lmc_path = Path(cfg.worktree_root).resolve() / cfg.raw.get(
        "CONFIG_KNLP_LMCACHE_DIR", "lmcache"
    )
    if not lmc_path.is_dir():
        return StageResult(
            name=ctx.name, status="failed",
            reason=f"lmcache dir not found at {lmc_path}",
        )

    # Resolve targets: vm_disk = CONFIG_KNLP_NVME_PATH or fallback,
    # tmpfs = /dev/shm subdir.
    nvme_root = (
        cfg.nvme_path
        or os.environ.get("KNLP_NVME_PATH", "")
        or str(ctx.stage_dir / "fallback_tmpdir")
    )
    nvme_was_user_supplied = bool(cfg.nvme_path or os.environ.get("KNLP_NVME_PATH"))

    tmpfs_root = "/dev/shm"
    if not os.path.isdir(tmpfs_root):
        tmpfs_root = tempfile.mkdtemp(prefix="serdes_tmpfs_", dir=str(ctx.stage_dir))
    tmpfs_root = os.path.join(tmpfs_root, f"serdes_bench_{os.getpid()}")

    targets = [("vm_disk", nvme_root), ("tmpfs", tmpfs_root)]

    try:
        asym_serde_mod = _install_serde_shim(lmc_path)
    except Exception as e:
        return StageResult(
            name=ctx.name, status="failed",
            reason=f"failed to import serde-multi-output-extensions modules: {e}",
        )

    AsymK16V8MultiSerializer = asym_serde_mod.AsymK16V8MultiSerializer
    AsymK16V8MultiDeserializer = asym_serde_mod.AsymK16V8MultiDeserializer
    # The V-only / split-tier classes are sibling additions to the
    # same module on the serde-multi-output-extensions branch.
    # ``getattr`` with a default lets older checkouts of the branch
    # (without commit bc0d5873) skip Mode 2 cleanly rather than
    # blowing up with AttributeError.
    AsymK16V8VOnlyMultiSerializer = getattr(
        asym_serde_mod, "AsymK16V8VOnlyMultiSerializer", None
    )
    AsymK16V8VOnlyMultiDeserializer = getattr(
        asym_serde_mod, "AsymK16V8VOnlyMultiDeserializer", None
    )
    MemoryLayoutDesc = sys.modules["lmcache.v1.distributed.api"].MemoryLayoutDesc

    asym_s = AsymK16V8MultiSerializer()
    asym_d = AsymK16V8MultiDeserializer()
    asym_v_only_s = (
        AsymK16V8VOnlyMultiSerializer()
        if AsymK16V8VOnlyMultiSerializer is not None
        else None
    )
    asym_v_only_d = (
        AsymK16V8VOnlyMultiDeserializer()
        if AsymK16V8VOnlyMultiDeserializer is not None
        else None
    )

    def _encode_asym(k, v):
        layout = (
            MemoryLayoutDesc(shapes=[k.shape], dtypes=[k.dtype]),
            MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]),
        )
        cap = asym_s.estimate_serialized_size(layout)
        buf = _byte_buffer(cap)
        n = asym_s.serialize(
            (_FakeMemoryObj(tensor=k), _FakeMemoryObj(tensor=v)), buf
        )
        return bytes(buf.tensor[:n].numpy().tobytes())

    def _decode_asym(blob, k_shape, v_shape, dtype):
        import torch
        src = _FakeMemoryObj(tensor=torch.frombuffer(bytearray(blob), dtype=torch.uint8))
        k_out = _FakeMemoryObj(tensor=torch.zeros(k_shape, dtype=dtype))
        v_out = _FakeMemoryObj(tensor=torch.zeros(v_shape, dtype=dtype))
        asym_d.deserialize(src, (k_out, v_out))
        return k_out.tensor, v_out.tensor

    def _encode_asym_v_only(k, v):
        # Mode 2 / split-tier: K is NOT written to the byte buffer
        # (it stays in CPU-pinned host RAM in real deployment).
        layout = (None, MemoryLayoutDesc(shapes=[v.shape], dtypes=[v.dtype]))
        cap = asym_v_only_s.estimate_serialized_size(layout)
        buf = _byte_buffer(cap)
        n = asym_v_only_s.serialize((None, _FakeMemoryObj(tensor=v)), buf)
        return bytes(buf.tensor[:n].numpy().tobytes())

    def _decode_asym_v_only(blob, k_shape, v_shape, dtype):
        # Returns ``(None, V_dq)`` — K is not in the blob.  The bench
        # records ``k_max_abs_err = None`` for this lane and the pass-
        # criteria check skips the bit-exact-K assertion accordingly.
        import torch
        src = _FakeMemoryObj(
            tensor=torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        )
        v_out = _FakeMemoryObj(tensor=torch.zeros(v_shape, dtype=dtype))
        asym_v_only_d.deserialize(src, (None, v_out))
        return None, v_out.tensor

    # Bench sweep.
    import torch
    cfg_model = dict(_LLAMA_8B)
    cfg_model["native_dtype"] = torch.bfloat16

    chunk_seqlens = [256, 1024, 4096]
    seeds = [0, 1, 2, 3]

    rows: List[dict] = []
    for target_label, target_root in targets:
        target_dir = Path(target_root) / "asym_vs_sym_storage_bench"
        for chunk_seqlen in chunk_seqlens:
            k_shape = (
                cfg_model["n_layers"],
                chunk_seqlen,
                cfg_model["n_kv_heads"],
                cfg_model["head_dim"],
            )
            v_shape = k_shape
            for seed in seeds:
                g = torch.Generator().manual_seed(seed)
                k = torch.randn(*k_shape, dtype=cfg_model["native_dtype"], generator=g).contiguous()
                v = torch.randn(*v_shape, dtype=cfg_model["native_dtype"], generator=g).contiguous()
                encs = {
                    "bf16_baseline": (_encode_bf16, lambda b: _decode_bf16(b, k_shape, v_shape)),
                    "sym_fp8": (
                        _encode_sym_fp8,
                        lambda b: _decode_sym_fp8(b, k_shape, v_shape, cfg_model["native_dtype"]),
                    ),
                    "asym_k16_v8": (
                        _encode_asym,
                        lambda b: _decode_asym(b, k_shape, v_shape, cfg_model["native_dtype"]),
                    ),
                }
                if asym_v_only_s is not None and asym_v_only_d is not None:
                    encs["asym_v_only"] = (
                        _encode_asym_v_only,
                        lambda b: _decode_asym_v_only(
                            b, k_shape, v_shape, cfg_model["native_dtype"]
                        ),
                    )
                for name, (enc_fn, dec_fn) in encs.items():
                    r = _measure_cell(name, enc_fn, dec_fn, k, v, target_dir, seed)
                    r.update({
                        "target_label": target_label,
                        "target_root": target_root,
                        "chunk_seqlen": chunk_seqlen,
                        "seed": seed,
                        "n_layers": cfg_model["n_layers"],
                        "n_kv_heads": cfg_model["n_kv_heads"],
                        "head_dim": cfg_model["head_dim"],
                        "native_dtype": "torch.bfloat16",
                    })
                    rows.append(r)
                    ctx.log_metric(
                        f"{name}/{target_label}/chunk{chunk_seqlen}/seed{seed}/n_bytes",
                        r["n_bytes"],
                    )
                    ctx.log_metric(
                        f"{name}/{target_label}/chunk{chunk_seqlen}/seed{seed}/write_MBps",
                        r["write_MBps"],
                    )
                    ctx.log_metric(
                        f"{name}/{target_label}/chunk{chunk_seqlen}/seed{seed}/read_MBps",
                        r["read_MBps"],
                    )

    # Pass criteria: byte ratio + V quality.
    fails: List[str] = []
    for r in rows:
        if r["target_label"] != "vm_disk":
            continue  # tmpfs only used for codec-cost characterization
        if r["encoding"] == "bf16_baseline":
            continue
        bf = next(
            (
                x
                for x in rows
                if x["target_label"] == r["target_label"]
                and x["chunk_seqlen"] == r["chunk_seqlen"]
                and x["seed"] == r["seed"]
                and x["encoding"] == "bf16_baseline"
            ),
            None,
        )
        if bf is None:
            continue
        ratio = r["n_bytes"] / bf["n_bytes"]
        expected = _EXPECTED_RATIOS[r["encoding"]]
        if abs(ratio - expected) > _BYTE_RATIO_TOL:
            fails.append(
                f"{r['encoding']} chunk={r['chunk_seqlen']} seed={r['seed']} "
                f"ratio {ratio:.4f} != expected {expected:.4f} "
                f"(tol {_BYTE_RATIO_TOL})"
            )
        if r["v_rel_err_median"] > _V_REL_ERR_TOL:
            fails.append(
                f"{r['encoding']} chunk={r['chunk_seqlen']} seed={r['seed']} "
                f"V rel err median {r['v_rel_err_median']:.4f} "
                f"exceeds {_V_REL_ERR_TOL}"
            )
        if (
            r["encoding"] == "asym_k16_v8"
            and r["k_max_abs_err"] is not None
            and r["k_max_abs_err"] != 0.0
        ):
            fails.append(
                f"asym_k16_v8 chunk={r['chunk_seqlen']} seed={r['seed']} "
                f"K not bit-exact (max abs err {r['k_max_abs_err']})"
            )

    payload = {
        "model": cfg_model | {"native_dtype": "torch.bfloat16"},
        "rows": rows,
        "context": _probe_hardware(),
        "argv": {
            "chunk_seqlens": chunk_seqlens,
            "seeds": seeds,
            "targets": targets,
            "nvme_was_user_supplied": nvme_was_user_supplied,
            "storage_device_label": cfg.storage_device_label,
            "lmcache_branch_expected": cfg.raw.get("CONFIG_KNLP_LMCACHE_REF", ""),
        },
        "pass_criteria": {
            "byte_ratio_tol": _BYTE_RATIO_TOL,
            "v_rel_err_tol": _V_REL_ERR_TOL,
            "expected_ratios": _EXPECTED_RATIOS,
        },
        "failures": fails,
    }
    out_path = ctx.stage_dir / "serdes_storage_bench.json"
    out_path.write_text(json.dumps(payload, indent=2))
    ctx.telemetry.log_artifact(out_path, "serdes_storage_bench")

    if fails:
        return StageResult(
            name=ctx.name,
            status="failed",
            reason=f"{len(fails)} pass-criteria violations; first: {fails[0]}",
        )
    if not nvme_was_user_supplied:
        ctx.stderr_path.open("a").write(
            "WARN: CONFIG_KNLP_NVME_PATH not set — fell back to tmpdir under run "
            "directory.  Byte-ratio claims still validated; absolute throughput "
            "is container/host-stack dependent and not portable.\n"
        )
    ctx.mark_done({"n_rows": len(rows), "nvme_user_supplied": nvme_was_user_supplied})
    return StageResult(name=ctx.name, status="passed")
