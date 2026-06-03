"""Memory + latency bench: Trellis bounded-state vs dense growing-KV.

Reports, per context length: forward tokens/sec, peak GPU memory, and the
model's memory-state bytes (Trellis bounded vs the dense full-KV estimate).
Untrained models are fine here — this measures the systems envelope, not
quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trellis_lm.config import TrellisConfig
from trellis_lm.model import build_model


def dense_full_kv_bytes(cfg, B, T, elem=2):
    return 2 * cfg.n_layers * B * cfg.n_heads * T * cfg.d_head * elem


@torch.no_grad()
def bench_one(kind, cfg, B, T, device, dt, iters=3):
    model = build_model(cfg, kind).to(device)
    if cfg.dtype != "fp32":
        model = model.to(dt)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    # warmup
    with torch.autocast(device_type=device.type, dtype=dt, enabled=cfg.dtype != "fp32"):
        model(idx, training=False)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        with torch.autocast(device_type=device.type, dtype=dt, enabled=cfg.dtype != "fp32"):
            model(idx, training=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt_s = (time.time() - t0) / iters
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    state = model.memory_state_bytes(B) if kind == "trellis" else dense_full_kv_bytes(cfg, B, T)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"kind": kind, "seq_len": T, "tok_per_s": round(B * T / dt_s, 1),
            "peak_gpu_bytes": int(peak), "state_bytes": int(state)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_lens", default="256,512,1024,2048")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_slots", type=int, default=64)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[a.dtype]
    rows = []
    for L in (int(x) for x in a.seq_lens.split(",")):
        cfg = TrellisConfig(vocab_size=4096, d_model=a.d_model, n_layers=a.n_layers,
                            n_heads=a.n_heads, d_head=a.d_head, n_slots=a.n_slots,
                            max_seq_len=L, dtype=a.dtype)
        for kind in ("dense", "trellis"):
            try:
                r = bench_one(kind, cfg, a.batch, L, device, dt)
            except RuntimeError as e:
                r = {"kind": kind, "seq_len": L, "error": str(e)[:80]}
            rows.append(r)
            print(r, flush=True)
    if a.output:
        Path(a.output).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
