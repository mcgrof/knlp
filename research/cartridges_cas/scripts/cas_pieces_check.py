#!/usr/bin/env python3
"""Check the piece-aware training patch (cas_pieces.py) on a real reader.

Run once per mode; each run installs the patch, as the trainer does.

``MODE=isolated``:
  A. A batch of piece-0 conversations gives exactly zero gradient on
     every slot of piece 1 and on piece 1's sink.
  B. On the same batch, piece 0's loss and gradient in the combined
     cache equal those of a cache holding piece 0 alone, with the
     library's own mask and positions (piece 0 at standalone positions).
``MODE=joint``:
  C. With every piece visible (P_ISO=0), the patched mask gives the same
     logits as the library's mask on the same bundle.

Piece cartridges are random tensors of the reader's shape (the checks
are about masks and positions, not quality); the conversations are real
Self-Study rows, one parquet per piece.

Env:
    MODEL           reader (default Qwen/Qwen3-0.6B)
    PIECE_PARQUETS  comma list, piece order
    LENGTHS         slots per piece incl. its sink (default 40,24)
    MODE            isolated | joint
    N_BATCH         packed sequences per check (default 2)
    DTYPE           fp32 runs reader and cartridges in float32 (default bf16)
"""

import os
import sys

os.environ.setdefault("CARTRIDGES_OUTPUT_DIR", "/tmp/cas_pieces_check")
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import cartridges.models.qwen.modeling_qwen3 as mq
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.datasets import DataSource, TrainDataset
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

import cas_pieces

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
PQ = os.environ["PIECE_PARQUETS"].split(",")
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "40,24").split(",")]
MODE = os.environ.get("MODE", "isolated")
NB = int(os.environ.get("N_BATCH", "2"))
DTYPE = torch.float32 if os.environ.get("DTYPE") == "fp32" else torch.bfloat16
DEV = "cuda"
ORIG_MASK = mq.create_block_mask_w_cache


def piece_parts(cfg):
    g = torch.Generator().manual_seed(0)
    parts = []
    for n in LENGTHS:
        shape = (1, cfg.num_key_value_heads, n, cfg.head_dim)
        k = [
            torch.randn(shape, generator=g).to(DTYPE)
            for _ in range(cfg.num_hidden_layers)
        ]
        v = [
            torch.randn(shape, generator=g).to(DTYPE) * 0.1
            for _ in range(cfg.num_hidden_layers)
        ]
        parts.append((k, v, 1))
    return parts


def make_cache(attn, keys, vals):
    return TrainableCache(
        config=attn,
        init_keys=[k.to(DEV) for k in keys],
        init_values=[v.to(DEV) for v in vals],
        num_frozen_tokens=1,
    ).to(DEV)


def loss_of(model, cache, b):
    cache.clear()
    with torch.amp.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=DTYPE == torch.bfloat16
    ):
        out = model(
            input_ids=b.input_ids.to(DEV),
            seq_ids=b.element_ids.to(DEV),
            position_ids=b.position_ids.to(DEV),
            use_cache=True,
            past_key_values=cache,
        )
        lp = F.log_softmax(out.logits, dim=-1)[
            0, b.topk_token_idxs.to(DEV) - 1, b.topk_token_ids.to(DEV)
        ]
        return (-b.topk_logprobs.to(DEV).exp() * lp).mean(), out.logits


def grads(cache):
    return [
        p.grad.float().clone()
        for p in list(cache.trainable_keys) + list(cache.trainable_values)
    ]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = (
        FlexQwen3ForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEV).eval()
    )
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config
    attn = AttnConfig(
        n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    cas_pieces.install(TrainDataset, mq, TrainableCache, LENGTHS, MODE, 0.0, 42)
    ds = TrainDataset.Config(
        data_sources=[DataSource(path=p, type="local") for p in PQ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ).instantiate(tokenizer=tok, seed=42)
    n0 = ds._cr_bounds[0]
    parts = piece_parts(cfg)
    total = sum(LENGTHS)
    ok = True
    if MODE == "isolated":
        # batches of piece-0 elements only
        ds.batches = [list(range(i * 4, i * 4 + 4)) for i in range(NB)]
        assert all(i < n0 for b in ds.batches for i in b)
        k, v, _ = cas_pieces.concat(parts, shift=False)
        comb = make_cache(attn, k, v)
        solo = make_cache(attn, parts[0][0], parts[0][1])
        for bi in range(NB):
            b = ds[bi]
            for c in (comb, solo):
                for p in c.parameters():
                    p.grad = None
            lc, _ = loss_of(model, comb, b)
            lc.backward()
            gc = grads(comb)
            # standalone: library ids/positions (undo the isolated shift)
            b.position_ids = b.position_ids - (LENGTHS[0] - total)
            ls, _ = loss_of(model, solo, b)
            ls.backward()
            gs = grads(solo)
            n_tr0 = LENGTHS[0] - 1
            leak = max(float(g[:, :, n_tr0:].abs().max()) for g in gc)
            a = torch.cat([g[:, :, :n_tr0].reshape(-1) for g in gc])
            s = torch.cat([g.reshape(-1) for g in gs])
            cos = float(F.cosine_similarity(a, s, dim=0))
            rel = float((a - s).norm() / s.norm())
            print(f"PIECES_CHECK_A batch={bi} piece1_grad_max={leak:.3e}", flush=True)
            print(
                f"PIECES_CHECK_B batch={bi} loss_comb={float(lc):.6f} loss_solo={float(ls):.6f} "
                f"grad_cos={cos:.6f} grad_rel={rel:.2e}",
                flush=True,
            )
            ok &= (
                leak == 0.0
                and abs(float(lc) - float(ls)) < 1e-4
                and cos > 0.999
                and rel < 2e-2
            )
    else:
        ds.batches = [list(range(n0 - 2, n0 + 2)) for _ in range(NB)]  # both pieces
        k, v, _ = cas_pieces.concat(parts, shift=True)
        cache = make_cache(attn, k, v)
        for bi in range(NB):
            b = ds[bi]
            with torch.no_grad():
                _, lp = loss_of(model, cache, b)
                mq.create_block_mask_w_cache, keep = (
                    ORIG_MASK,
                    mq.create_block_mask_w_cache,
                )
                _, lo = loss_of(model, cache, b)
                mq.create_block_mask_w_cache = keep
            d = float((lp.float() - lo.float()).abs().max())
            print(f"PIECES_CHECK_C batch={bi} logits_maxabs={d:.3e}", flush=True)
            ok &= d < 1e-2
    print(f"PIECES_CHECK_{'PASS' if ok else 'FAIL'} mode={MODE}", flush=True)


if __name__ == "__main__":
    main()
