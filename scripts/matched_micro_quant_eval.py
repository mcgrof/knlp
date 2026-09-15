#!/usr/bin/env python3
"""Precision of the recurrent state and of attention K/V, evaluated on
trained matched micro checkpoints without retraining.

A KV cache is written once and read many times; a Gated DeltaNet state
is an accumulator, read and rewritten every token by an error-correcting
rule behind a forget gate. This harness replays fla's GatedDeltaNet layer
through its own kernels in pieces (a chunk of tokens, or one token at a
time, the decode case) and rounds the recurrent state to a chosen format
between pieces, so the number reported is what a serving stack that
stored the state in that format would see. Attention layers get the
asymmetric-KV treatment: keys and values fake-quantized to fp8 with a
per-token, per-head scale.

Tasks: `lm` scores held-out validation loss on the language-model
checkpoints; `recall` scores the associative-recall quiz on the recall
cells. fp32 state with no chunking must reproduce the archived numbers;
that is the correctness gate for the replay path.

    python3 scripts/matched_micro_quant_eval.py --task lm --checkpoints a.pt \\
        --state-dtypes fp32,bf16,e4m3,e5m2,int8 --chunks 64,1 --kv none,v8,k8v8
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_micro_train import (
    CONTRACT,
    Attention,
    StackLM,
    TokenStream,
    ce_loss,
)  # noqa: E402

FP8_MAX = dict(e4m3=448.0, e5m2=57344.0)


# ---------------------------------------------------------------------------
# quantizers (fake-quant: round trip through the format, compute stays fp32)
# ---------------------------------------------------------------------------


def quantize(x, fmt, scale_dims):
    """Round x through fmt with an absmax scale shared over scale_dims
    (a tuple of trailing dims to reduce over). Returns fp32."""
    if fmt == "fp32":
        return x
    if fmt in ("bf16", "fp16"):
        return x.to(torch.bfloat16 if fmt == "bf16" else torch.float16).to(x.dtype)
    amax = x.abs().amax(dim=scale_dims, keepdim=True).clamp_min(1e-12)
    if fmt in FP8_MAX:
        scale = amax / FP8_MAX[fmt]
        dt = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
        return (x / scale).to(dt).to(x.dtype) * scale
    if fmt == "int8":
        scale = amax / 127.0
        return torch.round(x / scale).clamp_(-127, 127) * scale
    if fmt == "int8sr":
        # stochastic rounding: unbiased per write, so rounding error does
        # not accumulate as a drift in the state; no extra storage
        scale = amax / 127.0
        y = x / scale
        return (torch.floor(y + torch.rand_like(y))).clamp_(-127, 127) * scale
    raise ValueError(fmt)


def state_quantizer(spec):
    """spec = '<fmt>' (per-head scale over the whole state matrix),
    '<fmt>:row' (one scale per state row), '<fmt>:col' (per column),
    '<fmt>:ef' (error feedback: the rounding residual is carried in
    fp32 and added back before the next rounding; a diagnostic that
    separates accumulated drift from per-step noise, not a storage
    format, since the residual would cost the bits back)."""
    fmt, _, scope = spec.partition(":")
    dims = {"": (-2, -1), "head": (-2, -1), "row": (-1,), "col": (-2,), "ef": (-2, -1)}[
        scope
    ]
    residual = {}

    def q(s):
        x = s.float()
        if scope == "ef":
            r = residual.get("r")
            if r is not None and r.shape == x.shape:
                x = x + r
            y = quantize(x, fmt, dims)
            residual["r"] = x - y
            return y.to(s.dtype)
        return quantize(x, fmt, dims).to(s.dtype)

    return q


# ---------------------------------------------------------------------------
# replayed GatedDeltaNet forward with a quantized state between pieces
# ---------------------------------------------------------------------------


def gdn_replay_forward(mixer, x, chunk, qstate):
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    b, t, _ = x.shape
    conv_q = conv_k = conv_v = None
    state = None
    outs = []
    common = dict(
        A_log=mixer.A_log,
        dt_bias=mixer.dt_bias,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=mixer.allow_neg_eigval,
        state_v_first=True,
    )
    for s in range(0, t, chunk):
        h = x[:, s : s + chunk]
        q, conv_q = mixer.q_conv1d(
            x=mixer.q_proj(h), cache=conv_q, output_final_state=True
        )
        k, conv_k = mixer.k_conv1d(
            x=mixer.k_proj(h), cache=conv_k, output_final_state=True
        )
        v, conv_v = mixer.v_conv1d(
            x=mixer.v_proj(h), cache=conv_v, output_final_state=True
        )
        q, k = (
            rearrange(z, "... (h d) -> ... h d", d=mixer.head_k_dim) for z in (q, k)
        )
        v = rearrange(v, "... (h d) -> ... h d", d=mixer.head_v_dim)
        op = (
            chunk_gated_delta_rule
            if h.shape[1] > 64
            else fused_recurrent_gated_delta_rule
        )
        o, state = op(
            q=q,
            k=k,
            v=v,
            g=mixer.a_proj(h),
            beta=mixer.b_proj(h),
            initial_state=state,
            **common,
        )
        state = qstate(state)
        outs.append(o)
    o = torch.cat(outs, dim=1)
    if mixer.use_gate:
        g = rearrange(mixer.g_proj(x), "... (h d) -> ... h d", d=mixer.head_v_dim)
        o = mixer.o_norm(o, g)
    else:
        o = mixer.o_norm(o)
    return mixer.o_proj(rearrange(o, "b t h d -> b t (h d)")), state


def attention_kv_forward(attn, x, kfmt, vfmt):
    b, t, d = x.shape
    q, k, v = attn.qkv(x).chunk(3, dim=-1)
    if hasattr(attn, "convs"):  # the recall ceiling's convolution-equipped attention
        q, k, v = (conv(z)[0] for conv, z in zip(attn.convs, (q, k, v)))
    q, k, v = (
        z.view(b, t, attn.heads, attn.dim_head).transpose(1, 2) for z in (q, k, v)
    )
    q, k = attn.rope(q, k)
    # per-token, per-head absmax scale, as a KV cache quantizer would apply
    k = quantize(k, kfmt, (-1,))
    v = quantize(v, vfmt, (-1,))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return attn.out(o.transpose(1, 2).reshape(b, t, d))


class Replay:
    """Installs the replay forwards on a StackLM for one configuration."""

    def __init__(self, model, chunk, state_spec, kv):
        self.model = model
        self.chunk = chunk
        self.qstate = state_quantizer(state_spec)
        kfmt, vfmt = {
            "none": ("fp32", "fp32"),
            "v8": ("fp32", "e4m3"),
            "k8v8": ("e4m3", "e4m3"),
        }[kv]
        self.kfmt, self.vfmt = kfmt, vfmt
        self.state_shape = None

    def __enter__(self):
        self.saved = []
        for blk in self.model.blocks:
            m = blk.mixer
            self.saved.append((m, m.forward))
            if blk.mixer_kind == "G":
                m.forward = self._gdn(m)
            elif blk.mixer_kind == "A" and isinstance(m, Attention):
                m.forward = self._attn(m)
        return self

    def __exit__(self, *a):
        for m, f in self.saved:
            m.forward = f

    def _gdn(self, m):
        def fwd(x, **kw):
            o, state = gdn_replay_forward(m, x, self.chunk, self.qstate)
            self.state_shape = tuple(state.shape)
            return o, None, None

        return fwd

    def _attn(self, m):
        def fwd(x):
            return attention_kv_forward(m, x, self.kfmt, self.vfmt)

        return fwd


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@torch.no_grad()
def lm_loss(model, batches):
    return sum(ce_loss(model(idx), idx).item() for idx in batches) / len(batches)


@torch.no_grad()
def recall_accuracy(model, device, levels, examples, seed, eval_batch, vocab):
    import numpy as np

    import mqar_recall as m

    m.set_vocab(vocab)
    out = {}
    for n in levels:
        rng = np.random.default_rng(seed + n)
        x, y, _, _ = m.make_examples(rng, n, examples, 4 * n)
        correct = total = 0
        for s in range(0, examples, eval_batch):
            xb = torch.from_numpy(x[s : s + eval_batch]).to(device)
            yb = torch.from_numpy(y[s : s + eval_batch]).to(device)
            pred = model(xb).argmax(-1)
            mask = yb != -100
            correct += (pred[mask] == yb[mask]).sum().item()
            total += mask.sum().item()
        out[n] = correct / max(1, total)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["lm", "recall"], required=True)
    ap.add_argument("--checkpoints", required=True, help="comma-separated .pt paths")
    ap.add_argument("--state-dtypes", default="fp32,bf16,e4m3,e5m2,int8")
    ap.add_argument("--chunks", default="64,1")
    ap.add_argument("--kv", default="none", help="comma-separated: none,v8,k8v8")
    ap.add_argument("--data-dir", default="matched-micro-data")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--eval-pairs", default="4,16,64,128,256,512")
    ap.add_argument("--eval-examples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    rows = []
    batches = None
    if args.task == "lm":
        stream = TokenStream(args.data_dir, CONTRACT["seq_len"], args.batch)
        batches = stream.val_batches(device, max_batches=args.val_batches)
    for path in args.checkpoints.split(","):
        ck = torch.load(path, map_location="cpu")
        cfg = ck["config"]
        vocab = CONTRACT["vocab_size"] if args.task == "lm" else ck["args"]["vocab"]
        if args.task == "lm":
            model = StackLM(cfg, vocab)
        else:
            import mqar_recall as mq

            mq.set_vocab(vocab)
            model = mq.build(ck["arm"], torch.device("cpu"), 0, cfg.get("dim"))
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        has_gdn = any(b.mixer_kind == "G" for b in model.blocks)
        has_attn = any(b.mixer_kind == "A" for b in model.blocks)
        configs = []
        for kv in args.kv.split(","):
            if kv != "none" and not has_attn:
                continue
            for sd in args.state_dtypes.split(","):
                for ch in (int(c) for c in args.chunks.split(",")):
                    if not has_gdn and (
                        sd != "fp32" or ch != int(args.chunks.split(",")[0])
                    ):
                        continue
                    configs.append((kv, sd, ch))
        for kv, sd, ch in configs:
            with Replay(model, ch, sd, kv) as rp:
                if args.task == "lm":
                    metric = dict(val_loss=lm_loss(model, batches))
                else:
                    levels = [int(n) for n in args.eval_pairs.split(",")]
                    metric = dict(
                        recall=recall_accuracy(
                            model,
                            device,
                            levels,
                            args.eval_examples,
                            args.seed,
                            args.batch,
                            vocab,
                        )
                    )
                row = dict(
                    checkpoint=path,
                    arm=ck["arm"],
                    kv=kv,
                    state_dtype=sd,
                    chunk=ch,
                    state_shape=rp.state_shape,
                    **metric,
                )
            rows.append(row)
            print(json.dumps(row), flush=True)
            with open(args.out, "w") as f:
                json.dump(dict(task=args.task, args=vars(args), rows=rows), f, indent=1)
        del model
        torch.cuda.empty_cache()
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
