"""TrellisRetrofit — initialize a Trellis bounded-memory LM from a pretrained
GPT-2 and distill it against the GPT-2 full-cache teacher (Phase 4).

Rationale: training Trellis from scratch is data-hungry; a retrofit reuses the
teacher's token embedding, LM head, and attention q/k/v/o projections (which map
cleanly onto the Trellis mixer's q/k/v + out projection) and learns only the new
bounded-memory machinery (alpha/beta/gamma) plus a corrective fine-tune. This is
the fair counterpart to KRI-FT: matched base model, matched trainable budget.

Honesty: GPT-2's GELU MLP does not map onto the SwiGLU block, and RMSNorm != the
teacher's LayerNorm, so MLP/norm are re-initialized (not transferred); we
transfer wte, lm_head (tied), and attention q/k/v/o. The bounded-memory mixer is
a different operator from softmax attention, so this is a warm start, not a
weight-equivalent conversion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import TrellisConfig
from .model import TrellisLM


def _gpt2_to_trellis_cfg(hf_cfg, n_slots: int, dtype: str) -> TrellisConfig:
    return TrellisConfig(
        vocab_size=hf_cfg.vocab_size,
        d_model=hf_cfg.n_embd,
        n_layers=hf_cfg.n_layer,
        n_heads=hf_cfg.n_head,
        d_head=hf_cfg.n_embd // hf_cfg.n_head,
        n_slots=n_slots,
        max_seq_len=hf_cfg.n_positions,
        dtype=dtype,
        exact_inner=False,   # retrofit trains in the fast stale mode
        tie_embeddings=True,
    )


class TrellisRetrofit(TrellisLM):
    @classmethod
    def from_gpt2(cls, gpt2_name: str = "openai-community/gpt2",
                  n_slots: int = 64, dtype: str = "bf16") -> "TrellisRetrofit":
        from transformers import GPT2LMHeadModel
        hf = GPT2LMHeadModel.from_pretrained(gpt2_name)
        cfg = _gpt2_to_trellis_cfg(hf.config, n_slots, dtype)
        model = cls(cfg)
        sd = hf.state_dict()
        d = cfg.d_model
        with torch.no_grad():
            # token embedding + tied LM head
            model.wte.weight.copy_(sd["transformer.wte.weight"])
            # final norm: copy LN weight as an RMSNorm-weight warm start
            model.norm_f.w.copy_(sd["transformer.ln_f.weight"])
            for i in range(cfg.n_layers):
                p = f"transformer.h.{i}.attn."
                # GPT-2 Conv1D weight is [in, out]; nn.Linear wants [out, in].
                c_attn = sd[p + "c_attn.weight"]          # [d, 3d]
                q, k, v = c_attn[:, :d], c_attn[:, d:2 * d], c_attn[:, 2 * d:]
                blk = model.blocks[i].mixer
                blk.q_proj.weight.copy_(q.t().contiguous())
                blk.k_proj.weight.copy_(k.t().contiguous())
                blk.v_proj.weight.copy_(v.t().contiguous())
                blk.out_proj.weight.copy_(sd[p + "c_proj.weight"].t().contiguous())
                # alpha_proj, beta_proj, gamma, conv, gate, MLP, RMSNorms keep
                # their fresh init (no compatible teacher source).
        del hf
        return model


def freeze_to_lora(model: TrellisRetrofit, rank: int = 16,
                   targets=("q_proj", "k_proj", "v_proj", "out_proj",
                            "alpha_proj", "beta_proj")):
    """Freeze all weights, attach LoRA adapters on the named linears (plus keep
    gamma trainable). Returns the list of trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(t) for t in targets):
            lo = _LoRALinear(mod, rank)
            _set_module(model, name, lo)
            trainable += [lo.A, lo.B]
    # gamma stays trainable (cheap, per-head)
    for name, p in model.named_parameters():
        if name.endswith("gamma_raw"):
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


class _LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.A, std=0.02)
        self.scale = 1.0 / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def _set_module(root, dotted, new):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    setattr(obj, parts[-1], new)
