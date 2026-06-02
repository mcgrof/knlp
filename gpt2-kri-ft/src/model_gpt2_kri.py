"""Custom GPT-2 small that accepts an arbitrary [B, H, Tq, Tk] attention mask.

We deliberately do not subclass HuggingFace GPT-2: the internals make it
awkward to pass a per-(layer, head, query, key) boolean mask through to
the softmax. We instead reimplement the model in a nanoGPT-style class
and provide a checked weight-loader from HF GPT-2. A round-trip exporter
is provided in `export_hf_gpt2.py`.

Numerical conventions match HF GPT-2:
- pre-LayerNorm transformer blocks
- NewGELU (tanh approximation) — matches transformers.activations.NewGELU
- Tied wte / lm_head
- causal scaled dot-product attention with explicit boolean mask
- attn weights kept in attention dtype, cast back to value dtype before V

The attention path is plain torch.matmul + softmax. No FlashAttention
dependency. This works on ROCm and accepts arbitrary masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    mlp_hidden: int = 3072
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    initializer_range: float = 0.02


def gpt2_small_config() -> GPT2Config:
    return GPT2Config()


def new_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with arbitrary additive mask support.

    Accepts an optional `attn_mask` of shape `[B, H, Tq, Tk]` (bool).
    True = keep, False = mask. The causal portion is always enforced;
    `attn_mask`, if given, is intersected with the causal mask.

    When `return_kv=True`, returns the per-head Q/K/V used in this call.
    The KRI router uses K and V to compute block centroids and energies.
    """

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        # Fused Q,K,V projection.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

        self.register_buffer(
            "causal_bias",
            torch.tril(torch.ones(cfg.n_positions, cfg.n_positions, dtype=torch.bool)).view(
                1, 1, cfg.n_positions, cfg.n_positions
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Q @ K^T / sqrt(d_head). Matches HF dtype handling.
        att = torch.matmul(q, k.transpose(-1, -2))
        att = att / math.sqrt(self.head_dim)

        causal = self.causal_bias[:, :, :T, :T]
        if attn_mask is None:
            keep = causal
        else:
            assert attn_mask.dtype == torch.bool, "attn_mask must be bool"
            assert attn_mask.shape[-2:] == (T, T), (
                f"attn_mask query/key dims must be ({T},{T}); got {tuple(attn_mask.shape)}"
            )
            keep = causal & attn_mask

        mask_value = torch.finfo(att.dtype).min
        att = att.masked_fill(~keep, mask_value)
        att = F.softmax(att, dim=-1)
        att = att.to(v.dtype)
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        if return_kv:
            return y, (k, v)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=True)
        self.c_proj = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=True)
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ):
        a = self.attn(self.ln_1(x), attn_mask=attn_mask, return_kv=return_kv)
        if return_kv:
            attn_out, kv = a
        else:
            attn_out, kv = a, None
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if return_kv:
            return x, kv
        return x


class GPT2KRI(nn.Module):
    """GPT-2 small with arbitrary attention-mask support, suitable for KRI."""

    def __init__(self, cfg: GPT2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_epsilon)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Tie input/output embeddings.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        self._gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing = False

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_kv: bool = False,
    ):
        """Forward.

        Args:
            input_ids: [B, T] int64 token ids
            labels: optional [B, T] int64; if given, returns cross-entropy
            attn_mask: optional [B, H, T, T] bool — True means attend
            return_kv: if True, also return list of (k, v) per layer
        """
        B, T = input_ids.shape
        assert T <= self.cfg.n_positions, f"sequence length {T} > n_positions {self.cfg.n_positions}"
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)

        kvs = [] if return_kv else None
        for block in self.h:
            if self._gradient_checkpointing and self.training and not return_kv:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, attn_mask, False, use_reentrant=False
                )
            else:
                out = block(x, attn_mask=attn_mask, return_kv=return_kv)
                if return_kv:
                    x, kv = out
                    kvs.append(kv)
                else:
                    x = out

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_kv:
            return logits, loss, kvs
        return logits, loss

    @torch.no_grad()
    def collect_kv(self, input_ids: torch.Tensor):
        """Run a dense forward and return per-layer (k, v) tensors.

        Each entry is shape [B, n_head, T, head_dim]. Used by KRI to
        compute per-layer, per-head block centroids and energies.
        """
        was_training = self.training
        self.eval()
        try:
            _, _, kvs = self.forward(input_ids, return_kv=True)
        finally:
            if was_training:
                self.train()
        return kvs

    # -------- HF round-trip --------

    @classmethod
    def from_hf_gpt2(cls, hf_model_or_name="openai-community/gpt2") -> "GPT2KRI":
        """Load HuggingFace GPT-2 weights into a fresh GPT2KRI instance.

        Accepts either a HuggingFace model name/path or a loaded
        GPT2LMHeadModel.
        """
        from transformers import GPT2LMHeadModel, GPT2Config as HFGPT2Config

        if isinstance(hf_model_or_name, str):
            hf_model = GPT2LMHeadModel.from_pretrained(hf_model_or_name)
        else:
            hf_model = hf_model_or_name

        hf_cfg = hf_model.config
        cfg = GPT2Config(
            vocab_size=hf_cfg.vocab_size,
            n_positions=hf_cfg.n_positions,
            n_embd=hf_cfg.n_embd,
            n_layer=hf_cfg.n_layer,
            n_head=hf_cfg.n_head,
            mlp_hidden=hf_cfg.n_embd * 4,
            layer_norm_epsilon=hf_cfg.layer_norm_epsilon,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            resid_pdrop=0.0,
            initializer_range=hf_cfg.initializer_range,
        )
        model = cls(cfg)

        sd_src = hf_model.state_dict()
        sd_dst = model.state_dict()

        # HF Conv1D stores weight as [in, out]; our Linear stores [out, in].
        conv1d_keys = {
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        }

        renamed = {}
        for k, v in sd_src.items():
            # HF: transformer.wte.weight, transformer.wpe.weight,
            #     transformer.h.<i>.<sub>, transformer.ln_f.{weight,bias}
            # We:           wte.weight,            wpe.weight,
            #                       h.<i>.<sub>,            ln_f.{weight,bias}
            if k.startswith("transformer."):
                kk = k[len("transformer.") :]
            elif k == "lm_head.weight":
                # tied; already covered by wte
                continue
            else:
                kk = k

            # Drop the HF causal-mask buffer keys. They look like
            # `h.<i>.attn.bias` / `h.<i>.attn.masked_bias`, NOT
            # `h.<i>.attn.c_attn.bias`. Split-and-check is precise.
            parts = kk.split(".")
            if len(parts) >= 2 and parts[-2] == "attn" and parts[-1] in (
                "bias",
                "masked_bias",
            ):
                continue

            # Transpose Conv1D weights into Linear shape.
            for suffix in conv1d_keys:
                if kk.endswith(suffix):
                    v = v.t().contiguous()
                    break
            renamed[kk] = v

        # Now load
        missing, unexpected = [], []
        for k in sd_dst.keys():
            if k in renamed:
                if sd_dst[k].shape != renamed[k].shape:
                    raise RuntimeError(
                        f"shape mismatch for {k}: dst={tuple(sd_dst[k].shape)} src={tuple(renamed[k].shape)}"
                    )
                sd_dst[k] = renamed[k]
            else:
                # buffers like causal_bias aren't expected in HF state,
                # and lm_head.weight is tied to wte after load.
                if k.endswith("causal_bias") or k == "lm_head.weight":
                    continue
                missing.append(k)
        for k in renamed.keys():
            if k not in sd_dst:
                unexpected.append(k)

        # Re-tie after copying.
        model.load_state_dict(sd_dst, strict=False)
        model.lm_head.weight = model.wte.weight

        if missing:
            print("WARN: missing keys when loading HF GPT-2:", missing[:5], "...")
        if unexpected:
            print("WARN: unexpected keys when loading HF GPT-2:", unexpected[:5], "...")

        model.eval()
        return model
