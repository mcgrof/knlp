"""Phase 2.9 — KRI-FT training for SmolLM2 (RoPE + GQA + SwiGLU).

Wraps HuggingFace `SmolLM2-*` (Llama-style) so it accepts arbitrary
boolean attention masks for KRI-Q+N curriculum training, the same
way `GPT2KRI` does for GPT-2 small. Two design decisions:

1. **HF eager attention only, no FlashAttention.** Eager attention
   accepts a `[B, 1, T, T]` additive mask, which is enough for our
   KRI use case (per_head=False mode in `kri_mask.py`). FlashAttn-2
   does not support arbitrary masks. We trade ~2x slower attention
   for the ability to pass any mask shape through, which is the
   whole point of this experiment.

2. **Per-layer K/V capture via forward hooks.** The HF model does
   not natively return K/V tensors per layer; we register pre-output
   hooks on each `LlamaAttention` (or `SmolLM2Attention` — the class
   resolves at runtime) that stash `k_states` and `v_states` into a
   per-call list. `collect_kv()` runs a no-grad forward with those
   hooks active and returns the captured tensors.

The wrapper preserves the HF model unchanged on the weight side, so
the trained checkpoint round-trips back to a standard SmolLM2 model
that vLLM can serve.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmolLM2KRI(nn.Module):
    """Thin wrapper around HF SmolLM2 / Llama-style model.

    Public API mirrors `GPT2KRI` so the existing training loop in
    `src/train_kri.py` can switch base models with one import.
    """

    def __init__(self, hf_model, eos_token_id: int):
        super().__init__()
        self.hf = hf_model
        self.cfg_n_embd = self.hf.config.hidden_size
        self.cfg_n_head = self.hf.config.num_attention_heads
        # GQA: num_key_value_heads can be < num_attention_heads
        self.cfg_n_kv_head = getattr(
            self.hf.config, "num_key_value_heads", self.cfg_n_head
        )
        self.cfg_n_layer = self.hf.config.num_hidden_layers
        self.cfg_n_positions = getattr(self.hf.config, "max_position_embeddings", 4096)
        self.eos_token_id = eos_token_id
        self._capture: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._capture_active = False
        self._hooks: List = []

    # --- KRI-style API shims that mirror GPT2KRI ----------------------

    @property
    def cfg(self):
        # Small shim so callers that do `model.cfg.n_head` still work
        class _C:
            pass
        c = _C()
        c.n_head = self.cfg_n_head
        c.n_embd = self.cfg_n_embd
        c.n_layer = self.cfg_n_layer
        c.n_positions = self.cfg_n_positions
        c.vocab_size = self.hf.config.vocab_size
        return c

    def gradient_checkpointing_enable(self) -> None:
        self.hf.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        self.hf.gradient_checkpointing_disable()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --- Attention-mask conversion ------------------------------------

    @staticmethod
    def _bool_mask_to_additive(mask: Optional[torch.Tensor],
                               dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Convert our [B, H, T, T] bool mask (True=keep) to HF's
        additive [B, 1, T, T] format (0=keep, -inf=mask). If H>1, we
        OR across heads to get a [B, 1, T, T] shared mask, since the
        HF eager attention path expects head-broadcast masks.
        """
        if mask is None:
            return None
        if mask.dtype != torch.bool:
            raise TypeError("attn_mask must be bool")
        B, H, T, _ = mask.shape
        if H > 1:
            mask = mask.any(dim=1, keepdim=True)  # [B, 1, T, T]
        # Convert to additive
        neg_inf = torch.finfo(dtype).min
        add = torch.zeros_like(mask, dtype=dtype)
        add = add.masked_fill(~mask, neg_inf)
        return add

    # --- Forward / collect_kv -----------------------------------------

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                return_kv: bool = False):
        """Mirror GPT2KRI.forward signature."""
        # HF expects positions to be inferred from attention_mask if
        # provided; we pass a custom 4D mask directly via the
        # `attention_mask` kwarg in eager attention.
        # `attention_mask` in HF is interpreted as additive when 4D.
        # We need to figure out the desired dtype for the mask: HF
        # uses the same dtype as the hidden states (which is whatever
        # autocast picks).
        # We compute the additive mask in a way that the HF code path
        # casts it to the right dtype downstream.
        kvs_local: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if return_kv:
            self._enable_capture()
        try:
            # Convert mask to additive [B, 1, T, T]; HF will cast
            # to compute dtype.
            add_mask = None
            if attn_mask is not None:
                ref_dtype = self.hf.dtype if hasattr(self.hf, 'dtype') else torch.float32
                add_mask = self._bool_mask_to_additive(attn_mask, ref_dtype)
            out = self.hf(
                input_ids=input_ids,
                attention_mask=add_mask,
                labels=None,  # we compute loss outside so the loss
                              # accounting matches GPT2KRI
                use_cache=False,
            )
            logits = out.logits
            loss = None
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
        finally:
            if return_kv:
                kvs_local = list(self._capture)
                self._disable_capture()

        if return_kv:
            return logits, loss, kvs_local
        return logits, loss

    @torch.no_grad()
    def collect_kv(self, input_ids: torch.Tensor):
        was_training = self.training
        self.hf.eval()
        try:
            _, _, kvs = self.forward(input_ids, return_kv=True)
        finally:
            if was_training:
                self.hf.train()
        return kvs

    # --- Capture mechanism --------------------------------------------

    def _attn_module_iter(self):
        """Yield attention modules in layer order."""
        # HF Llama/SmolLM2: self.hf.model = LlamaModel (has .layers).
        # With PeftModel wrapping: self.hf.model = LlamaForCausalLM
        # (no .layers), so drill one more level to get LlamaModel.
        inner = self.hf.model
        if not hasattr(inner, "layers"):
            inner = inner.model
        for layer in inner.layers:
            yield layer.self_attn

    def _enable_capture(self):
        self._capture = []
        self._capture_active = True
        for attn in self._attn_module_iter():
            # forward_pre_hook with with_kwargs=True lets us read
            # the kwargs the HF DecoderLayer passes (it uses kwarg
            # form, so plain hooks see empty inputs).
            h = attn.register_forward_pre_hook(
                self._capture_pre_hook(), with_kwargs=True
            )
            self._hooks.append(h)

    def _disable_capture(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._capture_active = False

    def _capture_pre_hook(self):
        """Capture K and V *before* attention runs by recomputing the
        k_proj / v_proj on the hidden_states the layer just passed in.
        That's cheaper than redoing the whole attention forward and
        catches the post-projection K/V we need for KRI scoring.
        """
        def hook(module, args, kwargs):
            # HF LlamaAttention 4.40+: kwargs include 'hidden_states',
            # 'position_embeddings', 'attention_mask', etc. Older
            # versions may pass hidden_states positionally.
            hidden = kwargs.get("hidden_states")
            if hidden is None and len(args) > 0:
                hidden = args[0]
            if hidden is None:
                return  # nothing to do
            B, T, _ = hidden.shape
            k = module.k_proj(hidden)
            v = module.v_proj(hidden)
            n_kv = self.cfg_n_kv_head
            head_dim = k.shape[-1] // n_kv
            k = k.view(B, T, n_kv, head_dim).transpose(1, 2)
            v = v.view(B, T, n_kv, head_dim).transpose(1, 2)
            self._capture.append((k.detach(), v.detach()))
            return None  # do not modify args
        return hook

    # --- From HF loader / round-trip ----------------------------------

    @classmethod
    def from_hf_smollm2(cls, hf_model_name="HuggingFaceTB/SmolLM2-360M") -> "SmolLM2KRI":
        return cls.from_hf(hf_model_name)

    @classmethod
    def from_hf(cls, hf_model_name: str) -> "SmolLM2KRI":
        """Generic HF causal-LM loader for Llama-family models.

        The wrapper is architecture-agnostic for any HF causal LM
        whose decoder layers expose `self_attn.{k_proj,v_proj}` and
        whose top-level access path is `model.model.layers[...]`.
        Tested with SmolLM2-360M, Qwen3-0.6B, Llama-3.2-1B. Other
        models in the Llama-style family (Mistral, Gemma2/3, Phi-3)
        should also work; verify by running the self-test under
        __main__ on the target model first.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_name, attn_implementation="eager", torch_dtype=torch.float32,
        )
        tok = AutoTokenizer.from_pretrained(hf_model_name)
        eos = tok.eos_token_id if tok.eos_token_id is not None else 0
        return cls(hf_model, eos_token_id=eos)


if __name__ == "__main__":
    # Self-test: load the small SmolLM2, run a forward, capture K/V
    import torch
    m = SmolLM2KRI.from_hf_smollm2("HuggingFaceTB/SmolLM2-360M")
    m.eval()
    ids = torch.randint(0, m.cfg.vocab_size, (2, 64))
    with torch.no_grad():
        logits, _, kvs = m.forward(ids, return_kv=True)
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"n layers: {len(kvs)}")
    print(f"K shape per layer: {tuple(kvs[0][0].shape)}")
    print(f"V shape per layer: {tuple(kvs[0][1].shape)}")
