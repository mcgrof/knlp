"""CAS gap #2 -- mixed/joint-visibility cartridge initializer.

Builds a cache whose layout is [ frozen distractor cartridges | trainable target ]:
  - the TARGET cartridge is trainable, initialized from the patient's own record text
    (cart-specific truncation init, same as KVFromText);
  - one or more DISTRACTOR cartridges (already-trained isolated .pt files) are loaded as
    FROZEN, globally-visible (seq_id=-1) tokens that sit in front of the target.

Because every cartridge token carries CARTRIDGE_SEQ_ID (-1), the target is trained while
the distractors are co-attended -- exactly the co-loaded condition that makes isolated
cartridges collapse. Training the target under this pressure is the CAS rescue (a fixed-
distractor, always-joint approximation of the P_iso mixed-visibility rule; noted as such).

The saved cache keeps only `trainable_keys` = the rescued target, so downstream
combine-at-inference concatenates the rescued targets exactly like the isolated ones."""
import os
from pathlib import Path
from typing import List, Optional
import torch

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.initialization.tokenization_utils import MODEL_TO_SYSTEM_PROMPT_TOKENIZER


def _load_cart_trainable(path):
    # RELOAD FIX: reconstruct the FULL trained distractor cart = [frozen | trainable] so a
    # co-loaded distractor carries the sink token it trained with (see cas_combine_eval).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    def _t(p):
        return torch.as_tensor(p.data if hasattr(p, "data") else p)

    def cat(fro, tra):
        t = [_t(p) for p in tra]
        if fro:
            f = [_t(p) for p in fro]
            return [torch.cat([f[i], t[i]], dim=2) for i in range(len(t))]
        return t

    tk = cat(ckpt.get("frozen_keys"), ckpt["trainable_keys"])
    tv = cat(ckpt.get("frozen_values"), ckpt["trainable_values"])
    return tk, tv


class KVFromCarts(KVCacheFactory):
    class Config(KVCacheFactory.Config):
        target_text_source: str
        target_max_tokens: int = 1024
        distractor_paths: List[str] = None
        system_prompt_template: Optional[str] = "{text}"

    def initialize_kv_cache(self, tokenizer, model, attn_config: AttnConfig) -> TrainableCache:
        # 1. Build TARGET trainable KV from the patient record (same path as KVFromText).
        content = Path(self.config.target_text_source).read_text()
        if self.config.system_prompt_template is not None:
            content = self.config.system_prompt_template.format(text=content)
        tok_fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]
        input_ids = tok_fn(tokenizer=tokenizer, content=content,
                           max_tokens=self.config.target_max_tokens).squeeze(0)
        tmp = TrainableCache(config=attn_config)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_ids = input_ids.to(model.device)
                seq_ids = torch.full_like(input_ids, 0, dtype=torch.long)
                position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long).to(model.device)
                model(input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
                      use_cache=True, past_key_values=tmp, mode="generate")
        target_k = [k.detach().cpu() for k in tmp._keys]   # (1,H,T_tgt,D) per layer
        target_v = [v.detach().cpu() for v in tmp._values]

        # 2. Load frozen DISTRACTOR carts and concatenate: [distractors | target].
        dpaths = self.config.distractor_paths or []
        dk_per = [_load_cart_trainable(p) for p in dpaths]
        n_layers = attn_config.n_layers
        init_keys, init_values = [], []
        for li in range(n_layers):
            ks = [dc[0][li] for dc in dk_per] + [target_k[li]]
            vs = [dc[1][li] for dc in dk_per] + [target_v[li]]
            init_keys.append(torch.cat(ks, dim=2).to(torch.bfloat16))
            init_values.append(torch.cat(vs, dim=2).to(torch.bfloat16))
        distractor_tokens = sum(dc[0][0].shape[2] for dc in dk_per) if dk_per else 0

        return TrainableCache(
            config=attn_config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=distractor_tokens,  # distractors frozen, target trainable
        )
