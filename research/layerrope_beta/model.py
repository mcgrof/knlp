# SPDX-License-Identifier: MIT
"""Keep the author training path and expose measured residual states for evaluation."""

import torch

from research.tinystories_depth.model import Model as Backbone, mhc
from research.tinystories_depth.float_quant import scaled_float


class Model(Backbone):
    def __init__(self, arm, depth, seed, beta=-0.5, vocab=50257):
        if arm not in ("R", "L", "M", "ML"):
            raise ValueError(
                "This study supports residual, full LayerRoPE, mHC and their combination"
            )
        super().__init__(arm, depth, seed, vocab=vocab)
        self.beta_initialization = beta
        rope = self.lm.model.layerrope
        if rope is not None:
            with torch.no_grad():
                for site in rope.sites:
                    getattr(rope, site + "_beta").fill_(beta)
                    getattr(rope, site + "_beta_rot").fill_(beta)
        self.observer = None
        self.residual_bits = None

    def track(self, state, pre, post, carry):
        # Distribution measurements run on fixed evaluation batches, off the training path.
        return None

    def tap(self, layer, site, state):
        if self.residual_bits is not None and site.startswith("post_"):
            state = scaled_float(state, self.residual_bits)
        if self.observer is not None:
            self.observer(layer, site, state)
        return state

    def forward(self, ids):
        if self.observer is None and self.residual_bits is None:
            return super().forward(ids)
        h = self.lm.model.embed_tokens(ids).bfloat16()
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        if self.routes is not None:
            h = mhc.expand_to_mhc_ref(h, 4)
        for layer, block in enumerate(self.lm.model.layers):
            h = self.tap(layer, "residual_input", h)
            for index, site in enumerate(("attn", "mlp")):
                if self.routes is None:
                    inp = h
                else:
                    pre, post, carry, _ = self.routes[2 * layer + index](h)
                    with torch.autocast(device_type=h.device.type, enabled=False):
                        inp = mhc.mhc_pre_apply_mix_ref(h, pre)
                if site == "attn":
                    inp = block.input_layernorm(inp, block._gain("in_attn"))
                    self.tap(layer, "attention_read", inp)
                    branch = block.self_attn(
                        inp, position_ids=positions, use_cache=False
                    )[0]
                else:
                    inp = block.post_attention_layernorm(inp, block._gain("in_mlp"))
                    self.tap(layer, "mlp_read", inp)
                    branch = block.mlp(inp)
                branch = block._block_output(branch, site)
                self.tap(layer, site + "_write", branch)
                if self.routes is None:
                    h = h + branch
                else:
                    with torch.autocast(device_type=h.device.type, enabled=False):
                        h = mhc.mhc_post_ref(branch, h, post, carry)
                h = self.tap(layer, "post_" + site, h)
        if self.routes is not None:
            h = h.float().mean(-2).bfloat16()
        return self.lm.lm_head(self.lm.model.norm(h))
