"""The finite, predeclared depth-code variants above the executed backbone."""

import math

import torch
from torch import nn

from research.tinystories_depth.model import (
    Model as InheritedModel,
    generator,
    mhc,
)


def wave(phi):
    return torch.stack((phi.cos(), phi.sin(), -phi.cos(), -phi.sin()))


def bounded_write(post, code, rho, family="minimum"):
    beta = post.squeeze(-1)
    margins = torch.cat((beta, 2 - beta), dim=-1)
    if family == "pair":
        cosine = torch.stack(
            (beta[..., 0], 2 - beta[..., 0], beta[..., 2], 2 - beta[..., 2]), -1
        ).amin(-1)
        sine = torch.stack(
            (beta[..., 1], 2 - beta[..., 1], beta[..., 3], 2 - beta[..., 3]), -1
        ).amin(-1)
        slack = torch.stack((cosine, sine, cosine, sine), -1)
    elif family == "reciprocal":
        positive = margins > 0
        safe = torch.where(positive, margins, torch.ones_like(margins))
        slack = safe.reciprocal().sum(-1, keepdim=True).reciprocal()
        slack = torch.where(
            positive.all(-1, keepdim=True), slack, torch.zeros_like(slack)
        )
    else:
        slack = margins.amin(-1, keepdim=True)
    return (beta + rho * slack * code).unsqueeze(-1)


class Model(InheritedModel):
    def __init__(self, variant, depth, seed, vocab=50257, mapping="ordered"):
        assert variant in {f"V{i:02}" for i in range(32)}
        assert mapping in ("ordered", "shuffle", "constant")
        if depth < 2:
            raise ValueError(
                "The predeclared depth search requires at least two blocks"
            )
        inherited = "M" if variant == "V00" else "ML-O" if variant == "V31" else "M-O"
        super().__init__(inherited, depth, seed, vocab=vocab)
        self.variant, self.mapping, self.step = variant, mapping, 0
        self.strength_override = None
        self.code_map = []
        self.mapping_lookup = []
        for site in ("out_attn", "out_mlp"):
            indices = torch.arange(depth)
            active = torch.arange(depth)
            if variant == "V13":
                active = active[: depth // 2]
            elif variant == "V14":
                active = active[depth // 2 :]
            if variant == "V02":
                if len(active) < 2:
                    raise ValueError(
                        "A nonidentity active-block shuffle needs two blocks"
                    )
                g = generator(seed, "mo-code:" + site + ":shuffle")
                perm = torch.randperm(len(active), generator=g)
                while torch.equal(perm, torch.arange(len(active))):
                    perm = torch.randperm(len(active), generator=g)
                indices[active] = active[perm]
            elif variant == "V03":
                indices[active] = (depth - 1) // 2
            elif variant == "V04":
                indices = indices.flip(0)
            lookup = torch.arange(depth)
            if mapping == "shuffle":
                if len(active) < 2:
                    raise ValueError(
                        "A nonidentity active-block shuffle needs two blocks"
                    )
                g = generator(seed, "mo-code:" + site + ":control-shuffle")
                perm = torch.randperm(len(active), generator=g)
                while torch.equal(perm, torch.arange(len(active))):
                    perm = torch.randperm(len(active), generator=g)
                lookup[active] = active[perm]
            elif mapping == "constant":
                lookup[active] = (depth - 1) // 2
            indices = indices[lookup]
            self.mapping_lookup.append(lookup.tolist())
            self.code_map.append(indices.tolist())
        if variant == "V06":
            self.angular.requires_grad_(False)
        if variant in ("V15", "V16"):
            # Sharing means two learned scalars, rather than two unused extras.
            del self.angular.out_mlp_alpha_rot
            del self.angular.out_mlp_beta_rot
        if variant == "V09":
            self.strength_logits = nn.Parameter(torch.zeros(2))
        if variant == "V05":
            angles = torch.stack(
                [
                    torch.rand(
                        depth, generator=generator(seed, "mo-code:" + site + ":random")
                    )
                    * 2
                    * math.pi
                    for site in ("out_attn", "out_mlp")
                ]
            )
            self.register_buffer("fixed_angles", angles)
        if variant == "V20":
            self.normalized_frequency_logits = nn.Parameter(torch.zeros(2))
            self.normalized_offsets = nn.Parameter(torch.zeros(2))
        if variant == "V24":
            # The executed initial phase is exp(0 - .5 log(l+1)).
            angles = torch.stack(
                [
                    (torch.tensor(0.0) + torch.tensor(-0.5) * math.log(l + 1.0)).exp()
                    for l in range(depth)
                ]
            )
            self.free_angles = nn.Parameter(angles.repeat(2, 1))
        if variant in {"V05", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24"}:
            self.angular = None

    def phase(self, layer, site):
        index = self.code_map[site][layer]
        u = index / (self.depth - 1) if self.depth > 1 else 0.0
        device = self.lm.lm_head.weight.device
        dtype = self.lm.lm_head.weight.dtype
        if self.variant == "V05":
            return self.fixed_angles[site, index]
        if self.variant in ("V17", "V18", "V19"):
            k = {"V17": 0.5, "V18": 1.0, "V19": 2.0}[self.variant]
            return torch.tensor(2 * math.pi * k * u, device=device, dtype=dtype)
        if self.variant == "V20":
            k = 2 * self.normalized_frequency_logits[site].sigmoid()
            return 2 * math.pi * k * u + self.normalized_offsets[site]
        if self.variant == "V21":
            return torch.tensor(2 * math.pi * u * u, device=device, dtype=dtype)
        if self.variant == "V24":
            return self.free_angles[site, index]
        schedule = self.angular if self.angular is not None else self.lm.model.layerrope
        selected_site = (
            "out_attn"
            if self.variant in ("V15", "V16")
            else ("out_attn", "out_mlp")[site]
        )
        alpha = getattr(schedule, selected_site + "_alpha_rot")
        beta = getattr(schedule, selected_site + "_beta_rot")
        phi = (alpha + beta * math.log(index + 1.0)).exp()
        return phi + (math.pi / 2 if self.variant == "V16" and site == 1 else 0)

    def code(self, layer, site):
        index = self.code_map[site][layer]
        device = self.lm.lm_head.weight.device
        dtype = self.lm.lm_head.weight.dtype
        if self.variant == "V22":
            u = index / (self.depth - 1) if self.depth > 1 else 0.0
            # Keep the explicitly frozen blend; its energy varies by depth.
            a = torch.tensor(2 * math.pi * u, device=device, dtype=dtype)
            return (wave(a) + wave(2 * a)) / 2
        if self.variant == "V23":
            return torch.tensor(
                ((1, 0, -1, 0), (0, 1, 0, -1), (-1, 0, 1, 0), (0, -1, 0, 1))[index % 4],
                device=device,
                dtype=dtype,
            )
        return wave(self.phase(layer, site))

    def strength(self, layer, site):
        if self.strength_override is not None:
            return self.strength_override
        if self.variant == "V11" and site == 1 or self.variant == "V12" and site == 0:
            return 0.0
        if (
            self.variant == "V13"
            and layer >= self.depth // 2
            or self.variant == "V14"
            and layer < self.depth // 2
        ):
            return 0.0
        if self.variant == "V09":
            return self.strength_logits[site].sigmoid()
        if self.variant == "V10":
            return 0.5 * min(self.step / 500, 1.0)
        return {"V07": 0.25, "V08": 0.75}.get(self.variant, 0.5)

    def routing(self, route, state, layer, site):
        with torch.autocast(device_type=state.device.type, enabled=False):
            if self.variant not in ("V27", "V28", "V29", "V30"):
                pre, post, carry, logits = route(state)
                base_post = post
            else:
                mixes = mhc.mhc_pre_norm_fn_ref(
                    state, route.phi, route.norm_weight, 1e-5
                )
                scale = torch.cat(
                    (
                        route.scale[0].expand(4),
                        route.scale[1].expand(4),
                        route.scale[2].expand(16),
                    )
                )
                z = mixes * scale + route.base
                read, write = z[..., :4], z[..., 4:8]
                base_post = (write.sigmoid() * 2).unsqueeze(-1)
                logits = z[..., 8:].reshape(*state.shape[:2], 4, 4)
                code = self.code(layer, site)
                strength = self.strength(layer, site)
                if self.variant == "V27":
                    write = write + strength * code
                if self.variant in ("V28", "V29"):
                    read = read + strength * code
                if self.variant == "V30":
                    logits = logits + strength * torch.outer(code, code) / 2
                pre, post = read.sigmoid().unsqueeze(-1), (
                    write.sigmoid() * 2
                ).unsqueeze(-1)
                carry = mhc.sinkhorn_normalize_ref(logits, repeat=20, eps=1e-6)
            if self.variant not in ("V00", "V27", "V28", "V30"):
                family = {"V25": "pair", "V26": "reciprocal"}.get(
                    self.variant, "minimum"
                )
                post = bounded_write(
                    post, self.code(layer, site), self.strength(layer, site), family
                )
        return pre, base_post, post, carry, logits

    @torch.no_grad()
    def capture_sample(self, h, branch, base, post, carry, logits, index):
        super().capture_sample(h, branch, base, post, carry, logits, index)
        sample = self.samples[-1]
        ix = sample["indices"].to(h.device)
        sample["base_write"] = base.flatten(0, 1)[ix].cpu()
        sample["coded_write"] = post.flatten(0, 1)[ix].cpu()
        margins = torch.cat((base.squeeze(-1), 2 - base.squeeze(-1)), -1)
        sample["margins"] = margins.flatten(0, 1)[ix].cpu()
        sample["immediate_write_sum_change_max"] = (
            (post - base).sum(-2).abs().amax().item()
        )
        sample["effective_coefficient_perturbation_rms"] = (
            (post - base).square().mean().sqrt().item()
        )
        if self.variant != "V00":
            layer, site = divmod(index, 2)
            sample["depth_code"] = self.code(layer, site).cpu()
            sample["strength"] = torch.as_tensor(self.strength(layer, site)).item()

    def forward(self, ids):
        # Exact inherited rows use the executed forward, including operation order.
        if (
            self.variant in ("V00", "V01", "V31")
            and self.mapping == "ordered"
            and self.strength_override is None
        ):
            return super().forward(ids)
        self.samples, self.invariants = [], None
        h = mhc.expand_to_mhc_ref(self.lm.model.embed_tokens(ids).bfloat16(), 4)
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        for layer, block in enumerate(self.lm.model.layers):
            for site_i, site in enumerate(("attn", "mlp")):
                pre, base, post, carry, logits = self.routing(
                    self.routes[2 * layer + site_i], h, layer, site_i
                )
                with torch.autocast(device_type=h.device.type, enabled=False):
                    inp = mhc.mhc_pre_apply_mix_ref(h, pre)
                if site == "attn":
                    inp = block.input_layernorm(inp, block._gain("in_attn"))
                    branch = block.self_attn(
                        inp, position_ids=positions, use_cache=False
                    )[0]
                else:
                    inp = block.post_attention_layernorm(inp, block._gain("in_mlp"))
                    branch = block.mlp(inp)
                branch = block._block_output(branch, site)
                with torch.autocast(device_type=h.device.type, enabled=False):
                    updated = mhc.mhc_post_ref(branch, h, post, carry)
                self.track(updated, pre, post, carry)
                if self.capture:
                    self.capture_sample(
                        h, branch, base, post, carry, logits, 2 * layer + site_i
                    )
                h = updated
        h = self.lm.model.norm(h.float().mean(-2).bfloat16())
        return self.lm.lm_head(h)

    def identities(self):
        return {
            **super().identities(),
            "variant": self.variant,
            "mapping": self.mapping,
            "code_map": self.code_map,
            "mapping_lookup": self.mapping_lookup,
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
            "shared_schedule_parameters": 2 if self.variant in ("V15", "V16") else None,
            "fixed_multiscale_energy_varies": self.variant == "V22",
        }
