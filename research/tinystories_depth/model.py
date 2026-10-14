"""Pristine author LLaMA branches and portable upstream finite mHC."""

import hashlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import LlamaConfig

os.environ["NORM_TYPE"] = "pre"
ROOT = Path(__file__).parent / "upstream"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


author = load(
    "author_layer_rope", ROOT / "layer_rope/peft_pretraining/modeling_llama.py"
)
mhc = load("author_mhc", ROOT / "tile_kernels/tile_kernels/torch/mhc.py")
ARMS = ("R", "L", "M", "ML", "ML-O", "ML-S", "ML-C", "M-O")


def digest(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def generator(seed, name):
    g = torch.Generator()
    g.manual_seed(
        int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
        % (2**63)
    )
    return g


def config(depth, layerrope, vocab=50257):
    c = LlamaConfig(
        vocab_size=vocab,
        hidden_size=128,
        intermediate_size=352,
        num_hidden_layers=depth,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        initializer_range=0.02,
        max_position_embeddings=256,
        pad_token_id=None,
        bos_token_id=50256,
        eos_token_id=50256,
        tie_word_embeddings=False,
        use_cache=False,
    )
    c.layerrope = layerrope
    c.peri_norm = False
    c.layerrope_alpha_init = c.layerrope_alpha_rot_init = 0.0
    c.layerrope_beta_init = c.layerrope_beta_rot_init = -0.5
    c.layerrope_base_freq = 100.0
    c.logits_fp32 = False
    return c


@torch.no_grad()
def pair_weights(model, seed):
    """Same author distributions, named random tensors independent of RNG order."""
    receipt = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            continue
        std = math.sqrt(2 / 5) if isinstance(module, nn.Embedding) else 0.02
        t = (
            torch.randn(
                module.weight.shape, generator=generator(seed, name + ".weight")
            )
            * std
        )
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            t[module.padding_idx].zero_()
        module.weight.copy_(t)
        receipt[name + ".weight"] = digest(t)
    return receipt


class Routing(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(24, 512))
        self.norm_weight = nn.Parameter(torch.ones(512))
        self.scale = nn.Parameter(torch.full((3,), 0.01))
        pre = torch.full((4,), -1.0)
        pre[index % 4] = 1.0
        carry = torch.full((4, 4), -1.0)
        carry.fill_diagonal_(1.0)
        self.base = nn.Parameter(torch.cat((pre, torch.zeros(4), carry.flatten())))

    def forward(self, state):
        with torch.autocast(device_type=state.device.type, enabled=False):
            mixes = mhc.mhc_pre_norm_fn_ref(state, self.phi, self.norm_weight, 1e-5)
            pre, post, logits = mhc.mhc_pre_split_mixes_ref(
                mixes, self.scale, self.base, 4, 2.0, 0.0
            )
            carry = mhc.sinkhorn_normalize_ref(logits, repeat=20, eps=1e-6)
        return pre, post, carry, logits


class AngularSchedules(nn.Module):
    def __init__(self):
        super().__init__()
        for site in ("out_attn", "out_mlp"):
            setattr(self, site + "_alpha_rot", nn.Parameter(torch.tensor(0.0)))
            setattr(self, site + "_beta_rot", nn.Parameter(torch.tensor(-0.5)))


def write_code(post, schedule, site, code_index, rho=0.5):
    alpha = getattr(schedule, site + "_alpha_rot")
    beta = getattr(schedule, site + "_beta_rot")
    phi = (alpha + beta * math.log(code_index + 1.0)).exp()
    code = torch.stack((phi.cos(), phi.sin(), -phi.cos(), -phi.sin()))
    base = post.squeeze(-1)
    slack = torch.cat((base, 2 - base), dim=-1).amin(-1, keepdim=True)
    return (base + rho * slack * code).unsqueeze(-1)


class Model(nn.Module):
    def __init__(self, arm, depth, seed, vocab=50257, rho=0.5):
        super().__init__()
        assert arm in ARMS
        self.arm, self.depth, self.seed, self.rho = arm, depth, seed, rho
        layerrope = arm in ("L", "ML", "ML-O", "ML-S", "ML-C")
        self.lm = author.LlamaForCausalLM(config(depth, layerrope, vocab))
        self.common_hashes = pair_weights(self.lm, seed)
        self.routes = (
            nn.ModuleList([Routing(i) for i in range(2 * depth)])
            if arm.startswith("M")
            else None
        )
        self.angular = AngularSchedules() if arm == "M-O" else None
        assignment = []
        for site in ("out_attn", "out_mlp"):
            if arm == "ML-S":
                g = generator(seed, site + ":depth_shuffle")
                perm = torch.randperm(depth, generator=g)
                while torch.equal(perm, torch.arange(depth)):
                    perm = torch.randperm(depth, generator=g)
            elif arm == "ML-C":
                perm = torch.full((depth,), (depth - 1) // 2, dtype=torch.long)
            else:
                perm = torch.arange(depth)
            assignment.append(perm.tolist())
        self.assignment = assignment
        self.capture = False
        self.samples = []
        self.invariants = None

    def track(self, state, pre, post, carry):
        with torch.no_grad():
            v = torch.stack(
                (
                    (carry.sum(-1) - 1).abs().amax(),
                    (carry.sum(-2) - 1).abs().amax(),
                    (-carry).amax().clamp_min(0),
                    torch.maximum(-pre, pre - 1).amax().clamp_min(0),
                    torch.maximum(-post, post - 2).amax().clamp_min(0),
                    (~torch.isfinite(state)).any().float(),
                    (~torch.isfinite(carry)).any().float(),
                )
            )
            v = torch.nan_to_num(v, nan=float("inf"))
            self.invariants = (
                v if self.invariants is None else torch.maximum(self.invariants, v)
            )

    def forward(self, ids):
        self.samples = []
        self.invariants = None
        h = self.lm.model.embed_tokens(ids).bfloat16()
        if self.routes is None:
            return self.lm(inputs_embeds=h, use_cache=False).logits
        h = mhc.expand_to_mhc_ref(h, 4)
        positions = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        for l, block in enumerate(self.lm.model.layers):
            for site_i, site in enumerate(("attn", "mlp")):
                r = self.routes[2 * l + site_i]
                pre, base_post, carry, logits = r(h)
                post = base_post
                if self.arm in ("ML-O", "ML-S", "ML-C", "M-O"):
                    schedule = (
                        self.angular
                        if self.angular is not None
                        else self.lm.model.layerrope
                    )
                    post = write_code(
                        post,
                        schedule,
                        "out_" + site,
                        self.assignment[site_i][l],
                        self.rho,
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
                        h, branch, base_post, post, carry, logits, 2 * l + site_i
                    )
                h = updated
        h = self.lm.model.norm(h.float().mean(-2).bfloat16())
        return self.lm.lm_head(h)

    @torch.no_grad()
    def capture_sample(self, h, branch, base, post, carry, logits, index):
        n = h.shape[0] * h.shape[1]
        fixed = torch.linspace(0, n - 1, 16, device=h.device).long()
        errors = torch.maximum(
            (carry.sum(-1) - 1).abs().amax(-1), (carry.sum(-2) - 1).abs().amax(-1)
        ).flatten()
        worst = errors.argmax().reshape(1)
        ix = torch.cat((fixed, worst))
        mean = h.float().mean(-2, keepdim=True)
        self.samples.append(
            {
                "sublayer": index,
                "indices": ix.cpu(),
                "logits": logits.flatten(0, 1)[ix].cpu(),
                "carry": carry.flatten(0, 1)[ix].cpu(),
                "mean_energy": mean.square().mean().item(),
                "difference_energy": (h.float() - mean).square().mean().item(),
                "write_rms": (post * branch.float().unsqueeze(-2))
                .square()
                .mean()
                .sqrt()
                .item(),
                "write_l2_change": ((post - base) * branch.float().unsqueeze(-2))
                .norm()
                .item(),
            }
        )

    def identities(self):
        return {
            "arm": self.arm,
            "depth": self.depth,
            "seed": self.seed,
            "parameters": sum(p.numel() for p in self.parameters()),
            "common_weights": self.common_hashes,
            "code_assignment": self.assignment,
            "routing_initialization": (
                [{k: digest(v) for k, v in r.state_dict().items()} for r in self.routes]
                if self.routes
                else []
            ),
            "residual_dtype": "BF16; stream mean in FP32 then BF16",
            "precision": "FP32 master/state/norm/gains/routing/Sinkhorn/CE; BF16 block matrix ops",
        }


def loss(logits, targets):
    """Targets are shifted by the loader exactly once, never by the author LM."""
    return nn.functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
