# SPDX-License-Identifier: MIT
"""MNIST LayerNorm adaptation and bounded, shared-phase mHC write codes."""

import torch
from torch import nn

from research.subliminal_learning.data import generator, tensor_digest
from research.subliminal_learning.models import ResidualFamily, init_model

ARMS = ("R", "L", "M", "ML", "ML-O", "ML-S", "ML-C", "M-O")


def phase(schedule, depth):
    opts = {"device": schedule.alpha_rot.device, "dtype": schedule.alpha_rot.dtype}
    logs = torch.arange(1, depth + 1, **opts).log()
    return (schedule.alpha_rot[:, None] + schedule.beta_rot[:, None] * logs).exp()


def code_vectors(angles):
    c, s = angles.cos(), angles.sin()
    return torch.stack((c, s, -c, -s), -1)


def coded_write(beta, code, rho=0.5):
    """amin uses PyTorch's equal split of the subgradient at tied minima."""
    slack = torch.cat((beta, 2 - beta), -1).amin(-1, keepdim=True)
    return beta + rho * slack * code, rho * slack


class AngularSchedule(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.alpha_rot = nn.Parameter(torch.zeros(models))
        self.beta_rot = nn.Parameter(torch.full((models,), -0.5))


class SupervisedModel(ResidualFamily):
    def __init__(self, arm, depth, keys, iters=20, width=256, rho=0.5):
        if arm not in ARMS or depth < 1 or width % 2:
            raise ValueError("invalid supervised architecture")
        if arm == "ML-S" and depth < 2:
            raise ValueError("shuffled assignment requires at least two blocks")
        lr = arm in ("L", "ML", "ML-O", "ML-S", "ML-C")
        mhc = arm.startswith("M")
        super().__init__(
            len(keys),
            "mhc" if mhc else "residual_mlp",
            10,
            width,
            depth,
            4,
            1e-5,
            sinkhorn_iters=iters,
            mhc_init_logit=1.0,
            routing={"sinkhorn_log": True},
            layerrope=(
                {"base": 100.0, "mag_slope": 0.0, "rot_slope": -0.5} if lr else None
            ),
        )
        self.arm, self.depth, self.rho, self.width = arm, depth, rho, width
        self.channel_schedule = AngularSchedule(len(keys)) if arm == "M-O" else None
        assignment = []
        for key in keys:
            if arm == "ML-S":
                g = generator(key, "depth_code_order")
                perm = torch.randperm(depth, generator=g)
                while torch.equal(perm, torch.arange(depth)):
                    perm = torch.randperm(depth, generator=g)
            elif arm == "ML-C":
                perm = torch.full((depth,), (depth - 1) // 2, dtype=torch.long)
            else:
                perm = torch.arange(depth)
            assignment.append(perm)
        self.register_buffer("code_assignment", torch.stack(assignment))
        self.monitor = True
        self.capture = False
        self.terms = []
        self.invariants = None
        init_model(self, keys, "supervised_init_common")

    @property
    def angular_schedule(self):
        # Return the actual module; never copy or detach shared scalars.
        return self.layerrope if self.layerrope is not None else self.channel_schedule

    def channel_codes(self):
        if self.arm not in ("ML-O", "ML-S", "ML-C", "M-O"):
            return None
        angles = phase(self.angular_schedule, self.depth)
        return code_vectors(angles.gather(1, self.code_assignment))

    def clear_invariants(self):
        self.invariants = None

    def track(self, values):
        if not self.monitor:
            return
        values = torch.nan_to_num(values.detach(), nan=float("inf"))
        self.invariants = (
            values
            if self.invariants is None
            else torch.maximum(self.invariants, values)
        )

    def forward(self, x):
        h = self.stem(x)
        gains = self.layerrope() if self.layerrope is not None else None
        codes = self.channel_codes()
        self.terms = []
        if self.conns is not None:
            h = h.unsqueeze(2).expand(*h.shape[:2], 4, h.shape[-1])
        for l, block in enumerate(self.blocks):
            gain = gains[:, l] if gains is not None else None
            if self.conns is None:
                inp, carry, change = h, h, torch.zeros_like(h)
                out = block(inp, gain)
                write = out
                updated = carry + write
                pre = beta = base = matrix = strength = None
            else:
                pre, base, matrix = self.conns[l].maps(h)
                beta, strength = (
                    coded_write(base, codes[:, l, None], self.rho)
                    if codes is not None
                    else (base, torch.zeros_like(base[..., :1]))
                )
                inp = torch.einsum("mbn,mbnd->mbd", pre, h)
                out = block(inp, gain)
                carry = torch.einsum("mbij,mbjd->mbid", matrix, h)
                change = carry - h
                write = beta[..., None] * out[:, :, None]
                updated = carry + write
                if self.monitor:
                    self.track(
                        torch.stack(
                            (
                                (matrix.sum(-1) - 1).abs().amax(),
                                (matrix.sum(-2) - 1).abs().amax(),
                                torch.maximum(
                                    (-matrix).amax(), (matrix - 1).amax()
                                ).clamp_min(0),
                                torch.maximum(
                                    (-pre).amax(), (pre - 1).amax()
                                ).clamp_min(0),
                                torch.maximum(
                                    (-beta).amax(), (beta - 2).amax()
                                ).clamp_min(0),
                                (~torch.isfinite(updated)).any().to(h.dtype),
                            )
                        )
                    )
            if self.capture:
                self.terms.append(
                    {
                        k: v.detach() if v is not None else None
                        for k, v in dict(
                            state=h,
                            block_input=inp,
                            block_output=out,
                            carried=carry,
                            carry_change=change,
                            write=write,
                            updated=updated,
                            pre=pre,
                            beta=beta,
                            base_beta=base,
                            matrix=matrix,
                            code_strength=strength,
                        ).items()
                    }
                )
            h = updated
        if self.conns is not None:
            h = h.mean(2)
        return self.head(self.final_norm(h))

    def identities(self):
        return {
            "common_weights": {
                str(i): tensor_digest(l.weight)
                for i, l in enumerate(self.common_linears())
            },
            "code_assignment": self.code_assignment.cpu().tolist(),
            "code_assignment_hash": tensor_digest(self.code_assignment),
            "parameters": sum(p.numel() for p in self.parameters()),
            "angular_shared": self.channel_schedule is None
            and self.arm in ("ML-O", "ML-S", "ML-C"),
        }
