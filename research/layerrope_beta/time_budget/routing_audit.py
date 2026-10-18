# SPDX-License-Identifier: MIT
"""Replay captured router logits without changing the executed finite operator."""

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import time

import numpy as np
import torch

from research.layerrope_beta.model import Model
from research.layerrope_beta.time_budget.routing import RoutingObservation
from research.layerrope_beta.time_budget.state import synchronize
from research.layerrope_beta.time_budget.state import json_hash
from research.layerrope_beta.time_budget.train import source_identity
from research.mo_variants.stage_c_common import read, sha, write
from research.tinystories_depth.model import mhc
from research.tinystories_depth.train import state_hash
from research.tinystories_depth.kv_cache import evaluate

REPEATS = (20, 40, 80, 160, 320)
EPS = 1e-6
BOUND = 1e-4


def normalize(logits, repeats, with_previous=False):
    """Same softmax, epsilon placement, axes and final column step as upstream."""
    if repeats not in REPEATS:
        raise ValueError("Undeclared finite solver")
    value = logits.softmax(-1) + EPS
    value = value / (value.sum(-2, keepdim=True) + EPS)
    previous = value
    for _ in range(repeats - 1):
        previous = value
        value = value / (value.sum(-1, keepdim=True) + EPS)
        value = value / (value.sum(-2, keepdim=True) + EPS)
    return (value, previous) if with_previous else value


def errors(value):
    return torch.stack(
        ((value.sum(-1) - 1).abs().amax(-1), (value.sum(-2) - 1).abs().amax(-1)), -1
    ).reshape(-1, 2)


def distribution(values, bound=BOUND):
    values = values.detach().double().flatten()
    finite = torch.isfinite(values)
    if not finite.all().item():
        raise FloatingPointError("Nonfinite diagnostic distribution")
    return dict(
        count=values.numel(),
        maximum=values.max().item(),
        percentiles=dict(
            zip(
                ("p50", "p90", "p99"),
                torch.quantile(values, values.new_tensor([0.5, 0.9, 0.99])).tolist(),
            )
        ),
        above_bound=int((values > bound).sum().item()),
        fraction_above_bound=(values > bound).double().mean().item(),
    )


@contextmanager
def solver(repeats):
    """One worker owns this scoped substitution; restore the upstream default."""
    if repeats not in REPEATS:
        raise ValueError("Undeclared finite solver")
    original = mhc.sinkhorn_normalize_ref

    def chosen(logits, repeat=20, eps=EPS):
        if repeat != 20 or eps != EPS or logits.dtype != torch.float32:
            raise ValueError("The declared FP32 routing recipe changed")
        return normalize(logits, repeats)

    mhc.sinkhorn_normalize_ref = chosen
    try:
        yield
    finally:
        mhc.sinkhorn_normalize_ref = original


class DetailedObservation(RoutingObservation):
    """Equal-cadence distributions for observed matrices, without projection repair."""

    def __init__(self):
        super().__init__()
        self.residuals, self.activation = [], []

    @torch.no_grad()
    def add(self, state, pre, post, carry):
        super().add(state, pre, post, carry)
        self.residuals.append(errors(carry))
        self.activation.append(
            torch.stack(
                (state.float().square().mean().sqrt(), state.float().abs().amax())
            )
        )

    def report(self, expected_calls, **kwargs):
        value = super().report(expected_calls, **kwargs)
        if self.residuals and value["finite"]:
            residuals = torch.cat(self.residuals)
            activation = torch.stack(self.activation)
            value.update(
                row_distribution=distribution(residuals[:, 0]),
                column_distribution=distribution(residuals[:, 1]),
                maximum_observed_state_rms=activation[:, 0].max().item(),
                maximum_observed_state_abs=activation[:, 1].max().item(),
            )
        return value


@torch.no_grad()
def capture(model, ids):
    """Observe every site while preserving its inputs, arithmetic and outputs."""
    model.eval()
    before = state_hash(model.state_dict())
    with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16):
        baseline = model(ids)
    sites, handles = [], []
    observation = DetailedObservation()

    def hook(index):
        def observe(module, args, output):
            pre, post, carry, logits = output
            if index != len(sites):
                raise RuntimeError("Unexpected routing-site order or reuse")
            if logits.dtype != torch.float32 or carry.dtype != torch.float32:
                raise ValueError("Executed routing is not FP32")
            sites.append(
                dict(
                    site=index,
                    logits=logits.detach().clone(),
                    matrix=carry.detach().clone(),
                    pre=pre.detach().clone(),
                    post=post.detach().clone(),
                )
            )

        return observe

    try:
        model.routing_observation = observation
        for index, route in enumerate(model.routes):
            handles.append(route.register_forward_hook(hook(index)))
        with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16):
            observed = model(ids)
    finally:
        model.routing_observation = None
        for handle in handles:
            handle.remove()
    if not torch.equal(baseline, observed) or before != state_hash(model.state_dict()):
        raise RuntimeError("Capture changed logits or parameters")
    if len(sites) != 2 * model.depth:
        raise RuntimeError("Incomplete routing capture")
    return sites, dict(
        logits_equal=True,
        parameters_unchanged=True,
        sites=len(sites),
        batch_size=ids.shape[0],
        window=ids.shape[1],
        original_routing=observation.report(2 * model.depth),
    )


def sample(sites, count=16):
    """Select deterministic typical and original-worst IDs once for all recipes."""
    saved, gradients = [], []
    for site in sites:
        z, matrix = (site[k].reshape(-1, 4, 4) for k in ("logits", "matrix"))
        typical = torch.linspace(
            0, len(z) - 1, min(count, len(z)), device=z.device
        ).long()
        worst = errors(matrix).amax(-1).argsort(descending=True, stable=True)[:count]
        selected = torch.cat((typical, worst)).unique(sorted=True)
        saved.append(
            dict(
                site=site["site"],
                dtype=str(z.dtype),
                typical_indices=typical.cpu(),
                worst_indices=worst.cpu(),
                indices=selected.cpu(),
                logits=z[selected].cpu(),
                matrix=matrix[selected].cpu(),
                pre=site["pre"].reshape(len(z), 4)[selected].cpu(),
                post=site["post"].reshape(len(z), 4)[selected].cpu(),
                token_ids=dict(
                    window_index=(selected // site["logits"].shape[1]).cpu(),
                    token_position=(selected % site["logits"].shape[1]).cpu(),
                ),
            )
        )
        gradients.append(z[torch.stack((typical[0], worst[0]))])
    return saved, torch.cat(gradients)


def gradient_check(logits, repeats, seed=0):
    """Compare VJPs for the same finite recipe, never different operators."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cot = torch.randn(logits.shape, generator=generator).to(logits.device)
    gradients = []
    for dtype in (torch.float32, torch.float64):
        z = logits.detach().to(dtype).requires_grad_()
        value = normalize(z, repeats)
        gradient = torch.autograd.grad((value * cot.to(dtype)).sum(), z)[0]
        if not torch.isfinite(gradient).all().item():
            raise FloatingPointError("Nonfinite solver VJP")
        gradients.append(gradient.detach().double())
    delta = (gradients[0] - gradients[1]).flatten(1).norm(dim=-1)
    scale = gradients[1].flatten(1).norm(dim=-1)
    passes = delta <= 1e-6 + 1e-3 * scale
    # A directional finite difference on typical and worst sampled inputs.
    z = logits.detach()[[0, 1]].double().requires_grad_()
    c = cot[[0, 1]].double()
    direction = torch.randn(z.shape, generator=generator).to(z.device).double()
    direction /= direction.norm()
    exact = (
        (torch.autograd.grad((normalize(z, repeats) * c).sum(), z)[0] * direction)
        .sum()
        .item()
    )
    differences = []
    for epsilon in (1e-4, 1e-5):
        estimate = (
            (
                (
                    normalize(z.detach() + epsilon * direction, repeats)
                    - normalize(z.detach() - epsilon * direction, repeats)
                )
                * c
            ).sum()
            / (2 * epsilon)
        ).item()
        ok = abs(estimate - exact) <= 1e-8 + 1e-3 * abs(exact)
        differences.append(
            dict(epsilon=epsilon, autograd=exact, estimate=estimate, passed=ok)
        )
    return dict(
        finite=True,
        same_recipe_vjp_pass=bool(passes.all().item()),
        maximum_absolute_l2_error=delta.max().item(),
        maximum_reference_l2=scale.max().item(),
        failed_matrices=int((~passes).sum().item()),
        finite_differences=differences,
        passed=bool(passes.all().item()) and all(x["passed"] for x in differences),
    )


def replay(sites, gradient_inputs, seed=0):
    """Use identical captured FP32 logits for every finite solver calculation."""
    z = torch.cat([s["logits"].reshape(-1, 4, 4) for s in sites])
    original = torch.cat([s["matrix"].reshape(-1, 4, 4) for s in sites])
    if not torch.equal(normalize(z, 20), original):
        raise RuntimeError(
            "Independent finite recipe differs from original 20-step output"
        )
    rows, fp32_values = [], {}
    for dtype in (torch.float32, torch.float64):
        previous = None
        for repeats in REPEATS:
            synchronize(z.device)
            started = time.monotonic()
            with torch.no_grad():
                value, last_iterate = normalize(
                    z.to(dtype), repeats, with_previous=True
                )
            synchronize(z.device)
            seconds = time.monotonic() - started
            if dtype == torch.float32:
                fp32_values[repeats] = value
            residuals = errors(value)
            row, column = (distribution(residuals[:, i]) for i in (0, 1))
            gradients = (
                gradient_check(gradient_inputs, repeats, seed)
                if dtype == torch.float32
                else None
            )
            rows.append(
                dict(
                    repeats=repeats,
                    dtype=str(dtype),
                    seconds=seconds,
                    matrices=len(z),
                    row=row,
                    column=column,
                    minimum_entry=value.min().item(),
                    finite=True,
                    maximum_last_iteration_change=(value - last_iterate)
                    .abs()
                    .max()
                    .item(),
                    same_count_fp32_maximum_output_difference=(
                        value.double() - fp32_values[repeats].double()
                    )
                    .abs()
                    .max()
                    .item(),
                    maximum_change_from_previous_repeat=(
                        (value - previous).abs().max().item()
                        if previous is not None
                        else None
                    ),
                    gradient=gradients,
                    constraint_pass=row["above_bound"] == column["above_bound"] == 0,
                    at_diagnostic_cap=repeats == REPEATS[-1],
                )
            )
            previous = value
    return rows


def load_model(receipt, device):
    if sha(receipt["path"]) != receipt["sha256"]:
        raise ValueError("Changed checkpoint")
    state = torch.load(receipt["path"], map_location="cpu", weights_only=False)
    if (
        json_hash(state["identity"]) != receipt["identity_sha256"]
        or state["identity"]["job"]["arm"] != receipt["row"]
        or state["identity"]["job"]["seed"] != receipt["seed"]
    ):
        raise ValueError("Checkpoint recipe differs")
    if state_hash(state["model"]) != receipt["model_sha256"]:
        raise ValueError("Changed model payload")
    if (
        state["step"] != receipt["step"]
        or state["training_seconds"] != receipt["training_seconds"]
    ):
        raise ValueError("Checkpoint clock or step differs")
    model = Model(
        receipt["row"],
        receipt.get("depth", 32),
        receipt["seed"],
        beta=0.0,
        vocab=receipt.get("vocab", 50257),
    ).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def validation_parity(actual, expected):
    """Bind replay to saved same-shape losses as well as immutable state bytes."""
    a, b = (np.asarray(v["window_ce"], dtype=np.float64) for v in (actual, expected))
    if (
        a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Original validation coverage or finiteness differs")
    maximum = float(np.abs(a - b).max())
    if (
        maximum > 1e-6
        or not math.isfinite(actual["ce"])
        or abs(actual["ce"] - expected["ce"]) > 1e-6
    ):
        raise ValueError("Original checkpoint validation replay differs")
    return dict(
        actual_ce=actual["ce"],
        recorded_ce=expected["ce"],
        maximum_window_ce_difference=maximum,
        absolute_tolerance=1e-6,
        passed=True,
    )


def run(inputs, out, device="cuda"):
    card = read(inputs)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if card["source"] != source_identity() or not card["executed_forward_source"]:
        raise ValueError("Frozen audit source differs")
    if len(card["checkpoints"]) != 6 or {
        (r["row"], r["target_seconds"]) for r in card["checkpoints"]
    } != {(row, t) for row in ("M", "ML") for t in (0.0, 200.0, 787.0)}:
        raise ValueError(
            "The initial replay requires the six declared qualification states"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for path, expected in card["executed_forward_source"].items():
        if sha(path) != expected:
            raise ValueError("Executed forward source differs: " + path)
    if sha(card["validation"]["path"]) != card["validation"]["sha256"]:
        raise ValueError("Validation changed")
    tokens = np.load(card["validation"]["path"], mmap_mode="r")
    if tokens.dtype != np.uint16 or len(tokens) != 32769:
        raise ValueError("The declared validation capability is required")
    window, batch = card.get("window", 256), card.get("batch_size", 32)
    if (window, batch) != (256, 32):
        raise ValueError("The declared diagnostic shape is 32 by 256")
    ids = torch.as_tensor(
        np.array(tokens[: batch * window], dtype=np.int64).reshape(batch, window),
        device=device,
    )
    records = []
    for index, receipt in enumerate(card["checkpoints"]):
        model = load_model(receipt, device)
        legacy = evaluate(model, tokens, dict(name="clean"), device, window, batch)
        legacy_parity = validation_parity(legacy, receipt["validation"])
        sites, parity = capture(model, ids)
        samples, gradients = sample(sites)
        sample_path = out / (str(index) + "-samples.pt")
        torch.save(dict(checkpoint=receipt, ids=ids.cpu(), sites=samples), sample_path)
        results = replay(sites, gradients, card["diagnostic_seed"] + index)
        record = dict(
            checkpoint=receipt,
            parity=parity,
            original_validation_replay=legacy_parity,
            comparisons=results,
            samples=dict(path=str(sample_path), sha256=sha(sample_path)),
        )
        write(out / (str(index) + "-replay.json"), record)
        records.append(record)
        del model, sites, samples, gradients
    passing = [
        n
        for n in REPEATS[1:]
        if all(
            next(
                x
                for x in r["comparisons"]
                if x["dtype"] == "torch.float32" and x["repeats"] == n
            )["constraint_pass"]
            and next(
                x
                for x in r["comparisons"]
                if x["dtype"] == "torch.float32" and x["repeats"] == n
            )["gradient"]["passed"]
            for r in records
        )
    ]
    candidate = min(passing) if passing else None
    sensitivity = []
    if candidate is not None:
        for receipt in card["checkpoints"]:
            model = load_model(receipt, device)
            before = state_hash(model.state_dict())
            with torch.no_grad(), torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ):
                original_logits = model(ids)
            baseline = evaluate(
                model, tokens, dict(name="clean"), device, window, batch
            )
            observation = DetailedObservation()
            model.routing_observation = observation
            with solver(candidate):
                with torch.no_grad(), torch.autocast(
                    device_type=device, dtype=torch.bfloat16
                ):
                    changed_logits = model(ids)
                changed = evaluate(
                    model, tokens, dict(name="clean"), device, window, batch
                )
            model.routing_observation = None
            routing = observation.report(
                2 * model.depth * (1 + math.ceil(((len(tokens) - 1) // window) / batch))
            )
            delta = changed_logits.float() - original_logits.float()
            if before != state_hash(model.state_dict()):
                raise RuntimeError("Sensitivity evaluation changed model parameters")
            sensitivity.append(
                dict(
                    checkpoint=receipt,
                    original_ce=baseline["ce"],
                    candidate_ce=changed["ce"],
                    delta_ce=changed["ce"] - baseline["ce"],
                    first_batch_maximum_logit_change=delta.abs().max().item(),
                    first_batch_rms_logit_change=delta.square().mean().sqrt().item(),
                    routing=routing,
                )
            )
            del model, original_logits, changed_logits, delta
    qualified = candidate is not None and all(
        x["routing"]["constraint_pass"] for x in sensitivity
    )
    result = dict(
        status="candidate_qualified" if qualified else "stop",
        candidate_repeats=candidate,
        input_sha256=sha(inputs),
        records=records,
        sensitivity=sensitivity,
        policy="FP64 is a same-input numerical reference; select the least passing FP32 repeat count; finite capped calculations are not exact projections",
        final_holdout_access=False,
    )
    write(out / "AUDIT.json", result)
    return result


def smoke(out, device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = Model("M", 2, 7, vocab=97).to(device)
    ids = torch.arange(16, device=device).reshape(2, 8)
    sites, parity = capture(model, ids)
    _, gradients = sample(sites, 2)
    result = dict(
        parity=parity, comparisons=replay(sites, gradients), scientific_results=False
    )
    write(out, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke(args.out, args.device)
    else:
        run(args.inputs, args.out, args.device)
