# SPDX-License-Identifier: MIT
"""Measure digit accuracy and residual/weight ranges on retained MNIST states."""

import contextlib
from pathlib import Path

import torch
from torch.nn import functional as F

from research.layerrope_beta.measure import describe
from research.mo_variants.stage_c_common import read, sha, write
from research.subliminal_learning.models import build, MLinear
from research.subliminal_learning.recipe import Recipe
from research.subliminal_learning.run import load_data, state_digest
from research.tinystories_depth.float_quant import scaled_float
from research.tinystories_depth.kv_cache import fakequant


def policies():
    rows = []
    for bits in (4, 8):
        for scope in ("blocks", "all"):
            for integer in (False, True):
                rows.append(
                    dict(
                        name=f"{scope}-{'INT' if integer else 'FP'}{bits}-weights",
                        bits=bits,
                        scope=scope,
                        integer=integer,
                    )
                )
        for weights in (False, True):
            rows.append(
                dict(
                    name=(
                        f"block-W{bits}A{bits}" if weights else f"block-inputs-FP{bits}"
                    ),
                    bits=bits,
                    scope="blocks",
                    activations=True,
                    weights=weights,
                )
            )
    return rows


def matrices(model, scope):
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MLinear)
        and (scope == "all" or name.startswith("blocks."))
    ]


@torch.no_grad()
def evaluate(model, x, y, batch=1024):
    models = model.stem.weight.shape[0]
    correct = torch.zeros(models, device=x.device)
    ce = torch.zeros(models, device=x.device)
    for start in range(0, len(x), batch):
        labels = y[start : start + batch]
        logits = model(x[start : start + batch])[..., :10].float()
        correct += (logits.argmax(-1) == labels).sum(-1)
        targets = labels.expand(models, -1)
        ce += (
            F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
            .reshape(models, -1)
            .sum(-1)
        )
    return dict(
        accuracy=(correct / len(x)).cpu().tolist(),
        ce=(ce / len(x)).cpu().tolist(),
        examples=len(x),
    )


@contextlib.contextmanager
def policy(model, bits, scope="all", integer=False, activations=False, weights=True):
    originals, handles = [], []
    quant = lambda x: (
        fakequant(x, bits, "symmetric") if integer else scaled_float(x, bits)
    )
    try:
        with torch.no_grad():
            if weights:
                for _, module in matrices(model, scope):
                    originals.append((module, module.weight.detach().clone()))
                    module.weight.copy_(quant(module.weight))
        if activations:

            def hook(module, args):
                return (quant(args[0]),) + args[1:]

            for block in model.blocks:
                for module in block.modules():
                    if isinstance(module, MLinear):
                        handles.append(module.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for module, original in originals:
                module.weight.copy_(original)


@torch.no_grad()
def statistics(model, x):
    rows, handles = [], []
    before = model(x)

    def observe(site, layer, state):
        for m in range(state.shape[0]):
            row = dict(model=m, site=site, layer=layer, per_stream=describe(state[m]))
            if state.ndim == 4:
                row["stream_mean"] = describe(state[m].mean(-2))
            rows.append(row)

    if model.conns is None:
        for layer, block in enumerate(model.blocks):
            handles.append(
                block.register_forward_pre_hook(
                    lambda module, args, layer=layer: observe(
                        "residual_input", layer, args[0]
                    )
                )
            )
    else:
        for layer, conn in enumerate(model.conns):
            handles.append(
                conn.register_forward_pre_hook(
                    lambda module, args, layer=layer: observe(
                        "residual_input", layer, args[0]
                    )
                )
            )
        handles.append(
            model.conns[-1].register_forward_hook(
                lambda module, args, out: observe(
                    "final_streams", len(model.blocks) - 1, out
                )
            )
        )
    handles.append(
        model.final_norm.register_forward_pre_hook(
            lambda module, args: observe("final_mean", len(model.blocks) - 1, args[0])
        )
    )
    try:
        after = model(x)
        if not torch.equal(before, after):
            raise RuntimeError("MNIST observation changed logits")
    finally:
        for handle in handles:
            handle.remove()
    return dict(
        residuals=rows,
        example_count=len(x),
        observer_logits_identical=True,
        weights={
            name: [describe(w) for w in module.weight]
            for name, module in matrices(model, "all")
        },
        gains={
            site: dict(
                state={
                    name: value.detach().cpu().tolist()
                    for name, value in gain.state_dict().items()
                },
                by_layer=gain().cpu().tolist(),
            )
            for site, gain in (
                ("input", model.layerrope),
                ("output", model.output_layerrope),
            )
            if gain is not None
        },
    )


@torch.no_grad()
def quantization(model, x, y, batch=1024):
    before = state_digest(model)
    clean = evaluate(model, x, y, batch)
    cases = []
    model_count = model.stem.weight.shape[0]
    total = sum(p.numel() for p in model.parameters())
    for row in policies():
        options = {k: v for k, v in row.items() if k != "name"}
        with policy(model, **options):
            value = evaluate(model, x, y, batch)
        case = dict(**row, **value)
        case["accuracy_damage_pp"] = [
            100 * (a - b) for a, b in zip(clean["accuracy"], case["accuracy"])
        ]
        case["delta_ce"] = [a - b for a, b in zip(case["ce"], clean["ce"])]
        if row.get("weights", True):
            selected = matrices(model, row["scope"])
            count = sum(m.weight.numel() for _, m in selected)
            scales = sum(m.weight.numel() // m.weight.shape[-1] for _, m in selected)
            case["weight_parameter_fraction"] = count / total
            case["ideal_weight_bytes_per_model"] = (
                count * row["bits"] / 8 + scales * 4 + (total - count) * 4
            ) / model_count
        else:
            case["weight_parameter_fraction"] = 0.0
        cases.append(case)
        if state_digest(model) != before:
            raise RuntimeError("MNIST quantization did not restore the checkpoint")
    return dict(clean=clean, cases=cases, weights_unchanged=True)


def run(root, name):
    root = Path(root)
    plan = read(root / "PLAN.json")
    recipe = Recipe(**plan["mnist"]["recipes"][name])
    recipe.device = "cuda"
    device = torch.device("cuda")
    _, _, test_x, test_y = load_data(recipe, device)
    folder = root / "mnist/runs" / name
    manifest = read(folder / "manifest.json")
    results = []
    for role in ("teacher", "aux_shared", "full_shared"):
        entries = [v for k, v in manifest.items() if k.endswith("_" + role + ".pt")]
        if len(entries) != 1:
            raise ValueError("Expected one ten-model checkpoint pack for " + role)
        entry = entries[0]
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("MNIST checkpoint changed")
        model = build(recipe, 10).to(device).eval().requires_grad_(False)
        model.load_state_dict(
            torch.load(entry["path"], map_location=device, weights_only=False)
        )
        before = state_digest(model)
        stats = statistics(model, test_x[:256])
        quantized = quantization(model, test_x, test_y)
        if state_digest(model) != before:
            raise RuntimeError("MNIST quantization did not restore the checkpoint")
        results.append(dict(role=role, checkpoint=entry, statistics=stats, **quantized))
        write(
            folder / "QUANTIZATION.json",
            dict(status="running", seed_order=recipe.model_keys, roles=results),
        )
        del model
        torch.cuda.empty_cache()
    write(
        folder / "QUANTIZATION.json",
        dict(
            status="complete",
            seed_order=recipe.model_keys,
            roles=results,
            scope="Digit accuracy and CE on all 10000 MNIST test images; FP32 inference; no calibration or retraining; same weight/activation coverage definitions as the story study",
        ),
    )
