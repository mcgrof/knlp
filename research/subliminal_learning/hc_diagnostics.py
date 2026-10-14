# SPDX-License-Identifier: MIT
"""Detached Hyper-Connection mechanism and optimizer-step diagnostics."""

import torch

from .data import tensor_digest
from .run import state_digest

QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def _summary(value):
    flat = value.detach().reshape(value.shape[0], -1).double()
    return {
        "elements": int(flat.shape[1]),
        "mean": flat.mean(1).cpu().tolist(),
        "rms": flat.square().mean(1).sqrt().cpu().tolist(),
        "l2": torch.linalg.vector_norm(flat, dim=1).cpu().tolist(),
        "max_abs": flat.abs().amax(1).cpu().tolist(),
        "nonzero": (flat != 0).sum(1).cpu().tolist(),
        "finite": torch.isfinite(flat).all(1).cpu().tolist(),
    }


def _quantiles(value):
    flat = value.detach().reshape(value.shape[0], -1).double()
    values = torch.quantile(
        flat, torch.tensor(QUANTILES, dtype=flat.dtype, device=flat.device), dim=1
    ).transpose(0, 1)
    return {
        "levels": list(QUANTILES),
        "values": values.cpu().tolist(),
    }


def _relative_l2(value, reference):
    left = value.detach().reshape(value.shape[0], -1).double()
    right = reference.detach().reshape(reference.shape[0], -1).double()
    numerator = torch.linalg.vector_norm(left, dim=1)
    denominator = torch.linalg.vector_norm(right, dim=1)
    return [
        None if float(d) == 0 else float(n / d)
        for n, d in zip(numerator.cpu(), denominator.cpu())
    ]


def _alignment(left, right):
    a = left.detach().reshape(left.shape[0], -1).double()
    b = right.detach().reshape(right.shape[0], -1).double()
    dot = (a * b).sum(1)
    na = torch.linalg.vector_norm(a, dim=1)
    nb = torch.linalg.vector_norm(b, dim=1)
    cosine = []
    for value, x, y in zip(dot.cpu(), na.cpu(), nb.cpu()):
        cosine.append(
            None if float(x) == 0 or float(y) == 0 else float(value / (x * y))
        )
    return {
        "inner_product": dot.cpu().tolist(),
        "cosine": cosine,
        "undefined": [value is None for value in cosine],
    }


def _stream_structure(value):
    x = (
        value.detach()
        .double()
        .permute(0, 2, 1, 3)
        .reshape(value.shape[0], value.shape[2], -1)
    )
    centred = x - x.mean(-1, keepdim=True)
    covariance = torch.bmm(centred, centred.transpose(1, 2)) / max(1, x.shape[-1] - 1)
    diagonal = torch.diagonal(covariance, dim1=1, dim2=2)
    scale = torch.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
    correlation = torch.where(
        scale > 0, covariance / scale, torch.zeros_like(covariance)
    )
    eigenvalues = torch.linalg.eigvalsh(covariance)
    n = value.shape[2]
    mask = ~torch.eye(n, dtype=torch.bool, device=value.device)
    off_diagonal = correlation[:, mask]
    return {
        "mean_variance_across_streams": value.detach()
        .double()
        .var(dim=2, unbiased=False)
        .mean(dim=(1, 2))
        .cpu()
        .tolist(),
        "covariance": covariance.cpu().tolist(),
        "correlation": correlation.cpu().tolist(),
        "mean_off_diagonal_correlation": off_diagonal.mean(1).cpu().tolist(),
        "covariance_eigenvalues_ascending": eigenvalues.cpu().tolist(),
    }


def _carry_matrix_stats(matrix):
    p = matrix.detach().double()
    n = p.shape[-1]
    identity = torch.eye(n, dtype=p.dtype, device=p.device)
    delta = p - identity
    frobenius = torch.linalg.matrix_norm(delta, ord="fro", dim=(-2, -1))
    singular = torch.linalg.svdvals(p)
    levels = torch.tensor((0.1, 0.5, 0.9), dtype=p.dtype, device=p.device)
    singular_quantiles = torch.quantile(singular, levels, dim=1).permute(1, 2, 0)
    return {
        "p_minus_identity_frobenius_mean": frobenius.mean(1).cpu().tolist(),
        "p_minus_identity_frobenius_rms": frobenius.square()
        .mean(1)
        .sqrt()
        .cpu()
        .tolist(),
        "p_singular_values_mean": singular.mean(1).cpu().tolist(),
        "p_singular_value_quantile_levels": [0.1, 0.5, 0.9],
        "p_singular_value_quantiles": singular_quantiles.cpu().tolist(),
    }


def _parameter_summary(value):
    return _summary(value.detach())


def _layer_diagnostics(terms, conn, block):
    h = terms["h"]
    carry = terms["carry"]
    write = terms["write"]
    combined = terms["combined"]
    updated = terms["updated"]
    h_mean = h.mean(2)
    carry_mean = carry.mean(2)
    write_mean = write.mean(2)
    combined_mean = combined.mean(2)
    return {
        "state_terms": {
            "h": _summary(h),
            "carry": {
                **_summary(carry),
                "l2_relative_to_h": _relative_l2(carry, h),
            },
            "write": {
                **_summary(write),
                "l2_relative_to_h": _relative_l2(write, h),
            },
            "carry_plus_write": {
                **_summary(combined),
                "l2_relative_to_h": _relative_l2(combined, h),
            },
            "updated": _summary(updated),
        },
        "stream_mean_terms": {
            "h": _summary(h_mean),
            "carry": {
                **_summary(carry_mean),
                "l2_relative_to_h_mean": _relative_l2(carry_mean, h_mean),
            },
            "write": {
                **_summary(write_mean),
                "l2_relative_to_h_mean": _relative_l2(write_mean, h_mean),
            },
            "carry_plus_write": {
                **_summary(combined_mean),
                "l2_relative_to_h_mean": _relative_l2(combined_mean, h_mean),
            },
            "carry_write_alignment": _alignment(carry_mean, write_mean),
        },
        "full_carry_write_alignment": _alignment(carry, write),
        "block_input": _summary(terms["block_input"]),
        "block_output": _summary(terms["block_output"]),
        "beta": {
            "unscaled": _quantiles(terms["beta_unscaled"]),
            "scaled": _quantiles(terms["beta_scaled"]),
        },
        "read": {
            "mean_direction": terms["read"].double().mean(1).cpu().tolist(),
            "norm": _summary(torch.linalg.vector_norm(terms["read"], dim=-1)),
            "component_quantiles": _quantiles(terms["read"]),
        },
        "normalization_gains": {
            "connection_weight": _parameter_summary(conn.norm.weight),
            "connection_bias": _parameter_summary(conn.norm.bias),
            "block_weight": _parameter_summary(block.norm.weight),
            "block_bias": _parameter_summary(block.norm.bias),
        },
        "carry_matrix": _carry_matrix_stats(terms["carry_matrix"]),
        "stream_structure": {
            "before": _stream_structure(h),
            "after": _stream_structure(updated),
        },
    }


@torch.no_grad()
def collect_hc_probe(model, inputs, name):
    """Run one fixed probe and reduce all captured values to JSON-safe stats."""
    if getattr(model, "arch", None) != "hc":
        raise ValueError("HC diagnostics require arch=hc")
    rows = next(model.parameters()).shape[0]
    before = [state_digest(model, row) for row in range(rows)]
    model.set_hc_capture(True)
    try:
        logits = model(inputs)
        layers = [
            _layer_diagnostics(conn.last_terms, conn, block)
            for conn, block in zip(model.conns, model.blocks)
        ]
    finally:
        model.set_hc_capture(False)
    after = [state_digest(model, row) for row in range(rows)]
    if after != before:
        raise AssertionError("HC probe forward changed model state")
    return {
        "name": name,
        "examples": int(inputs.shape[-2]),
        "input_digest": (
            [tensor_digest(inputs[row].cpu()) for row in range(inputs.shape[0])]
            if inputs.dim() == 3
            else tensor_digest(inputs.cpu())
        ),
        "logits_digest": [tensor_digest(logits[row].cpu()) for row in range(rows)],
        "layers": layers,
        "final_normalization": {
            "weight": _parameter_summary(model.final_norm.weight),
            "bias": _parameter_summary(model.final_norm.bias),
        },
        "state_digest_before": before,
        "state_digest_after": after,
    }


def parameter_partition(model):
    """Disjoint HC parameter slices for gradient/update attribution."""
    parts = []
    for name, parameter in model.named_parameters():
        layer = None
        selector = "all"
        if name.startswith("stem."):
            group = "stem"
        elif name.startswith("head."):
            group = "head"
        elif name.startswith("final_norm."):
            group = "normalization"
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1])
            group = "normalization" if ".norm." in name else "block_weights"
        elif name.startswith("conns."):
            layer = int(name.split(".")[1])
            leaf = name.split(".", 2)[2]
            if leaf in ("static_alpha", "dynamic_alpha_fn"):
                for split_group, split_selector in (
                    ("read", "first_column"),
                    ("carry", "remaining_columns"),
                ):
                    parts.append(
                        {
                            "name": name,
                            "parameter": parameter,
                            "group": split_group,
                            "layer": layer,
                            "selector": split_selector,
                        }
                    )
                continue
            if leaf in ("dynamic_alpha_scale", "dynamic_beta_scale"):
                group = "routing_shared_scalars"
            elif leaf in ("static_beta", "dynamic_beta_fn"):
                group = "write_coefficients"
            elif leaf.startswith("norm."):
                group = "normalization"
            else:
                raise ValueError(f"unassigned HC parameter {name}")
        else:
            raise ValueError(f"unassigned model parameter {name}")
        parts.append(
            {
                "name": name,
                "parameter": parameter,
                "group": group,
                "layer": layer,
                "selector": selector,
            }
        )
    selected = sum(
        _select(part["parameter"], part["selector"]).numel() for part in parts
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    if selected != total:
        raise AssertionError(f"HC parameter partition covers {selected} != {total}")
    return parts


def _select(value, selector):
    if selector == "all":
        return value
    if selector == "first_column":
        return value[..., :1]
    if selector == "remaining_columns":
        return value[..., 1:]
    raise ValueError(selector)


def capture_step_before(parts):
    return {
        f"{part['name']}::{part['selector']}": _select(
            part["parameter"].detach(), part["selector"]
        ).clone()
        for part in parts
    }


def _parts_summary(parts, value):
    if not parts:
        return None
    rows = parts[0]["parameter"].shape[0]
    joined = [[] for _ in range(rows)]
    for part in parts:
        selected = value(part)
        if selected is None:
            selected = torch.zeros_like(_select(part["parameter"], part["selector"]))
        flat = selected.detach().reshape(rows, -1).double()
        for row in range(rows):
            joined[row].append(flat[row])
    vectors = [torch.cat(items) for items in joined]
    elements = int(vectors[0].numel())
    return {
        "elements": elements,
        "l2": [float(torch.linalg.vector_norm(item)) for item in vectors],
        "rms": [float(item.square().mean().sqrt()) for item in vectors],
        "max_abs": [float(item.abs().max()) for item in vectors],
        "nonzero": [int((item != 0).sum()) for item in vectors],
        "finite": [bool(torch.isfinite(item).all()) for item in vectors],
    }


def optimizer_step_diagnostics(model, parts, before, initial):
    """Gradient, actual Adam step, and cumulative W0 distance by disjoint group."""

    def values(kind, part):
        key = f"{part['name']}::{part['selector']}"
        current = _select(part["parameter"].detach(), part["selector"])
        if kind == "gradient":
            gradient = part["parameter"].grad
            return None if gradient is None else _select(gradient, part["selector"])
        if kind == "step":
            return current - before[key]
        if kind == "cumulative":
            start = _select(initial[part["name"]].to(current.device), part["selector"])
            return current - start
        raise ValueError(kind)

    groups = sorted({part["group"] for part in parts})
    out = {"groups": {}, "by_layer": {}}
    for group in groups:
        selected = [part for part in parts if part["group"] == group]
        out["groups"][group] = {
            kind: _parts_summary(selected, lambda part, kind=kind: values(kind, part))
            for kind in ("gradient", "step", "cumulative")
        }
    for layer in sorted({part["layer"] for part in parts if part["layer"] is not None}):
        out["by_layer"][str(layer)] = {}
        for group in groups:
            selected = [
                part
                for part in parts
                if part["layer"] == layer and part["group"] == group
            ]
            if selected:
                out["by_layer"][str(layer)][group] = {
                    kind: _parts_summary(
                        selected, lambda part, kind=kind: values(kind, part)
                    )
                    for kind in ("gradient", "step", "cumulative")
                }
    return out
