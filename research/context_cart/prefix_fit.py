# SPDX-License-Identifier: MIT
"""Bounded three-arm comparison on the diagnostic worlds.

Generated only: one writer forward pass. Direct fit: 100 batch-one updates of
a residual MLP added to a fixed source-independent prefix. Generated plus
refinement: the same MLP added to the generated prefix. Both optimisation
arms use the same embedding and hidden-layer initialisation per world and the
same fixed order of 32 fit questions; the eight evaluation questions never
enter the optimizer.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import dataset as D
from . import evaluate as E
from . import model as M
from . import prompts as P
from . import runs as R


@dataclasses.dataclass
class FitConfig:
    updates: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    embed_width: int = 128
    hidden_width: int = 128
    max_new_tokens: int = 16
    eval_batch: int = 8


class ResidualPrefix(nn.Module):
    """P = P0 + MLP(E); zero final projection makes the initial prefix P0."""

    def __init__(
        self,
        rows: int,
        geometry: M.Geometry,
        cfg: FitConfig,
        generator: torch.Generator,
    ):
        super().__init__()
        out = 2 * geometry.layers * geometry.kv_heads * geometry.head_dim
        self.geometry = geometry
        self.embed = nn.Parameter(
            torch.randn(rows, cfg.embed_width, generator=generator) * 0.02
        )
        self.hidden = nn.Linear(cfg.embed_width, cfg.hidden_width)
        with torch.no_grad():
            self.hidden.weight.copy_(
                torch.randn(cfg.hidden_width, cfg.embed_width, generator=generator)
                * (1.0 / cfg.embed_width**0.5)
            )
            self.hidden.bias.zero_()
        self.out = nn.Linear(cfg.hidden_width, out)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def delta(self) -> torch.Tensor:
        g = self.geometry
        values = self.out(torch.tanh(self.hidden(self.embed)))
        return values.view(self.embed.shape[0], 2, g.layers, g.kv_heads, g.head_dim)

    def prefix(
        self, base: list[tuple[torch.Tensor, torch.Tensor]], dtype: torch.dtype
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        delta = self.delta()
        out = []
        for layer, (k0, v0) in enumerate(base):
            dk = delta[:, 0, layer].permute(1, 0, 2).unsqueeze(0)
            dv = delta[:, 1, layer].permute(1, 0, 2).unsqueeze(0)
            out.append(((k0.float() + dk).to(dtype), (v0.float() + dv).to(dtype)))
        return out


@torch.no_grad()
def generic_prefix(
    reader: M.Reader, token_ids: list[int], first_position: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Reader key/value rows for fixed ordinary tokens at the memory positions,
    with no source: the source-independent base for direct fitting."""
    ids = torch.tensor([token_ids], dtype=torch.long, device=reader.device)
    _, cache = reader.forward(None, ids, torch.ones_like(ids), first_position)
    return [
        (k.clone(), v.clone())
        for k, v in (M.cache_layer(cache, i) for i in range(reader.geometry.layers))
    ]


def fit_world(
    reader: M.Reader,
    tokenizer: Any,
    base: list[tuple[torch.Tensor, torch.Tensor]],
    fit_items: list[E.Item],
    first_position: int,
    cfg: FitConfig,
    seed: int,
    dtype: torch.dtype,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    module = ResidualPrefix(
        int(base[0][0].shape[2]), reader.geometry, cfg, generator
    ).to(reader.device)
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    examples = [
        (
            P.prompt_ids(tokenizer, P.user_message("learned", item.question)),
            P.answer_ids(tokenizer, item.answer),
        )
        for item in fit_items
    ]
    losses: list[float] = []
    grad_checks: list[dict[str, float]] = []
    sync = torch.cuda.synchronize if reader.device.type == "cuda" else (lambda: None)

    @torch.no_grad()
    def panel_loss() -> float:
        """Mean answer-token loss over the complete fit panel."""
        total, count = 0.0, 0
        prefix = module.prefix(base, dtype)
        for prompt, answer in examples:
            ids, mask, labels = M.right_pad_with_labels(
                [prompt], [answer], reader.pad_id, reader.device
            )
            loss_sum, n, _ = reader.answer_loss(
                prefix, ids, mask, labels, first_position
            )
            total += float(loss_sum.item())
            count += n
        return total / max(count, 1)

    panel_before = panel_loss()
    sync()
    t0 = time.perf_counter()
    for update in range(cfg.updates):
        prompt, answer = examples[update % len(examples)]
        ids, mask, labels = M.right_pad_with_labels(
            [prompt], [answer], reader.pad_id, reader.device
        )
        prefix = module.prefix(base, dtype)
        loss_sum, count, _ = reader.answer_loss(
            prefix, ids, mask, labels, first_position
        )
        loss = loss_sum / count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if update < 3:
            grad_checks.append(
                {
                    "update": update + 1,
                    "out_weight_grad_norm": float(module.out.weight.grad.norm().item()),
                    "hidden_weight_grad_norm": (
                        float(module.hidden.weight.grad.norm().item())
                        if module.hidden.weight.grad is not None
                        else 0.0
                    ),
                    "embed_grad_norm": (
                        float(module.embed.grad.norm().item())
                        if module.embed.grad is not None
                        else 0.0
                    ),
                }
            )
        torch.nn.utils.clip_grad_norm_(module.parameters(), cfg.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
    sync()
    seconds = time.perf_counter() - t0
    panel_after = panel_loss()
    with torch.no_grad():
        final = [
            (k.detach().clone(), v.detach().clone())
            for k, v in module.prefix(base, dtype)
        ]
    finite = all(math.isfinite(l) for l in losses) and all(
        math.isfinite(v) for v in (panel_before, panel_after)
    )
    gradients = grad_checks[0]["out_weight_grad_norm"] > 0 and any(
        c["embed_grad_norm"] > 0 for c in grad_checks[1:]
    )
    # the pre/post comparison uses the same complete panel, not the loss of
    # whichever question happened to come first and last
    qualified = gradients and finite and panel_after < panel_before
    record = {
        "updates": cfg.updates,
        "losses": losses,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_10": sum(losses[-10:]) / min(10, len(losses)),
        "panel_loss_before": panel_before,
        "panel_loss_after": panel_after,
        "grad_checks": grad_checks,
        "finite": bool(finite),
        "qualified": bool(qualified),
        "seconds": seconds,
        "trainable_parameters": int(sum(p.numel() for p in module.parameters())),
    }
    return final, record


def run(
    writer: M.Writer,
    reader: M.Reader,
    tokenizer: Any,
    worlds: list[dict[str, Any]],
    cfg: FitConfig,
    out_dir: Path,
    data_seed: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slots = int(writer.memory.shape[0])
    source_tokens = len(worlds[0]["source_ids"])
    first_position = source_tokens + slots
    base_generic = generic_prefix(reader, writer.init_rows, source_tokens)
    arms = {"generated": [], "direct": [], "refined": []}
    per_world: dict[str, Any] = {}
    bytes_per_prefix = (
        M.Geometry.from_config(reader.model.config).bytes_per_position() * slots
    )
    for world in worlds:
        wid = world["world_id"]
        uncovered = D.uncovered_support(world["fit_questions"], world["questions"])
        if uncovered:
            raise ValueError(
                f"{wid}: evaluation depends on records the fit questions "
                f"never supervise: {uncovered}"
            )
        fit_items = E.items_from_worlds([world], key="fit_questions")
        eval_items = E.items_from_worlds([world], key="questions")
        sync = (
            torch.cuda.synchronize if reader.device.type == "cuda" else (lambda: None)
        )
        with torch.no_grad():
            sync()
            t0 = time.perf_counter()
            source = torch.tensor(
                [world["source_ids"]], dtype=torch.long, device=reader.device
            )
            generated, length = writer.encode(source)
            generated = [(k.detach().clone(), v.detach().clone()) for k, v in generated]
            sync()
            gen_seconds = time.perf_counter() - t0
        seed = D.stable_seed(data_seed, "prefix-fit", wid) % (2**31)
        direct, direct_record = fit_world(
            reader, tokenizer, base_generic, fit_items, first_position, cfg, seed, dtype
        )
        refined, refined_record = fit_world(
            reader, tokenizer, generated, fit_items, first_position, cfg, seed, dtype
        )
        fit_panel = {}
        for arm, prefix in (
            ("generic", base_generic),
            ("direct", direct),
            ("generated", generated),
            ("refined", refined),
        ):
            rows_fit, _, _ = E.answer_memory_arm(
                reader,
                tokenizer,
                arm,
                {wid: prefix},
                fit_items,
                first_position,
                cfg.eval_batch,
                cfg.max_new_tokens,
            )
            fit_panel[arm] = E.metrics(rows_fit)["em"]
        results = {"fit_panel_em": fit_panel}
        for arm, prefix, build_seconds in (
            ("generated", generated, gen_seconds),
            ("direct", direct, direct_record["seconds"]),
            ("refined", refined, refined_record["seconds"] + gen_seconds),
        ):
            rows, timing, _ = E.answer_memory_arm(
                reader,
                tokenizer,
                arm,
                {wid: prefix},
                eval_items,
                first_position,
                cfg.eval_batch,
                cfg.max_new_tokens,
            )
            arms[arm].extend(rows)
            results[arm] = {
                "em": E.metrics(rows)["em"],
                "build_seconds": build_seconds,
                "answer_seconds": timing.prefill_seconds + timing.decode_seconds,
                "prefix_bytes": bytes_per_prefix,
            }
        results["direct_fit"] = direct_record
        results["refined_fit"] = refined_record
        per_world[wid] = results
        print(
            f"{wid}: generated {results['generated']['em']:.3f} direct {results['direct']['em']:.3f} refined {results['refined']['em']:.3f}",
            flush=True,
        )
    for arm, rows in arms.items():
        E.write_rows(out_dir / f"{arm}_predictions.jsonl", rows)
    summary = {
        "config": dataclasses.asdict(cfg),
        "worlds": len(worlds),
        "answers_per_arm": len(arms["generated"]),
        "metrics": {arm: E.metrics(rows) for arm, rows in arms.items()},
        "direct_qualified_worlds": sum(
            1 for r in per_world.values() if r["direct_fit"]["qualified"]
        ),
        "refined_qualified_worlds": sum(
            1 for r in per_world.values() if r["refined_fit"]["qualified"]
        ),
        "total_updates": 2 * cfg.updates * len(worlds),
        "construction_seconds": {
            arm: sum(r[arm]["build_seconds"] for r in per_world.values())
            for arm in ("generated", "direct", "refined")
        },
        "per_world": per_world,
        "generic_prefix_rows": writer.init_rows,
        "peak_memory": R.peak_memory(),
    }
    R.dump_json(out_dir / "fit_summary.json", summary)
    return summary
