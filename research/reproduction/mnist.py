"""Prepare historical and repaired MNIST controls from public recipes."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from research.subliminal_learning.recipe import Recipe, resolve
from research.subliminal_learning.run import cmd_run
from research.reproduction.run import REPO, write


def recipes(config, root):
    settings = dict(
        device=config["device"],
        data_dir=config["data_dir"],
        output_dir=str(root / "runs"),
        checkpoint_dir=config["checkpoint_dir"] or str(root / "checkpoints"),
        max_seconds=config["max_seconds"],
    )
    selected = []
    if config["study"] == "mnist":
        for name in [
            "subliminal-mnist-paper",
            *[
                "subliminal-mnist-" + arm + "-cohort20"
                for arm in ("residual", "hc", "mhc", "gr", "altup")
            ],
        ]:
            recipe, _ = resolve(str(REPO / "defconfigs" / name), [])
            selected.append(replace(recipe, **settings, run_name=name))
        for depth, epochs in ((8, 5), (8, 20), (16, 5), (16, 20), (32, 5)):
            for arch in ("residual_mlp", "noskip_mlp", "hc", "mhc", "gr", "altup"):
                selected.append(
                    Recipe(
                        **settings,
                        arch=arch,
                        protocol="per_example_v1",
                        student_loss_mode="legacy_kl_v1",
                        seeds=list(range(2101, 2121)),
                        blocks=depth,
                        student_epochs=epochs,
                        pack_size=5,
                        run_name=f"historical-{arch}-d{depth}-e{epochs}",
                    )
                )
        for scale in (1.0, 0.25):
            selected.append(
                Recipe(
                    **settings,
                    arch="residual_mlp",
                    protocol="per_example_v1",
                    student_loss_mode="exact_cancel_v1",
                    seeds=list(range(73001, 73006)),
                    residual_scale=scale,
                    student_epochs=80,
                    pack_size=0,
                    student_eval_epochs=[5, 20, 80],
                    run_name=f"repaired-residual-scale-{scale}",
                )
            )
        for scale in (1.0, 0.25):
            selected.append(
                Recipe(
                    **settings,
                    arch="hc",
                    protocol="per_example_v1",
                    student_loss_mode="exact_cancel_v1",
                    seeds=list(range(73001, 73006)),
                    post_scale=scale,
                    student_epochs=80,
                    pack_size=0,
                    student_eval_epochs=[5, 20, 80],
                    run_name=f"repaired-hc-post-{scale}",
                )
            )
    selected.append(
        Recipe(
            **settings,
            arch="residual_mlp",
            protocol="per_example_v1",
            student_loss_mode="exact_cancel_v1",
            target_scale=0.0,
            seeds=config["seeds"] or list(range(73001, 73006)),
            student_epochs=80,
            student_eval_epochs=[5, 20, 80],
            pack_size=0,
            run_name="repaired-shared-null",
        )
    )
    return selected


def run(root, config):
    cells = recipes(config, root)
    write(root / "MNIST-RECIPES.json", [asdict(r) for r in cells])
    for recipe in cells:
        sources = {field: "public study recipe" for field in asdict(recipe)}
        status = cmd_run(recipe, sources, ["mnist reproduction", recipe.run_name])
        if status:
            raise RuntimeError(f"Incomplete {recipe.run_name}; retained resume state")
    # The experiment runner writes every W0, teacher and student observation.
    write(
        root / "MNIST-STATUS.json",
        {
            "status": "complete",
            "runs": [r.run_name for r in cells],
            "loss_modes": {r.run_name: r.student_loss_mode for r in cells},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, json.loads(args.config_json.read_text()))
