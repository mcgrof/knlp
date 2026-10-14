"""Exercise actual training and resume paths without a scientific verdict."""

import os
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from research.reproduction.run import write


def mnist(root, device, data_dir):
    from research.subliminal_learning.layerrope_study import ARMS, qualify
    from research.subliminal_learning.recipe import Recipe
    from research.subliminal_learning.run import cmd_run

    recipes = {
        name: Recipe(
            arch=arch,
            layerrope=rope,
            protocol="per_example_v1",
            student_loss_mode="exact_cancel_v1",
            seeds=[74001, 74002],
            width=16,
            blocks=2,
            batch_size=16,
            pack_size=2,
            teacher_epochs=1,
            student_epochs=1,
            noise_samples=64,
            heldout_noise=32,
            train_limit=128,
            test_limit=64,
            data_dir=data_dir,
            device=device,
            smoke=True,
            max_seconds=120,
            output_dir=str(root / "runs"),
            checkpoint_dir=str(root / "checkpoints"),
            run_name=name,
        )
        for name, (arch, rope) in ARMS.items()
    }
    checks = {name: qualify(recipe) for name, recipe in recipes.items()}
    if not all(q["all_pass"] for q in checks.values()):
        raise RuntimeError("MNIST qualification failed before smoke training")
    write(root / "QUALIFICATION.json", checks)
    for name, recipe in recipes.items():
        sources = {field: "smoke recipe" for field in asdict(recipe)}
        if cmd_run(recipe, sources, ["residual smoke", name]):
            raise RuntimeError("Incomplete MNIST smoke trajectory: " + name)
    write(
        root / "SMOKE.json",
        {
            "status": "complete",
            "device": device,
            "scope": "smoke",
            "scientific_confirmation_allowed": False,
            "recipes": {n: asdict(r) for n, r in recipes.items()},
        },
    )


def tinystories(root, device):
    from research.mo_variants.stage_c_common import ROWS
    from research.mo_variants.stage_c_fixtures import fixtures
    from research.mo_variants.model import Model
    from research.tinystories_depth.model import loss
    from research.tinystories_depth.probe import finite

    (root / "code").symlink_to(
        Path(__file__).resolve().parents[2], target_is_directory=True
    )
    (root / "records").mkdir()
    fixtures(root, "cpu")
    rows = []
    for row, variant, mapping in ROWS:
        model = Model(variant, 2, 76411, vocab=128, mapping=mapping).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.003,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0,
        )
        generator = torch.Generator().manual_seed(76411)
        tokens = torch.randint(0, 128, (2, 17), generator=generator).to(device)
        losses = []
        for step in range(1, 3):
            model.step = step
            optimizer.param_groups[0]["lr"] = 0.003 * step / 100
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                value = loss(model(tokens[:, :-1]), tokens[:, 1:])
            value.backward()
            finite(model)
            optimizer.step()
            finite(model, optimizer)
            losses.append(float(value.detach()))
        checkpoint = root / "checkpoints" / (row + ".pt")
        checkpoint.parent.mkdir(exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_cpu": torch.get_rng_state(),
                "rng_gpu": torch.cuda.get_rng_state() if device == "cuda" else None,
                "step": 2,
                "scope": "synthetic-token smoke",
                "row": row,
            },
            checkpoint,
        )
        rows.append({"row": row, "loss": losses, "checkpoint": str(checkpoint)})
    write(
        root / "SMOKE.json",
        {
            "status": "complete",
            "scope": "synthetic-token smoke",
            "device": device,
            "scientific_confirmation_allowed": False,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=("mnist", "tinystories"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--data-dir", default="~/.cache/knlp/mnist")
    args = parser.parse_args()
    torch.set_num_threads(min(2, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if args.kind == "mnist":
        mnist(args.root, args.device, args.data_dir)
    else:
        tinystories(args.root, args.device)
