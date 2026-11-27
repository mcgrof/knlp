import argparse
import json

from research.kv_translate.swat import recipe


def test_short_recipe_preserves_matched_evaluation(tmp_path):
    parent = tmp_path / "scripts" / "exp1"
    parent.mkdir(parents=True)
    json_path = parent / "config.json"
    json_path.write_text(
        json.dumps(
            {
                "train_common": {"args": {}},
                "stage1": {"args": {}},
                "stage2": {"args": {"datasets": ["wiki", "{traces}"]}},
                "eval": {},
            }
        )
    )
    args = argparse.Namespace(
        sender=tmp_path / "sender",
        receiver=tmp_path / "receiver",
        seed=44,
        reconstruction_steps=30,
        task_steps=20,
        context=512,
        eval_context=1024,
        samples=16,
        tasks=["squad", "hellaswag"],
    )
    config = recipe(tmp_path, args)
    assert config["train_common"]["args"]["max-length"] == 512
    assert config["stage1"]["args"]["max-steps"] == 30
    assert config["stage1"]["args"]["save-every"] == -1
    assert config["stage2"]["args"]["datasets"][-1].endswith("gold_solutions.jsonl")
    squad, hellaswag = config["eval"]["benchmarks"]
    assert squad["args"]["tasks"] == "squad+terse:validation"
    assert "vanilla_args" not in squad
    assert hellaswag["renderer_args"]["renderer-decode-prefix"] == "generation"
