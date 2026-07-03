#!/usr/bin/env python3
"""Orchestrate a code-reason run: manifest -> prompt -> agent loop -> artifacts.

For every manifest row, every model the row targets, and every reasoning mode
the config enables, this builds the task prompt from the Ticket-3 templates,
runs the execution-free agent loop against the row's repository, and writes
the full artifact tree (prompt, transcript, result, and a certificate in
semi-formal mode) through the ArtifactWriter. The default client is the
offline MockModelClient so a run needs no network or key; pass a real
client_factory for live models. Nothing here decides policy -- modes, models,
and the execution guard all come from the config flags carried in the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "datasets"))

from artifact_writer import ArtifactWriter  # noqa: E402
from agent_loop import run_agent  # noqa: E402
from tool_router import ToolRouter, tool_specs  # noqa: E402
from model_client import MockModelClient, mock_default_script  # noqa: E402

_PROMPTS = os.path.join(_ROOT, "prompts")

STEM = {
    "patch_equiv": "patch_equiv",
    "fault_localization": "fault_loc",
    "code_qa": "code_qa",
}

MODE_FLAGS = [
    ("CONFIG_CODE_REASON_MODE_STANDARD", "standard"),
    ("CONFIG_CODE_REASON_MODE_SEMIFORMAL", "semiformal"),
]


def enabled_modes(flags):
    modes = [m for f, m in MODE_FLAGS if flags.get(f) in (True, "y")]
    return modes or ["standard"]


def _placeholders(row):
    p = row.get("payload", {})
    repo_root = row.get("repo", ".")
    tt = row["task_type"]
    if tt == "patch_equiv":
        return {
            "repo_root": repo_root,
            "patch_a": p.get("patch_reference", ""),
            "patch_b": p.get("patch_candidate", ""),
        }
    if tt == "fault_localization":
        return {
            "repo_root": repo_root,
            "bug_report": p.get("symptom") or p.get("question", ""),
        }
    return {"repo_root": repo_root, "question": p.get("question", "")}


def build_prompt(row, mode, prompts_dir=_PROMPTS, execution_free=True):
    stem = STEM[row["task_type"]]
    with open(os.path.join(prompts_dir, f"{stem}_{mode}.md")) as fh:
        body = fh.read()
    fields = _placeholders(row)
    for key, val in fields.items():
        body = body.replace("{" + key + "}", str(val))
    if execution_free:
        with open(os.path.join(prompts_dir, "_shared_constraints.md")) as fh:
            body = fh.read() + "\n\n" + body
    return body


def _default_factory(model_id, task_type, mode):
    return MockModelClient(mock_default_script(task_type, mode), model_id=model_id)


def run_manifest(
    manifest_rows,
    flags,
    run_dir,
    dataset_dir,
    config_path=None,
    client_factory=_default_factory,
    repo_root=".",
    max_steps=8,
):
    writer = ArtifactWriter(run_dir, config_path=config_path, repo_root=repo_root)
    modes = enabled_modes(flags)
    execution_free = flags.get("CONFIG_CODE_REASON_NO_TARGET_REPO_EXECUTION", True) in (
        True,
        "y",
    )
    results = []
    for row in manifest_rows:
        task_id, tt = row["task_id"], row["task_type"]
        repo_path = os.path.join(dataset_dir, row.get("repo", "repo"))
        for model in row["model_ids"]:
            for mode in modes:
                prompt = build_prompt(row, mode, execution_free=execution_free)
                writer.write_prompt(task_id, model, mode, prompt)
                router = ToolRouter.for_repo(repo_path, flags)
                client = client_factory(model, tt, mode)
                tools = tool_specs(semiformal=(mode == "semiformal"))
                out = run_agent(
                    task_id,
                    model,
                    mode,
                    prompt,
                    client,
                    router,
                    tools,
                    writer=writer,
                    max_steps=max_steps,
                )
                result = {
                    "task_id": task_id,
                    "task_type": tt,
                    "model": model,
                    "mode": mode,
                    "answer": out["answer"],
                    "correct": None,
                    "score": None,
                }
                if out["certificate"]:
                    cert = dict(out["certificate"])
                    cert.setdefault("task_id", task_id)
                    cert.setdefault("model", model)
                    cert.setdefault("mode", mode)
                    cert.setdefault("task_type", tt)
                    writer.write_certificate(task_id, model, mode, cert)
                    result["certificate_ref"] = "certificate.json"
                writer.write_result(task_id, model, mode, result)
                writer.write_costs(task_id, model, mode, out["tokens"])
                writer.append_manifest(
                    {"task_id": task_id, "task_type": tt, "dataset": row["dataset"]}
                )
                results.append(result)
    return results


def _self_test():
    import tempfile

    sys.path.insert(0, os.path.join(_ROOT, "datasets"))
    from manifest_builder import load_dataset, build_manifest

    ds_dir = os.path.join(_ROOT, "datasets", "fixtures", "smoke")
    ds = load_dataset(ds_dir)
    flags = {
        "CONFIG_CODE_REASON_TASK_PATCH_EQUIV": True,
        "CONFIG_CODE_REASON_TASK_FAULT_LOCALIZATION": True,
        "CONFIG_CODE_REASON_TASK_CODE_QA": True,
        "CONFIG_CODE_REASON_MODEL_OPUS_4_5": True,
        "CONFIG_CODE_REASON_MODE_STANDARD": True,
        "CONFIG_CODE_REASON_MODE_SEMIFORMAL": True,
    }
    rows = build_manifest(ds, flags)
    run_dir = tempfile.mkdtemp()
    results = run_manifest(rows, flags, run_dir, ds_dir, repo_root=_ROOT)
    # 4 tasks x 1 model x 2 modes = 8 results
    assert len(results) == 8, len(results)
    # every task dir has prompt + transcript + answer; semiformal has a cert
    tid = "smoke-qa-clamp"
    base = os.path.join(run_dir, "tasks", tid, "claude-opus-4-5")
    for mode in ("standard", "semiformal"):
        for f in ("prompt.md", "transcript.jsonl", "answer.json", "costs.json"):
            assert os.path.exists(os.path.join(base, mode, f)), (mode, f)
    assert os.path.exists(os.path.join(base, "semiformal", "certificate.json"))
    assert not os.path.exists(os.path.join(base, "standard", "certificate.json"))
    # the transcript actually recorded a real grep tool result from the repo
    tpath = os.path.join(base, "standard", "transcript.jsonl")
    steps = [json.loads(l) for l in open(tpath)]
    assert any(
        s.get("role") == "tool" and s.get("tool_result", {}).get("hits") for s in steps
    ), "no grep hits recorded -> tools not wired to the repo"
    print(
        f"[runner] self-test PASS: 8 results, artifact tree + live tool calls in {run_dir}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--dataset-dir")
    ap.add_argument("--config")
    ap.add_argument("--run-dir")
    ap.add_argument("--max-steps", type=int, default=8)
    args = ap.parse_args()
    if args.self_test:
        _self_test()
        return
    if not (args.manifest and args.dataset_dir and args.run_dir):
        ap.error("--manifest, --dataset-dir, --run-dir required (or --self-test)")
    flags = {}
    if args.config and os.path.exists(args.config):
        flags = json.load(open(args.config)).get("flags", {})
    rows = [json.loads(l) for l in open(args.manifest)]
    results = run_manifest(
        rows,
        flags,
        args.run_dir,
        args.dataset_dir,
        config_path=args.config,
        max_steps=args.max_steps,
    )
    print(f"[runner] {len(results)} results -> {args.run_dir}")


if __name__ == "__main__":
    main()
