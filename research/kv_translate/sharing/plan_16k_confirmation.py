#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0
"""Inventory outcome-blind, independent 16K confirmation source groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from . import build_workload, code_tasks, contexts

LENGTH = 16384


def json_lines(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def query_record(query):
    record = {
        "gold_or_test_fixture_id": query["record"],
        "suffix_tokens": len(query["suffix"]),
        "suffix_tokens_sha256": contexts.digest(query["suffix"]),
    }
    if "accepted" in query:
        record["accepted_answers"] = query["accepted"]
    return record


def describe(ctx, *, group, domain, dataset, extra=None):
    return {
        "source_group_id": group,
        "domain": domain,
        "length_bucket": LENGTH,
        "actual_prefix_tokens": len(ctx["prefix"]),
        "prefix_tokens_sha256": contexts.digest(ctx["prefix"]),
        "dataset": dataset,
        "dataset_record_id": [q["record"] for q in ctx["queries"]],
        "evidence_pieces": sum(1 for item in ctx["layout"] if item["evidence"]),
        "pieces": len(ctx["layout"]),
        "evidence_whole": not any(
            item["cut"] for item in ctx["layout"] if item["evidence"]
        ),
        "fixed_queries": [query_record(query) for query in ctx["queries"]],
        **(extra or {}),
    }


def document_candidates(args, encode, development):
    questions = []
    for path in args.documents:
        questions.extend(json_lines(path))
    questions = [question for question in questions if question.get("answerable", True)]
    pool = build_workload.DocumentPool(questions, encode, args.workload_seed)
    expected = {
        row["context_id"]: row for row in development if row["domain"] == "document"
    }
    replay = []
    for length, count in build_workload.DEVELOPMENT.items():
        for index in range(count):
            name = f"dev-doc-{length}-{index:02d}"
            ctx = pool.build(length, name)
            if ctx is None:
                raise RuntimeError(f"cannot replay frozen development context {name}")
            got = contexts.digest(ctx["prefix"])
            wanted = expected[name]["prefix_tokens_sha256"]
            if got != wanted:
                raise RuntimeError(f"development replay mismatch for {name}: {got} != {wanted}")
            replay.append(name)

    candidates = []
    while True:
        name = f"confirm-doc-16384-{len(candidates):04d}"
        ctx = pool.build(LENGTH, name)
        if ctx is None:
            break
        if not build_workload.plain(ctx["prefix"], len(encode(contexts.DOCUMENT_HEAD))):
            continue
        identity = "|".join(sorted(query["record"] for query in ctx["queries"]))
        group = "doc:" + hashlib.sha256(identity.encode()).hexdigest()[:20]
        candidates.append(describe(ctx, group=group, domain="document", dataset="musique"))
    return candidates, replay, len(questions)


def code_candidates(args, encode, development):
    tasks = code_tasks.load(args.tasks)
    verdicts = json_lines(args.eligibility)
    by_repository, dropped = build_workload.eligible_by_repository(
        tasks, verdicts, args.source, encode
    )
    repositories = {
        repo: repo_tasks
        for repo, repo_tasks in by_repository.items()
        if len(repo_tasks) >= 2
    }
    used = {
        row["repository"] for row in development if row["domain"] == "code"
    }
    candidates = []
    for repo in sorted(repositories):
        if repo in used:
            continue
        ctx = build_workload.build_code(
            repo,
            repositories[repo],
            args.source,
            LENGTH,
            encode,
            args.workload_seed,
        )
        if ctx is None:
            continue
        candidates.append(
            describe(
                ctx,
                group="repo:" + repo,
                domain="code",
                dataset="deveval",
                extra={"repository": repo},
            )
        )
    return candidates, used, len(repositories), dict(dropped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--development-records", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--workload-seed", default="20260928")
    parser.add_argument("--selection-seed", default="20260930|native-16k-confirmation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
        local_files_only=True,
    )

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    development = json_lines(args.development_records)
    documents, replay, question_count = document_candidates(args, encode, development)
    code, used_repositories, usable_repositories, dropped = code_candidates(
        args, encode, development
    )
    selected_per_domain = min(len(documents), len(code))

    def order(rows):
        return sorted(
            rows,
            key=lambda row: build_workload.order_key(
                args.selection_seed, row["source_group_id"]
            ),
        )

    receipt = {
        "schema": "kv_native_16k_eligibility_v1",
        "status": "OUTCOME_BLIND_INVENTORY",
        "length_tokens": LENGTH,
        "workload_seed": args.workload_seed,
        "selection_seed": args.selection_seed,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.revision,
        "inputs": {
            "documents": [
                {"path": os.fspath(path), "sha256": sha_file(path)}
                for path in args.documents
            ],
            "tasks": {"path": args.tasks, "sha256": sha_file(args.tasks)},
            "eligibility": {
                "path": args.eligibility,
                "sha256": sha_file(args.eligibility),
            },
            "development_records": {
                "path": args.development_records,
                "sha256": sha_file(args.development_records),
            },
        },
        "exclusions": {
            "development_contexts_replayed": replay,
            "development_code_repositories": sorted(used_repositories),
            "fit_groups": "FineWeb calibration is outside both evaluation datasets",
            "probe_groups": "unrelated probe data is outside both evaluation datasets",
            "rule": "no development repository or used document title can re-enter",
        },
        "inventory": {
            "document_questions_considered": question_count,
            "document_groups_16k": len(documents),
            "code_repositories_with_two_usable_tasks": usable_repositories,
            "code_groups_16k": len(code),
            "code_task_drops": dropped,
            "max_equal_domain_groups_each": selected_per_domain,
            "max_total_groups": 2 * selected_per_domain,
        },
        "candidates": {"document": documents, "code": code},
        "deterministic_selection_order": {
            "document": [row["source_group_id"] for row in order(documents)],
            "code": [row["source_group_id"] for row in order(code)],
        },
        "candidate_cohorts": {
            "maximum_precision": {
                "document_groups": len(documents),
                "code_groups": len(code),
                "domain_weights": {"document": 0.5, "code": 0.5},
            },
            "equal_count": {
                "groups_per_domain": selected_per_domain,
                "domain_weights": {"document": 0.5, "code": 0.5},
            },
        },
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
