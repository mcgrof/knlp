# SPDX-License-Identifier: MIT
"""Command-line driver for the context cartridge experiment.

Subcommands follow the experiment order: ``prepare`` (CPU dataset),
``evaluate`` (reader-only arms, also used for the calibration gate),
``train`` (profile, tiny-fit, pilot and confirmation seeds), ``export``
(cartridges plus the live-prefix replay reference), ``replay`` (fresh
reader-only consumer), ``compare-replay``, ``fit`` (bounded direct-learning
comparison) and ``report``. Every run writes its resolved configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import dataset as D
from . import runs as R

ROOT = Path(__file__).resolve().parents[2]


def _model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", default=None, help="model name; 'tiny' selects the CPU smoke model"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--memory-slots", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--source-revision",
        default=None,
        help="40-hex source revision when git is unavailable",
    )


def _spec(args: argparse.Namespace):
    from . import model as M

    values: dict[str, Any] = {}
    if args.model:
        values["name"] = args.model
    if args.revision:
        values["revision"] = args.revision
    if args.dtype:
        values["dtype"] = args.dtype
    if args.memory_slots:
        values["memory_slots"] = args.memory_slots
    if args.allow_download:
        values["local_files_only"] = False
    spec = M.ModelSpec(**values)
    if spec.is_tiny and spec.revision == M.DEFAULT_REVISION:
        spec = dataclasses.replace(
            spec, revision="0" * 40, dtype=values.get("dtype", "float32")
        )
    return spec


def _source_revision(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "source_revision", None):
        return {"revision": args.source_revision, "dirty": None, "source": "argument"}
    info = R.git_revision(ROOT)
    info["source"] = "git"
    return info


def cmd_prepare(args: argparse.Namespace) -> None:
    from . import model as M
    from . import prompts as P

    spec = _spec(args)
    tokenizer = M.load_tokenizer(spec)
    tok = D.Tokenizer(tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = dict(D.POOLS)
    for override in args.count or []:
        pool, value = override.split("=")
        counts[pool] = int(value)
    pools = args.pools.split(",") if args.pools else list(D.POOL_ORDER)
    registry = D.NameRegistry()
    manifest: dict[str, Any] = {
        "data_seed": D.DATA_SEED,
        "source_tokens": D.SOURCE_TOKENS,
        "tokenizer": {
            "name": spec.tokenizer_name or spec.name,
            "revision": spec.revision,
            "digest": M.tokenizer_digest(tokenizer),
        },
        "template_digest": P.template_digest(tokenizer),
        "generator_digest": R.file_digest(Path(__file__).with_name("dataset.py")),
        "prompts_digest": R.file_digest(Path(__file__).with_name("prompts.py")),
        "source": _source_revision(args),
        "pools": {},
        "scoring": "lowercase, strip, drop trailing punctuation and quotes, drop a leading attribute word (team/locker/badge), exact string match",
        "instruction": P.INSTRUCTION,
        "summary_instruction": P.SUMMARY_INSTRUCTION,
    }
    digests: set[str] = set()
    started = time.perf_counter()
    for pool in D.POOL_ORDER:
        if pool not in pools:
            continue
        t0 = time.perf_counter()
        worlds = D.generate_pool(pool, tok, registry, counts[pool])
        rows = [w.to_dict() for w in worlds]
        for w in rows:
            if w["source_digest"] in digests:
                raise RuntimeError(f"duplicate source digest in {pool}")
            digests.add(w["source_digest"])
        entry: dict[str, Any] = {"count": len(rows), "stats": D.pool_stats(worlds)}
        entry["worlds_digest"] = D.write_jsonl(out / f"worlds_{pool}.jsonl", rows)
        entry["questions_digest"] = D.write_jsonl(
            out / f"questions_{pool}.jsonl", D.questions_only(rows)
        )
        if pool == "diagnostic":
            entry["fit_questions_digest"] = D.write_jsonl(
                out / f"fit_questions_{pool}.jsonl",
                D.questions_only(rows, key="fit_questions"),
            )
        if pool in (
            "calibration",
            "tiny",
            "validation",
            "test_ordinary",
            "test_shifted",
        ):
            try:
                mapping = D.derangement(rows, pool)
                entry["wrong_mapping_strict"] = True
            except RuntimeError:
                if len(rows) >= 16:
                    raise
                mapping = D.derangement(rows, pool, strict=False)
                entry["wrong_mapping_strict"] = False
                print(
                    f"warning: {pool} wrong-cartridge mapping is not answer-disjoint (smoke pool)"
                )
            (out / f"wrong_{pool}.json").write_text(
                json.dumps(mapping, indent=1, sort_keys=True)
            )
            entry["wrong_mapping_digest"] = R.file_digest(out / f"wrong_{pool}.json")
        if pool == "validation":
            panel = D.validation_panel(rows, min(64, len(rows)))
            (out / "validation_panel.json").write_text(json.dumps(panel, indent=1))
            entry["validation_panel_digest"] = R.file_digest(
                out / "validation_panel.json"
            )
        entry["seconds"] = time.perf_counter() - t0
        manifest["pools"][pool] = entry
        print(
            f"{pool}: {len(rows)} worlds in {entry['seconds']:.1f}s, filler {entry['stats']['filler_fraction_mean']:.3f}",
            flush=True,
        )
    manifest["names_total"] = len(registry.names)
    manifest["codes_total"] = len(registry.codes)
    manifest["seconds"] = time.perf_counter() - started
    R.dump_json(out / "manifest.json", manifest)
    print(f"dataset written to {out} in {manifest['seconds']:.1f}s")


def _reader_only(spec, device: str):
    from . import model as M

    tokenizer = M.load_tokenizer(spec)
    base = M.load_base(spec, device)
    eos = tuple(
        int(i)
        for i in (
            [tokenizer.eos_token_id]
            if isinstance(tokenizer.eos_token_id, int)
            else tokenizer.eos_token_id
        )
    )
    if tokenizer.pad_token_id not in eos:
        eos = eos + (int(tokenizer.pad_token_id),)
    reader = M.Reader(base, eos_ids=eos, pad_id=int(tokenizer.pad_token_id))
    identity = M.reader_identity(base, tokenizer, spec)
    return tokenizer, reader, identity


def cmd_evaluate(args: argparse.Namespace) -> None:
    import torch

    from . import artifact as A
    from . import evaluate as E

    R.configure_threads()
    spec = _spec(args)
    dataset_dir = Path(args.dataset)
    worlds = D.read_jsonl(dataset_dir / f"worlds_{args.pool}.jsonl")
    if args.worlds:
        worlds = worlds[: args.worlds]
    items = E.items_from_worlds(worlds)
    if args.panel:
        panel = {
            tuple(p)
            for p in json.loads((dataset_dir / "validation_panel.json").read_text())
        }
        items = [i for i in items if (i.world_id, i.qid) in panel]
        keep = {i.world_id for i in items}
        worlds = [w for w in worlds if w["world_id"] in keep]
    out_dir = Path(args.run_dir) / "evaluations" / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, reader, identity = _reader_only(spec, args.device)
    record: dict[str, Any] = {
        "pool": args.pool,
        "worlds": len(worlds),
        "items": len(items),
        "batch": args.batch,
        "max_new_tokens": args.max_new,
        "identity": identity,
        "environment": R.environment(),
        "arms": {},
        "source": _source_revision(args),
    }
    R.reset_peak_memory()
    text_arms = [a for a in (args.text_arms or "").split(",") if a]
    for arm in text_arms:
        context = None
        extra: dict[str, Any] = {}
        if arm == "tail":
            context = {
                w["world_id"]: E.tail_text(tokenizer, w, args.tail_tokens)
                for w in worlds
            }
        if arm == "summary":
            summaries, stats, seconds = E.write_summaries(
                reader, tokenizer, worlds, args.batch, args.summary_max_new
            )
            context = summaries
            extra = {"summary_stats": stats, "summary_seconds": seconds}
            R.dump_json(
                out_dir / "summaries.json",
                {"summaries": summaries, "stats": stats, "seconds": seconds},
            )
        rows, timing = E.answer_text_arm(
            reader, tokenizer, arm, worlds, items, args.batch, args.max_new, context
        )
        E.write_rows(out_dir / f"{arm}_predictions.jsonl", rows)
        metrics = E.metrics(rows)
        metrics["timing"] = dataclasses.asdict(timing)
        metrics.update(extra)
        R.dump_json(out_dir / f"{arm}_metrics.json", metrics)
        record["arms"][arm] = metrics
        print(
            f"{args.pool} {arm}: EM {metrics['em']:.3f} macro {metrics['macro_em']:.3f} n={metrics['n']}",
            flush=True,
        )
    for kind, specs in (
        ("artifact", args.artifact_arm or []),
        ("wrong", args.wrong_arm or []),
    ):
        for entry in specs:
            name, directory = entry.split("=", 1)
            directory = Path(directory)
            mapping = None
            if kind == "wrong":
                mapping = json.loads(
                    (dataset_dir / f"wrong_{args.pool}.json").read_text()
                )
            needed = {
                mapping[w["world_id"]] if mapping else w["world_id"] for w in worlds
            }
            sync = (
                torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
            )
            sync()
            t0 = time.perf_counter()
            prefixes = {}
            first_position = None
            load_seconds = {"read": 0.0, "validate": 0.0, "copy": 0.0}
            for world_id in sorted(needed):
                loaded = A.load_prefix(directory / world_id, identity, args.device)
                prefixes[world_id] = loaded.prefix
                load_seconds["read"] += loaded.read_seconds
                load_seconds["validate"] += loaded.validate_seconds
                load_seconds["copy"] += loaded.copy_seconds
                if first_position is None:
                    first_position = loaded.first_position
                elif first_position != loaded.first_position:
                    raise RuntimeError(
                        "artifacts disagree on the first reader position"
                    )
            sync()
            load_total = time.perf_counter() - t0
            rows, timing, _ = E.answer_memory_arm(
                reader,
                tokenizer,
                name,
                prefixes,
                items,
                first_position,
                args.batch,
                args.max_new,
                mapping=mapping,
            )
            E.write_rows(out_dir / f"{name}_predictions.jsonl", rows)
            metrics = E.metrics(rows)
            metrics["timing"] = dataclasses.asdict(timing)
            metrics["timing"]["artifact_load_seconds"] = load_total
            metrics["timing"]["artifact_load_breakdown"] = load_seconds
            metrics["artifacts_dir"] = str(directory)
            metrics["first_position"] = first_position
            metrics["wrong_mapping"] = bool(mapping)
            R.dump_json(out_dir / f"{name}_metrics.json", metrics)
            record["arms"][name] = metrics
            print(
                f"{args.pool} {name}: EM {metrics['em']:.3f} macro {metrics['macro_em']:.3f} n={metrics['n']}",
                flush=True,
            )
            del prefixes
    record["peak_memory"] = R.peak_memory()
    R.dump_json(out_dir / f"evaluate_{int(time.time())}.json", record)


def cmd_train(args: argparse.Namespace) -> None:
    from . import train as T

    values = json.loads(Path(args.config).read_text()) if args.config else {}
    for override in args.set or []:
        key, raw = override.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if key.startswith("model."):
            values.setdefault("model", dict(dataclasses.asdict(T.M.ModelSpec())))[
                key[len("model.") :]
            ] = value
        else:
            values[key] = value
    if "validate_at" in values:
        values["validate_at"] = tuple(values["validate_at"])
    cfg = T.TrainConfig(**values)
    if cfg.spec().is_tiny and cfg.model.get("revision") == T.M.DEFAULT_REVISION:
        cfg.model["revision"] = "0" * 40
        cfg.model["dtype"] = (
            cfg.model.get("dtype", "float32")
            if cfg.model.get("dtype") != "bfloat16"
            else "float32"
        )
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    R.dump_json(
        run_dir / "SOURCE_LOCK.json",
        {
            "source": _source_revision(args),
            "dataset_manifest": R.load_json(Path(cfg.dataset_dir) / "manifest.json"),
            "environment": R.environment(),
        },
    )
    summary = T.train(cfg)
    print(
        json.dumps(
            {
                k: summary.get(k)
                for k in (
                    "selected_checkpoint",
                    "selected_macro_em",
                    "train_seconds",
                    "peak_memory",
                    "profile",
                )
            },
            indent=1,
            default=str,
        )
    )


def cmd_export(args: argparse.Namespace) -> None:
    import torch

    from . import replay as RP
    from . import train as T

    R.configure_threads()
    spec = _spec(args)
    tokenizer = T.M.load_tokenizer(spec)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    seed = int(payload["config"]["seed"])
    writer, reader, identity, _ = T.build_models(spec, args.device, seed, tokenizer)
    writer.load_trainable_state(payload["trainable"])
    writer.eval()
    worlds = D.read_jsonl(Path(args.dataset) / f"worlds_{args.pool}.jsonl")
    if args.worlds:
        worlds = worlds[: args.worlds]
    source = _source_revision(args)
    out = Path(args.out)
    t0 = time.perf_counter()
    index = RP.produce(
        writer,
        reader,
        tokenizer,
        identity,
        worlds,
        out,
        source["revision"],
        R.file_digest(Path(args.checkpoint)),
        args.namespace,
        args.max_new,
        reference=args.replay_reference,
    )
    index["checkpoint"] = {
        "path": str(args.checkpoint),
        "digest": R.file_digest(Path(args.checkpoint)),
        "update": payload.get("update"),
        "seed": seed,
    }
    index["seconds"] = time.perf_counter() - t0
    index["source"] = source
    index["pool"] = args.pool
    index["peak_memory"] = R.peak_memory()
    R.dump_json(out / "index.json", index)
    total = sum(a["payload_bytes"] for a in index["artifacts"].values())
    print(
        f"exported {len(index['artifacts'])} cartridges to {out} ({total / 2**20:.1f} MiB payload) in {index['seconds']:.1f}s"
    )


def cmd_replay(args: argparse.Namespace) -> None:
    from . import replay as RP

    lo, hi = (int(x) for x in args.range.split(":"))
    record = RP.consume(
        Path(args.artifacts),
        Path(args.questions),
        Path(args.out),
        _spec(args),
        (lo, hi),
        args.max_new,
        args.device,
    )
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "process_id",
                    "answers",
                    "model_load_seconds",
                    "seconds",
                    "wall_seconds",
                )
            },
            indent=1,
        )
    )


def cmd_compare_replay(args: argparse.Namespace) -> None:
    from . import replay as RP

    verdict = RP.compare(
        Path(args.producer), [Path(c) for c in args.consumers], Path(args.out)
    )
    print(json.dumps(verdict, indent=1, default=str))


def cmd_fit(args: argparse.Namespace) -> None:
    import torch

    from . import prefix_fit as F
    from . import train as T

    R.configure_threads()
    spec = _spec(args)
    tokenizer = T.M.load_tokenizer(spec)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    writer, reader, identity, _ = T.build_models(
        spec, args.device, int(payload["config"]["seed"]), tokenizer
    )
    writer.load_trainable_state(payload["trainable"])
    writer.eval()
    worlds = D.read_jsonl(Path(args.dataset) / "worlds_diagnostic.jsonl")
    if args.worlds:
        worlds = worlds[: args.worlds]
    cfg = F.FitConfig(updates=args.updates, lr=args.lr, max_new_tokens=args.max_new)
    out = Path(args.run_dir) / "prefix_fit"
    R.reset_peak_memory()
    summary = F.run(
        writer, reader, tokenizer, worlds, cfg, out, D.DATA_SEED, spec.torch_dtype()
    )
    summary["checkpoint"] = {
        "path": str(args.checkpoint),
        "digest": R.file_digest(Path(args.checkpoint)),
    }
    summary["identity"] = identity
    summary["source"] = _source_revision(args)
    R.dump_json(out / "fit_summary.json", summary)
    print(
        json.dumps(
            {
                arm: summary["metrics"][arm]["em"]
                for arm in ("generated", "direct", "refined")
            },
            indent=1,
        )
    )


def cmd_report(args: argparse.Namespace) -> None:
    from . import report as REP

    report = REP.build(Path(args.run_dir))
    print((Path(args.run_dir) / "report.md").read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="context_cart")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="generate and hash the dataset pools (CPU)")
    p.add_argument("--out", required=True)
    p.add_argument("--pools", default=None)
    p.add_argument("--count", action="append", help="pool=N overrides for smoke runs")
    _model_args(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("evaluate", help="reader-only arms on one pool")
    p.add_argument("--dataset", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--worlds", type=int, default=0)
    p.add_argument(
        "--panel", action="store_true", help="restrict to the fixed validation panel"
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--summary-max-new", type=int, default=64)
    p.add_argument("--tail-tokens", type=int, default=64)
    p.add_argument("--text-arms", default="")
    p.add_argument(
        "--artifact-arm", action="append", help="name=directory of exported cartridges"
    )
    p.add_argument(
        "--wrong-arm",
        action="append",
        help="name=directory; uses the pool's fixed derangement",
    )
    _model_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("train", help="profile, tiny-fit, or main training")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", help="key=json overrides")
    p.add_argument("--source-revision", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="export cartridges from a checkpoint")
    p.add_argument("--dataset", required=True)
    p.add_argument("--pool", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--namespace", required=True)
    p.add_argument("--worlds", type=int, default=0)
    p.add_argument("--max-new", type=int, default=16)
    p.add_argument("--replay-reference", action="store_true")
    _model_args(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("replay", help="fresh reader-only consumer")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--range", default="0:8")
    p.add_argument("--max-new", type=int, default=16)
    _model_args(p)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser(
        "compare-replay", help="score consumers against the live reference"
    )
    p.add_argument("--producer", required=True)
    p.add_argument("--consumers", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_compare_replay)

    p = sub.add_parser("fit", help="bounded direct-learning comparison")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--worlds", type=int, default=0)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-new", type=int, default=16)
    _model_args(p)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("report", help="recompute scores and gates")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
