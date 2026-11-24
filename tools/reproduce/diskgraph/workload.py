# SPDX-License-Identifier: MIT
"""Run public DiskGraph sampling and materialize its selected features on a CPU."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import time

import diskgraph
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from observe import observe

REVISION = "21d7284df2a9621c71f21bc4a1a30b7e8effb8dd"
STACK_FILES = {
    "db/badges.parquet": "14d75037870632795cda368fd6516adb1e0fa24fcbdad8e21011f8f81d0c8bb6",
    "db/comments.parquet": "a31d0e7faf17298f3eddab6a13463be05ac60a0bc1cdbc84cafc45bdcbf80f94",
    "db/postHistory.parquet": "ceb2680328d30b2e5ce57572dcc659446556450481fadab4e6384a35979ec8ec",
    "db/postLinks.parquet": "a3b6f7fcd52115004bed2e077a842e3e625c19cec838daed77941990195ce40d",
    "db/posts.parquet": "a49e3c5e55e45212230c6ad290bb352930e9a4dc4a3a9882b36121a0566fbd3b",
    "db/users.parquet": "6326edcecdc02f0364114a12d8ea602b01e9b23798f3a71ca6d884030712e96f",
    "db/votes.parquet": "f2f90de004bfc41f187e8c43dab472e227480914261acf63af7d4c57e06f8e99",
    "tasks/user-engagement/test.parquet": "35489c874bcba4b6b7d4f01fb9db5eb41d9018ed893eabb8db9e0ce45063eeb3",
}
STACK = {
    "users": ("Id", "CreationDate", {}),
    "posts": ("Id", "CreationDate", {"OwnerUserId": "users", "ParentId": "posts"}),
    "badges": ("Id", "Date", {"UserId": "users"}),
    "comments": ("Id", "CreationDate", {"UserId": "users", "PostId": "posts"}),
    "postHistory": ("Id", "CreationDate", {"UserId": "users", "PostId": "posts"}),
    "postLinks": ("Id", "CreationDate", {"PostId": "posts", "RelatedPostId": "posts"}),
    "votes": ("Id", "CreationDate", {"PostId": "posts"}),
}
FIXTURE = {
    "users": ("Id", "Created", {}),
    "items": ("Id", "Created", {}),
    "events": ("Id", "Created", {"User": "users", "Item": "items"}),
}
BASE = 1704067200


def timestamp(seconds):
    return pa.array(seconds, type=pa.timestamp("s"))


def fixture(path):
    if (path / "seeds.parquet").is_file():
        return
    (path / "db").mkdir(parents=True, exist_ok=True)
    r = random.Random(4167)
    users, items, events = 8192, 4096, 196608
    for table, length in (("users", users), ("items", items)):
        pq.write_table(
            pa.table({"Id": range(length), "Created": timestamp([BASE] * length)}),
            path / "db" / f"{table}.parquet",
        )
    pq.write_table(
        pa.table(
            {
                "Id": range(events),
                "User": [r.randrange(users) for _ in range(events)],
                "Item": [r.randrange(items) for _ in range(events)],
                "Created": timestamp(range(BASE + 1, BASE + events + 1)),
                "Feature": [r.randbytes(64).hex() for _ in range(events)],
            }
        ),
        path / "db" / "events.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "Id": r.sample(range(users), users),
                "timestamp": timestamp([BASE + events * 3 // 4] * users),
            }
        ),
        path / "seeds.parquet",
    )


def stack_data(path, download):
    for relative, sha in STACK_FILES.items():
        dest = path / relative
        if not dest.is_file() and download:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".partial")
            url = f"https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/{REVISION}/rel-stack/{relative}"
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "3",
                    "--output",
                    str(tmp),
                    url,
                ],
                check=True,
            )
            with tmp.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != sha:
                    raise ValueError(f"Download hash differs for {relative}")
            tmp.replace(dest)
        with dest.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != sha:
                raise ValueError(f"Input hash differs for {relative}")


def native_dtype(kind):
    if pa.types.is_timestamp(kind):
        return "timestamp"
    if pa.types.is_large_string(kind):
        return "string"
    if pa.types.is_list(kind) and pa.types.is_string(kind.value_type):
        return "string_array"
    if pa.types.is_floating(kind):
        return "float64" if kind == pa.float64() else "float32"
    return str(kind)


def build(index, src, dataset, schema, threads):
    fingerprint = index.parent / "runtime"
    runtime_id = fingerprint.read_text().strip()
    receipt = index / ".knlp-runtime"
    if receipt.is_file() and receipt.read_text() == runtime_id:
        return
    if index.exists():
        raise ValueError(
            f"Choose a new DISKGRAPH_DIR for the changed runtime or incomplete index: {index}"
        )

    def keyed(name):
        return {
            "time": {
                "table": name,
                "column": schema[name][1],
                "bitwidth": 32,
                "unit": "s",
            }
        }

    tables = []
    for name, (key, created, foreign_keys) in schema.items():
        path = src / "db" / f"{name}.parquet"
        tables.append(
            {
                "name": name,
                "primary_key": key,
                "create_time": created,
                "columns": [
                    {"name": f.name, "ingested_dtype": native_dtype(f.type)}
                    for f in pq.read_schema(path)
                ],
                "fkeys": [
                    {
                        "src_column": column,
                        "dst_table": target,
                        "forward": keyed(target),
                        "reverse": keyed(name),
                    }
                    for column, target in foreign_keys.items()
                ],
                "source": {"type": "parquet", "paths": [str(path)]},
            }
        )
    config = {
        "dataset_name": dataset,
        "num_shards": 8 if dataset == "fixture" else 24,
        "tables": tables,
    }
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 32768:
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (min(32768, hard) if hard != -1 else 32768, hard)
        )
    diskgraph.ingest(
        json.dumps(config),
        str(index),
        col_stats=False,
        context_threads=threads,
        encode_threads=threads,
        pk_map_threads=threads,
        edges_map_threads=threads,
        index_threads=threads,
    )
    receipt.write_text(runtime_id)


def engine(index, threads):
    return diskgraph.SamplerEngine(str(index), random_seed=0, num_threads=threads)


def features(reader, table, ids, columns=()):
    requests = [{"node_type": table, "node_ids": ids, "feature_names": list(columns)}]
    result = pa.Table.from_batches(
        [e["batch"] for e in reader.fetch_features(requests) if "batch" in e]
    )
    if len(result) != len(ids):
        raise ValueError(f"DiskGraph did not deliver all {table} rows")
    return result


def seconds(column):
    return pc.cast(pc.cast(column, pa.timestamp("s"), safe=False), pa.int64())


def select_seeds(index, source, dataset, schema, count, threads):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 32768:
        resource.setrlimit(
            resource.RLIMIT_NOFILE, (min(32768, hard) if hard != -1 else 32768, hard)
        )
    is_stack = dataset == "rel-stack"
    name = "OwnerUserId" if is_stack else "Id"
    source = source / (
        "tasks/user-engagement/test.parquet" if is_stack else "seeds.parquet"
    )
    data = pq.read_table(source, columns=[name, "timestamp"])
    candidates = dict(
        zip(data[name].to_pylist(), seconds(data["timestamp"]).to_pylist(), strict=True)
    )
    metadata = json.loads((index / "index_metadata.json").read_text())
    total = (
        next(t["max_id"] for t in metadata["node_types"] if t["name"] == "users") + 1
    )
    lookup = engine(index, threads)
    primary, created, _ = schema["users"]
    requested_keys = pa.array(list(candidates), type=data[name].type)
    found = {}
    for start in range(0, total, 65536):
        gids = list(range(start, min(start + 65536, total)))
        table = features(lookup, "users", gids, (primary, created))
        keep = pc.is_in(table[primary], value_set=requested_keys).to_pylist()
        chosen = [gid for gid, good in zip(gids, keep, strict=True) if good]
        filtered = table.filter(pa.array(keep))
        for gid, key, born in zip(
            chosen,
            filtered[primary].to_pylist(),
            seconds(filtered[created]).to_pylist(),
            strict=True,
        ):
            if born is not None and born <= candidates[key]:
                found[key] = (gid, candidates[key])
    ordered = [found[key] for key in candidates if key in found][:count]
    if len(ordered) != count:
        raise ValueError(f"Only {len(ordered)} of {count} eligible seed rows")
    return ordered


def scan_bytes():
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    raise RuntimeError("Missing process read_bytes counter")


def passes(index, seeds, schema, args):
    reader = engine(index, args.threads)
    mapping = json.loads(reader.get_type_mapping())["node_types"]
    plan = json.dumps(
        {
            "layers": [
                {
                    "fanouts": {},
                    "default_fanout": {
                        "fanout": fanout,
                        "mode": "temporal_most_recent",
                        "absent_key_policy": "uniform_fallback",
                    },
                }
                for fanout in (24, 8, 6)
            ]
        }
    )
    runs = []
    for repeat in range(args.passes):
        digest = hashlib.sha256()
        rows = 0
        bounds = []
        io = scan_bytes()
        cpu = time.process_time()
        start_time = time.monotonic_ns()
        batch_size = 32
        for pos in range(0, len(seeds), batch_size):
            gids, times = zip(*seeds[pos : pos + batch_size], strict=True)
            sample_time = time.monotonic_ns()
            events = reader.sample(
                {"users": list(gids)},
                plan,
                fused=False,
                seed_timestamps={"users": list(times)},
                seed_timestamp_unit=diskgraph.TimeUnit.Seconds,
            )
            fetch_time = time.monotonic_ns()
            audited = defaultdict(list) if not args.no_audit else None
            for event in events:
                if event["type"] != "node" or not len(event["batch"]):
                    continue
                name = mapping[event["type_id"]]
                requested = event["batch"]
                table = features(reader, name, requested["id_b"].to_pylist())
                if audited is not None:
                    actual = seconds(table[schema[name][1]])
                    anchors = pa.array(
                        [times[i] for i in requested["id_a"].to_pylist()]
                    )
                    if pc.any(pc.fill_null(pc.greater(actual, anchors), False)).as_py():
                        raise ValueError(f"Future-dated values returned for {name}")
                    audited[name].append(
                        table.append_column("__seed", requested["id_a"]).append_column(
                            "__gid", requested["id_b"]
                        )
                    )
                rows += len(table)
            if audited is not None:
                for name, chunks in sorted(audited.items()):
                    ordered = pa.concat_tables(chunks).sort_by(
                        [("__seed", "ascending"), ("__gid", "ascending")]
                    )
                    digest.update(name.encode())
                    for batch in ordered.combine_chunks().to_batches():
                        digest.update(batch.serialize())
            bounds.append([sample_time, fetch_time, time.monotonic_ns()])
        stop = time.monotonic_ns()
        runs.append(
            {
                "pass": repeat + 1,
                "start_ns": start_time,
                "end_ns": stop,
                "wall_s": (stop - start_time) / 1e9,
                "cpu_s": time.process_time() - cpu,
                "process_read_bytes": scan_bytes() - io,
                "row_occurrences": rows,
                "feature_digest": None if args.no_audit else digest.hexdigest(),
                "batches": bounds,
            }
        )
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("fixture", "rel-stack"), required=True)
    p.add_argument("--backend", choices=("buffered", "mmap"), required=True)
    for field in ("work", "run", "source", "nvme-monitor"):
        p.add_argument(f"--{field}", type=Path, required=field in ("work", "run"))
    for field in ("seeds", "threads", "qd", "passes"):
        p.add_argument(f"--{field}", type=int, required=True)
    p.add_argument("--no-capture", action="store_true")
    p.add_argument("--no-audit", action="store_true")
    args = p.parse_args()
    if min(args.seeds, args.threads, args.qd, args.passes) < 1 or args.qd > 4096:
        p.error("Positive counts and a queue depth of at most 4096 are required")
    if args.source and args.dataset == "fixture":
        p.error("--source is for a local copy of the pinned RelBench input")
    os.environ.pop("KNLP_DISKGRAPH_INTENT", None)
    os.environ["DG_EDGE_BACKEND"] = "mmap"
    os.environ["DG_IO_QD"] = str(args.qd)
    os.environ["DG_IO_RINGS"] = str(min(args.threads, 64))
    os.environ["DG_IO_COALESCE_PAGES"] = "16"
    os.environ["DG_IO_BATCH_US"] = "0"
    os.environ["DG_IO_BATCH_MULT"] = "1"
    args.work = args.work.resolve()
    args.run = args.run.resolve()
    source = (args.source or args.work / args.dataset).resolve()
    schema = FIXTURE if args.dataset == "fixture" else STACK
    if args.dataset == "fixture":
        fixture(source)
    else:
        stack_data(source, download=args.source is None)
    index = args.work / f"{args.dataset}-index"
    build(index, source, args.dataset, schema, args.threads)
    seeds = select_seeds(index, source, args.dataset, schema, args.seeds, args.threads)
    args.run.mkdir(parents=True)
    for file in index.rglob("*"):
        if file.is_file():
            with file.open("rb") as handle:
                os.fsync(handle.fileno())
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    os.environ["DG_EDGE_BACKEND"] = args.backend
    if args.backend == "buffered" and not args.no_capture:
        os.environ["KNLP_DISKGRAPH_INTENT"] = str(args.run / "intent.jsonl")
    with observe(args.nvme_monitor, index, args.run):
        result = {
            "dataset": args.dataset,
            "backend": args.backend,
            "kernel": os.uname().release,
            "seeds_per_pass": len(seeds),
            "threads": args.threads,
            "queue_depth": args.qd,
            "runtime": (args.work / "runtime").read_text().strip(),
            "passes": passes(index, seeds, schema, args),
        }
    trace = args.run / "intent.jsonl"
    if args.backend == "buffered" and not args.no_capture and not trace.is_file():
        raise RuntimeError("Pinned capture patch did not produce intent.jsonl")
    if trace.is_file():
        with trace.open("a") as output:
            output.write('{"event":"end"}\n')
    with (args.run / "live.json").open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(f"Live DiskGraph receipt: {args.run / 'live.json'}")


if __name__ == "__main__":
    main()
