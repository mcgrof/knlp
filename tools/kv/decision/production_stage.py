# SPDX-License-Identifier: MIT
"""Run one fresh, owned worker invocation and retain every lifecycle boundary.

This command never allocates a cloud resource, retries a failed arm, selects
a quality policy, or interprets an incomplete panel as a certificate.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .production import ARMS, prepare, read_inputs, sha
from .production_resources import DriverSampler, placement


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cpu_contract():
    """Record the container's visible quota and allowed CPU set before pinning."""
    paths = (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpu.stat",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/proc/self/cgroup",
    )
    files = {p: Path(p).read_text() for p in paths if Path(p).is_file()}
    quota = None
    if "/sys/fs/cgroup/cpu.max" in files:
        amount, period = files["/sys/fs/cgroup/cpu.max"].split()
        quota = None if amount == "max" else int(amount) / int(period)
    elif "/sys/fs/cgroup/cpu/cpu.cfs_quota_us" in files:
        amount = int(files["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"])
        period = int(files["/sys/fs/cgroup/cpu/cpu.cfs_period_us"])
        quota = None if amount < 0 else amount / period
    else:
        raise ValueError("Container CPU quota interface is unavailable")
    allowed = sorted(os.sched_getaffinity(0))
    fd_limit = list(resource.getrlimit(resource.RLIMIT_NOFILE))
    if fd_limit[0] != resource.RLIM_INFINITY and fd_limit[0] < 8192:
        raise ValueError("File-descriptor limit is too small for the fixed workload")
    return {
        "allowed_cpus": allowed,
        "cpu_quota": quota,
        "placement": placement(allowed, quota),
        "fd_limit": fd_limit,
        "cgroup_files": files,
        "quota_scope": "Container-visible cgroup quota; inaccessible ancestor quotas are not inferred",
    }


def pinned(argv, cpus):
    return ["taskset", "--cpu-list", ",".join(map(str, cpus)), *argv]


def terminate(process):
    """Stop only this invocation's process group, even if its leader exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    # A child may outlive the group leader. This group was created exclusively
    # for our command; it does not contain the caller or another experiment.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.returncode


def resource_check(sampler, enforce):
    sampler.check()
    if (
        enforce
        and sampler.rows[-1]["driver_used_bytes"]
        > sampler.policy["driver_envelope_bytes"]
    ):
        raise RuntimeError("Observed driver memory exceeded the common ceiling")


def ready(server, sampler, output):
    attempts = []
    deadline = time.monotonic() + 1800
    try:
        while time.monotonic() < deadline:
            resource_check(sampler, True)
            if server.poll() is not None:
                raise RuntimeError(
                    f"Server exited before readiness: {server.returncode}"
                )
            row = {"monotonic_ns": time.monotonic_ns()}
            try:
                with urllib.request.urlopen(
                    "http://127.0.0.1:8000/health", timeout=2
                ) as response:
                    row["status"] = response.status
                    attempts.append(row)
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError) as error:
                row["error"] = repr(error)
                attempts.append(row)
            time.sleep(1)
        raise TimeoutError("Server did not become ready within the startup bound")
    finally:
        save(output / "readiness.json", attempts)


def child(argv, output, name, cpus, sampler, server, *, enforce):
    command = pinned(argv, cpus)
    started = time.monotonic_ns()
    record = {"argv": command, "started_ns": started, "application_retries": 0}
    failure = None
    with (output / (name + ".log")).open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                resource_check(sampler, enforce)
                if server.poll() is not None:
                    raise RuntimeError("Server exited while collecting the fixed panel")
                time.sleep(0.25)
            if process.returncode != 0:
                raise RuntimeError(
                    f"Fixed {name} collector failed: {process.returncode}"
                )
        except BaseException as error:
            failure = error
            record["error"] = repr(error)
        finally:
            record["returncode"] = terminate(process)
            record["ended_ns"] = time.monotonic_ns()
            try:
                record["resources"] = sampler.window(
                    started, record["ended_ns"], enforce=enforce
                )
            except Exception as error:
                record["resource_error"] = repr(error)
                failure = failure or error
            save(output / (name + "-command.json"), record)
    if failure is not None:
        raise failure
    if not record["resources"]["eligible"]:
        raise RuntimeError("Collector lacks valid common-resource evidence")


def run(install, inputs, output, arm, digest, phase, timing=None):
    protocol, data = read_inputs(inputs, digest)
    if phase == "timing":
        from .production_timing import validate_plan

        if timing is None or sha(timing["plan"]) != timing["plan_sha256"]:
            raise ValueError("Timing requires an immutable invocation plan")
        plan = json.loads(timing["plan"].read_text())
        if (
            plan["arm"] != arm
            or sha(timing["quality"]) != plan["quality_analysis_sha256"]
        ):
            raise ValueError("Timing arm or quality-certificate identity differs")
        validate_plan(
            plan, json.loads(timing["quality"].read_text()), protocol, digest, data
        )
    elif timing is not None:
        raise ValueError("Timing options supplied to a quality or allocation phase")
    cpu = cpu_contract()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "cpu-contract.json", cpu)
    allocation = "equal_slots" if phase == "equal_slots" else "capacity"
    argv, env = prepare(
        install, inputs, output / "server", arm, digest, allocation=allocation
    )
    # Restore the caller's affinity even on a failed collector. The server and
    # each client get explicit disjoint taskset arguments rather than relying
    # on an inherited shell or previously running process.
    previous_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpu["placement"]["control"])
    try:
        with DriverSampler(
            output / "driver-samples.jsonl", policy=protocol["resource_policy"]
        ) as sampler:
            save(output / "device-identity.json", sampler.identity)
            with (output / "server.log").open("x") as log:
                started = time.monotonic_ns()
                command = pinned(argv, cpu["placement"]["server"])
                server = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                exit_record = {
                    "argv": command,
                    "started_ns": started,
                    "pid": server.pid,
                }
                try:
                    ready(server, sampler, output)
                    boundary = time.monotonic_ns()
                    startup = sampler.window(started, boundary)
                    save(output / "startup-resources.json", startup)
                    if not startup["eligible"]:
                        raise RuntimeError("Startup did not meet the common envelope")
                    common = [
                        "--inputs",
                        str(inputs),
                        "--worker-output",
                        str(output / "server"),
                        "--protocol-sha256",
                        digest,
                    ]
                    if phase == "numerical_quality":
                        child(
                            [
                                sys.executable,
                                "-m",
                                "tools.kv.decision.production_qualify",
                                *common,
                                "--output",
                                str(output / "numerical"),
                            ],
                            output,
                            "numerical",
                            cpu["placement"]["client"],
                            sampler,
                            server,
                            enforce=False,
                        )
                        child(
                            [
                                sys.executable,
                                "-m",
                                "tools.kv.decision.production_quality",
                                *common,
                                "--output",
                                str(output / "quality"),
                                "--qualification",
                                str(output / "numerical/QUALIFIED.json"),
                            ],
                            output,
                            "quality",
                            cpu["placement"]["client"],
                            sampler,
                            server,
                            enforce=True,
                        )
                    elif phase == "timing":
                        child(
                            [
                                sys.executable,
                                "-m",
                                "tools.kv.decision.production_timing",
                                *common,
                                "--output",
                                str(output / "timing"),
                                "--plan",
                                str(timing["plan"]),
                                "--plan-sha256",
                                timing["plan_sha256"],
                                "--quality-analysis",
                                str(timing["quality"]),
                            ],
                            output,
                            "timing",
                            cpu["placement"]["client"],
                            sampler,
                            server,
                            enforce=True,
                        )
                except BaseException as error:
                    exit_record["error"] = repr(error)
                    raise
                finally:
                    exit_record["returncode"] = terminate(server)
                    exit_record["ended_ns"] = time.monotonic_ns()
                    save(output / "server-exit.json", exit_record)
        save(
            output / "DONE.json",
            {
                "arm": arm,
                "phase": phase,
                "protocol_sha256": digest,
                "worker_identity_sha256": sha(output / "server/worker-identity.json"),
                "source_sha": protocol["source_sha"],
                "scope": "Complete collection after server shutdown; independent CPU evidence analysis is still required",
            },
        )
    finally:
        os.sched_setaffinity(0, previous_affinity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument(
        "--phase", choices=("equal_slots", "numerical_quality", "timing"), required=True
    )
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--quality-analysis", type=Path)
    args = parser.parse_args()
    supplied = [
        args.plan is not None,
        args.plan_sha256 is not None,
        args.quality_analysis is not None,
    ]
    if any(supplied) and not all(supplied):
        parser.error(
            "The plan, plan hash and quality analysis must be supplied together"
        )
    run(
        args.install.resolve(),
        args.inputs.resolve(),
        args.output.resolve(),
        args.arm,
        args.protocol_sha256,
        args.phase,
        (
            {
                "plan": args.plan.resolve(),
                "plan_sha256": args.plan_sha256,
                "quality": args.quality_analysis.resolve(),
            }
            if all(supplied)
            else None
        ),
    )


if __name__ == "__main__":
    main()
