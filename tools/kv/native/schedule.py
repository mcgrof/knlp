# SPDX-License-Identifier: MIT
"""Record actual scheduler admissions and decoded sequences without changing them."""

import json
import time
from pathlib import Path

_sink = None
_label = None
_installed = False
_external_labels = False


def label(value):
    global _label
    _label = value
    if _sink is not None:
        flag = Path(str(_sink) + ".label")
        temporary = flag.with_suffix(".tmp")
        temporary.write_text(json.dumps(value))
        temporary.replace(flag)


def install(path, external_labels=False):
    global _sink, _installed, _external_labels
    if _installed:
        raise RuntimeError("scheduler observer already installed")
    from vllm.v1.core.sched.scheduler import Scheduler

    _sink = Path(path)
    _external_labels = external_labels
    _sink.parent.mkdir(parents=True, exist_ok=True)
    original = Scheduler.schedule

    def observed(self, *args, **kwargs):
        before = {
            rid: (r.num_computed_tokens, r.num_prompt_tokens)
            for rid, r in self.requests.items()
        }
        output = original(self, *args, **kwargs)
        phase = _label
        if _external_labels:
            flag = Path(str(_sink) + ".label")
            phase = json.loads(flag.read_text()) if flag.exists() else None
        if phase is not None:
            scheduled = dict(output.num_scheduled_tokens)
            decoded = [
                rid
                for rid, count in scheduled.items()
                if count == 1 and rid in before and before[rid][0] >= before[rid][1]
            ]
            row = {
                "label": phase,
                "monotonic": time.monotonic(),
                "scheduled_tokens": scheduled,
                "computed_before": before,
                "active_decode_sequences": len(decoded),
                "decode_request_ids": decoded,
                "scheduled_sequences": len(scheduled),
                "resident_computed_tokens_before": sum(
                    before[rid][0] for rid in scheduled if rid in before
                ),
                "running": len(self.running),
                "waiting": len(self.waiting),
                "preempted": list(getattr(output, "preempted_req_ids", [])),
            }
            with _sink.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        return output

    Scheduler.schedule = observed
    _installed = True
