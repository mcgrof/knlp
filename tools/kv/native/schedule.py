# SPDX-License-Identifier: MIT
"""Record actual scheduler admissions and decoded sequences without changing them."""

import atexit
import json
import os
import time
from pathlib import Path

_sink = None
_label = None
_installed = False
_external_labels = False
_buffer = []
_batch_id = 0
_completed = []
_request_ids = {}
_mode = os.environ.get("KNLP_NATIVE_OBSERVER", "legacy")


def internal_ids(external_ids):
    result = [_request_ids[rid] for rid in external_ids]
    if len(set(result)) != len(result):
        raise ValueError("request ID mapping must be one-to-one")
    return result


def flush():
    if _sink is not None and _buffer:
        with _sink.open("a") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in _buffer)
        _buffer.clear()


atexit.register(flush)


def label(value):
    global _label
    if value is None:
        flush()
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
    from vllm.v1.engine.input_processor import InputProcessor

    _sink = Path(path)
    _external_labels = external_labels
    _sink.parent.mkdir(parents=True, exist_ok=True)
    original = Scheduler.schedule
    original_update = Scheduler.update_from_output
    original_assign = InputProcessor.assign_request_id
    if _mode not in ("legacy", "buffered", "off"):
        raise ValueError("unknown observer mode")

    def assigned(request):
        result = original_assign(request)
        _request_ids[request.external_req_id] = request.request_id
        return result

    InputProcessor.assign_request_id = staticmethod(assigned)

    def updated(self, output, result, *args, **kwargs):
        values = original_update(self, output, result, *args, **kwargs)
        row = getattr(output, "_knlp_observation", None)
        if row is not None:
            completed = {}
            for group in values.values():
                for value in group.outputs:
                    completed[value.request_id] = len(value.new_token_ids)
            record = dict(
                row, completed_monotonic=time.monotonic(), completed_tokens=completed
            )
            _completed.append(record)
        return values

    Scheduler.update_from_output = updated

    def observed(self, *args, **kwargs):
        global _batch_id
        if _mode == "off":
            return original(self, *args, **kwargs)
        before = {
            rid: (r.num_computed_tokens, r.num_prompt_tokens)
            for rid, r in self.requests.items()
        }
        output = original(self, *args, **kwargs)
        phase = _label
        if _external_labels and _mode == "legacy":
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
                "batch_id": _batch_id,
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
            output._knlp_observation = row
            output._knlp_batch_id = _batch_id
            _batch_id += 1
            if _mode == "legacy":
                with _sink.open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            else:
                _buffer.append(row)
        return output

    Scheduler.schedule = observed
    _installed = True
