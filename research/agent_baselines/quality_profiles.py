# SPDX-License-Identifier: MIT
"""Frozen instruction profiles for the PrefixPlan quality experiment.

Profiles change only the invariant instruction block. Layout arms decide
whether that block appears before or after the task-specific PR description.
The explicit arm map makes wording and layout separate experimental factors.
"""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy

QUALITY_PROFILE_SCHEMA = "knlp.agent_baselines.quality_profile.v3"
CLEAR_TECHNICAL_PROFILE = "clear-technical-v1"
EVIDENCE_STATE_PROFILE = "evidence-state-v1"
FINISH_WINDOW_PROFILE = "finish-window-v1"
FINISH_PROTOCOL_PROFILE = "finish-protocol-v1"

ARM_SPECS = {
    "stock": {
        "instruction_profile": "current",
        "layout": "current",
        "execution_protocol": "current",
        "evidence_state": "current",
        "finish_window": "current",
    },
    "stable-prefix": {
        "instruction_profile": "current",
        "layout": "stable-prefix",
        "execution_protocol": "current",
        "evidence_state": "current",
        "finish_window": "current",
    },
    "clear": {
        "instruction_profile": CLEAR_TECHNICAL_PROFILE,
        "layout": "current",
        "execution_protocol": "current",
        "evidence_state": "current",
        "finish_window": "current",
    },
    "clear-stable-prefix": {
        "instruction_profile": CLEAR_TECHNICAL_PROFILE,
        "layout": "stable-prefix",
        "execution_protocol": "current",
        "evidence_state": "current",
        "finish_window": "current",
    },
    "evidence-state": {
        "instruction_profile": "current",
        "layout": "current",
        "execution_protocol": EVIDENCE_STATE_PROFILE,
        "evidence_state": EVIDENCE_STATE_PROFILE,
        "finish_window": "current",
    },
    "finish-window": {
        "instruction_profile": "current",
        "layout": "current",
        "execution_protocol": FINISH_WINDOW_PROFILE,
        "evidence_state": "current",
        "finish_window": FINISH_WINDOW_PROFILE,
    },
    "finish-protocol": {
        "instruction_profile": "current",
        "layout": "current",
        "execution_protocol": FINISH_PROTOCOL_PROFILE,
        "evidence_state": EVIDENCE_STATE_PROFILE,
        "finish_window": FINISH_WINDOW_PROFILE,
    },
}

EVIDENCE_STATE_INSTRUCTIONS = """

## Evidence and state protocol

Keep a short state in your reasoning with these headings:

- `Observed facts`: results established by commands that used the real code.
- `Unresolved checks`: hypotheses, failures, or acceptance criteria not yet
  established.
- `Next action`: the single most useful next command.

Do not describe a hypothesis as verified. An edit is not established until
you inspect the real diff and exercise the changed code. Use executable
assertions for properties that support correctness; a print-only script or a
mock that manually implements the proposed fix is not acceptance evidence.
"""

FINISH_WINDOW_INSTRUCTIONS = """

## Budget and finish protocol

After each command, the runner reports the exact model-call budget.

When five calls remain, decide which unresolved checks are essential. Do not
start optional exploration after that checkpoint. Reserve the final three
calls, in order, to create `patch.txt`, inspect it, and submit it with the
exact required command. If an earlier check failed, keep that failure visible
and make the safest justified submission decision within the remaining
budget.
"""

FINISH_PROTOCOL_INSTRUCTIONS = """

## Budget, evidence, and finish protocol

After each command, the runner reports the exact model-call budget. Keep a
short state in your reasoning with these headings:

- `Observed facts`: results established by commands that used the real code.
- `Unresolved checks`: hypotheses, failures, or acceptance criteria not yet
  established.
- `Next action`: the single most useful next command.

Do not describe a hypothesis as verified. An edit is not established until
you inspect the real diff and exercise the changed code. Use executable
assertions for properties that support correctness; a print-only script or a
mock that manually implements the proposed fix is not acceptance evidence.

When five calls remain, decide which unresolved checks are essential. Do not
start optional exploration after that checkpoint. Reserve the final three
calls, in order, to create `patch.txt`, inspect it, and submit it with the
exact required command. If an earlier check failed, keep that failure visible
and make the safest justified submission decision within the remaining
budget.
"""

FINISH_PROTOCOL_OBSERVATION = """
{% set calls_remaining = step_limit - n_model_calls %}
<finish_protocol_state>
model_calls_used={{ n_model_calls }}
model_calls_remaining={{ calls_remaining }}
{% if calls_remaining == 5 -%}
checkpoint=decide_essential_verification
required=List essential unresolved checks and stop optional exploration.
{% elif calls_remaining == 3 -%}
phase=finish_window
required=Create patch.txt in this call.
{% elif calls_remaining == 2 -%}
phase=finish_window
required=Inspect patch.txt in this call. Do not modify it without reinspecting.
{% elif calls_remaining == 1 -%}
phase=finish_window
required=Submit with the exact required command in this call.
{% elif calls_remaining <= 0 -%}
phase=budget_exhausted
{% else -%}
phase=work_and_verify
{% endif -%}
</finish_protocol_state>
"""

CLEAR_TECHNICAL_INSTRUCTIONS = """<instructions>
# Task instructions

## Goal

Act as a software engineer. Use the computer shell continuously to solve the
issue in the PR description. Modify only the necessary non-test source files
in `/testbed`. Make a general change that is consistent with the codebase.

Do not modify tests or configuration files such as `pyproject.toml` and
`setup.cfg`.

## Work process

1. Find and read the relevant code.
2. Create a script that reproduces the issue.
3. Modify the source code to fix the issue.
4. Run the reproduction script again to verify the fix.
5. Test relevant edge cases.

For every response:

1. State what you observed and what you will do next.
2. Make at least one `bash` tool call.
3. Read the command result before you choose the next action.

You can make multiple tool calls in one response when the commands are
independent. Each command runs in a new subshell. Directory changes and
environment-variable changes do not persist. Put required setup in the same
command, for example:

```bash
MY_ENV_VAR=VALUE cd /path/to/working/dir && command
```

The environment provides a full Linux shell. Use non-interactive flags such
as `-y` and `-f`. Do not use interactive programs such as `vi` or `nano`. You
can create helper scripts and install a missing tool when necessary.

## Submission

When the work is complete, do these steps in order and as separate commands:

1. Create `patch.txt` with `git diff -- path/to/file1 path/to/file2 > patch.txt`.
   List only the source files that you modified to fix the issue. Do not commit.
2. Inspect `patch.txt`. Confirm that it contains only the intended changes and
   that its headers use `--- a/` and `+++ b/` paths.
3. Submit with this exact command:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

The submitted patch must not contain reproduction files, helper scripts,
tests, generated files, binaries, or packaging and setup changes unless a
packaging or setup file is directly part of the issue. If you modify
`patch.txt` after inspection, inspect it again. Do not read, edit, test, or
run another command after submission.
</instructions>"""

_TEMPLATE = re.compile(
    r"(<pr_description>.*?</pr_description>)(\s+)"
    r"(<instructions>.*?</instructions>)(\s*)",
    re.DOTALL,
)


def _blocks(template: str) -> tuple[str, str, str, str]:
    match = _TEMPLATE.fullmatch(template)
    if not match or match[1].count("{{task}}") != 1 or "{{task}}" in match[3]:
        raise ValueError(
            "Unsupported instance template; expected pinned SWE-bench form"
        )
    return match[1], match[2], match[3], match[4]


def arm_spec(arm: str) -> dict[str, str]:
    try:
        return deepcopy(ARM_SPECS[arm])
    except KeyError as error:
        raise ValueError(f"Unknown quality arm: {arm}") from error


def render_instance_template(template: str, arm: str) -> str:
    """Apply one frozen wording profile and one layout to the pinned template."""
    spec = arm_spec(arm)
    if spec == ARM_SPECS["stock"]:
        return template
    task, separator, current_instructions, trailing = _blocks(template)
    instructions = (
        current_instructions
        if spec["instruction_profile"] == "current"
        else CLEAR_TECHNICAL_INSTRUCTIONS
    )
    protocol = ""
    if spec["execution_protocol"] == FINISH_PROTOCOL_PROFILE:
        protocol = FINISH_PROTOCOL_INSTRUCTIONS
    elif spec["evidence_state"] == EVIDENCE_STATE_PROFILE:
        protocol = EVIDENCE_STATE_INSTRUCTIONS
    elif spec["finish_window"] == FINISH_WINDOW_PROFILE:
        protocol = FINISH_WINDOW_INSTRUCTIONS
    if protocol:
        instructions = instructions.replace(
            "</instructions>", protocol + "\n</instructions>"
        )
    if spec["layout"] == "stable-prefix":
        return instructions + separator + task + trailing
    return task + separator + instructions + trailing


def apply_observation_protocol(observation_template: str, arm: str) -> str:
    """Append live budget facts only for arms with a finish window."""
    spec = arm_spec(arm)
    if spec["finish_window"] != FINISH_WINDOW_PROFILE:
        return observation_template
    return observation_template.rstrip() + "\n" + FINISH_PROTOCOL_OBSERVATION


def quality_profile_receipt(
    template: str, arm: str, observation_template: str | None = None
) -> dict:
    """Bind a rendered arm to its immutable wording and layout identities."""
    spec = arm_spec(arm)
    task_position = template.index("<pr_description>")
    instruction_position = template.index("<instructions>")
    stable = instruction_position < task_position
    if stable != (spec["layout"] == "stable-prefix"):
        raise ValueError("Rendered template does not match its declared layout")
    instruction_start = instruction_position
    instruction_end = template.index("</instructions>") + len("</instructions>")
    instructions = template[instruction_start:instruction_end]
    receipt = {
        "schema": QUALITY_PROFILE_SCHEMA,
        "arm": arm,
        **spec,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
        "template_characters": len(template),
        "instruction_characters": len(instructions),
    }
    if spec["finish_window"] == FINISH_WINDOW_PROFILE:
        if (
            observation_template is None
            or "<finish_protocol_state>" not in observation_template
        ):
            raise ValueError("Finish protocol requires live observation state")
        receipt.update(
            observation_template_sha256=hashlib.sha256(
                observation_template.encode()
            ).hexdigest(),
            decision_checkpoint_calls_remaining=5,
            finish_window_calls=3,
        )
    return receipt
