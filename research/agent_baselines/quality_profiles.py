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

QUALITY_PROFILE_SCHEMA = "knlp.agent_baselines.quality_profile.v1"
CLEAR_TECHNICAL_PROFILE = "clear-technical-v1"

ARM_SPECS = {
    "stock": {"instruction_profile": "current", "layout": "current"},
    "stable-prefix": {
        "instruction_profile": "current",
        "layout": "stable-prefix",
    },
    "clear": {
        "instruction_profile": CLEAR_TECHNICAL_PROFILE,
        "layout": "current",
    },
    "clear-stable-prefix": {
        "instruction_profile": CLEAR_TECHNICAL_PROFILE,
        "layout": "stable-prefix",
    },
}

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
    if spec == {"instruction_profile": "current", "layout": "current"}:
        return template
    task, separator, current_instructions, trailing = _blocks(template)
    instructions = (
        current_instructions
        if spec["instruction_profile"] == "current"
        else CLEAR_TECHNICAL_INSTRUCTIONS
    )
    if spec["layout"] == "stable-prefix":
        return instructions + separator + task + trailing
    return task + separator + instructions + trailing


def quality_profile_receipt(template: str, arm: str) -> dict:
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
    return {
        "schema": QUALITY_PROFILE_SCHEMA,
        "arm": arm,
        **spec,
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
        "template_characters": len(template),
        "instruction_characters": len(instructions),
    }
