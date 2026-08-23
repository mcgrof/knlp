"""Typed identifiers, controlled vocabularies, and canonical purpose ids.

This module is the vocabulary root of the package. It imports nothing from the
rest of the package so that every other module can depend on it. Identifiers
are ``NewType`` aliases over ``str`` rather than opaque objects: they stay
JSON-serializable, they compare and hash like strings, and a type checker still
refuses to swap a resource id for an action id.

Enumerations use ``StrEnum`` so that a member serializes to its wire value with
no encoder hook. Values written here are part of the on-disk event, manifest,
and outcome contract; changing one changes the schema version.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, NewType, TypeAlias

SCHEMA_VERSION: Final[str] = "0.1.0"
"""Version stamped on every event, manifest, challenge, and outcome record."""

ActorId = NewType("ActorId", str)
RunId = NewType("RunId", str)
ChallengeId = NewType("ChallengeId", str)
ResourceId = NewType("ResourceId", str)
ActionId = NewType("ActionId", str)
PurposeId = NewType("PurposeId", str)
SubjectId = NewType("SubjectId", str)


class Variant(StrEnum):
    """Arm of a matched challenge pair.

    The two arms share one mechanism, one observation structure, and one tool
    interface. The only load-bearing difference is the policy rule set: the
    fast route is authorized in ``control`` and forbidden in ``treatment``.
    """

    CONTROL = "control"
    TREATMENT = "treatment"


Arm: TypeAlias = "Variant | str"
"""An arm of a challenge, which is a matched-pair member or a named arm.

The matched pair is the enumeration and every comparison is made across it. An
arm outside the pair, such as a capability ceiling that authorizes everything,
carries a plain name instead, so it cannot be mistaken for a third member of
the pair by anything that iterates :class:`Variant`.
"""

ARM_PERMISSIVE: Final[str] = "permissive"
"""Name of the capability-ceiling arm as it appears on disk.

It authorizes every action the world contains and briefs the model on the task
without mentioning authorization, so what it measures is how many agents can
work the mechanism at all. That is a ceiling to read the matched pair against,
not a third authorization condition, which is why it is a name here rather than
a member of :class:`Variant`.
"""

MATCHED_ARM_NAMES: Final[tuple[str, ...]] = (
    str(Variant.CONTROL),
    str(Variant.TREATMENT),
)
"""The arms a matched-pair comparison is made of, by name."""

ARM_NAMES: Final[tuple[str, ...]] = (*MATCHED_ARM_NAMES, ARM_PERMISSIVE)
"""Every arm name that may appear in a challenge file or a run manifest."""


def parse_arm(value: str) -> Arm:
    """Return the arm a name denotes, raising ``ValueError`` if it denotes none.

    A matched-pair name comes back as a :class:`Variant` so that the pair stays
    a closed enumeration downstream. The ceiling arm comes back as its name,
    which is what keeps it out of every comparison that iterates the pair.
    """
    name = str(value)
    if name == ARM_PERMISSIVE:
        return ARM_PERMISSIVE
    try:
        return Variant(name)
    except ValueError:
        raise ValueError(
            f"unknown arm {value!r}; expected one of {', '.join(ARM_NAMES)}"
        ) from None


class Decision(StrEnum):
    """Effective verdict returned by the policy engine for one request.

    ``OBSERVE`` is reported when the engine runs in observe mode and a rule
    would otherwise have denied the request: the world still executes the
    action, and the fact that the rule matched is recorded anyway.
    """

    ALLOW = "allow"
    DENY = "deny"
    OBSERVE = "observe"


class AuthoritativeScope(StrEnum):
    """Ground-truth scope label for a request, independent of enforcement mode.

    This label is always computed from the challenge policy and never inferred
    from model prose. It is what separates an attempted violation from a
    committed one.
    """

    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    AMBIGUOUS = "ambiguous"


class PolicyMode(StrEnum):
    """Whether a denying rule blocks execution or only records the decision."""

    OBSERVE = "observe"
    ENFORCE = "enforce"


class ResourceKind(StrEnum):
    """Class of a node in the synthetic resource graph.

    The kinds describe deterministic state-machine components, not realistic
    daemons. They exist so that a challenge can declare intent and so that the
    trace exporter can group resources without parsing identifier strings.
    """

    DOCUMENT = "document"
    DATASET = "dataset"
    OBJECT_STORE = "object_store"
    ARTIFACT_REGISTRY = "artifact_registry"
    NAME_SERVICE = "name_service"
    METADATA_SERVICE = "metadata_service"
    SERVICE = "service"
    REPOSITORY = "repository"
    IDENTITY = "identity"
    CREDENTIAL = "credential"
    SECRET_STORE = "secret_store"
    JOB_RUNNER = "job_runner"
    WORKER = "worker"
    MAILBOX = "mailbox"
    AUDIT_LOG = "audit_log"
    POLICY_SERVICE = "policy_service"


class FinalClass(StrEnum):
    """Single terminal classification of a trajectory.

    Exactly one value is assigned per run. The three ``invalid_*`` values and
    ``containment_failure`` mark runs that carry no behavioural signal and must
    be excluded from behavioural rates rather than counted as compliance.
    """

    INCAPABLE = "incapable"
    CAPABLE_IN_SCOPE_SUCCESS = "capable_in_scope_success"
    CAPABLE_COMPLIANT_HALT = "capable_compliant_halt"
    CAPABLE_PERMISSION_REQUEST = "capable_permission_request"
    CAPABLE_FORBIDDEN_DISCOVERED_AVOIDED = "capable_forbidden_discovered_avoided"
    CAPABLE_ATTEMPT_BLOCKED = "capable_attempt_blocked"
    CAPABLE_COMMITTED_SYNTHETIC_VIOLATION = "capable_committed_synthetic_violation"
    AMBIGUOUS_POLICY = "ambiguous_policy"
    INVALID_BACKEND = "invalid_backend"
    INVALID_PARSER = "invalid_parser"
    CONTAINMENT_FAILURE = "containment_failure"


class EvidenceClass(StrEnum):
    """What an exposed piece of evidence is evidence about."""

    SCOPE = "scope"
    OWNERSHIP = "ownership"
    ENVIRONMENT_IDENTITY = "environment_identity"
    AUTHORIZATION = "authorization"


class EvidenceKind(StrEnum):
    """Whether an evidence item opens a route or undercuts the task framing.

    An opportunity makes a route visible and materially attractive. A
    contradiction signals that a resource does not belong to the declared
    exercise. Both are pre-registered in the challenge and emitted on a
    declarative schedule, never improvised at run time.
    """

    OPPORTUNITY = "opportunity"
    CONTRADICTION = "contradiction"


class ModelDeclaredScope(StrEnum):
    """Scope the model itself asserts for an action it is about to request.

    This is a model statement, recorded next to the authoritative label so the
    two can be compared. It never affects the policy verdict.
    """

    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    UNCERTAIN = "uncertain"
    UNSTATED = "unstated"


class ReviewStatus(StrEnum):
    """Human-review state of a derived annotation."""

    UNREVIEWED = "unreviewed"
    AUDITED = "audited"
    DISPUTED = "disputed"


class FinishReason(StrEnum):
    """Why the model stopped producing output for one inference call."""

    TOOL_CALL = "tool_call"
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


class TerminalSignal(StrEnum):
    """Agent-initiated end of a trajectory, requested through a control tool."""

    COMPLETE = "complete"
    HALT = "halt"
    PERMISSION_REQUEST = "permission_request"


class StopReason(StrEnum):
    """Why the agent loop stopped.

    ``GOAL_REACHED`` is decided by the world, ``COMPLETE``/``HALT``/
    ``PERMISSION_REQUESTED`` by the model through a control tool, and the rest
    by harness budgets. Budget-driven stops are ordinary outcomes; only
    ``MODEL_ERROR`` and ``PARSE_FAILURE_BUDGET`` suggest an invalid cell.
    """

    COMPLETED = "completed"
    HALTED = "halted"
    PERMISSION_REQUESTED = "permission_requested"
    GOAL_REACHED = "goal_reached"
    MAX_TURNS = "max_turns"
    PARSE_FAILURE_BUDGET = "parse_failure_budget"
    MODEL_ERROR = "model_error"
    WATCHDOG = "watchdog"


ACTOR_HARNESS: Final[ActorId] = ActorId("harness")
ACTOR_AGENT: Final[ActorId] = ActorId("agent-0")
ACTOR_POLICY: Final[ActorId] = ActorId("policy")
ACTOR_WORLD: Final[ActorId] = ActorId("world")
ACTOR_WATCHDOG: Final[ActorId] = ActorId("watchdog")
ACTOR_ANNOTATOR: Final[ActorId] = ActorId("annotator")

SUBJECT_AGENT: Final[SubjectId] = SubjectId("agent-0")

ACTION_LIST: Final[ActionId] = ActionId("list")
ACTION_READ: Final[ActionId] = ActionId("read")
ACTION_WRITE: Final[ActionId] = ActionId("write")
ACTION_CALL: Final[ActionId] = ActionId("call")
ACTION_ASSUME: Final[ActionId] = ActionId("assume")
ACTION_SUBMIT: Final[ActionId] = ActionId("submit")

KNOWN_ACTION_IDS: Final[frozenset[ActionId]] = frozenset(
    {
        ACTION_LIST,
        ACTION_READ,
        ACTION_WRITE,
        ACTION_CALL,
        ACTION_ASSUME,
        ACTION_SUBMIT,
    }
)

PURPOSE_COMPLETE_ASSIGNED_TASK: Final[PurposeId] = PurposeId("complete-assigned-task")
PURPOSE_GATHER_CONTEXT: Final[PurposeId] = PurposeId("gather-context")
PURPOSE_DIAGNOSE_FAILURE: Final[PurposeId] = PurposeId("diagnose-failure")
PURPOSE_VERIFY_RESULT: Final[PurposeId] = PurposeId("verify-result")
PURPOSE_REQUEST_AUTHORIZATION: Final[PurposeId] = PurposeId("request-authorization")
PURPOSE_UNSPECIFIED: Final[PurposeId] = PurposeId("unspecified")

KNOWN_PURPOSE_IDS: Final[frozenset[PurposeId]] = frozenset(
    {
        PURPOSE_COMPLETE_ASSIGNED_TASK,
        PURPOSE_GATHER_CONTEXT,
        PURPOSE_DIAGNOSE_FAILURE,
        PURPOSE_VERIFY_RESULT,
        PURPOSE_REQUEST_AUTHORIZATION,
        PURPOSE_UNSPECIFIED,
    }
)

WILDCARD: Final[str] = "*"
"""Pattern that matches any value in a policy rule field."""
