"""Authorization tuples, rules, and the default-deny policy engine.

The engine answers one question: for the tuple

    (subject, action, resource, purpose, conditions)

what does the challenge policy say, and what should happen next. Those are two
different answers and the separation is the whole point. The authoritative
scope label is computed from the rules in every mode, while the effective
decision depends on whether the engine is enforcing. That is what lets a run
record an attempted violation that never became a committed one.

Matching is ordered and first-match-wins over an explicit rule list, with a
default deny when nothing matches. Rule fields accept an exact value, the
wildcard ``*``, or a prefix glob such as ``store/project-a/*``. There is no
regular-expression support: rules are meant to be read by a reviewer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from .events import EventLog, EventType
from .ids import (
    ACTOR_POLICY,
    WILDCARD,
    ActionId,
    AuthoritativeScope,
    Decision,
    PolicyMode,
    PurposeId,
    ResourceId,
    SubjectId,
)
from .manifest import digest_json

DEFAULT_DENY_REASON_CODE: Final[str] = "default_deny_no_matching_rule"
"""Reason code attached to a request no rule matched."""

RULE_EFFECTS: Final[frozenset[Decision]] = frozenset({Decision.ALLOW, Decision.DENY})
"""Effects a rule may declare. ``observe`` is a mode, not a rule outcome."""


def pattern_matches(pattern: str, value: str) -> bool:
    """Report whether a rule field pattern matches a concrete value.

    Three forms are supported: ``*`` matches anything, a pattern ending in
    ``/*`` matches any value under that prefix, and anything else must match
    exactly. Prefix matching is on the literal string, so ``store/project-a/*``
    matches ``store/project-a/item-7`` but not ``store/project-ab/item-7``.
    """
    if pattern == WILDCARD:
        return True
    if pattern.endswith("/*"):
        return value.startswith(pattern[:-1])
    return pattern == value


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    """One authorization question about a concrete action.

    ``conditions`` carries challenge-defined context such as the identity the
    agent is currently acting as. Rules do not match on conditions in this
    version; the field is recorded so that a decision can be re-derived later
    when conditional rules exist.
    """

    subject_id: SubjectId
    action_id: ActionId
    resource_id: ResourceId
    purpose_id: PurposeId
    conditions: Mapping[str, str] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the request as an event payload fragment."""
        return {
            "subject_id": str(self.subject_id),
            "action_id": str(self.action_id),
            "resource_id": str(self.resource_id),
            "purpose_id": str(self.purpose_id),
            "conditions": {
                str(key): str(value) for key, value in sorted(self.conditions.items())
            },
        }


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One ordered authorization rule.

    Every field except ``effect`` and ``reason_code`` is a pattern. The
    ``reason_code`` is a stable machine token, not prose: it appears in events
    and in aggregate tables, so it must be reusable across challenges.
    """

    rule_id: str
    subject: str
    action: str
    resource: str
    purpose: str
    effect: Decision
    reason_code: str

    def __post_init__(self) -> None:
        if self.effect not in RULE_EFFECTS:
            raise ValueError("rule effect must be allow or deny")
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if not self.reason_code:
            raise ValueError("reason_code must be non-empty")

    def matches(self, request: AuthorizationRequest) -> bool:
        """Report whether all four patterns match the request."""
        return (
            pattern_matches(self.subject, str(request.subject_id))
            and pattern_matches(self.action, str(request.action_id))
            and pattern_matches(self.resource, str(request.resource_id))
            and pattern_matches(self.purpose, str(request.purpose_id))
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Return the rule as a JSON object, as stored in a challenge file."""
        return {
            "rule_id": self.rule_id,
            "subject": self.subject,
            "action": self.action,
            "resource": self.resource,
            "purpose": self.purpose,
            "effect": str(self.effect),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_json_dict(cls, obj: Mapping[str, Any]) -> "PolicyRule":
        """Rebuild a rule from a challenge file entry."""
        return cls(
            rule_id=str(obj["rule_id"]),
            subject=str(obj["subject"]),
            action=str(obj["action"]),
            resource=str(obj["resource"]),
            purpose=str(obj["purpose"]),
            effect=Decision(obj["effect"]),
            reason_code=str(obj["reason_code"]),
        )


ALLOW_ALL_RULE: Final[PolicyRule] = PolicyRule(
    rule_id="policy/allow-all",
    subject=WILDCARD,
    action=WILDCARD,
    resource=WILDCARD,
    purpose=WILDCARD,
    effect=Decision.ALLOW,
    reason_code="unrestricted",
)
"""Catch-all rule, useful for smoke tests. Real challenges enumerate rules."""


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Verdict for one request.

    ``decision`` is what the gateway acts on. ``authoritative_scope`` is the
    ground truth for classification and is identical in observe and enforce
    mode. When those two disagree, the run is in observe mode and the world is
    about to execute something the policy calls forbidden.
    """

    decision: Decision
    authoritative_scope: AuthoritativeScope
    rule_id: str | None
    reason_code: str

    @property
    def blocked(self) -> bool:
        """Report whether execution must not proceed."""
        return self.decision is Decision.DENY

    @property
    def forbidden(self) -> bool:
        """Report whether the rules label this request out of scope."""
        return self.authoritative_scope is AuthoritativeScope.FORBIDDEN

    @property
    def event_type(self) -> str:
        """Return the outcome event type that follows ``policy.evaluated``."""
        if self.authoritative_scope is AuthoritativeScope.AMBIGUOUS:
            return EventType.POLICY_AMBIGUOUS
        return DECISION_EVENT_TYPES[self.decision]

    def to_json_dict(self) -> dict[str, Any]:
        """Return the decision as an event payload fragment."""
        return {
            "decision": str(self.decision),
            "authoritative_scope": str(self.authoritative_scope),
            "rule_id": self.rule_id,
            "reason_code": self.reason_code,
        }


class PolicyEngine:
    """Ordered, default-deny authorization engine with two enforcement modes.

    In enforce mode a matching deny rule blocks the world call. In observe mode
    the same rule produces a recorded ``observe`` decision and the world still
    executes, which is how a committed synthetic violation is produced on
    purpose. The scope label does not change between modes.

    Rules listed in ``ambiguous_rule_ids`` mark boundary-adjacent requests: the
    rule's effect still decides enforcement, but the scope is reported as
    ambiguous so those trajectories can be held out of the primary endpoint.
    """

    def __init__(
        self,
        rules: Sequence[PolicyRule],
        *,
        mode: PolicyMode = PolicyMode.ENFORCE,
        ambiguous_rule_ids: frozenset[str] = frozenset(),
        default_effect: Decision = Decision.DENY,
        default_reason_code: str = DEFAULT_DENY_REASON_CODE,
    ) -> None:
        if default_effect not in RULE_EFFECTS:
            raise ValueError("default_effect must be allow or deny")
        self.rules: tuple[PolicyRule, ...] = tuple(rules)
        self.mode = mode
        self.ambiguous_rule_ids = ambiguous_rule_ids
        self.default_effect = default_effect
        self.default_reason_code = default_reason_code

    @property
    def policy_revision(self) -> str:
        """Return a stable ``sha256:<hex>`` digest of the rule set.

        The digest covers the ordered rules, the ambiguous rule ids, and the
        default effect. It does not cover the mode: observe and enforce runs of
        one challenge arm share a policy revision by design.
        """
        return policy_revision(
            self.rules,
            ambiguous_rule_ids=self.ambiguous_rule_ids,
            default_effect=self.default_effect,
        )

    def match(self, request: AuthorizationRequest) -> PolicyRule | None:
        """Return the first rule matching the request, or ``None``."""
        for rule in self.rules:
            if rule.matches(request):
                return rule
        return None

    def authoritative_scope(self, request: AuthorizationRequest) -> AuthoritativeScope:
        """Return the ground-truth scope label, ignoring the enforcement mode."""
        rule = self.match(request)
        if rule is None:
            effect = self.default_effect
        elif rule.rule_id in self.ambiguous_rule_ids:
            return AuthoritativeScope.AMBIGUOUS
        else:
            effect = rule.effect
        if effect is Decision.ALLOW:
            return AuthoritativeScope.ALLOWED
        return AuthoritativeScope.FORBIDDEN

    def evaluate(self, request: AuthorizationRequest) -> PolicyDecision:
        """Return the verdict for a request under the current mode."""
        rule = self.match(request)
        scope = self.authoritative_scope(request)
        effect = rule.effect if rule is not None else self.default_effect
        # Observe mode never blocks. Any deny, whether it came from a rule or from
        # the default, is recorded as an observation while the world executes, which
        # is how a committed synthetic violation is produced on purpose.
        if effect is Decision.ALLOW:
            decision = Decision.ALLOW
        elif self.mode is PolicyMode.OBSERVE:
            decision = Decision.OBSERVE
        else:
            decision = Decision.DENY
        return PolicyDecision(
            decision=decision,
            authoritative_scope=scope,
            rule_id=rule.rule_id if rule is not None else None,
            reason_code=(
                rule.reason_code if rule is not None else self.default_reason_code
            ),
        )

    def evaluate_and_log(
        self,
        request: AuthorizationRequest,
        log: EventLog,
        *,
        step_id: int,
        correlation_id: str,
        parent_event_id: int | None = None,
    ) -> PolicyDecision:
        """Evaluate a request and emit the policy events for it.

        Emits ``policy.evaluated`` followed by exactly one of
        ``policy.allowed``, ``policy.denied``, ``policy.observe_only``, or
        ``policy.ambiguous``. Both events carry the full tuple, the scope label,
        the decision, the matched rule id, and the reason code, so the verdict
        is reconstructible from the stream without the engine.
        """
        decision = self.evaluate(request)
        payload = {**request.to_json_dict(), **decision.to_json_dict()}
        evaluated = log.emit(
            EventType.POLICY_EVALUATED,
            payload,
            actor_id=POLICY_ACTOR,
            step_id=step_id,
            parent_event_id=parent_event_id,
            correlation_id=correlation_id,
        )
        log.emit(
            decision.event_type,
            dict(payload),
            actor_id=POLICY_ACTOR,
            step_id=step_id,
            parent_event_id=evaluated.event_id,
            correlation_id=correlation_id,
        )
        return decision

    def with_mode(self, mode: PolicyMode) -> "PolicyEngine":
        """Return a copy of this engine running in a different mode."""
        return PolicyEngine(
            self.rules,
            mode=mode,
            ambiguous_rule_ids=self.ambiguous_rule_ids,
            default_effect=self.default_effect,
            default_reason_code=self.default_reason_code,
        )

    def rule_by_id(self, rule_id: str) -> PolicyRule | None:
        """Return a rule by id, or ``None`` if this engine does not hold it."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def to_json_list(self) -> list[dict[str, Any]]:
        """Return the ordered rules as JSON objects."""
        return [rule.to_json_dict() for rule in self.rules]


def policy_revision(
    rules: Sequence[PolicyRule],
    *,
    ambiguous_rule_ids: frozenset[str] = frozenset(),
    default_effect: Decision = Decision.DENY,
) -> str:
    """Compute the policy digest for a rule set without building an engine."""
    return digest_json(
        {
            "default_effect": str(default_effect),
            "ambiguous_rule_ids": sorted(ambiguous_rule_ids),
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "subject": rule.subject,
                    "action": rule.action,
                    "resource": rule.resource,
                    "purpose": rule.purpose,
                    "effect": str(rule.effect),
                    "reason_code": rule.reason_code,
                }
                for rule in rules
            ],
        }
    )


DECISION_EVENT_TYPES: Final[Mapping[Decision, str]] = {
    Decision.ALLOW: EventType.POLICY_ALLOWED,
    Decision.DENY: EventType.POLICY_DENIED,
    Decision.OBSERVE: EventType.POLICY_OBSERVE_ONLY,
}
"""Outcome event emitted after ``policy.evaluated`` for each decision. An
ambiguous scope overrides this mapping with ``policy.ambiguous``."""

POLICY_ACTOR = ACTOR_POLICY
"""Actor id stamped on every event this module emits."""
