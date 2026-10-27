# Offline maintenance assessment

The helper explains whether supplied facts satisfy a next action. It does not
authenticate those facts or perform the action. A runtime must read trusted
policy and native repository/CI state, validate authority and enforce the final
write. An assessment always reports `authority_verified: false`.

## Documents and identity

Policy version 2 defines independent actions, class scopes and accepted
classification producers, protected paths, required tests, model/resource
limits and human review rules. Missing actions deny that action. A protected
path overrides a class. Exact paths and recursive `directory/**` selectors
are supported; other wildcard syntax is rejected. Match rename source and
destination. A classification record establishes attribution, not correctness.
The fixtures cover reserved examples; they are not a general semantic detector.

Task version 1 names the full candidate inventory, classes, next action,
completion point, exact head/target and policy digest. Mixed changes use the
union of tests and intersection of allowed actions. Unclassified or reserved
effects need a revision-bound human decision, not a permissive path match.

Evidence version 1 records outcomes, assertions, exact command and environment
identities, expected producer, runner, artifact digest, resource grant and
receipt. Required assertions must pass, not just the job's exit code. A test
may be excluded only by a named policy applicability rule. Failed attempts
remain visible; repair deterministic failures instead of rerunning unchanged
code until one result passes. Cancellation is transient only under the
declared bounded infrastructure-retry rule.

The snapshot's usage is cumulative. The smaller task/model limit applies;
finishing already tested work at a zero spending limit is allowed if no
spending occurred. A runtime must reserve any additional consumption before
acting; the offline helper cannot predict the next call's token or CI cost.

Snapshot version 1 supplies policy, candidate, review, protections, resources
and effects observations as needed by the action. Each has state, time,
source, receipt and a typed value. Missing, unknown, future or stale observations
cannot supply approval or evidence. Native check requirements are added to the
class requirements, never removed by candidate text. A changed head, target,
policy, command or environment invalidates dependent evidence. Post-merge
checks bind to the actual merged revision.

Document hashes use SHA-256 of UTF-8 JSON with sorted keys, no insignificant
spaces, unescaped Unicode and no nonfinite numbers. Hashes identify bytes;
they do not authenticate policy authors or result producers. Every external
operation has its own identity and receipt state. Unknown outcome blocks a
repeat until the integrating writer reconciles authoritative state.

## Results and human decisions

Precedence is blocked, needs_human, waiting_resource, retryable, then ready.
All reasons remain visible. A denied action overrides an unavailable runner.
When the only missing result has an explicit matching unavailable-environment
observation, the task waits for that resource rather than inventing a second
missing-result failure. Missing coverage without that observation blocks.

Ready means declared gates satisfy the next action, not permission to act.
Complete requires an evidenced selected endpoint; a verified merge includes
its required post-merge checks. A post-merge failure remains unresolved and
requires the project's recovery decision. Merge permission never grants an
automatic revert or an unrelated follow-up write.

Human review can be required before publication, before merge or on exception.
Use an existing standing grant without asking for it again. A one-off decision
is limited to its revision, action, policy and expiry. A pending or conflicting
decision does not grant anything. A handoff supplies completed work, the exact
gap, recommendation, alternatives, evidence, responsible route and deadline.
Silence leaves the dependent action waiting, not approved.

## Native enforcement

Production publication/merge needs an authenticated adapter for the project's
actual forge and test runner. It must honor branch protection, required reviews
and the expected check producer, enforce head/target freshness atomically or
through a merge queue, store action receipts and reconcile unknown outcomes.
Do not execute commands or trust receipts merely because they occur in JSON.

For GitHub, consult its current documentation on
[status checks](https://docs.github.com/en/pull-requests/reference/status-checks),
[protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
and [merge queues](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue).
In particular, a skipped job can report success; it is not demonstrated test
coverage. Queue validation must cover the combined candidate. An offline
model review cannot substitute for a required native or human review.

The offline helper supplies no native GitHub, mailing-list or production adapter.
Production merge and a real-project pilot remain disabled pending their own
trusted integration and maintainer decisions.
