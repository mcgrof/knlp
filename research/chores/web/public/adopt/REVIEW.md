# Review an incoming change

Give the maintainer a decision they can act on, supported by the diff and
tests. Do not treat the pull-request title or a model summary as the change.
A review assignment permits analysis, not a public comment, approval, squash
or merge. Follow the target project's contribution rules before local style
preferences. This applies to GitHub PRs and to versioned mailing-list series.

## Freeze the review

Record repository, target branch, base/head hashes, merge base, series order,
policy revision and prior review state. A rewritten head is a new review
input. Keep a queue entry with owner, branch, dependencies, test coverage,
outstanding findings and next action. Separate draft, reviewed, tested,
ready-to-post and owner-accepted states. A dashboard cannot create acceptance.

Read the full patch of every commit and the series diff, including tests,
build/CI changes and documentation. For each hunk ask what behavior it
changes, why this commit owns it and whether its message explains that
reason. Inspect error paths, cleanup, concurrency, data types and trust
boundaries. Treat changed workflows, hooks, dependency scripts and generated
code as execution surfaces. Do not execute them with reviewer credentials.

Report findings by severity: correctness and unsafe intermediate commits;
series structure; documentation and claims; then optional style comments.
Each finding needs a precise location, an observed or reasoned failure,
affected configuration and a bounded remedy. Separate a confirmed bug from a
question, an untested concern and a preference. Never invent a test result.

## Ask for a squash when it repairs the series

| Situation | Recommended review request |
| --- | --- |
| A later commit repairs, formats or adapts code introduced earlier in this unmerged series | Fold each affected hunk and its regression test into the commit that owns the code. Check intervening overlaps and retest affected intermediate commits. |
| One cleanup touches code from several earlier commits | Distribute its hunks to those commits; do not attach the whole cleanup to an arbitrary predecessor. |
| A feature commit contains unrelated cleanup | Split the independent cleanup or move it to its owning change. |
| Commits are separate, useful and individually correct | Keep the sequence; do not ask for one giant squash merely for convenience. |
| The bug exists on the target's current upstream branch | Prepare an independently reviewable fix with its reproducer, then make the feature depend on it. Follow the project's landing policy. |

Bisect safety means each proposed intermediate commit remains usable enough
for the project's tests and supported build. Preserve a meaningful regression
test with the fix. Request history changes from the author; do not rewrite or
force-push their branch. If the owner separately authorizes editing a series,
preserve a recovery ref, verify the intended final tree/patch equivalence,
rerun checks and invalidate reviews tied to old hashes. Never squash a
released upstream bug fix away into an unrelated new feature.

## Identify an unlabeled fix from evidence

Trace the old failing behavior and the intended behavior. Reproduce on base
and verify on head when execution is permitted. Use blame/log and the actual
introducing diff to distinguish an upstream bug from a defect introduced by
this series. A title such as “cleanup” is not proof that behavior is unchanged.
Conversely, refactoring or adding a new capability is not automatically a fix.

For an upstream bug, recommend a subject and body that describe the defect,
trigger, user-visible effect and remedy. Recommend a `Fixes:` trailer only
when the project uses it and the introducing commit is verified. Do not guess
its hash, add a stable-backport recipient mechanically, or call a new-path
comparison a regression. If the bug belongs only to the unmerged series,
request a fold instead of manufacturing upstream fix metadata.

Keep commit attribution, sign-off, AI disclosure, subject style and line
length consistent with the target project and employer rules. Do not invent
human co-authors, model identities, usage receipts or attestations. Correct
the public message to explain the code; keep internal campaign names and
private review conversations out of it.

## Validate the recommendation

Run the project's full applicable mechanical gates, not only the easiest
linter. Use [TEST-PLAN.md](TEST-PLAN.md) to separate unit, emulated, physical
and GPU coverage. Compare base/head failures before blaming the patch.
Record each skip and untested required configuration. A missing device can
mean blocked, not green. Check messages and docs against what actually ran.
Avoid requesting unrelated redesign or a stronger proof than the claim needs.

Return a short summary, severity-ranked findings, commit disposition and a
test matrix with exact identities. Recommend accept, revise or blocked with a
reason; the maintainer retains the decision. Draft public wording separately
and post only through an explicitly approved route with duplicate prevention.
If delivery is uncertain, record unknown rather than blindly posting twice.

## Finish within the selected policy

A review-only assignment ends with a reviewed patch. A maintenance assignment
may also include repair, publication or merge under independent standing
grants; reuse those grants without repeatedly asking for the same permission.
Repair findings within scope, rerun affected tests and review the final series.
Recheck native protections, current head/target and expected check producers
before merging; only the native writer can enforce freshness atomically.
Verify the actual merged revision before recording completion. A failing
follow-up check needs a recovery handoff and pauses affected automatic merges.
Use [HANDOFF.md](HANDOFF.md) for decisions beyond the assignment, with completed
work, the exact question, recommendation, alternatives, evidence and deadline.
