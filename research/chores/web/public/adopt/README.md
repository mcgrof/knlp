# AI code maintenance with human delegation

Routine maintenance takes more than a review comment. Someone must reproduce
the problem, prepare the fix, test it, respond to findings, and get the change
merged. Chores provides a way to organize that work around an existing AI
agent and the project's own maintenance rules.

A maintainer decides which changes the agent can complete and which decisions
need a person. One project may want patches ready for review. Another may let
the agent merge routine fixes after required tests pass. Both need a useful
handoff when the agent lacks evidence, access, or a decision about intended
behavior.

This guide describes that integration. Your agent, test infrastructure, and
repository service carry out the work and enforce access. Reading the guide
or validating a configuration does not install a runner or grant permissions.

## Start with one repair

Give your agent this request:

> Read the code-maintenance guide at chores.knlp.io and inspect this
> repository's contributor instructions, tests, and existing automation.
> Prepare an integration that uses those conventions. Identify the maintenance
> changes you can handle, the tests each requires, and the actions already
> authorized. Propose clear rules for publishing, merging, and handing a
> decision to a person. Then take one small maintenance task as far as the
> existing authorization permits. Preserve the patch and evidence if blocked.
> Show me the few remaining policy decisions with a recommendation for each.

The first integration should fit the project. Keep its issue tracker, review
channel, CI, and merge process. Start with one worker and a small task such as
a reproducible test failure. Recurring work becomes useful after that task
can finish with a clear record of what happened.

Use the AI access you already have where its supported tools and account
rules permit the intended work. Record whether execution is interactive or
unattended, which source data the agent may send, and its usage limit. If the
session expires or quota runs out, save progress and pause. Changing accounts
or buying extra capacity needs its own authorization.

## Choose what the agent may finish

Choose permissions per action and kind of change. A single autonomy switch
cannot express permission to fix a test while reserving public API decisions
for a maintainer.

| Starting policy | Agent's normal completion point | Human decision |
| --- | --- | --- |
| Prepare changes | A reviewed patch with test results | Publish and merge |
| Publish for review | A pull request or patch series, updated after review | Merge |
| Merge eligible maintenance | An eligible change merged and checked afterward | Exceptions and changes outside the approved scope |

These are starting policies. A project can allow automatic documentation
fixes while requiring review for implementation changes. Permission to merge
does not imply permission to publish releases, alter infrastructure, or
revert unrelated work.

For each maintenance class, name the affected paths and behavior, required
tests, review requirements, allowed actions, and resource limits. Also name
protected areas such as public interfaces, persistent formats, authentication,
and the maintenance policy itself. Use the most restrictive applicable rule
when a change crosses classes. An unclassified change needs a decision.

Path matching helps select rules; it cannot establish that a change is
harmless. A one-line change can alter an interface, and a documentation file
can contain an executable example. Review the effect as well as the filename.
Read the effective policy from a trusted revision. An agent's proposed policy
edit must not give that same change new privileges.

Once the maintainer adopts a rule, the agent should use it. Repeatedly asking
for permission already granted adds work without improving the decision.

## Carry the task through to completion

Begin with a specific problem and a stopping condition. For a bug, reproduce
the failure or record the best available evidence. For maintenance without a
failure, establish the behavior that must remain unchanged. Check whether
someone already owns the task before preparing a competing patch.

Make a focused repair and run the tests selected by its effects. Read the
whole change, including generated files and intermediate commits when the
project reviews a series. Fix findings within the task's scope. If repeated
attempts reach the agreed time or cost limit, preserve the useful work and
explain the unresolved problem.

Publish through the project's normal channel when allowed. A pull request and
an emailed patch series need different adapters, but both need a stable task
identity, the submitted revision, and a record of the outcome. An uncertain
submission response requires checking what happened before sending it again.

Before merging, verify that the candidate is still the reviewed change, its
required tests and approvals are current, blocking findings are resolved,
and the project's native protections permit the merge. A rebase or target
branch update can make earlier evidence stale. Revalidate the affected checks
or use the project's merge queue to test the combined result. Do not weaken
checks to make a task eligible.

After the merge, record the resulting revision and run the agreed follow-up
checks. A merged change with failing validation remains open work. Stop
related automatic merges, report the failure, and prepare a repair or revert
under the project's recovery policy.

## Match tests to the behavior changed

Test requirements describe the evidence needed. Resource permissions describe
where and how the test may run. Keep both explicit: access to a powerful
machine does not make it the right test environment.

| Project type | Common checks | Changes that need additional evidence |
| --- | --- | --- |
| Portable library | Build, unit tests, sanitizers, compatibility checks | Device transport or firmware interaction needs the required device coverage |
| Kernel integration | Build and checks in a disposable guest | Loading, attaching, and observing behavior needs a supported kernel, configuration, and features |
| Build tool | Clean and incremental builds, deterministic outputs, cache invalidation | Platform behavior needs the affected platform; dependency changes need the relevant dependency cases |
| Storage tool | File-backed correctness tests and emulated feature tests | Device behavior needs suitable hardware; performance claims need controlled comparisons |

A mock can test error handling without proving real-device behavior. An
emulated device can exercise a protocol without qualifying physical timing.
A successful build cannot substitute for a required runtime test. Accept a
substitute only when the maintainer's test policy says it answers the same
question.

Each test should identify its assertion, environment, permitted resources,
expected duration, and cleanup. Destructive tests require an explicit
disposable target and operator-approved scope. Hardware identity and access
details can stay in the private runner configuration.

Record passed, failed, skipped, unavailable, inconclusive and cancelled
results separately. Exclude a test as not applicable only under a trusted
rule for this change. Bind results to the tested source revision and relevant
environment. Missing required coverage blocks merging. Optional coverage
may remain missing if the policy
allows it and the report says so plainly.

## Hand over the decision, with the work attached

A useful handoff explains what is complete, what prevents progress, and the
smallest decision or action needed. Include the patch, relevant test results,
recommended next step, alternatives, and what the agent will do while waiting.

Projects choose who receives each kind of handoff and when to interrupt them.
Routine reviews can be batched. A busy test machine can remain queued until
its resource deadline. Suspected corruption or unauthorized activity needs
the project's immediate incident channel. Silence leaves a decision pending.

Three examples show the difference:

- **A documentation repair completes.** The change fits an approved class,
  required checks pass, and the merge policy permits it. The agent merges,
  checks the result, and includes the completion in the next summary.
- **A device change waits for evidence.** Unit tests pass, but a required
  hardware test cannot run. The agent preserves the patch and sends the
  operator the test, prerequisites, and expected result. It resumes when
  valid evidence arrives.
- **A behavior change needs judgment.** Two plausible fixes produce different
  public behavior. The agent explains the tradeoff and recommends one. The
  maintainer decides; the agent then completes the authorized implementation.

Record whether a human decision applies only to the current revision or
changes the standing policy. That distinction lets later routine work finish
without reopening the same question.

## Check the integration before expanding it

Try one real repair through its chosen completion point. Also exercise a
missing required test, a changed candidate revision, and an action the policy
denies. Confirm that interrupted work resumes without duplicate submissions
or lost evidence, and that failed post-merge checks reach the right person.

Judge the integration by completed repairs, maintainer time, missed defects,
unnecessary interruptions, and total resource use. Compare it with the
project's existing agent-and-CI workflow. Add scheduling or more workers only
when they solve an observed bottleneck.

The adoption materials are a starting point for that work. Keep the project
policy small, the evidence specific, and the remaining human decision clear.

## Offline tools and integration status

The [portable skill](SKILL.md), [policy proposal](policy.json),
[test plan](TEST-PLAN.md), [review procedure](REVIEW.md),
[human handoff](HANDOFF.md) and [generic profiles](PROJECTS.md) belong together.
The policy proposal is not an input to the existing Chores status builder.
A project's accepted task events can feed a private briefing through a separate
adapter; a dashboard never grants permission.

[maintenance.py](maintenance.py) prepares bounded local inventory, checks policy
format/completeness and assesses supplied task facts. Install the existing
Chores dependency `jsonschema>=4.20` in a task-local Python 3.11+ environment.
Keep the [schemas](schemas/policy-v2.json) directory beside the script. Download
and inspect source before running it; never pipe a remote installer to a shell.

```sh
python3 maintenance.py init --repo . --output /tmp/maintenance-proposal
python3 maintenance.py check-profile /tmp/maintenance-proposal/policy.proposed.json
python3 maintenance.py check-profile policy.approved.json --complete
python3 maintenance.py assess-task --policy policy.approved.json --task task.json \
  --evidence evidence.json --snapshot snapshot.json --now 2030-01-01T12:00:00Z
```

Choose a new output directory: init never replaces an existing directory.
It reads at most 64 fixed instruction/build/workflow files, at most 32 KiB
each and 512 KiB total. It uses read-only Git metadata inspection and records
command candidates by source line, revision and hash without executing them.
It does not follow repository-content symlinks or copy raw file contents.
Missing or skipped files become visible inventory gaps.
Dirty or untracked files keep their content hash but have a null source
revision; a baseline commit must not be presented as their command source.

The versioned schemas for [policy](schemas/policy-v2.json),
[task](schemas/task-v1.json), [evidence](schemas/evidence-v1.json),
[snapshot](schemas/snapshot-v1.json) and [assessment](schemas/assessment-v1.json)
share [fixed types](schemas/common-v1.json). Inputs are limited to 256 KiB per
document. Unknown keys/versions, duplicate keys, nonfinite values, malformed
timestamps and hashes are rejected. Schema references resolve locally only.
The old shape-only proposal format is retired; [check_profile.py](check_profile.py)
is a compatibility entrypoint for the new proposal checker.

An incomplete draft may pass syntax checking but fails `--complete`.
Even a complete document is not authenticated approval. No helper executes
a policy command, calls a provider, posts a review or merges a branch.
Every assessment includes `authority_verified: false`. See
[the assessment contract](ASSESSMENT.md) before consuming its output.

These offline tools do not qualify a production forge, hostile-code runner,
corporate account or real hardware. No native integration is supplied here.
A production adapter must authenticate policy, CI/review sources and grants,
enforce current head/target conditions at the writer, and reconcile operation
receipts. Automatic production merge remains unsupported until that adapter
is selected and qualified. No real adopter pilot has run.

Source: [knlp Chores starter](https://github.com/mcgrof/knlp/tree/main/research/chores/web/public/adopt).
