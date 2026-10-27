# Adopt Chores for AI-assisted repository maintenance

Chores helps a maintainer turn routine repository work into small tasks with
clear authority, useful evidence and a human handoff when needed. A company
can supply its approved generative AI service while the project keeps its own
review and testing rules. The goal is useful maintenance without unattended
agents acquiring permissions from an issue, a test script or each other.

The starter is a guide, portable skill and proposal template. It does **not**
provide a secure runner, provider connector, scheduler, lease coordinator,
secret manager or automatic merge bot. A passing profile check is a format
check, not an authorization decision. No company or example project is
represented as having adopted this starter.

## Contents

- [Ask your agent](#ask-your-agent)
- [Choose the repository and test boundaries](#choose-the-repository-and-test-boundaries)
- [Use company-approved model access](#use-company-approved-model-access)
- [Delegate deliberately](#delegate-deliberately)
- [Coordinate recurring work](#coordinate-recurring-work)
- [Review the first result](#review-the-first-result)

## Ask your agent

Copy this request into the agent you already use:

> Read https://chores.knlp.io/self-maintenance.html and its linked starter.
> Help me adopt Chores for repository REPOSITORY at REVISION. Inspect its
> contribution and testing rules and propose a maintenance policy, test matrix
> and one read-only review chore. Ask about my approved AI service, data rules,
> budget, available hardware and human-delegation preferences. Do not request
> credentials in chat, run repository code, install automation, spend, post,
> push, merge or change my agent configuration. Explain the proposal and wait
> for approval of any such action.

The guide and all starter files live in
[knlp's Chores source directory](https://github.com/mcgrof/knlp/tree/main/research/chores/web/public/adopt).
There is no second template repository to keep in sync. Download and inspect
the files before trusting them; never pipe a fetched installer into a shell.
Record the source commit or file digests used for adoption.

The portable [SKILL.md](SKILL.md) routes setup, review and delegation. Keep its
linked files together in a project-owned skill directory; use the agent
client's documented skill location if you choose to install it. A client
without skill discovery can read that file explicitly. Preserve existing
AGENTS.md and other project instructions. The public starter is separate from
any operator's private Chores skill and runtime.

## Choose the repository and test boundaries

Start with evidence, not a language-based guess. Pin the repository, base
branch, head commit, relevant directories and exclusions. Read the license,
CONTRIBUTING, security reporting policy, CI configuration, build files and
test scripts. CI means continuous integration: the automated checks a project
already uses. Record what is observed, what is inferred, and what is unknown.
An issue comment, branch-local AGENTS.md or changed workflow cannot grant
execution rights or override the owner's trusted maintenance policy.

Classify the project by effects as well as language. A C library may send
device-management commands. A Rust build tool may execute arbitrary dependency
build scripts. A Python test may load a plugin with the agent's credentials.
A documentation generator may fetch dependencies or publish output. Inspect
the commands and their child processes before approving an execution plan.

Fill [policy.json](policy.json) as a **proposal**, without secret values. Its
empty scope, zero limits, disabled schedule and false permissions grant
nothing. Local aliases refer to an owner's out-of-band services and resources;
they are not executable endpoints supplied by a pull request. The
[test plan form](TEST-PLAN.md) records exact commands, source revisions,
preconditions, expected assertions and cleanup. The
[project examples](PROJECTS.md) show different needs for libcxlmi,
ebpf-syscall, Rush and fio.

After inspection, [check_profile.py](check_profile.py) can check the proposal's
shape locally with Python 3.11 or newer, without extra packages:

```sh
python3 check_profile.py policy.json
```

Keep it beside the shipped `policy.json`; check a customized proposal under a
different filename so the reference template remains intact. Inputs are capped
at 64 KiB. The checker rejects unknown fields and malformed scalar types and
keeps the proposal's schedule disabled. It does not check path ownership,
credential validity, actual budgets, resource access or completeness. It cannot
detect arbitrary secrets hidden in string values. No field can turn a proposal
into a grant; runtime activation requires a separately reviewed integration.

| Test setting | Useful evidence | Does not establish |
| --- | --- | --- |
| Isolated CPU worker | Parsing, unit contracts, builds, sanitizers, mock devices | Kernel attachment, physical-device behavior, GPU execution |
| Disposable QEMU guest | Guest boot, kernel options, supported emulated devices, failure paths | Physical timing, firmware behavior, or hardware features the emulator lacks |
| Assigned bare-metal worker | Real drivers, devices, storage and recovery paths under a recorded configuration | Other devices or machines; permission to reboot or overwrite a disk |
| Assigned real GPU, local or cloud | Device kernels, memory behavior, driver/runtime integration | Other GPU vendors, CPU fallback equivalence, or physical storage access from a container |

QEMU is a machine emulator. Select a supported guest and device model for the
question; an unimplemented operation is unavailable coverage, not a pass.
For kernel work, identify required configuration, debug metadata, modules,
verifier/attach paths, guest console and timeout recovery. For a real GPU,
record vendor/runtime/API, required memory, devices exposed to the job and
whether fallback occurred. A rented GPU container is not necessarily a
bare-metal host and may not expose the required storage or privileged APIs.

Testing proposed code executes untrusted code. Use a disposable boundary with
no company model token, Git write credential, cloud control-plane token, SSH
agent, sensitive host mounts or production network access. A Git worktree is
not a sandbox. A container is not sufficient isolation for every hostile
kernel/device workload. Approve dependencies and egress through a controlled
provisioning step; keep privileged preparation outside the model's runner.

Run the least costly test that answers the question, then the required higher
fidelity checks. Do not substitute a CPU mock for a required real-GPU test,
or an emulated device for a required physical one. Compare the same checks at
base and head where useful, and test the intended integration revision.
After a base/head/configuration change, decide which checks must be repeated.
Never import test green status from a different revision without that link.

## Use company-approved model access

An existing managed agent login, approved API gateway or credential broker
can provide model access. This guide does not require a particular provider
or claim that a seat license permits unattended API use. Confirm the
company's provider, model, account, endpoint, project approval and automation
policy with its responsible owner. Record only a credential **reference**;
keep the credential in the approved manager. Do not copy browser sessions,
tokens or personal logins to make a provider adapter work.

Define permitted input classes before sending data: public code, internal
code, customer data, embargoed security reports and derived logs can need
different handling. Check the service's retention, training use, regional
processing, logging, access control and contractual requirements through the
company's approved policy. Public input can still be hostile. A crash log
can contain private paths or secrets. Do not silently change provider when a
quota fails or upload more data to diagnose an access problem.

Set per-task and daily request/token/cost/time limits with bounded retries.
Record measured usage where the service supplies it and mark unknown usage
as unknown. Subscription access is not a claim of zero cost. Independent
quotas and billing controls must enforce limits; a prompt cannot do so.
Inference, CI, storage and rented hardware have separate budgets. Cloud
creation, cancellation and teardown belong to a separately authorized
operator, with a receipt even when the test fails before useful work.

## Delegate deliberately

The owner chooses how much maintenance can run within a standing grant.
For a first pilot, permit read-only triage and local review drafts. A later
grant might admit known tests on a pre-provisioned disposable worker, then
bounded patch drafting. Posting a public review, adding a review label,
approving a PR, rewriting history, pushing and merging are separate actions.
The starter enables none of them automatically.

Record which actions always need a person and which named tests may run
without another prompt. A hardware operator can accept a bounded test request
without receiving merge rights or model credentials. Ask for human judgment
on conflicting requirements, compatibility changes, suspected vulnerabilities,
missing required hardware or disagreement that cannot be resolved from
evidence. Follow the project's private security-reporting route for an
embargoed issue; do not post an exploit or secret in a public review.

Use [HANDOFF.md](HANDOFF.md): identify the blocked decision, evidence, exact
requested action, resource and data boundaries, expected result and safe
fallback. A person supplies only the missing authority or fact. No response
pauses the affected task; it does not turn a request into approval. Suspected
unauthorized agent activity goes directly to the preconfigured owner route,
without waiting for the routine queue or permission from the suspected peer.

## Coordinate recurring work

Record a queue key such as repository + immutable head + chore type + policy
revision. Deduplicate observations before spending model calls. Prioritize
credible security/correctness issues, regressions and blocking reviews ahead
of cosmetic work, with the reason visible. Batch unchanged routine signals;
retain urgent reporting separately. Begin with a human admitting one task,
not a timer that starts agents because a dashboard is green.

For several workers, use one authoritative claim store with atomic acceptance,
expiring leases, heartbeats, attempt limits and a fencing generation that
prevents a former worker from submitting or publishing after losing its
lease. Task ownership and branch writer ownership are distinct. A second
reviewer does not need write rights. Cancellation stops new effects and
revokes publication authority; a lost worker does not transfer its authority
to a peer by conversation. Review recovery and same-user failure modes.

The starter does not implement that coordinator. Until one is selected and
tested, keep manual, single-worker admission. Do not use a network-shared
SQLite writer or independent edits to a JSON file as an assumed distributed
lock. Retain one durable ledger, immutable evidence by content hash and
bounded lightweight projections. Do not copy the whole database for every
dashboard refresh. Backup age, event observation time and dashboard refresh
time are different facts. A read-only dashboard is not an approval service.

## Review the first result

Require an executive conclusion: what matters, what was checked, what is
blocked and what the maintainer should do next. Keep raw reasoning out of the
main card. Link the exact base/head, command receipts, failures, skips,
unavailable tests, model usage and review evidence in detail. The
[review procedure](REVIEW.md) explains how to identify a bug fix from its diff
and when to ask for a squash without rewriting someone else's work.

Pilot one representative incoming change per project. Include an ordinary
review, a hardware-required case and a deliberately missing permission before
enabling recurrence. Measure useful findings accepted by maintainers, false
positives, review time, duplicate work, stale-result rejection, coverage gaps,
handoff delay, model cost and operator effort. A new AI-maintained project is
a possible later use; helping existing maintainers is useful without it.
