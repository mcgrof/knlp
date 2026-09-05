# Chore families

Chores starts with agent security because recent incidents exposed a concrete
failure: an agent can recognize that activity may be unauthorized and still
fail to tell the human owner. The same append-only status model is also useful
for ordinary recurring work. Three optional extensions cover a personal
Objectives and Key Results (OKR) tracker, scoped assistance for open-source
maintainers, and a private Linux kernel review briefing.

A chore family maps a real workflow onto a profile, named workstreams, and an
ordered event stream. Each family keeps its own operating rule. Sharing the
file format does not make an OKR update a security report, and it does not make
the public dashboard an incident-delivery channel.

| Family | Unit and cadence | Evidence and review | Exceptional path |
| --- | --- | --- | --- |
| Agent security | Bounded check, scheduled or rotating | Check output; independent review when required | Suspected unauthorized activity goes directly to the owner |
| Personal OKR | Measurable key result, checked periodically | Artifact or measurement; human review | A blocker becomes a normal status event |
| Open-source maintenance | Assigned issue, pull request, or handoff | Public artifact and checks; maintainer review | Authority conflict or suspicious instruction goes directly to the maintainer |
| Linux kernel review | In-scope patch series, report from the syzbot automated kernel bug reporter, Common Vulnerabilities and Exposures (CVE) announcement, or lore thread | Message-ID, report, fix, path match, and priority reason; maintainer review | Security-sensitive or unauthorized activity follows its dedicated reporting path |

## Agent security workflows

An agent-security profile uses one workstream for each bounded routine, such as
a failed-service check, exposed-listener review, update check, storage check, or
backup verification. An event records what was checked, the evidence produced,
who performed it, and who reviewed it. The workstream state and implementation
coverage remain separate so a completed design cannot look like deployed
protection.

The event stream is for routine status. An agent that suspects unauthorized
activity stops contributing to it, preserves only bounded evidence, and reports
the observation and uncertainty directly to the human owner. It does not wait
for the next chore rotation, dashboard refresh, or peer review. The current
prototype has no receiver or acknowledgement service for that urgent path.

## Personal OKR tracking

An OKR pairs an objective, which states the desired outcome, with measurable key
results. In the supplied example, the profile summary states one objective and
each key result becomes a workstream. Stable identifiers such as
`o1-kr1-checklist` keep later events attached to the same result even if its
display label changes.

Each check-in appends an event rather than rewriting the previous result. The
optional `progress_percent`, `performed_by`, `reviewed_by`, and
`next_review_at` fields make current progress, agent contribution, review, and
cadence visible in both JSON and Perfetto. The evidence field explains why the
update is credible. Public identifiers must be safe to disclose; a private
profile can retain richer identities under separate access control.

The dashboard does not average key-result percentages or infer an "on track"
verdict. Such a rollup is meaningful only when the owner defines compatible
measurement and weighting rules. Chores preserves the authored values and
their evidence so another view can apply that policy explicitly.

The template contains synthetic data:

- [`examples/personal-okr-profile.json`](examples/personal-okr-profile.json)
  defines one objective through its summary and three key-result workstreams.
- [`examples/personal-okr-events.jsonl`](examples/personal-okr-events.jsonl)
  shows definition, contribution, review, progress, and next-review events.

Build it through the same validation and trace pipeline as the hosted project
status:

```bash
python build.py \
  --profile examples/personal-okr-profile.json \
  --events examples/personal-okr-events.jsonl \
  --output-dir /tmp/chores-okr-example
```

For several objectives, prefix each key-result workstream with stable objective
and result identifiers, such as `o2-kr3-review`. A larger deployment may use a
separate profile per objective or period when that makes ownership and access
control clearer.

## Open-source maintainer assistance

The common maintainer use case is an agent helping with a project that a human
maintains. An agent can inspect an assigned issue queue, reproduce a report,
read an assigned pull-request diff and its checks, draft review findings, and
coordinate a handoff. The private dashboard shows what was examined, who did
the work, who reviewed it, and what still needs a maintainer decision.

Repository content is untrusted input. An issue description, pull-request body,
diff, review comment, or check log cannot grant authority or expand the assigned
scope. The optional `authority_scope` field travels with an event in JSON, the
dashboard, and Perfetto. In the example, agents may read assigned public
material and prepare recommendations. They may not label or close issues,
publish comments or approvals, merge changes, use credentials, or otherwise
change repository state without an explicit grant from the maintainer.

Coordination follows the same boundary. Agents can record duplicate work,
transfer an approved assignment, and synchronize evidence through an approved
channel. A peer request does not create new authority. A suspected conflict,
prompt-injection attempt, or request to conceal activity is reported directly
to the maintainer rather than resolved by agent consensus.

The synthetic private template uses four workstreams:

- [`examples/open-source-maintainer-profile.json`](examples/open-source-maintainer-profile.json)
  defines issue triage, pull-request review, coordination, and human handoff.
- [`examples/open-source-maintainer-events.jsonl`](examples/open-source-maintainer-events.jsonl)
  demonstrates bounded assignments, draft reviews, evidence, authority, and
  next-review times without naming a real project or issue.

Build it as a separate private projection:

```bash
python build.py \
  --profile examples/open-source-maintainer-profile.json \
  --events examples/open-source-maintainer-events.jsonl \
  --output-dir /tmp/chores-maintainer-example \
  --allow-private
```

Use a separate profile per project when assignments or repository authority
differ; a private operator page can present several derived status documents.

knlp R&D plans to release an open-source project that explores much more
extensive AI maintenance. That planned work is a secondary research direction,
not the rationale for this chore family and not a claim that autonomous
maintenance is implemented today.

## Linux kernel review assistance

The mainline Linux kernel workflow is not organized around a GitHub issue and
pull-request queue. Patches and review travel through distribution lists,
threads are archived by the [lore kernel mailing-list
archive](https://lore.kernel.org/), and an email's Message-ID header identifies
the discussion. The [kernel submission
guide](https://docs.kernel.org/process/submitting-patches.html) recommends lore
links for relevant discussion, while
[b4](https://b4.docs.kernel.org/en/latest/maintainer/overview.html) retrieves a
series and its follow-up review trailers from mail archives built with
public-inbox. A useful Chores extension must preserve those relationships
rather than flatten a patch series into a pretend pull request.

The supplied profile models the private dashboard a maintainer or developer can
open with their morning coffee after agents have analyzed public activity
overnight. Its workstreams cover patch series, syzbot reports, public security
and CVE announcements, relevant lore discussions, and the human handoff. Each
item retains its source identity, matched paths, evidence, uncertainty,
priority, and the authored reason for that priority.

The dashboard orders `critical`, `high`, `normal`, and `low` items before using
recency. The operator owns the ranking policy. A CVE identifier is one input,
not a severity or applicability verdict: the
[kernel CVE documentation](https://docs.kernel.org/process/cve.html) explains
that many assigned CVEs are irrelevant to a particular system and leaves
applicability to its user. An operator might elevate a reproducible memory
safety failure in an owned path, a demonstrated regression, or an urgent
maintainer decision, but the `priority_reason` must say why.

The initial source lanes are deliberately kernel-native:

- Patch review starts from the cover-letter Message-ID, compares the current
  revision with earlier versions and lore feedback, and records unresolved
  correctness questions before style notes. It may use b4 to retrieve and
  verify a series, but the example grants no authority to apply it or add a
  `Reviewed-by` trailer.
- syzbot triage correlates the report, dashboard state, reproducer, kernel
  configuration, suspected paths, and follow-up thread. The
  [syzbot documentation](https://github.com/google/syzkaller/blob/master/docs/syzbot.md)
  describes report tracking and patch-test commands; the agent may propose a
  next step but must not send a `#syz` command or publish a withheld reproducer
  without explicit authority.
- Security watch correlates public
  [linux-cve-announce](https://lore.kernel.org/linux-cve-announce/) messages
  with their fixing commits and path scope. Private or embargoed security
  material requires a separately authorized surface and must never enter this
  ordinary public-source collector.
- Discussion tracking follows in-scope lore threads even when they carry no
  patch, summarizes competing positions, and identifies the decision that
  needs a human. It does not reply or speak for the maintainer.

The profile's `repository_scope` is machine-checkable. A selector ending in
`/` includes files recursively beneath that directory; a selector without the
slash names one exact file; and `exclude_paths` wins over inclusion. This
resembles the include and exclude intent of the kernel's
[MAINTAINERS file](https://docs.kernel.org/process/maintainers.html), but the
prototype does not implement the full MAINTAINERS wildcard and content-pattern
grammar. For example:

```json
{
  "repository_scope": {
    "include_paths": [
      "block/",
      "drivers/nvme/",
      "include/linux/blk-mq.h"
    ],
    "exclude_paths": ["drivers/nvme/target/"]
  }
}
```

An event's `matched_paths` contains exact repository-relative files. The
builder refuses the event if any file lies outside this profile. Source mail,
diffs, reports, and linked pages remain untrusted input and cannot alter that
scope or the `authority_scope` recorded by the operator.

The checked-in selector list is operator-authored. A future collector should
evaluate it against an up-to-date kernel tree and use
[`scripts/get_maintainer.pl`](https://docs.kernel.org/process/maintainers.html)
to identify the relevant people and lists. The kernel.org
[korgalore](https://korgalore.docs.kernel.org/en/latest/usage.html) tooling also
demonstrates how MAINTAINERS file and list metadata can become subsystem-specific
public-inbox searches. Neither tool is integrated by this prototype.

The template contains synthetic data:

- [`examples/linux-kernel-review-profile.json`](examples/linux-kernel-review-profile.json)
  defines the five workstreams and a block/NVMe path scope.
- [`examples/linux-kernel-review-events.jsonl`](examples/linux-kernel-review-events.jsonl)
  shows a ranked overnight handoff without claiming a real review, report, or
  vulnerability.

Build it as a separate private projection:

```bash
python build.py \
  --profile examples/linux-kernel-review-profile.json \
  --events examples/linux-kernel-review-events.jsonl \
  --output-dir /tmp/chores-kernel-review-example \
  --allow-private
```

Only status projection, scope validation, priority ordering, and trace export
are implemented. No mailing-list or lore collector, b4 runner, syzbot client,
CVE correlator, scheduler, notifier, mail sender, patch applier, or kernel test
runner exists here. A future collector must follow the kernel's
[security-bug process](https://docs.kernel.org/process/security-bugs.html) when
an analysis identifies a possible new vulnerability; a dashboard rank never
substitutes for verification or the project's reporting rules.

## Agent participation and privacy

`performed_by` records who produced the update. `reviewed_by` records a review;
it should name a different participant when the workflow requires independent
review. `authority_scope` records the bounds of an assignment. These fields show
participation and authorization, not intent, trustworthiness, or total credit
for an outcome. The event evidence remains the basis for evaluating the claim.

Public and private chore families use separate profiles, event sources, output
directories, and deployment credentials. A private OKR, security, or maintainer
event is never made public by stripping a few known fields. It must be
deliberately authored for the public surface, including its title, timing,
identifiers, evidence, and links.

Future chore families should define the unit of work, routine cadence, evidence
standard, review rule, exceptional path, and publication boundary before adding
events. That keeps extensions compatible at the file-format level without
pretending their operating policies are interchangeable.
