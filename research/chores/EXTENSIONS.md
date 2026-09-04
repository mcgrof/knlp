# Chore families

Chores starts with agent security because recent incidents exposed a concrete
failure: an agent can recognize that activity may be unauthorized and still
fail to tell the human owner. The same append-only status model is also useful
for ordinary recurring work. Two optional extensions cover a personal
Objectives and Key Results (OKR) tracker and scoped assistance for open-source
maintainers.

A chore family maps a real workflow onto a profile, named workstreams, and an
ordered event stream. Each family keeps its own operating rule. Sharing the
file format does not make an OKR update a security report, and it does not make
the public dashboard an incident-delivery channel.

| Family | Unit and cadence | Evidence and review | Exceptional path |
| --- | --- | --- | --- |
| Agent security | Bounded check, scheduled or rotating | Check output; independent review when required | Suspected unauthorized activity goes directly to the owner |
| Personal OKR | Measurable key result, checked periodically | Artifact or measurement; human review | A blocker becomes a normal status event |
| Open-source maintenance | Assigned issue, pull request, or handoff | Public artifact and checks; maintainer review | Authority conflict or suspicious instruction goes directly to the maintainer |

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
