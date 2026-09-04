# Chore families

Chores starts with agent security because recent incidents exposed a concrete
failure: an agent can recognize that activity may be unauthorized and still
fail to tell the human owner. The same append-only status model is also useful
for ordinary recurring work. The first optional extension is a personal
Objectives and Key Results (OKR) tracker.

A chore family maps a real workflow onto a profile, named workstreams, and an
ordered event stream. Each family keeps its own operating rule. Sharing the
file format does not make an OKR update a security report, and it does not make
the public dashboard an incident-delivery channel.

| Concern | Agent security workflows | Personal OKR tracking |
| --- | --- | --- |
| Unit of work | A bounded check or review | A measurable key result |
| Routine cadence | Scheduled or rotating | Periodic check-in |
| Event evidence | Check output or review record | Artifact or measurement behind an update |
| Agent participation | Performer and, when required, an independent reviewer | Contributor and human reviewer |
| Exceptional path | Suspected unauthorized activity goes directly to the owner | A blocker becomes a normal status event |
| Progress percentage | Usually omitted | Optional and never a substitute for evidence |

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

## Agent participation and privacy

`performed_by` records who produced the update. `reviewed_by` records a review;
it should name a different participant when the workflow requires independent
review. These fields show participation, not intent, trustworthiness, or total
credit for an outcome. The event evidence remains the basis for evaluating the
claim.

Public and private chore families use separate profiles, event sources, output
directories, and deployment credentials. A private OKR or security event is
never made public by stripping a few known fields. It must be deliberately
authored for the public surface, including its title, timing, identifiers,
evidence, and links.

Future chore families should define the unit of work, routine cadence, evidence
standard, review rule, exceptional path, and publication boundary before adding
events. That keeps extensions compatible at the file-format level without
pretending their operating policies are interchangeable.
