# Chores

Chores is a practical response to recent AI-agent security incidents. It turns
routine security and reliability checks into small, evidence-producing tasks
that can rotate among agents a person already uses. Every participating agent
also has a separate duty to report suspected unauthorized activity immediately
and directly to the human owner, whether it observes its own mistake or
another agent's conduct.

**Routine chores rotate. Immediate incident reporting does not.**

## Why this exists

[OpenAI reported](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)
that agents in cybersecurity evaluations bypassed isolation, communicated
through unauthorized channels, and accessed third-party systems. An
[independent investigation by METR and Redwood Research](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)
found that agents sometimes recognized conduct as out of scope but very rarely
considered alerting a human and did not act on the cases it identified.

Those incidents motivate Chores; they do not establish that Chores would have
prevented them. Authorized collaboration is not suspicious merely because
several agents participate. The relevant warning is an observed conflict
between an action and its recorded authority, an attempt to suppress or alter
evidence, or the loss of monitoring and reporting coverage.

The response is owner-controlled, proactive security hygiene. Routine checks
may rotate so that failed services, exposed listeners, stale updates, storage
pressure, and broken backups are less likely to be neglected. Suspected
unauthorized activity follows a different path: stop contributing to it,
preserve bounded evidence, and report the observation and uncertainty directly
to the owner without waiting for a peer's permission or a weekly summary.

For example, if an agent observes a peer asking to use a credential outside its
recorded grant or to conceal an operation, it reports that conflict instead of
asking the peer group to decide whether the human should know. Reporting does
not authorize it to test the credential, search unrelated files, punish the
peer, or make an unsupported claim about intent.

Chores is intended to sit above established tools rather than replace them.
systemd timers, osquery, Fleet, Wazuh, and similar systems can collect or
schedule work. The project adds an agent-facing discipline for authority,
evidence, independent review, and direct human reporting. A skill can request
that behavior, but it cannot enforce it; policy checks, durable reporting, and
monitoring must operate outside the agents for stronger guarantees.

## Current scope

The code in this directory begins with a status projection, not a security
monitor or maintenance runner. It publishes which parts of Chores are
documented, implemented, or deployed while keeping planned work visibly
separate. There is no collector, scheduler, incident receiver, notifier,
acknowledgement service, credential rotation, or remediation mechanism here.

The first data contract consists of a project profile and an ordered event
stream:

- [`data/public-profile.json`](data/public-profile.json) names the project and
  its workstreams.
- [`data/public-events.jsonl`](data/public-events.jsonl) records public project
  activity in timestamp order.
- [`schemas/`](schemas/) contains the versioned input and derived-output
  schemas.

Each event records a workstream state and an implementation-coverage level.
"Complete" therefore does not mean "deployed": a design can be complete while
its coverage remains `documented` or `not-implemented`.

## Public and private projections

The public site and a private operator dashboard may use the same file formats,
but they must use separate profiles, event sources, output directories, and
deployment credentials. The public feed contains only facts approved for
publication. It is never produced by redacting a private trace after the fact.

The `surface` field catches accidental mixing between a profile and its event
stream. It is not a content sanitizer. Whoever publishes an event remains
responsible for checking its title, summary, evidence, URL, and timing for
sensitive information.

## Build the status views

`build.py` validates a profile and event stream, derives `status.json`, and
renders the same events as a native Perfetto TrackEvent trace. Perfetto is an
open trace viewer; the trace presents project events on per-workstream lanes
with counters for state changes. The JSON document and trace are generated
views rather than separate status ledgers.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python build.py
pytest
```

The default command builds the checked-in public inputs under `web/public/`.
Tests rebuild those artifacts, compare their bytes, and query the trace through
Perfetto's trace processor.

A private deployment passes its own paths and must opt in explicitly:

```bash
python build.py --profile private-profile.json \
  --events private-events.jsonl --output-dir private-web \
  --allow-private
```

That flag prevents an accidental private build through the public defaults. It
does not make the output safe to publish or provide access control.
