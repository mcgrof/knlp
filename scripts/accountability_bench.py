#!/usr/bin/env python3
"""Accountability benchmark generator and scorer for long-history memory.

Needle tests ask whether one planted fact can be recovered. This suite
asks the harder question the Titans/Hope evaluation plan pre-registered:
whether a memory preserves source, order, multiplicity, correction
history, and coexisting contradictions, rather than collapsing them into
only the latest semantic state. Episodes are generated fresh and
deterministically from a seed, so any fixed test set can be regenerated,
and contamination is detectable by comparing accuracy on the declared
seed against a fresh one.

Eight families, one per failure mode:

    sources         the same fact stated by several people; who first,
                    how many, and who never said it
    correction      a value corrected one or more times; the current
                    value, the original, and who corrected it
    aba             a value changed away and back; the final state and
                    the history must remain distinguishable
    duplicates      semantically identical events with distinct
                    identities; exact counts and ordinals
    contradiction   incompatible claims that must coexist per source,
                    beside a control pair that agrees
    nesting         references through other events: values before a
                    correction, ordinal mentions, quoted attributions
                    through a chain of speakers
    exact           arbitrary material (identifiers, numbers, code)
                    that semantic smoothing destroys
    density         the interference matrix: live facts, restatements,
                    value similarity, filler, and placement varied
                    under recorded controls

Every query carries the question text, the canonical answer, ranking
options for log-prob evaluation of small models, machine-readable
parameters, and the supporting sentence indices. The scorer reports
normalized exact-match accuracy with bootstrap confidence intervals,
broken down by family, query type, and control.

The self-test regenerates the suite and runs two machine readers over
the rendered text alone. A parser oracle rebuilds the event log from
the sentences and re-derives every answer through the same resolver the
generator used; anything below 100% is a rendering or ground-truth bug.
A latest-state reader retains only each key's final value and most
recent speaker — the state a history-free memory would keep — and its
per-family scores measure which questions actually require history.

    python3 scripts/accountability_bench.py generate --seed 1 --out eps.jsonl
    python3 scripts/accountability_bench.py score --episodes eps.jsonl \\
        --predictions preds.jsonl --out report.json
    python3 scripts/accountability_bench.py selftest --out selftest.json
"""

import argparse
import hashlib
import json
import random
import re
import string
import sys

# ---------------------------------------------------------------------------
# vocabulary pools
# ---------------------------------------------------------------------------

COMMON_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "Dana",
    "Erin",
    "Frank",
    "Grace",
    "Henry",
    "Ines",
    "Jack",
    "Kara",
    "Liam",
    "Mona",
    "Nate",
    "Olga",
    "Pete",
]
RARE_NAMES = [
    "Thaddeus",
    "Ermentrude",
    "Beauregard",
    "Wilhelmina",
    "Ignatius",
    "Perpetua",
    "Bartholomew",
    "Seraphina",
    "Evanthia",
    "Cornelius",
]
KEY_ADJECTIVES = [
    "build",
    "backup",
    "staging",
    "primary",
    "fallback",
    "edge",
    "audit",
    "billing",
    "export",
    "ingest",
    "legacy",
    "mirror",
    "canary",
    "vault",
]
KEY_NOUNS = [
    "server",
    "gateway",
    "cluster",
    "queue",
    "bucket",
    "schedule",
    "index",
    "router",
    "token",
    "manifest",
    "pipeline",
    "registry",
    "ledger",
    "probe",
]
VALUE_POOL = [
    "zeus",
    "atlas",
    "hera",
    "argus",
    "helios",
    "rhea",
    "orion",
    "talos",
    "juno",
    "vesta",
    "milo",
    "echo",
    "iris",
    "leda",
    "numa",
    "remus",
    "castor",
    "pollux",
    "hydra",
    "lyra",
    "vega",
    "altair",
    "deneb",
    "mira",
]
CITIES = [
    "portland",
    "dover",
    "salem",
    "fresno",
    "tulsa",
    "boise",
    "reno",
    "fargo",
    "waco",
    "provo",
]
CODE_IDENTS = ["retries", "limit", "depth", "quota", "stride", "offset"]
FILLER = [
    "Routine checks continued through the afternoon.",
    "Nothing unusual appeared in the overnight summary.",
    "The team paused briefly for a scheduled meeting.",
    "Ambient readings stayed within the expected range.",
    "A quiet interval followed with no new activity.",
    "Monitoring dashboards showed steady load overall.",
    "The afternoon shift proceeded without incident.",
    "General maintenance chores filled the remaining hour.",
    "A short standup covered the usual coordination items.",
    "The weekly summary was filed at the usual time.",
    "Background jobs ran on their normal cadence.",
    "The on-call rotation changed hands without issues.",
]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

FAMILIES = (
    "sources",
    "correction",
    "aba",
    "duplicates",
    "contradiction",
    "nesting",
    "exact",
    "density",
)

DEFAULT_CONTROLS = dict(
    source_vocab="common",  # common | rare | random
    sources_per_fact=3,  # family: sources
    corrections=2,  # family: correction (chain length)
    duplicate_counts=(4, 2),  # family: duplicates (city1, city2)
    contradictors=2,  # family: contradiction
    quote_depth=2,  # family: nesting (speakers in the chain)
    exact_facts=4,  # family: exact
    n_live=6,  # family: density (live keys)
    mentions_per_key=1,  # family: density (restatements)
    value_similarity="distinct",  # distinct | similar
    filler_sentences=6,  # neutral sentences interleaved
    placement="spread",  # early | late | spread (density)
)


# ---------------------------------------------------------------------------
# deterministic randomness
# ---------------------------------------------------------------------------


def stable_rng(*parts):
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def random_name(rng):
    syllables = ["zor", "quim", "vel", "tak", "mur", "pex", "lod", "gan", "rus", "fim"]
    name = "".join(rng.choice(syllables) for _ in range(2))
    return name.capitalize()


def pick_sources(rng, controls, count):
    vocab = controls["source_vocab"]
    if vocab == "rare":
        pool = list(RARE_NAMES)
    elif vocab == "random":
        pool = []
        while len(pool) < count + 4:
            name = random_name(rng)
            if name not in pool:
                pool.append(name)
    else:
        pool = list(COMMON_NAMES)
    rng.shuffle(pool)
    return pool[: count + 4]  # extras serve as non-speakers and distractors


def pick_keys(rng, count):
    combos = [(a, n) for a in KEY_ADJECTIVES for n in KEY_NOUNS]
    rng.shuffle(combos)
    return [f"{a} {n}" for a, n in combos[:count]]


def pick_values(rng, controls, count):
    if controls["value_similarity"] == "similar":
        base = rng.choice(VALUE_POOL)
        values = [base]
        letters = string.ascii_lowercase
        while len(values) < count:
            pos = rng.randrange(len(base))
            repl = rng.choice(letters)
            cand = base[:pos] + repl + base[pos + 1 :]
            if cand not in values:
                values.append(cand)
        return values
    pool = list(VALUE_POOL)
    rng.shuffle(pool)
    return pool[:count]


def exact_value(rng, kind):
    if kind == "alnum":
        return "".join(
            rng.choice(string.ascii_lowercase + string.digits) for _ in range(10)
        )
    if kind == "uuid":
        hexd = "0123456789abcdef"
        parts = ("".join(rng.choice(hexd) for _ in range(n)) for n in (8, 4, 4))
        return "-".join(parts)
    if kind == "integer":
        return str(rng.randrange(10**6, 10**9))
    if kind == "hex":
        return "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(8))
    return f"{rng.choice(CODE_IDENTS)}={rng.randrange(2, 500)}"


# ---------------------------------------------------------------------------
# events, rendering, and the shared resolver
# ---------------------------------------------------------------------------
# An event is a dict: kind (claim|set|correct|note|record|quote), source,
# key, value, and for quotes the chain of intermediate speakers.  The
# generator builds ground truth by resolving queries against its event
# list; the parser oracle resolves the same queries against events parsed
# back out of the rendered text.  One resolver serves both.


def render(events, controls, rng):
    """Interleave events with neutral filler; return text and the sentence
    index of each event."""
    sentences = []
    for event in events:
        sentences.append(("event", event))
    filler_count = controls["filler_sentences"]
    for _ in range(filler_count):
        pos = rng.randrange(len(sentences) + 1)
        sentences.insert(pos, ("filler", rng.choice(FILLER)))
    text_parts = []
    event_index = {}
    for idx, (kind, payload) in enumerate(sentences):
        if kind == "filler":
            text_parts.append(payload)
        else:
            text_parts.append(event_text(payload))
            event_index[id(payload)] = idx
    positions = [event_index[id(e)] for e in events]
    return " ".join(text_parts), positions


def event_text(e):
    if e["kind"] == "claim":
        return f"{e['source']} says the {e['key']} is {e['value']}."
    if e["kind"] == "set":
        return f"{e['source']} sets the {e['key']} to {e['value']}."
    if e["kind"] == "correct":
        return f"{e['source']} corrects the {e['key']} to {e['value']}."
    if e["kind"] == "note":
        return f"{e['source']} notes the {e['key']} is {e['value']}."
    if e["kind"] == "record":
        return f"{e['source']} records a shipment to {e['value']}."
    if e["kind"] == "quote":
        chain = e["chain"]  # outermost speaker first, original speaker last
        head = f"{chain[0]} says that "
        middle = " ".join(f"{name} said that" for name in chain[1:-1])
        if middle:
            middle += " "
        return head + middle + f"{chain[-1]} said the {e['key']} is {e['value']}."
    raise ValueError(e["kind"])


PARSERS = [
    (
        "quote",
        re.compile(
            r"^(\w+) says that ((?:\w+ said that )*)(\w+) said the (\w+ \w+) is (.+)\.$"
        ),
    ),
    ("claim", re.compile(r"^(\w+) says the (\w+ \w+) is (.+)\.$")),
    ("set", re.compile(r"^(\w+) sets the (\w+ \w+) to (.+)\.$")),
    ("correct", re.compile(r"^(\w+) corrects the (\w+ \w+) to (.+)\.$")),
    ("note", re.compile(r"^(\w+) notes the (\w+ \w+) is (.+)\.$")),
    ("record", re.compile(r"^(\w+) records a shipment to (\w+)\.$")),
]


def parse_text(text):
    """Rebuild the event list from rendered text alone."""
    sentences = re.findall(r"[^.]+\.", text)
    events = []
    for sentence in sentences:
        sentence = sentence.strip()
        for kind, pattern in PARSERS:
            m = pattern.match(sentence)
            if not m:
                continue
            if kind == "quote":
                outer, middle, inner, key, value = m.groups()
                chain = [outer] + re.findall(r"(\w+) said that", middle) + [inner]
                events.append(
                    dict(kind="quote", chain=chain, source=inner, key=key, value=value)
                )
            elif kind == "record":
                source, city = m.groups()
                events.append(
                    dict(kind="record", source=source, key="shipment", value=city)
                )
            else:
                source, key, value = m.groups()
                events.append(dict(kind=kind, source=source, key=key, value=value))
            break
    return events


def resolve(events, params):
    """Answer a query from an event list.  This is the single source of
    truth for both ground-truth generation and the parser oracle."""
    kind = params["kind"]
    key = params.get("key")
    value = params.get("value")

    def for_key(k):
        return [e for e in events if e.get("key") == k and e["kind"] != "quote"]

    def value_history(k):
        return [
            e for e in for_key(k) if e["kind"] in ("claim", "set", "correct", "note")
        ]

    if kind == "who_first":
        hits = [e for e in for_key(key) if e["value"] == value]
        return hits[0]["source"]
    if kind == "source_count":
        hits = [e for e in for_key(key) if e["value"] == value]
        return str(len({e["source"] for e in hits}))
    if kind == "did_say":
        hits = [
            e
            for e in for_key(key)
            if e["value"] == value and e["source"] == params["source"]
        ]
        return "yes" if hits else "no"
    if kind == "current_value":
        return value_history(key)[-1]["value"]
    if kind == "original_value":
        return value_history(key)[0]["value"]
    if kind == "who_corrected_last":
        return [e for e in for_key(key) if e["kind"] == "correct"][-1]["source"]
    if kind == "who_original":
        return value_history(key)[0]["source"]
    if kind == "middle_value":
        return value_history(key)[-2]["value"]
    if kind == "change_count":
        history = value_history(key)
        return str(
            sum(1 for a, b in zip(history, history[1:]) if a["value"] != b["value"])
        )
    if kind == "ever_held":
        return "yes" if any(e["value"] == value for e in value_history(key)) else "no"
    if kind == "shipment_count":
        return str(
            sum(1 for e in events if e["kind"] == "record" and e["value"] == value)
        )
    if kind == "who_ordinal_shipment":
        hits = [e for e in events if e["kind"] == "record" and e["value"] == value]
        return hits[params["ordinal"]]["source"]
    if kind == "according_to":
        hits = [e for e in for_key(key) if e["source"] == params["source"]]
        return hits[-1]["value"]
    if kind == "agree":
        a = [e for e in for_key(key) if e["source"] == params["source_a"]][-1]["value"]
        b = [e for e in for_key(key) if e["source"] == params["source_b"]][-1]["value"]
        return "yes" if a == b else "no"
    if kind == "before_correction":
        history = value_history(key)
        idx = next(
            i
            for i, e in enumerate(history)
            if e["kind"] == "correct" and e["source"] == params["source"]
        )
        return history[idx - 1]["value"]
    if kind == "attributed":
        hits = [
            e
            for e in events
            if e["kind"] == "quote"
            and e["key"] == key
            and e["chain"][0] == params["source"]
            and e["chain"][-1] == params["speaker"]
        ]
        return hits[-1]["value"]
    if kind == "ordinal_mention":
        hits = [e for e in for_key(key) if e["source"] == params["source"]]
        return hits[params["ordinal"]]["value"]
    if kind == "reverse_lookup":
        hits = [e for e in events if e["kind"] == "note" and e["value"] == value]
        return hits[-1]["key"]
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# query assembly
# ---------------------------------------------------------------------------


def make_query(
    episode, qtype, question, params, events, positions, support_events, options, rng
):
    answer = resolve(events, params)
    index_by_id = {id(e): i for i, e in enumerate(events)}
    support = sorted(positions[index_by_id[id(e)]] for e in support_events)
    # cap the distractors first, then place the answer by shuffle, so the
    # answer's position carries no signal and every list holds at most six
    distractors = [o for o in dict.fromkeys(options) if o and o != answer][:5]
    opts = distractors + [answer]
    rng.shuffle(opts)
    episode["queries"].append(
        dict(
            query_id=f"{episode['episode_id']}-q{len(episode['queries'])}",
            query_type=qtype,
            question=question,
            answer=answer,
            params=params,
            options=opts,
            support=support,
        )
    )


def count_options(n, rng):
    opts = {n}
    while len(opts) < 4:
        delta = rng.choice([-2, -1, 1, 2, 3])
        if n + delta >= 0:
            opts.add(n + delta)
    # sorted so the emitted artifact is byte-identical across processes
    return [str(v) for v in sorted(opts)]


def base_episode(family, index, base_seed, controls):
    return dict(
        episode_id=f"{family}-{index:04d}-s{base_seed}",
        family=family,
        seed=base_seed,
        controls={
            k: (list(v) if isinstance(v, tuple) else v) for k, v in controls.items()
        },
        context=None,
        queries=[],
    )


# ---------------------------------------------------------------------------
# family generators
# ---------------------------------------------------------------------------


def gen_sources(rng, episode, controls):
    m = controls["sources_per_fact"]
    sources = pick_sources(rng, controls, m + 2)
    keys = pick_keys(rng, 3)
    values = pick_values(rng, controls, 3)
    key, val = keys[0], values[0]
    speakers = sources[:m]
    non_speaker = sources[m]
    events = [dict(kind="claim", source=s, key=key, value=val) for s in speakers]
    # decoy facts on other keys keep the target from being the only content
    for k, v, s in zip(keys[1:], values[1:], sources[m + 1 :]):
        events.append(dict(kind="claim", source=s, key=k, value=v))
    rng.shuffle(events)
    claim_events = [e for e in events if e["key"] == key]
    return events, [
        (
            "who_said_first",
            f"Who said the {key} is {val} first?",
            dict(kind="who_first", key=key, value=val),
            claim_events[:1],
            sources,
        ),
        (
            "source_count",
            f"How many people said the {key} is {val}? Answer with a number.",
            dict(kind="source_count", key=key, value=val),
            claim_events,
            count_options(m, rng),
        ),
        (
            "did_say_yes",
            f"Did {speakers[-1]} say the {key} is {val}? Answer yes or no.",
            dict(kind="did_say", key=key, value=val, source=speakers[-1]),
            [e for e in claim_events if e["source"] == speakers[-1]],
            ["yes", "no"],
        ),
        (
            "did_say_no",
            f"Did {non_speaker} say the {key} is {val}? Answer yes or no.",
            dict(kind="did_say", key=key, value=val, source=non_speaker),
            claim_events,
            ["yes", "no"],
        ),
    ]


def gen_correction(rng, episode, controls):
    chain = controls["corrections"]
    sources = pick_sources(rng, controls, chain + 1)
    key = pick_keys(rng, 1)[0]
    values = pick_values(rng, controls, chain + 1)
    events = [dict(kind="claim", source=sources[0], key=key, value=values[0])]
    for i in range(chain):
        events.append(
            dict(kind="correct", source=sources[i + 1], key=key, value=values[i + 1])
        )
    return events, [
        (
            "current_value",
            f"What is the {key} now?",
            dict(kind="current_value", key=key),
            events[-1:],
            values,
        ),
        (
            "original_value",
            f"What value was first given for the {key}?",
            dict(kind="original_value", key=key),
            events[:1],
            values,
        ),
        (
            "who_corrected_last",
            f"Who corrected the {key} last?",
            dict(kind="who_corrected_last", key=key),
            events[-1:],
            sources,
        ),
        (
            "who_original",
            f"Who gave the first value for the {key}?",
            dict(kind="who_original", key=key),
            events[:1],
            sources,
        ),
    ]


def gen_aba(rng, episode, controls):
    sources = pick_sources(rng, controls, 3)
    key = pick_keys(rng, 1)[0]
    a, b = pick_values(rng, controls, 2)
    never = pick_values(rng, controls, 3)[-1]
    if never in (a, b):
        never = [v for v in VALUE_POOL if v not in (a, b)][0]
    events = [
        dict(kind="set", source=sources[0], key=key, value=a),
        dict(kind="set", source=sources[1], key=key, value=b),
        dict(kind="set", source=sources[2], key=key, value=a),
    ]
    return events, [
        (
            "current_value",
            f"What is the {key} now?",
            dict(kind="current_value", key=key),
            events[-1:],
            [a, b, never],
        ),
        (
            "middle_value",
            f"What value did the {key} hold immediately before its final change?",
            dict(kind="middle_value", key=key),
            events[1:2],
            [a, b, never],
        ),
        (
            "change_count",
            f"How many times was the {key} changed after it was first set? Answer with a number.",
            dict(kind="change_count", key=key),
            events,
            count_options(2, rng),
        ),
        (
            "ever_held_yes",
            f"Did the {key} ever hold {b}? Answer yes or no.",
            dict(kind="ever_held", key=key, value=b),
            events[1:2],
            ["yes", "no"],
        ),
        (
            "ever_held_no",
            f"Did the {key} ever hold {never}? Answer yes or no.",
            dict(kind="ever_held", key=key, value=never),
            events,
            ["yes", "no"],
        ),
    ]


def gen_duplicates(rng, episode, controls):
    m1, m2 = controls["duplicate_counts"]
    sources = pick_sources(rng, controls, m1 + m2)
    cities = list(CITIES)
    rng.shuffle(cities)
    city1, city2 = cities[:2]
    events = [
        dict(kind="record", source=sources[i], key="shipment", value=city1)
        for i in range(m1)
    ]
    events += [
        dict(kind="record", source=sources[m1 + i], key="shipment", value=city2)
        for i in range(m2)
    ]
    rng.shuffle(events)
    city1_events = [e for e in events if e["value"] == city1]
    ordinal = rng.randrange(min(m1, len(ORDINALS)))
    return events, [
        (
            "shipment_count",
            f"How many shipments to {city1} were recorded? Answer with a number.",
            dict(kind="shipment_count", value=city1),
            city1_events,
            count_options(m1, rng),
        ),
        (
            "shipment_count_2",
            f"How many shipments to {city2} were recorded? Answer with a number.",
            dict(kind="shipment_count", value=city2),
            [e for e in events if e["value"] == city2],
            count_options(m2, rng),
        ),
        (
            "who_ordinal_shipment",
            f"Who recorded the {ORDINALS[ordinal]} shipment to {city1}?",
            dict(kind="who_ordinal_shipment", value=city1, ordinal=ordinal),
            city1_events[ordinal : ordinal + 1],
            sources[: m1 + m2],
        ),
    ]


def gen_contradiction(rng, episode, controls):
    n = controls["contradictors"]
    sources = pick_sources(rng, controls, n + 2)
    keys = pick_keys(rng, 2)
    values = pick_values(rng, controls, n + 1)
    key = keys[0]
    events = [
        dict(kind="claim", source=sources[i], key=key, value=values[i])
        for i in range(n)
    ]
    # control pair that agrees, on a second key
    agree_key, agree_val = keys[1], values[n]
    events.append(dict(kind="claim", source=sources[n], key=agree_key, value=agree_val))
    events.append(
        dict(kind="claim", source=sources[n + 1], key=agree_key, value=agree_val)
    )
    rng.shuffle(events)
    return events, [
        (
            "according_to_a",
            f"According to {sources[0]}, what is the {key}?",
            dict(kind="according_to", key=key, source=sources[0]),
            [e for e in events if e["key"] == key and e["source"] == sources[0]],
            values,
        ),
        (
            "according_to_b",
            f"According to {sources[1]}, what is the {key}?",
            dict(kind="according_to", key=key, source=sources[1]),
            [e for e in events if e["key"] == key and e["source"] == sources[1]],
            values,
        ),
        (
            "agree_no",
            f"Do {sources[0]} and {sources[1]} agree about the {key}? Answer yes or no.",
            dict(kind="agree", key=key, source_a=sources[0], source_b=sources[1]),
            [e for e in events if e["key"] == key],
            ["yes", "no"],
        ),
        (
            "agree_yes",
            f"Do {sources[n]} and {sources[n + 1]} agree about the {agree_key}? Answer yes or no.",
            dict(
                kind="agree",
                key=agree_key,
                source_a=sources[n],
                source_b=sources[n + 1],
            ),
            [e for e in events if e["key"] == agree_key],
            ["yes", "no"],
        ),
    ]


def gen_nesting(rng, episode, controls):
    depth = max(2, controls["quote_depth"])
    sources = pick_sources(rng, controls, depth + 3)
    keys = pick_keys(rng, 2)
    values = pick_values(rng, controls, 5)
    key, key2 = keys
    v0, v1, w1, w2, w3 = values
    claimer, corrector = sources[0], sources[1]
    chain = sources[2 : 2 + depth]
    # three mentions of key2 so the queried middle one is answerable
    # neither from the final state nor from the first impression
    events = [
        dict(kind="claim", source=claimer, key=key, value=v0),
        dict(kind="correct", source=corrector, key=key, value=v1),
        dict(
            kind="quote",
            chain=[*chain[:-1], claimer],
            source=claimer,
            key=key,
            value=v0,
        ),
        dict(kind="claim", source=corrector, key=key2, value=w1),
        dict(kind="claim", source=corrector, key=key2, value=w2),
        dict(kind="claim", source=corrector, key=key2, value=w3),
    ]
    quoter = chain[0]
    return events, [
        (
            "before_correction",
            f"What was the value of the {key} before {corrector}'s correction?",
            dict(kind="before_correction", key=key, source=corrector),
            events[:1],
            values,
        ),
        (
            "attributed",
            f"According to {quoter}, what did {claimer} say the {key} is?",
            dict(kind="attributed", key=key, source=quoter, speaker=claimer),
            events[2:3],
            values,
        ),
        (
            "ordinal_mention",
            f"The second time {corrector} gave a value for the {key2}, what value did they give?",
            dict(kind="ordinal_mention", key=key2, source=corrector, ordinal=1),
            events[4:5],
            values,
        ),
    ]


def gen_exact(rng, episode, controls):
    n = controls["exact_facts"]
    kinds = ["alnum", "uuid", "integer", "hex", "code"]
    sources = pick_sources(rng, controls, n)
    keys = pick_keys(rng, n)
    events = []
    exacts = []
    for i in range(n):
        val = exact_value(rng, kinds[i % len(kinds)])
        exacts.append(val)
        events.append(dict(kind="note", source=sources[i], key=keys[i], value=val))
    rng.shuffle(events)
    target = rng.randrange(n)
    tkey, tval = keys[target], exacts[target]
    tevent = [e for e in events if e["key"] == tkey]
    rtarget = (target + 1) % n
    rkey, rval = keys[rtarget], exacts[rtarget]
    return events, [
        (
            "exact_value",
            f"What is the {tkey}?",
            dict(kind="current_value", key=tkey),
            tevent,
            exacts,
        ),
        (
            "reverse_lookup",
            f"Which item has the value {rval}?",
            dict(kind="reverse_lookup", value=rval),
            [e for e in events if e["key"] == rkey],
            keys,
        ),
    ]


def gen_density(rng, episode, controls):
    n = controls["n_live"]
    mentions = controls["mentions_per_key"]
    sources = pick_sources(rng, controls, n * mentions)
    keys = pick_keys(rng, n)
    values = pick_values(rng, controls, n)
    events = []
    for i in range(n):
        for j in range(mentions):
            src = sources[(i * mentions + j) % len(sources)]
            events.append(dict(kind="claim", source=src, key=keys[i], value=values[i]))
    placement = controls["placement"]
    if placement == "spread":
        rng.shuffle(events)
        # placement of the queried key is decided after the shuffle below
    target_key = keys[0]
    target_events = [e for e in events if e["key"] == target_key]
    if placement in ("early", "late"):
        others = [e for e in events if e["key"] != target_key]
        rng.shuffle(others)
        events = (
            (target_events + others)
            if placement == "early"
            else (others + target_events)
        )
    probe = rng.choice(keys[1:] or [target_key])
    queries = [
        (
            "value_of",
            f"What is the {target_key}?",
            dict(kind="current_value", key=target_key),
            target_events[-1:],
            values,
        ),
        (
            "value_of_2",
            f"What is the {probe}?",
            dict(kind="current_value", key=probe),
            [e for e in events if e["key"] == probe][-1:],
            values,
        ),
        (
            "who_said_first",
            f"Who said the {target_key} is {values[0]} first?",
            dict(kind="who_first", key=target_key, value=values[0]),
            [e for e in events if e["key"] == target_key][:1],
            sources,
        ),
    ]
    return events, queries


GENERATORS = dict(
    sources=gen_sources,
    correction=gen_correction,
    aba=gen_aba,
    duplicates=gen_duplicates,
    contradiction=gen_contradiction,
    nesting=gen_nesting,
    exact=gen_exact,
    density=gen_density,
)


def generate_episode(family, index, base_seed, controls):
    rng = stable_rng(base_seed, family, index, "episode")
    episode = base_episode(family, index, base_seed, controls)
    events, query_specs = GENERATORS[family](rng, episode, controls)
    text, positions = render(
        events, controls, stable_rng(base_seed, family, index, "render")
    )
    episode["context"] = text
    episode["words"] = len(text.split())
    qrng = stable_rng(base_seed, family, index, "queries")
    for qtype, question, params, support_events, options in query_specs:
        make_query(
            episode,
            qtype,
            question,
            params,
            events,
            positions,
            support_events,
            options,
            qrng,
        )
    return episode


def generate(base_seed, per_family, families, controls_override=None):
    controls = dict(DEFAULT_CONTROLS)
    if controls_override:
        controls.update(controls_override)
    episodes = []
    for family in families:
        for i in range(per_family):
            episodes.append(generate_episode(family, i, base_seed, controls))
    return episodes


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def normalize(text):
    text = str(text).strip().lower()
    text = text.strip(".,;:'\"!?")
    return re.sub(r"\s+", " ", text)


def score(episodes, predictions, bootstrap=1000, seed=0):
    """Every query counts: an unanswered query is scored as wrong, so
    selective answering cannot inflate accuracy.  Coverage is reported
    separately.  Duplicate predictions for one query keep the last
    submission; predictions for unknown query ids are counted and
    ignored."""
    known = {q["query_id"] for ep in episodes for q in ep["queries"]}
    preds = {}
    unknown = 0
    for pred in predictions:
        if pred["query_id"] in known:
            preds[pred["query_id"]] = pred["prediction"]
        else:
            unknown += 1
    rows = []
    for ep in episodes:
        for q in ep["queries"]:
            answered = q["query_id"] in preds
            rows.append(
                dict(
                    episode_id=ep["episode_id"],
                    family=ep["family"],
                    query_type=q["query_type"],
                    answered=answered,
                    correct=answered
                    and normalize(preds[q["query_id"]]) == normalize(q["answer"]),
                )
            )

    def acc(subset):
        return (
            sum(r["correct"] for r in subset) / len(subset) if subset else float("nan")
        )

    report = dict(
        overall=acc(rows),
        coverage=(sum(r["answered"] for r in rows) / len(rows)) if rows else 0.0,
        n_queries=len(rows),
        n_episodes=len(episodes),
        unknown_prediction_ids=unknown,
        by_family={},
        by_query_type={},
    )
    for r in rows:
        report["by_family"].setdefault(r["family"], []).append(r["correct"])
        report["by_query_type"].setdefault(r["query_type"], []).append(r["correct"])
    for keyname in ("by_family", "by_query_type"):
        report[keyname] = {
            k: dict(accuracy=sum(v) / len(v), n=len(v))
            for k, v in sorted(report[keyname].items())
        }
    # bootstrap over episodes
    rng = random.Random(seed)
    per_ep = {}
    for r in rows:
        per_ep.setdefault(r["episode_id"], []).append(r["correct"])
    ep_ids = sorted(per_ep)
    if ep_ids:
        means = []
        for _ in range(bootstrap):
            sample = [rng.choice(ep_ids) for _ in ep_ids]
            flat = [c for e in sample for c in per_ep[e]]
            means.append(sum(flat) / len(flat))
        means.sort()
        report["bootstrap_95ci"] = [
            means[int(0.025 * len(means))],
            means[int(0.975 * len(means)) - 1],
        ]
    else:
        report["bootstrap_95ci"] = [float("nan"), float("nan")]
    return report


# ---------------------------------------------------------------------------
# machine readers for the self-test
# ---------------------------------------------------------------------------


def oracle_predict(episodes):
    """Parse the rendered text back into events and re-derive every answer
    with the shared resolver.  Anything below 100% is a generator bug."""
    preds = []
    for ep in episodes:
        events = parse_text(ep["context"])
        for q in ep["queries"]:
            preds.append(
                dict(query_id=q["query_id"], prediction=resolve(events, q["params"]))
            )
    return preds


def latest_state_predict(episodes):
    """A reader that keeps, per key, only the final asserted value and
    the most recent speaker — the state a history-free memory retains —
    plus, per shipment destination, its last recorder.  Quoted
    attributions are treated as reports, not assertions, matching the
    resolver's definition of a value history."""
    preds = []
    for ep in episodes:
        events = parse_text(ep["context"])
        final = {}
        last_recorder = {}
        for e in events:
            if e["kind"] == "quote":
                continue
            if e["kind"] == "record":
                last_recorder[e["value"]] = e["source"]
                continue
            final[e["key"]] = (e["value"], e["source"])
        rng = stable_rng(ep["episode_id"], "latest-state")

        def answer(q):
            params = q["params"]
            kind = params["kind"]
            key = params.get("key")
            state = final.get(key, (None, None))
            if kind in (
                "current_value",
                "original_value",
                "middle_value",
                "before_correction",
                "attributed",
                "ordinal_mention",
                "according_to",
            ):
                return state[0] or rng.choice(q["options"])
            if kind in ("who_first", "who_corrected_last", "who_original"):
                return state[1] or rng.choice(q["options"])
            if kind == "who_ordinal_shipment":
                return last_recorder.get(params["value"]) or rng.choice(q["options"])
            if kind in ("source_count", "change_count", "shipment_count"):
                return "1"
            if kind == "did_say":
                ok = state == (params["value"], params["source"])
                return "yes" if ok else "no"
            if kind == "ever_held":
                return "yes" if state[0] == params["value"] else "no"
            if kind == "agree":
                return "yes"
            if kind == "reverse_lookup":
                for k, (v, _) in final.items():
                    if v == params["value"]:
                        return k
                return rng.choice(q["options"])
            return rng.choice(q["options"])

        for q in ep["queries"]:
            preds.append(dict(query_id=q["query_id"], prediction=answer(q)))
    return preds


def selftest(base_seed, per_family, out_path, keep_episodes=None):
    grids = [
        dict(),
        dict(source_vocab="random", value_similarity="similar"),
        dict(corrections=3, quote_depth=3, filler_sentences=12),
        dict(n_live=10, mentions_per_key=2, placement="early"),
    ]
    episodes = []
    for gi, grid in enumerate(grids):
        controls = dict(DEFAULT_CONTROLS)
        controls.update(grid)
        for family in FAMILIES:
            for i in range(per_family):
                episodes.append(
                    generate_episode(family, gi * per_family + i, base_seed, controls)
                )
    if keep_episodes:
        with open(keep_episodes, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

    results = {}
    for name, reader in (
        ("parser_oracle", oracle_predict),
        ("latest_state", latest_state_predict),
    ):
        preds = reader(episodes)
        results[name] = score(episodes, preds, bootstrap=500, seed=base_seed)

    summary = dict(
        seed=base_seed,
        n_episodes=len(episodes),
        n_queries=sum(len(e["queries"]) for e in episodes),
        mean_words=sum(e["words"] for e in episodes) / len(episodes),
        readers=results,
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)

    oracle_acc = results["parser_oracle"]["overall"]
    latest = results["latest_state"]
    print(
        f"episodes {len(episodes)}, queries {summary['n_queries']}, "
        f"mean context {summary['mean_words']:.0f} words"
    )
    print(f"parser oracle: {oracle_acc:.4f} (must be 1.0)")
    print(
        f"latest-state reader: {latest['overall']:.4f} "
        f"(CI {latest['bootstrap_95ci'][0]:.3f}-{latest['bootstrap_95ci'][1]:.3f})"
    )
    print("latest-state by family:")
    for fam, entry in latest["by_family"].items():
        print(f"  {fam:14s} {entry['accuracy']:.3f}  (n={entry['n']})")
    ok = oracle_acc == 1.0 and latest["overall"] < 0.9
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write episodes as JSONL")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--per-family", type=int, default=25)
    g.add_argument("--families", default="all")
    g.add_argument("--controls", default=None, help="JSON dict of control overrides")
    g.add_argument("--out", required=True)

    s = sub.add_parser("score", help="score a predictions JSONL against episodes")
    s.add_argument("--episodes", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--bootstrap", type=int, default=1000)

    t = sub.add_parser("selftest", help="generate, run machine readers, verify")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--per-family", type=int, default=10)
    t.add_argument("--out", default="accountability_selftest.json")
    t.add_argument("--keep-episodes", default=None)

    args = ap.parse_args()

    if args.cmd == "generate":
        families = (
            FAMILIES if args.families == "all" else tuple(args.families.split(","))
        )
        overrides = json.loads(args.controls) if args.controls else None
        episodes = generate(args.seed, args.per_family, families, overrides)
        with open(args.out, "w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")
        print(f"{len(episodes)} episodes -> {args.out}")
        return 0

    if args.cmd == "score":
        with open(args.episodes) as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        with open(args.predictions) as f:
            predictions = [json.loads(line) for line in f if line.strip()]
        report = score(episodes, predictions, bootstrap=args.bootstrap)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
        print(
            f"overall {report['overall']:.4f} "
            f"(CI {report['bootstrap_95ci'][0]:.3f}-{report['bootstrap_95ci'][1]:.3f}) "
            f"-> {args.out}"
        )
        return 0

    if args.cmd == "selftest":
        return selftest(args.seed, args.per_family, args.out, args.keep_episodes)

    return 1


if __name__ == "__main__":
    sys.exit(main())
