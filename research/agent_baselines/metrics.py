"""Measure local prefix-cache token counters on a dedicated vLLM server.

The vLLM 0.13 metrics logger defines queries and hits in tokens:
https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/metrics/loggers/
Server totals cannot attribute reuse to individual or concurrent agents.
"""

import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request

QUERIES = "vllm:prefix_cache_queries_total"
HITS = "vllm:prefix_cache_hits_total"
PROMPTS = "vllm:prompt_tokens_total"
_SAMPLE = re.compile(r"([^\s{]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\[\\"n])*)"\s*(?:,|$)')


def snapshot(url):
    """Return raw exposition or an observable fetch failure; never invent zeros."""
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Metrics URL must be HTTP(S) without credentials or query")
    result = {"url": url, "ok": False, "raw_text": None, "error": None}
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            result["raw_text"] = response.read().decode("utf-8")
        result["ok"] = True
    except (OSError, UnicodeError, ValueError) as exc:
        # Exception messages can contain response bodies or proxy credentials.
        result["error"] = type(exc).__name__
    result["time_ns"] = time.time_ns()
    return result


def _parse(raw):
    counters = {name: {} for name in (QUERIES, HITS, PROMPTS)}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s{]", line, maxsplit=1)[0]
        if name not in counters:
            continue
        sample = _SAMPLE.fullmatch(line)
        if not sample:
            raise ValueError("Malformed counter sample")
        labels, position = {}, 0
        label_text = sample[2] or ""
        while position < len(label_text):
            match = _LABEL.match(label_text, position)
            if not match or match[1] in labels:
                raise ValueError("Malformed or duplicate label")
            labels[match[1]] = json.loads('"' + match[2] + '"')
            position = match.end()
        key = tuple(sorted(labels.items()))
        value = float(sample[3])
        if not math.isfinite(value) or value < 0 or key in counters[name]:
            raise ValueError("Invalid or duplicate counter")
        counters[name][key] = value
    return counters


def _deltas(before, after, name):
    old, new = before[name], after[name]
    if not old or not new:
        return None, "missing_counter"
    if old.keys() != new.keys():
        return None, "changed_series"
    delta = {key: new[key] - value for key, value in old.items()}
    if any(value < 0 for value in delta.values()):
        return None, "counter_reset"
    return delta, None


def compare_snapshots(before, after):
    """Compare matching series, then divide total hit tokens by query tokens.

    Two endpoint samples cannot detect a reset whose counters already surpassed
    their old values. Keep the server running and isolated throughout the run.
    """
    result = {
        "valid": False,
        "reason": None,
        "token_hit_ratio": None,
        "queried_tokens": None,
        "hit_tokens": None,
        "series": [],
        "prompt_tokens_delta": None,
        "prompt_tokens_reason": None,
        "scope": "server local prefix cache; no agent attribution",
    }
    if not before.get("ok") or not after.get("ok"):
        result["reason"] = "snapshot_unavailable"
        return result
    if before.get("url") != after.get("url"):
        result["reason"] = "changed_endpoint"
        return result
    if after["time_ns"] <= before["time_ns"]:
        result["reason"] = "unordered_snapshots"
        return result
    try:
        old, new = _parse(before["raw_text"]), _parse(after["raw_text"])
    except (ValueError, TypeError):
        result["reason"] = "invalid_exposition"
        return result
    prompts, error = _deltas(old, new, PROMPTS)
    result["prompt_tokens_reason"] = error
    if prompts is not None:
        result["prompt_tokens_delta"] = sum(prompts.values())
    queries, query_error = _deltas(old, new, QUERIES)
    hits, hit_error = _deltas(old, new, HITS)
    if query_error or hit_error:
        result["reason"] = query_error or hit_error
        return result
    if queries.keys() != hits.keys():
        result["reason"] = "unpaired_hit_query_series"
        return result
    if any(hits[key] > value for key, value in queries.items()):
        result["reason"] = "hits_exceed_queries"
        return result
    if prompts is not None and prompts.keys() != queries.keys():
        result["prompt_tokens_delta"] = None
        result["prompt_tokens_reason"] = "unpaired_prompt_series"
    for key in sorted(queries):
        result["series"].append(
            {
                "labels": dict(key),
                "queried_tokens": queries[key],
                "hit_tokens": hits[key],
            }
        )
    result["queried_tokens"] = sum(queries.values())
    result["hit_tokens"] = sum(hits.values())
    result["valid"] = True
    if result["queried_tokens"]:
        result["token_hit_ratio"] = result["hit_tokens"] / result["queried_tokens"]
    else:
        result["reason"] = "no_queried_tokens"
    return result
