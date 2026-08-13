# SPDX-License-Identifier: MIT
"""hydra: trace-driven GPU residency simulator and policy boundary map.

Discrete-event simulation of one GPU with a fixed HBM budget serving
a small model catalog under different routing policies, driven by two
measured inputs:

  * an engine cost model (cost_model_probe.py output): per-model
    cold/warm engine bring-up medians, first-token medians, and
    weight bytes;
  * a labeled routing trace (phase2 bench + labels): per-query
    correctness and GPU-seconds for every model in the catalog plus
    a scalar requirement label in [0, 1].

Per-model residency states and transition costs:

  awake       serves queries; holds weight_bytes * HBM_OVERHEAD of HBM
  asleep_l1   weights backed to CPU RAM (level-1 sleep); wake 0.47 s,
              eviction into it 3.74 s (both measured); asleep weights
              count against a CPU RAM budget
  asleep_l2   weights dropped (level-2 sleep); wake 0.41 s measured
              BUT token parity after wake was never verified, so this
              state is gated behind --allow-l2 and always reported as
              unverified
  down_warm   engine process down, weight file in page cache; start =
              measured warm engine-init median (~18-22 s)
  down_cold   process down, page cache cold; start = measured cold
              engine-init median
  absent      not staged on local NVMe; start = cold start plus a
              declared 1 GB/s staging placeholder (unmeasured; unused
              by the standard sweeps)

Declared modeling assumptions (limitations, echoed into the output
JSON so no reader can miss them):

  1. Service times are the trace's batch-amortized GPU-seconds
     (milliseconds per query), not interactive prefill+decode
     latencies, so queueing almost never differentiates policies and
     residency transitions (0.5-29 s) dominate.  TTFT is modeled as
     queue wait + transition wait + the serving model's measured
     first-token median.
  2. Single server: the GPU serves one query at a time
     (work-conserving, earliest arrival among awake models).  Real
     vLLM batches concurrently; C_queue here is a model, not a
     measurement.
  3. One residency transition chain runs at a time (evict, then
     wake); engine init and wakes do not block serving awake models.
  4. Wake/sleep constants were measured once on a 3B model and are
     applied to every model in the catalog.
  5. Sessions are synthetic: the trace is single-turn, so sessions
     are consecutive same-group draws with a geometric length, and
     the prefix-cache benefit of staying on the same model is a
     declared fraction of service time saved, not a measurement.
  6. The requirement column stands in for the capability predictor:
     requirement >= STRONG_REQ routes to the strong class, otherwise
     to the cheapest model whose per-dimension trace accuracy covers
     the requirement.
  7. The oracle is correctness-clairvoyant per query and greedy over
     residency; it is not the offline-optimal schedule.

Policies (each picks a model per query; a shared resolver realizes
the pick against residency state):

  semantic   requirement-threshold pick, residency-blind
  load       + per-model queued-work awareness, residency-blind
  cache      sticky-session: keep the session's previous model when
             it stays semantically eligible (prefix-cache retention)
  residency  semantic eligibility + minimize realized cost including
             wake/start transitions, with min-dwell hysteresis
  oracle     correctness-clairvoyant pick, residency-aware realization

Determinism: a seeded numpy Generator drives arrivals and query
draws; policies and the event loop are deterministic (heap ties break
on a sequence counter).  No wall-clock dependence.
"""

import argparse
import heapq
import json
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Measured transition constants (engine cost probe, level-1/2 sleep
# probe).  Wake/sleep were measured on the 3B model and applied
# pool-wide (declared assumption 4).
WAKE_L1_S = 0.47
SLEEP_L1_S = 3.74
WAKE_L2_S = 0.41  # token parity UNVERIFIED; gated behind --allow-l2
SLEEP_L2_S = 0.01
STOP_S = 1.0  # process stop cost, declared assumption (unmeasured)
ABSENT_STAGE_BPS = 1.0e9  # declared staging placeholder (unmeasured)

HBM_OVERHEAD = 1.25  # declared KV/activation reserve over weights
RAM_BUDGET_BYTES = 59 * 2**30  # CPU RAM budget for L1-backed weights

STRONG_REQ = 0.5  # requirement >= this routes to the strong class
STRONG_BAND = 0.05  # strong class = within this of the best accuracy
CACHE_HIT_FRACTION = 0.30  # declared prefix-cache service saving
DWELL_DEFAULT_S = 60.0  # min-dwell hysteresis

MMPP_HI_MULT = 3.0
MMPP_LO_MULT = 1.0 / 3.0
MMPP_HI_DWELL_S = 60.0
MMPP_LO_DWELL_S = 180.0

SESSION_MEAN_LEN = 3
SESSION_MAX_LEN = 8

AWAKE = "awake"
WAKING = "waking"
SLEEPING = "sleeping"
ASLEEP_L1 = "asleep_l1"
ASLEEP_L2 = "asleep_l2"
DOWN_WARM = "down_warm"
DOWN_COLD = "down_cold"
ABSENT = "absent"

CATALOG5 = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
CATALOG3 = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

ASSUMPTIONS = [
    "service times are batch-amortized trace GPU-seconds, not "
    "interactive latencies; C_queue is a model, not a measurement",
    "single-server GPU, work-conserving earliest-arrival dispatch "
    "among awake models",
    "one transition chain at a time; transitions do not block "
    "serving of awake models",
    "wake/sleep constants measured once on a 3B model, applied to " "all models",
    "sessions synthesized from a single-turn trace; prefix-cache "
    "benefit is a declared 30% service saving on sticky hits",
    "requirement column is a stand-in for the capability predictor",
    "oracle is correctness-clairvoyant greedy, not offline-optimal",
    "asleep_l2 wake time measured but token parity UNVERIFIED; "
    "excluded from headline results",
]


@dataclass
class ModelSpec:
    name: str
    weight_bytes: int
    hbm_bytes: float
    start_warm_s: float
    start_cold_s: float
    first_token_s: float
    mean_gpu_s: float
    cap: dict = field(default_factory=dict)  # dim -> train accuracy


def load_specs(cost_model_path, trace_df, names):
    with open(cost_model_path) as f:
        cm = json.load(f)
    train = trace_df[trace_df["split"] == "train"]
    specs = {}
    for name in names:
        warm = cm[f"{name}|warm"]
        cold = cm[f"{name}|cold"]
        cap = train.groupby("dim")[f"correct::{name}"].mean().to_dict()
        specs[name] = ModelSpec(
            name=name,
            weight_bytes=warm["weight_bytes"],
            hbm_bytes=warm["weight_bytes"] * HBM_OVERHEAD,
            start_warm_s=warm["init_seconds"]["median"],
            start_cold_s=cold["init_seconds"]["median"],
            first_token_s=warm["first_token_seconds"]["median"],
            mean_gpu_s=float(trace_df[f"gpu_s::{name}"].mean()),
            cap=cap,
        )
    return specs


class Workload:
    """Arrival times plus a query stream with synthetic sessions."""

    def __init__(
        self,
        rng,
        trace,
        zipf_s=1.0,
        proc="poisson",
        rate=0.5,
        horizon=3600.0,
        sessions=True,
        forced_frac=0.0,
        thrash_period=None,
    ):
        self.rng = rng
        self.trace = trace
        self.sessions = sessions
        self.forced_frac = forced_frac
        self.thrash_period = thrash_period
        df = trace
        self.bench_order = list(df["benchmark"].value_counts().index)
        ranks = np.arange(1, len(self.bench_order) + 1, dtype=float)
        w = ranks**-zipf_s
        self.bench_p = w / w.sum()
        self.bench_groups = {
            b: sorted(df[df["benchmark"] == b]["group"].unique())
            for b in self.bench_order
        }
        self.group_idx = {
            g: df.index[df["group"] == g].to_numpy() for g in df["group"].unique()
        }
        self.hi_pool = df.index[df["requirement"] >= 0.75].to_numpy()
        self.lo_pool = df.index[df["requirement"] == 0.0].to_numpy()
        self.arrivals = self._arrival_times(proc, rate, horizon)
        self._sess_left = 0
        self._sess_group = None
        self._sess_id = -1

    def _arrival_times(self, proc, rate, horizon):
        rng = self.rng
        ts = []
        t = 0.0
        if proc == "poisson":
            while True:
                t += rng.exponential(1.0 / rate)
                if t > horizon:
                    break
                ts.append(t)
            return ts
        # two-state MMPP, mean rate preserved by construction
        hi = bool(rng.integers(2))
        switch = t + rng.exponential(MMPP_HI_DWELL_S if hi else MMPP_LO_DWELL_S)
        while t <= horizon:
            r = rate * (MMPP_HI_MULT if hi else MMPP_LO_MULT)
            dt = rng.exponential(1.0 / r)
            if t + dt > switch:
                t = switch
                hi = not hi
                switch = t + rng.exponential(MMPP_HI_DWELL_S if hi else MMPP_LO_DWELL_S)
                continue
            t += dt
            if t > horizon:
                break
            ts.append(t)
        return ts

    def _draw_session(self):
        rng = self.rng
        bench = self.bench_order[int(rng.choice(len(self.bench_order), p=self.bench_p))]
        groups = self.bench_groups[bench]
        self._sess_group = groups[int(rng.integers(len(groups)))]
        length = 1 + int(rng.geometric(1.0 / SESSION_MEAN_LEN))
        self._sess_left = min(length, SESSION_MAX_LEN)
        self._sess_id += 1

    def query_at(self, t):
        """Return (trace row index, session id, forced requirement)."""
        rng = self.rng
        if self.thrash_period is not None:
            pool = (
                self.hi_pool if int(t // self.thrash_period) % 2 == 0 else self.lo_pool
            )
            self._sess_id += 1
            qi = int(pool[int(rng.integers(len(pool)))])
            return qi, self._sess_id, None
        if not self.sessions or self._sess_left == 0:
            self._draw_session()
        self._sess_left -= 1
        idx = self.group_idx[self._sess_group]
        qi = int(idx[int(rng.integers(len(idx)))])
        forced = None
        if self.forced_frac > 0.0 and rng.random() < self.forced_frac:
            forced = 1.0
        return qi, self._sess_id, forced


class Sim:
    """One policy, one workload, one GPU: the discrete-event core."""

    def __init__(
        self,
        specs,
        catalog,
        hbm_budget,
        policy,
        dwell_s,
        fallback,
        allow_l2=False,
    ):
        self.specs = specs
        # HBM admission: a model whose reserved bytes exceed the
        # budget can never be awake and leaves the effective catalog.
        self.catalog = [m for m in catalog if specs[m].hbm_bytes <= hbm_budget]
        self.hbm_budget = hbm_budget
        self.policy = policy
        self.dwell_s = dwell_s
        self.fallback = fallback
        self.allow_l2 = allow_l2
        caps = {m: np.mean(list(specs[m].cap.values())) for m in self.catalog}
        self.strongest = max(self.catalog, key=lambda m: caps[m])
        self.cost_order = sorted(self.catalog, key=lambda m: specs[m].mean_gpu_s)
        self._elig_cache = {}

    # -- semantic layer -------------------------------------------------
    def eligible(self, dim, req):
        key = (dim, req)
        got = self._elig_cache.get(key)
        if got is not None:
            return got
        if req >= STRONG_REQ:
            best = max(self.specs[m].cap[dim] for m in self.catalog)
            out = [
                m
                for m in self.cost_order
                if self.specs[m].cap[dim] >= best - STRONG_BAND
            ]
        else:
            out = [m for m in self.cost_order if self.specs[m].cap[dim] >= req]
            if not out:
                out = [self.strongest]
        self._elig_cache[key] = out
        return out

    def semantic_pick(self, dim, req):
        if req >= STRONG_REQ:
            return max(self.catalog, key=lambda m: self.specs[m].cap[dim])
        return self.eligible(dim, req)[0]

    # -- residency machinery --------------------------------------------
    def _reserved_hbm(self):
        return sum(
            self.specs[m].hbm_bytes
            for m in self.catalog
            if self.status[m] in (AWAKE, WAKING, SLEEPING)
        )

    def _wake_cost(self, m):
        st = self.status[m]
        sp = self.specs[m]
        if st == ASLEEP_L1:
            return WAKE_L1_S
        if st == ASLEEP_L2:
            return WAKE_L2_S
        if st == DOWN_WARM:
            return sp.start_warm_s
        if st == DOWN_COLD:
            return sp.start_cold_s
        if st == ABSENT:
            return sp.start_cold_s + sp.weight_bytes / ABSENT_STAGE_BPS
        return 0.0

    def realized_score(self, m, now):
        sp = self.specs[m]
        st = self.status[m]
        trans = 0.0
        if st in (AWAKE, WAKING):
            trans = 0.0
        elif m in self.wake_set:
            trans = 0.5
        else:
            trans = self._wake_cost(m)
            need = sp.hbm_bytes
            if self.hbm_budget - self._reserved_hbm() < need:
                trans += SLEEP_L2_S if self.allow_l2 else SLEEP_L1_S
        queued = len(self.queues[m]) * sp.mean_gpu_s
        return trans + queued + sp.mean_gpu_s

    # -- policy picks ----------------------------------------------------
    def pick(self, qi, dim, req, sess, now):
        pol = self.policy
        if pol == "semantic":
            return self.semantic_pick(dim, req)
        if pol == "load":
            elig = self.eligible(dim, req)
            return min(
                elig,
                key=lambda m: (
                    len(self.queues[m]) * self.specs[m].mean_gpu_s
                    + self.specs[m].mean_gpu_s,
                    self.specs[m].mean_gpu_s,
                    m,
                ),
            )
        if pol == "cache":
            prev = self.sess_model.get(sess)
            if prev is not None and prev in self.eligible(dim, req):
                return prev
            return self.semantic_pick(dim, req)
        if pol == "residency":
            elig = self.eligible(dim, req)
            return min(
                elig,
                key=lambda m: (
                    round(self.realized_score(m, now), 6),
                    self.specs[m].mean_gpu_s,
                    m,
                ),
            )
        if pol == "oracle":
            correct = [m for m in self.catalog if self.correct[m][qi]]
            base = correct if correct else [self.strongest]
            return min(
                base,
                key=lambda m: (
                    round(self.realized_score(m, now), 6),
                    self.specs[m].mean_gpu_s,
                    m,
                ),
            )
        raise ValueError(pol)

    # -- event loop ------------------------------------------------------
    def run(self, workload, trace):
        specs = self.specs
        df = trace
        self.correct = {}
        gpu = {}
        req_col = df["requirement"].to_numpy()
        dim_col = df["dim"].to_numpy()
        for m in self.catalog:
            self.correct[m] = df[f"correct::{m}"].to_numpy()
            gpu[m] = df[f"gpu_s::{m}"].to_numpy()
        # to allow dict-style row access in pick()
        self.correct = {
            m: {i: bool(v) for i, v in enumerate(self.correct[m])} for m in self.catalog
        }

        self.status = {m: DOWN_COLD for m in self.catalog}
        self.status[self.strongest] = AWAKE
        self.awake_since = {m: 0.0 for m in self.catalog}
        self.last_used = {m: 0.0 for m in self.catalog}
        self.queues = {m: deque() for m in self.catalog}
        self.wake_fifo = deque()
        self.wake_set = set()
        self.wake_override = {}
        self.wake_noevict = set()
        self.trans_busy = False
        self.sess_model = {}
        self.last_activation = -1e9
        self.serving = None
        self.server_busy = False
        self.ram_used = 0.0

        counts = {
            "wake_l1": 0,
            "wake_l2": 0,
            "start_warm": 0,
            "start_cold": 0,
            "sleep": 0,
            "stop": 0,
            "fallback_retarget": 0,
            "bg_wake_req": 0,
        }
        records = []
        heap = []
        seq = 0

        def push(t, kind, data):
            nonlocal seq
            heapq.heappush(heap, (t, seq, kind, data))
            seq += 1

        for i, t in enumerate(workload.arrivals):
            qi, sess, forced = workload.query_at(t)
            push(t, "arrive", (i, qi, sess, forced))

        def request_wake(m, now, override, noevict=False):
            if self.status[m] in (AWAKE, WAKING) or m in self.wake_set:
                return
            self.wake_set.add(m)
            self.wake_override[m] = override
            if noevict:
                self.wake_noevict.add(m)
            self.wake_fifo.append(m)
            process_transitions(now)

        def pick_victim(now, override):
            cands = [
                m for m in self.catalog if self.status[m] == AWAKE and m != self.serving
            ]
            if not override and self.dwell_s > 0:
                cands = [m for m in cands if now - self.awake_since[m] >= self.dwell_s]
            if not cands:
                return None
            empty = [m for m in cands if not self.queues[m]]
            pool = empty if empty else cands
            return min(pool, key=lambda m: (self.last_used[m], m))

        def process_transitions(now):
            if self.trans_busy or not self.wake_fifo:
                return
            m = self.wake_fifo[0]
            need = specs[m].hbm_bytes
            free = self.hbm_budget - self._reserved_hbm()
            if free < need and m in self.wake_noevict:
                # background wakes never evict: drop the request
                self.wake_fifo.popleft()
                self.wake_set.discard(m)
                self.wake_noevict.discard(m)
                process_transitions(now)
                return
            if free >= need:
                self.wake_fifo.popleft()
                self.wake_set.discard(m)
                self.wake_noevict.discard(m)
                cost = self._wake_cost(m)
                st = self.status[m]
                if st == ASLEEP_L1:
                    counts["wake_l1"] += 1
                    self.ram_used -= specs[m].weight_bytes
                elif st == ASLEEP_L2:
                    counts["wake_l2"] += 1
                elif st == DOWN_WARM:
                    counts["start_warm"] += 1
                else:
                    counts["start_cold"] += 1
                self.status[m] = WAKING
                self.trans_busy = True
                self.last_activation = now
                push(now + cost, "wake_done", m)
                return
            victim = pick_victim(now, self.wake_override.get(m, False))
            if victim is None:
                push(now + 1.0, "retry", None)
                return
            self.trans_busy = True
            self.status[victim] = SLEEPING
            if self.allow_l2:
                push(now + SLEEP_L2_S, "sleep_done", (victim, ASLEEP_L2))
                counts["sleep"] += 1
            elif self.ram_used + specs[victim].weight_bytes <= RAM_BUDGET_BYTES:
                push(now + SLEEP_L1_S, "sleep_done", (victim, ASLEEP_L1))
                counts["sleep"] += 1
            else:
                push(now + STOP_S, "sleep_done", (victim, DOWN_WARM))
                counts["stop"] += 1

        def dispatch(now):
            if self.server_busy:
                return
            best = None
            for m in self.catalog:
                if self.status[m] == AWAKE and self.queues[m]:
                    head = self.queues[m][0]
                    if best is None or head[0] < best[0][0]:
                        best = (head, m)
            if best is None:
                # progress guarantee: re-request the earliest stalled
                # queue's model if nothing is awake or pending for it
                stalled = [
                    (self.queues[m][0][0], m)
                    for m in self.catalog
                    if self.queues[m]
                    and self.status[m] not in (AWAKE, WAKING)
                    and m not in self.wake_set
                ]
                if stalled:
                    stalled.sort()
                    request_wake(stalled[0][1], now, True)
                return
            (arr, qi, sess, forced), m = best
            self.queues[m].popleft()
            svc = gpu[m][qi]
            hit = self.sess_model.get(sess) == m
            if hit:
                svc *= 1.0 - CACHE_HIT_FRACTION
            self.server_busy = True
            self.serving = m
            self.last_used[m] = now
            ttft = (now - arr) + specs[m].first_token_s
            records.append((arr, ttft, 1.0 if self.correct[m][qi] else 0.0, svc))
            push(now + svc, "svc_done", (m, sess))

        drain_cap = (workload.arrivals[-1] if workload.arrivals else 0.0) + 7200.0
        while heap:
            now, _, kind, data = heapq.heappop(heap)
            if now > drain_cap:
                # safety net: a policy that cannot drain within two
                # hours past the last arrival is saturated; count the
                # backlog as unserved instead of looping
                break
            if kind == "arrive":
                _, qi, sess, forced = data
                req = forced if forced is not None else req_col[qi]
                dim = dim_col[qi]
                m = self.pick(qi, dim, req, sess, now)
                if self.status[m] not in (AWAKE, WAKING) and m not in self.wake_set:
                    awake_elig = [
                        e
                        for e in self.eligible(dim, req)
                        if self.status[e] in (AWAKE, WAKING)
                    ]
                    if (
                        self.fallback
                        and awake_elig
                        and now - self.last_activation < self.dwell_s
                    ):
                        m = min(
                            awake_elig,
                            key=lambda e: (
                                round(self.realized_score(e, now), 6),
                                self.specs[e].mean_gpu_s,
                                e,
                            ),
                        )
                        counts["fallback_retarget"] += 1
                    else:
                        request_wake(m, now, not awake_elig)
                if self.policy in ("residency", "oracle"):
                    # background non-evicting wake: converge toward
                    # the semantic-ideal working set whenever free HBM
                    # allows, so stickiness never inflates GPU cost
                    # when capacity is actually available
                    ideal = self.semantic_pick(dim, req)
                    if (
                        ideal != m
                        and self.status[ideal] not in (AWAKE, WAKING)
                        and ideal not in self.wake_set
                        and self.hbm_budget - self._reserved_hbm()
                        >= specs[ideal].hbm_bytes
                    ):
                        counts["bg_wake_req"] += 1
                        request_wake(ideal, now, False, noevict=True)
                self.queues[m].append((now, qi, sess, forced))
                dispatch(now)
            elif kind == "svc_done":
                m, sess = data
                self.server_busy = False
                self.serving = None
                self.sess_model[sess] = m
                dispatch(now)
                process_transitions(now)
            elif kind == "wake_done":
                m = data
                self.status[m] = AWAKE
                self.awake_since[m] = now
                self.trans_busy = False
                # dispatch BEFORE further transitions: the woken model
                # must get to serve (becoming eviction-protected) or a
                # standing wake FIFO would evict it instantly and two
                # queued models could mutually evict forever
                dispatch(now)
                process_transitions(now)
            elif kind == "sleep_done":
                victim, target = data
                self.status[victim] = target
                if target == ASLEEP_L1:
                    self.ram_used += specs[victim].weight_bytes
                self.trans_busy = False
                process_transitions(now)
                dispatch(now)
            elif kind == "retry":
                process_transitions(now)
                dispatch(now)

        unserved = sum(len(q) for q in self.queues.values())
        return records, counts, unserved


def metrics_from(records, counts, unserved, horizon, warmup_s):
    rec = [r for r in records if r[0] >= warmup_s]
    if not rec:
        return None
    ttft = np.array([r[1] for r in rec])
    quality = float(np.mean([r[2] for r in rec]))
    gpu_s = float(np.mean([r[3] for r in rec]))
    hours = horizon / 3600.0
    trans = (
        counts["wake_l1"]
        + counts["wake_l2"]
        + counts["start_warm"]
        + counts["start_cold"]
        + counts["sleep"]
        + counts["stop"]
    )
    n_all = len(records) + unserved
    return {
        "n": len(rec),
        "unserved": unserved,
        "saturated": bool(unserved > 0.05 * max(n_all, 1)),
        "ttft_mean": float(ttft.mean()),
        "ttft_p50": float(np.quantile(ttft, 0.50)),
        "ttft_p95": float(np.quantile(ttft, 0.95)),
        "ttft_p99": float(np.quantile(ttft, 0.99)),
        "quality": quality,
        "gpu_s_per_query": gpu_s,
        "transitions_per_hour": trans / hours,
        "counts": dict(counts),
    }


def pooled_metrics(per_seed):
    """Average scalar metrics across seeds (declared aggregation)."""
    keys = [
        "ttft_mean",
        "ttft_p50",
        "ttft_p95",
        "ttft_p99",
        "quality",
        "gpu_s_per_query",
        "transitions_per_hour",
    ]
    out = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
    out["n"] = int(np.sum([m["n"] for m in per_seed]))
    out["unserved"] = int(np.sum([m["unserved"] for m in per_seed]))
    out["saturated"] = bool(any(m["saturated"] for m in per_seed))
    out["counts"] = {
        k: int(np.sum([m["counts"][k] for m in per_seed]))
        for k in per_seed[0]["counts"]
    }
    return out


POLICY_DEFS = {
    "semantic": dict(dwell_s=0.0, fallback=False),
    "load": dict(dwell_s=0.0, fallback=False),
    "cache": dict(dwell_s=0.0, fallback=False),
    "residency": dict(dwell_s=DWELL_DEFAULT_S, fallback=True),
    "oracle": dict(dwell_s=DWELL_DEFAULT_S, fallback=True),
}


def run_cell(
    specs,
    trace,
    catalog,
    hbm_budget,
    policy,
    seeds,
    horizon,
    warmup_s,
    workload_kw,
    dwell_s=None,
    fallback=None,
    allow_l2=False,
):
    pdef = POLICY_DEFS[policy]
    dw = pdef["dwell_s"] if dwell_s is None else dwell_s
    fb = pdef["fallback"] if fallback is None else fallback
    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        wl = Workload(rng, trace, horizon=horizon, **workload_kw)
        sim = Sim(
            specs,
            catalog,
            hbm_budget,
            policy,
            dw,
            fb,
            allow_l2=allow_l2,
        )
        rec, counts, unserved = sim.run(wl, trace)
        m = metrics_from(rec, counts, unserved, horizon, warmup_s)
        if m is not None:
            per_seed.append(m)
    return pooled_metrics(per_seed)


def classify(deltas):
    win = (deltas["dp95"] >= 0.30 or deltas["dcost"] >= 0.15) and deltas[
        "dq_pp"
    ] >= -2.0
    neutral = deltas["dp95"] < 0.05 and deltas["dcost"] < 0.05
    if win:
        return "residency_wins"
    if neutral:
        return "neutral"
    return "mid"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--cost-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--horizon", type=float, default=3600.0)
    ap.add_argument("--warmup", type=float, default=300.0)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.5, 2.0])
    ap.add_argument("--zipf", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--budgets-gb", type=float, nargs="+", default=[20.0, 24.0, 48.0])
    ap.add_argument(
        "--allow-l2",
        action="store_true",
        help="enable the UNVERIFIED level-2 sleep state (appendix only)",
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    trace = pd.read_parquet(args.trace)
    specs = load_specs(args.cost_model, trace, CATALOG5)
    seeds = list(range(1, args.seeds + 1))
    horizon = args.horizon
    warmup = args.warmup
    if args.quick:
        horizon, warmup = 600.0, 60.0
        args.rates = [0.5]
        args.zipf = [1.0]
        args.budgets_gb = [20.0, 48.0]

    catalogs = {"3": CATALOG3, "5": CATALOG5}
    policies = list(POLICY_DEFS)

    sweep = []
    for proc in ["poisson", "mmpp"]:
        for rate in args.rates:
            for zipf_s in args.zipf:
                for cname, cat in catalogs.items():
                    for bgb in args.budgets_gb:
                        cell = {
                            "arrival": proc,
                            "rate_per_s": rate,
                            "zipf_s": zipf_s,
                            "catalog": cname,
                            "hbm_gb": bgb,
                            "policies": {},
                        }
                        for pol in policies:
                            cell["policies"][pol] = run_cell(
                                specs,
                                trace,
                                cat,
                                bgb * 1e9,
                                pol,
                                seeds,
                                horizon,
                                warmup,
                                dict(zipf_s=zipf_s, proc=proc, rate=rate),
                            )
                        sem = cell["policies"]["semantic"]
                        res = cell["policies"]["residency"]
                        deltas = {
                            "dp95": 1.0 - res["ttft_p95"] / max(sem["ttft_p95"], 1e-9),
                            "dcost": 1.0
                            - res["gpu_s_per_query"]
                            / max(sem["gpu_s_per_query"], 1e-12),
                            "dq_pp": (res["quality"] - sem["quality"]) * 100.0,
                        }
                        cell["residency_vs_semantic"] = deltas
                        cell["region"] = classify(deltas)
                        sweep.append(cell)
                        print(
                            f"[sweep] {proc} rate={rate} zipf={zipf_s} "
                            f"cat={cname} hbm={bgb} -> {cell['region']} "
                            f"dp95={deltas['dp95']:+.2f} "
                            f"dcost={deltas['dcost']:+.2f} "
                            f"dq={deltas['dq_pp']:+.1f}pp",
                            flush=True,
                        )

    # thrash-adversarial arm: oscillating strong/cheap demand at
    # several periods, hysteresis on/off
    thrash = []
    periods = [30.0, 60.0, 120.0, 300.0] if not args.quick else [60.0]
    thrash_arms = [
        ("semantic", dict(dwell_s=0.0, fallback=False)),
        ("semantic_hyst", dict(dwell_s=DWELL_DEFAULT_S, fallback=True)),
        ("residency_d0", dict(dwell_s=0.0, fallback=True)),
        ("residency_d60", dict(dwell_s=DWELL_DEFAULT_S, fallback=True)),
    ]
    for period in periods:
        entry = {"period_s": period, "hbm_gb": 20.0, "policies": {}}
        for label, over in thrash_arms:
            pol = "semantic" if label.startswith("semantic") else "residency"
            entry["policies"][label] = run_cell(
                specs,
                trace,
                CATALOG5,
                20.0e9,
                pol,
                seeds,
                horizon,
                warmup,
                dict(
                    zipf_s=1.0,
                    proc="poisson",
                    rate=0.5,
                    sessions=False,
                    thrash_period=period,
                ),
                **over,
            )
        thrash.append(entry)
        print(f"[thrash] period={period}s done", flush=True)

    # adversarial arm: a fraction of queries steered to requirement
    # 1.0; measure induced cost inflation vs the f=0 baseline
    adversarial = []
    fracs = [0.0, 0.1, 0.3] if not args.quick else [0.0, 0.3]
    for bgb in [20.0, 48.0]:
        base = {}
        for f in fracs:
            entry = {"hbm_gb": bgb, "forced_frac": f, "policies": {}}
            for pol in ["semantic", "residency"]:
                entry["policies"][pol] = run_cell(
                    specs,
                    trace,
                    CATALOG5,
                    bgb * 1e9,
                    pol,
                    seeds,
                    horizon,
                    warmup,
                    dict(zipf_s=1.0, proc="poisson", rate=0.5, forced_frac=f),
                )
            if f == 0.0:
                base = entry["policies"]
            else:
                entry["inflation_vs_f0"] = {
                    pol: {
                        "gpu_s": entry["policies"][pol]["gpu_s_per_query"]
                        / max(base[pol]["gpu_s_per_query"], 1e-12),
                        "p95": entry["policies"][pol]["ttft_p95"]
                        / max(base[pol]["ttft_p95"], 1e-9),
                        "transitions": entry["policies"][pol]["transitions_per_hour"]
                        / max(base[pol]["transitions_per_hour"], 1e-9),
                    }
                    for pol in ["semantic", "residency"]
                }
            adversarial.append(entry)
            print(f"[adv] hbm={bgb} f={f} done", flush=True)

    # optional appendix: the UNVERIFIED level-2 sleep state
    l2_appendix = []
    if args.allow_l2:
        for pol in ["semantic", "residency"]:
            l2_appendix.append(
                {
                    "hbm_gb": 20.0,
                    "policy": pol,
                    "token_parity": "UNVERIFIED",
                    "metrics": run_cell(
                        specs,
                        trace,
                        CATALOG5,
                        20.0e9,
                        pol,
                        seeds,
                        horizon,
                        warmup,
                        dict(zipf_s=1.0, proc="poisson", rate=0.5),
                        allow_l2=True,
                    ),
                }
            )

    wins = [c for c in sweep if c["region"] == "residency_wins"]
    neutrals = [c for c in sweep if c["region"] == "neutral"]
    gate = {
        "win_cells": len(wins),
        "neutral_cells": len(neutrals),
        "mid_cells": len(sweep) - len(wins) - len(neutrals),
        "verdict": "PASS" if wins and neutrals else "FAIL",
        "criteria": (
            "PASS requires >=1 cell with (p95 TTFT improvement >= 30% "
            "or GPU-seconds saving >= 15%) at quality within 2 pp, AND "
            ">=1 cell with both improvements < 5%"
        ),
    }

    out = {
        "config": {
            "seeds": seeds,
            "horizon_s": horizon,
            "warmup_s": warmup,
            "hbm_overhead": HBM_OVERHEAD,
            "ram_budget_bytes": RAM_BUDGET_BYTES,
            "strong_req": STRONG_REQ,
            "strong_band": STRONG_BAND,
            "cache_hit_fraction": CACHE_HIT_FRACTION,
            "dwell_default_s": DWELL_DEFAULT_S,
            "wake_l1_s": WAKE_L1_S,
            "sleep_l1_s": SLEEP_L1_S,
            "wake_l2_s": WAKE_L2_S,
            "sleep_l2_s": SLEEP_L2_S,
            "stop_s": STOP_S,
            "mmpp": {
                "hi_mult": MMPP_HI_MULT,
                "lo_mult": MMPP_LO_MULT,
                "hi_dwell_s": MMPP_HI_DWELL_S,
                "lo_dwell_s": MMPP_LO_DWELL_S,
            },
            "session_mean_len": SESSION_MEAN_LEN,
            "assumptions": ASSUMPTIONS,
        },
        "catalog_specs": {
            m: {
                "weight_bytes": s.weight_bytes,
                "hbm_bytes": s.hbm_bytes,
                "start_warm_s": s.start_warm_s,
                "start_cold_s": s.start_cold_s,
                "first_token_s": s.first_token_s,
                "mean_gpu_s": s.mean_gpu_s,
                "cap": s.cap,
            }
            for m, s in specs.items()
        },
        "sweep": sweep,
        "thrash": thrash,
        "adversarial": adversarial,
        "l2_appendix": l2_appendix,
        "gate": gate,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(
        f"gate: {gate['verdict']} "
        f"(win={gate['win_cells']} neutral={gate['neutral_cells']} "
        f"mid={gate['mid_cells']})"
    )


if __name__ == "__main__":
    main()
