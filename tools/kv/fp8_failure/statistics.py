"""GPU-free statistics for the failure atlas: paired bootstrap CIs, Benjamini-Hochberg FDR, and a
leave-one-factor-out (LOFO) attribution. The atlas makes many claims ("FP8-K hurts model X",
"prebias recovers Y%", "subspace Z is the culprit") across many cells; without multiplicity control
and CIs those claims are noise-mining. Deterministic given a seed -- no Date/Random reliance here;
the caller passes the seed.
"""

import math


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0, statistic="mean"):
    """Percentile bootstrap CI for the mean (or median) of a 1-D sample. Pure-Python LCG so it is
    reproducible without numpy and without the banned Random/Date globals. Returns
    (point, lo, hi)."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    def stat(xs):
        if statistic == "median":
            s = sorted(xs)
            m = len(s)
            return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])
        return sum(xs) / len(xs)

    point = stat(vals)
    state = (seed * 2654435761 + 12345) & 0xFFFFFFFF
    boots = []
    for _ in range(n_boot):
        sample = []
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            sample.append(vals[state % n])
        boots.append(stat(sample))
    boots.sort()
    lo = _quantile(boots, alpha / 2)
    hi = _quantile(boots, 1 - alpha / 2)
    return point, lo, hi


def paired_bootstrap_delta(a, b, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap CI for the paired difference mean(a - b) (e.g. error_fp8 - error_repair across the
    same eval items). a and b must be aligned same-length samples. Returns (delta, lo, hi).
    """
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_delta needs aligned same-length samples")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    return bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed, statistic="mean")


def benjamini_hochberg(pvalues, q=0.05):
    """BH step-up FDR control. Returns a list of booleans (reject H0 / 'significant') aligned to the
    input order, plus the BH-adjusted q-values. Controls the false-discovery rate across the atlas's
    many cells so we do not call sampling noise a failure."""
    m = len(pvalues)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = pvalues[i] * m / (rank + 1)
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    reject = [adj[i] <= q for i in range(m)]
    return reject, adj


def lofo_attribution(full_error, ablated_errors):
    """Leave-one-factor-out attribution. full_error: scalar error with every factor present.
    ablated_errors: {factor_name: error_with_that_factor_removed}. A factor's importance is how much
    the error DROPS when it is removed (positive => that factor was hurting). Returns a dict sorted
    high-to-low by attributed importance."""
    imp = {f: full_error - e for f, e in ablated_errors.items()}
    return dict(sorted(imp.items(), key=lambda kv: kv[1], reverse=True))
