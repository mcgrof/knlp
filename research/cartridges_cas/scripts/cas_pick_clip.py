#!/usr/bin/env python3
"""Decide the clip threshold for the LR-0.1 arm from the gradient-norm probe.

The probe runs the frozen regime unchanged for 210 optimizer steps with the
pre-clip gradient norm logged every step.  This script reads that log and
decides two things at once: whether a clipped arm is worth nine hours of GPU
time at all, and if so at what threshold.

The gate exists because a clip is only an ablation when it fires on the event
it was built for and on nothing else.  The event in question is a loss spike.
Archived full-parquet runs put their spikes at very different steps (patient_06
at 288-297, patient_01 at 425-434 and 976-986, patients 03 and 05 never), and
no run in the archive ever logged a gradient norm, so the magnitude a clip
would have to catch has never been measured.  A threshold picked from ordinary
step statistics is therefore a guess: on this probe the 95th percentile of the
post-warmup norms lands under 18 of the 20 init-transient steps and would
rescale the largest of them by more than tenfold, which perturbs warmup rather
than clipping a spike.

So the rule is: only emit a firing threshold when the probe actually caught a
loss event whose gradient norm stands clear of every other step by a factor of
three.  Otherwise emit a guard far above anything observed.  A guard that never
fires leaves the arm numerically identical to an unclipped run, which is the
per-step loss and gradient-norm curve of the reference regime that the archive
is missing -- a better use of the GPU than a clip that cannot fire.

Output contract, imposed by the calling lane, which does
``pick_clip.py LOG | grep -oE '^CLIP_NORM=[0-9.e+-]+' | cut -d= -f2``:
exactly one line starts with ``CLIP_NORM=``, every other line is prefixed
``REPORT``, the exit status is always 0, and a threshold is printed on every
path including a parse failure -- an empty result parks a $3/h GPU.
"""

import re
import sys

# Both markers are flushed into the middle of the library's tqdm bar, so they
# are almost never at column 0 and must not be anchored.  The field patterns
# cannot cross a comma or whitespace and the record must end at the line end,
# so a line caught half-written is skipped rather than mis-parsed.
GRAD_RE = re.compile(r"GRADNORM_CSV,(\d+),([^,\s]+),([01])(?![^\r\n])")
LOSS_RE = re.compile(r"LOSS_CSV,(\d+),([^,\s]+),([^,\s]+),(\d+)(?![^\r\n])")

INIT_END = 20  # steps 0..19 are the init transient, excluded from STEADY
EVENT_RATIO = 1.5  # loss over its trailing median that counts as an event
EVENT_MERGE = 6  # hits this close together are one event
EVENT_PRE, EVENT_POST = 2, 8  # event window, in steps around the hits
SEP_REQUIRED = 3.0  # event norm over every other norm, to gate the arm
GUARD_MULT = 4.0  # NO-GO threshold, as a multiple of the largest norm seen
FIRE_MULT = 2.0  # GO threshold, as a multiple of the largest non-event norm
FALLBACK = 1e6  # never fires, never overflows clip_coef


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs, q):
    s = sorted(xs)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _finite(x):
    return x == x and x not in (float("inf"), float("-inf"))


def _parse(path):
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8", errors="replace")
    grads, clipped_rows, dropped = {}, 0, 0
    for m in GRAD_RE.finditer(text):
        try:
            v = float(m.group(2))
        except ValueError:
            dropped += 1
            continue
        if not _finite(v):
            dropped += 1
            continue
        grads[int(m.group(1))] = v
        clipped_rows += int(m.group(3))
    losses = {}
    for m in LOSS_RE.finditer(text):
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        if _finite(v):
            losses[int(m.group(1))] = v
    return grads, losses, clipped_rows, dropped


def _events(losses):
    """Steps where the loss rose clear of its own trailing median."""
    hits = []
    for k in sorted(losses):
        if k < 2 * INIT_END:
            continue
        prev = [losses[j] for j in range(k - 20, k) if j in losses]
        if len(prev) < 10:
            continue
        base = _median(prev)
        if base > 0 and losses[k] / base >= EVENT_RATIO:
            hits.append((k, losses[k] / base))
    events = []
    for step, ratio in hits:
        if events and step - events[-1][1] <= EVENT_MERGE:
            events[-1] = (events[-1][0], step, max(events[-1][2], ratio))
        else:
            events.append((step, step, ratio))
    return events


def main():
    out = []

    def report(line):
        out.append("REPORT " + line)

    threshold, verdict = FALLBACK, "FALLBACK"
    try:
        grads, losses, clipped_rows, dropped = _parse(sys.argv[1])
        steps = sorted(grads)
        report(
            "parsed %d gradient rows (steps %s..%s), %d loss rows, "
            "%d already-clipped, %d dropped"
            % (
                len(steps),
                steps[0] if steps else "-",
                steps[-1] if steps else "-",
                len(losses),
                clipped_rows,
                dropped,
            )
        )
        if len(steps) < 40:
            report("FEWER THAN 40 GRADIENT ROWS -- probe did not run; guard only")
            raise ValueError("too few rows")
        missing = [s for s in range(steps[0], steps[-1] + 1) if s not in grads]
        if missing:
            report(
                "missing %d step(s) in range, first few: %s"
                % (len(missing), missing[:8])
            )
        if clipped_rows:
            report(
                "WARNING %d rows report clipped=1; the probe should be unclipped"
                % clipped_rows
            )

        init = [grads[s] for s in steps if s < INIT_END]
        steady = [grads[s] for s in steps if s >= INIT_END]
        gmax = max(grads.values())
        smed = _median(steady) if steady else float("nan")
        report(
            "steady window (step %d+): n=%d min=%.6g median=%.6g p90=%.6g "
            "p95=%.6g p99=%.6g max=%.6g max/median=%.3g"
            % (
                INIT_END,
                len(steady),
                min(steady),
                smed,
                _pct(steady, 0.90),
                _pct(steady, 0.95),
                _pct(steady, 0.99),
                max(steady),
                max(steady) / smed if smed else float("nan"),
            )
        )
        if init:
            report(
                "init transient (steps 0-%d): n=%d max=%.6g = %.3gx the steady median"
                % (
                    INIT_END - 1,
                    len(init),
                    max(init),
                    max(init) / smed if smed else float("nan"),
                )
            )
        report(
            "the old p95 rule would have emitted %.6g, firing on %d of %d steps"
            % (
                _pct(steady, 0.95),
                sum(1 for v in grads.values() if v > _pct(steady, 0.95)),
                len(grads),
            )
        )

        events = _events(losses)
        if not events:
            report(
                "loss events at ratio >= %.2g: NONE in %d loss rows"
                % (EVENT_RATIO, len(losses))
            )
        for a, b, r in events:
            report("loss event steps %d-%d, peak %.3gx its trailing median" % (a, b, r))

        in_ev = set()
        for a, b, _ in events:
            in_ev.update(range(a - EVENT_PRE, b + EVENT_POST + 1))
        ev_norms = [grads[s] for s in steps if s in in_ev]
        out_norms = [grads[s] for s in steps if s not in in_ev]
        if events and ev_norms and out_norms:
            g_ev, g_out = max(ev_norms), max(out_norms)
            sep = g_ev / g_out if g_out else float("inf")
            report(
                "event max norm %.6g, non-event max norm %.6g, separation %.3gx"
                % (g_ev, g_out, sep)
            )
            if sep >= SEP_REQUIRED:
                threshold = FIRE_MULT * g_out
                verdict = "GO"
                report(
                    "GATE PASS: separation %.3gx >= %.2gx, so a single scalar can fire on "
                    "the event and on nothing else the probe saw" % (sep, SEP_REQUIRED)
                )
            else:
                threshold = GUARD_MULT * gmax
                verdict = "NO-GO separation %.3gx < %.2gx" % (sep, SEP_REQUIRED)
                report(
                    "GATE FAIL: the loss event carries no distinguishable gradient signature, "
                    "so no threshold fires on it without also rescaling ordinary steps"
                )
        else:
            threshold = GUARD_MULT * gmax
            verdict = "NO-GO no loss event in the probe window"
            report(
                "GATE FAIL: the probe caught no loss event, so the magnitude a clip would "
                "have to catch remains unmeasured"
            )

        if verdict.startswith("NO-GO"):
            report(
                "emitting a guard at %.3gx the largest norm seen (%.6g); it fires on nothing "
                "the probe observed, so the arm is numerically an unclipped replicate of the "
                "reference regime and supplies the per-step curve the archive lacks"
                % (GUARD_MULT, gmax)
            )
    except Exception as exc:  # never leave the lane without a threshold
        report("EXCEPTION %s: %r" % (type(exc).__name__, exc))
        report("emitting the inert fallback; the arm runs as an unclipped replicate")
        threshold, verdict = FALLBACK, "FALLBACK"

    report("verdict: %s" % verdict)
    print("\n".join(out))
    print("CLIP_NORM=%.6g" % threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
