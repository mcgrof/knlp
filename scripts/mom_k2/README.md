# Normalized top-2 memory study

Code for the four-arm comparison of a gated-delta baseline against a
mixture-of-memories layer at top-1 and top-2 routing, plus a control that keeps
top-2 selection but fixes the readout coefficients.

## Inputs and outputs

The runners use `scripts/matched_micro_train.py` and
`scripts/matched_micro_data.py` from this checkout. To use a separate
copy of those modules, set `MATCHED_MICRO_SOURCE` to its directory.

Pass `--data-dir /path/to/packed-tokens` to `qualify2.py`,
`timing.py`, `profile_arms.py`, and `train.py`, or set
`MATCHED_MICRO_DATA`. Prepare that directory with
`matched_micro_data.prepare`; it must contain the packed training
tokens, `dev_selection.npy`, and `meta.json`.

`train.py` also requires `--checkpoint-dir /path/to/checkpoints`
or `MOM_K2_CHECKPOINT_DIR`. Direct all `--out` files and checkpoints
to an external experiment directory. These commands generate result
artifacts; those artifacts do not belong in this source tree.

## Do not reformat the pinned or generated files

`_pinned_mom_c2.py`, `_pinned_mom_r1.py` and the four `mom_*.py` files that
`make_modules.py` writes are **evidence, not source to be tidied**. Their
SHA256 values are recorded in every qualification receipt and inside every
training checkpoint, and the two pinned files are identified in the study
contract by their git blob hashes:

```
_pinned_mom_c2.py  d611bad0ddd624814e890468173e1d6a2a314b31
_pinned_mom_r1.py  eb9663c2ec61db28089e095ee7f636ce77f97e0f
```

Running a formatter over them changes those bytes and silently breaks the link
between the archived result and the code that produced it. This happened once
during development and was caught by comparing the recorded hashes against the
files; the fix was to restore the executed bytes from the machine that ran them.
If a repository-wide format is ever applied, exclude this directory and
re-verify the hashes above.

The hand-written files here (`arms.py`, `qualify.py`, `qualify2.py`,
`timing.py`, `profile_arms.py`, `train.py`) also have their hashes recorded at
process startup, so reformatting them after a run makes the receipt disagree
with the tree. Format them only alongside a fresh run.

## What each file does

`make_modules.py` writes the two readout policies from one pinned source, so the
candidate and its control differ in exactly one documented place. `arms.py`
defines the four arms and interrogates each built model for the routing width
and readout policy it actually implements, rather than trusting its label.
`qualify.py` compares the optimized top-2 path against a less-optimized
reference of the same policy across four deliberate routing layouts in both
precisions, with tolerances calibrated from identical reruns and from a
mathematically equivalent reordering before any candidate error is read.
`qualify2.py` covers causality through the packed layout, where the router's
gradient comes from under each policy, and a full-model update with a disk save
and reload. `timing.py` measures update cost without a profiler attached and
prices the whole cohort. `profile_arms.py` attributes device time by category
over every event. `train.py` runs the cohort.
