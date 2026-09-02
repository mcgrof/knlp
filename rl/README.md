# Reinforcement learning in knlp

This directory holds knlp's reinforcement-learning scaffolding: a readable
single-file PPO trainer, a Gymnasium environment contract, non-learning
reference policies, and a pause/resume protocol so unattended training shares
one GPU with a person. The first task is racing in Extreme Tux Racer (ETR), a
GPL game whose physics runs cleanly without rendering. The pieces are split so
that the trainer, the vector wrapper and the GPU lease carry over unchanged to
the next environment; only `rl/envs/` is ETR-specific.

The lane exists for two reasons. It is an educational vehicle: every term of
the observation, the action table, the reward and the PPO update is in plain
code in this tree, small enough to read in one sitting. And it is scaffolding:
once one course is learned reproducibly, the same trainer is the starting
point for later RL work in knlp.

## Contents

- [`ppo.py`](ppo.py): PPO with generalised advantage estimation for discrete
  actions, in the CleanRL style, plus exact-resume checkpoints, a wall-clock
  budget and the yield protocol.
- [`vec.py`](vec.py): a minimal synchronous vector environment with
  same-step reset and per-episode statistics.
- [`envs/`](envs/): the ETR bridge client, a pure-Python simulator that speaks
  the same protocol, and the frozen environment contract (`etr_env.py`).
- [`controls/policies.py`](controls/policies.py): no-op, random and a
  centreline proportional controller, with a runner that scores them.
- [`pace/`](pace/): the GPU lease (`lease.py`) and its command line (`ctl.py`).
- [`etr/`](etr/): the game patch, its build script and the
  [wire protocol](etr/protocol.md).

## Building the game bridge

The bridge is a patch against Extreme Tux Racer 0.8.4 (the Debian source
package). It adds `etr --rl`, a mode that never opens a window, GL context or
audio device and drives the unchanged steering and physics code over
stdin/stdout. Build it with:

```
rl/etr/build_etr.sh            # fetches the source if needed, patches, builds
export ETR_RL_BIN=~/devel/extremetuxracer-0.8.4/src/etr
```

Build dependencies on Debian are `build-essential autoconf automake pkgconf
libsfml-dev libglu1-mesa-dev`; the course data comes from the distribution's
`extremetuxracer-data` package. One bridge process steps physics at roughly
170k ticks per second (about 2800x real time) and uses no GPU.

## The v0 contract

`rl/envs/etr_env.py` fixes the observation, action and reward. The observation
is a 73-value vector in course coordinates: progress fraction, lateral offset
from the centre, heading error, velocity components, terrain tilt, contact and
control flags, a 4 x 5 fan of forward terrain heights and frictions, and the
four nearest trees and items. The action is one of nine macro-actions
(steer left/straight/right x coast/paddle/brake) held for four physics ticks
at 60 Hz. The reward is the sum of named terms, each logged separately:
progress counts only new maximum distance down the course (so oscillating
across a point earns nothing), herrings, a small per-step time cost, a
collision penalty per tree contact, a finish bonus, and a penalty when the
episode is truncated for making no progress. Changing any of these is a
visible diff to that file, not a flag.

The `sim` backend (`envs/etr_sim.py`) is a crude point-mass on a slope that
returns the same raw dictionary as the game. It is a contract test double, not
a model of ETR; it lets the trainer and the tests run anywhere in seconds.

## Running

```
# score the reference policies
python -m rl.controls.policies --env etr:bunny_hill --episodes 5

# train on one course; the run directory holds args, metrics.csv, checkpoint.pt
python -m rl.ppo --env etr:bunny_hill --num-envs 8 --run-name bh-s1 --seed 1

# continue a stopped run from its checkpoint
python -m rl.ppo --run-name bh-s1 --resume

# same, under the supervisor that restarts after every yield
python -m rl.pace.ctl run --name bh-s1 -- python -m rl.ppo --env etr:bunny_hill --run-name bh-s1 --resume
```

Environment ids are `<backend>:<course>`; `sim:sim_trees` needs no game.
With a tiny policy network the learner is often faster on CPU than on a GPU;
`--device` selects, and the run's `sps` column in `metrics.csv` is the number
to compare.

## Sharing the GPU: pause and resume

Training runs on a workstation whose GPU a person also uses interactively.
Instead of a schedule, the runs honour a lease:

```
python -m rl.pace.ctl status                  # hold state and live workers
python -m rl.pace.ctl pause --wait            # reserve the GPU; workers checkpoint and exit
python -m rl.pace.ctl resume                  # release it; supervisors relaunch
```

`pause` writes a hold record and signals every registered worker. A trainer
checks the hold at each update boundary (and on SIGUSR1/SIGTERM), saves its
checkpoint and exits with status 75; a supervisor started with `ctl run` waits
for the hold to clear and restarts the worker with `--resume`. Nothing is
lost, no run has to be re-planned, and the state is plain files under
`~/.local/state/knlp/gpu-lease` (override with `KNLP_GPU_LEASE_DIR`), so any
shell or agent on the host can drive it. `--max-hours` bounds a single
session the same way; the run resumes on the next launch.

## Tests

```
python -m pytest tests/test_rl_pace.py tests/test_rl_env.py tests/test_rl_ppo_smoke.py
ETR_RL_BIN=... python -m pytest tests/test_rl_etr_bridge.py     # against the real game
```

The bridge tests are the first integrity gate: byte-identical trajectories
for the same seed across processes and across in-process resets, items and
counters restored on reset, a step rate far above real time, the Gymnasium
checker over the game itself, and the centreline controller finishing
`bunny_hill`. The PPO smoke test trains on the simulator on CPU, yields to a
hold, resumes from the checkpoint and finishes.

## Phases

The work is staged so GPU commitment grows only with evidence:

1. Scaffolding and environment integrity (this tree): bridge, contract,
   controls, trainer, tests. CPU only.
2. One course: PPO on `bunny_hill`, three development seeds then five
   reporting seeds, learning curves in steps and wall clock, checkpoint
   replays. Runs are bounded by `--max-hours` and pausable at any time.
3. Generalisation: a held-out course, perturbed starts, course-relative versus
   raw observations. This is where the first honest finding lives.
4. Coexistence: the same frozen build measured with and without an
   interactive session on the GPU, reported separately from the learning
   result.

Later hypotheses (pixels, world models, curriculum over game parameters) are
separate studies that start only after these gates pass.
