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
  ETR actions or bounded continuous flight actions, in the CleanRL style,
  plus exact-resume checkpoints, a wall-clock budget and the yield protocol.
- [`continuous.py`](continuous.py): a bounded squashed-Gaussian actor-critic
  for flight-control environments.  It is separate from the discrete actor so
  existing ETR checkpoints retain their original layout.
- [`flight/`](flight/): hashed observation, goal and action-vector contracts,
  timestamped telemetry, expiring control messages and explicit coordinate
  transforms.
- [`envs/ufo_env.py`](envs/ufo_env.py): a Gymnasium adapter over the shared
  `xplane-ufo` C dynamics library.  The aircraft repository owns the physical
  model and concrete vector schema; knlp owns the learning environment.
- [`controls/ufo.py`](controls/ufo.py): zero-wrench, hover and velocity-target
  reference controllers for the standalone flight environment.
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
is a 77-value vector in course coordinates: progress fraction, lateral offset
from the centre, heading error, velocity components, terrain tilt, contact and
control flags, a 4 x 5 fan of forward terrain heights and frictions, and the
four nearest trees and items. The action is one of nine macro-actions
(steer left/straight/right x coast/paddle/brake) held for four physics ticks
at 60 Hz. The reward is the sum of named terms, each logged separately:
progress counts only new maximum distance down the course (so oscillating
across a point earns nothing), herrings, a per-step time cost (0.2 per 1/15 s step, so 3 per race second, the same size as the finish bonus over the 16 s that separate the no-op racer from the centreline controller), a
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

# CPU smoke for the L2 raw-wrench UFO policy
XPLANE_UFO_ROOT=~/devel/xplane-ufo-ai-runtime \
  python -m rl.ppo --env ufo:hover --device cpu --num-envs 8 \
  --num-steps 256 --num-minibatches 8 --update-epochs 5 \
  --learning-rate 3e-4 --gamma 0.995 --run-name ufo-hover-s1

# fixed-seed physical comparison against zero and reference control
XPLANE_UFO_ROOT=~/devel/xplane-ufo-ai-runtime \
  python -m rl.evaluate_ufo --run-dir runs/rl/ufo-hover-s1 \
  --random-start --output runs/rl/ufo-hover-s1/evaluation.json

# clone the deterministic controller, then start a fresh PPO run from it
XPLANE_UFO_ROOT=~/devel/xplane-ufo-ai-runtime \
  python -m rl.clone_ufo --run-dir runs/rl/ufo-hover-clone-s1
XPLANE_UFO_ROOT=~/devel/xplane-ufo-ai-runtime \
  python -m rl.ppo --env ufo:hover --init-agent \
  runs/rl/ufo-hover-clone-s1/checkpoint.pt --run-name ufo-hover-bc-ppo-s1

# clone control across changing combat-maneuver goals
XPLANE_UFO_ROOT=~/devel/xplane-ufo \
  python -m rl.clone_ufo --env ufo:maneuver \
  --run-dir runs/rl/ufo-maneuver-clone-s1

# read X-Plane telemetry and log policy proposals without sending controls
python -m rl.export_ufo_actor \
  --contract ~/devel/xplane-ufo/schemas/ufo-wrench-v1.json \
  --checkpoint /data/knlp-key-results/xplane-ufo-20260906/runs/ufo-hover-dagger-s1/checkpoint.pt \
  --output /data/knlp-key-results/xplane-ufo-20260906/runs/ufo-hover-dagger-s1/actor.npz
python -m rl.flight.shadow_ufo \
  --contract ~/devel/xplane-ufo/schemas/ufo-wrench-v1.json \
  --model /data/knlp-key-results/xplane-ufo-20260906/runs/ufo-hover-dagger-s1/actor.npz \
  --output runs/rl/xplane-shadow.jsonl

# same, under the supervisor that restarts after every yield
python -m rl.pace.ctl run --name bh-s1 -- python -m rl.ppo --env etr:bunny_hill --run-name bh-s1 --resume
```

Environment ids are `<backend>:<course>`; `sim:sim_trees` needs no game.
The synthetic `ufo:maneuver` task changes body velocity and yaw-rate targets
every four simulated seconds. Its descent limit uses the current altitude so
the sampled command does not deliberately point through the terrain safety
margin during that interval. This trains the continuous flight actor; target
selection, firing, and camera control remain separate mission-director
responsibilities.

`ufo:showcase` widens that task to the Hollywood demonstration envelope:
90 m/s forward, 50 m/s laterally, 24 m/s vertically, and 1.4 rad/s yaw.
Its reference controller permits 24 m/s² acceleration instead of the
certified profile's 8 m/s². Keep checkpoints trained for these two profiles
separate; passing the certified combat gate does not cover showcase control.

Before exporting a maneuver actor, compare it with the reference pilot on at
least 100 fixed seeds and gate the resulting report:

```
python -m rl.certify_ufo \
  --evaluation runs/rl/ufo-maneuver-s1/evaluation.json \
  --output runs/rl/ufo-maneuver-s1/certification.json

# probe every corner of the combat director goal envelope
python -m rl.stress_ufo \
  --run-dir runs/rl/ufo-maneuver-s1 \
  --output runs/rl/ufo-maneuver-s1/combat-stress.json
```

The gate rejects any terrain or contract-envelope termination, any episode
that tilts past 20 degrees, mean velocity error above 1.25 times the reference,
or mean yaw-rate error above 1.5 times the reference. Passing covers only the
standalone dynamics; it does not authorize or arm live X-Plane control.
The combat stress adds an adversarial gate over all 16 Cartesian corners of
the director velocity and yaw-rate limits. It applies the same terminal,
tilt, and relative tracking limits instead of relying only on average random
goal coverage.

The accepted maneuver actor is a DAgger behavioral clone, not a PPO result.
Use it as an initialization checkpoint when running PPO rather than describing
imitation training as reinforcement learning.

`python -m rl.ufo_duel` loads independent actor and enemy motor checkpoints
in the shared headless dynamics. A deterministic tactical layer supplies
pursuit and firing goals to both policies:

```
python -m rl.ufo_duel \
  --actor-run-dir runs/rl/ufo-showcase-candidate \
  --enemy-run-dir runs/rl/ufo-maneuver-accepted \
  --output runs/rl/ufo-showcase-candidate/duel.json
```

Omit `--enemy-run-dir` for a symmetric same-checkpoint duel. This is a gate
for composing two learned motor policies, not learned combat tactics or live
control of an X-Plane AI aircraft.

The F-14 actor has a separate live adapter for the user-selected stock
Laminar F-14D. Start X-Plane first, load an airborne F-14, and then run:

```
F14_RESULTS=/data/knlp-key-results/xplane-f14-20260912
python -m rl.flight.control_f14 \
  --contract rl/contracts/fighter-controls-v2.json \
  --model "$F14_RESULTS/runs/f14-live-dagger-s2-v1/actor.npz" \
  --transport web \
  --airspeed 180 --climb-rate 0 --turn-rate 0 \
  --output "$F14_RESULTS/live/player.jsonl"
```

The live adapter uses X-Plane's 10 Hz WebSocket stream and never installs the
UFO plugin into the fighter. REST polling is too slow for control and remains
diagnostic only. Before taking over, the adapter requires ten samples at no
less than 8 Hz, 140--230 m/s, 500 m AGL, at most 5 m/s climb or descent,
attitude within 10 degrees, and body rates below 0.15 rad/s. Its yaw-invariant
revision-2 actor then blends in over three seconds. Surface commands are
bounded to 35 percent and slew limited. Pause, stale input, an unsafe live
envelope, interruption, and normal exit all release the joystick and engine
overrides. This first live gate holds a fixed speed, climb rate, and turn rate;
mission choreography and learned formation tactics remain separate.

The live multiplayer adapter can run the same frozen motor actor once per
configured UFO. Its formation behavior places the player at the point of a V
and gives every follower a body-velocity and yaw-rate goal that closes its
error from a fixed slot behind the player. The player remains manual and the
followers do not fire. This isolates heading match, spacing, cohesion, and
recovery before combat adds target selection and weapon timing. A large
one-frame leader relocation, including BLINK, translates every formation
member by the same vector so the group keeps its slots instead of crossing
the intervening scenery at normal flight speed.

The stock F-14 can also lead configured AI F-14s without loading the UFO
aircraft plugin. `rl.flight.control_f14_formation` reads the player pose from
X-Plane's Web API, advances one frozen motor actor and fixed-wing dynamics
instance per follower, and writes only multiplayer aircraft paths. Its slot
velocity includes the full three-dimensional rigid-formation motion and a
bounded position correction. Followers therefore remain in their slots while
the leader rolls or completes a vertical loop. Once acquired, the adapter
accepts continuous aerobatic attitudes instead of releasing and later
respawning the formation. Pause, stale telemetry, landing, interruption,
and normal exit still release every acquired AI path; aircraft zero remains
manual.
Neutral and friend AI slots are accepted; enemy slots are rejected before
path ownership begins. X-Plane exposes the team-status array as read-only, so
the adapter cannot change an enemy slot into a formation follower.
The `xplane-ufo` helper `tools/ai_f14_formation.sh` supervises this adapter in
a named tmux session.

`rl.evaluate_f14_formation` is the closed-loop formation gate. It flies the
learned followers through sustained left and right turns plus complete inside
and outside vertical loops. It records slot error, minimum pair separation,
linear acceleration, and angular speed. The ordinary motor-policy gate is
necessary but not sufficient: it can pass while small tracking errors
accumulate into a visibly broken formation.

Combat exposes a one-use shield request beside each enemy pose. The current
director raises it deterministically on the first close attack. That field is
an interface for a future tactical policy, not evidence that shield timing is
learned. The motor actor still receives only velocity and yaw-rate goals and
never chooses targets, fire, or shields.

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
