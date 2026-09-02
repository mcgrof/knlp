# Extreme Tux Racer headless bridge protocol

The patch in [patches/](patches/) adds an `--rl` mode to Extreme Tux Racer
0.8.4 that runs the race physics with no window, GL context or audio device,
so a learner can reset and step the game thousands of times faster than real
time. This page is the wire contract between that process and
[`rl/envs/etr_bridge.py`](../envs/etr_bridge.py).

## Transport

`etr --rl` reads one request per line on stdin and writes one JSON object per
line on stdout. Everything the game would normally print (its banner, load
messages) goes to stderr, so stdout carries only replies. The process exits on
`quit` or end of input. One process is one racer; run several processes for
several environments.

## Requests

```
info
reset seed=<u32> course=<dir> [group=default] [wind=0] [light=0] [mirror=0] [dt=0.016667]
step turn=<-1..1> brake=<0|1> paddle=<0|1> jump=<0|1> [ticks=1] [dt=0.016667]
quit
```

`reset` loads the course (courses are listed by `info`; `bunny_hill` is the
shortest default course), seeds the C library random generator, replays the
start animation at the fixed step and leaves the racer at the starting line
with every collectable item restored. `wind` 1 to 3 enables the game's wind
model, which draws from the seeded generator. Snow is never enabled because it
is a particle effect with no physical influence.

`step` applies one action for `ticks` physics ticks of `dt` seconds each and
returns the state after the last tick. A tick runs the exact steering and
physics code the rendered game runs per frame; `turn` feeds the joystick path
(so `-1`, `0`, `1` reproduce the keyboard), `paddle` and `brake` are the
matching keys, and `jump` charges a jump while held and releases it when
dropped. The loop stops early when the finish line is crossed.

## Replies

Every reply carries `"ok": true`; an error reply is
`{"ok": false, "error": "<message>"}` and leaves the state untouched.

`info` returns the game version, the character in use, the data directory and
the course table (`group`, `dir`, `name`, sizes, start point, `use_keyframe`).

`reset` and `step` return the observation:

| key | meaning |
|---|---|
| `tick`, `time` | physics ticks and race clock since reset |
| `pos`, `vel` | racer position and velocity in course coordinates (the course runs toward negative z) |
| `speed`, `way` | speed magnitude and path length travelled |
| `progress` | distance along the course, i.e. `-pos.z` |
| `herring`, `items_left` | items collected and still on the course |
| `collisions` | tree contact events since reset (one per contact, not per tick) |
| `airborne`, `braking`, `paddling`, `jumping`, `turn` | control and contact state |
| `friction`, `nml` | surface friction and terrain normal under the racer |
| `finish`, `done` | finish-line state; `done` is what a learner should treat as terminal |
| `size`, `play` | course and playable extents (width, length) |
| `probe_h`, `probe_f` | terrain height relative to the racer, and friction, on a 4 x 5 forward fan (3, 6, 12, 24 m ahead; -6 to +6 m sideways) |
| `trees` | up to 4 nearest trees as `[dx, dz, diameter]` |
| `items` | up to 4 nearest collectable items as `[dx, dz]` |

Numbers are written with 12 significant digits; two runs with the same seed
and request stream produce byte-identical replies, which is the determinism
check the environment tests rely on.

## Performance

Measured on prune (Ryzen, 32 threads, single process): about 170k physics
ticks per second on `bunny_hill`, including JSON encoding at one reply every
four ticks. Real time is 60 ticks per second, so one process is roughly
2800x real time; the game process uses no GPU at all.
