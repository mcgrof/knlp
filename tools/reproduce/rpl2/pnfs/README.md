# pNFS lab for the KV-offload read path

These scripts build a pNFS system on one Linux host and compare stock
LMCache with LMCache's native file connector keeping read descriptors open
(`held_fd_capacity`). The usual entry points are the knlp defconfigs
`rpl2-pnfs-lab`, `rpl2-pnfs-sweep` and `rpl2-pnfs-remote`. The
[reproduction page](../../../../docs/lmcache-readplan-l2-reproduce.html#pnfs-lab)
explains them; this file maps the scripts.

## Pieces

| Script | Role |
|---|---|
| [setup.sh](setup.sh) | Fetch and build Lattice (pNFS metadata server), RonDB, ntirpc and LMCache at checked pins into `RPL2P_ROOT`, without root. Creates a uv venv with CPU torch when `RPL2P_PYTHON` is unset. |
| [env.sh](env.sh) | Lab parameters: `RPL2P_ROOT` (required), `RPL2P_PREFIX`, `RPL2P_NET`, memory caps, addresses. |
| [labctl.sh](labctl.sh) | Root. Brings namespaces, the NFSv3 data servers, RonDB, the metadata server and the NFSv4.2 client up and down as capped transient units. Also runs commands inside them and handles extra clients and the remote route. Teardown checks the host's NFS and firewall state against its baseline. |
| [ds-inner.sh](ds-inner.sh), [rondb-inner.sh](rondb-inner.sh), [mds-inner.sh](mds-inner.sh), [cli-inner.sh](cli-inner.sh) | What each unit runs inside its namespaces. |
| [smoke.sh](smoke.sh), [check_receipt.py](check_receipt.py) | One-minute functional test: one object written and read back directly from the data server, with no fallback and a clean teardown. Writes `receipt.json` with PASS or FAIL. |
| [fixture.sh](fixture.sh) | Seals the shared seeded test objects (8 MiB and 256 KiB). |
| [native_reader.py](native_reader.py) | The reader: drives LMCache's native connector directly (no torch import) to seal objects, run timed requests or a phased fingerprint. `--held` sets `held_fd_capacity`. |
| [run.sh](run.sh) | Runs the reader as the owner inside a lab client, an extra client or the remote client. |
| [fingerprint.sh](fingerprint.sh), [trace.sh](trace.sh), [trace_summary.py](trace_summary.py) | RPCs per phase, server and procedure, from a private tracefs instance. |
| [screen.sh](screen.sh) | Paired screen, stock against held, in alternating blocks. |
| [sweep.sh](sweep.sh) | Many clients, each its own NFS client identity, reading the same objects at once. |
| [remote-client.sh](remote-client.sh), [remote-holder.sh](remote-holder.sh) | A client on another host, mounted in a private mount namespace. |
| [summarize.py](summarize.py) | Screen, sweep and fingerprint summaries. |
| [tests/](tests/) | Unit tests on synthetic records (`make rpl2-pnfs-test`). |

## Safety

Everything the lab creates carries its prefix:
- network namespaces `<prefix>-*`;
- systemd units `<prefix>-*`;
- veth pairs `<prefix>a-*` and `<prefix>b-*`;
- the tracefs instance `<prefix>`.

The data servers bind-mount their own `/run`, `/var/lib/nfs`,
`/etc/exports` and nfsd filesystem inside a private mount namespace. The
host's NFS server, exports and rpcbind are never edited or restarted.

Remote clients need two `DOCKER-USER` accept rules for the lab subnet
while the lab is up. Docker sets the forwarding policy to drop, and these
rules reopen forwarding for that subnet only. Nothing persists across a
reboot. Run `labctl.sh down` (or `make rpl2-pnfs-down`) after an
interrupted run; it removes the lab and reports any leftover.

## Licences

Lattice is MIT-licensed except for one GPL-2.0-or-later file, its RonDB
shim, which makes a RonDB-enabled `pnfs-mds` binary a combined
GPL-2.0-or-later binary. RonDB, ntirpc and LMCache keep their own
licences. The scripts fetch and build these projects locally and do not
redistribute them.
