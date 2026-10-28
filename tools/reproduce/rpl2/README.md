# KV-offload read path on a mounted filesystem

These tools give a basic, reproducible baseline for one question: how long
does it take to read a whole KV-cache-sized object back from a filesystem,
and does every request return exactly the bytes that were written? The
page [knlp.io/lmcache-readplan-l2-reproduce.html](https://knlp.io/lmcache-readplan-l2-reproduce.html)
explains the setup, output and troubleshooting in more detail.

Two defconfigs select the profile:

```
make defconfig-rpl2-smoke && make
make defconfig-rpl2-mounted-fs && \
    make RPL2_TARGET=/mnt/share/rpl2 RPL2_OUTPUT=$HOME/rpl2-records
```

The smoke profile runs the unit tests in [tests/](tests/) on temporary
local directories. The mounted-filesystem profile writes a small set of
immutable objects to `RPL2_TARGET`, reads each one back as a request
(open, read, close), checks it against its SHA-256 digest after the timed
region and appends one JSON record per request under `RPL2_OUTPUT`. Both
directories must exist outside the source tree. `make rpl2-help` lists
the individual targets.

Nothing here mounts a filesystem, needs root, drops caches or changes a
kernel or service setting. Cache state is recorded as uncontrolled. The
record schema is defined in [records.py](records.py).
