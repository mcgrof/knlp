# Phase 1 150M driver map

## Canonical result root
- `/home/mcgrof/devel/results-archive/paper-fim/150m/`

## Lane assignments

### Lane P1-A (A100 80GB)
Runs:
- pruning family
- kvsplice family

Results:
- `.../pruning/`
- `.../kvsplice/`

Teardown condition:
- all assigned run IDs complete
- summaries/logs committed to `results-archive`
- checksum or manifest updated

### Lane P1-B (A100 80GB)
Runs:
- ra family
- composition family

Results:
- `.../ra/`
- `.../composition/`

Teardown condition:
- all assigned run IDs complete
- summaries/logs committed to `results-archive`
- checksum or manifest updated

### Lane P1-C (H100 80GB, optional but recommended)
Runs:
- perf family
- latency/wall-clock validation on frozen checkpoints

Results:
- `.../perf/`

Teardown condition:
- perf report generated
- latency tables written
- committed to `results-archive`

## Naming pattern
- `{date}_{regime}_{family}_{variant}_s{seed}`
- example: `2026-03-19_150m_pruning_bitter7_s50_s2`
