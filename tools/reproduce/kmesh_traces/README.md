# Agent-trace fan-out and resume counts

These scripts count two things in released agent traces. The first is how
often one agent launch starts four or more children that share a large
saved ancestor. That is the case where reading shared KV from storage once
for all children (singleflight) or sharing attention reads (cascade
attention) has work to do. The second is how much long-context work
resumes after a long idle gap, which is the case for storing a context at
the end of a turn and restoring it at the next one. The findings are on
the [KMesh agent workloads page](https://knlp.io/kmesh-workloads.html).

The scripts only read published data. They run on a CPU with Python 3.10
or later and numpy; the FreeInference reader also needs pandas and
pyarrow. No model, GPU or storage path is used.

## Inputs

| Trace | Revision | License |
| --- | --- | --- |
| [Weka Claude Code corpus](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126) | `23f152f6f0f9399a85901b89a6458def0ef16729` | Apache-2.0 |
| [FreeInference agentic trace](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace), weeks 2026-05-17 and 2026-07-12 | `16d3c04dae6ccaa0c48a9064e204e3a8f1eb9ade` | CC BY 4.0 |
| [FreeInference analysis artifact](https://github.com/HarvardMadSys/freeinference_agentic_trace), its `release` reader | `925a718eed7cc01255e7dcea024e70389cf021e5` | see repository |
| [GitHub Copilot coding agent traces](https://github.com/Azure/AzurePublicDataset/blob/215becdacba1ce682c7368642ce97d5a332de7a6/GitHubCopilotCodingAgentDataset2026.md), days 2026-06-02 and 2026-06-06 | release `ghcp-coding-agent-2026` | CC BY |

The Copilot archives must match the release digests:

```
95b0450041095dbe88181b88841c64b1e42f55a0127aa7abe91c1c9d3b424240  date.2026-06-02.tar.gz
9546e3fd156d379e2420b8a0ada496ff507cf74aab32b9fc97bd1b6b353fa182  date.2026-06-06.tar.gz
```

```sh
hf download semianalysisai/cc-traces-weka-062126 --repo-type dataset \
  --revision 23f152f6f0f9399a85901b89a6458def0ef16729 --local-dir /path/to/weka

hf download harvardMadsys/freeinference_agentic_trace --repo-type dataset \
  --revision 16d3c04dae6ccaa0c48a9064e204e3a8f1eb9ade --local-dir /path/to/fi \
  --include 'release/2026-05-17_2026-05-23/*' 'release/2026-07-12_2026-07-18/*'
git clone https://github.com/HarvardMadSys/freeinference_agentic_trace /path/to/fi-artifact
git -C /path/to/fi-artifact checkout 925a718eed7cc01255e7dcea024e70389cf021e5

cd /path/to/copilot
for d in 2026-06-02 2026-06-06; do
  curl -LO https://github.com/Azure/AzurePublicDataset/releases/download/ghcp-coding-agent-2026/date.$d.tar.gz
done
sha256sum -c digests.txt   # the two lines above
for d in 2026-06-02 2026-06-06; do tar -xzf date.$d.tar.gz; done
```

The FreeInference reader expects each week directory to hold
`manifest.json` beside the files of its `processed/` directory. A layout
of symbolic links is enough:

```sh
for w in 2026-05-17_2026-05-23 2026-07-12_2026-07-18; do
  mkdir -p /path/to/fi-layout/$w
  ln -s /path/to/fi/release/$w/manifest.json /path/to/fi-layout/$w/
  ln -s /path/to/fi/release/$w/processed/* /path/to/fi-layout/$w/
done
```

## Run

```sh
python3 fanout_spans.py --weka /path/to/weka/traces.jsonl --window-s 1 --output weka-w1.json
python3 fanout_spans.py --weka /path/to/weka/traces.jsonl --window-s 5 --output weka-w5.json
for w in 2026-05-17 2026-07-12; do
  PYTHONPATH=/path/to/fi-artifact python3 fanout_spans.py \
    --release /path/to/fi-layout --week $w --output fi-$w.json
done
python3 weka_resume.py --traces /path/to/weka/traces.jsonl --output weka-resume.json
for d in 2026-06-02 2026-06-06; do
  python3 copilot_resume.py --data /path/to/copilot --days $d --output copilot-$d.json
done
```

Each script prints a short summary and writes one JSON report. The fan-out
report holds one row per launch.

## Definitions

A launch is one parent step that starts one or more children. Weka groups
children whose spawn times fall within `--window-s` seconds; FreeInference
groups by the parent's spawning step. Width is the number of children.
Spans are equal leading block ids in each child's first request: 64-token
hash blocks local to a trace in Weka, 16-token block ids local to a week
in FreeInference. Equal ids are a workload intention, not verified KV
identity on any model.

- Group span L: the leading blocks common to every child of the launch.
- Preamble span X: the longest leading run the first child shares with a
  child of a different launch (same trace in Weka; same week and harness
  in FreeInference).
- Prior span E: the longest leading run the first child shares with any
  request that started before the launch.
- Saved family ancestor S = max(0, min(L, E) - X): shared by these
  children only and present before the launch, so it could be in storage.
- New family span N = L - max(E, X): shared by these children only but
  first seen at the launch.
- Saved shared span S2 = min(L, E), with the time since the last earlier
  request that held all of it ended. This is the span a wide launch would
  read back once per child if it had left GPU memory.

In the Copilot release, an LLM call's `timestamp` marks the end of the
call: consecutive calls fit that reading in 99.94% of 81,519 checked
pairs and the start reading in 75%. The script therefore places a call at
`timestamp - duration_ms` to `timestamp`, uses LLM calls only for turn
times, and counts two calls as overlapping only beyond the release's 1 s
timestamp resolution. A compaction is the release's own heuristic: the
prompt shrinks by more than 30% after a prompt above 10,000 tokens.
