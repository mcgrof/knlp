# vLLM through the LMCache MP server

These scripts test LMCache's `held_fd_capacity` end to end: vLLM serves
a model through the LMCache multiprocess (MP) server, and the server's
`fs_native` L2 adapter keeps the KV cache on the filesystem under test.
Each of `PROMPTS` prompts has a distinct `PREFIX_TOKENS`-token prefix.
LMCache's host-memory L1 is smaller than the working set and the prompts
go round robin, so every request loads its prefix from the filesystem.
Each request asks for one token, so its latency is the time to first
token. The knlp defconfigs `rpl2-vllm` (your mount) and
`rpl2-vllm-pnfs-lab` (the one-host pNFS lab) run them; the
[NFS and pNFS page](../../../../docs/lmcache-nfs-held-descriptors.html#vllm)
explains the test.

| Script | Role |
|---|---|
| [setup.sh](setup.sh) | Builds LMCache with its CUDA extensions against an existing vLLM environment, in the task root, and puts missing LMCache dependencies in a separate directory. Nothing is installed into the vLLM environment. |
| [serve.sh](serve.sh) | Starts the LMCache MP server with the `fs_native` adapter and vLLM with the `LMCacheMPConnector`, or vLLM alone for the recompute baseline. |
| [bench.py](bench.py) | Sends the prompts, times each request, checks the cached-token count, and records the NFS client's RPCs for the mount. |
| [run.sh](run.sh) | The run: recompute baseline, one populate arm per I/O mode, then stock and held arms in alternating order, each in a fresh server. In the pNFS lab it also runs each phase while the lab's extra clients load the metadata server. |
| [summarize.py](summarize.py) | `summary.md`: time to first token, requests per second, NFS calls and metadata-server operations per request, and the change from stock to held. |
