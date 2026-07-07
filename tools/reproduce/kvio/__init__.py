# SPDX-License-Identifier: MIT
"""kvio storage-IO capture / replay reproduce orchestrator.

Standalone effort (imports nothing from the other reproduce orchestrators):
drive real request traces through vLLM + LMCache and capture the KV-offload
storage behaviour so it can be analysed -- and reproduced -- GPU-free later.

Two profiles, selected by the kvio-* defconfigs:

  * mooncake -- replay a production Mooncake (FAST25) request trace (real
    arrival timing + real prefix-reuse structure; token IDs synthesised to
    preserve the hash structure).  Measures prefix-cache reuse, timing, and
    KV-offload traffic.

  * content  -- replay REAL datasets (LMSYS-Chat-1M conversations or LongBench
    long-context questions), tokenized with the target model, and CAPTURE the
    LMCache KV corpus plus the semantic trace for offline, GPU-free
    KV-geometry / reuse / storage analysis.

Invoked via ``python3 -m tools.reproduce.kvio.run <subcommand> --config .config``
(see Makefile.kvio).
"""
