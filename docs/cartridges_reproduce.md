# Reproducing CartridgeConnector Validation Tests

This guide explains how to reproduce all CartridgeConnector serving
validation tests from the `20260429-cartridges-code-only` branch.

## Quick start (knlp pipeline)

If you have the [knlp](https://github.com/mcgrof/knlp) repo:

```bash
cd knlp
make defconfig-cartridges-vllm-tests
make
```

This runs: doctor → fetch → build → test → report. It clones the
vLLM branch, installs it, and runs all 178 unit tests. If the vLLM
repo already exists next to knlp, it reuses it (no re-clone).

## Manual reproduction

### Prerequisites

- GPU: NVIDIA H100/A100 (1 GPU for Tiers -1 through 5, 2 GPUs for Tier 6)
- Python 3.12
- [uv](https://astral.sh/uv) package manager
- HuggingFace token at `~/.cache/huggingface/token` (for gated models)

### Step 1: Clone and install

```bash
git clone --branch 20260429-cartridges-code-only \
  https://github.com/mcgrof/vllm.git
cd vllm

VLLM_USE_PRECOMPILED=1 uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install transformers==4.55.4 pytest pytest-timeout tblib requests
```

### Step 2: Run unit tests (Tier -1, no GPU)

```bash
pytest tests/v1/core/test_cartridge*.py -v --timeout=120
```

Expected: 178 passed.

These tests cover:
- `test_cartridge_connector.py` — KV injection, slot mapping, fake writer
- `test_cartridge_connector_integration.py` — scheduler API state machine
- `test_cartridge_manifest.py` — JSON manifest validation
- `test_cartridge_registry.py` — SQLite registry operations
- `test_cartridge_store.py` — store lifecycle, refcount, thread safety
- `test_cartridge_router.py` — explicit/static/label/composite dispatch
- `test_cartridge_routing.py` — per-request isolation, no cross-contamination
- `test_cartridge_fault.py` — error path safety
- `test_cartridge_gpu_residency.py` — LRU eviction, pin/unpin, memory
- `test_cartridge_lmcache_plugin.py` — optional import, plugin behavior

### Step 3: Smoke serve (Tier 0, needs GPU + cartridge)

You need a trained cartridge `.pt` file. If you don't have one, you
can build a prefilled-cache canary from any text:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.bfloat16).to("cuda").eval()

text = "Your document text here..."
ids = tok.encode(text, add_special_tokens=False)
cache = DynamicCache()
with torch.no_grad():
    model(input_ids=torch.tensor([ids], device="cuda"),
          past_key_values=cache, use_cache=True, return_dict=True)

Ks, Vs = [], []
for li in range(len(cache.layers)):
    layer = cache.layers[li]
    k = layer.keys.cpu() if hasattr(layer, "keys") else layer.key_cache[0].cpu()
    v = layer.values.cpu() if hasattr(layer, "values") else layer.value_cache[0].cpu()
    Ks.append(k); Vs.append(v)

torch.save({"trainable_keys": Ks, "trainable_values": Vs,
            "frozen_keys": [], "frozen_values": []}, "canary.pt")
```

Then serve:

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "CartridgeConnector",
    "kv_connector_module_path":
      "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector",
    "kv_connector_extra_config": {
      "cartridge_path": "canary.pt"
    },
    "kv_role": "kv_both"
  }'
```

Send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct",
       "messages":[{"role":"user","content":"Your document text here... What is the answer?"}],
       "max_tokens":50,"temperature":0}'
```

The prompt must include the document text as a prefix so the
CartridgeConnector has tokens to match against. The connector
replaces the KV computation for those prefix tokens with the
cartridge's pre-computed KV.

### Step 4: Multi-cartridge dispatch (Tier 3)

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "CartridgeConnector",
    "kv_connector_module_path":
      "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector",
    "kv_connector_extra_config": {
      "cartridges": [
        {"cartridge_id": "doc_a", "path": "cart_a.pt"},
        {"cartridge_id": "doc_b", "path": "cart_b.pt"}
      ],
      "router": {"type": "explicit"}
    },
    "kv_role": "kv_both"
  }'
```

Send per-cartridge requests:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct",
       "messages":[{"role":"user","content":"Doc A text... Question?"}],
       "max_tokens":50,"temperature":0,
       "extra_body":{"kv_transfer_params":{"cartridge_id":"doc_a"}}}'
```

### Step 5: TP=2 (Tier 6, needs 2 GPUs)

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
  --tensor-parallel-size 2 \
  --kv-transfer-config '{...same as above...}'
```

The CartridgeConnector automatically shards the cartridge KV
heads across TP ranks.

## What the tests validate

| Tier | What | How to run |
|---|---|---|
| -1 | Unit tests (178) | `pytest tests/v1/core/test_cartridge*.py` |
| 0 | Server starts with cartridge | `vllm serve` + curl |
| 0.5 | Prefill canary (logit match) | Build cart from known text, compare output |
| 1 | Context questions (10/10) | Send factual questions about cartridge document |
| 2 | 16 concurrent requests | Thread pool, same cartridge |
| 3 | Multi-cart dispatch | 3 carts, ExplicitCartridgeRouter |
| 4 | 50 sequential stability | Repeated requests, server stays alive |
| 5 | Error handling | Empty prompt, short prompt |
| 6 | TP=2 | 2 GPUs, KV head sharding |

## Known issues

- `echo=True` + `max_tokens=0` returns 500 (V1 API edge case)
- Prompt must include document text as prefix for cartridge context
- Tokenizer patch needed for fork-point compatibility (automated in knlp pipeline)
