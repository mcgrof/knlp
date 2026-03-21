# Cartridges: Pre-Trained KV Caches for Document-Grounded Inference

A Cartridge is a pre-trained KV cache. Train it once per document offline,
inject it into the model's KV cache at serving time, skip prefill entirely.

The paper: [CARTRIDGE: Compact Document Representation for LLM Inference](https://arxiv.org/abs/2508.17032).

## Why this matters

Long-context prefill is expensive. A 16K-token document takes over 2 seconds
on a W7900 before the model generates a single output token. Cartridges
eliminate that cost by precomputing the KV cache offline and loading it as a
CPU-to-GPU memcpy at request time.

Cartridges also compress: a 16K-token document can be represented by 4K or
even 256 trained KV entries, reducing both memory and injection latency while
retaining answer quality.

## What we tested

55 cartridge configurations across 3 model architectures and 6 documents
spanning 2K--49K tokens:

| Model | Arch | Layers | KV Heads | Configs |
|-------|------|--------|----------|---------|
| Qwen2.5-7B-Instruct | GQA | 28 | 4 | 23 |
| Qwen2.5-1.5B-Instruct | GQA | 28 | 2 | 12 |
| Llama-2-7b-hf | MHA | 32 | 32 | 20 |

Documents: ML research (8K, 20K tokens), legal (GPL v3, 7K), medical (12K),
technical (RFC 9110, 49K), and short reference (Wikipedia, 2K).

All experiments validated on NVIDIA H100 80GB and AMD Radeon Pro W7900 48GB
(ROCm 6.4.3).

## Key results

**10.6x prefill speedup** at 16K tokens on W7900. The speedup scales
sub-linearly with prefix length because loading pre-computed KV (memcpy) is
cheaper than running a forward pass:

| Prefix tokens | ICL prefill | Cartridge prefill | Speedup |
|---------------|-------------|-------------------|---------|
| 256 | 0.037s | 0.038s | 1.0x |
| 1,024 | 0.072s | 0.041s | 1.8x |
| 4,096 | 0.312s | 0.071s | 4.4x |
| 8,192 | 0.758s | 0.110s | 6.9x |
| 16,384 | 2.027s | 0.192s | **10.6x** |

Measured on Llama-3.2-1B-Instruct, AMD W7900, FlexAttention + torch.compile.

**Quality at compression.** LLM-as-judge scores (0--5 scale) for
Qwen2.5-7B-Instruct: 3.7--4.5 at full size, 2.8--4.1 at 4x compression,
2.4--3.7 at p=256. Instruct models compress well; base models struggle at
4x+ compression.

**GQA models produce dramatically smaller cartridges.** Llama-2-7b (32 KV
heads) creates cartridges 8.5x larger than Qwen-7B (4 KV heads) at the same
prefix length. Checkpoint size scales linearly: ~57 KB per token for Qwen-7B,
~469 KB per token for Llama-2-7b. At p=256: Qwen-7B = 15 MB, Llama-2-7b =
128 MB.

## SCI vs First-k initialization

Two strategies for initializing the trainable KV cache before optimization:

**First-k.** Initialize with the first *p* key-value vectors from running the
document through the model. Simple and deterministic.

**SCI (Sampled Chunk Initialization).** Initialize by sampling random 64-token
chunks from across the document (seed=42). Captures broader document content
in the initial state.

Both produce nearly identical training times, decode throughput, and end-to-end
performance. SCI's broader initialization does not translate into measurable
quality advantage over First-k in our experiments. Training times on W7900
(270 steps, 1 epoch):

| Prefix | First-k | SCI |
|--------|---------|-----|
| 256 | 661s | 682s |
| 4,096 | 1,144s | 1,131s |
| 16,384 | 2,684s | 2,610s |

End-to-end throughput at p=16,384: First-k = 44.5 tok/s, SCI = 44.2 tok/s,
ICL baseline = 27.4 tok/s.

## Prefix tuning background

Cartridges build on **Prefix Tuning** (Li and Liang, 2021). The original
idea: prepend a sequence of *p* trainable continuous vectors to the key and
value matrices at every attention layer. These vectors are optimized via
gradient descent while the model weights stay frozen.

The initialization strategy matters. Li and Liang found that initializing
prefix vectors from real word embeddings (e.g., sampling from the model's
vocabulary embedding matrix) outperforms random Gaussian initialization.
Activations from running real tokens through the model provide an even
stronger starting point --- this is essentially what First-k initialization
does in cartridges: the initial KV state comes from actually processing the
document, giving the optimizer a warm start in a region of KV space the model
already understands.

Cartridges extend prefix tuning from a fine-tuning technique to an inference
optimization: the trained prefix *is* the document representation, and
injecting it replaces prefill entirely.

## vLLM integration

CartridgeConnector plugs into vLLM as a `KVConnectorBase_V1` plugin
with **zero vLLM core modifications**. The connector is about 300 lines of
Python.

A branch with the connector already in place is available at
[mcgrof/vllm:20260320-cartridge-connector](https://github.com/mcgrof/vllm/tree/20260320-cartridge-connector).

### Minimum requirements

- vLLM >= 0.16.0 (needs `KVConnectorBase_V1` API)
- A trained cartridge checkpoint (`.pt` file)
- The tokenized prefix (`prefix_token_ids.json`)
- The connector file placed in vLLM's connector directory

### Connector architecture

The connector owns three responsibilities:

1. **Load cartridge state** at server startup: deserialize the checkpoint and
   tokenized prefix.
2. **Report prefix match length** for each incoming request: compare request
   token IDs against stored prefix, return block-aligned match count.
3. **Scatter KV into GPU blocks** using vLLM's slot mapping: copy pre-trained
   KV from CPU to the physical GPU addresses vLLM assigned.

The connector does not allocate KV memory, does not replace attention, and
does not bypass the scheduler. vLLM owns allocation; the connector only fills
the slots vLLM assigned.

### Starting the server

```bash
export CARTRIDGE=/path/to/cartridge.pt
export PREFIX_IDS=/path/to/prefix_token_ids.json

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.40 \
  --kv-transfer-config '{
    "kv_connector": "CartridgeConnector",
    "kv_connector_module_path": "vllm.distributed.kv_transfer.kv_connector.v1.cartridge_connector",
    "kv_connector_extra_config": {
      "cartridge_path": "'"$CARTRIDGE"'",
      "prefix_token_ids_path": "'"$PREFIX_IDS"'"
    },
    "kv_role": "kv_both"
  }'
```

- `kv_connector_module_path` points vLLM's factory at the custom module via
  dynamic import --- no forked registration needed.
- `cartridge_path` and `prefix_token_ids_path` are the two runtime artifacts.
- `kv_role=kv_both` tells vLLM the connector participates in both load/save
  sides of the connector contract.

### Connector implementation

```python
class CartridgeConnector(KVConnectorBase_V1):
    def __init__(self, rank, local_rank, config):
        self.cartridge = torch.load(config.cartridge_path, map_location="cpu")
        self.prefix_token_ids = load_json(config.prefix_token_ids_path)
        self.layer_k = [layer["k"].contiguous() for layer in self.cartridge]
        self.layer_v = [layer["v"].contiguous() for layer in self.cartridge]

    def get_num_new_matched_tokens(self, request, **kwargs):
        matched = longest_common_prefix(
            request.prompt_token_ids, self.prefix_token_ids
        )
        return align_down_to_block_size(matched, block_size=16)

    def start_load_kv(self, connector_meta, **kwargs):
        slot_mapping = connector_meta["slot_mapping"]
        matched = connector_meta["matched_tokens"]
        for layer_idx in range(len(self.layer_k)):
            scatter_to_gpu_blocks(
                self.layer_k[layer_idx][:matched],
                self.layer_v[layer_idx][:matched],
                slot_mapping[:matched],
            )
```

The full API surface: `__init__`, `get_num_new_matched_tokens`,
`update_state_after_alloc`, `build_connector_meta`, `start_load_kv`,
`save_kv_layer` (no-op for cartridges).

### Connector file location

Place the connector file inside the installed vLLM package:

```
<venv>/lib/python3.12/site-packages/vllm/
  distributed/kv_transfer/kv_connector/v1/
    cartridge_connector.py
```

No modifications to `factory.py` or any other vLLM file are needed.

### Sending requests

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system",
       "content": "<full training document text>"},
      {"role": "user",
       "content": "What is the main contribution?"}
    ]
  }'
```

The system prompt must contain the same document the cartridge was trained on.
The connector matches token IDs to determine how much KV to inject.

## AMD ROCm validation

All cartridge training and inference components work on W7900 (RDNA 3,
gfx1100, ROCm 6.4.3):

- FlexAttention compiles and runs correctly
- torch.compile graph compilation succeeds (first-run ~5 min overhead)
- Full training sweep (7 prefix sizes x 2 init strategies = 14 runs) completed
- vLLM serving via Docker (`rocm/vllm:latest`)

Native vLLM installation fails on Debian testing due to ROCm version mismatch
(vLLM ships torch 2.9.1+rocm700, system has ROCm 6.4.3) and GCC 15
incompatibility. Docker is the working path:

```bash
docker pull rocm/vllm:latest
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --network=host \
  -v /data:/data \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HIP_VISIBLE_DEVICES=0 \
  rocm/vllm:latest \
  python -c "import vllm; print(vllm.__version__)"
```

## Why LMCache integration was not needed

LMCache is content-addressed: it identifies cached KV by hashing token IDs.
Cartridges are identity-addressed: a compressed cartridge (p=512 for a
16K-token document) contains KV for 512 "virtual" positions that were
*trained*, not computed from any specific 512-token sequence. No token hash
maps to this trained KV.

CartridgeConnector implements the same `KVConnectorBase_V1` interface as
`LMCacheConnectorV1`. When multi-cartridge serving is needed, LMCache's
`StorageManager` can serve as the backend storage layer while keeping
CartridgeConnector's matching logic.

## Next steps

### Marin integration

Evaluate integrating cartridge support into the
[Marin project](https://marin.community/). Marin's training infrastructure
could provide a natural path for offline cartridge training at scale, and
its evaluation framework could benchmark cartridge quality across a broader
model and document set than the current 55-configuration sweep.

### Broader architecture coverage

Extend testing to MLA architectures (DeepSeek) and larger models (70B+).
The current results suggest GQA models are strongly preferred for deployment
due to 8x smaller checkpoints, but MLA's compressed KV representation may
interact differently with cartridge training.

### Production serving patterns

Evaluate multi-cartridge serving (one server, many documents) using vLLM's
`MultiConnector` to multiplex between cartridges based on request routing.

## Related work

- [CARTRIDGE paper](https://arxiv.org/abs/2508.17032): the original
  pre-trained KV cache technique
- [Prefix Tuning](https://arxiv.org/abs/2101.00190) (Li and Liang, 2021):
  the foundation -- trainable continuous prefix vectors prepended to attention
- [MoBA: Mixture of Block Attention](https://arxiv.org/abs/2511.11571):
  theoretical foundation for key-based routing in block attention, relevant
  to understanding why key-based KV compression works
- [Fused KV quantization](https://github.com/mcgrof/knlp/blob/main/docs/fused_kv_quantization.md):
  complementary approach -- cartridges skip prefill, fused quantization
  reduces decode traffic. Both target KV cache cost from different angles
- [Interactive visualization](https://mcgrof.github.io/knlp/cartridges_visualization.html):
  tabbed walkthrough of cartridge findings and prefix tuning background
