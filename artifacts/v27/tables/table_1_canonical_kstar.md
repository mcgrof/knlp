| Model | Arch | D | GPU | Quant | Ranking | eps | k*(3%) | kv_ratio | max_delta | PASS_3% | PASS_1% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | Qwen2 | 24 | W7900 | g32 INT4/INT8 | oracle | 3% | 2 | 0.3008 | 2.85% | Y | N |
| Qwen2.5-1.5B | Qwen2 | 28 | W7900 | g32 INT4/INT8 | oracle | 3% | 2 | 0.2974 | 1.05% | Y | Y |
| Qwen2.5-7B | Qwen2 | 28 | H100 | g32 INT4/INT8 | oracle | 3% | 2 | 0.2974 | 1.05% | Y | Y |
| Mistral-7B | Mistral | 32 | H100 | g32 INT4/INT8 | oracle | 3% | 0 | 0.2812 | 0.48% | Y | Y |
| Llama-2-7b | Llama | 32 | H100 | g32 INT4/INT8 | oracle | 3% | 0 | 0.2812 | 1.01% | Y | Y |
