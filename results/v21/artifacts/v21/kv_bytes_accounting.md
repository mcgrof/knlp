# True KV Bytes Accounting — BPA v21

## Model: Qwen2.5-0.5B
- Layers: 24, KV heads: 2, head_dim: 64
- Dense bf16 bytes/token: 12288

## Accounting Table

| Config | Bytes/token | kv_ratio | Scale overhead % | Description |
|--------|-------------|----------|-----------------|-------------|
| dense_bf16 | 12288.0 | 1.0000 | 0.0% | Dense bf16 baseline |
| INT8_all | 6336.0 | 0.5156 | 0.0% | All layers INT8 symmetric |
| INT4_g32_k0 | 3456.0 | 0.2812 | 11.1% | INT4 g=32, 0 INT8 layers |
| INT4_g32_k1 | 3576.0 | 0.2910 | 11.1% | INT4 g=32, 1 INT8 layers |
| INT4_g32_k2 | 3696.0 | 0.3008 | 11.1% | INT4 g=32, 2 INT8 layers |
| INT4_g32_k3 | 3816.0 | 0.3105 | 11.1% | INT4 g=32, 3 INT8 layers |
| INT4_g32_k4 | 3936.0 | 0.3203 | 11.1% | INT4 g=32, 4 INT8 layers |
| INT4_g32_k6 | 4176.0 | 0.3398 | 11.1% | INT4 g=32, 6 INT8 layers |
| INT4_g8_k0 | 4608.0 | 0.3750 | 33.3% | INT4 g=8, 0 INT8 layers |
| INT4_g8_k2 | 4752.0 | 0.3867 | 33.3% | INT4 g=8, 2 INT8 layers |
| INT4_g8_k4 | 4896.0 | 0.3984 | 33.3% | INT4 g=8, 4 INT8 layers |
| INT4_g4_k0 | 6144.0 | 0.5000 | 50.0% | INT4 g=4, 0 INT8 layers |
| INT4_g4_k1 | 6152.0 | 0.5007 | 50.0% | INT4 g=4, 1 INT8 layers |
| INT4_g4_k2 | 6160.0 | 0.5013 | 50.0% | INT4 g=4, 2 INT8 layers |
| INT4_g4_k3 | 6168.0 | 0.5020 | 50.0% | INT4 g=4, 3 INT8 layers |
| INT4_g4_k4 | 6176.0 | 0.5026 | 50.0% | INT4 g=4, 4 INT8 layers |
| INT4_g4_k6 | 6192.0 | 0.5039 | 50.0% | INT4 g=4, 6 INT8 layers |
| INT4_perchan_k4 | 3776.0 | 0.3073 | 5.9% | INT4 per-channel (g=64), 4 INT8 layers |

## Key Findings

- g=4 scale overhead: 50.0% vs g=32: 11.1%
- g=4 kv_ratio (k=0): 0.5000 vs g=32 (k=0): 0.2812

## Configs with kv_ratio < 0.333

- **INT4_g32_k0**: kv_ratio=0.2812
- **INT4_g32_k1**: kv_ratio=0.2910
- **INT4_g32_k2**: kv_ratio=0.3008
- **INT4_g32_k3**: kv_ratio=0.3105
- **INT4_g32_k4**: kv_ratio=0.3203
- **INT4_perchan_k4**: kv_ratio=0.3073

## Configs with kv_ratio < 0.333 (PASS candidates)

These configs could potentially beat S2_k6 if they PASS@3%:

- **INT4_g32_k0**: kv_ratio=0.2812
- **INT4_g32_k1**: kv_ratio=0.2910
- **INT4_g32_k2**: kv_ratio=0.3008
- **INT4_g32_k3**: kv_ratio=0.3105
- **INT4_g32_k4**: kv_ratio=0.3203
- **INT4_perchan_k4**: kv_ratio=0.3073
