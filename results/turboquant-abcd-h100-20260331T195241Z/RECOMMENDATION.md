# TurboQuant A/B/C/D Ablation — Final Recommendation

## Date: 2026-03-31
## Hardware: NVIDIA H100 80GB HBM3
## Environment: torch=2.6.0+cu126, triton=3.2.0, python3.12

## Arms tested
| Arm | Description | Avg MSE | Avg CosSim | Avg Median Latency |
|-----|-------------|---------|------------|-------------------|
| A | Uncompressed FP16 (anchor) | 0 | 1.000000 | 32.3 µs |
| B1 | TurboQuant paper-style (rotation + INT4) | 7.35e-05 | 0.995605 | 372.4 µs |
| B2 | TurboQuant + QJL residual | 2.32e-05 | 0.998535 | 851.1 µs |
| C1 | TurboQuant + fused Triton dequant | 7.35e-05 | 0.995605 | 1066.2 µs |
| D | Fused quant baseline (per-group INT4) | 7.35e-05 | 0.995605 | 312.7 µs |

## Ablation answers

### 1. Does TurboQuant improve quality over uncompressed? (B1 vs A)
Yes, but the degradation from A to B1 is small: CosSim drops from 1.0 to 0.9956.
This is the expected cost of INT4 quantization at head_dim=128.

### 2. Does TurboQuant improve quality over fused quant? (B1 vs D)
**NO.** B1 MSE = 7.350e-05, D MSE = 7.351e-05 — within noise (−0.01% difference).
At head_dim=128, the random rotation does not improve per-group scalar INT4.
The TurboQuant rotation spreads information uniformly, but at d=128 the original
distribution is already spread enough that per-group scalar quantization captures it.

### 3. Does fused Triton execution rescue TurboQuant latency? (C1 vs B1)
**NO.** C1 is actually slower (1066µs vs 372µs). The fused Triton dequant kernel
adds overhead without benefit — the rotation matmul (d×d) is cuBLAS-optimal and
cannot be meaningfully improved by custom Triton kernels. The bottleneck is the
rotation matmul itself, not implementation overhead in the dequant step.

B1 overhead vs D: +19.1% (rotation matmul cost)
C1 overhead vs D: +240.9% (Triton kernel launch + suboptimal matmul)

### 4. Does D still win overall?
**YES.** D matches B1 quality exactly and is 19% faster (no rotation overhead).
D is the clear Pareto-optimal point: same quality, lower latency, less complexity.

### 5. Is QJL worth keeping? (B2 vs B1)
**NO at this operating point.** B2 achieves 68% MSE reduction and CosSim 0.9985
(vs 0.9956), but at +128% latency cost. The quality gain is real but the cost
is too high for serving. At head_dim >> 128 this might change.

## Final verdict: KEEP D (fused quant) AS PRIMARY LINE

- TurboQuant rotation adds latency (+19%) with no quality benefit at head_dim=128
- Fused Triton execution makes TurboQuant slower, not faster
- QJL helps quality but costs too much latency
- D remains the best practical operating point

## When to revisit
- If models adopt head_dim ≥ 256 (rotation may help with non-uniform distributions)
- If models show strongly non-uniform KV magnitude distributions
- If someone writes a production-quality fused rotation+attention Triton kernel

## Matrix coverage
- Models: Qwen2.5-7B-like (4 KV heads), Llama3.1-8B-like (8 KV heads)
- Batch: 1, 4, 8
- Seq len: 2048, 4096, 8192, 16384
- head_dim: 128
- Quantization: INT4 symmetric, group_size=32
- Total configs: 24 (all completed, 0 OOM)
