#!/usr/bin/env python3
"""Full-model decode-step Amdahl analysis for the certified LM-head kernel.

The kernel's 3.2x / 25.3%-bytes win is on the LM HEAD GEMV IN ISOLATION. A decode
step also reads all non-head weights (once per token) and the KV cache (grows with
context T, per sequence). So the FULL decode-step speedup is diluted by the head's
share of total per-token decode traffic (Amdahl on bytes). This quantifies that
share and the resulting full-step speedup per model / context / batch, so the
component win is not mis-sold as a full-model win.

Per-token decode traffic (bf16, E=2):
  head_dense = E * V * d
  non_head   = E * P - head_dense          # all layer weights, read once/token
  kv(T,B)    = B * L * 2 * n_kv * dh * T * E
  dense_step = non_head + head_dense + kv
  head_share = head_dense / dense_step
With the certified head at byte ratio q (mean 0.253, p95 0.659):
  step_ratio = (non_head + q*head_dense + kv) / dense_step
  speedup    = 1 / step_ratio = 1 / (1 - head_share*(1-q))
Weights are reused across batch B; KV is per-sequence (scales with B); the head
output is per-token. So head_share SHRINKS as B*T grows (KV dilutes it).
"""

E = 2
Q_MEAN = 0.25313   # mean certified head-table byte ratio (shadow + 7.4% fetch)
Q_P95 = 0.659      # p95 (48% fetch tail) -- the honest worst-case per token

# (name, P_params, V, d, L, n_kv, dh)
MODELS = [
    ("Qwen2.5-7B", 7.62e9, 152064, 3584, 28, 4, 128),
    ("Llama-3.2-3B", 3.21e9, 128256, 3072, 28, 8, 128),
    ("Qwen2.5-1.5B", 1.54e9, 151936, 1536, 28, 2, 128),
    ("Qwen2.5-0.5B", 0.49e9, 151936, 896, 24, 2, 64),
    ("(hypo) 1B/152k/d2048", 1.0e9, 152064, 2048, 24, 4, 128),
]
CONTEXTS = [0, 2048, 8192, 32768]
BATCHES = [1, 8, 16]


def gb(x):
    return x / 1e9


def analyze(P, V, d, L, n_kv, dh, T, B, q):
    head = E * V * d
    non_head = E * P - head
    kv = B * L * 2 * n_kv * dh * T * E
    dense = non_head + head + kv
    share = head / dense
    speedup = 1.0 / (1.0 - share * (1.0 - q))
    return share, speedup, head, dense


print("Certified LM-head: full decode-step speedup (Amdahl on bytes)")
print(f"q_mean={Q_MEAN:.3f}  q_p95={Q_P95:.3f}  (head-table byte ratio)\n")
for (name, P, V, d, L, n_kv, dh) in MODELS:
    head = E * V * d
    print(f"## {name}  head={gb(head):.2f}GB  weights={gb(E*P):.1f}GB")
    print(f"   {'B':>2} {'T':>6}  head_share  full_speedup(mean)  full_speedup(p95)")
    for B in BATCHES:
        for T in CONTEXTS:
            share, sp_mean, _, _ = analyze(P, V, d, L, n_kv, dh, T, B, Q_MEAN)
            _, sp_p95, _, _ = analyze(P, V, d, L, n_kv, dh, T, B, Q_P95)
            print(f"   {B:>2} {T:>6}  {share*100:>8.1f}%   {sp_mean:>8.3f}x (+{(sp_mean-1)*100:.1f}%)"
                  f"   {sp_p95:>6.3f}x (+{(sp_p95-1)*100:.1f}%)")
    print()

# headline thresholds (Codex): gain ~ (1-q_mean)*head_share
print("Full-step gain thresholds (mean q): gain ~ 0.747 * head_share")
for g in (0.05, 0.10, 0.20):
    print(f"  >{int(g*100)}% full-step gain needs head_share > {g/(1-Q_MEAN)*100:.1f}%")
