import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Config -----------------
B = 4       # batch
H = 16      # heads
T = 1024    # sequence length
D = 64      # head dim
warmup = 10
iters = 50

print(f"Device: {device}")
print(f"Shape: q, k = [{B}, {H}, {T}, {D}]")

q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)


# ----------------- View sanity check -----------------
def inspect_view():
    scores = torch.matmul(q, k.transpose(-2, -1))  # [B,H,T,T]
    scores_t = scores.transpose(-2, -1)

    print("\n[View inspection]")
    print("scores_t._base is scores:", scores_t._base is scores)
    print("scores.data_ptr == scores_t.data_ptr:",
          scores.storage().data_ptr() == scores_t.storage().data_ptr())
    print("scores.is_contiguous():", scores.is_contiguous())
    print("scores_t.is_contiguous():", scores_t.is_contiguous())
    print("scores.shape:", scores.shape, "scores_t.shape:", scores_t.shape)


inspect_view()


# ----------------- Timed functions (eager) -----------------
def qk_only(q, k):
    # Standard GPT-2: Q K^T
    return torch.matmul(q, k.transpose(-2, -1))

def qk_then_transpose(q, k):
    scores = torch.matmul(q, k.transpose(-2, -1))
    return scores.transpose(-2, -1)

def kq_direct(q, k):
    # Reciprocal: K Q^T
    return torch.matmul(k, q.transpose(-2, -1))


def benchmark(fn, name):
    # warmup
    for _ in range(warmup):
        out = fn(q, k)
        if device == "cuda":
            torch.cuda.synchronize()

    # timed
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k)
        if device == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()

    ms = (end - start) * 1000.0 / iters
    print(f"{name:20s}: {ms:8.3f} ms")
    return out


print("\n[Eager mode timings]")
out_qk   = benchmark(qk_only,          "qk_only")
out_qk_t = benchmark(qk_then_transpose,"qk_then_transpose")
out_kq   = benchmark(kq_direct,        "kq_direct")

# Numerical sanity: KQ^T == (QK^T)^T
diff = (out_kq - out_qk_t).abs().max().item()
print(f"\nMax |kq_direct - (qk_then_transpose)|: {diff:.3e}")


# ----------------- torch.compile section -----------------
if hasattr(torch, "compile"):
    print("\n[torch.compile timings]")

    compiled_qk_only = torch.compile(qk_only, mode="max-autotune", fullgraph=False)
    compiled_qk_t    = torch.compile(qk_then_transpose, mode="max-autotune", fullgraph=False)
    compiled_kq      = torch.compile(kq_direct, mode="max-autotune", fullgraph=False)

    # ---- correctness check (single run, with clone) ----
    out_qk_c   = compiled_qk_only(q, k).clone()
    out_qk_t_c = compiled_qk_t(q, k).clone()
    out_kq_c   = compiled_kq(q, k).clone()
    if device == "cuda":
        torch.cuda.synchronize()

    diff_c = (out_kq_c - out_qk_t_c).abs().max().item()
    print(f"\nMax |compiled_kq - compiled(qk_then_T)|: {diff_c:.3e}")

    # ---- perf benchmark (ignore outputs) ----
    print("\n[torch.compile perf only]")
    benchmark(compiled_qk_only, "compiled_qk_only")
    benchmark(compiled_qk_t,    "compiled_qk_then_T")
    benchmark(compiled_kq,      "compiled_kq_direct")
else:
    print("\n[torch.compile not available in this PyTorch build]")

