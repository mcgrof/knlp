# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 fused mixture-of-experts kernel for EMO, taken from vLLM.

EMO's reference code runs each expert as its own small matrix multiply in a
Python loop, which is far slower than a serving engine. This module replaces
that loop with the kernel that vLLM uses for unquantized MoE layers, so that
timing measurements reflect production compute rather than Python overhead.

The Triton kernel is vLLM's ``fused_moe_kernel`` from
``vllm/model_executor/layers/fused_moe/fused_moe.py`` at upstream commit
3b364baef5 (2026-10-22), reduced to its BF16 path: the quantization, bias,
tensor-descriptor and swapped-operand branches are removed and nothing else
is changed. The host code follows vLLM's ``fused_experts_impl``: route,
align token slots to expert blocks, run the gate/up kernel, apply SiLU and
multiply, run the down kernel with the router weight, and sum the slots.
vLLM's ``moe_align_block_size`` is a C++ operator, so it is reimplemented
here with PyTorch operations that do not synchronize with the host. The
default tile configuration is vLLM's ``get_default_config`` for ROCm; a
tuned table from ``tune_fused_moe.py`` can replace it.

The file keeps vLLM's Apache-2.0 license.
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

VLLM_SOURCE = {
    "file": "vllm/model_executor/layers/fused_moe/fused_moe.py",
    "upstream_commit": "3b364baef5cd731b5e1f14ff16387feb69d01f10",
    "blob": "e0610c878d2c584375b4539c4fd408763cb2b97d",
}


@triton.jit
def fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    # Grouped ordering of output blocks, as in vLLM, for L2 reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def default_config(M, E):
    """vLLM's get_default_config for BF16 on ROCm (two pipeline stages)."""
    if M <= 32:
        block_m = 16
    elif M <= 96:
        block_m = 32
    elif M <= 512:
        block_m = 64
    else:
        block_m = 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64 if M <= 64 else 128,
        "BLOCK_SIZE_K": 128 if M <= 64 else 64,
        "GROUP_SIZE_M": 16 if M // max(E, 1) > 128 else 1,
        "num_warps": 4 if M <= 128 else 8,
        "num_stages": 2,
    }


class ConfigTable:
    """Pick a tile configuration per (experts, tokens).

    With a tuned table, the entry with the nearest token count is used, as
    vLLM does with its per-device JSON files; otherwise the default.
    """

    def __init__(self, path=None):
        self.table = {}
        self.source = "vllm-default-rocm"
        if path:
            with open(path) as f:
                raw = json.load(f)["configs"]
            self.table = {
                int(e): {int(m): c for m, c in per.items()} for e, per in raw.items()
            }
            self.source = path

    def __call__(self, M, E):
        per = self.table.get(E)
        if not per:
            return default_config(M, E)
        return per[min(per, key=lambda m: abs(m - M))]


def moe_align_block_size(topk_ids, block_m, num_experts):
    """Sort token slots by expert and pad each expert to a block multiple.

    Same outputs as vLLM's operator: sorted slot ids padded with the slot
    count, the expert of each block, and the padded slot total.
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    numel = flat.numel()
    dev = flat.device
    max_pad = numel + num_experts * (block_m - 1)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=dev)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block_m - 1) // block_m * block_m
    pad_end = torch.cumsum(padded, 0)
    pad_start = pad_end - padded
    start = torch.cumsum(counts, 0) - counts
    order = torch.argsort(flat, stable=True)
    se = flat[order]
    dest = pad_start[se] + (torch.arange(numel, device=dev) - start[se])
    sorted_ids = torch.full((max_pad,), numel, dtype=torch.int32, device=dev)
    sorted_ids[dest] = order.to(torch.int32)
    nblocks = triton.cdiv(max_pad, block_m)
    starts = torch.arange(nblocks, device=dev) * block_m
    expert_ids = torch.searchsorted(pad_end, starts, right=True)
    expert_ids = expert_ids.clamp_(max=num_experts - 1).to(torch.int32)
    return sorted_ids, expert_ids, pad_end[-1:].to(torch.int32)


def _invoke(A, B, C, topk_weights, sorted_ids, expert_ids, ntpp, mul_w, top_k, cfg):
    M = A.size(0)
    num_tokens = M * top_k
    bm = cfg["BLOCK_SIZE_M"]
    if sorted_ids is not None:
        EM = sorted_ids.size(0)
        if M < bm:
            EM = min(EM, M * top_k * bm)
    else:
        EM = num_tokens * bm
    grid = (triton.cdiv(EM, bm) * triton.cdiv(B.size(1), cfg["BLOCK_SIZE_N"]),)
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_ids,
        expert_ids,
        ntpp,
        B.size(1),
        B.size(2),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        naive_block_assignment=sorted_ids is None,
        MUL_ROUTED_WEIGHT=mul_w,
        top_k=top_k,
        compute_type=tl.bfloat16,
        **cfg,
    )


def fused_experts(x, w13, w2, topk_weights, topk_ids, cfg):
    """Routed expert output for BF16 ``x`` [T, H].

    ``w13`` is [E, 2I, H] (gate rows, then up rows), ``w2`` is [E, H, I],
    ``topk_weights`` is float32 [T, k] and ``topk_ids`` is int [T, k].
    """
    T, H = x.shape
    E, N2, _ = w13.shape
    I = N2 // 2
    k = topk_ids.size(1)
    assert x.is_contiguous() and topk_weights.is_contiguous()
    if T * k * 4 <= E:
        # vLLM's low-latency path: one block per token slot, no sorting.
        sorted_ids = None
        expert_ids = topk_ids.reshape(-1).to(torch.int32)
        ntpp = torch.full(
            (1,),
            topk_ids.numel() * cfg["BLOCK_SIZE_M"],
            dtype=torch.int32,
            device=x.device,
        )
    else:
        sorted_ids, expert_ids, ntpp = moe_align_block_size(
            topk_ids, cfg["BLOCK_SIZE_M"], E
        )
    c1 = torch.empty((T, k, N2), dtype=x.dtype, device=x.device)
    _invoke(x, w13, c1, topk_weights, sorted_ids, expert_ids, ntpp, False, k, cfg)
    c1 = c1.view(-1, N2)
    c2 = F.silu(c1[:, :I]) * c1[:, I:]
    c3 = torch.empty((T, k, H), dtype=x.dtype, device=x.device)
    _invoke(c2, w2, c3, topk_weights, sorted_ids, expert_ids, ntpp, True, 1, cfg)
    return c3.sum(dim=1)


def _stack(experts):
    w13 = torch.stack(
        [torch.cat([e.gate_proj.weight, e.up_proj.weight], 0) for e in experts]
    ).contiguous()
    w2 = torch.stack([e.down_proj.weight for e in experts]).contiguous()
    return w13, w2


class FusedEmoMoe(nn.Module):
    """EMO's legacy-path MoE block with the fused routed computation.

    Routing is EMO's legacy shared-expert path: a softmax over the routed
    experts picks ``top_k - num_shared``, and the shared experts (the last
    indices) always run with weight 1. Routed experts go through the fused
    kernel; shared experts run as ordinary dense matrix multiplies, as
    serving engines do. Router weights are rounded to BF16 first, as EMO's
    code does before it scales each expert output.
    """

    def __init__(self, moe, configs):
        super().__init__()
        assert not moe.norm_topk_prob
        aa = getattr(moe, "always_active_experts", None)
        assert aa is None or len(aa) == 0
        assert moe.num_shared_experts > 0
        self.gate = moe.gate
        self.top_k = moe.top_k
        self.num_shared = moe.num_shared_experts
        self.num_experts = moe.num_experts
        routed = self.num_experts - self.num_shared
        w13, w2 = _stack(moe.experts[:routed])
        s13, s2 = _stack(moe.experts[routed:])
        self.w13 = nn.Parameter(w13, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self.s13 = nn.Parameter(s13, requires_grad=False)
        self.s2 = nn.Parameter(s2, requires_grad=False)
        self.configs = configs

    @property
    def num_routed(self):
        return self.num_experts - self.num_shared

    def route(self, x):
        logits = self.gate(x)
        std = F.softmax(logits[:, : -self.num_shared], dim=1, dtype=torch.float)
        w, ids = torch.topk(std, self.top_k - self.num_shared, dim=-1)
        return logits, w.to(x.dtype).float().contiguous(), ids

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        x = hidden_states.reshape(-1, h).contiguous()
        logits, w, ids = self.route(x)
        out = fused_experts(
            x, self.w13, self.w2, w, ids, self.configs(x.size(0), self.num_routed)
        )
        inter = self.s13.size(1) // 2
        for j in range(self.num_shared):
            g = F.linear(x, self.s13[j])
            out = out + F.linear(F.silu(g[:, :inter]) * g[:, inter:], self.s2[j])
        return out.reshape(b, s, h), logits


def install_fused(model, configs=None):
    """Replace every MoE block of ``model`` in place; frees the expert lists."""
    configs = configs or ConfigTable()
    for layer in model.model.layers:
        old = layer.mlp
        layer.mlp = FusedEmoMoe(old, configs)
        old.experts = None
        del old
    torch.cuda.empty_cache()
    return configs


def _self_test():
    """Compare fused_experts with an FP32 per-expert loop on random data."""
    torch.manual_seed(0)
    H, I, k = 2048, 1024, 7
    worst = 0.0
    for E in (127, 32):
        w13 = (torch.randn(E, 2 * I, H, device="cuda") * 0.02).bfloat16()
        w2 = (torch.randn(E, H, I, device="cuda") * 0.02).bfloat16()
        for T in (1, 3, 7, 128, 1000):
            x = torch.randn(T, H, device="cuda").bfloat16()
            w, ids = torch.topk(torch.randn(T, E, device="cuda").softmax(-1), k)
            w = w.bfloat16().float().contiguous()
            y = fused_experts(x, w13, w2, w, ids, default_config(T, E)).float()
            ref = torch.zeros(T, H, device="cuda")
            for e in range(E):
                t, s = torch.where(ids == e)
                if t.numel():
                    g = x[t].float() @ w13[e].float().T
                    h = (F.silu(g[:, :I]) * g[:, I:]) @ w2[e].float().T
                    ref.index_add_(0, t, h * w[t, s, None])
            r = float((y - ref).norm() / ref.norm())
            worst = max(worst, r)
            print(f"E={E} T={T}: relative L2 error {r:.2e}", flush=True)
    assert worst < 1e-2, worst
    print(f"self-test pass, worst {worst:.2e}")


if __name__ == "__main__":
    _self_test()
