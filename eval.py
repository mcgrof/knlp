# === Same-FLOP RA (D_std+R=D) — fast path with head-selective routing, shared W,
# === zero-copy folding, and fused bias — designed to stay on Flash SDPA.

# Inputs:
#   Q,K,V:        [B,H,T,D]  (bf16/fp16, contiguous in last dim)
#   gates:        [H,3]      (per-head logits for [w_std, w_rec, w_disc])
#   W:            [D,R]      (shared across heads in this layer; bf16/fp16)
#   disc_vec_u:   [H,D] or [D]  (discoverability direction)
#   tau:          scalar threshold for w_rec to enable RA (e.g., 0.08)
#   causal/window mask options as flags
#
# Contract:
# - Keeps total head dim = D (baseline FLOPs).
# - Only heads with w_rec > tau use RA (folded). Others run pure SDPA.
# - Compute QW, KW once (shared W) and reuse for all RA heads.
# - Avoids torch.cat overhead by writing into preallocated folded buffers.

def ra_same_flop_fast(Q, K, V, gates, W, disc_vec_u=None, tau=0.08,
                      causal=False, ra_window=None, dropout_p=0.0):

    B, H, T, D = Q.shape
    device = Q.device
    dtype  = Q.dtype

    # ---- 0) Hygiene: enforce bf16/fp16 & contiguity (channels-last in last dim) ----
    assert Q.is_contiguous(memory_format=torch.contiguous_format)
    assert K.is_contiguous(memory_format=torch.contiguous_format)
    assert V.is_contiguous(memory_format=torch.contiguous_format)

    # ---- 1) Per-head gates → scales (stay in the small tensor space) ----
    w = torch.softmax(gates, dim=-1)            # [H,3]
    w_std = w[:, 0]                              # [H]
    w_rec = w[:, 1]                              # [H]
    w_disc = w[:, 2]                             # [H]
    use_ra = (w_rec > tau)                       # [H] boolean

    # Early exit: if no RA heads, run baseline SDPA once
    if use_ra.sum() == 0:
        return _sdpa_call(Q, K, V, disc_vec_u, w_disc, causal, ra_window, dropout_p)

    # ---- 2) Shared low-rank features (compute once per layer) ----
    # QW/KW: [B,H,T,R]
    # NOTE: matmul is batched; proj cost small & reused for all RA heads.
    QW = torch.matmul(Q, W)                      # [B,H,T,R]
    KW = torch.matmul(K, W)                      # [B,H,T,R]

    # ---- 3) Pack heads into two groups to avoid per-head branching ----
    idx_ra = torch.nonzero(use_ra, as_tuple=False).squeeze(1)       # [H_ra]
    idx_bl = torch.nonzero(~use_ra, as_tuple=False).squeeze(1)      # [H_bl]

    # Gather head slices (view-like, cheap)
    Q_bl, K_bl, V_bl = Q[:, idx_bl], K[:, idx_bl], V[:, idx_bl]     # [B,H_bl,T,D]
    Q_ra, K_ra, V_ra = Q[:, idx_ra], K[:, idx_ra], V[:, idx_ra]     # [B,H_ra,T,D]
    QW_ra, KW_ra     = QW[:, idx_ra], KW[:, idx_ra]                 # [B,H_ra,T,R]

    # Per-pack gate scales
    ws_bl = w_std[idx_bl]       # [H_bl]
    wd_bl = w_disc[idx_bl]      # [H_bl]
    ws_ra = w_std[idx_ra]       # [H_ra]
    wr_ra = w_rec[idx_ra]       # [H_ra]
    wd_ra = w_disc[idx_ra]      # [H_ra]

    # ---- 4) Build a SINGLE fused attention bias per pack (stay on Flash path) ----
    # Discoverability column bias (zero-mean per head), shape → [B,H,1,T]
    def make_disc_bias(K_pack, heads_idx, wd_pack):
        if disc_vec_u is None:
            return None
        u = disc_vec_u
        if u.dim() == 1:                    # [D] → broadcast per head
            u_eff = u.expand(len(heads_idx), -1)    # [H_pack,D]
        else:                                # [H,D] → index heads
            u_eff = u[heads_idx]                    # [H_pack,D]
        # d_raw: [B,H_pack,T]
        d_raw = torch.einsum("bhtd,hd->bht", K_pack, u_eff.to(K_pack.dtype))
        d = d_raw - d_raw.mean(dim=-1, keepdim=True)
        d = d.unsqueeze(-2)                  # [B,H_pack,1,T]
        d = d * wd_pack.view(1, -1, 1, 1)    # scale by w_disc
        return d.to(dtype)

    disc_bl = make_disc_bias(K_bl, idx_bl, wd_bl)   # or None
    disc_ra = make_disc_bias(K_ra, idx_ra, wd_ra)   # or None

    # Causal/window mask -> additive bias [B,H,T,T] or [1,1,T,T]
    attn_mask = _make_mask(B, H, T, device, dtype, causal, ra_window)  # may be None
    # Slice masks per pack (view), avoid new allocations
    mask_bl = None if attn_mask is None else attn_mask[:, idx_bl]
    mask_ra = None if attn_mask is None else attn_mask[:, idx_ra]

    # Merge biases: prefer in-place add to a single tensor per pack
    def fuse_bias(mask_pack, disc_pack):
        if mask_pack is None and disc_pack is None:
            return None
        if mask_pack is None:
            return disc_pack
        if disc_pack is None:
            return mask_pack
        # both exist; sum with minimal casts
        return mask_pack + disc_pack

    bias_bl = fuse_bias(mask_bl, disc_bl)
    bias_ra = fuse_bias(mask_ra, disc_ra)

    # ---- 5) Baseline heads: pure SDPA (no folding) ----
    out_bl = None
    if Q_bl.numel() > 0:
        # row-scale by sqrt(w_std) as a Q scaling to avoid extra ops in-kernel
        s_bl = ws_bl.clamp_min(1e-8).sqrt().view(1, -1, 1, 1)   # [1,H_bl,1,1]
        Q_bl_scaled = Q_bl * s_bl
        K_bl_scaled = K_bl * s_bl   # symmetric scaling keeps logits balanced
        out_bl = _sdpa_call(Q_bl_scaled, K_bl_scaled, V_bl, None, None,
                            causal=False, ra_window=None, dropout_p=dropout_p,
                            fused_bias=bias_bl)

    # ---- 6) RA heads: folded same-FLOP (D_std + R = D), zero-copy folding ----
    # Split standard channels (first D_std) and reciprocal channels (R from QW/KW)
    D_std = D - KW_ra.shape[-1]
    # Preallocate folded buffers to avoid torch.cat overhead
    Qf = torch.empty((B, Q_ra.shape[1], T, D), device=device, dtype=dtype)
    Kf = torch.empty_like(Qf)

    # Left half: sqrt(ws) * standard channels
    s_ra = ws_ra.clamp_min(1e-8).sqrt().view(1, -1, 1, 1)               # [1,H_ra,1,1]
    Qf[..., :D_std].copy_(Q_ra[..., :D_std] * s_ra)                     # no cat
    Kf[..., :D_std].copy_(K_ra[..., :D_std] * s_ra)

    # Right half: sqrt(wr) * reciprocal low-rank channels
    r_ra = wr_ra.clamp_min(1e-8).sqrt().view(1, -1, 1, 1)
    Qf[..., D_std:].copy_(KW_ra * r_ra)                                  # KW for Q_fold
    Kf[..., D_std:].copy_(QW_ra * r_ra)                                  # QW for K_fold

    out_ra = _sdpa_call(Qf, Kf, V_ra, None, None,
                        causal=False, ra_window=None, dropout_p=dropout_p,
                        fused_bias=bias_ra)

    # ---- 7) Stitch heads back in original order (single scatter) ----
    out = torch.empty_like(Q)
    if out_bl is not None:
        out[:, idx_bl].copy_(out_bl)
    out[:, idx_ra].copy_(out_ra)
    return out


# --- Helpers: single fused SDPA call with Flash/mem-efficient backend ---
def _sdpa_call(Q, K, V, disc_vec_u, w_disc, causal, ra_window, dropout_p, fused_bias=None):
    B, H, T, D = Q.shape

    # If we’re here from “no RA heads” path and we still want discoverability:
    if fused_bias is None and disc_vec_u is not None and w_disc is not None:
        # Build a [B,H,1,T] bias as in make_disc_bias() above and assign to fused_bias.
        raise NotImplementedError("Pass prebuilt fused_bias for baseline pack to stay on fast path.")

    # Reshape to [B*H, T, D] for PyTorch SDPA
    q = Q.transpose(1, 2).reshape(B*H, T, D)
    k = K.transpose(1, 2).reshape(B*H, T, D)
    v = V.transpose(1, 2).reshape(B*H, T, D)

    # Bias: None or [B,H,T,T] → [B*H,T,T]
    bias = None if fused_bias is None else fused_bias.reshape(B*H, T, T)

    # Keep Flash/mem-efficient path
    with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                        enable_mem_efficient=True,
                                        enable_math=False):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=dropout_p, is_causal=False
        )
    return out.reshape(B, T, H, D).transpose(1, 2)


def _make_mask(B, H, T, device, dtype, causal, ra_window):
    if not causal and ra_window is None:
        return None
    # Start with causal lower-tri mask
    M = torch.ones((1,1,T,T), device=device, dtype=torch.bool)
    if causal:
        M = torch.tril(M)
    if ra_window is not None:
        band = torch.zeros_like(M)
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        Dm = (i[:,None] - j[None,:]).abs()
        band[..., :, :] = Dm <= int(ra_window)
        M = M & band
    # Convert to additive bias (keep dtype to match SDPA expectations)
    bias = torch.empty((B, H, T, T), device=device, dtype=dtype)
    bias.masked_fill_(~M, float("-inf"))
    return bias

