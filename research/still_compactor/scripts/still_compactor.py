#!/usr/bin/env python3
"""STILL reference compactor (arXiv:2606.07878v1) - fidelity-corrected.

Corrections after the an external review:
  * non-parametric RMSNorm (was nn.LayerNorm);
  * INDEPENDENT parameters per block (was one shared set across both blocks);
  * full Appendix-C-style identity init: positional routing AND an identity
    [K;V] transport write-path in block 1 (block 2 + self-attn zeroed to no-ops),
    so the init is near-pass-through in DIRECTION at t=T, not just argmax-routing.

One compactor per layer; per-KV-head latent bank; projections shared across KV
heads but INDEPENDENT per block. Input per head X=concat(K_unrot,V); source keys
inverse-RoPE'd, values never rotated; compactor-internal RoPE base 10 on
L2-normalized (cosine) q,k; logits = d_latent*(q.k); 2 blocks, cross-attn each +
1-head latent self-attn, no FFN; output K/V heads; base-model RoPE reapplied to
output keys; beta=0; d_latent=2*head_dim=256.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

MAT_GUARD = 8192  # the reference materializes t x T scores; forbid long T here


def rmsnorm(x, eps=1e-6):
    """Non-parametric RMSNorm (paper's norm)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _rope_tables(positions, dim, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device,
                                             dtype=torch.float32) / dim))
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([ang, ang], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, positions, theta, inverse=False):
    cos, sin = _rope_tables(positions, x.shape[-1], theta, x.device, x.dtype)
    shape = [1] * (x.dim() - 2) + list(cos.shape)
    cos, sin = cos.view(shape), sin.view(shape)
    if inverse:
        sin = -sin
    return x * cos + _rotate_half(x) * sin


class _Block(nn.Module):
    """One compactor block: cross-attention + 1-head latent self-attention."""

    def __init__(self, head_dim, d_latent):
        super().__init__()
        d, dl = head_dim, d_latent
        self.W_q = nn.Linear(dl, dl, bias=True)      # latent query
        self.W_k = nn.Linear(2 * d, dl, bias=True)   # [K;V] -> key
        self.W_v = nn.Linear(2 * d, dl, bias=False)  # [K;V] -> value
        self.W_o = nn.Linear(dl, dl, bias=False)     # cross-attn out
        self.sq = nn.Linear(dl, dl, bias=False)
        self.sk = nn.Linear(dl, dl, bias=False)
        self.sv = nn.Linear(dl, dl, bias=False)
        self.so = nn.Linear(dl, dl, bias=False)


class STILLCompactorLayer(nn.Module):
    def __init__(self, n_kv_heads, head_dim, t=128, d_latent=None, n_blocks=2,
                 compactor_theta=10.0, base_theta=1e6, identity_init=True):
        super().__init__()
        self.H, self.d, self.t = n_kv_heads, head_dim, t
        self.d_latent = d_latent or 2 * head_dim
        self.n_blocks, self.compactor_theta, self.base_theta = (
            n_blocks, compactor_theta, base_theta)
        d, dl = head_dim, self.d_latent
        self.Z = nn.Parameter(torch.zeros(n_kv_heads, t, dl))
        self.blocks = nn.ModuleList(_Block(d, dl) for _ in range(n_blocks))
        self.W_key = nn.Linear(dl, d, bias=False)
        self.W_val = nn.Linear(dl, d, bias=False)
        if identity_init:
            self._identity_init()

    def _identity_init(self):
        d, dl = self.d, self.d_latent
        with torch.no_grad():
            self.Z.zero_()
            for bi, blk in enumerate(self.blocks):
                # constant q,k with energy in ALL dims so every RoPE frequency
                # contributes -> a sharp, non-aliased routing kernel peaked only
                # at p_i=p_j (a single-dim e0 activates one frequency and aliases
                # latent i onto i +/- 2*pi*k, smearing the transported value).
                blk.W_q.weight.zero_(); blk.W_q.bias.fill_(1.0)
                blk.W_k.weight.zero_(); blk.W_k.bias.fill_(1.0)
                blk.so.weight.zero_()                    # self-attn is a no-op
                if bi == 0:
                    # block 1 = identity [K;V] transport write path
                    blk.W_v.weight.zero_()
                    blk.W_v.weight[:2 * d, :2 * d] = torch.eye(2 * d)
                    blk.W_o.weight.copy_(torch.eye(dl))
                else:
                    # block 2+ start as no-ops (zero cross-attn output)
                    blk.W_v.weight.zero_()
                    blk.W_o.weight.zero_()
            self.W_key.weight.zero_(); self.W_key.weight[:, :d] = torch.eye(d)
            self.W_val.weight.zero_(); self.W_val.weight[:, d:2 * d] = torch.eye(d)

    def _cross(self, blk, Zn, k_unrot, v, src_pos, lat_pos):
        H, T, d = k_unrot.shape
        X = torch.cat([k_unrot, v], dim=-1)
        q = F.normalize(blk.W_q(Zn), dim=-1)
        k = F.normalize(blk.W_k(X), dim=-1)
        val = blk.W_v(X)
        q = apply_rope(q, lat_pos, self.compactor_theta)
        k = apply_rope(k, src_pos, self.compactor_theta)
        if T > MAT_GUARD:
            raise RuntimeError(f"reference materializes {self.t}x{T} (T>{MAT_GUARD})")
        logits = self.d_latent * torch.einsum("htl,hTl->htT", q, k)
        attn = torch.softmax(logits.float(), dim=-1).to(val.dtype)
        return blk.W_o(torch.einsum("htT,hTl->htl", attn, val))

    def _self(self, blk, Zn):
        q, k, v = blk.sq(Zn), blk.sk(Zn), blk.sv(Zn)
        a = torch.softmax((q @ k.transpose(-1, -2)).float()
                          / math.sqrt(self.d_latent), dim=-1).to(v.dtype)
        return blk.so(a @ v)

    @torch.no_grad()
    def first_cross_attn(self, k_unrot, v, src_pos):
        H, T, d = k_unrot.shape
        lat_pos = torch.linspace(float(src_pos[0]), float(src_pos[-1]), self.t,
                                 device=k_unrot.device)
        Z = self.Z.to(k_unrot.dtype).expand(H, self.t, self.d_latent)
        blk = self.blocks[0]
        q = F.normalize(blk.W_q(rmsnorm(Z)), dim=-1)
        k = F.normalize(blk.W_k(torch.cat([k_unrot, v], -1)), dim=-1)
        q = apply_rope(q, lat_pos, self.compactor_theta)
        k = apply_rope(k, src_pos, self.compactor_theta)
        return torch.softmax(self.d_latent
                             * torch.einsum("htl,hTl->htT", q, k).float(), dim=-1)

    def forward(self, k_unrot, v, src_pos):
        H, T, d = k_unrot.shape
        dev, dt = k_unrot.device, k_unrot.dtype
        lat_pos = torch.linspace(float(src_pos[0]), float(src_pos[-1]), self.t,
                                 device=dev)
        Z = self.Z.to(dt).expand(H, self.t, self.d_latent).contiguous()
        for blk in self.blocks:
            Z = Z + self._cross(blk, rmsnorm(Z), k_unrot, v, src_pos, lat_pos)
            Z = Z + self._self(blk, rmsnorm(Z))
        Zo = rmsnorm(Z)
        c_k, c_v = self.W_key(Zo), self.W_val(Zo)
        out_pos = lat_pos.round().long()
        c_k = apply_rope(c_k, out_pos, self.base_theta)
        return c_k, c_v, out_pos


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_device(0)
    H, d, T, t = 8, 128, 256, 128
    dt = torch.float32
    print(f"device={dev} H={H} d={d} T={T} t={t}")

    comp = STILLCompactorLayer(H, d, t=t).to(dev, dt)
    k = torch.randn(H, T, d, device=dev, dtype=dt)
    v = torch.randn(H, T, d, device=dev, dtype=dt)
    pos = torch.arange(T, device=dev)
    c_k, c_v, out_pos = comp(k, v, pos)
    assert c_k.shape == (H, t, d) and torch.isfinite(c_k).all()
    print(f"[t1 shapes/finite] c_k{tuple(c_k.shape)} OK")

    x = torch.randn(2, 5, 64, device=dev, dtype=dt); p = torch.arange(5, device=dev)
    err = (apply_rope(apply_rope(x, p, 1e6), p, 1e6, inverse=True) - x).abs().max().item()
    assert err < 1e-4; print(f"[t2 rope roundtrip] {err:.2e} OK")

    # independent per-block params
    b0, b1 = comp.blocks[0], comp.blocks[1]
    assert b0.W_o.weight.data_ptr() != b1.W_o.weight.data_ptr()
    print("[t3 per-block params] block0 and block1 are independent OK")

    # full identity init: DIRECTION pass-through at t=T (RMSNorm rescales mag)
    cid = STILLCompactorLayer(H, d, t=T).to(dev, dt)
    attn = cid.first_cross_attn(k, v, pos)
    argmax_hit = (attn.argmax(-1) == torch.arange(T, device=dev)).float().mean().item()
    ck2, cv2, _ = cid(k, v, pos)
    cos_v = F.cosine_similarity(cv2.flatten(0, 1), v.flatten(0, 1), dim=-1).mean().item()
    cos_k_un = F.cosine_similarity(
        apply_rope(ck2, pos, comp.base_theta, inverse=True).flatten(0, 1),
        k.flatten(0, 1), dim=-1).mean().item()
    print(f"[t4 identity init @t=T] argmax-route={argmax_hit:.3f}, "
          f"cos(c_v,V)={cos_v:.3f}, cos(unrot c_k,K)={cos_k_un:.3f} "
          f"(direction near-pass-through; RMSNorm rescales magnitude)")
    assert argmax_hit > 0.9 and cos_v > 0.9

    comp_tr = STILLCompactorLayer(H, d, t=t).to(dev, dt)
    Wro = torch.randn(2 * d, 32, device=dev, dtype=dt)
    teacher = torch.softmax(torch.randn(t, 32, device=dev, dtype=dt), dim=-1)
    opt = torch.optim.Adam(comp_tr.parameters(), lr=1e-3)
    l0 = None
    for step in range(60):
        ck, cv, _ = comp_tr(k, v, pos)
        logp = F.log_softmax(torch.cat([ck, cv], -1).mean(0) @ Wro, dim=-1)
        loss = F.kl_div(logp, teacher, reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            l0 = loss.item()
    print(f"[t5 forward-KL train] {l0:.4f} -> {loss.item():.4f} "
          f"({'REDUCES' if loss.item() < l0 else 'FAILED'})")
    assert loss.item() < l0
    print("\nALL REFERENCE TESTS PASSED (fidelity-corrected)")
