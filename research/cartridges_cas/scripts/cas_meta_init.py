#!/usr/bin/env python3
"""Meta-initialization for cartridges: audit, fit, select and apply a
shared displacement learned from other documents' training runs.

Every cartridge in the CAS (Cartridges at Scale) line starts from the
same kind of anchor, the KV state of the first p tokens of its document,
and training moves it from there. The question this tool answers is
whether the *direction* training takes from the anchor is shared across
documents, and if so whether adding that shared direction to a new
document's anchor before training saves optimizer steps. The
displacement of one run is D_d(t) = C_d(t) - A_d, the checkpoint at
step t minus the anchor, per layer, KV head and slot.

Three commands:

audit
    For a manifest of documents (anchor + checkpoints per step), report
    where the displacement lives (its size relative to the anchor, the
    slot profile, key versus value energy, the fraction of a document's
    own displacement that a single per-(layer, head) mean vector
    explains, and the direction's stability along the trajectory) and
    whether it is shared across documents: the energy fraction
    f_hat = ||mean_d m_d||^2 / mean_d ||m_d||^2 of the documents'
    content-slot means against a null that rotates each document's
    mean by an independent random orthogonal matrix.

fit
    Fit a displacement phi on a set of fitting documents at a target
    step. Families, nested: bias (one mean vector per layer, head and
    K/V over content slots, plus per-slot means for the template slots
    every document shares), slotwise (a mean per slot index, aligned
    from the start for header and content and from the end for the
    footer), gain (bias plus a scalar gain on the anchor), affine (a
    ridge-regressed matrix on the anchor plus bias). The bias means are
    shrunk toward zero by a positive-part James-Stein factor from the
    across-document scatter, and the key branch is capped at a fraction
    of the anchor's RMS key norm. Leave-one-out over the fitting
    documents scores each family by how much of the held-out document's
    displacement it predicts (R^2 in displacement space) and by the
    naive step-equivalent T * <D_e(T), phi> / ||D_e(T)||^2, so a family
    can be selected without touching the target document. The phi file
    records the documents it was fitted on and ``apply`` refuses a
    target that is among them.

apply
    anchor + alpha * phi -> a cartridge file the trainer can start from
    (CART_INIT=). Modes: meta, flip (minus phi), konly, vonly. Keys are
    handled in the frame phi was fitted in and rotated back per slot.
    The result is rounded to bf16 and the retained fraction of the
    applied change is asserted per (layer, head).

Key frames. Stored keys are rotated by their absolute slot position, so
the frame in which displacements are compared across documents is a
choice: ``raw`` compares stored keys as they are; ``slot`` removes each
slot's own rotation (pre-RoPE keys); ``doc`` removes one rotation by
the document's own cartridge length p, which is the frame a
conversation token at offset j past the cartridge sees every slot in,
because the score between a query at p + j and slot i depends on the
key only through R(-p) k_i. The audit reports all three; fit and apply
carry the frame in the phi file.

Manifest (JSON):
    {"docs": {"patient_01": {"anchor": ".../patient_01_init.pt",
                             "steps": {"60": ".../cache-step60.pt", ...}},
              ...},
     "theta": 1000000.0}

Usage:
    cas_meta_init.py audit --manifest m.json --out audit.json
    cas_meta_init.py fit --manifest m.json --docs patient_01,patient_03 \\
        --step 300 --frame doc --family bias --out phi.pt
    cas_meta_init.py apply --anchor patient_02_init.pt --phi phi.pt \\
        --alpha 1.0 --mode meta --out patient_02_meta.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cas_kv_rope import rotate  # noqa: E402

FRAMES = ("raw", "slot", "doc")
FAMILIES = ("bias", "slotwise", "gain", "affine")
N_HEADER = 2  # slots 1, 2 after the frozen sink: "system", "\n"
N_FOOTER = 2  # the closing <|im_end|>, "\n"
KEY_CAP = 0.25  # |phi_K| <= KEY_CAP * RMS anchor key norm, per (layer, head)
RETENTION_MIN = 0.9
RESOLVABLE_REL = 2.0**-6  # four bf16 ulps of the anchor's norm


# ----------------------------------------------------------------------
# cartridge files, pure torch (no cartridges dependency for analysis)


def _as_t(p):
    return torch.as_tensor(p.data if hasattr(p, "data") else p).detach()


def read_cart(path):
    """Stacked float32 K, V of shape [L, H, P, D] (frozen slots first) and
    the frozen count, read off the token axis."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tk = torch.stack([_as_t(p)[0] for p in ck["trainable_keys"]]).float()
    tv = torch.stack([_as_t(p)[0] for p in ck["trainable_values"]]).float()
    fk = ck.get("frozen_keys") or []
    if len(fk):
        fk = torch.stack([_as_t(p)[0] for p in fk]).float()
        fv = torch.stack([_as_t(p)[0] for p in ck["frozen_values"]]).float()
        nfrozen = fk.shape[2]
        return torch.cat([fk, tk], 2), torch.cat([fv, tv], 2), nfrozen
    return tk, tv, 0


def write_cart(K, V, nfrozen, path):
    """The library's dict layout, bf16, one [1, H, T, D] tensor per layer."""
    K = K.to(torch.bfloat16)
    V = V.to(torch.bfloat16)
    out = {
        "trainable_keys": [
            K[l, :, nfrozen:][None].contiguous() for l in range(K.shape[0])
        ],
        "trainable_values": [
            V[l, :, nfrozen:][None].contiguous() for l in range(V.shape[0])
        ],
        "frozen_keys": (
            [K[l, :, :nfrozen][None].contiguous() for l in range(K.shape[0])]
            if nfrozen
            else []
        ),
        "frozen_values": (
            [V[l, :, :nfrozen][None].contiguous() for l in range(V.shape[0])]
            if nfrozen
            else []
        ),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)


def to_frame(K, frame, theta):
    """Stored keys [L, H, P, D] -> the comparison frame."""
    P = K.shape[2]
    if frame == "raw":
        return K
    if frame == "slot":
        return rotate(K, torch.arange(P), theta, inverse=True)
    if frame == "doc":
        return rotate(K, torch.full((P,), P), theta, inverse=True)
    raise ValueError(frame)


def from_frame(K, frame, theta):
    """Inverse of ``to_frame`` for a tensor with P slots."""
    P = K.shape[2]
    if frame == "raw":
        return K
    if frame == "slot":
        return rotate(K, torch.arange(P), theta, inverse=False)
    if frame == "doc":
        return rotate(K, torch.full((P,), P), theta, inverse=False)
    raise ValueError(frame)


def slot_classes(P, nfrozen):
    """(header, content, footer) slot index tensors for a P-slot cart."""
    h = torch.arange(nfrozen, nfrozen + N_HEADER)
    c = torch.arange(nfrozen + N_HEADER, P - N_FOOTER)
    f = torch.arange(P - N_FOOTER, P)
    return h, c, f


# ----------------------------------------------------------------------
# displacements


class Doc:
    def __init__(self, name, spec, theta, steps=None):
        self.name = name
        self.theta = theta
        self.AK, self.AV, self.nfrozen = read_cart(spec["anchor"])
        self.P = self.AK.shape[2]
        self.header, self.content, self.footer = slot_classes(self.P, self.nfrozen)
        self.step_paths = {int(k): v for k, v in spec["steps"].items()}
        if steps is not None:
            self.step_paths = {t: p for t, p in self.step_paths.items() if t in steps}
        self._cache = {}

    def steps(self):
        return sorted(self.step_paths)

    def delta(self, t, frame):
        """(dK, dV) at step t, dK in ``frame``, both [L, H, P, D]."""
        key = (t, frame)
        if key not in self._cache:
            CK, CV, nf = read_cart(self.step_paths[t])
            assert nf == self.nfrozen and CK.shape == self.AK.shape, (
                self.name,
                t,
            )
            # the frozen sink must not have moved
            fz = (CK[:, :, : self.nfrozen] - self.AK[:, :, : self.nfrozen]).abs().max()
            assert fz.item() == 0.0, f"{self.name} step {t}: frozen slot moved {fz}"
            dK = to_frame(CK, frame, self.theta) - to_frame(self.AK, frame, self.theta)
            self._cache[key] = (dK, CV - self.AV)
        return self._cache[key]

    def anchor_in(self, frame):
        return to_frame(self.AK, frame, self.theta)


def load_docs(manifest, names=None, steps=None):
    m = json.load(open(manifest))
    theta = float(m.get("theta", 1e6))
    docs = {}
    for name, spec in m["docs"].items():
        if names is None or name in names:
            docs[name] = Doc(name, spec, theta, steps)
    if names is not None:
        missing = set(names) - set(docs)
        assert not missing, f"docs not in manifest: {sorted(missing)}"
    return docs, theta


def haar_rotations(n, D, gen):
    """n Haar-distributed orthogonal D x D matrices."""
    A = torch.randn(n, D, D, generator=gen)
    Q, R = torch.linalg.qr(A)
    s = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    return Q * s[:, None, :]


def shared_fraction(M):
    """M [G, N, D]: per group, ||mean_d M||^2 / mean_d ||M_d||^2."""
    num = M.mean(1).pow(2).sum(-1)
    den = M.pow(2).sum(-1).mean(1).clamp_min(1e-30)
    return num / den


def shared_null(M, draws, gen):
    """Null of ``shared_fraction`` from independent rotations per doc;
    returns [draws, G]."""
    G, N, D = M.shape
    R = haar_rotations(draws * N, D, gen).view(draws, N, D, D)
    # rotated[s, g, n, :] = R[s, n] @ M[g, n]
    rot = torch.einsum("snij,gnj->sgni", R, M)
    return shared_fraction(rot.reshape(draws * G, N, D)).view(draws, G)


# ----------------------------------------------------------------------
# audit


def cmd_audit(a):
    docs, theta = load_docs(a.manifest)
    names = sorted(docs)
    gen = torch.Generator().manual_seed(a.seed)
    report = {"manifest": os.path.abspath(a.manifest), "docs": {}, "shared": {}}
    steps_all = sorted(set.intersection(*[set(d.steps()) for d in docs.values()]))
    print(f"[audit] docs {names}, common steps {steps_all}", flush=True)

    # the template slots every document shares: header keys (slot
    # frame) and values must agree bitwise across documents, the footer
    # pair carries the same tokens but a document-dependent state
    ref = docs[names[0]]
    hd_k = (
        max(
            (
                docs[n].anchor_in("slot")[:, :, docs[n].header]
                - ref.anchor_in("slot")[:, :, ref.header]
            )
            .abs()
            .max()
            .item()
            for n in names[1:]
        )
        if len(names) > 1
        else 0.0
    )
    hd_v = (
        max(
            (docs[n].AV[:, :, docs[n].header] - ref.AV[:, :, ref.header])
            .abs()
            .max()
            .item()
            for n in names[1:]
        )
        if len(names) > 1
        else 0.0
    )
    ft_v = (
        max(
            (docs[n].AV[:, :, docs[n].footer] - ref.AV[:, :, ref.footer])
            .abs()
            .max()
            .item()
            for n in names[1:]
        )
        if len(names) > 1
        else 0.0
    )
    report["template_slots"] = {
        "header_K_maxdiff": hd_k,
        "header_V_maxdiff": hd_v,
        "footer_V_maxdiff": ft_v,
    }
    print(
        f"[audit] template slots: header K max diff {hd_k:.3g} V {hd_v:.3g}; footer V max diff {ft_v:.3g}",
        flush=True,
    )

    # per-document geometry
    for name in names:
        d = docs[name]
        rep = {"P": d.P, "nfrozen": d.nfrozen, "steps": {}}
        aK = d.AK[:, :, d.content]
        rms_K = aK.pow(2).sum(-1).mean(-1).sqrt()  # [L,H]
        rms_V = d.AV[:, :, d.content].pow(2).sum(-1).mean(-1).sqrt()
        prev = None
        for t in d.steps():
            dK, dV = d.delta(t, "slot")
            r = {}
            for nm, dX, rms in (("K", dK, rms_K), ("V", dV, rms_V)):
                cont = dX[:, :, d.content]  # [L,H,C,D]
                e_tot = dX.pow(2).sum().item()
                e_cont = cont.pow(2).sum().item()
                slot_e = dX.pow(2).sum((0, 1, 3))  # [P]
                m = cont.mean(2)  # [L,H,D]
                f_slot = m.pow(2).sum(-1) / cont.pow(2).sum(-1).mean(-1).clamp_min(
                    1e-30
                )
                r[nm] = {
                    "energy": e_tot,
                    "rel_rms": (cont.pow(2).sum(-1).mean(-1).sqrt() / rms)
                    .mean()
                    .item(),
                    "frac_header": slot_e[d.header].sum().item() / max(e_tot, 1e-30),
                    "frac_footer": slot_e[d.footer].sum().item() / max(e_tot, 1e-30),
                    "frac_content": e_cont / max(e_tot, 1e-30),
                    "frac_last10": slot_e[-10:].sum().item() / max(e_tot, 1e-30),
                    "frac_first10": slot_e[d.nfrozen : d.nfrozen + 10].sum().item()
                    / max(e_tot, 1e-30),
                    "mean_explained": f_slot.mean().item(),
                    "mean_explained_by_layer": f_slot.mean(1).tolist(),
                }
            # fast versus slow key pairs, pre-RoPE frame
            half = dK.shape[-1] // 2
            fast = dK[..., :32].pow(2).sum() + dK[..., half : half + 32].pow(2).sum()
            r["K"]["frac_fast_pairs"] = (fast / dK.pow(2).sum().clamp_min(1e-30)).item()
            r["KV_energy_ratio"] = r["K"]["energy"] / max(r["V"]["energy"], 1e-30)
            if prev is not None:
                pK, pV = prev
                r["cos_prev_step"] = {
                    "K": torch.nn.functional.cosine_similarity(
                        dK.flatten(), pK.flatten(), dim=0
                    ).item(),
                    "V": torch.nn.functional.cosine_similarity(
                        dV.flatten(), pV.flatten(), dim=0
                    ).item(),
                }
            prev = (dK, dV)
            rep["steps"][str(t)] = r
            print(
                f"[audit] {name} step {t}: relRMS K {r['K']['rel_rms']:.4f} "
                f"V {r['V']['rel_rms']:.4f}  explained-by-mean K "
                f"{r['K']['mean_explained']:.3f} V {r['V']['mean_explained']:.3f}  "
                f"last10 K {r['K']['frac_last10']:.3f} V {r['V']['frac_last10']:.3f}",
                flush=True,
            )
        report["docs"][name] = rep
        d._cache.clear()

    # across-document sharing, per frame and step
    for frame in FRAMES:
        report["shared"][frame] = {}
        for t in steps_all:
            out = {}
            for nm in ("K", "V"):
                M = []
                for name in names:
                    d = docs[name]
                    dK, dV = d.delta(t, frame)
                    dX = dK if nm == "K" else dV
                    M.append(dX[:, :, d.content].mean(2))  # [L,H,D]
                M = torch.stack(M, 2)  # [L,H,N,D]
                L, H, N, D = M.shape
                Mg = M.reshape(L * H, N, D)
                f = shared_fraction(Mg)
                null = shared_null(Mg, a.null_draws, gen)
                q99 = torch.quantile(null, 0.99, dim=0)
                # pairwise cosine between documents' means
                Mn = torch.nn.functional.normalize(Mg, dim=-1)
                cos = torch.einsum("gnd,gmd->gnm", Mn, Mn)
                iu = torch.triu_indices(N, N, 1)
                pair = cos[:, iu[0], iu[1]].mean(-1)
                out[nm] = {
                    "f_hat_mean": f.mean().item(),
                    "f_hat_by_layer": f.view(L, H).mean(1).tolist(),
                    "null99_mean": q99.mean().item(),
                    "frac_groups_above_null99": (f > q99).float().mean().item(),
                    "frac_layers_majority_above": (
                        (f > q99).view(L, H).float().mean(1) > 0.5
                    )
                    .float()
                    .mean()
                    .item(),
                    "pairwise_cos_mean": pair.mean().item(),
                    "expected_null_f": 1.0 / N,
                }
            report["shared"][frame][str(t)] = out
            print(
                f"[audit] shared frame={frame} step {t}: K f_hat {out['K']['f_hat_mean']:.3f} "
                f"(null99 {out['K']['null99_mean']:.3f}, groups above "
                f"{out['K']['frac_groups_above_null99']:.2f}, layers majority "
                f"{out['K']['frac_layers_majority_above']:.2f})  V f_hat "
                f"{out['V']['f_hat_mean']:.3f} (null99 {out['V']['null99_mean']:.3f}, "
                f"groups above {out['V']['frac_groups_above_null99']:.2f}, layers "
                f"majority {out['V']['frac_layers_majority_above']:.2f})",
                flush=True,
            )
        for d in docs.values():
            d._cache.clear()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"AUDIT_DONE {a.out}", flush=True)


# ----------------------------------------------------------------------
# fitting


def js_shrink(M, pool=None):
    """M [G, N, D] -> (mean [G, D], alpha [G]) with a positive-part
    James-Stein factor alpha = max(0, 1 - noise / ||s||^2): s is the
    across-document mean, noise = (unbiased mean squared residual over
    the N documents) / N is the variance of that mean. With ``pool`` =
    number of consecutive groups that share one alpha (e.g. the heads of
    a layer), ||s||^2 and noise are summed over the pool first, so N = 4
    documents estimate 72 factors rather than 576."""
    N = M.shape[1]
    s = M.mean(1)
    G = M.shape[0]
    if N < 2:
        return s, torch.ones(G)
    scatter = (M - s[:, None]).pow(2).sum(-1).sum(1) / (N - 1)
    noise = scatter / N
    sig = s.pow(2).sum(-1)
    if pool and pool > 1:
        assert G % pool == 0, (G, pool)
        noise = noise.view(-1, pool).sum(1, keepdim=True).expand(-1, pool).reshape(G)
        sig = sig.view(-1, pool).sum(1, keepdim=True).expand(-1, pool).reshape(G)
    alpha = (1 - noise / sig.clamp_min(1e-30)).clamp(0, 1)
    return s, alpha


def _pool(level, L, H, extra=1):
    """Groups per shrinkage factor for [L*H*extra] flattened groups."""
    return {"head": 1, "layer": H * extra, "global": L * H * extra}[level]


def fit_family(docs, names, t, frame, family, ridge, shrink="layer"):
    """Return a phi dict describing the displacement model."""
    L, H, D = (
        docs[names[0]].AK.shape[0],
        docs[names[0]].AK.shape[1],
        docs[names[0]].AK.shape[3],
    )
    phi = {
        "family": family,
        "frame": frame,
        "step": t,
        "docs": list(names),
        "L": L,
        "H": H,
        "D": D,
        "n_header": N_HEADER,
        "n_footer": N_FOOTER,
        "shrink": shrink,
    }
    Pmin = min(docs[n].P for n in names)
    nfz = docs[names[0]].nfrozen
    assert all(docs[n].nfrozen == nfz for n in names)
    phi["nfrozen"] = nfz
    for nm in ("K", "V"):
        cont, head, foot, slotw = [], [], [], []
        A_all, D_all = [], []
        for n in names:
            d = docs[n]
            dK, dV = d.delta(t, frame)
            dX = dK if nm == "K" else dV
            cont.append(dX[:, :, d.content].mean(2))  # [L,H,D]
            head.append(dX[:, :, d.header])  # [L,H,2,D]
            foot.append(dX[:, :, d.footer])
            slotw.append(dX[:, :, nfz + N_HEADER : Pmin - N_FOOTER])
            A = d.anchor_in(frame) if nm == "K" else d.AV
            A_all.append(A[:, :, d.content])
            D_all.append(dX[:, :, d.content])
        M = torch.stack(cont, 2).reshape(L * H, len(names), D)
        s, alpha = js_shrink(M, _pool(shrink, L, H))
        _, alpha_head = js_shrink(M)
        phi[f"{nm}_content"] = (alpha[:, None] * s).view(L, H, D)
        phi[f"{nm}_alpha"] = alpha.view(L, H)
        phi[f"{nm}_alpha_head"] = alpha_head.view(L, H)  # diagnostic only
        phi[f"{nm}_content_raw"] = s.view(L, H, D)
        for tag, lst in (("header", head), ("footer", foot)):
            X = torch.stack(lst, 0)  # [N,L,H,2,D]
            S = X.shape[3]
            Mx = X.permute(1, 2, 3, 0, 4).reshape(L * H * S, len(names), D)
            sx, ax = js_shrink(Mx, _pool(shrink, L, H, S))
            phi[f"{nm}_{tag}"] = (ax[:, None] * sx).view(L, H, S, D)
        if family == "slotwise":
            X = torch.stack(slotw, 0)  # [N,L,H,S,D]
            S = X.shape[3]
            Mx = X.permute(1, 2, 3, 0, 4).reshape(L * H * S, len(names), D)
            # one factor per slot across heads of a layer, or per layer
            sx, ax = js_shrink(
                Mx,
                (
                    _pool(shrink, L, H, S)
                    if shrink == "global"
                    else (H * S if shrink == "layer" else 1)
                ),
            )
            phi[f"{nm}_slotwise"] = (ax[:, None] * sx).view(L, H, S, D)
        if family in ("gain", "affine"):
            A = torch.cat(A_all, 2)  # [L,H,C_all,D]
            Dl = torch.cat(D_all, 2)
            Am, Dm = A.mean(2, keepdim=True), Dl.mean(2, keepdim=True)
            Ac, Dc = A - Am, Dl - Dm
            if family == "gain":
                g = (Ac * Dc).sum((2, 3)) / Ac.pow(2).sum((2, 3)).clamp_min(1e-30)
                phi[f"{nm}_gain"] = g  # [L,H]
                phi[f"{nm}_gain_bias"] = (Dm - g[..., None, None] * Am)[:, :, 0]
            else:
                # ridge: W = (Ac^T Ac + lam I)^-1 Ac^T Dc, per (l,h)
                G = torch.einsum("lhcd,lhce->lhde", Ac, Ac)
                lam = ridge * torch.diagonal(G, dim1=-2, dim2=-1).mean(-1)
                G = G + lam[..., None, None] * torch.eye(D)
                B = torch.einsum("lhcd,lhce->lhde", Ac, Dc)
                W = torch.linalg.solve(G, B)  # [L,H,D,D]; delta ~ A @ W
                phi[f"{nm}_affine_W"] = W
                phi[f"{nm}_affine_bias"] = (
                    Dm - torch.einsum("lhcd,lhde->lhce", Am, W)
                )[:, :, 0]
    return phi


def predict_delta(phi, AK_frame, AV, P, nfrozen):
    """The displacement phi prescribes for an anchor with P slots, in
    phi's frame: (dK, dV) each [L, H, P, D]."""
    out = {}
    for nm, A in (("K", AK_frame), ("V", AV)):
        L, H, _, D = A.shape
        dX = torch.zeros(L, H, P, D)
        header, content, footer = slot_classes(P, nfrozen)
        fam = phi["family"]
        if fam == "slotwise":
            S = phi[f"{nm}_slotwise"].shape[2]
            n = min(S, len(content))
            dX[:, :, content[:n]] = phi[f"{nm}_slotwise"][:, :, :n]
            if n < len(content):
                dX[:, :, content[n:]] = phi[f"{nm}_content"][:, :, None]
        elif fam == "gain":
            g = phi[f"{nm}_gain"][..., None, None]
            dX[:, :, content] = (
                g * A[:, :, content] + phi[f"{nm}_gain_bias"][:, :, None]
            )
        elif fam == "affine":
            W, b = phi[f"{nm}_affine_W"], phi[f"{nm}_affine_bias"]
            dX[:, :, content] = (
                torch.einsum("lhcd,lhde->lhce", A[:, :, content], W) + b[:, :, None]
            )
        else:
            dX[:, :, content] = phi[f"{nm}_content"][:, :, None]
        dX[:, :, header] = phi[f"{nm}_header"]
        dX[:, :, footer] = phi[f"{nm}_footer"]
        out[nm] = dX
    return out["K"], out["V"]


def cap_keys(dK, AK_frame, nfrozen, cap=KEY_CAP):
    """Scale the key displacement down per (layer, head) so its RMS over
    slots stays within ``cap`` of the anchor's RMS key norm."""
    rms_a = AK_frame[:, :, nfrozen:].pow(2).sum(-1).mean(-1).sqrt()  # [L,H]
    rms_d = dK[:, :, nfrozen:].pow(2).sum(-1).mean(-1).sqrt()
    scale = (cap * rms_a / rms_d.clamp_min(1e-30)).clamp(max=1.0)
    return dK * scale[..., None, None], scale


def score_prediction(dK_true, dV_true, dK_pred, dV_pred, content):
    """R^2 in displacement space and the projected fraction, over
    content slots, K and V pooled and separately."""
    res = {}
    tot_num = tot_den = tot_proj = 0.0
    for nm, tr, pr in (("K", dK_true, dK_pred), ("V", dV_true, dV_pred)):
        tr = tr[:, :, content].flatten()
        pr = pr[:, :, content].flatten()
        den = tr.pow(2).sum().item()
        num = (tr - pr).pow(2).sum().item()
        proj = (tr * pr).sum().item()
        res[nm] = {"r2": 1 - num / max(den, 1e-30), "proj_frac": proj / max(den, 1e-30)}
        tot_num += num
        tot_den += den
        tot_proj += proj
    res["all"] = {
        "r2": 1 - tot_num / max(tot_den, 1e-30),
        "proj_frac": tot_proj / max(tot_den, 1e-30),
    }
    return res


def step_equivalent(doc, frame, dK_p, dV_p):
    """Smallest step k at which the document's own displacement,
    projected on the unit vector of the prescribed displacement, reaches
    that displacement's norm; interpolated between saved steps. Also the
    norm ratios against the earliest and the last saved displacement."""
    u = torch.cat(
        [dK_p[:, :, doc.content].flatten(), dV_p[:, :, doc.content].flatten()]
    )
    n_phi = u.norm().item()
    if n_phi == 0:
        return {
            "step_equivalent": 0.0,
            "phi_norm": 0.0,
            "phi_over_first_step_norm": 0.0,
            "phi_over_last_norm": 0.0,
        }
    u = u / n_phi
    proj, norms = {0: 0.0}, {}
    for t in doc.steps():
        dK, dV = doc.delta(t, frame)
        v = torch.cat(
            [dK[:, :, doc.content].flatten(), dV[:, :, doc.content].flatten()]
        )
        proj[t] = (v * u).sum().item()
        norms[t] = v.norm().item()
    ts = sorted(proj)
    k = None
    for i in range(1, len(ts)):
        if proj[ts[i]] >= n_phi:
            a, b = ts[i - 1], ts[i]
            fa, fb = proj[a], proj[b]
            k = a + (b - a) * (n_phi - fa) / max(fb - fa, 1e-30)
            break
    first, last = doc.steps()[0], doc.steps()[-1]
    return {
        "step_equivalent": k if k is not None else float("inf"),
        "phi_norm": n_phi,
        "phi_over_first_step_norm": n_phi / max(norms[first], 1e-30),
        "first_saved_step": first,
        "phi_over_last_norm": n_phi / max(norms[last], 1e-30),
        "last_saved_step": last,
        "projection_by_step": {str(t): proj[t] for t in ts},
    }


def loo_scores(docs, names, step, frame, fam, ridge, shrink):
    """Leave-one-out over ``names``: fit on the others, score the held
    document's own displacement at ``step``."""
    per = {}
    for held in names:
        fit_names = [n for n in names if n != held]
        phi = fit_family(docs, fit_names, step, frame, fam, ridge, shrink)
        d = docs[held]
        dK_t, dV_t = d.delta(step, frame)
        dK_p, dV_p = predict_delta(phi, d.anchor_in(frame), d.AV, d.P, d.nfrozen)
        dK_p, _ = cap_keys(dK_p, d.anchor_in(frame), d.nfrozen)
        sc = score_prediction(dK_t, dV_t, dK_p, dV_p, d.content)
        sc["naive_step_equivalent"] = step * sc["all"]["proj_frac"]
        sc.update(step_equivalent(d, frame, dK_p, dV_p))
        per[held] = sc
        for dd in docs.values():
            dd._cache.clear()
    reached = [
        v["step_equivalent"]
        for v in per.values()
        if v["step_equivalent"] != float("inf")
    ]
    out = {
        "per_doc": per,
        "mean_r2": sum(v["all"]["r2"] for v in per.values()) / len(per),
        "mean_r2_K": sum(v["K"]["r2"] for v in per.values()) / len(per),
        "mean_r2_V": sum(v["V"]["r2"] for v in per.values()) / len(per),
        "n_docs_positive_r2": sum(v["all"]["r2"] > 0 for v in per.values()),
        "n_step_equivalent_reached": len(reached),
        "mean_step_equivalent_reached": (
            (sum(reached) / len(reached)) if reached else None
        ),
        "mean_naive_step_equivalent": sum(
            v["naive_step_equivalent"] for v in per.values()
        )
        / len(per),
    }
    se = out["mean_step_equivalent_reached"]
    print(
        f"[fit] LOO family={fam} frame={frame} step={step}: mean R2 "
        f"{out['mean_r2']:.4f} (K {out['mean_r2_K']:.4f}, V {out['mean_r2_V']:.4f}), "
        f"positive on {out['n_docs_positive_r2']}/{len(per)}, step-equivalent reached "
        f"{len(reached)}/{len(per)}"
        + (f" mean {se:.0f}" if se is not None else "")
        + f", naive mean {out['mean_naive_step_equivalent']:.0f}; per doc "
        + ", ".join(
            f"{k}: R2 {v['all']['r2']:.3f} steq "
            + (
                f"{v['step_equivalent']:.0f}"
                if v["step_equivalent"] != float("inf")
                else f">{v['last_saved_step']}"
            )
            + f" phi/first {v['phi_over_first_step_norm']:.2f}"
            for k, v in per.items()
        ),
        flush=True,
    )
    return out


def cmd_fit(a):
    names = [n for n in a.docs.split(",") if n]
    docs, theta = load_docs(a.manifest, names)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    families = FAMILIES if a.family == "all" else (a.family,)
    frames = FRAMES if a.frame == "all" else (a.frame,)
    loo = {}
    frame = a.frame
    if a.loo and len(names) >= 3:
        # frame first, on the simplest family; then the families in it
        if len(frames) > 1:
            for fr in frames:
                loo[f"bias/{fr}"] = loo_scores(
                    docs, names, a.step, fr, "bias", a.ridge, a.shrink
                )
            frame = max(frames, key=lambda fr: loo[f"bias/{fr}"]["mean_r2"])
            print(
                f"[fit] selected frame {frame} by LOO mean R2 (bias family)", flush=True
            )
        for fam in families:
            key = f"{fam}/{frame}"
            if key not in loo:
                loo[key] = loo_scores(
                    docs, names, a.step, frame, fam, a.ridge, a.shrink
                )
    elif len(frames) > 1:
        raise SystemExit("--frame all needs --loo and at least 3 docs")
    chosen = a.family
    if a.family == "all":
        if loo:
            # nested: the simplest family within the margin of the best
            best = max(loo[f"{f}/{frame}"]["mean_r2"] for f in families)
            chosen = next(
                f
                for f in families
                if loo[f"{f}/{frame}"]["mean_r2"] >= best - a.select_margin
            )
        else:
            chosen = "bias"
        print(f"[fit] selected family {chosen} (margin {a.select_margin})", flush=True)
    phi = fit_family(docs, names, a.step, frame, chosen, a.ridge, a.shrink)
    phi["loo"] = loo
    phi["theta"] = theta
    phi["manifest"] = os.path.abspath(a.manifest)
    torch.save(phi, a.out)
    aK, aV = phi["K_alpha"].mean().item(), phi["V_alpha"].mean().item()
    print(
        f"FIT_DONE family={chosen} frame={frame} step={a.step} docs={names} "
        f"shrink={a.shrink} shrink_mean K {aK:.3f} V {aV:.3f} -> {a.out}",
        flush=True,
    )


# ----------------------------------------------------------------------
# apply


def cmd_apply(a):
    phi = torch.load(a.phi, map_location="cpu", weights_only=False)
    target = a.target or Path(a.anchor).stem.replace("_init", "")
    assert (
        target not in phi["docs"]
    ), f"refusing: target {target} is among the fitting docs {phi['docs']}"
    AK, AV, nfrozen = read_cart(a.anchor)
    assert nfrozen == phi["nfrozen"], (nfrozen, phi["nfrozen"])
    theta = phi.get("theta", 1e6)
    frame = phi["frame"]
    AKf = to_frame(AK, frame, theta)
    P = AK.shape[2]
    dK, dV = predict_delta(phi, AKf, AV, P, nfrozen)
    dK, kscale = cap_keys(dK, AKf, nfrozen)
    match = None
    if a.match_phi:
        # a single-donor (or any other) phi rescaled per (layer, head) to
        # the norm the reference phi applies to this anchor, so the
        # control differs from the meta arm in direction only
        ref = torch.load(a.match_phi, map_location="cpu", weights_only=False)
        assert ref["frame"] == frame, (ref["frame"], frame)
        rK, rV = predict_delta(ref, AKf, AV, P, nfrozen)
        rK, _ = cap_keys(rK, AKf, nfrozen)
        match = {}
        for nm in ("K", "V"):
            X, R = (dK, rK) if nm == "K" else (dV, rV)
            nx = X[:, :, nfrozen:].pow(2).sum((2, 3)).sqrt()
            nr = R[:, :, nfrozen:].pow(2).sum((2, 3)).sqrt()
            s = nr / nx.clamp_min(1e-30)
            X.mul_(s[..., None, None])
            nx2 = X[:, :, nfrozen:].pow(2).sum((2, 3)).sqrt()
            dev = ((nx2 - nr).abs() / nr.clamp_min(1e-30))[nr > 0]
            assert dev.max().item() < 0.05, f"{nm} norm match off by {dev.max():.3f}"
            match[nm] = {
                "scale_mean": s.mean().item(),
                "scale_min": s.min().item(),
                "scale_max": s.max().item(),
            }
    if a.mode == "flip":
        dK, dV = -dK, -dV
    elif a.mode == "konly":
        dV = torch.zeros_like(dV)
    elif a.mode == "vonly":
        dK = torch.zeros_like(dK)
    elif a.mode != "meta":
        raise ValueError(a.mode)
    dK, dV = a.alpha * dK, a.alpha * dV
    # back to stored keys: the frame rotation is linear, so rotate the
    # displacement alone and add
    dK_stored = from_frame(dK, frame, theta)
    K_new = AK + dK_stored
    V_new = AV + dV
    K_new[:, :, :nfrozen] = AK[:, :, :nfrozen]
    V_new[:, :, :nfrozen] = AV[:, :, :nfrozen]
    # bf16 retention per (layer, head): what survives rounding
    K_b, V_b = K_new.to(torch.bfloat16).float(), V_new.to(torch.bfloat16).float()
    A_bK, A_bV = AK.to(torch.bfloat16).float(), AV.to(torch.bfloat16).float()
    rep = {}
    for nm, X_b, A_b, dX in (("K", K_b, A_bK, dK_stored), ("V", V_b, A_bV, dV)):
        applied = (X_b - A_b)[:, :, nfrozen:]
        want = dX[:, :, nfrozen:]
        num = (applied * want).sum((2, 3))
        den = want.pow(2).sum((2, 3))
        ret = num / den.clamp_min(1e-30)
        rel = (
            want.pow(2).sum((2, 3)).sqrt()
            / A_b[:, :, nfrozen:].pow(2).sum((2, 3)).sqrt()
        )
        # a head whose wanted change is below bf16 resolution relative to
        # its anchor cannot retain it and was shrunk to nothing anyway;
        # judge retention on the heads the change is resolvable in, and
        # on the energy-weighted whole
        used = (den > 0) & (rel > RESOLVABLE_REL)
        rep[nm] = {
            "retention_min": ret[used].min().item() if used.any() else 1.0,
            "retention_mean": ret[used].mean().item() if used.any() else 1.0,
            "retention_energy": (
                (num.sum() / den.sum()).item() if den.sum() > 0 else 1.0
            ),
            "heads_judged": int(used.sum().item()),
            "rel_norm_mean": rel.mean().item(),
            "rel_norm_max": rel.max().item(),
        }
    write_cart(K_new, V_new, nfrozen, a.out)
    meta = {
        "anchor": os.path.abspath(a.anchor),
        "phi": os.path.abspath(a.phi),
        "phi_docs": phi["docs"],
        "family": phi["family"],
        "frame": frame,
        "step": phi["step"],
        "mode": a.mode,
        "alpha": a.alpha,
        "key_cap_scale_min": kscale.min().item(),
        "key_cap_scale_mean": kscale.mean().item(),
        "retention": rep,
        "target": target,
        "norm_matched_to": os.path.abspath(a.match_phi) if a.match_phi else None,
        "norm_match": match,
    }
    json.dump(meta, open(str(a.out) + ".json", "w"), indent=1)
    ok = a.alpha == 0 or all(
        v["retention_min"] >= RETENTION_MIN and v["retention_energy"] >= RETENTION_MIN
        for v in rep.values()
    )
    print(
        f"APPLY_DONE mode={a.mode} alpha={a.alpha} family={phi['family']} "
        f"frame={frame} K rel {rep['K']['rel_norm_mean']:.4f} retention "
        f"{rep['K']['retention_min']:.3f}/{rep['K']['retention_energy']:.3f}  "
        f"V rel {rep['V']['rel_norm_mean']:.4f} retention "
        f"{rep['V']['retention_min']:.3f}/{rep['V']['retention_energy']:.3f} "
        f"(min over {rep['K']['heads_judged']}+{rep['V']['heads_judged']} "
        f"resolvable heads / energy-weighted)  keycap min {kscale.min():.3f} "
        f"-> {a.out} {'OK' if ok else 'RETENTION_LOW'}",
        flush=True,
    )
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--null-draws", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("fit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--docs", required=True, help="comma list of fitting docs")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--frame", choices=FRAMES + ("all",), default="doc")
    p.add_argument("--family", choices=FAMILIES + ("all",), default="bias")
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--shrink", choices=("head", "layer", "global"), default="layer")
    p.add_argument(
        "--select-margin",
        type=float,
        default=0.01,
        help="with --family all: pick the simplest family within this R2 of the best",
    )
    p.add_argument("--loo", action="store_true", help="LOO family scores")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("apply")
    p.add_argument("--anchor", required=True)
    p.add_argument("--phi", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--mode", choices=("meta", "flip", "konly", "vonly"), default="meta")
    p.add_argument("--target", default="", help="target doc name (default from anchor)")
    p.add_argument(
        "--match-phi",
        default="",
        help="rescale the applied displacement per (layer, head) to the norm this reference phi applies",
    )
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_apply)
    a = ap.parse_args()
    rc = a.fn(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
