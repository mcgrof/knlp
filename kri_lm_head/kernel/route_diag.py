#!/usr/bin/env python3
"""Diagnostic: can the U_b gap / rho predict the hard (high-fetch) tail at all?
Prints the oracle routing ceiling and the best gK-threshold, so a weak predictor
is distinguished from a calibration bug."""
import argparse
import json
from pathlib import Path

import torch
from certdecode_kernel import certified_decode, shadow_upper_bound

ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--f-bad", type=float, default=0.15)
ap.add_argument("--K", type=int, default=52)
a = ap.parse_args()
dev = "cuda:0"
A = Path(a.artifact)
m = json.loads((A / "meta.json").read_text())
V, C = m["V"], m["C"]; S = V // C
aq = torch.load(A / "aq.pt").to(dev); scale = torch.load(A / "scale.pt").to(dev)
delta = torch.load(A / "delta.pt").to(dev); B = torch.load(A / "B.pt").to(dev)
H = torch.load(A / "H.pt").to(dev); W_U = torch.load(A / "W_U.pt").to(dev)
ae = m["aq_err_norm"]; n = min(a.n, H.shape[0]); K = min(a.K, C)
T_EASY, T_S1, T_DENSE, T_TAIL = 1.06, 0.48, 1.68, 5.0

rho_n, gK, frac = [], [], []
for i in range(n):
    h = H[i]; hf = h.float(); q = hf @ B
    hn = (hf * hf).sum().clamp_min(1e-9).sqrt()
    rho = (hf * hf).sum().sub((q * q).sum()).clamp_min(0).sqrt()
    U = shadow_upper_bound(aq, scale, delta, q, float(rho), ae)
    Ub = U.view(C, S).amax(1)
    Us, _ = Ub.sort(descending=True)
    _, fetched, _ = certified_decode(h, B, aq, scale, delta, W_U, S, ae)
    rho_n.append(float(rho / hn)); gK.append(float(Us[0] - Us[K - 1])); frac.append(fetched / V)
rho_n = torch.tensor(rho_n); gK = torch.tensor(gK); frac = torch.tensor(frac)
hard = frac > a.f_bad
print(f"n={n} hard={int(hard.sum())} ({100*hard.float().mean():.1f}%)")
# correlations
print(f"corr(gK, fetch) = {torch.corrcoef(torch.stack([gK, frac]))[0,1]:.3f}  "
      f"corr(rho, fetch) = {torch.corrcoef(torch.stack([rho_n, frac]))[0,1]:.3f}")
no_route = (T_EASY * (~hard).float() + T_TAIL * hard.float()).mean()
oracle = (T_EASY * (~hard).float() + T_DENSE * hard.float()).mean()
print(f"no-route sim mean = {no_route:.3f} ms   ORACLE (perfect route) = {oracle:.3f} ms")
# best gK threshold (route if gK <= tau): sweep
best = None
for tau in torch.quantile(gK, torch.linspace(0.01, 0.6, 40)):
    route = gK <= tau
    fn = (hard & ~route).float().sum() / n
    fp = (~hard & route).float().sum() / n
    lat = torch.where(route, torch.full_like(gK, T_S1 + T_DENSE),
                      torch.where(hard, torch.full_like(gK, T_TAIL),
                                  torch.full_like(gK, T_EASY))).mean()
    if best is None or lat < best[0]:
        best = (float(lat), float(tau), float(fn), float(fp), float(route.float().mean()))
print(f"best gK-threshold: mean={best[0]:.3f} ms  tau={best[1]:.3f}  "
      f"FN={best[2]*100:.1f}%  FP={best[3]*100:.1f}%  routed={best[4]*100:.0f}%")
