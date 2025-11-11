# MIT-0
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


@torch.no_grad()
def pca_fit(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # SVD doesn't support bfloat16 on CPU, convert to float32 if needed
    orig_dtype = x.dtype
    if x.dtype == torch.bfloat16:
        x = x.float()

    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    U, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    Vk = Vh[:k].T.contiguous()  # [D,k]

    # Convert back to original dtype if needed
    if orig_dtype == torch.bfloat16:
        mu = mu.to(orig_dtype)
        Vk = Vk.to(orig_dtype)

    return mu.contiguous(), Vk


@torch.no_grad()
def pca_proj(x: torch.Tensor, mu: torch.Tensor, Vk: torch.Tensor) -> torch.Tensor:
    return (x - mu) @ Vk


@torch.no_grad()
def pca_back(z: torch.Tensor, mu: torch.Tensor, Vk: torch.Tensor) -> torch.Tensor:
    return z @ Vk.T + mu


class PWLSpline(nn.Module):
    def __init__(self, x_knots: torch.Tensor):
        super().__init__()
        self.register_buffer("xk", x_knots.contiguous())  # [D,K]
        D, K = self.xk.shape
        self.K = K
        self.delta_raw = nn.Parameter(torch.zeros(D, K - 1))
        self.scale_raw = nn.Parameter(torch.zeros(D))
        self.shift = nn.Parameter(torch.zeros(D))
        self.eps = 1e-4

    def _slopes_yk(self):
        xk = self.xk
        seg_dx = xk[:, 1:] - xk[:, :-1]
        slopes = F.softplus(self.delta_raw) + self.eps
        # Normalize slopes to preserve total "mass" across segments
        # Clamp avg to prevent numerical issues
        avg = (slopes * seg_dx).sum(1, keepdim=True) / (
            seg_dx.sum(1, keepdim=True) + 1e-8
        )
        avg = torch.clamp(avg, min=1e-6)  # Prevent division by near-zero
        slopes = slopes / avg
        yk = torch.zeros_like(self.xk)
        yk[:, 1:] = torch.cumsum(slopes * seg_dx, dim=1)
        return slopes, yk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, K = self.xk.shape
        slopes, yk = self._slopes_yk()
        outs = []
        for j in range(D):
            xj = x[..., j].contiguous()
            xkj = self.xk[j].contiguous()
            idx = torch.searchsorted(xkj, xj)
            idx = torch.clamp(idx, 1, K - 1)
            i0 = idx - 1
            x0 = xkj[i0]
            y0 = yk[j, i0]
            m = slopes[j, i0]
            outs.append((y0 + m * (xj - x0)).unsqueeze(-1))
        y = torch.cat(outs, dim=-1)
        scale = F.softplus(self.scale_raw) + 1e-3
        y = y * scale + self.shift
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        D, K = self.xk.shape
        slopes, yk = self._slopes_yk()
        scale = F.softplus(self.scale_raw) + 1e-3
        y_ = (y - self.shift) / (scale + 1e-8)  # Prevent division by zero
        outs = []
        for j in range(D):
            yj = y_[..., j].contiguous()
            ykj = yk[j].contiguous()
            idx = torch.searchsorted(ykj, yj)
            idx = torch.clamp(idx, 1, K - 1)
            i0 = idx - 1
            y0 = ykj[i0]
            m = slopes[j, i0]
            # Clamp slope to prevent division by near-zero
            m_safe = torch.clamp(m, min=1e-6)
            x0 = self.xk[j, i0]
            outs.append((x0 + (yj - y0) / m_safe).unsqueeze(-1))
        return torch.cat(outs, dim=-1)


@torch.no_grad()
def build_spline_from_data(x: torch.Tensor, K: int = 7) -> PWLSpline:
    qs = torch.linspace(0, 1, K, device=x.device, dtype=x.dtype)
    xs, _ = torch.sort(x, dim=0)
    idxs = (qs * (x.shape[0] - 1)).long()
    idxs = torch.clamp(idxs, 0, x.shape[0] - 1)  # Safety clamp
    xk = xs[idxs, :].T.contiguous()  # [D,K]
    return PWLSpline(xk)


class KVGeometryV(nn.Module):
    """
    V-only geometry compressor: y = PCA(g(x)), g = per-dim monotone spline.
    API:
      fit(calib_V), compress(V), decompress(Vc)
    """

    def __init__(self, Hd: int, k_latent: int, knots: int = 7):
        super().__init__()
        self.Hd = Hd
        self.k = k_latent
        self.knots = knots
        self.register_buffer("mu", torch.zeros(1, Hd))
        self.register_buffer("Vk", torch.zeros(Hd, k_latent))
        self.register_buffer("x_mu", torch.zeros(1, Hd))
        self.register_buffer("x_std", torch.ones(1, Hd))
        self.geom: Optional[nn.Module] = None
        self.fitted = False

    def fit(
        self, calib_V: torch.Tensor, epochs: int = 8, lr: float = 2e-3, wd: float = 1e-4
    ):
        x = calib_V
        self.x_mu = x.mean(dim=0, keepdim=True)
        self.x_std = x.std(dim=0, keepdim=True) + 1e-6
        x_n = (x - self.x_mu) / self.x_std
        geom = build_spline_from_data(x_n, K=self.knots).to(x.device, x.dtype)
        geom.train()  # Ensure training mode
        opt = torch.optim.AdamW(geom.parameters(), lr=lr, weight_decay=wd)
        bs = min(4096, max(256, x.shape[0] // 8))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_n),
            batch_size=bs,
            shuffle=True,
            drop_last=True,
        )
        for _ in range(epochs):
            with torch.no_grad():
                z_all = geom(x_n)
                mu_z, Vk = pca_fit(z_all, self.k)
            # Detach mu_z and Vk since they're constants for this epoch
            mu_z = mu_z.detach()
            Vk = Vk.detach()
            for (xb,) in loader:
                # Forward: x -> z -> PCA -> z_compressed -> z_decompressed -> x_reconstructed
                zb = geom(
                    xb
                )  # xb is target (no grad needed), gradients flow through geom params
                zc = (
                    zb - mu_z
                ) @ Vk @ Vk.T + mu_z  # PCA round-trip (constants, no grad)
                xr = geom.inverse(zc)  # Gradients flow through geom params in inverse
                loss = F.mse_loss(xr, xb)  # Compare reconstruction to target
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            z_all = geom(x_n)
            mu_z, Vk = pca_fit(z_all, self.k)
            self.mu.copy_(mu_z)
            self.Vk.copy_(Vk)
        self.geom = geom.eval()
        for p in self.geom.parameters():
            p.requires_grad_(False)
        self.fitted = True

    @torch.no_grad()
    def compress(self, V: torch.Tensor) -> torch.Tensor:
        assert self.fitted
        V2 = V.reshape(-1, self.Hd).contiguous()
        x_n = (V2 - self.x_mu) / self.x_std
        z = self.geom(x_n)
        zc = pca_proj(z, self.mu, self.Vk)
        return zc.reshape(*V.shape[:-1], self.k).contiguous()

    @torch.no_grad()
    def decompress(self, Vc: torch.Tensor) -> torch.Tensor:
        assert self.fitted
        Vc2 = Vc.reshape(-1, self.k).contiguous()
        z = pca_back(Vc2, self.mu, self.Vk)
        x = self.geom.inverse(z)
        V = x * self.x_std + self.x_mu
        return V.reshape(*Vc.shape[:-1], self.Hd).contiguous()
