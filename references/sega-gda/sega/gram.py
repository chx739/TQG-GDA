"""Deformation Gram matrix G_r (Sec. 3.8.4).

[G_r]_{lm} = int tr[ g^{-1} V_l g^{-1} V_m ] d mu_bar_r,   mu_bar_r = 1/2 mu_0 + 1/2 mu_r^path.

Because V_l(z) = T^B_l + sum_k alpha_k(z) T^E_{kl}, in symmetric coordinates
vec_s(delta g(z)) = T(z) Delta with T(z) = [Q_B, gamma_1(z) I_s, ..., gamma_{K-1}(z) I_s],
gamma(z) = U^T alpha(z). With psi(z) = (1, gamma(z)) and the s x s matrix
Omega(g)_{mn} = tr(g^{-1} S_m g^{-1} S_n) we get

    G = blocks of  sum_nodes chi * (psi psi^T) (x) Omega(g(z)),

and g(z) depends on z only through alpha(z). Nodes are grouped by alpha on a
grid of resolution 1/res: the polynomial moments psi psi^T are accumulated
exactly per cell, and Omega is evaluated at the cell's weighted mean alpha.
This is the only approximation (it is exact whenever g is constant on a cell,
e.g. at b = 0), and it makes the integral over millions of path nodes tractable.
"""
from __future__ import annotations

import numpy as np
import torch


class GramAccumulator:
    def __init__(self, field, res: int, dtype=torch.float64, device="cpu"):
        self.field = field
        self.K = field.K
        self.res = int(res)
        self.dtype, self.device = dtype, device
        self.U = field.U_t
        self.npsi = 1 + field.KE                                   # psi = (1, gamma_1..gamma_KE)
        self.cells, self.wsum, self.wa, self.mom = [], [], [], []
        self.total_weight = 0.0

    def _psi(self, alpha: torch.Tensor) -> torch.Tensor:
        gamma = alpha @ self.U                                     # [M, K-1]
        return torch.cat([torch.ones_like(alpha[:, :1]), gamma], dim=1)

    @torch.no_grad()
    def add(self, alpha: torch.Tensor, weight: torch.Tensor) -> None:
        alpha = alpha.reshape(-1, self.K).to(self.dtype)
        weight = weight.reshape(-1).to(self.dtype)
        q = torch.round(alpha * self.res).to(torch.int64)          # integer grid cell, one row per node
        uniq, inv = torch.unique(q, dim=0, return_inverse=True)     # row-wise: no overflow for any K
        nb = len(uniq)
        psi = self._psi(alpha)
        ws = torch.zeros(nb, dtype=self.dtype, device=alpha.device).index_add_(0, inv, weight)
        wa = torch.zeros(nb, self.K, dtype=self.dtype, device=alpha.device).index_add_(0, inv, weight[:, None] * alpha)
        outer = (weight[:, None, None] * psi[:, :, None] * psi[:, None, :]).reshape(len(psi), -1)
        mom = torch.zeros(nb, outer.shape[1], dtype=self.dtype, device=alpha.device).index_add_(0, inv, outer)
        self.cells.append(uniq); self.wsum.append(ws); self.wa.append(wa); self.mom.append(mom)
        self.total_weight += float(weight.sum())

    @torch.no_grad()
    def _merge(self):
        cells = torch.cat(self.cells)
        uniq, inv = torch.unique(cells, dim=0, return_inverse=True)
        nb = len(uniq)
        dev = cells.device
        ws = torch.zeros(nb, dtype=self.dtype, device=dev).index_add_(0, inv, torch.cat(self.wsum))
        wa = torch.zeros(nb, self.K, dtype=self.dtype, device=dev).index_add_(0, inv, torch.cat(self.wa))
        mom = torch.cat(self.mom)
        mo = torch.zeros(nb, mom.shape[1], dtype=self.dtype, device=dev).index_add_(0, inv, mom)
        return ws, wa / ws[:, None], mo.reshape(nb, self.npsi, self.npsi)

    @torch.no_grad()
    def finalize(self, b: torch.Tensor, bin_chunk: int = 64) -> np.ndarray:
        f = self.field
        if f.P == 0:
            return np.zeros((0, 0))
        ws, abar, Psi = self._merge()
        mats = f.mats(b)                                           # [K+1, d, d]
        B, E = mats[0], mats[1:]
        s = f.s
        S = f.S                                                    # [s, d, d]
        Sflat = S.reshape(s, -1)
        Mom = torch.zeros(self.npsi, self.npsi, s, s, dtype=self.dtype, device=B.device)
        for i in range(0, len(ws), bin_chunk):
            a = abar[i:i + bin_chunk]
            g = B + torch.einsum("bk,kde->bde", a, E)
            Gi = torch.linalg.inv(g)
            Gi = 0.5 * (Gi + Gi.transpose(-1, -2))
            Y = torch.einsum("bde,sef,bfg->bsdg", Gi, S, Gi)       # g^{-1} S_m g^{-1}
            Om = torch.einsum("bsx,tx->bst", Y.reshape(len(a), s, -1), Sflat)
            Mom += torch.einsum("bij,bst->ijst", Psi[i:i + bin_chunk], Om)
        Mom = Mom.cpu().numpy()
        Qb, PB, KE = f.Qb, f.PB, f.KE
        G = np.zeros((f.P, f.P))
        if PB:
            G[:PB, :PB] = Qb.T @ Mom[0, 0] @ Qb
        for j in range(KE):
            sl_j = slice(PB + j * s, PB + (j + 1) * s)
            if PB:
                G[:PB, sl_j] = Qb.T @ Mom[0, 1 + j]
                G[sl_j, :PB] = G[:PB, sl_j].T
            for i in range(KE):
                sl_i = slice(PB + i * s, PB + (i + 1) * s)
                G[sl_i, sl_j] = Mom[1 + i, 1 + j]
        G = 0.5 * (G + G.T)
        self.n_bins = len(ws)
        return G
