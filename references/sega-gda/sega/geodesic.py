"""Numerical geodesic distance contract (Sec. 3.10).

Discrete energy of a piecewise-linear path P = (p_0, ..., p_L):
    E_b(P) = L * sum_l sum_q w_q (dp_l)^T g_b(p_l + t_q dp_l) dp_l
Straight-line initialisation, J_path fixed gradient steps on the interior nodes
with fixed step eta_path, then the lowest-energy finite iterate (ties -> smallest
iteration index) defines D^num_b(x, y). Unordered pairs are canonicalised by the
lexicographic order of the endpoints (energy is reversal-symmetric because the
quadrature rule is symmetric), and identical endpoints give zero distance with a
constant path.

Matrices may be shared ([K+1, d, d]) or per query ([n, K+1, d, d]); the latter is
used by the Jacobian pass so that one backward pass returns one Jacobian row per
query (each query's scores only depend on its own copy of b).
"""
from __future__ import annotations

import numpy as np
import torch


class GeodesicSolver:
    def __init__(self, coverage, L_path: int, Q_quad: int, J_path: int, eta_path: float,
                 dtype=torch.float64, device="cpu"):
        self.cov = coverage
        self.L = int(L_path)
        self.J = int(J_path)
        self.eta = float(eta_path)
        t, w = np.polynomial.legendre.leggauss(int(Q_quad))
        self.t = torch.as_tensor((t + 1.0) / 2.0, dtype=dtype, device=device)    # symmetric nodes in [0,1]
        self.w = torch.as_tensor(w / 2.0, dtype=dtype, device=device)            # positive, sum to 1
        self.dtype, self.device = dtype, device
        frac = torch.arange(self.L + 1, dtype=dtype, device=device) / self.L
        self.frac = frac                                                          # [L+1]

    # ------------------------------------------------------------- energy
    def node_alpha(self, pts: torch.Tensor) -> torch.Tensor:
        """alpha at quadrature nodes of each segment: pts [..., L+1, d] -> [..., L, Q, K]."""
        dp = pts[..., 1:, :] - pts[..., :-1, :]
        nodes = pts[..., :-1, None, :] + self.t[:, None] * dp[..., None, :]
        return self.cov.alpha(nodes)

    def energy(self, pts: torch.Tensor, mats: torch.Tensor) -> torch.Tensor:
        """pts [n, m, L+1, d]; mats [K+1, d, d] or [n, K+1, d, d] -> energy [n, m]."""
        dp = pts[..., 1:, :] - pts[..., :-1, :]                                   # [n, m, L, d]
        alpha = self.node_alpha(pts)                                              # [n, m, L, Q, K]
        abar = (alpha * self.w[:, None]).sum(-2)                                  # [n, m, L, K]
        if mats.dim() == 3:
            Md = torch.einsum("nmld,kde->nmlke", dp, mats)
        else:
            Md = torch.einsum("nmld,nkde->nmlke", dp, mats)
        qf = (Md * dp[..., None, :]).sum(-1)                                      # [n, m, L, K+1]
        per_seg = qf[..., 0] + (abar * qf[..., 1:]).sum(-1)                       # sum_q w_q = 1
        return self.L * per_seg.sum(-1)

    @torch.no_grad()
    def energy_and_grad(self, pts: torch.Tensor, mats: torch.Tensor):
        """Energy and its analytic gradient w.r.t. the interior nodes p_1..p_{L-1} (no autograd graph).

        dE/dp_l = L [ 2 g dp_{l-1} + sum_q w_q t_q h_{l-1,q} - 2 g dp_l + sum_q w_q (1 - t_q) h_{l,q} ],
        g dp_l := sum_q w_q g(x_lq) dp_l,  h = grad_x (dp^T g(x) dp) = sum_k qf_k grad alpha_k(x),
        grad alpha_k = alpha_k (xi_k - sum_j alpha_j xi_j) / sigma^2."""
        dp = pts[..., 1:, :] - pts[..., :-1, :]                                   # [n, m, L, d]
        nodes = pts[..., :-1, None, :] + self.t[:, None] * dp[..., None, :]       # [n, m, L, Q, d]
        alpha = self.cov.alpha(nodes)                                             # [n, m, L, Q, K]
        abar = (alpha * self.w[:, None]).sum(-2)                                  # [n, m, L, K]
        if mats.dim() == 3:
            Md = torch.einsum("nmld,kde->nmlke", dp, mats)
        else:
            Md = torch.einsum("nmld,nkde->nmlke", dp, mats)
        qf = (Md * dp[..., None, :]).sum(-1)                                      # [n, m, L, K+1]
        E = self.L * (qf[..., 0] + (abar * qf[..., 1:]).sum(-1)).sum(-1)
        gdp = Md[..., 0, :] + (abar[..., None] * Md[..., 1:, :]).sum(-2)          # [n, m, L, d]
        qa = qf[..., None, 1:] * alpha                                            # [n, m, L, Q, K]
        xi = self.cov.centers
        h = (qa @ xi - qa.sum(-1, keepdim=True) * (alpha @ xi)) / (self.cov.sigma ** 2)   # [n, m, L, Q, d]
        h_end = (h * (self.w * self.t)[:, None]).sum(-2)                          # weight t_q   (node -> p_{l+1})
        h_start = (h * (self.w * (1.0 - self.t))[:, None]).sum(-2)                # weight 1-t_q (node -> p_l)
        grad = self.L * (2.0 * gdp[..., :-1, :] + h_end[..., :-1, :] - 2.0 * gdp[..., 1:, :] + h_start[..., 1:, :])
        return E, grad

    # ------------------------------------------------------------ solving
    def _endpoints(self, X: torch.Tensor, Y: torch.Tensor):
        n, m = X.shape[0], Y.shape[0]
        Xe = X[:, None, :].expand(n, m, -1)
        Ye = Y[None, :, :].expand(n, m, -1)
        diff = Xe - Ye
        nz = diff != 0
        first = nz.to(torch.int8).argmax(dim=-1, keepdim=True)
        swap = (diff.gather(-1, first).squeeze(-1) > 0)                           # x >_lex y -> swap
        start = torch.where(swap[..., None], Ye, Xe)
        end = torch.where(swap[..., None], Xe, Ye)
        return start, end

    def solve(self, X: torch.Tensor, Y: torch.Tensor, mats: torch.Tensor, mode: str = "eval",
              return_path: bool = False):
        """Numerical squared distance D^num_b between every x in X and y in Y.

        mode = "eval":      no graph w.r.t. mats (forward-only evaluation)
        mode = "unrolled":  differentiable through the finite relaxation (default Jacobian, 3.10.2)
        mode = "envelope":  path relaxed without graph, energy re-evaluated at the selected path
        """
        start, end = self._endpoints(X, Y)
        f = self.frac[:, None]
        pts0 = start[..., None, :] + f * (end - start)[..., None, :]              # straight line
        if self.L == 1 or self.J == 0:
            best_pts = pts0
            mats_e = mats if mode != "eval" else mats.detach()
            with torch.set_grad_enabled(mode != "eval"):
                D = self.energy(pts0, mats_e)
            return (D, best_pts.detach()) if return_path else D

        create_graph = mode == "unrolled"
        mats_relax = mats if create_graph else mats.detach()
        if not create_graph:
            # forward-only relaxation with the analytic gradient (same iterates as autograd)
            with torch.no_grad():
                interior = pts0[..., 1:-1, :].clone()
                s_, e_ = start[..., None, :], end[..., None, :]
                best_E, best_int = None, None
                for j in range(self.J + 1):
                    pts = torch.cat([s_, interior, e_], dim=-2)
                    E, g = self.energy_and_grad(pts, mats_relax)
                    finite = torch.isfinite(E) & torch.isfinite(interior).all(-1).all(-1)
                    E_sel = torch.where(finite, E, torch.full_like(E, float("inf")))
                    if best_E is None:
                        best_E, best_int = E_sel, interior
                    else:
                        better = E_sel < best_E                                   # strict: ties keep earlier
                        best_E = torch.where(better, E_sel, best_E)
                        best_int = torch.where(better[..., None, None], interior, best_int)
                    if j == self.J:
                        break
                    new = interior - self.eta * g
                    ok = torch.isfinite(new).all(-1).all(-1)
                    interior = torch.where(ok[..., None, None], new, interior)
            best_pts = torch.cat([start[..., None, :], best_int, end[..., None, :]], dim=-2)
            if mode == "envelope":
                D = self.energy(best_pts, mats)
            else:
                D = best_E
            return (D, best_pts) if return_path else D

        with torch.enable_grad():
            interior = pts0[..., 1:-1, :].detach().clone().requires_grad_(True)
            s_, e_ = start[..., None, :], end[..., None, :]
            best_E, best_int = None, None
            for j in range(self.J + 1):
                pts = torch.cat([s_, interior, e_], dim=-2)
                E = self.energy(pts, mats_relax)
                finite = torch.isfinite(E) & torch.isfinite(interior).all(-1).all(-1)
                E_sel = torch.where(finite, E, torch.full_like(E, float("inf")))
                if best_E is None:
                    best_E, best_int = E_sel, interior
                else:
                    better = E_sel < best_E                                       # strict: ties keep earlier
                    best_E = torch.where(better, E_sel, best_E)
                    best_int = torch.where(better[..., None, None], interior, best_int)
                if j == self.J:
                    break
                # unrolled: keep the graph so that D depends on b through every relaxation step
                g = torch.autograd.grad(E.sum(), interior, create_graph=True)[0]
                new = interior - self.eta * g
                ok = torch.isfinite(new).all(-1).all(-1)
                interior = torch.where(ok[..., None, None], new, interior)
        best_pts = torch.cat([start[..., None, :], best_int, end[..., None, :]], dim=-2)
        return (best_E, best_pts.detach()) if return_path else best_E
