"""Shared global metric + bounded local corrections (Sec. 3.5.2) in effective coordinates (Sec. 3.8.1).

    g_b(z) = B(b) + sum_k alpha_k(z) E_k(b)
    B(b)   = I_d + sum_l b_l T^B_l          (tr T^B_l = 0)
    E_k(b) = sum_l b_l T^E_{kl}             (sum_k nu_k T^E_{kl} = 0)

Construction of the orthonormal basis (in the joint Frobenius inner product):
  * S_1..S_s, s = d(d+1)/2, is the orthonormal basis of symmetric matrices
    (e_i e_i^T and (e_i e_j^T + e_j e_i^T)/sqrt 2);
  * the B block uses Q_B (s x (s-1)), an orthonormal basis of {beta : sum_diag beta = 0};
  * the E block uses U (K x (K-1)), an orthonormal basis of {u : nu^T u = 0}, so that
    E_k = sum_j U_{kj} C_j with free symmetric C_j; ||(E_k)||^2 = ||C||^2.
Hence P = (s-1) + (K-1) s = K d(d+1)/2 - 1 when kappa > 1 and rho > 0.
Distinct coverage centres make alpha_1..alpha_K linearly independent functions,
so no remaining direction maps to the identically-zero field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import null_space


def sym_basis(d: int) -> np.ndarray:
    mats, is_diag = [], []
    for i in range(d):
        for j in range(i, d):
            S = np.zeros((d, d))
            if i == j:
                S[i, i] = 1.0
            else:
                S[i, j] = S[j, i] = 1.0 / np.sqrt(2.0)
            mats.append(S)
            is_diag.append(i == j)
    return np.stack(mats), np.array(is_diag, dtype=np.float64)


class FieldParam:
    def __init__(self, d: int, nu: np.ndarray, kappa: float, rho: float, dtype=torch.float64, device="cpu"):
        self.d = d
        self.K = len(nu)
        self.nu = np.asarray(nu, dtype=np.float64)
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.dtype, self.device = dtype, device
        S_np, diag = sym_basis(d)
        self.s = S_np.shape[0]
        self.S_np = S_np
        self.S = torch.as_tensor(S_np, dtype=dtype, device=device)          # [s, d, d]
        self.use_B = self.kappa > 1.0
        self.use_E = (self.rho > 0.0) and (self.K > 1)
        self.Qb = null_space(diag[None, :]) if self.use_B else np.zeros((self.s, 0))   # [s, s-1]
        self.U = null_space(self.nu[None, :]) if self.use_E else np.zeros((self.K, 0))  # [K, K-1]
        self.PB = self.Qb.shape[1]
        self.KE = self.U.shape[1]
        self.PE = self.KE * self.s
        self.P = self.PB + self.PE
        self.Qb_t = torch.as_tensor(self.Qb, dtype=dtype, device=device)
        self.U_t = torch.as_tensor(self.U, dtype=dtype, device=device)
        self.I = torch.eye(d, dtype=dtype, device=device)

    # ------------------------------------------------------------------ maps
    def split(self, b: torch.Tensor):
        beta = b[..., :self.PB]
        C = b[..., self.PB:].reshape(*b.shape[:-1], self.KE, self.s)
        return beta, C

    def _mat(self, vs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...s,sde->...de", vs, self.S)

    def mats(self, b: torch.Tensor) -> torch.Tensor:
        """[K+1, d, d]: index 0 is B, index k = 1..K is E_k. b: [P] (or [n, P] -> [n, K+1, d, d])."""
        beta, C = self.split(b)
        B = self.I + self._mat(beta @ self.Qb_t.T)
        vecE = torch.einsum("kj,...js->...ks", self.U_t, C)                 # [..., K, s]
        E = self._mat(vecE)                                                  # [..., K, d, d]
        return torch.cat([B.unsqueeze(-3), E], dim=-3)

    def BE_numpy(self, b: np.ndarray):
        M = self.mats(torch.as_tensor(b, dtype=self.dtype, device=self.device)).cpu().numpy()
        return M[0], M[1:]

    def linear_maps(self):
        """Affine maps used inside the convex sub-problem: vec_C(B(b+D)) = vec_C(B(b)) + A_B D, same for E_k."""
        Sflat = self.S_np.reshape(self.s, -1).T                              # [d^2, s]
        A_B = np.zeros((self.d * self.d, self.P))
        if self.PB:
            A_B[:, :self.PB] = Sflat @ self.Qb
        A_E = []
        for k in range(self.K):
            A = np.zeros((self.d * self.d, self.P))
            if self.PE:
                A[:, self.PB:] = Sflat @ np.kron(self.U[k][None, :], np.eye(self.s))
            A_E.append(A)
        return A_B, A_E

    # ----------------------------------------------------------- feasibility
    def feasibility(self, b: np.ndarray) -> dict:
        B, E = self.BE_numpy(b)
        eB = np.linalg.eigvalsh(B)
        out = {"trace_B": float(np.trace(B)), "eig_B_min": float(eB.min()), "eig_B_max": float(eB.max()),
               "nu_sum_E": float(np.abs(np.tensordot(self.nu, E, axes=1)).max()) if len(E) else 0.0}
        lo, hi = np.inf, np.inf
        for k in range(self.K):
            lo = min(lo, float(np.linalg.eigvalsh(self.rho * B - E[k]).min()))
            hi = min(hi, float(np.linalg.eigvalsh(self.rho * B + E[k]).min()))
        out["eig_rhoB_minus_E_min"] = lo
        out["eig_rhoB_plus_E_min"] = hi
        return out

    def is_feasible(self, b: np.ndarray, tol: float = 1e-10) -> bool:
        f = self.feasibility(b)
        ok = (f["eig_B_min"] >= 1.0 / self.kappa - tol) and (f["eig_B_max"] <= self.kappa + tol)
        ok = ok and abs(f["trace_B"] - self.d) <= 1e-8 * self.d and f["nu_sum_E"] <= 1e-8
        if self.K > 0:
            ok = ok and f["eig_rhoB_minus_E_min"] >= -tol and f["eig_rhoB_plus_E_min"] >= -tol
        return bool(ok)

    @property
    def m_minus(self) -> float:
        return (1.0 - self.rho) / self.kappa

    @property
    def m_plus(self) -> float:
        return (1.0 + self.rho) * self.kappa
