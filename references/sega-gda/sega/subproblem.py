"""Convex prox-linear sub-problem for one candidate step (Sec. 3.8.3 - 3.8.5).

    min_D   1/2 L^F_hat(D) + 1/2 L^R_hat(D) + 1/(2 eta) D^T M D
    s.t.    R^S_hat_c(D) <= b_c^risk           for every class c
            b_r + D in B_geo                    (LMIs, tightened by sdp_margin)

Only the geometry-to-score map is linearised (S_hat = S + J D); the full
max-hinge and the cross entropy are kept, so the problem is convex but not a QP.
`eta` is a cvxpy Parameter, so back-tracking re-solves without re-compiling.
"""
from __future__ import annotations

from typing import Dict, Optional

import cvxpy as cp
import numpy as np


def _diff_rows(S: np.ndarray, J: np.ndarray, pos: np.ndarray):
    """For row i with reference class pos[i]: S_i[c] - S_i[q] (q != c) and the matching Jacobian rows.

    Returns D0 [n, C-1], JD [n*(C-1), P]."""
    n, C = S.shape
    others = np.array([[q for q in range(C) if q != c] for c in range(C)], dtype=np.int64)  # [C, C-1]
    oth = others[pos]                                                                      # [n, C-1]
    r = np.arange(n)[:, None]
    D0 = S[np.arange(n), pos][:, None] - S[r, oth]
    JD = J[np.arange(n), pos][:, None, :] - J[r, oth]
    return D0, JD.reshape(n * (C - 1), -1)


def _hinge_np(S, pos, m):
    n, C = S.shape
    sc = S[np.arange(n), pos]
    Sm = S.copy()
    Sm[np.arange(n), pos] = -np.inf
    return np.maximum(m - sc + Sm.max(1), 0.0)


def _ce_np(S, pos):
    mx = S.max(1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(S - mx).sum(1))
    return lse - S[np.arange(len(S)), pos]


class ProxSubproblem:
    def __init__(self, cfg, field, meta: Dict, cur: Dict, budgets: Optional[np.ndarray], M: np.ndarray,
                 b_cur: np.ndarray):
        self.cfg, self.field, self.meta = cfg, field, meta
        self.P = field.P
        self.budgets = budgets
        self.M = M
        self.cur = cur
        P = self.P
        C = meta["C"]
        Delta = cp.Variable(P)
        self.Delta = Delta
        self.inv_eta = cp.Parameter(nonneg=True, value=1.0)
        obj_terms = []

        # ---- forward task: source library classifies accepted targets (all C classes)
        if meta["use_F"]:
            n = cur["S_tp"].shape[0]
            D0, JD = _diff_rows(cur["S_tp"], cur["J_tp"], meta["yhat_pos"])
            diff = cp.reshape(D0.ravel() + JD @ Delta, (n, C - 1), order="C")
            hinge = cp.pos(cp.max(meta["m_tp"][:, None] - diff, axis=1))
            obj_terms.append(0.5 * (meta["w_tp"] / n) @ hinge)
        # ---- reverse task: target library classifies source queries (classes C_T)
        if meta["use_R"]:
            CT = len(meta["C_T"])
            n = cur["S_fitT"].shape[0]
            D0, JD = _diff_rows(cur["S_fitT"], cur["J_fitT"], meta["y_fitT_posT"])
            diff = cp.reshape(D0.ravel() + JD @ Delta, (n, CT - 1), order="C")
            hinge = cp.pos(cp.max(meta["m_fitT"][:, None] - diff, axis=1))
            obj_terms.append(0.5 * cp.sum(hinge) / n)

        L = np.linalg.cholesky(M)
        obj_terms.append(0.5 * self.inv_eta * cp.sum_squares(L.T @ Delta))
        cons = []

        # ---- per-class source CE budget with linearised scores (CE curvature kept)
        if budgets is not None:
            n = cur["S_fit"].shape[0]
            D0, JD = _diff_rows(cur["S_fit"], cur["J_fit"], meta["y_fit_pos"])      # s_y - s_q
            neg = cp.reshape(-(D0.ravel() + JD @ Delta), (n, C - 1), order="C")     # s_q - s_y
            ce = cp.log_sum_exp(cp.hstack([np.zeros((n, 1)), neg]), axis=1)
            for c, ix in enumerate(meta["fit_class_idx"]):
                cons.append(cp.sum(ce[ix]) / len(ix) <= budgets[c])

        # ---- geometric feasible set B_geo (trace / nu-centering hold by construction)
        if P > 0:
            d = field.d
            A_B, A_E = field.linear_maps()
            B_r, E_r = field.BE_numpy(b_cur)
            I = np.eye(d)
            delta = cfg.sdp_margin
            Bx = cp.reshape(B_r.ravel() + A_B @ Delta, (d, d), order="C")
            Bs = 0.5 * (Bx + Bx.T)
            if field.use_B:
                cons.append(Bs - (1.0 / field.kappa + delta) * I >> 0)
                cons.append((field.kappa - delta) * I - Bs >> 0)
            if field.use_E:
                for k in range(field.K):
                    Ex = cp.reshape(E_r[k].ravel() + A_E[k] @ Delta, (d, d), order="C")
                    Es = 0.5 * (Ex + Ex.T)
                    cons.append(field.rho * Bs - Es - delta * I >> 0)
                    cons.append(field.rho * Bs + Es - delta * I >> 0)
        self.prob = cp.Problem(cp.Minimize(cp.sum(cp.hstack(obj_terms))), cons)

    # ------------------------------------------------------------ local model
    def local_values(self, D: np.ndarray) -> Dict:
        meta, cur = self.meta, self.cur
        out = {"L_F": 0.0, "L_R": 0.0}
        if meta["use_F"]:
            S = cur["S_tp"] + cur["J_tp"] @ D
            out["L_F"] = float((meta["w_tp"] * _hinge_np(S, meta["yhat_pos"], meta["m_tp"])).mean())
        if meta["use_R"]:
            S = cur["S_fitT"] + cur["J_fitT"] @ D
            out["L_R"] = float(_hinge_np(S, meta["y_fitT_posT"], meta["m_fitT"]).mean())
        out["L_DA"] = 0.5 * out["L_F"] + 0.5 * out["L_R"]
        S = cur["S_fit"] + cur["J_fit"] @ D
        ce = _ce_np(S, meta["y_fit_pos"])
        out["R_c"] = np.array([ce[ix].mean() for ix in meta["fit_class_idx"]])
        out["normM"] = float(np.sqrt(max(D @ self.M @ D, 0.0)))
        return out

    def objective(self, D: np.ndarray, eta: float) -> float:
        lv = self.local_values(D)
        return lv["L_DA"] + 0.5 / eta * float(D @ self.M @ D)

    # ------------------------------------------------------------------ solve
    def solve(self, eta: float) -> Dict:
        self.inv_eta.value = 1.0 / eta
        status, D = None, None
        for solver in [self.cfg.solver, "SCS"]:
            try:
                kw = {"max_iters": 20000, "eps": 1e-9} if solver == "SCS" else {}
                self.prob.solve(solver=solver, verbose=self.cfg.solver_verbose, **kw)
                status = self.prob.status
                if status in ("optimal", "optimal_inaccurate") and self.Delta.value is not None:
                    D = np.asarray(self.Delta.value, dtype=np.float64).copy()
                    break
            except cp.error.SolverError as e:  # numerical solver failure -> try the fallback solver
                status = f"SolverError({solver}): {str(e)[:120]}"
        res = {"status": status, "Delta": D, "valid": False}
        if D is None:
            return res
        # validity (3.8.5): local feasibility and objective not above the zero step
        lv = self.local_values(D)
        res["local"] = lv
        ok = True
        if self.budgets is not None:
            tol = self.cfg.local_feas_tol * np.maximum(1.0, np.abs(self.budgets))
            ok &= bool(np.all(lv["R_c"] <= self.budgets + tol))
        obj_D = lv["L_DA"] + 0.5 / eta * float(D @ self.M @ D)
        obj_0 = self.local_values(np.zeros_like(D))["L_DA"]
        ok &= obj_D <= obj_0 + 1e-12
        res["obj"], res["obj0"] = obj_D, obj_0
        res["valid"] = bool(ok)
        return res
