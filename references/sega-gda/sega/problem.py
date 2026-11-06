"""The two cross-domain margin tasks, the source risk, and their score Jacobians (Sec. 3.6-3.8.2).

Fixed data during adaptation:
  A_S     : full source reference library (coords a_i, labels y_i, omega = 1)
  I_fit   : source queries (source-risk constraint set and reverse-task queries)
  A_T     : accepted target references (z_j, yhat_j, w_j), classes C_T
Only b (the metric field) changes. All scores use the numerical distance D^num_b
of `GeodesicSolver` and the full-source-library normaliser N_b(z).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from .classifier import (RefLibrary, class_positions, class_scores, cross_entropy, hinge_gap,
                         normaliser)


class SegaProblem:
    def __init__(self, cfg, field, solver, z_ref, y_ref, v, z_fit, y_fit, z_tp, yhat_tp, w_tp, C_T,
                 m_c: np.ndarray, num_classes: int):
        self.cfg = cfg
        self.field, self.solver = field, solver
        self.dtype, self.device = z_ref.dtype, z_ref.device
        self.C = num_classes
        self.classes = list(range(num_classes))
        self.C_T = list(C_T)
        self.tau = cfg.tau
        self.z_ref, self.y_ref, self.v = z_ref, y_ref, v
        self.lib_S = RefLibrary.build(z_ref, y_ref, classes=self.classes)
        self.z_fit, self.y_fit = z_fit, y_fit
        self.y_fit_pos = class_positions(y_fit, self.classes)
        self.use_F = bool(cfg.use_forward) and len(z_tp) > 0
        self.use_R = bool(cfg.use_reverse) and len(self.C_T) >= 2
        self.z_tp = z_tp
        self.yhat_tp = yhat_tp
        self.w_tp = w_tp
        m_t = torch.as_tensor(m_c, dtype=self.dtype, device=self.device)
        if len(z_tp):
            self.lib_T = RefLibrary.build(z_tp, yhat_tp, weights=w_tp, classes=self.C_T)
            self.yhat_pos = class_positions(yhat_tp, self.classes)
            self.m_tp = m_t[yhat_tp]
        # reverse task queries: I_fit^T = {i in I_fit : y_i in C_T}
        cset = set(self.C_T)
        self.fitT_mask = torch.tensor([int(c) in cset for c in y_fit.tolist()], device=self.device)
        self.fitT_idx = torch.nonzero(self.fitT_mask).squeeze(1)
        if self.fitT_idx.numel():
            self.y_fitT_posT = class_positions(y_fit[self.fitT_idx], self.C_T)
            self.m_fitT = m_t[y_fit[self.fitT_idx]]
        self.fit_class_idx = [torch.nonzero(y_fit == c).squeeze(1) for c in self.classes]
        self.n_fit_c = np.array([len(ix) for ix in self.fit_class_idx], dtype=np.float64)
        if (self.n_fit_c == 0).any():
            raise ValueError("every class needs at least one I_fit query")
        # number of distinct unordered endpoint pairs used by the tasks (path measure, 3.8.4)
        n_ref, n_fit, n_tp = len(z_ref), len(z_fit), len(z_tp)
        self.n_paths = n_fit * n_ref + (n_tp * n_ref if self.use_F else 0) + \
            (len(self.fitT_idx) * n_tp if self.use_R else 0)

    # ------------------------------------------------------------- losses
    def losses_from_scores(self, S_fit: torch.Tensor, S_fitT: Optional[torch.Tensor],
                           S_tp: Optional[torch.Tensor]) -> Dict:
        out = {}
        if self.use_F:
            hF = hinge_gap(S_tp, self.yhat_pos, self.m_tp)
            out["L_F"] = float((self.w_tp * hF).sum() / len(hF))
        else:
            out["L_F"] = 0.0
        if self.use_R:
            hR = hinge_gap(S_fitT, self.y_fitT_posT, self.m_fitT)
            out["L_R"] = float(hR.mean())
        else:
            out["L_R"] = 0.0
        out["L_DA"] = 0.5 * out["L_F"] + 0.5 * out["L_R"]
        ce = cross_entropy(S_fit, self.y_fit_pos)
        out["R_c"] = np.array([float(ce[ix].mean()) for ix in self.fit_class_idx])
        out["R_S"] = float((self.n_fit_c / self.n_fit_c.sum() * out["R_c"]).sum())
        out["src_fit_acc"] = float((S_fit.argmax(1) == self.y_fit_pos).double().mean())
        return out

    # --------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, b: torch.Tensor) -> Dict:
        """Non-linear re-evaluation of every score, normaliser and loss at state b."""
        mats = self.field.mats(b)
        n_ref, n_tp = len(self.z_ref), len(self.z_tp)
        S_fit = torch.empty(len(self.z_fit), self.C, dtype=self.dtype, device=self.device)
        S_fitT = torch.empty(len(self.fitT_idx), len(self.C_T), dtype=self.dtype, device=self.device) \
            if self.use_R else None
        cols = n_ref + (n_tp if self.use_R else 0)
        step = max(1, self.cfg.pair_chunk_eval // max(cols, 1))
        rev_pos = torch.full((len(self.z_fit),), -1, dtype=torch.long, device=self.device)
        rev_pos[self.fitT_idx] = torch.arange(len(self.fitT_idx), device=self.device)
        for s in range(0, len(self.z_fit), step):
            sl = slice(s, s + step)
            Xc = self.z_fit[sl]
            D = self.solver.solve(Xc, self.z_ref, mats, "eval")
            N = normaliser(D, self.v)
            S_fit[sl] = class_scores(D, N, self.lib_S, self.tau)
            if self.use_R:
                rows = self.fitT_mask[sl]
                if bool(rows.any()):
                    DT = self.solver.solve(Xc[rows], self.z_tp, mats, "eval")
                    S_fitT[rev_pos[sl][rows]] = class_scores(DT, N[rows], self.lib_T, self.tau)
        S_tp = None
        if self.use_F:
            S_tp = torch.empty(n_tp, self.C, dtype=self.dtype, device=self.device)
            step = max(1, self.cfg.pair_chunk_eval // n_ref)
            for s in range(0, n_tp, step):
                sl = slice(s, s + step)
                D = self.solver.solve(self.z_tp[sl], self.z_ref, mats, "eval")
                S_tp[sl] = class_scores(D, normaliser(D, self.v), self.lib_S, self.tau)
        out = self.losses_from_scores(S_fit, S_fitT, S_tp)
        out.update({"S_fit": S_fit, "S_fitT": S_fitT, "S_tp": S_tp})
        return out

    # ------------------------------------------------------ Jacobian pass
    def jacobian_pass(self, b: torch.Tensor, gram_acc=None) -> Dict:
        """Scores at b and their Jacobians w.r.t. the effective coordinates (Sec. 3.8.2).

        Each query gets its own copy of b, so d(sum_i S_c(z_i))/d b_rep[i] is row i of J.
        The default mode differentiates through the finite path relaxation (3.10.2)."""
        P = self.field.P
        mode = self.cfg.jac_mode
        n_ref, n_tp = len(self.z_ref), len(self.z_tp)
        w_node = None
        if gram_acc is not None:
            # path measure: equal weight per path, equal per segment, quadrature weights inside
            w_node = (0.5 / self.n_paths) * (1.0 / self.solver.L) * self.solver.w      # [Q]
        S_fit = torch.empty(len(self.z_fit), self.C, dtype=self.dtype, device=self.device)
        J_fit = torch.zeros(len(self.z_fit), self.C, P, dtype=self.dtype, device=self.device)
        nT = len(self.fitT_idx)
        S_fitT = torch.empty(nT, len(self.C_T), dtype=self.dtype, device=self.device) if self.use_R else None
        J_fitT = torch.zeros(nT, len(self.C_T), P, dtype=self.dtype, device=self.device) if self.use_R else None
        rev_pos = torch.full((len(self.z_fit),), -1, dtype=torch.long, device=self.device)
        rev_pos[self.fitT_idx] = torch.arange(nT, device=self.device)

        def add_nodes(path):
            if gram_acc is not None:
                with torch.no_grad():
                    a = self.solver.node_alpha(path)                                      # [n, m, L, Q, K]
                    gram_acc.add(a, w_node.expand(a.shape[:-1]))                          # chi per node

        cols = n_ref + (n_tp if self.use_R else 0)
        step = max(1, self.cfg.pair_chunk_jac // max(cols, 1))
        for s in range(0, len(self.z_fit), step):
            sl = slice(s, min(s + step, len(self.z_fit)))
            Xc = self.z_fit[sl]
            nc = Xc.shape[0]
            b_rep = b.detach().expand(nc, P).clone().requires_grad_(P > 0)
            mats_q = self.field.mats(b_rep)
            D, path = self.solver.solve(Xc, self.z_ref, mats_q, mode, return_path=True)
            N = normaliser(D, self.v)
            S = class_scores(D, N, self.lib_S, self.tau)
            S_fit[sl] = S.detach()
            add_nodes(path)
            rows = None
            if self.use_R:
                rows = self.fitT_mask[sl]
                if bool(rows.any()):
                    DT, pathT = self.solver.solve(Xc[rows], self.z_tp, mats_q[rows], mode, return_path=True)
                    ST = class_scores(DT, N[rows], self.lib_T, self.tau)
                    S_fitT[rev_pos[sl][rows]] = ST.detach()
                    add_nodes(pathT)
                else:
                    rows = None
            if P > 0:
                for c in range(self.C):
                    g = torch.autograd.grad(S[:, c].sum(), b_rep, retain_graph=True)[0]
                    J_fit[sl, c] = g
                if rows is not None:
                    for k in range(len(self.C_T)):
                        g = torch.autograd.grad(ST[:, k].sum(), b_rep, retain_graph=True)[0]
                        J_fitT[rev_pos[sl][rows], k] = g[rows]
            del D, N, S, mats_q, b_rep

        S_tp = J_tp = None
        if self.use_F:
            S_tp = torch.empty(n_tp, self.C, dtype=self.dtype, device=self.device)
            J_tp = torch.zeros(n_tp, self.C, P, dtype=self.dtype, device=self.device)
            step = max(1, self.cfg.pair_chunk_jac // n_ref)
            for s in range(0, n_tp, step):
                sl = slice(s, min(s + step, n_tp))
                Xc = self.z_tp[sl]
                nc = Xc.shape[0]
                b_rep = b.detach().expand(nc, P).clone().requires_grad_(P > 0)
                mats_q = self.field.mats(b_rep)
                D, path = self.solver.solve(Xc, self.z_ref, mats_q, mode, return_path=True)
                S = class_scores(D, normaliser(D, self.v), self.lib_S, self.tau)
                S_tp[sl] = S.detach()
                add_nodes(path)
                if P > 0:
                    for c in range(self.C):
                        J_tp[sl, c] = torch.autograd.grad(S[:, c].sum(), b_rep, retain_graph=True)[0]
                del D, S, mats_q, b_rep

        if gram_acc is not None:
            # fixed reference measure mu_0 enters with weight 1/2
            a0 = self.solver.cov.alpha(self.solver.cov.mu_points)
            gram_acc.add(a0, 0.5 * self.solver.cov.mu_weights)

        out = self.losses_from_scores(S_fit, S_fitT, S_tp)
        out.update({"S_fit": S_fit, "S_fitT": S_fitT, "S_tp": S_tp,
                    "J_fit": J_fit, "J_fitT": J_fitT, "J_tp": J_tp})
        return out

    # ----------------------------------------------- data for sub-problem
    def numpy_meta(self) -> Dict:
        meta = {
            "C": self.C, "C_T": self.C_T,
            "y_fit_pos": self.y_fit_pos.cpu().numpy(),
            "fit_class_idx": [ix.cpu().numpy() for ix in self.fit_class_idx],
            "use_F": self.use_F, "use_R": self.use_R,
        }
        if self.use_F:
            meta.update({"yhat_pos": self.yhat_pos.cpu().numpy(), "w_tp": self.w_tp.cpu().numpy(),
                         "m_tp": self.m_tp.cpu().numpy()})
        if self.use_R:
            meta.update({"y_fitT_posT": self.y_fitT_posT.cpu().numpy(), "m_fitT": self.m_fitT.cpu().numpy()})
        return meta
