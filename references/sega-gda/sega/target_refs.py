"""Fixed trusted target references (Sec. 3.4). Everything here uses g_0 = I_d and frozen coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch

from .classifier import RefLibrary, class_scores, normaliser, sq_euclidean
from .utils import empirical_quantile


@dataclass
class TargetRefs:
    accepted: np.ndarray          # J_+ : positions into the target coordinate array
    yhat: np.ndarray              # pseudo-class of each accepted target (same order as `accepted`)
    w: np.ndarray                 # fixed weights w_j in (0, 1]
    C_T: List[int]                # covered classes (sorted)
    m: np.ndarray                 # m_c, NaN where undefined
    R: np.ndarray                 # R_c, NaN where undefined
    C_cal: List[int]
    C_rel: List[int]
    n_ties: int
    info: Dict = field(default_factory=dict)


def robust_margins(Z: torch.Tensor, z_ref: torch.Tensor, libs, v: torch.Tensor, tau: float):
    """Scores of the three heads h in {0,1,2} and M_0(z, c) for every class c.

    All heads share the denominator N_0(z) of the FULL source library."""
    D = sq_euclidean(Z, z_ref)
    N = normaliser(D, v)
    S_heads = []
    for idx, lib in libs:
        S_heads.append(class_scores(D[:, idx], N, lib, tau))
    C = S_heads[0].shape[1]
    M = torch.full((Z.shape[0], C), float("inf"), dtype=Z.dtype, device=Z.device)
    for S in S_heads:
        for c in range(C):
            others = torch.cat([S[:, :c], S[:, c + 1:]], dim=1)
            M[:, c] = torch.minimum(M[:, c], S[:, c] - others.max(dim=1).values)
    return M, S_heads, D


def select_target_refs(z_ref, y_ref, sub1_pos, sub2_pos, z_cal, y_cal, z_tgt, v, num_classes: int,
                       tau: float, q_m: float, q_R: float, n_cal_min: int, mode: str = "robust") -> TargetRefs:
    classes = list(range(num_classes))
    full = (torch.arange(len(y_ref), device=z_ref.device), RefLibrary.build(z_ref, y_ref, classes=classes))
    libs = [full]
    for pos in (sub1_pos, sub2_pos):
        pos_t = torch.as_tensor(pos, device=z_ref.device)
        libs.append((pos_t, RefLibrary.build(z_ref[pos_t], y_ref[pos_t], classes=classes)))

    # ---- 3.4.2: class margins and absolute support from stable & correct calibration samples
    M_cal, _, D_cal = robust_margins(z_cal, z_ref, libs, v, tau)
    m = np.full(num_classes, np.nan)
    R = np.full(num_classes, np.nan)
    C_cal, C_rel = [], []
    n_V = {}
    for c in classes:
        sel = (y_cal == c) & (M_cal[:, c] > 0)
        n_V[c] = int(sel.sum())
        if n_V[c] < n_cal_min:
            continue
        C_cal.append(c)
        ref_c = (y_ref == c)
        delta = D_cal[sel][:, ref_c].min(dim=1).values           # delta_c(z), NOT normalised
        m[c] = empirical_quantile(M_cal[sel, c].cpu().numpy(), q_m)
        R[c] = empirical_quantile(delta.cpu().numpy(), q_R)
        if m[c] > 0 and R[c] > 0:
            C_rel.append(c)

    # ---- 3.4.3: accept target references
    M_t, S_heads_t, D_t = robust_margins(z_tgt, z_ref, libs, v, tau)
    S_full = S_heads_t[0]
    top2 = torch.topk(S_full, k=2, dim=1)
    tie = top2.values[:, 0] == top2.values[:, 1]
    yhat_all = S_full.argmax(dim=1)
    n_ties = int(tie.sum())

    if mode == "robust":
        ok = ~tie
        delta_t = torch.full((len(z_tgt),), float("inf"), dtype=z_tgt.dtype, device=z_tgt.device)
        for c in classes:
            rows = yhat_all == c
            if bool(rows.any()):
                delta_t[rows] = D_t[rows][:, y_ref == c].min(dim=1).values
        rel = torch.zeros_like(ok)
        m_t = torch.full_like(delta_t, float("nan"))
        R_t = torch.full_like(delta_t, float("nan"))
        for c in C_rel:
            rows = yhat_all == c
            rel |= rows
            m_t[rows] = float(m[c]); R_t[rows] = float(R[c])
        Mhat = M_t.gather(1, yhat_all[:, None]).squeeze(1)
        ok = ok & rel & (Mhat >= m_t) & (delta_t <= R_t)
        acc = torch.nonzero(ok).squeeze(1)
        # w_j = min_h p_0^{A^(h)}(yhat | z)
        P = torch.stack([torch.softmax(S, 1).gather(1, yhat_all[:, None]).squeeze(1) for S in S_heads_t], 1)
        w_all = P.min(dim=1).values
    elif mode == "all":
        # ablation: every non-tied target with its argmax class and unit weight
        acc = torch.nonzero(~tie).squeeze(1)
        w_all = torch.ones(len(z_tgt), dtype=z_tgt.dtype, device=z_tgt.device)
        for c in classes:                      # m_c must exist for the hinge; use 0 if undefined
            if np.isnan(m[c]):
                m[c] = 0.0
    else:
        raise ValueError(mode)

    acc_np = acc.cpu().numpy()
    yhat = yhat_all[acc].cpu().numpy()
    w = w_all[acc].cpu().numpy()
    C_T = sorted(set(yhat.tolist()))
    info = {"n_V": n_V, "n_target": int(len(z_tgt)), "n_accepted": int(len(acc_np)),
            "accepted_per_class": {int(c): int((yhat == c).sum()) for c in C_T}}
    return TargetRefs(acc_np, yhat, w, C_T, m, R, C_cal, C_rel, n_ties, info)
