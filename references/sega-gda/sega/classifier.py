"""Unified reference classifier under any geometry (Sec. 3.2).

Every classifier in the method (pre-training, sub-library heads, forward head,
reverse head, final prediction) is the same function of a distance matrix:

    N_g(z)      = sum_i v_i D_g(z, a_i)               (full source library only)
    r_g(z, a)   = D_g(z, a) / N_g(z)
    S_c(z)      = log sum_{a in A_c} u_{a|c} exp(-r_g(z, a) / tau)
    p(c | z)    = softmax_c S_c(z)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


@dataclass
class RefLibrary:
    """A fixed set of reference records (coordinate, class, positive weight)."""
    coords: torch.Tensor            # [n, d]
    labels: torch.Tensor            # [n] long
    weights: torch.Tensor           # [n] omega_a > 0
    classes: List[int]              # C_A in fixed (sorted) order
    log_u: torch.Tensor             # [n] log of class-normalised weight u_{a|c}
    masks: List[torch.Tensor]       # per class in `classes`: bool mask over records

    @staticmethod
    def build(coords: torch.Tensor, labels: torch.Tensor, weights: Optional[torch.Tensor] = None,
              classes: Optional[Sequence[int]] = None) -> "RefLibrary":
        if weights is None:
            weights = torch.ones(len(labels), dtype=coords.dtype, device=coords.device)
        weights = weights.to(coords.dtype)
        if classes is None:
            classes = sorted(set(labels.tolist()))
        classes = list(classes)
        log_u = torch.empty_like(weights)
        masks = []
        for c in classes:
            m = labels == c
            if not bool(m.any()):
                raise ValueError(f"class {c} has no reference record")
            log_u[m] = torch.log(weights[m]) - torch.log(weights[m].sum())
            masks.append(m)
        return RefLibrary(coords, labels, weights, classes, log_u, masks)

    def subset(self, idx: torch.Tensor) -> "RefLibrary":
        return RefLibrary.build(self.coords[idx], self.labels[idx], self.weights[idx])


def scale_weights(labels: torch.Tensor, num_classes: int, dtype=torch.float64) -> torch.Tensor:
    """v_i = 1 / (C n_{y_i}); sums to one; equal total weight per class (Sec. 3.2.2)."""
    counts = torch.bincount(labels, minlength=num_classes).to(dtype)
    if bool((counts[labels] == 0).any()):
        raise ValueError("empty class in source reference library")
    return 1.0 / (num_classes * counts[labels])


def sq_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact squared Euclidean distances (differentiable everywhere, unlike cdist at 0)."""
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)


def normaliser(D_ref: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """N_g(z) from distances to the FULL source library, D_ref: [nq, n_ref]."""
    return D_ref @ v


def class_scores(D: torch.Tensor, N: torch.Tensor, lib: RefLibrary, tau: float) -> torch.Tensor:
    """S^{A}_{g,c}(z) for c in lib.classes. D: [nq, n_lib] distances to lib records."""
    logits = -(D / N[:, None]) / tau + lib.log_u[None, :]
    return torch.stack([torch.logsumexp(logits[:, m], dim=1) for m in lib.masks], dim=1)


def cross_entropy(S: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
    """CE(c, s) = -s_c + logsumexp(s), per row. y_pos indexes columns of S."""
    return torch.logsumexp(S, dim=1) - S.gather(1, y_pos[:, None]).squeeze(1)


def margin_vs_rest(S: torch.Tensor, c_pos: torch.Tensor) -> torch.Tensor:
    """s_c - max_{q != c} s_q, per row."""
    sc = S.gather(1, c_pos[:, None]).squeeze(1)
    masked = S.clone()
    masked.scatter_(1, c_pos[:, None], float("-inf"))
    return sc - masked.max(dim=1).values


def hinge_gap(S: torch.Tensor, c_pos: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """l_m(c, s; C') = [m_c - s_c + max_{q != c} s_q]_+ (Sec. 3.6.1)."""
    return torch.clamp(m - margin_vs_rest(S, c_pos), min=0.0)


def class_positions(labels: torch.Tensor, classes: Sequence[int]) -> torch.Tensor:
    """Map class ids to column positions in a score matrix ordered by `classes`."""
    lut = {c: i for i, c in enumerate(classes)}
    return torch.tensor([lut[int(c)] for c in labels.tolist()], dtype=torch.long, device=labels.device)
