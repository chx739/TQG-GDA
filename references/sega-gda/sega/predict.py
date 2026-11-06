"""Unified target prediction (Sec. 3.11, stage D): full source library + final shared metric g_*."""
from __future__ import annotations

import torch

from .classifier import RefLibrary, class_scores, normaliser


@torch.no_grad()
def predict(Z: torch.Tensor, z_ref: torch.Tensor, y_ref: torch.Tensor, v: torch.Tensor, field, solver,
            b: torch.Tensor, tau: float, num_classes: int, pair_chunk: int = 131072):
    lib = RefLibrary.build(z_ref, y_ref, classes=list(range(num_classes)))
    mats = field.mats(b)
    step = max(1, pair_chunk // len(z_ref))
    S_all = []
    for s in range(0, len(Z), step):
        D = solver.solve(Z[s:s + step], z_ref, mats, "eval")
        S_all.append(class_scores(D, normaliser(D, v), lib, tau))
    S = torch.cat(S_all, 0)
    # ties -> first class in the fixed class order (torch.argmax returns the first maximum)
    return S.argmax(1), torch.softmax(S, 1), S
