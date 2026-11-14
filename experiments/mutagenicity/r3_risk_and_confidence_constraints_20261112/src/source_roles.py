"""Source role allocator used by R3/R4, preserved from SEGA-ref (ab08590)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class SourceRoles:
    """Disjoint source roles, all stored as ORIGINAL graph ids (dataset indices)."""
    ref: np.ndarray          # I_ref : final-prediction source reference library
    fit: np.ndarray          # I_fit : pre-training queries + source-risk constraint set
    cal: np.ndarray          # I_cal : checkpoint selection + threshold calibration
    ref_sub1: np.ndarray     # per-class split of I_ref into two non-empty halves (3.4.1)
    ref_sub2: np.ndarray


def split_source_roles(src_ids: np.ndarray, y: np.ndarray, frac_ref: float, frac_fit: float,
                       num_classes: int, seed: int) -> SourceRoles:
    """Class-stratified split of the source domain into I_ref / I_fit / I_cal by graph id."""
    rng = np.random.RandomState(seed)
    ref, fit, cal, s1, s2 = [], [], [], [], []
    for c in range(num_classes):
        ids_c = np.sort(src_ids[y[src_ids] == c])
        ids_c = ids_c[rng.permutation(len(ids_c))]
        n = len(ids_c)
        if n < 3:
            raise ValueError(f"class {c} has only {n} source graphs; need >= 3 (2 ref + 1 fit)")
        n_ref = min(max(2, int(round(frac_ref * n))), n - 1)
        n_fit = max(1, int(round(frac_fit * n)))
        if n_ref + n_fit > n:
            n_fit = n - n_ref                                   # >= 1 because n_ref <= n - 1
        r, f, k = ids_c[:n_ref], ids_c[n_ref:n_ref + n_fit], ids_c[n_ref + n_fit:]
        ref.append(r); fit.append(f); cal.append(k)
        half = len(r) // 2                       # both halves non-empty since n_ref >= 2
        s1.append(r[:half]); s2.append(r[half:])
    roles = SourceRoles(*(np.concatenate(a) for a in (ref, fit, cal, s1, s2)))
    if len(roles.cal) == 0:
        raise ValueError("I_cal is empty; lower frac_ref / frac_fit")
    assert len(np.intersect1d(roles.ref, roles.fit)) == 0
    assert len(np.intersect1d(roles.ref, roles.cal)) == 0
    assert len(np.intersect1d(roles.fit, roles.cal)) == 0
    return roles
