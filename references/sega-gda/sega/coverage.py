"""Fixed spatial coverage (Sec. 3.5.1): mu_0, centres xi_k, bandwidth sigma_cov, alpha_k(z), nu_k."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans


@dataclass
class Coverage:
    centers: torch.Tensor     # [K, d]
    sigma: float              # sigma_cov
    nu: np.ndarray            # [K], nu_k = int alpha_k dmu_0
    mu_points: torch.Tensor   # support of mu_0, [n, d]
    mu_weights: torch.Tensor  # weights of mu_0, [n]
    s_cov: float
    K_requested: int

    @property
    def K(self) -> int:
        return int(self.centers.shape[0])

    def alpha(self, z: torch.Tensor) -> torch.Tensor:
        """alpha_k(z): normalised Gaussian weights with the common bandwidth, z: [..., d] -> [..., K].

        softmax_k(-||z - xi_k||^2 / 2 sigma^2) = softmax_k((z.xi_k - ||xi_k||^2 / 2) / sigma^2) exactly,
        since ||z||^2 is common to all k."""
        logits = (z @ self.centers.T - 0.5 * (self.centers ** 2).sum(-1)) / (self.sigma ** 2)
        return torch.softmax(logits, dim=-1)


def build_coverage(z_geom_src: torch.Tensor, z_tgt: torch.Tensor, K: int, c_h: float, c_min: float,
                   seed: int, n_init: int = 10) -> Coverage:
    if c_h < c_min or c_min <= 0:
        raise ValueError("need c_h >= c_min > 0")
    dtype, device = z_tgt.dtype, z_tgt.device
    n_s, n_t = len(z_geom_src), len(z_tgt)
    X = torch.cat([z_geom_src, z_tgt], 0)
    w = torch.cat([torch.full((n_s,), 0.5 / n_s, dtype=dtype, device=device),
                   torch.full((n_t,), 0.5 / n_t, dtype=dtype, device=device)])
    Xn, wn = X.cpu().numpy(), w.cpu().numpy()
    n_distinct = len(np.unique(Xn, axis=0))
    K_eff = int(min(K, n_distinct))
    km = KMeans(n_clusters=K_eff, init="k-means++", n_init=n_init, random_state=seed)
    km.fit(Xn, sample_weight=wn)
    centers = km.cluster_centers_
    assign = km.labels_
    mass = np.bincount(assign, weights=wn, minlength=K_eff)
    centers = centers[mass > 0]                            # drop empty centres
    centers = np.unique(centers, axis=0)                   # drop duplicate centres
    zbar = (wn[:, None] * Xn).sum(0)
    s_cov = float(np.sqrt((wn * ((Xn - zbar) ** 2).sum(1)).sum()))
    if s_cov <= 0:
        raise ValueError("coverage coordinates are all identical")
    sigma = c_h * s_cov
    cov = Coverage(torch.as_tensor(centers, dtype=dtype, device=device), sigma, None, X, w, s_cov, K)
    with torch.no_grad():
        nu = (w[:, None] * cov.alpha(X)).sum(0).cpu().numpy()
    cov.nu = nu
    return cov
