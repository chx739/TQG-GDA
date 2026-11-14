"""Frozen, label-free R1 Gaussian KDE and finite-support observations.

These are diagnostic observations, not a trained score or an adaptation step.
Every bootstrap uses the same target-graph counts within a task/seed cell.
"""
import hashlib
import json
import math

import torch


def derived_seed(protocol_id, purpose, seed, domain, fold=-1, feature="*", head="*", rep=-1):
    payload = json.dumps([protocol_id, purpose, seed, domain, fold, feature, head, rep],
                         ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def bootstrap_counts(n, repeats, seed, return_indices=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(n, (repeats, n), generator=generator)
    counts = torch.zeros((repeats, n), dtype=torch.float64)
    counts.scatter_add_(1, indices, torch.ones_like(indices, dtype=torch.float64))
    return (counts, indices) if return_indices else counts


def percentile_interval(values):
    return torch.quantile(values, values.new_tensor([0.025, 0.975]), interpolation="linear")


class GaussianKDE:
    """Exact equal-weight Gaussian KDE; stable log-count bootstrap, float64."""

    def __init__(self, x, bandwidth, chunk=32):
        self.x = x.detach().to(device="cpu", dtype=torch.float64)
        self.bandwidth = float(bandwidth)
        self.chunk = int(chunk)
        self.query_count = 0
        if self.x.ndim != 2 or not len(self.x) or not bool(torch.isfinite(self.x).all()):
            raise ValueError("KDE needs finite, nonempty coordinates")
        if not math.isfinite(self.bandwidth) or self.bandwidth <= 0:
            raise ValueError("KDE bandwidth must be positive")
        self.log_normalizer = (self.x.shape[1] * math.log(self.bandwidth)
                               + self.x.shape[1] / 2 * math.log(2 * math.pi))

    def log_kernel(self, z):
        z = z.to(self.x)
        blocks = []
        for block in z.split(self.chunk):
            # Direct cdist avoids cancellation near an input and bounds memory.
            squared = torch.cdist(block, self.x, compute_mode="donot_use_mm_for_euclid_dist").square()
            blocks.append(-squared / (2 * self.bandwidth**2))
        self.query_count += len(z)
        result = torch.cat(blocks)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("non-finite KDE log kernel")
        return result

    def from_log_kernel(self, log_kernel, counts=None, drop_index=None, projections=None):
        """Return [rep,query] logp/ESS and optional normal derivative.

        Counts multiply graph kernels. ESS treats each original graph as a
        support unit, rather than pretending duplicated bootstrap draws create
        new independent locations. Primary ESS always uses equal weights.
        """
        if counts is None:
            counts = torch.ones((1, len(self.x)), dtype=torch.float64)
        else:
            counts = counts.to(self.x).clone()
        if drop_index is not None:
            counts[:, drop_index] = 0
        totals = counts.sum(-1)
        if bool((totals <= 0).any()):
            raise ValueError("KDE counts empty after leave-origin-out")
        log_counts = counts.log()
        values, effective, slopes = [], [], []
        # Chunk repetitions as well as queries: no all-query x N x d allocation.
        for start in range(0, len(counts), self.chunk):
            joint = log_counts[start:start + self.chunk, None, :] + log_kernel[None, :, :]
            denominator = torch.logsumexp(joint, dim=-1)
            logp = denominator - totals[start:start + self.chunk, None].log() - self.log_normalizer
            weights = torch.exp(joint - denominator[..., None])
            values.append(logp)
            effective.append(weights.square().sum(-1).reciprocal())
            if projections is not None:
                slopes.append((weights * projections[None, :, :]).sum(-1) / self.bandwidth**2)
        result = {"logp": torch.cat(values), "ess": torch.cat(effective)}
        if projections is not None:
            result["g"] = torch.cat(slopes)
        if not all(bool(torch.isfinite(value).all()) for value in result.values()):
            raise FloatingPointError("non-finite weighted KDE observation")
        return result

    def evaluate(self, z, counts=None, drop_index=None, normals=None):
        log_kernel = self.log_kernel(z)
        projections = None
        if normals is not None:
            # [query,N] without forming a query x N x d intermediate.
            projections = normals.to(self.x) @ self.x.T - (normals.to(self.x) * z.to(self.x)).sum(-1)[:, None]
        return self.from_log_kernel(log_kernel, counts, drop_index, projections)


def finite_support(x, ids, origin_id, point, normal, sigma, shoulder_sigma, B_ids,
                   min_unlabeled=20, min_B=10):
    """Finite tubular shoulders, preserving all neighbor IDs for sealed audit."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    delta = x - point
    axial = delta @ normal
    tangent = (delta - axial[:, None] * normal).norm(dim=-1)
    distance = delta.norm(dim=-1)
    h = shoulder_sigma * sigma
    eligible = (tangent <= 4 * sigma) & (ids != origin_id)
    positive = eligible & (axial >= h / 2) & (axial < 3 * h / 2)
    negative = eligible & (axial > -3 * h / 2) & (axial <= -h / 2)
    B = set(int(v) for v in B_ids)
    result = {}
    for name, mask in (("positive", positive), ("negative", negative)):
        selected = ids[mask].tolist()
        result[name + "_ids"] = selected
        result[name + "_B_ids"] = [i for i in selected if i in B]
        result[name + "_axial"] = axial[mask].clone()
        result[name + "_tangent"] = tangent[mask].clone()
        result[name + "_distance"] = distance[mask].clone()
        result[name + "_distance_median"] = float(distance[mask].quantile(.5)) if selected else None
        result[name + "_distance_max"] = float(distance[mask].max()) if selected else None
        result[name + "_tangent_median"] = float(tangent[mask].quantile(.5)) if selected else None
        result[name + "_tangent_max"] = float(tangent[mask].max()) if selected else None
    result["unlabeled_supported"] = all(len(result[k + "_ids"]) >= min_unlabeled for k in ("positive", "negative"))
    result["B_supported"] = all(len(result[k + "_B_ids"]) >= min_B for k in ("positive", "negative"))
    return result


def valley_observation(kde, point, normal, sigma, shoulder_sigma, counts, drop_index=None):
    """The valley requires *both* shoulders denser than its center."""
    points = torch.stack((point, point + shoulder_sigma * sigma * normal,
                          point - shoulder_sigma * sigma * normal))
    log_kernel = kde.log_kernel(points)
    projections = normal[None] @ kde.x.T - (normal[None] * points).sum(-1)[:, None]
    direct = kde.from_log_kernel(log_kernel, drop_index=drop_index, projections=projections)
    boot = kde.from_log_kernel(log_kernel, counts=counts, drop_index=drop_index, projections=projections)
    logp, ess = direct["logp"][0], direct["ess"][0]
    V = float(logp[1:].min() - logp[0])
    V_bootstrap = boot["logp"][:, 1:].min(-1).values - boot["logp"][:, 0]
    interval = percentile_interval(V_bootstrap)
    stable = bool((ess >= 5).all())
    state = ("kernel_concentrated" if not stable else
             "valley_supported" if float(interval[0]) > math.log(1.1) else
             "valley_opposed" if float(interval[1]) < 0 else "valley_uncertain")
    g = float(direct["g"][0, 0])
    g_bootstrap = boot["g"][:, 0]
    sigma_g_ci = percentile_interval(sigma * g_bootstrap)
    identifiable = bool(ess[0] >= 5) and (bool(sigma_g_ci[0] > .001) or bool(sigma_g_ci[1] < -.001))
    return {"logp": logp, "ess": ess, "V": V, "V_bootstrap": V_bootstrap, "V_ci": interval,
            "valley_state": state, "kernel_supported": stable,
            "g": g, "g_bootstrap": g_bootstrap, "sigma_g_ci": sigma_g_ci,
            "direction_identifiable": identifiable,
            "density_direction": (-1 if g > 0 else 1) if identifiable else None,
            "ess_center": float(ess[0]), "log_kernel": log_kernel,
            "drop_index": drop_index, "query_points": points}
