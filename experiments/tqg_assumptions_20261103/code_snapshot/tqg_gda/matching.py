"""Detached source-only-calibrated matching for the revision-v1 protocol.

The gate is an empirical support diagnostic.  It is deliberately returned
separately from the weights so that matching and rejection can be ablated.
"""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MatchingConfig:
    gamma: float = .75
    tau: float = 1.
    eps_distance: float = 1e-12
    ess_fraction: float = .25
    coverage_quantile: float = .95

    def __post_init__(self):
        for name in ('gamma', 'tau', 'eps_distance', 'ess_fraction', 'coverage_quantile'):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not 0 <= self.gamma <= 1:
            raise ValueError('gamma must lie in [0, 1]')
        if self.tau <= 0 or self.eps_distance <= 0:
            raise ValueError('tau and eps_distance must be positive')
        if not 0 <= self.ess_fraction <= 1:
            raise ValueError('ess_fraction must lie in [0, 1]')
        if not 0 <= self.coverage_quantile <= 1:
            raise ValueError('coverage_quantile must lie in [0, 1]')


def _detached_float64(value, name):
    if not isinstance(value, torch.Tensor) or value.is_complex():
        raise ValueError(f'{name} must be a real tensor')
    value = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} contains a nonfinite value')
    return value


def normalize_embedding(embedding, epsilon=1e-8):
    """Normalize rows by max(L2 norm, epsilon), without retaining autograd."""
    embedding = _detached_float64(embedding, 'embedding')
    if embedding.ndim != 2 or min(embedding.shape) <= 0:
        raise ValueError('embedding must have shape [n_graphs, n_features] with positive sizes')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be finite and positive')
    norm = torch.linalg.vector_norm(embedding, dim=1, keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise ValueError('embedding norm is nonfinite')
    return (embedding / norm.clamp_min(epsilon)).detach()


def matching_weights(source_signature, target_signature, config=None):
    """Compute local weights [nS,nT] and a separate coverage/ESS gate [nT].

    Distances average over *all* nonsample signature coordinates.  Both scale
    and coverage threshold are calibrated from source signatures only.  A
    constant source signature provides no distance scale: return uniform
    weights, a false gate, and nullable undefined scale diagnostics.
    """
    if config is None:
        config = MatchingConfig()
    if not isinstance(config, MatchingConfig):
        raise ValueError('config must be a MatchingConfig')
    source = _detached_float64(source_signature, 'source_signature')
    target = _detached_float64(target_signature, 'target_signature')
    if source.ndim < 2 or target.ndim != source.ndim:
        raise ValueError('signatures must have equally many axes, including a graph axis')
    if source.shape[1:] != target.shape[1:] or any(size <= 0 for size in source.shape + target.shape):
        raise ValueError('signatures must have matching positive feature dimensions and nonempty graph axes')
    if source.device != target.device:
        raise ValueError('source and target signatures must share a device')
    source, target = source.flatten(1), target.flatten(1)
    distances = (source[:, None, :] - target[None, :, :]).square().mean(-1)
    source_distances = (source[:, None, :] - source[None, :, :]).square().mean(-1)
    if not bool(torch.isfinite(distances).all() and torch.isfinite(source_distances).all()):
        raise ValueError('signature distances are nonfinite')
    n_source, n_target = distances.shape
    upper = torch.triu_indices(n_source, n_source, offset=1, device=source.device)
    pair_distances = source_distances[upper[0], upper[1]]
    positive = pair_distances[pair_distances > config.eps_distance]
    uniform = torch.full_like(distances, 1 / n_source)
    if positive.numel() == 0:
        ess = torch.full((n_target,), float(n_source), dtype=torch.float64, device=source.device)
        return {
            'weights': uniform, 'gate': torch.zeros(n_target, dtype=torch.bool, device=source.device),
            'distances': distances, 'source_distances': source_distances,
            'distance_scale': None, 'coverage_threshold': None,
            'nearest_scaled_distance': None, 'ess': ess,
            'statistics': {'uninformative_signature': True, 'n_source': n_source,
                           'n_target': n_target, 'distance_scale': None,
                           'coverage_threshold': None, 'acceptance_rate': 0.,
                           'ess_min': float(n_source), 'ess_mean': float(n_source)},
        }
    # torch.median chooses the lower middle item for an even count, whereas
    # the specified mathematical median interpolates the two middle items.
    scale = torch.quantile(positive, .5, interpolation='linear')
    scaled = distances / scale
    scaled_source = source_distances / scale
    if not bool(torch.isfinite(scaled).all() and torch.isfinite(scaled_source).all()):
        raise ValueError('scaled signature distances are nonfinite')
    # Subtract each column minimum first. This leaves softmax unchanged and
    # avoids an irrelevant large common offset in far-away target batches.
    logits = -(scaled - scaled.amin(dim=0, keepdim=True)) / config.tau
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('matching logits are nonfinite')
    weights = (1 - config.gamma) * uniform + config.gamma * logits.softmax(dim=0)
    ess = weights.square().sum(dim=0).reciprocal()
    nearest_source = scaled_source.clone()
    nearest_source.fill_diagonal_(float('inf'))
    leave_one_out = nearest_source.amin(dim=1)
    threshold = torch.quantile(leave_one_out, config.coverage_quantile, interpolation='linear')
    nearest_target = scaled.amin(dim=0)
    gate = (nearest_target <= threshold) & (ess >= n_source * config.ess_fraction)
    return {
        'weights': weights.detach(), 'gate': gate.detach(),
        'distances': distances.detach(), 'source_distances': source_distances.detach(),
        'distance_scale': scale.detach(), 'coverage_threshold': threshold.detach(),
        'nearest_scaled_distance': nearest_target.detach(), 'ess': ess.detach(),
        'statistics': {'uninformative_signature': False, 'n_source': n_source,
                       'n_target': n_target, 'distance_scale': float(scale),
                       'coverage_threshold': float(threshold),
                       'acceptance_rate': float(gate.double().mean()),
                       'ess_min': float(ess.min()), 'ess_mean': float(ess.mean())},
    }


def permute_weights(weights, generator):
    """Permute source attribution independently per target, preserving masses."""
    weights = _detached_float64(weights, 'weights')
    if weights.ndim != 2 or min(weights.shape) <= 0:
        raise ValueError('weights must have shape [n_source, n_target] with positive sizes')
    if bool((weights < 0).any()) or not torch.allclose(
            weights.sum(dim=0), torch.ones_like(weights.sum(dim=0)), atol=1e-10, rtol=1e-10):
        raise ValueError('each weight column must be nonnegative and sum to one')
    if not isinstance(generator, torch.Generator):
        raise ValueError('generator must be an explicit torch.Generator')
    permutations = [torch.randperm(weights.shape[0], generator=generator, device=generator.device)
                    .to(weights.device) for _ in range(weights.shape[1])]
    return torch.stack([weights[order, j] for j, order in enumerate(permutations)], dim=1).detach()
