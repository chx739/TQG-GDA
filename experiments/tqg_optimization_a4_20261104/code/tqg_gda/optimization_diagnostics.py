"""Label-boundary-safe numerical components for the bounded optimization study."""
from dataclasses import dataclass
import numpy as np
import torch

from .assumption_diagnostics import metrics


def bandwidth_kernel(distance, multiplier):
    if multiplier <= 0 or not np.isfinite(distance).all():
        raise ValueError('Invalid distance or bandwidth')
    return np.exp(-np.maximum(distance, 0)/(2*multiplier**2))


def effective_lambda(base, count, reference, convention):
    if min(base, count, reference) <= 0:
        raise ValueError('Positive regularization and sample sizes required')
    if convention == 'FIXED':
        return float(base)
    if convention == 'NORMALIZED':
        return float(base)*count/reference
    raise ValueError(convention)


def kernel_factor(kernel):
    values, vectors = np.linalg.eigh((kernel+kernel.T)/2)
    if values.min() < -1e-8:
        raise ValueError('Indefinite kernel')
    return values.clip(0), vectors


@dataclass
class RidgeFit:
    probabilities: np.ndarray  # [label variant, lambda, evaluation sample]
    alpha: np.ndarray          # probability residual coefficients [variant, lambda, train]
    eigenvalues: np.ndarray
    effective_dimensions: np.ndarray
    rkhs_squared: np.ndarray   # probability-coordinate norm; multiply by ||Q1-Q0||^2 for contrast


def ridge_grid(kernel, evaluation_kernel, labels, train_probability, evaluation_probability,
               penalties, factor=None):
    """Evaluation labels are deliberately absent from the fitting interface."""
    y = np.asarray(labels, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(kernel) or np.any(penalties <= 0):
        raise ValueError('Invalid regression dimensions or regularization')
    values, vectors = kernel_factor(kernel) if factor is None else factor
    residual = y-np.asarray(train_probability)[None, :]
    projected = residual@vectors
    scaled = projected[:, None, :]/(values[None, None, :]+penalties[None, :, None])
    alpha = scaled@vectors.T
    probabilities = np.asarray(evaluation_probability)[None, None, :]+alpha@evaluation_kernel.T
    if not np.isfinite(probabilities).all():
        raise FloatingPointError('Nonfinite ridge predictions')
    return RidgeFit(probabilities, alpha, values,
        (values[None, :]/(values[None, :]+penalties[:, None])).sum(1),
        (scaled**2*values[None, None, :]).sum(2))


def choose_kernel(scores, multipliers, penalties):
    """scores[bandwidth, lambda]; exact ties favor stronger/wider regularization."""
    if not np.isfinite(scores).all():
        raise ValueError('All CV scores must be finite')
    return min(np.ndindex(scores.shape), key=lambda ij: (scores[ij], -penalties[ij[1]], -multipliers[ij[0]]))


def mean_readout(kernel, cross_kernel, target_response, penalty):
    """Binary specialization, used only where q=1 is asserted."""
    if target_response.shape[1] != 1:
        raise ValueError('Binary reader requires q=1')
    hst = cross_kernel@target_response[:, 0, :]/len(target_response)
    return np.linalg.solve(kernel+penalty*np.eye(len(kernel)), hst).T


def closest_bank(source_ids, cross_kernel, count):
    if not 0 < count <= len(source_ids):
        raise ValueError('Invalid bank size')
    maximum = np.asarray(cross_kernel).max(1)
    order = sorted(range(len(source_ids)), key=lambda i: (-maximum[i], int(source_ids[i])))
    return np.array(sorted(order[:count]), dtype=int)


def check_mean_identity(kss, kst, ktt, target_response, residual, penalty, calibration):
    """Independent multi-class Kronecker oracle, not the production mean solver."""
    from .d34_diagnostics import solve_observed
    q = target_response.shape[1]
    bt = target_response.reshape(-1, target_response.shape[-1])/len(target_response)
    block_k = torch.kron(kss.contiguous(), torch.eye(q, dtype=kss.dtype))
    block_ts = torch.kron(kst.mT.contiguous(), torch.eye(q, dtype=kss.dtype))
    alpha = torch.linalg.solve(block_k+penalty*torch.eye(len(block_k), dtype=kss.dtype), residual)
    direct = bt.mT@(block_ts@alpha)
    w, _ = solve_observed((kss, kst, ktt), target_response, 'MEAN', penalty, calibration)
    iso, _ = solve_observed((kss, kst, ktt), target_response, 'ISO', 2*penalty, calibration)
    return dict(mean_vs_pointwise_max=float((w@residual-direct).abs().max()),
                mean_vs_iso_max=float((w-iso).abs().max()))
