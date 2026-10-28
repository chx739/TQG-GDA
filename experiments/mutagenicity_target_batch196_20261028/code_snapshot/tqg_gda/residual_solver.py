"""Detached float64 residual regression and Euclidean simplex projection."""

import math

import torch


def _tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.is_complex():
        raise ValueError(f'{name} must be a real tensor')
    value = value.detach().to(dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} contains a nonfinite value')
    return value


def _regression_inputs(phi, residual, ridge):
    phi, residual = _tensor(phi, 'phi'), _tensor(residual, 'residual')
    if phi.ndim != 3 or residual.ndim != 2 or phi.shape[:2] != residual.shape:
        raise ValueError('expected phi [n_graphs, n_classes, n_columns] and residual [n_graphs, n_classes]')
    if phi.shape[0] <= 0 or phi.shape[1] <= 0:
        raise ValueError('regression needs nonempty graph and class axes')
    if phi.device != residual.device:
        raise ValueError('phi and residual must share a device')
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError('ridge must be finite and positive')
    return phi, residual


def uniform_ridge(phi, residual, ridge):
    """Minimize mean_graph ||phi_i a-r_i||² + ridge ||a||²."""
    phi, residual = _regression_inputs(phi, residual, ridge)
    n, _, columns = phi.shape
    if columns == 0:
        return phi.new_empty((0,))
    gram = torch.einsum('ncm,nck->mk', phi, phi) / n
    cross = torch.einsum('ncm,nc->m', phi, residual) / n
    answer = torch.linalg.solve(gram + ridge * torch.eye(columns, dtype=phi.dtype, device=phi.device), cross)
    if not bool(torch.isfinite(answer).all()):
        raise ValueError('ridge solution is nonfinite')
    return answer.detach()


def weighted_ridge(phi, residual, weights, ridge):
    """One local regression per target; columns of weights already sum to 1.

    Return [n_target, n_columns]. No additional division by n_source is made.
    """
    phi, residual = _regression_inputs(phi, residual, ridge)
    weights = _tensor(weights, 'weights')
    if weights.ndim != 2 or weights.shape[0] != phi.shape[0] or weights.shape[1] <= 0:
        raise ValueError('weights must have shape [n_source, n_target]')
    if phi.device != weights.device:
        raise ValueError('weights and phi must share a device')
    if bool((weights < 0).any()) or not torch.allclose(
            weights.sum(0), torch.ones_like(weights.sum(0)), atol=1e-10, rtol=1e-10):
        raise ValueError('weights must be nonnegative and sum to one for each target')
    columns = phi.shape[-1]
    if columns == 0:
        return phi.new_empty((weights.shape[1], 0))
    gram = torch.einsum('nt,ncm,nck->tmk', weights, phi, phi)
    cross = torch.einsum('nt,ncm,nc->tm', weights, phi, residual)
    regularizer = ridge * torch.eye(columns, dtype=phi.dtype, device=phi.device)
    answer = torch.linalg.solve(gram + regularizer, cross.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(answer).all()):
        raise ValueError('weighted ridge solution is nonfinite')
    return answer.detach()


def select_ridge_cv(phi, residual, fold_permutation, candidates=(.01, .1, 1.), tie_tol=1e-8):
    """Four-fold source CV using independently supplied graph-index ordering.

    Unequal fold sizes (n not divisible by four) are pooled by graph count,
    giving every held-out graph equal weight. Each training fold's solve uses
    its own n_fit normalization, exactly as uniform_ridge specifies.
    """
    if not candidates:
        raise ValueError('candidates must be nonempty')
    candidates = tuple(float(value) for value in candidates)
    phi, residual = _regression_inputs(phi, residual, candidates[0])
    if any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError('ridge candidates must be finite and positive')
    if len(set(candidates)) != len(candidates):
        raise ValueError('ridge candidates must be unique')
    if not math.isfinite(tie_tol) or tie_tol < 0:
        raise ValueError('tie_tol must be finite and nonnegative')
    n = phi.shape[0]
    if n < 4:
        raise ValueError('four-fold CV requires at least four source graphs')
    if not isinstance(fold_permutation, torch.Tensor) or fold_permutation.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('fold_permutation must be an integer tensor')
    order = fold_permutation.detach().to(device=phi.device, dtype=torch.long)
    if order.ndim != 1 or order.shape[0] != n or not torch.equal(
            order.sort().values, torch.arange(n, device=phi.device)):
        raise ValueError('fold_permutation must contain each source position exactly once')
    folds = torch.tensor_split(order, 4)
    scores = {}
    for ridge in candidates:
        total = phi.new_zeros(())
        for fold_index, audit in enumerate(folds):
            fit = torch.cat([fold for index, fold in enumerate(folds) if index != fold_index])
            coefficients = uniform_ridge(phi[fit], residual[fit], ridge)
            total += (phi[audit] @ coefficients - residual[audit]).square().sum()
        score = float(total / n)
        if not math.isfinite(score):
            raise ValueError('CV score is nonfinite')
        scores[ridge] = score
    minimum = min(scores.values())
    selected = max(ridge for ridge, score in scores.items() if score <= minimum + tie_tol)
    return selected, scores


def simplex_projection(value):
    """Euclidean projection onto the probability simplex along the last axis."""
    value = _tensor(value, 'value')
    if value.ndim < 1 or value.shape[-1] <= 0 or value.numel() <= 0:
        raise ValueError('simplex projection needs a nonempty final class axis')
    # Translation invariance allows subtraction of max, reducing overflow for
    # finite inputs that have a large common logit offset.
    translated = value - value.amax(dim=-1, keepdim=True)
    if not bool(torch.isfinite(translated).all()):
        raise ValueError('simplex projection translation is nonfinite')
    ordered = translated.sort(dim=-1, descending=True).values
    partial = ordered.cumsum(dim=-1) - 1
    positions = torch.arange(1, value.shape[-1] + 1, dtype=value.dtype, device=value.device)
    active = ordered - partial / positions > 0
    count = active.sum(dim=-1, keepdim=True)
    threshold = partial.gather(-1, count - 1) / count
    answer = (translated - threshold).clamp_min(0)
    if not bool(torch.isfinite(answer).all()):
        raise ValueError('simplex projection is nonfinite')
    return answer.detach()


def project_residual(probabilities, correction):
    """Project p+d onto the simplex and return its residual relative to p."""
    probabilities = _tensor(probabilities, 'probabilities')
    correction = _tensor(correction, 'correction')
    if probabilities.shape != correction.shape or probabilities.device != correction.device:
        raise ValueError('probabilities and correction must share shape and device')
    if probabilities.ndim < 1 or probabilities.shape[-1] <= 0 or probabilities.numel() <= 0:
        raise ValueError('probabilities need a nonempty class axis')
    if bool((probabilities < 0).any()) or not torch.allclose(
            probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=1e-10, rtol=1e-10):
        raise ValueError('probabilities must be nonnegative and sum to one')
    return (simplex_projection(probabilities + correction) - probabilities).detach()
