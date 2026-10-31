"""Stopped kernel query calibration from design v2, sections 3.9--3.11.

All solves use FP64. The scalar source kernel is decomposed once; graph-major
class blocks are handled as tensors, without materializing K_SS kron I_q.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CalibrationConfig:
    lambda_w: float = .01
    gap_atol: float = 1e-8
    gap_rtol: float = 1e-6
    max_iterations: int = 200
    initial_step: float = 1.
    max_backtracks: int = 40
    mode: str = 'joint'

    def __post_init__(self):
        for name in ('lambda_w', 'initial_step'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' must be finite and positive')
        for name in ('gap_atol', 'gap_rtol'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if self.gap_atol == self.gap_rtol == 0:
            raise ValueError('At least one gap tolerance must be positive')
        for name in ('max_iterations', 'max_backtracks'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if self.mode not in ('joint', 'mean_only'):
            raise ValueError('Calibration mode must be joint or the mean_only KRR control')


def _sym(matrix):
    return (matrix + matrix.mT) * .5


def _finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError('Nonfinite kernel calibration value')


def _psd_tolerance(matrix):
    scale = max(float(matrix.abs().max()), torch.finfo(matrix.dtype).tiny)
    return 100 * torch.finfo(matrix.dtype).eps * max(matrix.shape) * scale


def check_psd(matrix, name):
    _finite(matrix)
    values = torch.linalg.eigvalsh(_sym(matrix))
    if float(values[0]) < -_psd_tolerance(matrix):
        raise FloatingPointError(name + ' is not positive semidefinite beyond roundoff')


def project_density(matrix):
    """Euclidean projection onto symmetric PSD matrices with trace one."""
    values, vectors = torch.linalg.eigh(_sym(matrix))
    values = values - values.max()  # Simplex projection is invariant to scalar shifts.
    ordered = values.sort(descending=True).values
    cumulative = ordered.cumsum(0) - 1
    indices = torch.arange(1, len(values) + 1, dtype=values.dtype, device=values.device)
    valid = ordered - cumulative / indices > 0
    last = int(torch.nonzero(valid)[-1])
    threshold = cumulative[last] / (last + 1)
    projected = (values - threshold).clamp_min(0)
    return _sym((vectors * projected.unsqueeze(0)) @ vectors.mT)


@torch.no_grad()
def solve_calibration(k_ss, h_st, h_tt, config=CalibrationConfig(), *, return_details=False):
    """Return W[r, b_S*q], diagnostics for the exact v2 objective.

    h_st = (K_ST kron I_q) B_T; h_tt = B_T.T (K_TT kron I_q) B_T.
    The reported primal/dual gap always uses a matched (Omega, W_Omega).
    mean_only removes the spectral term and is explicitly a KRR ablation.
    return_details adds stopped matrices for offline diagnostics only; default
    training results and JSON-compatible statistics are unchanged.
    """
    k_ss, h_st, h_tt = [x.detach().to(dtype=torch.float64) for x in (k_ss, h_st, h_tt)]
    if k_ss.ndim != 2 or k_ss.shape[0] != k_ss.shape[1] or not len(k_ss):
        raise ValueError('Expected a nonempty square scalar source kernel')
    n = len(k_ss)
    if (h_st.ndim != 2 or h_st.shape[0] % n or h_st.shape[0] < n
            or h_tt.shape != (h_st.shape[1], h_st.shape[1]) or not h_st.shape[1]):
        raise ValueError('Incompatible query matrix dimensions')
    _finite(k_ss, h_st, h_tt)
    if any(x.device != k_ss.device for x in (h_st, h_tt)):
        raise ValueError('Calibration matrices must be on one device')
    if not torch.allclose(k_ss, k_ss.mT, atol=_psd_tolerance(k_ss), rtol=0):
        raise FloatingPointError('Asymmetric source kernel')
    if not torch.allclose(h_tt, h_tt.mT, atol=_psd_tolerance(h_tt), rtol=0):
        raise FloatingPointError('Asymmetric target query Gram matrix')
    k_ss, h_tt = _sym(k_ss), _sym(h_tt)
    eigenvalues, source_vectors = torch.linalg.eigh(k_ss)
    if float(eigenvalues[0]) < -_psd_tolerance(k_ss):
        raise FloatingPointError('Source kernel is not positive semidefinite beyond roundoff')
    # Do not silently repair a substantively indefinite matrix.
    eigenvalues = eigenvalues.clamp_min(0)
    r, q = h_st.shape[1], h_st.shape[0] // n
    identity = torch.eye(r, device=k_ss.device, dtype=k_ss.dtype)
    right = torch.einsum('aic,ij->ajc', h_st.mT.reshape(r, n, q), source_vectors)

    def error(w):
        blocks = w.reshape(r, n, q)
        wk = torch.einsum('aic,ij->ajc', blocks, k_ss).reshape(r, n*q)
        cross = w @ h_st
        e = _sym(h_tt - cross - cross.mT + wk @ w.mT)
        _finite(e)
        # Cancellation roundoff is measured against the terms, not only E.
        tolerance = 100 * torch.finfo(e.dtype).eps * max(n*q, r) * max(
            float(h_tt.abs().max()), float(cross.abs().max()),
            float((wk @ w.mT).abs().max()), torch.finfo(e.dtype).tiny)
        if float(torch.linalg.eigvalsh(e)[0]) < -tolerance:
            raise FloatingPointError('Query error Gram matrix is not positive semidefinite')
        return e

    def conditional(omega):
        if omega is None:
            direction_values, direction_vectors = torch.ones(r, dtype=k_ss.dtype, device=k_ss.device), identity
        else:
            direction_values, direction_vectors = torch.linalg.eigh(identity + r * omega)
        transformed = torch.einsum('ia,ibc->abc', direction_vectors, right)
        factors = direction_values[:, None] / (
            direction_values[:, None] * eigenvalues[None, :] + config.lambda_w)
        transformed = transformed * factors[..., None]
        w = torch.einsum('ai,ibc,jb->ajc', direction_vectors, transformed, source_vectors).reshape(r, n*q)
        e = error(w)
        base = e.trace() / r + config.lambda_w * w.square().sum() / r
        dual = base if omega is None else base + (omega * e).sum()
        _finite(w, dual)
        return w, e, base, dual

    if config.mode == 'mean_only':
        w, e, base, _ = conditional(None)
        info = dict(mode=config.mode, converged=True, iterations=1,
            termination='closed_form_krr', gap=0., primal=float(base), dual=float(base),
            mean_error=float(e.trace()/r), max_error=float(torch.linalg.eigvalsh(e)[-1]),
            weight_norm=float(w.norm()), backtracks=0)
        if return_details:
            info['_details'] = dict(omega=None, error=e.detach(), source_eigenvalues=eigenvalues.detach())
        return w, info

    omega = identity / r
    step_size, backtracks = config.initial_step, 0
    termination = 'iteration_limit'
    for iteration in range(1, config.max_iterations + 1):
        w, e, base, dual = conditional(omega)
        maximum = torch.linalg.eigvalsh(e)[-1]
        primal = base + maximum
        gap = (maximum - (omega * e).sum()).clamp_min(0)
        tolerance = config.gap_atol + config.gap_rtol * max(abs(float(primal)), abs(float(dual)))
        if float(gap) <= tolerance:
            termination = 'gap_tolerance'
            break
        if iteration == config.max_iterations:
            break
        accepted = False
        trial_step = step_size
        for _ in range(config.max_backtracks):
            candidate = project_density(omega + trial_step * e)
            displacement = candidate - omega
            _, _, _, candidate_dual = conditional(candidate)
            lower = dual + (e * displacement).sum() - displacement.square().sum()/(2*trial_step)
            rounding = 32 * torch.finfo(k_ss.dtype).eps * max(abs(float(dual)), abs(float(candidate_dual)), 1.)
            if float(candidate_dual) >= float(lower) - rounding:
                omega = candidate
                step_size = trial_step * 2
                accepted = True
                break
            trial_step *= .5
            backtracks += 1
        if not accepted:
            termination = 'line_search_limit'
            break
    # w, e, dual and gap all belong to the same current omega here.
    info = dict(mode=config.mode, converged=termination == 'gap_tolerance',
        iterations=iteration, termination=termination, gap=float(gap),
        primal=float(primal), dual=float(dual), mean_error=float(e.trace()/r),
        max_error=float(maximum), weight_norm=float(w.norm()), backtracks=backtracks)
    if return_details:
        info['_details'] = dict(omega=omega.detach().clone(), error=e.detach(),
                                source_eigenvalues=eigenvalues.detach())
    return w.detach(), info
