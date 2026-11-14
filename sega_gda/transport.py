"""§3.7: hard TARGET columns, KL-relaxed SOURCE rows; no OT differentiation."""
from dataclasses import dataclass, asdict
import math
import torch


def _parameters(epsilon, rho):
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(rho) or rho < 0:
        raise ValueError('Require finite epsilon > 0 and rho >= 0')


def _problem(cost, a, b):
    if (cost.ndim != 2 or cost.shape != (a.numel(), b.numel()) or a.ndim != 1
            or b.ndim != 1 or min(cost.shape) < 1):
        raise ValueError('Expected nonempty cost [source,target], a [source], b [target]')
    if any(t.device != cost.device or not t.is_floating_point() for t in (cost, a, b)):
        raise ValueError('Floating tensors must share device')
    if not all(bool(torch.isfinite(t).all()) for t in (cost, a, b)) or (cost < 0).any():
        raise ValueError('Nonfinite input or negative squared-distance cost')
    if (a <= 0).any() or (b <= 0).any():
        raise ValueError('Reference masses must be strictly positive')
    tol = max(1e-12, 10 * max(torch.finfo(t.dtype).eps for t in (a, b)))
    if abs(float(a.sum())-1) > tol or abs(float(b.sum())-1) > tol:
        raise ValueError('Reference masses must each sum to one')


def generalized_kl(x, reference):
    """Zero mass has its exact limiting contribution, without adding pseudo-mass."""
    if (not torch.isfinite(x).all() or not torch.isfinite(reference).all()
            or (x < 0).any() or (reference <= 0).any()):
        raise ValueError('Invalid KL masses')
    return (torch.special.xlogy(x, x) - x * reference.log() - x + reference).sum()


def ot_objective(pi, cost, a, b, epsilon, rho):
    """Full F at a fixed Pi (KL terms are constants for model differentiation)."""
    _parameters(epsilon, rho)
    _problem(cost, a, b)
    if pi.shape != cost.shape or pi.requires_grad:
        raise ValueError('Pi must be a detached matrix of the cost shape')
    return _objective_terms(pi, cost, a, b, epsilon, rho)


def _objective_terms(pi, cost, a, b, epsilon, rho):
    transport = (pi * cost).sum()
    entropy = generalized_kl(pi, a[:, None] * b[None, :])
    marginal = generalized_kl(pi.sum(1), a)
    total = transport + epsilon * entropy + rho * marginal
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite full OT objective')
    return dict(total=total, transport=transport, entropy_kl=entropy, source_kl=marginal)


def class_balanced_masses(graph_ids, labels, classes, *, dtype=torch.float64, device='cpu'):
    """Only source-TRAIN labels. Each reference ID is unique and every class occurs."""
    if not graph_ids or len(graph_ids) != len(labels) or len(set(graph_ids)) != len(graph_ids):
        raise ValueError('Source references need distinct IDs and aligned labels')
    if not classes or len(set(classes)) != len(classes) or set(labels) != set(classes):
        raise ValueError('Reference set must represent exactly the source train classes')
    counts = {c: labels.count(c) for c in classes}
    return torch.tensor([1 / (len(classes)*counts[c]) for c in labels], dtype=dtype, device=device)


def sample_source_references(source_train_labels, per_class, seed):
    """Uniform without replacement per class, with an independent local RNG."""
    if not source_train_labels or not isinstance(per_class, int) or per_class < 1:
        raise ValueError('Need source train ID -> label map and positive class budget')
    generator = torch.Generator().manual_seed(seed)
    ids, labels = [], []
    for c in sorted(set(source_train_labels.values())):
        pool = sorted(g for g, y in source_train_labels.items() if y == c)
        chosen = torch.randperm(len(pool), generator=generator)[:per_class].tolist()
        ids.extend(pool[i] for i in chosen)
        labels.extend([c]*len(chosen))
    return tuple(ids), tuple(labels)


def uniform_target_masses(count, *, dtype=torch.float64, device='cpu'):
    if not isinstance(count, int) or count < 1:
        raise ValueError('Target input batch must be nonempty')
    return torch.full((count,), 1/count, dtype=dtype, device=device)


@dataclass(frozen=True)
class OTTolerances:
    """Tiny float64 development tolerances, NOT a calibrated production preset."""
    max_iter: int = 5000
    column_abs: float = 1e-10
    column_rel: float = 1e-10
    stationarity_abs: float = 1e-7
    stationarity_rel: float = 1e-8
    iterate: float = 1e-9
    objective: float = 1e-11
    export_mass: float = 1e-10
    export_objective_abs: float = 1e-8
    export_objective_rel: float = 1e-7


@dataclass
class OTResult:
    pi: torch.Tensor
    log_pi: torch.Tensor
    converged: bool
    status: str
    iterations: int
    diagnostics: dict
    epsilon: float
    rho: float


class OTConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f'Semi-relaxed OT {result.status}: {result.diagnostics}')


def fixed_plan_objective(result, cost, a, b):
    """Unconverged correspondence cannot silently enter a model update."""
    if not result.converged:
        raise OTConvergenceError(result)
    return ot_objective(result.pi, cost, a.detach(), b.detach(), result.epsilon, result.rho)


def _log_objective(ell, cost, la, lb, epsilon, rho):
    pi = ell.exp()
    logm = torch.logsumexp(ell, 1)
    entropy = (pi*(ell-la[:, None]-lb[None, :])-pi).sum() + la.exp().sum()*lb.exp().sum()
    marginal = (logm.exp()*(logm-la)-logm.exp()+la.exp()).sum()
    return (pi*cost).sum()+epsilon*entropy+rho*marginal


@torch.no_grad()
def solve_semi_relaxed(cost, a, b, epsilon, rho, tolerances=None, *, raise_on_failure=True):
    """Log-space scaling on stopped cost, float64 internals, explicit export checks.

    Returned Pi uses the requested cost dtype/device. No row renormalization,
    cost clipping, fallback distance, retry, or gradients through OT iterations.
    """
    _parameters(epsilon, rho)
    _problem(cost, a, b)
    tol = tolerances or OTTolerances()
    if not isinstance(tol.max_iter, int) or tol.max_iter < 1 or any(
            not math.isfinite(v) or v <= 0 for k, v in asdict(tol).items() if k != 'max_iter'):
        raise ValueError('Invalid OT tolerances')
    c, a64, b64 = (t.detach().to(torch.float64) for t in (cost, a, b))
    la, lb = a64.log(), b64.log()
    kernel = la[:, None]+lb[None, :]-c/epsilon
    if not torch.isfinite(kernel).all():
        raise FloatingPointError('Cost/epsilon exceeds float64 log-space range')
    logc = torch.zeros_like(lb)
    old_ell, old_f, stable = None, None, 0
    converged, status = False, 'MAX_ITER'
    for iteration in range(1, tol.max_iter+1):
        logr = (rho/(rho+epsilon))*(la-torch.logsumexp(kernel+logc[None, :], 1)) if rho else torch.zeros_like(la)
        logc = lb-torch.logsumexp(kernel+logr[:, None], 0)
        ell = logr[:, None]+kernel+logc[None, :]
        if not torch.isfinite(ell).all():
            raise FloatingPointError('Nonfinite log-space OT iterate')
        logm, logcol = torch.logsumexp(ell, 1), torch.logsumexp(ell, 0)
        entropy_gradient = epsilon*(ell-la[:, None]-lb[None, :])
        row_gradient = rho*(logm-la) if rho else torch.zeros_like(la)
        h = c+entropy_gradient+row_gradient[:, None]
        stat_abs = float((h-(h.amax(0)+h.amin(0))[None, :]/2).abs().max())
        stat_scale = max(1., float(c.abs().max()), float(entropy_gradient.abs().max()), float(row_gradient.abs().max()))
        f = _log_objective(ell, c, la, lb, epsilon, rho)
        if not torch.isfinite(f) or not math.isfinite(stat_abs):
            raise FloatingPointError('Nonfinite OT objective/residual')
        d_iter = float((ell-old_ell).abs().max()) if old_ell is not None else None
        d_f = float(abs(f-old_f)/max(1., abs(float(f)), abs(float(old_f)))) if old_f is not None else None
        col_abs = float((logcol.exp()-b64).abs().max())
        col_rel = float(torch.expm1(logcol-lb).abs().max())
        residual_ok = (col_abs <= tol.column_abs and col_rel <= tol.column_rel
                       and stat_abs <= tol.stationarity_abs and stat_abs/stat_scale <= tol.stationarity_rel)
        steady = rho == 0 or (d_iter is not None and d_iter <= tol.iterate and d_f <= tol.objective)
        stable = stable+1 if residual_ok and steady else 0
        if stable >= (1 if rho == 0 else 2):
            converged, status = True, 'CONVERGED'
            break
        old_ell, old_f = ell, f
    pi = ell.exp().to(cost.dtype)
    exported = pi.to(torch.float64)
    zeros = pi == 0
    log_lost = float(torch.logsumexp(ell[zeros], 0)) if zeros.any() else None
    # Conservative representable bound: never report zero merely due to underflow.
    lost_bound = math.exp(max(log_lost, -700.)) if log_lost is not None else 0.
    rounding = float((exported-ell.exp()).abs().sum())
    export_col_abs = float((exported.sum(0)-b64).abs().max())
    export_col_rel = float(((exported.sum(0)-b64)/b64).abs().max())
    # Per-column bound for both underflow and rounding (total underflow bound is conservative).
    export_mass_rel = float(((exported-ell.exp()).abs().sum(0)/b64).max()) + lost_bound/float(b64.min())
    # Inputs were validated in their ORIGINAL precision. Promotion must not
    # reinterpret a float32 third as a float64-defined probability mass.
    export_f = _objective_terms(exported, c, a64, b64, epsilon, rho)['total']
    f_error = float(abs(export_f-f))
    export_ok = (rounding+lost_bound <= tol.export_mass and export_mass_rel <= tol.export_mass
                 and export_col_abs <= tol.column_abs and export_col_rel <= tol.column_rel
                 and f_error <= tol.export_objective_abs+tol.export_objective_rel*abs(float(f)))
    if converged and not export_ok:
        converged, status = False, 'EXPORT_FAILED'
    diag = dict(column_abs=col_abs, column_rel=col_rel, stationarity_abs=stat_abs,
                stationarity_rel=stat_abs/stat_scale, iterate_change=d_iter, objective_change=d_f,
                objective_log=float(f), objective_export=float(export_f), export_objective_error=f_error,
                export_column_abs=export_col_abs, export_column_rel=export_col_rel,
                export_mass_error_bound=rounding+lost_bound, export_column_relative_error_bound=export_mass_rel,
                underflow_count=int(zeros.sum()), log_underflow_mass=log_lost,
                underflow_mass_upper_bound=lost_bound, total_mass=float(exported.sum()),
                source_row_l1=float((exported.sum(1)-a64).abs().sum()),
                tolerances=asdict(tol), internal_dtype='float64')
    result = OTResult(pi.detach(), ell.detach(), converged, status, iteration, diag, epsilon, rho)
    if not converged and raise_on_failure:
        raise OTConvergenceError(result)
    return result
