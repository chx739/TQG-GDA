"""Exact finite-update risks and a finite-candidate convex master problem.

Only the outer problem is convex. A local explanation search does not certify
the worst risk over the complete explanation set, or the true target risk.
Matrices stay in [class, feature] form; flattening is an implementation detail.
SciPy is imported only when optimization is requested.
"""
from dataclasses import asdict, is_dataclass
from copy import deepcopy
import math

import numpy as np


def _array(value, name, ndim):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _inputs(source_logits, features):
    z = _array(source_logits, "source_logits", 2)
    u = _array(features, "features", 2)
    if not len(z) or z.shape[1] < 2 or u.shape[0] != z.shape[0] or not u.shape[1]:
        raise ValueError("Expected matching nonempty target rows and at least two classes")
    return z, u


def _lse(values):
    maximum = values.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(values - maximum).sum(axis=1))


def _partition_terms(z, u, correction):
    source_offset = z.max(axis=1, keepdims=True)
    delta = u @ correction.T
    # TwoSum keeps the rounding residual until the common row offset is removed.
    total = z + delta
    rounded_delta = total - z
    residual = (z - (total - rounded_delta)) + (delta - rounded_delta)
    total_offset = total.max(axis=1, keepdims=True)
    updated = (total - total_offset) + residual
    reference = updated.argmax(axis=1)
    offset = updated[np.arange(len(z)), reference]
    centered = updated - offset[:, None]
    log_norm = _lse(centered)
    baseline_norm = _lse(z - source_offset)
    probabilities = np.exp(centered - log_norm[:, None])
    gradient = probabilities.T @ u / len(u)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("Non-finite log-partition gradient")
    offset_change = (total_offset - source_offset)[:, 0] + offset
    return reference, source_offset[:, 0], log_norm - baseline_norm, offset_change, gradient


def _partition(z, u, correction):
    _, _, norm_change, offset_change, gradient = _partition_terms(z, u, correction)
    change = float(np.mean(offset_change + norm_change))
    if not math.isfinite(change):
        raise FloatingPointError("Non-finite log-partition change")
    return change, gradient


def log_partition_change(source_logits, features, correction):
    """Return A(W), exactly for the specified finite affine logit update."""
    z, u = _inputs(source_logits, features)
    w = _array(correction, "correction", 2)
    if w.shape != (z.shape[1], u.shape[1]):
        raise ValueError("Correction has incompatible class/feature dimensions")
    return _partition(z, u, w)[0]


def _center_correction(w):
    # Relative cross-entropy is unchanged by adding the same feature
    # weights to every class. Remove the shared same-sign component before
    # matrix multiplication so it cannot mask smaller class differences.
    lower, upper = w.min(axis=0), w.max(axis=0)
    shared = np.clip(0., lower, upper)
    # Sterbenz's bound makes the subtraction exact for every class. An
    # inexact shift can erase a small class gap after adding source logits.
    exact = np.maximum(np.abs(lower), np.abs(upper)) * 0.5 <= np.abs(shared)
    return w - np.where(exact, shared, 0.)


def _risk_terms(z, u, w, statistics):
    centered = _center_correction(w)
    reference, source_offset, norm_change, _, gradient = _partition_terms(z, u, centered)
    rows = np.arange(len(z))
    one_hot = np.zeros_like(z)
    one_hot[rows, reference] = 1.
    reference_statistic = one_hot.T @ u / len(u)
    # The reference class's correction is already in A(W). Combine it
    # with S before taking the inner product, avoiding two large scalars.
    moment_change = np.einsum("kcd,cd->k", reference_statistic - statistics, centered)
    risks = (np.mean(z[rows, reference] - source_offset) + moment_change) + np.mean(norm_change)
    if not np.isfinite(risks).all():
        raise FloatingPointError("Non-finite relative risk")
    return risks, gradient


def _statistics(statistics, classes, features):
    stats = _array(statistics, "statistics", 3)
    if not len(stats) or stats.shape[1:] != (classes, features.shape[1]):
        raise ValueError("Expected at least one statistic with shape [classes, features]")
    # Probability-weighted class moments must sum to the unlabeled moment.
    tolerance = 1e-10 * np.maximum(1., np.abs(features).mean(axis=0))
    if (np.abs(stats.sum(axis=1) - features.mean(axis=0)) > tolerance).any():
        raise ValueError("Class statistics must sum to the mean target features")
    return stats


def relative_risks(source_logits, features, correction, statistics):
    """Return A(W)-<W,S_k> for all retained conditional-label statistics."""
    z, u = _inputs(source_logits, features)
    w = _array(correction, "correction", 2)
    if w.shape != (z.shape[1], u.shape[1]):
        raise ValueError("Correction has incompatible dimensions")
    stats = _statistics(statistics, z.shape[1], u)
    return _risk_terms(z, u, w, stats)[0]


def _kkt(gradient, stats, risks, epigraph, regularization, w,
         feasibility_tol, optimality_tol, minimize):
    """Find simplex multipliers on near-active constraints, then check KKT."""
    slack = epigraph - risks
    active = np.flatnonzero(slack <= max(10 * feasibility_tol, 1e-8))
    if not len(active):
        active = np.array([int(np.argmax(risks))])
    columns = stats[active].reshape(len(active), -1).T
    target = (gradient + regularization * w).ravel()
    residual_columns = columns - target[:, None]
    single = int(np.argmin(np.abs(residual_columns).max(axis=0)))
    weights = np.zeros(len(active))
    weights[single] = 1.
    scale = max(1., float(np.max(np.abs(target))), float(np.max(np.abs(columns))))
    if len(active) == 1 or np.max(np.abs(residual_columns[:, single])) <= optimality_tol * scale:
        multiplier_success = True
    else:
        # Center and scale before fitting simplex multipliers. With nearly
        # identical cuts the unscaled least squares can stop at its start.
        normalized = residual_columns / np.max(np.abs(residual_columns))

        def objective(weight):
            residual = normalized @ weight
            return .5 * float(residual @ residual), normalized.T @ residual

        fit = minimize(objective, weights, jac=True,
                       method="SLSQP", bounds=[(0., 1.)] * len(active),
                       constraints={"type": "eq", "fun": lambda v: v.sum() - 1,
                                    "jac": lambda v: np.ones_like(v)},
                       options={"ftol": 1e-14, "maxiter": 1000})
        weights = np.asarray(fit.x, dtype=np.float64)
        multiplier_success = bool(fit.success)
    mass_error = float(weights.sum()) - 1.
    stationarity = float(np.max(np.abs(residual_columns @ weights + mass_error * target)))
    complementarity = float(np.max(np.abs(weights * slack[active])))
    simplex_error = max(abs(mass_error), float(np.maximum(-weights, 0).max()))
    valid = (math.isfinite(stationarity) and stationarity <= optimality_tol * scale
             and simplex_error <= feasibility_tol
             and complementarity <= max(feasibility_tol, optimality_tol))
    return dict(valid=bool(valid), stationarity_residual=stationarity,
                stationarity_tolerance=optimality_tol * scale,
                complementarity_residual=complementarity, simplex_residual=simplex_error,
                multiplier_solver_success=multiplier_success,
                active_indices=active.tolist(), active_weights=weights.tolist())


def solve_master(source_logits, features, statistics, *, regularization,
                 feasibility_tol=1e-8, optimality_tol=1e-6,
                 master_maxiter=1000, initial=None):
    """Solve the convex epigraph master; return a candidate and audit metadata.

    A candidate with info['valid'] == False MUST NOT be deployed. The enclosing
    solve_shared_update function replaces such a candidate with the zero update.
    """
    from scipy.optimize import minimize

    z, u = _inputs(source_logits, features)
    shape = (z.shape[1], u.shape[1])
    stats = _statistics(statistics, z.shape[1], u)
    if (not math.isfinite(regularization) or regularization <= 0
            or not 0 < feasibility_tol < 1 or not 0 < optimality_tol < 1
            or master_maxiter < 1):
        raise ValueError("Invalid regularization, tolerance or master iteration budget")
    w0 = np.zeros(shape) if initial is None else _array(initial, "initial", 2)
    if w0.shape != shape:
        raise ValueError("Initial correction has incompatible dimensions")
    # Bound the magnitudes seen by SLSQP while optimizing the same logits
    # and the same L2 penalty on the original coefficients.
    feature_scales = np.maximum(1., np.abs(u).max(axis=0))
    scaled_u, scaled_stats = u / feature_scales, stats / feature_scales
    initial_scaled = w0 * feature_scales
    risk0 = _risk_terms(z, scaled_u, initial_scaled, scaled_stats)[0]
    x0 = np.r_[initial_scaled.ravel(), risk0.max()]
    flattened = scaled_stats.reshape(len(stats), -1)
    inverse_scale = np.broadcast_to(1. / feature_scales, shape).ravel()

    def objective(vector):
        w = vector[:-1] * inverse_scale
        return (float(vector[-1] + regularization * (w @ w) / 2),
                np.r_[regularization * w * inverse_scale, 1.])

    def constraints(vector):
        return vector[-1] - _risk_terms(z, scaled_u, vector[:-1].reshape(shape), scaled_stats)[0]

    def constraint_jacobian(vector):
        _, gradient = _risk_terms(z, scaled_u, vector[:-1].reshape(shape), scaled_stats)
        return np.column_stack((flattened - gradient.ravel(), np.ones(len(stats))))

    scaled_optimality_tol = optimality_tol / feature_scales.max()
    result = minimize(objective, x0, jac=True, method="SLSQP",
                      constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
                      options={"ftol": max(1e-14, min(1e-12, feasibility_tol * .001, scaled_optimality_tol ** 2)),
                               "maxiter": int(master_maxiter)})
    if not np.isfinite(result.x).all():
        return np.zeros(shape), dict(valid=False, reason="nonfinite_master_solution",
                                     solver_success=bool(result.success), solver_status=int(result.status),
                                     solver_message=str(result.message))
    w = result.x[:-1].reshape(shape) / feature_scales
    risks, gradient = _risk_terms(z, u, w, stats)
    epigraph = float(result.x[-1])
    worst = float(risks.max())
    penalty = regularization * float(np.sum(w * w)) / 2
    violation = max(0., worst - epigraph)
    kkt = _kkt(gradient, stats, risks, epigraph, regularization, w,
               feasibility_tol, optimality_tol, minimize)
    checks = dict(feasible=violation <= feasibility_tol, stationary=kkt["valid"],
                  no_worse_than_zero=worst + penalty <= feasibility_tol)
    failures = [name for name, passed in checks.items() if not passed]
    solved_reason = "solved" if result.success else "verified_after_solver_exit"
    return w, dict(valid=not failures, reason="+".join(failures) if failures else solved_reason,
                   solver_success=bool(result.success), solver_status=int(result.status), solver_message=str(result.message),
                   iterations=int(result.nit), candidates=len(stats), **checks,
                   epigraph=epigraph, constraint_violation=violation,
                   known_candidate_risks=risks.tolist(), found_worst_risk=worst,
                   regularization_penalty=penalty, objective=worst + penalty, kkt=kkt,
                   feature_scales=feature_scales.tolist())


def _json(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot serialize explanation evidence of type {type(value).__name__}")


def solve_shared_update(source_logits, features, explanations, config):
    """Fit a fixed explanation set, solve masters, and search for violations.

    source_logits are fixed *target* logits produced by the source classifier.
    Search starts are Explanation objects. Target class labels are never inputs.
    Search-budget exhaustion retains only a finite-candidate guarantee; it is
    explicitly not a certificate for the complete explanation set.
    """
    z, u = _inputs(source_logits, features)
    zero = np.zeros((z.shape[1], u.shape[1]), dtype=np.float64)
    regularization = float(config.regularization)
    feasibility_tol = float(config.feasibility_tol)
    separation_tol = float(config.separation_tol)
    optimality_tol = float(getattr(config, "optimality_tol", 1e-6))
    rounds = int(config.search_rounds)
    if (regularization <= 0 or not math.isfinite(regularization) or rounds < 0
            or separation_tol < 0 or not math.isfinite(separation_tol)
            or not 0 < feasibility_tol < 1 or not 0 < optimality_tol < 1):
        raise ValueError("Invalid shared-update optimization configuration")
    candidates, history = [], []
    info = dict(status="fallback", stop_reason=None, iterations=history,
                full_set_certified=False, true_target_risk_certified=False,
                guarantee_scope="retained_feasible_explanations_only",
                reference_loss=None, threshold=None, regularization=regularization)
    numerical_errors = (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError)

    def finish(w, reason, fallback=False):
        if fallback:
            w = zero.copy()
        stats = np.stack([c.statistic for c in candidates]) if candidates else None
        risks = relative_risks(z, u, w, stats).tolist() if candidates else []
        penalty = regularization * float(np.sum(w * w)) / 2
        info.update(status="fallback" if fallback else "completed", stop_reason=reason,
                    used_zero_fallback=fallback, correction_norm=float(np.linalg.norm(w)),
                    candidates=[dict(statistic=_json(c.statistic), observation_loss=float(c.observation_loss),
                                     parameters=_json(c.parameters)) for c in candidates],
                    candidate_count=len(candidates), known_candidate_risks=risks,
                    found_worst_risk=max(risks) if risks else None,
                    final_regularized_objective=max(risks) + penalty if risks else None)
        return w, _json(info)

    try:
        reference = explanations.fit()
        reference_loss = float(explanations.reference_loss)
        threshold = float(explanations.threshold)
        if not math.isfinite(reference_loss) or not math.isfinite(threshold) or threshold < reference_loss:
            raise ValueError("Invalid fixed observation reference/threshold")
        info.update(reference_loss=reference_loss, threshold=threshold,
                    reference_fit_diagnostics=_json(getattr(explanations, "fit_report", None)))
    except numerical_errors as error:
        info["error"] = str(error)
        return finish(zero, "reference_fit_failed", True)

    def validate(candidate):
        if candidate is None:
            raise ValueError("No explanation returned")
        _statistics([candidate.statistic], z.shape[1], u)
        loss = float(candidate.observation_loss)
        if not math.isfinite(loss) or loss > threshold + feasibility_tol:
            raise ValueError("Explanation is infeasible or has invalid dimensions")
        _json(candidate.parameters)

    try:
        validate(reference)
    except (TypeError, *numerical_errors) as error:
        info["error"] = str(error)
        return finish(zero, "invalid_reference", True)
    candidates.append(deepcopy(reference))
    w = zero.copy()
    stop_reason = "search_budget_exhausted"
    # An extra master solve after the last search is mandatory: the last found
    # violating statistic must constrain the returned correction as well.
    for iteration in range(rounds + 1):
        entry = dict(iteration=iteration, candidate_count=len(candidates))
        history.append(entry)
        try:
            w, master = solve_master(z, u, np.stack([c.statistic for c in candidates]),
                                     regularization=regularization, feasibility_tol=feasibility_tol,
                                     optimality_tol=optimality_tol, master_maxiter=config.master_maxiter,
                                     initial=w)
            entry["master"] = master
        except numerical_errors as error:
            entry["error"] = str(error)
            return finish(zero, "master_exception", True)
        if not master["valid"]:
            return finish(zero, "invalid_master", True)
        if iteration == rounds:
            break
        try:
            found = explanations.search(u @ _center_correction(w).T, deepcopy(candidates))
            entry["search_diagnostics"] = _json(getattr(explanations, "last_search", None))
            if (float(explanations.reference_loss) != reference_loss
                    or float(explanations.threshold) != threshold):
                raise ValueError("Explanation search changed the fixed observation set")
            if found is None:
                entry["search"] = "no_feasible_candidate_returned"
                stop_reason = "search_returned_no_candidate"
                break
            validate(found)
        except (TypeError, *numerical_errors) as error:
            entry["error"] = str(error)
            return finish(zero, "invalid_search_result", True)
        risk = float(relative_risks(z, u, w, np.asarray([found.statistic]))[0])
        violation = risk - master["epigraph"]
        entry.update(search_observation_loss=float(found.observation_loss),
                     search_risk=risk, separation_violation=violation,
                     searched_statistic=_json(found.statistic))
        # Exact duplicate statistics introduce no new constraint. Distinct
        # feasible results are retained even if the current violation is small.
        duplicate = any(np.array_equal(found.statistic, c.statistic) for c in candidates)
        entry["duplicate_statistic"] = duplicate
        if not duplicate:
            candidates.append(deepcopy(found))
        if violation <= separation_tol or duplicate:
            diagnostics = entry["search_diagnostics"] or {}
            stop_reason = ("local_search_incomplete" if diagnostics.get("status") == "local_search_failed"
                           else "no_significant_violation_found")
            if not duplicate:
                try:
                    w, final_master = solve_master(z, u, np.stack([c.statistic for c in candidates]),
                                                   regularization=regularization, feasibility_tol=feasibility_tol,
                                                   optimality_tol=optimality_tol, master_maxiter=config.master_maxiter,
                                                   initial=w)
                    entry["final_master"] = final_master
                    if not final_master["valid"]:
                        return finish(zero, "invalid_final_master", True)
                except numerical_errors as error:
                    entry["error"] = str(error)
                    return finish(zero, "final_master_exception", True)
            break
    risks = relative_risks(z, u, w, np.stack([c.statistic for c in candidates]))
    objective = float(risks.max() + regularization * np.sum(w * w) / 2)
    if not math.isfinite(objective) or objective > feasibility_tol:
        return finish(zero, "final_zero_comparison_failed", True)
    return finish(w, stop_reason)
