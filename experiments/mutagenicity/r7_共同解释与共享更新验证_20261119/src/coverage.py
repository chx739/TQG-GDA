"""Post-seal stage-S model, moment, direction, and marginal-fit diagnostics.

No model is fitted here.  The caller must verify the observation seal before
providing ``truth``.  A retained-candidate hull is an inner approximation of
the full explanation set, so its distances are not coverage certificates.
"""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr, softmax

from tsba.shared_update.explanations import ExplanationSet
from tsba.shared_update.solver import relative_risks


def _array(value, name, ndim=None):
    value = np.asarray(value, dtype=np.float64)
    if (ndim is not None and value.ndim != ndim) or not np.isfinite(value).all():
        raise ValueError(f"{name} has an invalid shape or non-finite values")
    return value


def _moment(probabilities, features):
    probabilities = _array(probabilities, "probabilities", 2)
    if (len(probabilities) != len(features) or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(1), 1., atol=1e-12, rtol=0.)):
        raise ValueError("Posterior rows do not match the target features or probability simplex")
    return probabilities.T @ features / len(features)


def hull_distance(statistic, bank, *, feasibility_tol=1e-8, optimality_tol=1e-6):
    """Frobenius distance to a finite hull, with simplex and optimality checks.

    The Frank-Wolfe gap of the convex squared-distance problem independently
    checks a returned optimizer state.  A failed check reports distance=None;
    an unverified optimizer objective is never presented as hull distance.
    """
    statistic, bank = _array(statistic, "statistic", 2), _array(bank, "bank", 3)
    if not len(bank) or bank.shape[1:] != statistic.shape:
        raise ValueError("Hull requires a nonempty bank matching the statistic")
    differences = (bank - statistic).reshape(len(bank), -1)
    if not np.isfinite(differences).all():
        raise FloatingPointError("Non-finite differences in the convex-hull distance problem")
    scale = float(np.max(np.abs(differences)))
    if not scale:
        return dict(valid=True, distance=0., weights=[1.] + [0.] * (len(bank) - 1),
                    simplex_residual=0., normalized_optimality_gap=0., optimizer_runs=[])
    matrix = differences / scale

    def objective(weights):
        residual = weights @ matrix
        return .5 * float(residual @ residual), matrix @ residual

    nearest = int(np.argmin(np.sum(matrix ** 2, axis=1)))
    starts = [np.eye(1, len(bank), nearest)[0]]
    if len(bank) > 1:
        starts.append(np.full(len(bank), 1 / len(bank)))
    candidates, reports = [], []
    for initial in starts:
        if len(bank) == 1:
            weights, success, status, message = initial, True, 0, "single_vertex"
        else:
            result = minimize(objective, initial, jac=True, method="SLSQP",
                              bounds=[(0., 1.)] * len(bank),
                              constraints={"type": "eq", "fun": lambda a: a.sum() - 1,
                                           "jac": lambda a: np.ones_like(a)},
                              options={"ftol": 1e-14, "maxiter": 1000})
            weights, success, status, message = result.x, bool(result.success), int(result.status), str(result.message)
        if np.isfinite(weights).all():
            mass_error = abs(float(weights.sum()) - 1)
            residual = max(mass_error, float(np.maximum(-weights, 0).max()),
                           float(np.maximum(weights - 1, 0).max()))
            value, gradient = objective(weights)
            gap = max(0., float(weights @ gradient - gradient.min()))
            valid = residual <= feasibility_tol and gap <= optimality_tol
            candidates.append(dict(valid=bool(valid), distance=float(np.linalg.norm(weights @ differences)),
                                   weights=weights.tolist(), simplex_residual=residual,
                                   normalized_optimality_gap=gap, normalized_objective=value))
        else:
            valid = False
        reports.append(dict(success=success, status=status, message=message, verified=bool(valid)))
    verified = [candidate for candidate in candidates if candidate["valid"]]
    if verified:
        answer = min(verified, key=lambda candidate: candidate["distance"])
    else:
        answer = dict(valid=False, distance=None, reason="convex_hull_solution_not_verified",
                      simplex_residual=min((item["simplex_residual"] for item in candidates), default=None),
                      normalized_optimality_gap=min((item["normalized_optimality_gap"] for item in candidates), default=None))
    return dict(answer, coordinate_scale=scale, optimizer_runs=reports)


def _generating_model(truth, mean, scale, classes, dimension):
    raw_means = _array(truth["raw_source_means"], "raw_source_means", 2)
    raw_variances = _array(truth["raw_source_variances"], "raw_source_variances", 2)
    diagonal = _array(truth["raw_L_diagonal"], "raw_L_diagonal", 1)
    raw_shift = _array(truth["raw_b"], "raw_b", 1)
    prior = _array(truth["target_prior"], "target_prior", 1)
    if (raw_means.shape != (classes, dimension) or raw_variances.shape != raw_means.shape
            or diagonal.shape != (dimension,) or raw_shift.shape != (dimension,)
            or prior.shape != (classes,) or np.any(raw_variances <= 0) or np.any(diagonal == 0)
            or np.any(prior <= 0) or not np.isclose(prior.sum(), 1., atol=1e-12, rtol=0.)):
        raise ValueError("Known generating parameters have invalid dimensions or values")
    return dict(means=(raw_means - mean) / scale, variances=raw_variances / scale ** 2,
                diagonal=diagonal, shift=((diagonal - 1) * mean + raw_shift) / scale,
                prior=prior)


def _true_log_components(features, model):
    transformed = features * model["diagonal"] + model["shift"]
    difference = transformed[:, None, :] - model["means"]
    return (-.5 * (np.log(2 * np.pi * model["variances"])
                   + difference ** 2 / model["variances"]).sum(2)
            + np.log(model["prior"]))


def _true_likelihood(source, labels, target, model):
    source_difference = source - model["means"][labels]
    source_log = -.5 * (np.log(2 * np.pi * model["variances"][labels])
                       + source_difference ** 2 / model["variances"][labels]).sum(1)
    target_log = logsumexp(_true_log_components(target, model), axis=1)
    # Reflection has negative determinant, but a valid density uses abs(det).
    target_log += np.log(np.abs(model["diagonal"])).sum()
    value = -float(source_log.sum() + target_log.sum()) / (len(source) + len(target))
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite generating-model observation likelihood")
    return value


def _true_parameters(model, family):
    prior_floor = float(family.config.prior_floor)
    latent = (model["prior"] - prior_floor) / (1 - family.classes * prior_floor)
    if np.any(model["diagonal"] <= 0):
        return None, "generating_map_is_not_positive_diagonal"
    if np.any(latent <= 0):
        return None, "generating_prior_is_not_representable_with_configured_floor"
    parameters = np.concatenate((model["means"].ravel(), np.log(model["variances"]).ravel(),
                                 np.log(model["diagonal"]), model["shift"],
                                 np.log(latent[:-1]) - np.log(latent[-1])))
    return parameters, None


def _cdf_diagnostic(family, reference_parameters, holdout):
    means, log_variances, log_scale, shift, _, prior = family._unpack(reference_parameters)
    scales, deviations = np.exp(log_scale), np.exp(.5 * log_variances)
    distances = []
    count = len(holdout)
    if not count:
        raise ValueError("Holdout CDF diagnostic requires nonempty holdout inputs")
    for coordinate in range(family.dimension):
        values = np.sort(holdout[:, coordinate])
        standardized = ((scales[coordinate] * values[:, None] + shift[coordinate]
                         - means[:, coordinate]) / deviations[:, coordinate])
        model_cdf = ndtr(standardized) @ prior
        right = np.arange(1, count + 1) / count
        left = np.arange(count) / count
        distances.append(float(max(np.max(right - model_cdf), np.max(model_cdf - left))))
    return dict(coordinate_sup=distances, maximum=max(distances), median=float(np.median(distances)),
                samples=count, scope="fixed_reference_marginal_fit_only_not_model_selection")


def boundary_angle(v0, w, truth, standard):
    """Signed-class binary boundary-normal angle in degrees, or None if flat.

    Both normals use the source-standardized coordinates.  Opposite class
    orientations give 180 degrees; absolute cosine would hide semantic flips.
    Intercepts do not enter this directional diagnostic.
    """
    v0, w = _array(v0, "v0", 2), _array(w, "w", 2)
    scale = _array(standard["scale"], "scale", 1)
    raw_target = _array(truth["target_head_weight"], "target_head_weight", 2)
    if (v0.shape != w.shape or v0.shape != (2, len(scale) + 1)
            or raw_target.shape != (2, len(scale)) or np.any(scale <= 0)):
        raise ValueError("Boundary angle requires compatible binary affine heads")
    predicted = (v0 + w)[1, :-1] - (v0 + w)[0, :-1]
    target = (raw_target[1] - raw_target[0]) * scale
    pscale, tscale = float(np.max(np.abs(predicted))), float(np.max(np.abs(target)))
    if pscale == 0 or tscale == 0:
        return None
    predicted, target = predicted / pscale, target / tscale
    cosine = float(predicted @ target / (np.linalg.norm(predicted) * np.linalg.norm(target)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1., 1.))))


def diagnostics(cfg, fit, standard, held, truth, observed, detail_j):
    """Evaluate frozen S explanations without using labels to change them.

    Unsuffixed moment distances and risk gap refer to A. ``holdout_*`` values
    use the SAME frozen parameter bank replayed on H, never the A statistics.
    Bounds membership concerns the specified generating parameterization;
    it does not establish that no observationally equivalent parameter exists.
    The true likelihood remains defined for a signed reflection outside the
    positive-map model.  All comparisons use the original frozen threshold.
    """
    mean, scale = _array(standard["mean"], "mean", 1), _array(standard["scale"], "scale", 1)
    source_u = _array(standard["source_u"], "source_u", 2)
    adapt_u = _array(standard["adapt_u"], "adapt_u", 2)
    holdout_u = _array(held["holdout_u"], "holdout_u", 2)
    z0 = _array(standard["adapt_z0"], "adapt_z0", 2)
    holdout_z0 = _array(held["holdout_z0"], "holdout_z0", 2)
    dimension, classes = len(mean), z0.shape[1]
    if (scale.shape != mean.shape or np.any(scale <= 0)
            or source_u.shape[1] != dimension + 1 or adapt_u.shape[1] != dimension + 1
            or holdout_u.shape[1] != dimension + 1 or len(z0) != len(adapt_u)
            or holdout_z0.shape != (len(holdout_u), classes)
            or not all(np.all(u[:, -1] == 1.) for u in (source_u, adapt_u, holdout_u))):
        raise ValueError("Inconsistent standardized features or fixed logits")
    labels = np.asarray(fit["source_y"])
    family = ExplanationSet(source_u[:, :-1], labels, adapt_u[:, :-1], cfg)
    holdout_family = ExplanationSet(source_u[:, :-1], labels, holdout_u[:, :-1], cfg)
    threshold, reference_loss = float(detail_j["threshold"]), float(detail_j["reference_loss"])
    if not math.isfinite(threshold) or not math.isfinite(reference_loss) or threshold < reference_loss:
        raise ValueError("Invalid frozen reference loss or likelihood threshold")
    parameters = _array(observed["theta"], "theta", 2)
    bank = _array(observed["stats"], "stats", 3)
    reference_parameters = _array(observed["ref_theta"], "ref_theta", 1)
    correction = _array(observed["w_J"], "w_J", 2)
    if (parameters.shape != (len(bank), family._size) or not len(bank)
            or bank.shape[1:] != (classes, dimension + 1)
            or reference_parameters.shape != (family._size,) or correction.shape != bank.shape[1:]):
        raise ValueError("Incompatible frozen explanation bank or correction")
    # Recompute every saved statistic and candidate likelihood without fitting.
    holdout_bank = []
    for theta, statistic in zip(parameters, bank):
        if np.any(theta < family._lower - cfg.feasibility_tol) or np.any(theta > family._upper + cfg.feasibility_tol):
            raise ValueError("Saved explanation is outside its fixed parameter box")
        observation_loss = family._observation(theta)[0]
        if observation_loss > threshold + cfg.feasibility_tol:
            raise ValueError("Saved explanation is outside its fixed observation set")
        replay = _moment(family._target_terms(theta)[-1], adapt_u)
        if not np.allclose(replay, statistic, atol=1e-12, rtol=0.):
            raise ValueError("Saved candidate statistic fails posterior replay")
        holdout_bank.append(_moment(holdout_family._target_terms(theta)[-1], holdout_u))
    holdout_bank = np.asarray(holdout_bank)
    reference_replay = family._observation(reference_parameters)[0]
    if not np.isclose(reference_replay, reference_loss, atol=1e-10, rtol=1e-12):
        raise ValueError("Saved reference likelihood fails replay")

    model = _generating_model(truth, mean, scale, classes, dimension)
    true_loss = _true_likelihood(source_u[:, :-1], labels, adapt_u[:, :-1], model)
    true_parameters, exclusion_reason = _true_parameters(model, family)
    box_violation = None
    in_box = False
    if true_parameters is not None:
        box_violation = float(max(0., np.max(family._lower - true_parameters),
                                  np.max(true_parameters - family._upper)))
        in_box = box_violation <= cfg.feasibility_tol
        if not in_box:
            exclusion_reason = "specified_generating_parameters_exceed_fixed_box"
        # This independently checks the standardized change-of-variables law.
        replay_true = family._observation(true_parameters)[0]
        if not np.isclose(replay_true, true_loss, atol=1e-10, rtol=1e-12):
            raise ValueError("Analytic generating likelihood disagrees with explanation coordinates")
    true_q = _array(truth["adapt_q"], "adapt_q", 2)
    holdout_q = _array(truth["holdout_q"], "holdout_q", 2)
    for features, q in ((adapt_u, true_q), (holdout_u, holdout_q)):
        derived = softmax(_true_log_components(features[:, :-1], model), axis=1)
        if derived.shape != q.shape or not np.allclose(derived, q, atol=1e-11, rtol=1e-10):
            raise ValueError("Sealed true posterior disagrees with standardized generating model")
    moments = dict(true=_moment(true_q, adapt_u), source=_moment(softmax(z0, axis=1), adapt_u),
                   holdout_true=_moment(holdout_q, holdout_u),
                   holdout_source=_moment(softmax(holdout_z0, axis=1), holdout_u))
    hulls = {name: hull_distance(statistic, holdout_bank if name.startswith("holdout_") else bank,
                                 feasibility_tol=cfg.feasibility_tol, optimality_tol=cfg.optimality_tol)
             for name, statistic in moments.items()}
    true_risk = float(relative_risks(z0, adapt_u, correction, moments["true"][None])[0])
    worst = float(relative_risks(z0, adapt_u, correction, bank).max())
    holdout_true_risk = float(relative_risks(holdout_z0, holdout_u, correction, moments["holdout_true"][None])[0])
    holdout_worst = float(relative_risks(holdout_z0, holdout_u, correction, holdout_bank).max())
    cdf = _cdf_diagnostic(family, reference_parameters, holdout_u[:, :-1])
    return dict(
        true_in_model_bounds=bool(in_box), true_parameter_box_violation=box_violation,
        true_parameter_exclusion_reason=exclusion_reason,
        true_observation_loss=true_loss, true_observation_threshold=threshold,
        true_likelihood_within_threshold=bool(true_loss <= threshold + cfg.feasibility_tol),
        true_in_observation_set=bool(in_box and true_loss <= threshold + cfg.feasibility_tol),
        true_moment_distance=hulls["true"]["distance"], source_moment_distance=hulls["source"]["distance"],
        holdout_true_moment_distance=hulls["holdout_true"]["distance"],
        holdout_source_moment_distance=hulls["holdout_source"]["distance"],
        true_directional_risk=true_risk, retained_worst_directional_risk=worst,
        true_directional_risk_gap=true_risk - worst,
        holdout_true_directional_risk=holdout_true_risk,
        holdout_retained_worst_directional_risk=holdout_worst,
        holdout_true_directional_risk_gap=holdout_true_risk - holdout_worst,
        reference_holdout_coordinate_cdf_sup=cdf["maximum"],
        reference_holdout_coordinate_cdf_sup_median=cdf["median"],
        reference_holdout_cdf=cdf, hull_diagnostics=hulls,
        hull_solutions_verified=all(item["valid"] for item in hulls.values()),
        candidate_count=len(bank), bounds_scope="specified_generating_parameterization",
        coverage_scope="original_retained_candidate_hull_only",
        target_labels_used_to_modify_models=False,
    )
