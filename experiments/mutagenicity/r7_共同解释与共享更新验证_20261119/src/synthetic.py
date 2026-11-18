"""Registered R7 Gaussian worlds, sealed truth, and stage-S decisions.

``generate`` deliberately returns separate dictionaries: only ``inputs`` may
reach adaptation.  ``truth`` contains target labels, posteriors and generating
parameters and must stay behind the runner's observation seal.  This module
does not fit an explanation, select a width, or run a research trajectory.
"""

import math

import numpy as np


def _positive_integer(value, name, *, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probabilities(value, name, classes):
    value = np.asarray(value, dtype=np.float64)
    if (value.shape != (classes,) or not np.isfinite(value).all()
            or np.any(value <= 0) or not np.isclose(value.sum(), 1., atol=1e-14, rtol=0.)):
        raise ValueError(f"{name} must contain positive class probabilities summing to one")
    return value


def _posterior(logits):
    centered = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(centered)
    return weights / weights.sum(axis=1, keepdims=True)


def generate(plan, world_index, seed):
    """Return raw features/source labels and separately protected target truth.

    The source, adaptation and holdout RNGs are independent children of
    ``SeedSequence([seed, world_index])`` in that order.  Within each stream
    labels are drawn first, followed by one [sample, coordinate] normal draw.
    Arrays are float64, except labels (int64).  There is no resampling if an
    unusually small source fixture misses a class.

    Truth parameters are in RAW coordinates.  For source-fitted standardizing
    mean ``m`` and scale ``s``, the same generating model has
    means=(raw_source_means-m)/s, variances=raw_source_variances/s**2,
    L=raw_L_diagonal, and b=((L-1)*m+raw_b)/s.  Signed L is retained so that
    the semantic-reflection world's exclusion from a positive-map family is
    visible, rather than silently replacing it with an allowed explanation.
    """
    phase = plan["phases"]["S"]
    world_index = _positive_integer(world_index, "world_index", allow_zero=True)
    seed = _positive_integer(seed, "seed", allow_zero=True)
    worlds = phase["worlds"]
    if world_index >= len(worlds):
        raise ValueError("world_index is outside the registered world list")
    world = worlds[world_index]
    means = np.asarray(world.get("source_class_means_override", phase["source_class_means"]),
                       dtype=np.float64)
    if means.ndim != 2 or means.shape[0] != 2 or not means.shape[1] or not np.isfinite(means).all():
        raise ValueError("Registered source means must be a finite [2, dimension] matrix")
    classes, dimension = means.shape
    if phase["class_order"] != list(range(classes)):
        raise ValueError("Synthetic class_order must be contiguous and match the source means")
    variance = np.asarray(phase["source_class_diagonal_variance"], dtype=np.float64)
    diagonal = np.asarray(world["L_diagonal"], dtype=np.float64)
    shift = np.asarray(world["b"], dtype=np.float64)
    if (variance.shape != (dimension,) or diagonal.shape != (dimension,)
            or shift.shape != (dimension,) or not np.isfinite(variance).all()
            or not np.isfinite(diagonal).all() or not np.isfinite(shift).all()
            or np.any(variance <= 0) or np.any(diagonal == 0)):
        raise ValueError("World variances, invertible diagonal map, or shift are invalid")
    source_prior = _probabilities(phase["source_prior"], "source_prior", classes)
    target_prior = _probabilities(world["target_prior"], "target_prior", classes)
    counts = [_positive_integer(phase[key], key) for key in
              ("source_count", "adaptation_count", "holdout_count")]
    streams = [np.random.default_rng(child)
               for child in np.random.SeedSequence([seed, world_index]).spawn(3)]

    def sample(rng, count, prior, target=False):
        labels = rng.choice(classes, size=count, p=prior).astype(np.int64)
        reference = means[labels] + rng.normal(size=(count, dimension)) * np.sqrt(variance)
        features = (reference - shift) / diagonal if target else reference
        return np.asarray(features, dtype=np.float64), labels

    source_h, source_y = sample(streams[0], counts[0], source_prior)
    adapt_h, adapt_y = sample(streams[1], counts[1], target_prior, True)
    holdout_h, holdout_y = sample(streams[2], counts[2], target_prior, True)

    # Equal class covariance makes the analytic source logits affine after
    # removing the common quadratic term. Centering classes is probability
    # preserving and supplies a unique source-head representation for J_cal.
    conditional_weight = means / variance
    conditional_bias = -0.5 * np.sum(means ** 2 / variance, axis=1)
    source_weight = conditional_weight - conditional_weight.mean(axis=0, keepdims=True)
    source_bias = conditional_bias + np.log(source_prior)
    source_bias -= source_bias.mean()
    target_weight = conditional_weight * diagonal
    target_bias = conditional_bias + conditional_weight @ shift + np.log(target_prior)
    target_weight -= target_weight.mean(axis=0, keepdims=True)
    target_bias -= target_bias.mean()
    adapt_z0 = adapt_h @ source_weight.T + source_bias
    holdout_z0 = holdout_h @ source_weight.T + source_bias
    adapt_q = _posterior(adapt_h @ target_weight.T + target_bias)
    holdout_q = _posterior(holdout_h @ target_weight.T + target_bias)
    inputs = dict(source_h=source_h, source_y=source_y, adapt_h=adapt_h,
                  holdout_h=holdout_h, adapt_z0=adapt_z0, holdout_z0=holdout_z0)
    truth = dict(
        adapt_y=adapt_y, holdout_y=holdout_y, adapt_q=adapt_q, holdout_q=holdout_q,
        world=str(world["id"]), world_index=world_index, seed=seed,
        class_order=list(range(classes)), raw_source_means=means.copy(),
        raw_source_variances=np.broadcast_to(variance, means.shape).copy(),
        raw_L_diagonal=diagonal.copy(), raw_b=shift.copy(),
        source_prior=source_prior.copy(), target_prior=target_prior.copy(),
        source_head_weight=source_weight, source_head_bias=source_bias,
        target_head_weight=target_weight, target_head_bias=target_bias,
        raw_target_class_means=(means - shift) / diagonal,
        raw_target_class_variances=np.broadcast_to(variance / diagonal ** 2, means.shape).copy(),
        positive_diagonal_generating_map=bool(np.all(diagonal > 0)),
        parameter_coordinates="raw_before_source_fit_standardization",
        parameter_box_and_likelihood_membership="evaluate_after_standardization_and_frozen_fit",
    )
    if not all(np.isfinite(value).all() for value in
               (*inputs.values(), adapt_q, holdout_q, source_weight, source_bias,
                target_weight, target_bias)):
        raise FloatingPointError("Non-finite synthetic inputs or analytic truth")
    return inputs, truth


def _finite_metric(row, key):
    value = row.get(key)
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def synthetic_gate(plan, rows):
    """Evaluate registered S gates without choosing a width or dropping seeds.

    Row keys are ``world``, ``seed``, ``tau``, ``method``, ``valid`` and the
    metrics ``holdout_expected_nll_delta``, ``centered_rms``, and
    ``true_directional_risk_gap``.  The latter is A's true conditional risk
    change MINUS the worst relative risk over retained explanations (not the
    regularized objective).  ``centered_rms`` is H's class-centered correction.

    Analytic-posterior oracle rows use method ``O_analytic``.  Sampled-label
    ``O`` is diagnostic and cannot substitute for it.  Every registered world
    and seed requires valid J and O_analytic primary-width rows; additional
    methods/widths are ignored here. The runner separately checks complete
    study artifacts, engineering gates and seals before permitting phase P.
    """
    phase = plan["phases"]["S"]
    gate = phase["gate"]
    primary = float(plan["shared_update"]["primary_tau"])
    worlds = [str(world["id"]) for world in phase["worlds"]]
    seeds = [_positive_integer(seed, "seed", allow_zero=True) for seed in phase["seeds"]]
    if len(set(worlds)) != len(worlds) or len(set(seeds)) != len(seeds):
        raise ValueError("Registered worlds and seeds must be unique")
    positive_worlds = list(gate["required_positive_worlds"])
    if any(world not in worlds for world in positive_worlds) or "no_shift" not in worlds:
        raise ValueError("Synthetic gate worlds are absent from the registered plan")
    minimum = _positive_integer(gate["minimum_seed_count"], "minimum_seed_count")
    if minimum > len(seeds):
        raise ValueError("Synthetic minimum_seed_count exceeds the registered seeds")
    methods = ("J", "O_analytic")
    expected = {(world, seed, method) for world in worlds for seed in seeds for method in methods}
    records, duplicates, unexpected, malformed = {}, [], [], []
    for index, row in enumerate(rows):
        if row.get("method") not in methods:
            continue
        tau = _finite_metric(row, "tau")
        if tau is None:
            malformed.append(dict(row=index, reason="invalid_tau"))
            continue
        if tau != primary:
            continue
        try:
            raw_seed = row["seed"]
            if isinstance(raw_seed, (bool, np.bool_)) or int(raw_seed) != float(raw_seed):
                raise ValueError("invalid seed")
            key = (str(row["world"]), int(raw_seed), str(row["method"]))
        except (KeyError, ValueError, TypeError, OverflowError):
            malformed.append(dict(row=index, reason="invalid_primary_key"))
            continue
        if key not in expected:
            unexpected.append(list(key))
        elif key in records:
            duplicates.append(list(key))
        else:
            records[key] = row
    missing = [list(key) for key in sorted(expected - records.keys())]
    invalid = []
    for key, row in records.items():
        metrics = ("holdout_expected_nll_delta",)
        if key[2] == "J":
            metrics += ("centered_rms", "true_directional_risk_gap")
        errors = [name for name in metrics if _finite_metric(row, name) is None]
        if not isinstance(row.get("valid"), (bool, np.bool_)) or not bool(row.get("valid")):
            errors.append("valid")
        if _finite_metric(row, "centered_rms") is not None and float(row["centered_rms"]) < 0:
            errors.append("negative_centered_rms")
        if errors:
            invalid.append(dict(key=list(key), fields=errors))
    invalid_keys = {tuple(item["key"]) for item in invalid}

    def usable(world, seed, method):
        key = (world, seed, method)
        return records.get(key) if key not in invalid_keys else None

    checks = {"complete_primary_rows": not (missing or duplicates or unexpected or malformed or invalid)}
    summaries = {}
    for world in worlds:
        joint_good, oracle_good, joint_deltas, oracle_deltas = [], [], [], []
        for seed in seeds:
            joint, oracle = usable(world, seed, "J"), usable(world, seed, "O_analytic")
            if joint is not None:
                delta = float(joint["holdout_expected_nll_delta"])
                joint_deltas.append(delta)
                if (delta < 0 and float(joint["centered_rms"]) > gate["centered_logit_rms_nonzero"]
                        and float(joint["true_directional_risk_gap"]) <= gate["true_directional_risk_slack_nats"]):
                    joint_good.append(seed)
            if oracle is not None:
                delta = float(oracle["holdout_expected_nll_delta"])
                oracle_deltas.append(delta)
                if delta < 0:
                    oracle_good.append(seed)
        complete_joint = len(joint_deltas) == len(seeds)
        summaries[world] = dict(
            joint_passing_seeds=joint_good, analytic_oracle_improving_seeds=oracle_good,
            joint_mean_expected_nll_delta=float(np.mean(joint_deltas)) if complete_joint else None,
            joint_max_expected_nll_delta=max(joint_deltas) if complete_joint else None,
            analytic_oracle_mean_expected_nll_delta=(float(np.mean(oracle_deltas))
                                                     if len(oracle_deltas) == len(seeds) else None),
            registered_seeds=len(seeds), valid_joint_rows=len(joint_deltas),
            valid_analytic_oracle_rows=len(oracle_deltas),
        )
        if world in positive_worlds:
            checks[f"{world}:joint_signal"] = len(joint_good) >= minimum
            checks[f"{world}:analytic_oracle"] = len(oracle_good) >= minimum
        if world == "no_shift":
            checks["no_shift:mean_harm_cap"] = bool(
                complete_joint and np.mean(joint_deltas) <= gate["no_shift_mean_nll_increase_cap"])
            checks["no_shift:each_seed_harm_cap"] = bool(
                complete_joint and max(joint_deltas) <= gate["no_shift_each_seed_nll_increase_cap"])
    passed = all(checks.values())
    return dict(passed=passed, valid=passed, primary_tau=primary, checks=checks,
                failed_checks=[name for name, value in checks.items() if not value],
                expected_primary_rows=len(expected), found_primary_rows=len(records),
                missing=missing, duplicates=duplicates, unexpected=unexpected,
                malformed=malformed, invalid_rows=invalid, worlds=summaries,
                scope="statistical_stage_S_gate_only; runner_must_also_verify_engineering_and_seals")


def check_known_moment():
    """Check the actual design fixture and zero update using the core master.

    This is a tiny deterministic engineering check, not a fitted synthetic
    world. The first probability column follows the design's displayed class.
    The registered positive 1e-9 L2 approximates its illustrative zero-L2 case.
    """
    from tsba.shared_update.solver import relative_risks, solve_master

    initial = np.array([.42, .53, .47, .48], dtype=np.float64)
    z0 = np.column_stack((np.log(initial / (1 - initial)), np.zeros(4)))
    features = np.array([[-1.], [-1.], [1.], [1.]])
    class_probability = np.array([[.10, .60, .35, .85], [.50, .30, .75, .35]])
    posteriors = np.stack((class_probability, 1 - class_probability), axis=-1)
    statistics = np.einsum("knc,nd->kcd", posteriors, features) / len(features)
    correction, info = solve_master(z0, features, statistics, regularization=1e-9)
    probabilities = _posterior(z0 + features @ correction.T)
    risks = relative_risks(z0, features, correction, statistics)
    direct = np.mean(np.sum(posteriors * (np.log(_posterior(z0)) - np.log(probabilities)), axis=2), axis=1)
    expected_probabilities = np.array([.34805146, .45395642, .54604358, .55596431])
    expected_risks = np.array([-.02663122, -.01138913])
    source_statistic = _posterior(z0).T @ features / len(features)
    augmented = np.concatenate((statistics, source_statistic[None]), axis=0)
    zero_correction, zero_info = solve_master(z0, features, augmented, regularization=1e-9)
    zero_probabilities = _posterior(z0 + features @ zero_correction.T)
    zero_risks = relative_risks(z0, features, zero_correction, augmented)
    checks = dict(
        finite_master_valid=bool(info["valid"]), zero_master_valid=bool(zero_info["valid"]),
        design_probabilities=bool(np.allclose(probabilities[:, 0], expected_probabilities, atol=1e-6, rtol=0.)),
        design_risks=bool(np.allclose(risks, expected_risks, atol=1e-6, rtol=0.)),
        risk_identity=bool(np.allclose(risks, direct, atol=1e-12, rtol=0.)),
        source_moment_restores_zero_probabilities=bool(np.allclose(zero_probabilities, _posterior(z0), atol=1e-6, rtol=0.)),
        source_moment_restores_zero_risks=bool(np.allclose(zero_risks, 0., atol=1e-6, rtol=0.)),
    )
    passed = all(checks.values())
    return dict(passed=passed, valid=passed, checks=checks,
                failed_checks=[name for name, value in checks.items() if not value],
                regularization=1e-9, numeric_atol=1e-6,
                source_probabilities=initial.tolist(), probabilities=probabilities.tolist(),
                risks=risks.tolist(), direct_risks=direct.tolist(),
                statistics=statistics.tolist(), source_statistic=source_statistic.tolist(),
                correction=correction.tolist(), master=info,
                zero_correction=zero_correction.tolist(), zero_probabilities=zero_probabilities.tolist(),
                zero_risks=zero_risks.tolist(), zero_master=zero_info)
