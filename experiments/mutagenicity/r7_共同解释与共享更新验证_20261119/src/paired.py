"""Paired graph bootstrap for already opened R7 P/C/D evaluation arrays.

No files or labels are opened here.  The caller must first pass the R7 label
gates.  Models, explanations and their predictions remain fixed.  One target
graph draw is shared by every method, model seed and source direction using
that setting/target cohort; model seeds themselves are never resampled.
"""
import hashlib
from itertools import permutations

import numpy as np

from .decisions import ALIASES, BALANCED, decision_policy


_METRICS = ("NLL", "BA", "Accuracy", "directional_risk_gap")
_SCALES = np.asarray([1., 100., 100., 1.])


def _expected(plan, phase):
    config = plan["phases"][phase]
    seeds = plan["real_model_seeds"]
    if phase in ("P", "D"):
        directions = {config["setting"]: [tuple(pair) for pair in config["directions"]]}
    elif phase == "C":
        directions = {setting: list(permutations(config["domains"], 2)) for setting in config["settings"]}
    else:
        raise ValueError("The registered real-data bootstrap supports only P, C and D")
    if (not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds)
            or any(not pairs or len(set(pairs)) != len(pairs) for pairs in directions.values())):
        raise ValueError("Invalid registered seed or direction grid")
    expected = {(setting, source, target, seed) for setting, pairs in directions.items()
                for source, target in pairs for seed in seeds}
    return directions, seeds, expected


def _settings(plan):
    config = plan["evaluation"]
    interval = config["paired_interval"]
    if (interval["strata"] != ["target_domain", "class"]
            or not interval["reuse_draws_across_methods_seeds_and_source_directions"]
            or not interval["conditional_on_fitted_models"]):
        raise ValueError("Unsupported bootstrap protocol")
    replicates, seed = config["cohort_bootstrap_replicates"], config["cohort_bootstrap_seed"]
    minimum = interval["minimum_graphs_per_class_for_interval"]
    percentiles = interval["percentiles"]
    if (type(replicates) is not int or replicates < 1 or type(seed) is not int or seed < 0
            or type(minimum) is not int or minimum < 1
            or percentiles != [2.5, 97.5] or interval["level"] != .95):
        raise ValueError("Invalid registered bootstrap count, seed or interval")
    return replicates, seed, minimum, percentiles


def _log_probabilities(logits, n):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (n, 2) or not np.isfinite(logits).all():
        raise ValueError("Every valid method requires finite binary logits [n_graphs, 2]")
    centered = logits - logits.max(axis=1, keepdims=True)
    result = centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite log probabilities")
    return result


def _cohort(setting, target, ids, labels, replicates, seed, minimum):
    counts = np.bincount(labels, minlength=2)
    eligible = bool(np.all(counts >= minimum))
    # Keyed streams make a P cohort's draws identical when it reappears in C.
    # All eight SHA256 words are retained, with an explicitly fixed byte order.
    words = np.frombuffer(hashlib.sha256(setting.encode("utf-8")).digest(), dtype="<u4").tolist()
    draws = np.zeros((replicates, len(ids)), dtype=np.int32) if eligible else None
    if eligible:
        for cls in (0, 1):
            positions = np.flatnonzero(labels == cls)
            rng = np.random.default_rng(np.random.SeedSequence([seed, *words, target, cls]))
            draws[:, positions] = rng.multinomial(len(positions), np.full(len(positions), 1. / len(positions)),
                                                  size=replicates)
        draw_hash = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
        weights = draws.astype(np.float64) / len(ids)
    else:
        draw_hash, weights = None, None
    return dict(ids=ids, y=labels, weights=weights, eligible=eligible,
                audit=dict(setting=setting, target=target, graphs=len(ids), class_counts=counts.tolist(),
                           interval_eligible=eligible, sorted_ids=ids.tolist(),
                           draw_counts_sha256=draw_hash, draw_counts_dtype="int32",
                           reason=None if eligible else "fewer_than_minimum_graphs_in_a_target_class"))


def _block_metrics(block, cohort, order, names, replicates):
    """Return point/sample metrics, recomputing the finite-bank max per draw."""
    y, weights = cohort["y"], cohort["weights"]
    n = len(y)
    counts = np.bincount(y, minlength=2)
    balanced = n / (2 * counts[y]) if bool(np.all(counts)) else None
    logits = block["logits"]
    logs = {name: _log_probabilities(value, n)[order]
            for name, value in logits.items() if name in names | {"B0"} and value is not None}
    banks = {}
    for bank_name in ('candidate_posteriors', 'expanded_candidate_posteriors'):
        bank = block.get(bank_name)
        if bank is None:
            continue
        bank = np.asarray(bank, dtype=np.float64)
        if (bank.ndim != 3 or bank.shape[1:] != (n, 2) or not len(bank)
                or not np.isfinite(bank).all() or np.any(bank < 0)
                or not np.allclose(bank.sum(axis=2), 1., rtol=0, atol=1e-10)):
            raise ValueError("Candidate posteriors must be finite probabilities [K, n_graphs, 2]")
        banks[bank_name] = bank[:, order, :]
    rows = np.arange(n)
    baseline_loss = -logs["B0"][rows, y] if "B0" in logs else None
    result = {}
    for name in names:
        point = np.full(4, np.nan)
        samples = np.full((replicates, 4), np.nan) if weights is not None else None
        reasons = [None] * 4
        if name not in logs:
            result[name] = (point, samples, ["missing_or_invalid_method"] * 4)
            continue
        logp = logs[name]
        loss = -logp[rows, y]
        decisions = block.get('decisions', {})
        if name in decisions:
            actions = np.asarray(decisions[name])
            if (actions.shape != (n,) or not np.issubdtype(actions.dtype, np.integer)
                    or not np.isin(actions, [0, 1]).all()):
                raise ValueError('Valid methods require one frozen binary decision per graph')
            prediction = actions[order]
        else:
            prediction = logp.argmax(axis=1)
        correct = (prediction == y).astype(np.float64)
        point[0], point[2] = loss.mean(), correct.mean()
        if weights is not None:
            samples[:, 0], samples[:, 2] = weights @ loss, weights @ correct
        if balanced is not None:
            point[1] = np.mean(correct * balanced)
            if weights is not None:
                samples[:, 1] = weights @ (correct * balanced)
        else:
            reasons[1] = "target_class_absent"
        key = 'expanded_candidate_posteriors' if name.endswith('_plus') else 'candidate_posteriors'
        bank = banks.get(key)
        if bank is None or baseline_loss is None:
            reasons[3] = "missing_candidate_bank" if bank is None else "missing_B0_for_relative_risk"
        else:
            candidate_risks = -np.einsum("knc,nc->kn", bank, logp - logs["B0"])
            true_relative = loss - baseline_loss
            point[3] = true_relative.mean() - candidate_risks.mean(axis=1).max()
            if weights is not None:
                samples[:, 3] = weights @ true_relative - (weights @ candidate_risks.T).max(axis=1)
        result[name] = (point * _SCALES, None if samples is None else samples * _SCALES, reasons)
    return result


def paired_bootstrap(plan, blocks, *, phase, comparisons, tau=None):
    """Compute registered paired percentile intervals from frozen H predictions.

    Each block is a dict with ``setting, source, target, seed, ids, y, logits``.
    ``logits`` maps method names to [n, 2] arrays; use None for an invalid or
    missing method, including a numerical fallback excluded by the protocol.
    Optional ``decisions`` maps names to frozen binary classification actions.
    These affect BA/Accuracy only; NLL and risk gaps always use unshifted logits.
    The bootstrap never estimates or refits decision thresholds from its draws.
    The optional ``candidate_posteriors`` is the retained joint bank's [K,n,2]
    posterior array evaluated on these same H graph IDs, in the same row order.
    Arms ending in `_plus` use `expanded_candidate_posteriors` instead for gap metrics.
    The caller supplies one fixed width/pool per call, after label seal checks.

    Comparisons are (left, right) method-name pairs; all deltas are left minus
    right. NLL and directional-risk-gap units are nats; BA and Accuracy are pp.
    Gap(W) is empirical relative NLL minus max_k retained-candidate relative
    NLL, with that max recomputed for every bootstrap sample (not per graph).

    ``interval_rows`` covers each registered direction and each whole setting.
    Direction means average all registered seeds. Setting means first average
    directions equally within each seed, then average seeds. Missing pairs are
    retained in denominators; the affected full estimate/CI is None, while
    complete individual seed estimates remain visible. A small class suppresses
    CIs, but does not suppress otherwise defined point estimates.
    """
    directions, seeds, expected = _expected(plan, phase)
    replicates, seed, minimum, percentiles = _settings(plan)
    comparisons = [tuple(pair) for pair in comparisons]
    if (not comparisons or len(set(comparisons)) != len(comparisons)
            or any(len(pair) != 2 or pair[0] == pair[1]
                   or any(not isinstance(name, str) or not name for name in pair) for pair in comparisons)):
        raise ValueError("Comparisons must be distinct nonempty method-name pairs")
    names = {name for pair in comparisons for name in pair}
    normalized, cohorts = {}, {}
    explicit_actions = decision_policy(plan) is not None
    for block in blocks:
        key = (block["setting"], block["source"], block["target"], block["seed"])
        if key not in expected or key in normalized:
            raise ValueError("Unexpected or duplicate setting/direction/seed block")
        if explicit_actions:
            for name in BALANCED | set(ALIASES):
                if block['logits'].get(name) is not None and name not in block.get('decisions', {}):
                    raise ValueError('Registered classification policy requires explicit frozen decisions: ' + name)
        ids, y = np.asarray(block["ids"]), np.asarray(block["y"])
        if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or len(np.unique(ids)) != len(ids)
                or y.shape != ids.shape or not np.issubdtype(y.dtype, np.integer)
                or not np.isin(y, [0, 1]).all()):
            raise ValueError("Each cohort requires unique graph IDs and binary integer labels")
        order = np.argsort(ids, kind="stable")
        ids, y = ids[order].astype(np.int64), y[order].astype(np.int64)
        cohort_key = key[0], key[2]
        if cohort_key in cohorts:
            previous = cohorts[cohort_key]
            if not np.array_equal(ids, previous["ids"]) or not np.array_equal(y, previous["y"]):
                raise ValueError("Shared target cohorts differ in graph IDs or labels across seeds/source directions")
        else:
            cohorts[cohort_key] = _cohort(*cohort_key, ids, y, replicates, seed, minimum)
        normalized[key] = block, order
    for setting in directions:
        joined = np.concatenate([row["ids"] for (name, _), row in cohorts.items() if name == setting]) \
            if any(name == setting for name, _ in cohorts) else np.empty(0, dtype=np.int64)
        if len(joined) != len(np.unique(joined)):
            raise ValueError("A setting's target domains contain overlapping graph IDs")
    metrics = {key: _block_metrics(block, cohorts[key[0], key[2]], order, names, replicates)
               for key, (block, order) in normalized.items()}
    output = []
    for setting, pairs in directions.items():
        scopes = [("direction", [pair]) for pair in pairs] + [("setting", pairs)]
        for scope, selected in scopes:
            for left, right in comparisons:
                for index, metric in enumerate(_METRICS):
                    reasons, seed_estimates, seed_curves = set(), [], []
                    complete = 0
                    for model_seed in seeds:
                        points, curves = [], []
                        for source, target in selected:
                            key = setting, source, target, model_seed
                            if key not in metrics:
                                reasons.add("missing_registered_block")
                                continue
                            lp, lb, lr = metrics[key][left]
                            rp, rb, rr = metrics[key][right]
                            if not np.isfinite(lp[index]) or not np.isfinite(rp[index]):
                                reasons.update(r for r in (lr[index], rr[index]) if r)
                                continue
                            complete += 1
                            points.append(float(lp[index] - rp[index]))
                            if lb is None or rb is None:
                                reasons.add("fewer_than_minimum_graphs_in_a_target_class")
                            elif not np.isfinite(lb[:, index]).all() or not np.isfinite(rb[:, index]).all():
                                reasons.add("nonfinite_bootstrap_metric")
                            else:
                                curves.append(lb[:, index] - rb[:, index])
                        seed_estimates.append(dict(seed=model_seed, estimate=float(np.mean(points))
                                                   if len(points) == len(selected) else None,
                                                   complete_pairs=len(points), expected_pairs=len(selected)))
                        if len(curves) == len(selected):
                            seed_curves.append(np.mean(curves, axis=0))
                    points = [row["estimate"] for row in seed_estimates]
                    point_complete = all(value is not None for value in points)
                    estimate = float(np.mean(points)) if point_complete else None
                    ci_complete = point_complete and len(seed_curves) == len(seeds)
                    if ci_complete:
                        distribution = np.mean(seed_curves, axis=0)
                        lower, upper = np.percentile(distribution, percentiles).tolist()
                    else:
                        lower, upper = None, None
                    output.append(dict(scope=scope, setting=setting,
                                       source=selected[0][0] if scope == "direction" else None,
                                       target=selected[0][1] if scope == "direction" else None,
                                       left=left, right=right, metric=metric,
                                       unit="pp" if metric in ("BA", "Accuracy") else "nats",
                                       estimate=estimate, lower=lower, upper=upper,
                                       seed_estimates=seed_estimates,
                                       seed_sample_sd=float(np.std(points, ddof=1))
                                       if point_complete and len(points) > 1 else None,
                                       expected_pairs=len(selected) * len(seeds), complete_pairs=complete,
                                       interval_replicates=replicates if ci_complete else 0,
                                       reason=";".join(sorted(reasons)) if reasons else None))
    return dict(phase=phase, tau=tau, expected_blocks=len(expected), supplied_blocks=len(normalized),
                interval_rows=output, cohort_audit=[cohorts[key]["audit"] for key in sorted(cohorts)],
                metadata=dict(replicates=replicates, seed=seed, percentiles=percentiles,
                              minimum_graphs_per_class=minimum, conditional_on_fitted_models=True,
                              resample_model_seeds=False, equal_directions_within_seed=True,
                              delta_convention="left_minus_right",
                              rng="numpy.default_rng(SeedSequence([seed, *sha256(setting) uint32_le words, target, class]))",
                              bootstrap="multinomial class-stratified graph multiplicities",
                              numpy_version=np.__version__,
                              directional_gap="empirical_relative_NLL - max_k mean_retained_candidate_relative_NLL",
                              candidate_max_recomputed_each_replicate=True))


def check_paired():
    """Small engineering fixture: pairing, gap max, weighting and missingness."""
    plan = dict(real_model_seeds=[1, 2], phases={"C": dict(settings=["toy"], domains=[0, 1, 2])},
                evaluation=dict(cohort_bootstrap_replicates=80, cohort_bootstrap_seed=20261119,
                                paired_interval=dict(strata=["target_domain", "class"],
                                                     reuse_draws_across_methods_seeds_and_source_directions=True,
                                                     conditional_on_fitted_models=True,
                                                     minimum_graphs_per_class_for_interval=2,
                                                     percentiles=[2.5, 97.5], level=.95)))
    blocks = []
    for source, target in permutations(range(3), 2):
        for seed in (1, 2):
            ids = np.arange(target * 6, target * 6 + 6)
            y = np.array([0, 0, 0, 1, 1, 1])
            b0 = np.zeros((6, 2))
            direction = np.array([-1., .5, 1., -.5, 1.5, -1.5])
            z = np.column_stack((np.zeros(6), direction))
            # Candidates swap which half they fit; bootstrap worst can change.
            q = np.stack((np.tile([.2, .8], (6, 1)), np.tile([.8, .2], (6, 1))))
            order = np.arange(5, -1, -1) if source == 2 else np.arange(6)
            blocks.append(dict(setting="toy", source=source, target=target, seed=seed,
                               ids=ids[order], y=y[order], logits={"B0": b0[order], "J": z[order], "copy": z[order]},
                               candidate_posteriors=q[:, order]))
    result = paired_bootstrap(plan, blocks, phase="C", comparisons=[("J", "B0"), ("J", "copy")])
    for row in result["interval_rows"]:
        if row["left"] == "J" and row["right"] == "copy":
            assert row["estimate"] == row["lower"] == row["upper"] == 0.
        assert row["complete_pairs"] == row["expected_pairs"]
    target_zero = [row for row in result["interval_rows"] if row["target"] == 0 and row["right"] == "B0"]
    for metric in _METRICS:
        same = [row for row in target_zero if row["metric"] == metric]
        assert same[0]["lower"] == same[1]["lower"] and same[0]["upper"] == same[1]["upper"]
    # Verify the gap against explicit per-replicate candidate means.
    first = blocks[0]
    cohort = _cohort("toy", first["target"], first["ids"], first["y"], 80, 20261119, 2)
    metrics = _block_metrics(first, cohort, np.arange(6), {"J"}, 80)["J"]
    lp = _log_probabilities(first["logits"]["J"], 6)
    base = _log_probabilities(first["logits"]["B0"], 6)
    true = -(lp - base)[np.arange(6), first["y"]]
    candidate = -np.einsum("knc,nc->kn", first["candidate_posteriors"], lp - base)
    explicit = np.array([float(weight @ true) - max(float(weight @ row) for row in candidate)
                         for weight in cohort["weights"]])
    np.testing.assert_allclose(metrics[1][:, 3], explicit, atol=1e-14, rtol=0)
    assert len(np.unique((cohort["weights"] @ candidate.T).argmax(axis=1))) == 2
    missing = paired_bootstrap(plan, blocks[:-1], phase="C", comparisons=[("J", "B0")])
    assert all(row["estimate"] is None and row["lower"] is None
               for row in missing["interval_rows"] if row["scope"] == "setting")
    small = dict(plan, evaluation=dict(plan["evaluation"], paired_interval=dict(
        plan["evaluation"]["paired_interval"], minimum_graphs_per_class_for_interval=4)))
    suppressed = paired_bootstrap(small, blocks, phase="C", comparisons=[("J", "B0")])
    assert all(row["estimate"] is not None and row["lower"] is None for row in suppressed["interval_rows"])
    # Unequal cohort sizes must not change the registered equal-direction mean.
    pplan = dict(plan, phases={"P": dict(setting="toy", directions=[[0, 1], [0, 2]])})
    unequal = []
    expected = []
    for target, n in ((1, 6), (2, 10)):
        y = np.arange(n) % 2
        z = np.eye(2)[y] * target
        expected.append(float(-_log_probabilities(z, n)[np.arange(n), y].mean() - np.log(2.)))
        for seed in (1, 2):
            unequal.append(dict(setting="toy", source=0, target=target, seed=seed,
                                ids=np.arange(100 * target, 100 * target + n), y=y,
                                logits={"B0": np.zeros((n, 2)), "J": z}))
    aggregated = paired_bootstrap(pplan, unequal, phase="P", comparisons=[("J", "B0")])
    whole = next(row for row in aggregated["interval_rows"]
                 if row["scope"] == "setting" and row["metric"] == "NLL")
    np.testing.assert_allclose(whole["estimate"], np.mean(expected), atol=1e-14, rtol=0)
    single_seed = paired_bootstrap(dict(pplan, real_model_seeds=[1]),
                                  [row for row in unequal if row["seed"] == 1],
                                  phase="P", comparisons=[("J", "B0")])
    for repeated, once in zip(aggregated["interval_rows"], single_seed["interval_rows"]):
        assert repeated["lower"] == once["lower"] and repeated["upper"] == once["upper"]
    return dict(passed=True, fixture_blocks=len(blocks), replicates=80,
                shared_draws=True, candidate_max_recomputed=True, missing_pairs_retained=True,
                small_class_intervals_suppressed=True, equal_direction_weighting=True,
                identical_seed_duplication_does_not_shrink_intervals=True)
