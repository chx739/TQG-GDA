"""Source-only diagonal GMM templates with shared affine target adaptation.

Each source class selects K in {1, 2, 4} by its own source-fit BIC.  Templates
are then frozen. Target fitting/search changes only shared positive diagonal
scales, shifts and class priors, keeping the original joint NLL denominator
and all original update features. The default covariance regularizer equals
variance_floor, but is ADDITIVE covariance regularization, not max(empirical
variance, variance_floor). Source EM uses only NumPy/SciPy; no sklearn runtime
dependency. Regularization and finite/convergence diagnostics are recorded.
"""

from copy import deepcopy

import numpy as np
from scipy.special import logsumexp, ndtr, softmax

from .projected_explanations import ProjectedExplanationSet, _readonly


def _component_log_density(points, weights, means, variances):
    difference = points[:, None, :]-means[None, :, :]
    return (np.log(weights)[None, :]
            -.5*(np.log(2*np.pi*variances)[None, :, :]+difference**2/variances[None, :, :]).sum(axis=2))


def _source_em(observations, count, *, seed, regularizer, maxiter, tolerance):
    """One deterministic k-means++ initialization and diagonal EM run.

    M-step variances are responsibility-weighted population residual variances
    plus ``regularizer``. Preserve the best finite likelihood iterate even if
    the iteration limit is reached or regularization makes likelihood fall.
    """
    rng = np.random.default_rng(seed)
    n, dimension = observations.shape
    centers = [observations[int(rng.integers(n))].copy()]
    distance = ((observations-centers[0])**2).sum(axis=1)
    for _ in range(1, count):
        total = float(distance.sum())
        index = int(rng.choice(n, p=distance/total)) if total > 0 else int(rng.integers(n))
        centers.append(observations[index].copy())
        distance = np.minimum(distance, ((observations-centers[-1])**2).sum(axis=1))
    means = np.stack(centers)
    variances = np.tile(observations.var(axis=0)+regularizer, (count, 1))
    weights = np.full(count, 1/count)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        log_components = _component_log_density(observations, weights, means, variances)
        log_norm = logsumexp(log_components, axis=1)
    mean_likelihood = float(log_norm.mean())
    if not np.isfinite(mean_likelihood):
        raise FloatingPointError("Non-finite initial source mixture likelihood")
    best = (mean_likelihood, weights.copy(), means.copy(), variances.copy(), 0)
    converged, decreases, error = False, 0, None
    change, iterations = None, 0
    for iteration in range(1, maxiter+1):
        iterations = iteration
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                responsibilities = np.exp(log_components-log_norm[:, None])
                mass = responsibilities.sum(axis=0)+10*np.finfo(np.float64).eps
                weights = mass/mass.sum()
                means = responsibilities.T@observations/mass[:, None]
                difference = observations[:, None, :]-means[None, :, :]
                variances = (responsibilities[:, :, None]*difference**2).sum(axis=0)/mass[:, None]+regularizer
                log_components = _component_log_density(observations, weights, means, variances)
                log_norm = logsumexp(log_components, axis=1)
                updated = float(log_norm.mean())
            if not np.isfinite(updated):
                raise FloatingPointError("Non-finite source EM likelihood")
            change = updated-mean_likelihood
            if updated > best[0]:
                best = (updated, weights.copy(), means.copy(), variances.copy(), iteration)
            decreases += int(change < -tolerance)
            mean_likelihood = updated
            if abs(change) < tolerance:
                converged = True
                break
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exception:
            error = str(exception)
            break
    likelihood, weights, means, variances, selected_iteration = best
    report = dict(seed=seed, converged=converged, iterations=iterations,
                  selected_iteration=selected_iteration, best_mean_log_likelihood=likelihood,
                  last_mean_log_likelihood=mean_likelihood, last_mean_log_likelihood_change=change,
                  likelihood_decrease_iterations=decreases, finite_candidate=True, error=error)
    return dict(weights=weights, means=means, variances=variances), report


class MixtureExplanationSet(ProjectedExplanationSet):
    """Frozen per-class mixtures, reduced affine/prior parameters, full moments.

    ``source_templates`` is a tuple of dictionaries containing readonly
    weights, means and diagonal variances. ``source_component_counts`` lists
    actual selected K values. Even two-dimensional synthetic inputs fit these
    source mixtures; only the projection is an identity in that case.

    CDF values use projected density coordinates. ``_unpack`` flattens mixture
    components for the existing CDF helper; its final weights are component
    masses, NOT class priors. Use ``class_prior(theta)`` for class priors.
    """

    def __init__(self, source_x, source_y, target_x, config, *, source_head_normal,
                 fisher_regularization=.1, rank_tolerance=1e-10,
                 component_counts=(1, 2, 4), mixture_n_init=5,
                 mixture_maxiter=300, mixture_tol=1e-6, mixture_reg_covar=None):
        grid = tuple(component_counts)
        if (not grid or any(isinstance(k, (bool, np.bool_)) or int(k) != k or k not in (1, 2, 4) for k in grid)
                or len(set(grid)) != len(grid) or 1 not in grid):
            raise ValueError("Component grid must be a distinct subset of {1, 2, 4} containing 1")
        for value in (mixture_n_init, mixture_maxiter):
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
                raise ValueError("Mixture initialization and iteration counts must be positive integers")
        reg = float(config.variance_floor if mixture_reg_covar is None else mixture_reg_covar)
        if not np.isfinite([reg, mixture_tol]).all() or reg <= 0 or mixture_tol <= 0:
            raise ValueError("Mixture covariance regularizer and tolerance must be positive")
        self.component_count_grid = tuple(sorted(int(k) for k in grid))
        self.mixture_n_init, self.mixture_maxiter = int(mixture_n_init), int(mixture_maxiter)
        self.mixture_tol, self.mixture_reg_covar = float(mixture_tol), reg
        super().__init__(source_x, source_y, target_x, config, source_head_normal=source_head_normal,
                         fisher_regularization=fisher_regularization, rank_tolerance=rank_tolerance)
        self.source_templates, self.source_template_report = self._fit_source_templates()
        self.source_component_counts = tuple(len(template["weights"]) for template in self.source_templates)
        self._component_means = _readonly(np.concatenate([t["means"] for t in self.source_templates]))
        self._component_log_variances = _readonly(np.log(np.concatenate([t["variances"] for t in self.source_templates])))
        self._component_weights = _readonly(np.concatenate([t["weights"] for t in self.source_templates]))
        self._component_classes = np.repeat(np.arange(self.classes), self.source_component_counts)
        self._component_classes.setflags(write=False)
        source_density, _ = self._density_terms(self.source)
        self._source_conditional_nll = -float(source_density[np.arange(len(self.source)), self.labels].sum())

    def _initial_parameters(self):
        initial, lower, upper = super()._initial_parameters()
        prefix = 2*self.classes*self.dimension
        self._size = len(initial)-prefix
        return initial[prefix:].copy(), lower[prefix:].copy(), upper[prefix:].copy()

    def _fit_source_templates(self):
        templates, reports = [], []
        for label in range(self.classes):
            observations = self.source[self.labels == label]
            candidates, trials = [], []
            for count in self.component_count_grid:
                trial = dict(components=count, source_rows=len(observations), eligible=False)
                if count > len(observations):
                    trials.append(trial | {"reason": "fewer_source_rows_than_components"})
                    continue
                runs, fits = [], []
                for initialization in range(self.mixture_n_init):
                    seed = int(np.random.SeedSequence([int(self.config.seed), label, count, initialization, 9817])
                               .generate_state(1)[0])
                    try:
                        template, run = _source_em(observations, count, seed=seed,
                            regularizer=self.mixture_reg_covar, maxiter=self.mixture_maxiter,
                            tolerance=self.mixture_tol)
                        runs.append(run)
                        fits.append((run["best_mean_log_likelihood"], initialization, template, run))
                    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                        runs.append(dict(seed=seed, converged=False, finite_candidate=False, error=str(error)))
                if fits:
                    mean_likelihood, initialization, template, selected_run = max(fits, key=lambda item: (item[0], -item[1]))
                    log_likelihood = mean_likelihood*len(observations)
                    parameter_count = count*(2*self.dimension+1)-1
                    bic = -2*log_likelihood+parameter_count*np.log(len(observations))
                    finite = (np.isfinite(bic) and all(np.isfinite(value).all() for value in template.values())
                              and np.all(template["variances"] > 0) and np.all(template["weights"] > 0))
                    trial.update(eligible=bool(finite), converged=selected_run["converged"],
                                 selected_initialization=initialization, runs=runs, bic=float(bic),
                                 source_log_likelihood=log_likelihood, parameter_count=parameter_count,
                                 reason="best_finite_iterate" if finite else "nonfinite")
                    if finite:
                        order = np.lexsort(tuple(template["means"][:, j] for j in reversed(range(self.dimension))))
                        weights = template["weights"][order]
                        template = dict(weights=_readonly(weights/weights.sum()),
                                        means=_readonly(template["means"][order]),
                                        variances=_readonly(template["variances"][order]))
                        candidates.append((float(bic), count, template))
                else:
                    trial.update(reason="no_finite_initialization", runs=runs)
                trials.append(trial)
            if not candidates:
                raise RuntimeError(f"No finite source GMM for class {label}: {trials}")
            bic, count, template = min(candidates, key=lambda item: (item[0], item[1]))
            templates.append(template)
            reports.append(dict(class_index=label, selected_components=count, selected_bic=bic,
                                trials=trials, selection="minimum_source_BIC_then_smaller_K",
                                bic_formula="-2*sum_source_log_likelihood + (K*(2*d+1)-1)*log(n_class)"))
        return tuple(templates), reports

    def _reduced_parameters(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (self._size,) or not np.isfinite(parameters).all():
            raise ValueError("Expected finite affine/prior mixture parameters")
        d = self.dimension
        latent = softmax(np.append(parameters[2*d:], 0.))
        prior = self.config.prior_floor+(1-self.classes*self.config.prior_floor)*latent
        return parameters[:d], parameters[d:2*d], latent, prior

    def class_prior(self, parameters):
        return self._reduced_parameters(parameters)[-1]

    def _unpack(self, parameters):
        """Legacy CDF-only representation with components flattened across classes."""
        log_scale, shift, latent, prior = self._reduced_parameters(parameters)
        component_mass = prior[self._component_classes]*self._component_weights
        return (self._component_means, self._component_log_variances, log_scale,
                shift, latent, component_mass)

    def _density_terms(self, points):
        """Per-class source log densities and gradients with respect to points."""
        log_density, scores = [], []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for template in self.source_templates:
                difference = points[:, None, :]-template["means"][None, :, :]
                component_log = (np.log(template["weights"])[None, :]
                    -.5*(np.log(2*np.pi*template["variances"])[None, :, :]
                           + difference**2/template["variances"][None, :, :]).sum(axis=2))
                normalizer = logsumexp(component_log, axis=1)
                responsibilities = np.exp(component_log-normalizer[:, None])
                score = -(responsibilities[:, :, None]*difference/template["variances"][None, :, :]).sum(axis=1)
                log_density.append(normalizer)
                scores.append(score)
        return np.column_stack(log_density), np.stack(scores, axis=1)

    def _target_terms(self, parameters):
        """Internal mixture terms; the final entry is the [n_target,C] posterior."""
        log_scale, shift, latent, prior = self._reduced_parameters(parameters)
        scaled_x = self.target*np.exp(log_scale)
        transformed = scaled_x+shift
        class_log, class_score = self._density_terms(transformed)
        joint_log = class_log+np.log(prior)
        normalizer = logsumexp(joint_log, axis=1)
        posterior = np.exp(joint_log-normalizer[:, None])
        return log_scale, shift, latent, prior, scaled_x, class_log, class_score, normalizer, posterior

    def _observation(self, parameters):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            log_scale, _, latent, prior, scaled_x, _, score, normalizer, posterior = self._target_terms(parameters)
            total = len(self.source)+len(self.target)
            loss = (self._source_conditional_nll-normalizer.sum()-len(self.target)*log_scale.sum())/total
            marginal_score = (posterior[:, :, None]*score).sum(axis=1)
            gscale = -(marginal_score*scaled_x).sum(axis=0)-len(self.target)
            gshift = -marginal_score.sum(axis=0)
            gprior = self._prior_gradient(-posterior.sum(axis=0)/prior, latent)
            gradient = np.concatenate((gscale, gshift, gprior))/total
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite mixture observation loss/gradient")
        return float(loss), gradient

    def _separation(self, parameters, correction):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _, _, latent, prior, scaled_x, _, score, _, posterior = self._target_terms(parameters)
            expectation = (posterior*correction).sum(axis=1)
            weights = posterior*(correction-expectation[:, None])/len(self.target)
            weighted_score = (weights[:, :, None]*score).sum(axis=1)
            gscale = (weighted_score*scaled_x).sum(axis=0)
            gshift = weighted_score.sum(axis=0)
            gprior = self._prior_gradient(weights.sum(axis=0)/prior, latent)
            gradient = np.concatenate((gscale, gshift, gprior))
            value = float(expectation.mean())
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite mixture separation objective/gradient")
        return value, gradient

    def coordinate_cdf(self, parameters, coordinate, values):
        """Target-coordinate marginal CDF in the projected density coordinates."""
        if (isinstance(coordinate, (bool, np.bool_)) or int(coordinate) != coordinate
                or not 0 <= coordinate < self.dimension):
            raise ValueError("Invalid projected coordinate")
        values = np.asarray(values, dtype=np.float64)
        if np.isnan(values).any():
            raise ValueError("CDF query contains NaN")
        means, log_variances, log_scale, shift, _, masses = self._unpack(parameters)
        standardized = ((np.exp(log_scale[coordinate])*values.reshape(-1, 1)+shift[coordinate]
                         -means[:, coordinate])/np.exp(.5*log_variances[:, coordinate]))
        return np.clip(ndtr(standardized)@masses, 0., 1.).reshape(values.shape)

    def marginal_cdf(self, parameters, features):
        """Return per-coordinate CDFs for an [n,density_dimension] matrix."""
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("CDF matrix must use projected density coordinates")
        return np.column_stack([self.coordinate_cdf(parameters, j, values[:, j]) for j in range(self.dimension)])

    def fit(self):
        reference = super().fit()
        report = deepcopy(self.fit_report or {})
        report.pop("variance_upper_multiplier", None)
        report.update(family="source_only_BIC_diagonal_GMM_affine_v1",
            parameter_layout=["log_scales", "shifts", "prior_logits_except_last"],
            parameter_sizes=[self.dimension, self.dimension, self.classes-1], parameter_dimension=self._size,
            lower_bounds=self._lower.tolist(), upper_bounds=self._upper.tolist(),
            source_component_counts=list(self.source_component_counts),
            source_templates=[{key: value.tolist() for key, value in template.items()} for template in self.source_templates],
            source_template_report=deepcopy(self.source_template_report),
            component_count_grid=list(self.component_count_grid), mixture_n_init=self.mixture_n_init,
            mixture_maxiter=self.mixture_maxiter, mixture_tol=self.mixture_tol,
            mixture_reg_covar=self.mixture_reg_covar,
            covariance_regularization="weighted population residual variance + reg_covar; not max-floor",
            mixture_backend="numpy_scipy_diagonal_EM_v1", mixture_numpy_version=np.__version__,
            mixture_initialization="source-seeded k-means++ centers, uniform masses, global diagonal variance",
            mixture_finite_policy="best finite iterate across all initializations; report convergence separately",
            source_component_estimation_uses_target=False, source_anchor_uncertainty_included=False,
            source_conditional_nll_constant=True,
            source_mean_conditional_nll=self._source_conditional_nll/len(self.source),
            observation_denominator=len(self.source)+len(self.target),
            observation_loss_definition="(sum_source_conditional_NLL + sum_target_marginal_NLL) / "
                                        "(n_source + n_target)",
            density_dimension=self.dimension, correction_dimension=self.full_dimension,
            projection=deepcopy(self.projection_report))
        self.fit_report = report
        return reference


def check_mixture_explanations():
    """Tiny generated-data regressions; no cached graph or target labels."""
    import json

    from tsba.shared_update.config import SharedUpdateConfig

    rng = np.random.default_rng(2027)
    cfg = SharedUpdateConfig(seed=41, inner_starts=2, inner_maxiter=150, tau=.02)
    labels = np.repeat([0, 1], 24)
    source = rng.normal(size=(48, 5))*.6+(2*labels[:, None]-1)*np.array([.8, .4, .2, 0., 0.])
    target, normal = rng.normal(size=(15, 5)), np.array([1., .3, -.2, 0., .1])
    single = MixtureExplanationSet(source, labels, target, cfg, source_head_normal=normal,
                                  component_counts=(1,), mixture_n_init=1)
    gaussian = ProjectedExplanationSet(source, labels, target, cfg, source_head_normal=normal)
    theta = single._initial+np.array([.03, -.05, .02, -.01, .04])
    full_theta = np.r_[np.concatenate([t["means"] for t in single.source_templates]).ravel(),
                      np.log(np.concatenate([t["variances"] for t in single.source_templates])).ravel(), theta]
    correction = rng.normal(size=(len(target), 2))
    max_error = 0.
    for name, extra in [("_observation", ()), ("_separation", (correction,))]:
        value, grad = getattr(single, name)(theta, *extra)
        expected, expected_grad = getattr(gaussian, name)(full_theta, *extra)
        np.testing.assert_allclose(value, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(grad, expected_grad[8:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(single._target_terms(theta)[-1], gaussian._target_terms(full_theta)[-1], atol=1e-12, rtol=0)
    assert single._candidate(theta).statistic.shape == (2, 6)

    # Multimodal source templates must be fitted even for identity 2D input.
    labels2 = np.repeat([0, 1], 80)
    components = np.tile(np.repeat([-1., 1.], 40), 2)
    source2 = rng.normal(scale=.18, size=(160, 2))
    source2[:, 0] += 1.8*components+.8*labels2
    source2[:, 1] += 1.5*(2*labels2-1)
    target2 = rng.normal(size=(19, 2))
    family = MixtureExplanationSet(source2, labels2, target2, cfg, source_head_normal=np.array([1., 0.]))
    assert family.source_component_counts == (2, 2)
    assert all(len(trial["runs"]) == 5 for report in family.source_template_report for trial in report["trials"])
    theta2 = family._initial+np.array([.04, -.03, .02, .01, -.08])
    correction2 = rng.normal(size=(19, 2))
    for name, extra in [("_observation", ()), ("_separation", (correction2,))]:
        function = getattr(family, name)
        _, grad = function(theta2, *extra)
        numeric = [(function(theta2+1e-6*axis, *extra)[0]-function(theta2-1e-6*axis, *extra)[0])/2e-6
                   for axis in np.eye(len(theta2))]
        np.testing.assert_allclose(grad, numeric, atol=2e-6, rtol=2e-5)
        max_error = max(max_error, float(np.max(np.abs(grad-numeric))))
    reference = family.fit()
    found = family.search(correction2, [reference])
    assert found is not None and found.observation_loss <= family.threshold+cfg.feasibility_tol
    assert found.statistic.shape == (2, 3)
    for coordinate in range(2):
        grid = np.r_[-np.inf, np.linspace(-8, 8, 61), np.inf]
        cdf = family.coordinate_cdf(reference.parameters, coordinate, grid)
        assert np.all(np.diff(cdf) >= -1e-14) and abs(cdf[0]) < 1e-14 and abs(cdf[-1]-1) < 1e-14
        means, logvar, logscale, shift, _, weights = family._unpack(reference.parameters)
        expected = ndtr((np.exp(logscale[coordinate])*grid[:, None]+shift[coordinate]-means[:, coordinate])
                        /np.exp(.5*logvar[:, coordinate]))@weights
        np.testing.assert_allclose(cdf, expected, atol=1e-14, rtol=0)
    replay = MixtureExplanationSet(source2, labels2, rng.normal(size=(7, 2)), cfg,
                                  source_head_normal=np.array([1., 0.]))
    assert replay.source_component_counts == family.source_component_counts
    for template, other in zip(family.source_templates, replay.source_templates):
        for key in template:
            np.testing.assert_array_equal(template[key], other[key])
    assert replay._target_terms(reference.parameters)[-1].shape == (7, 2)
    assert family.marginal_cdf(reference.parameters, target2).shape == target2.shape
    json.dumps(family.fit_report, allow_nan=False)
    return dict(passed=True, single_component_equivalent_gaussian_parity=True,
                source_component_counts=list(family.source_component_counts),
                optimized_parameters=family._size, maximum_gradient_error=max_error,
                full_statistic_shapes=[[2, 6], [2, 3]], source_only_selection_and_replay=True,
                legacy_CDF_component_unpack_exact=True, CDF_limits_monotonicity=True,
                search_status=family.last_search["status"], covariance_regularization="additive reg_covar")
