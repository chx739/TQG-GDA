"""V9 density-only angular ablation of V8 source class GMMs plus mass.

For original source-standardized x=(h-mean)/scale, let r=x+mean/scale.
The density model sees r/max(||r||_2, coordinate_floor): the raw h=0 origin
in the frozen source diagonal metric. Source projection and GMM estimation
use this view. B0, full update features, moments, risk and B0 mass do not.
Positive per-row raw rescaling is invariant above the numerical norm floor,
for fixed source mean/scale. This does not claim invariance if those source
statistics are re-estimated. Dimensions <=2 bypass the entire transform so
the original analytic S worlds retain their exact V8 observation model.
"""

from copy import deepcopy

import numpy as np

from .posterior_mass_explanations import MixtureMassExplanationSet
from .projected_explanations import _fingerprint, _readonly


_ANGULAR_VERSION = "angular_raw_origin_v1"


def _stable_angular_rows(values, floor):
    """Compute r/max(norm(r),floor) without squaring huge/tiny raw values."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Angular coordinates require a finite feature matrix")
    largest = np.max(np.abs(values), axis=1)
    scaled = np.divide(values, largest[:, None], out=np.zeros_like(values), where=largest[:, None] > 0)
    scaled_norm = np.sqrt(np.sum(scaled*scaled, axis=1))
    # Comparing max(abs(r)) to floor/norm(scaled) avoids forming an
    # overflowing norm. Rows inside the floor keep r/floor, including zero.
    limit = np.divide(floor, scaled_norm, out=np.full_like(scaled_norm, np.inf), where=scaled_norm > 0)
    active = largest > limit
    answer = np.empty_like(values)
    answer[active] = scaled[active]/scaled_norm[active, None]
    answer[~active] = values[~active]/floor
    if not np.isfinite(answer).all():
        raise FloatingPointError("Non-finite angular density coordinates")
    return answer, ~active


class AngularMixtureMassExplanationSet(MixtureMassExplanationSet):
    """Original full-coordinate inputs; source statistics are mandatory.

    Inherited ``project`` first applies this fixed view before the frozen
    source projection. Inherited ``full_target``, ``features`` and the mass
    constructor still receive the original standardized values. Query replay
    must reuse the original source statistics; it must not standardize H.
    """

    def __init__(self, source_x, source_y, target_x, config, *, source_mean,
                 source_scale, target_posterior_power=1., **mixture_mass_options):
        source = np.asarray(source_x, dtype=np.float64)
        mean, scale = np.asarray(source_mean, dtype=np.float64), np.asarray(source_scale, dtype=np.float64)
        if (source.ndim != 2 or not source.shape[1] or mean.shape != (source.shape[1],)
                or scale.shape != mean.shape or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("Angular density requires finite source mean and positive source scale of matching dimension")
        self.density_source_mean, self.density_source_scale = _readonly(mean), _readonly(scale)
        self.angular_floor = float(config.coordinate_floor)
        if not np.isfinite(self.angular_floor) or self.angular_floor <= 0:
            raise ValueError("Angular coordinate floor must be finite and positive")
        self.angular_active = source.shape[1] > 2
        if (isinstance(target_posterior_power, (bool, np.bool_)) or not np.isscalar(target_posterior_power)
                or not np.isfinite(target_posterior_power) or not 0 < target_posterior_power <= 1):
            raise ValueError("target_posterior_power must be finite and in (0, 1]")
        self.requested_target_posterior_power = float(target_posterior_power)
        self.target_posterior_power = self.requested_target_posterior_power if self.angular_active else 1.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            self.density_origin_offset = _readonly(mean/scale if self.angular_active else np.zeros_like(mean))
        super().__init__(source_x, source_y, target_x, config, **mixture_mass_options)
        if self.angular_active:
            _, floored = _stable_angular_rows(source+self.density_origin_offset, self.angular_floor)
            self.projection_report["input_coordinates"] = "raw-origin angular view in frozen source diagonal metric"
            self.projection_report["source_head_axis_scope"] = (
                "original standardized head normal retained as angular axis; radius discarded only for density")
        else:
            floored = np.zeros(len(source), dtype=bool)
        self.density_coordinate_report = dict(
            version=_ANGULAR_VERSION, active=self.angular_active,
            scope="density_projection_and_class_GMM_only",
            formula="r=x+source_mean/source_scale; a=r/max(norm(r),coordinate_floor)",
            raw_origin="h=0", metric="fixed_source_population_diagonal_scale",
            floor=self.angular_floor, source_norm_floored_rows=int(floored.sum()),
            source_mean=mean.tolist(), source_scale=scale.tolist(),
            origin_offset=self.density_origin_offset.tolist(),
            origin_offset_evaluated=self.angular_active,
            source_statistics_sha256=dict(mean=_fingerprint(mean), scale=_fingerprint(scale)),
            original_source_features_sha256=_fingerprint(self.full_source),
            source_density_features_sha256=_fingerprint(self.density_coordinates(self.full_source)),
            positive_raw_row_scale_invariance="fixed source statistics and both row norms above floor",
            low_dimension_rule="dimension <=2: exact V8 identity; no angular normalization or origin shift",
            update_features="original source-standardized full embedding plus intercept",
            source_logits_and_head_unchanged=True, aggregate_mass_anchor="original_full_target_B0_mean",
            source_statistics_use_target=False)
        self.projection_report["density_coordinates"] = deepcopy(self.density_coordinate_report)

    def density_coordinates(self, full_x):
        values = np.asarray(full_x, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != len(self.density_origin_offset)
                or not np.isfinite(values).all()):
            raise ValueError("Angular density inputs must use finite original standardized features")
        if not self.angular_active:
            return values.copy()
        with np.errstate(over="raise", invalid="raise"):
            shifted = values+self.density_origin_offset
        return _stable_angular_rows(shifted, self.angular_floor)[0]

    def fit(self):
        reference = super().fit()
        self.fit_report["density_coordinates"] = deepcopy(self.density_coordinate_report)
        self.fit_report["projection"] = deepcopy(self.projection_report)
        self.fit_report["source_templates_coordinate_scope"] = (
            "angular density view" if self.angular_active else "original <=2D synthetic coordinates")
        self.fit_report["unpenalized_likelihood_measure"] = (
            "projected angular density; not a likelihood on original raw embeddings"
            if self.angular_active else "original <=2D synthetic density")
        self.fit_report["target_posterior_power"] = dict(
            requested=self.requested_target_posterior_power, effective=self.target_posterior_power,
            active=self.target_posterior_power != 1.,
            formula="q_beta=softmax(beta*(source_class_log_density+log_mixture_weight))",
            joint="original target marginal p_theta(x) times q_beta(y|x)",
            raw_marginal_and_CDF_unchanged=True, raw_marginal_gradient_uses_unpowered_posterior=True,
            separation_and_mass_gradient_include_power=True,
            mixing_prior_scope="raw marginal mixture weights; need not equal integrated q_beta class masses",
            low_dimension_rule="full dimension <=2: effective beta=1, exact previous S behavior",
            classifier_logits_unchanged=True, update_features_unchanged=True,
            mass_anchor="unchanged original full-coordinate B0 probabilities on A")
        self.fit_report["mass_penalty"]["posterior_model"] = (
            "source class GMM target posterior with effective power " + str(self.target_posterior_power))
        return reference


def check_angular_explanations():
    """Focused generated-data fixture for the real-dimensional density view."""
    import json
    from scipy.special import expit
    from tsba.shared_update.config import SharedUpdateConfig
    from .fit import explanation_family
    from .separation_audit import independent_search

    rng = np.random.default_rng(919)
    dimension, source_count, query_count = 16, 64, 17
    labels = np.repeat([0, 1], source_count//2)
    raw_source = rng.normal(size=(source_count, dimension))+np.linspace(-.7, 1.1, dimension)
    raw_source[:, 0] += .7*(2*labels-1)
    raw_target = rng.normal(size=(query_count, dimension))+.4
    mean, scale = raw_source.mean(0), raw_source.std(0)
    source, target = (raw_source-mean)/scale, (raw_target-mean)/scale
    normal, bias = rng.normal(size=dimension)*.2, .13
    cfg = SharedUpdateConfig(seed=49, inner_starts=2, inner_maxiter=100, tau=.015)
    options = dict(source_head_normal=normal, source_head_bias=bias,
                   component_counts=(1,), mixture_n_init=1)
    family = AngularMixtureMassExplanationSet(source, labels, target, cfg,
        source_mean=mean, source_scale=scale, **options)
    raw = MixtureMassExplanationSet(source, labels, target, cfg, **options)
    np.testing.assert_array_equal(family.features, np.column_stack((target, np.ones(query_count))))
    np.testing.assert_array_equal(family.full_source, source)
    np.testing.assert_array_equal(family.full_target, target)
    np.testing.assert_array_equal(family.mass_reference_probabilities, raw.mass_reference_probabilities)
    np.testing.assert_array_equal(family.mass_reference_probabilities[1], expit(target@normal+bias).mean())
    expected = raw_target/scale
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(family.density_coordinates(target), expected, atol=3e-16, rtol=1e-14)
    factors = np.geomspace(.01, 100., query_count)[:, None]
    rescaled = (raw_target*factors-mean)/scale
    np.testing.assert_allclose(family.density_coordinates(rescaled), expected, atol=5e-14, rtol=1e-12)
    np.testing.assert_allclose(family.project(rescaled), family.target, atol=5e-13, rtol=1e-12)
    np.testing.assert_allclose(family.density_coordinates((-mean/scale)[None]), 0., atol=0, rtol=0)
    extreme, floors = _stable_angular_rows(np.array([[1e300, -1e300], [1e-300, 0.], [0., 0.]]), cfg.coordinate_floor)
    np.testing.assert_allclose(extreme[0], [1/np.sqrt(2), -1/np.sqrt(2)], atol=1e-15, rtol=0)
    np.testing.assert_allclose(extreme[1], [1e-294, 0.], atol=0, rtol=1e-14)
    np.testing.assert_array_equal(floors, [False, True, True])
    theta = family._initial+np.array([.04, -.03, .02, -.01, .12])
    candidate = family._candidate(theta)
    q = family._target_terms(theta)[-1]
    np.testing.assert_allclose(candidate.statistic, q.T@family.features/query_count, atol=1e-15, rtol=0)
    assert not np.allclose(candidate.statistic[:, :-1], q.T@family.density_coordinates(target)/query_count)
    correction = rng.normal(size=(query_count, 2))
    maximum_error = 0.
    for function in (family._observation, lambda p: family._separation(p, correction)):
        _, gradient = function(theta)
        numerical = [(function(theta+1e-6*axis)[0]-function(theta-1e-6*axis)[0])/2e-6 for axis in np.eye(len(theta))]
        maximum_error = max(maximum_error, float(np.max(np.abs(gradient-numerical))))
        np.testing.assert_allclose(gradient, numerical, atol=5e-7, rtol=2e-5)
    reference = family.fit()
    search = family.search(correction, [reference])
    assert search is not None and search.observation_loss <= family.threshold+cfg.feasibility_tol
    candidates, independent = independent_search(family, correction, seed=7050, starts=2, maxiter=80, known=[reference])
    assert candidates and independent['fresh_optimizer_launches'] == 2
    for candidate in candidates:
        assert candidate.observation_loss <= family.threshold+cfg.feasibility_tol
        q = family._target_terms(candidate.parameters)[-1]
        np.testing.assert_allclose(candidate.statistic, q.T@family.features/query_count, atol=1e-15, rtol=0)
    plan = dict(separation_variant='mixture_mass_v1', density_coordinates=_ANGULAR_VERSION,
                source_mixture_options=dict(component_counts=[1], mixture_n_init=1),
                posterior_mass_options=dict(mass_weight=1., mass_floor=1e-8))
    v0 = np.stack((-np.r_[normal, bias]/2, np.r_[normal, bias]/2))
    holdout = rng.normal(size=(11, dimension))
    replay = explanation_family(plan, source, labels, holdout, cfg, v0, source_mean=mean, source_scale=scale)
    assert replay.projection_report == family.projection_report
    np.testing.assert_array_equal(replay.project(target), family.target)
    assert replay._target_terms(reference.parameters)[-1].shape == (len(holdout), 2)
    assert replay.marginal_cdf(reference.parameters, replay.target).shape == (len(holdout), 2)
    for a, b in zip(replay.source_templates, family.source_templates):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    # Even finite source statistics whose ratio would overflow are untouched
    # in the <=2D bypass, preserving the original analytic S numerical path.
    small_options = options | dict(source_head_normal=normal[:2])
    small_raw = MixtureMassExplanationSet(source[:, :2], labels, target[:, :2], cfg, **small_options)
    small = AngularMixtureMassExplanationSet(source[:, :2], labels, target[:, :2], cfg,
        source_mean=np.array([1e300, -1e300]), source_scale=np.array([1e-300, 1e-300]), **small_options)
    a, b = small_raw.fit(), small.fit()
    np.testing.assert_array_equal(a.parameters, b.parameters)
    np.testing.assert_array_equal(a.statistic, b.statistic)
    assert a.observation_loss == b.observation_loss
    np.testing.assert_array_equal(small._observation(theta)[1], small_raw._observation(theta)[1])
    np.testing.assert_array_equal(small.mass_reference_probabilities, small_raw.mass_reference_probabilities)
    assert not small.density_coordinate_report['origin_offset_evaluated']
    assert not small.density_coordinate_report['active']
    for bad_scale in (np.zeros(dimension), np.full(dimension, np.inf), -scale):
        try:
            AngularMixtureMassExplanationSet(source, labels, target, cfg,
                source_mean=mean, source_scale=bad_scale, **options)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid source scale accepted')
    json.dumps(family.fit_report, allow_nan=False)
    return dict(passed=True, full_dimension=dimension, positive_row_scale_invariant=True,
                raw_origin_verified=True, original_B0_mass_and_full_moments=True,
                source_only_projection_and_GMM_replay=True, exact_2D_bypass=True,
                independent_search_verified=True, maximum_gradient_error=maximum_error)
