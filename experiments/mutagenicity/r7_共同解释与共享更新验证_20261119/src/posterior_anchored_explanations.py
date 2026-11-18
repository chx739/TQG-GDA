"""Frozen B0 posterior with a pooled source marginal and common affine shift.

Let g be the P5 source templates pooled with empirical source class masses.
The class densities are f_c(r)=g(r)q0(c|r)/pi0[c], where pi0=integral(g*q0).
Only a positive diagonal affine map and target class prior are fitted.  This
preserves q0 exactly at the identity map/prior pi0. Numerical normalization
uses deterministic GH quadrature; its refinement checks are not rigorous
integration error bounds. No source or target labels choose quadrature order.
"""

from copy import deepcopy
from functools import lru_cache

import numpy as np
from scipy.special import expit, log_expit, logsumexp, ndtr, ndtri, roots_hermitenorm, roots_legendre

from .mixture_explanations import MixtureExplanationSet, _component_log_density
from .projected_explanations import _readonly


@lru_cache(maxsize=None)
def _normal_quadrature(order):
    nodes, weights = roots_hermitenorm(order)
    return _readonly(nodes), _readonly(weights/weights.sum())


@lru_cache(maxsize=None)
def _unit_quadrature(order):
    nodes, weights = roots_legendre(order)
    return _readonly((nodes+1)/2), _readonly(weights/2)


def _orders(values, minimum):
    values = tuple(values)
    if (len(values) < minimum or any(isinstance(x, (bool, np.bool_)) or int(x) != x or x < 4 for x in values)
            or any(a >= b for a, b in zip(values, values[1:]))):
        raise ValueError("Quadrature orders must be strictly increasing positive integers >=4")
    return tuple(int(x) for x in values)


class PosteriorAnchoredExplanationSet(MixtureExplanationSet):
    """Same affine/prior vector and full moments; additionally requires bias.

    source_head_normal and source_head_bias are the B0 binary logit margin
    coefficients in FULL source-standardized input coordinates, not raw h.
    coordinate_cdf consumes projected density coordinates. _unpack rejects
    the legacy Gaussian representation: logistic tilting is not a Gaussian
    mixture with merely changed component weights.
    """

    def __init__(self, source_x, source_y, target_x, config, *, source_head_normal,
                 source_head_bias, fisher_regularization=.1, rank_tolerance=1e-10,
                 component_counts=(1, 2, 4), mixture_n_init=5, mixture_maxiter=300,
                 mixture_tol=1e-6, mixture_reg_covar=None,
                 normalization_orders=(16, 32, 64, 128, 256, 512, 1024),
                 normalization_tolerance=1e-10, cdf_orders=(32, 64, 128, 256, 512),
                 cdf_tolerance=1e-6):
        if not np.array_equal(np.unique(source_y), [0, 1]) or not np.isfinite(source_head_bias):
            raise ValueError("Posterior anchoring requires binary source labels and finite margin bias")
        self.normalization_orders = _orders(normalization_orders, 3)
        self.cdf_orders = _orders(cdf_orders, 2)
        if (not np.isfinite([normalization_tolerance, cdf_tolerance]).all()
                or normalization_tolerance <= 0 or cdf_tolerance <= 0):
            raise ValueError("Quadrature tolerances must be finite and positive")
        self.normalization_tolerance, self.cdf_tolerance = float(normalization_tolerance), float(cdf_tolerance)
        self.source_head_bias = float(source_head_bias)
        super().__init__(source_x, source_y, target_x, config,
            source_head_normal=source_head_normal, fisher_regularization=fisher_regularization,
            rank_tolerance=rank_tolerance, component_counts=component_counts,
            mixture_n_init=mixture_n_init, mixture_maxiter=mixture_maxiter,
            mixture_tol=mixture_tol, mixture_reg_covar=mixture_reg_covar)
        normal = np.asarray(source_head_normal, dtype=np.float64)
        projected_normal = normal@self.projection_basis
        if self.full_dimension > 2 and np.any(normal):
            # Axis one was explicitly constructed from this normal. Remove
            # reorthogonalization roundoff; no new second-axis ranking exists.
            projected_normal = np.zeros(self.dimension)
            projected_normal[0] = float(normal@self.projection_basis[:, 0])
        self.posterior_normal = _readonly(projected_normal*self.projection_scale)
        self.posterior_bias = float(source_head_bias+projected_normal@self.projection_mean)
        empirical = np.bincount(self.labels, minlength=2)/len(self.labels)
        self.source_empirical_prior = _readonly(empirical)
        self.pooled_component_weights = _readonly(empirical[self._component_classes]*self._component_weights)
        self._pooled_variances = _readonly(np.exp(self._component_log_variances))
        self._margin_means = self._component_means@self.posterior_normal+self.posterior_bias
        self._margin_variances = self._pooled_variances@(self.posterior_normal**2)
        self.source_prior_normalizer, self._component_posterior_prior, self.normalization_report = self._normalizers()
        self._log_source_prior = _readonly(np.log(self.source_prior_normalizer))
        latent = (self.source_prior_normalizer-self.config.prior_floor)/(1-2*self.config.prior_floor)
        if np.any(latent <= 0):
            raise ValueError("B0 identity prior lies outside the fixed target prior floor")
        identity = np.r_[np.zeros(2*self.dimension), np.log(latent[0]/latent[1])]
        if np.any(identity < self._lower) or np.any(identity > self._upper):
            raise ValueError("B0 identity prior lies outside the fixed target prior-logit bounds")
        self.identity_parameters = _readonly(identity)
        self._initial = identity.copy()
        source_log_density, _ = self._pooled_terms(self.source)
        source_log_q = self._log_q0(self.source)
        self._source_conditional_nll = -float((source_log_density
            +source_log_q[np.arange(len(self.source)), self.labels]-self._log_source_prior[self.labels]).sum())
        self.cdf_reports = {}
        expected = self.full_target@normal+self.source_head_bias
        recovered = self.target@self.posterior_normal+self.posterior_bias
        if not np.allclose(expected, recovered, atol=1e-10, rtol=1e-12):
            raise ValueError("Projected margin does not reproduce the frozen B0 head")

    def _normalizers(self):
        history, previous, consecutive = [], None, 0
        for order in self.normalization_orders:
            nodes, weights = _normal_quadrature(order)
            margins = self._margin_means[:, None]+np.sqrt(self._margin_variances)[:, None]*nodes
            components = np.column_stack((expit(-margins)@weights, expit(margins)@weights))
            prior = self.pooled_component_weights@components
            delta = None if previous is None else float(np.max(np.abs(components-previous)))
            consecutive = consecutive+1 if delta is not None and delta <= self.normalization_tolerance else 0
            history.append(dict(order=order, source_prior=prior.tolist(), maximum_component_change=delta))
            if consecutive >= 2:
                break
            previous = components
        if consecutive < 2:
            raise RuntimeError(f"Source posterior normalizer quadrature did not converge: {history}")
        prior = prior/prior.sum()
        if not np.isfinite(prior).all() or np.any(prior <= 0):
            raise ValueError("Source posterior prior is nonpositive or nonfinite")
        return _readonly(prior), _readonly(components), dict(
            method="Gauss-Hermite over Gaussian component logit margin; exact 2D-to-1D linear reduction",
            converged=True, tolerance=self.normalization_tolerance, order=order, refinements=history,
            required_consecutive_refinements=2, error_scope="successive-order differences, not rigorous bounds",
            component_class_probabilities=components.tolist(), uses_target=False)

    def _log_q0(self, points):
        margin = points@self.posterior_normal+self.posterior_bias
        return np.column_stack((log_expit(-margin), log_expit(margin)))

    def source_posterior(self, projected_features):
        return np.exp(self._log_q0(np.asarray(projected_features, dtype=np.float64)))

    def _pooled_terms(self, points):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            component_log = _component_log_density(points, self.pooled_component_weights,
                                                   self._component_means, self._pooled_variances)
            density = logsumexp(component_log, axis=1)
            responsibilities = np.exp(component_log-density[:, None])
            score = -(responsibilities[:, :, None]*(points[:, None, :]-self._component_means)
                      /self._pooled_variances).sum(axis=1)
        return density, score

    def _unpack(self, parameters):
        raise NotImplementedError("Logistic-tilted density: use coordinate_cdf, not Gaussian _unpack")

    def _target_terms(self, parameters):
        log_scale, shift, latent, prior = self._reduced_parameters(parameters)
        scaled_x = self.target*np.exp(log_scale)
        transformed = scaled_x+shift
        log_density, score = self._pooled_terms(transformed)
        log_q0 = self._log_q0(transformed)
        adjusted = log_q0+np.log(prior)-self._log_source_prior
        log_adjustment = logsumexp(adjusted, axis=1)
        posterior = np.exp(adjusted-log_adjustment[:, None])
        return (log_scale, shift, latent, prior, scaled_x, log_density, score,
                np.exp(log_q0), log_density+log_adjustment, posterior)

    def _observation(self, parameters):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            log_scale, _, latent, prior, scaled_x, _, score, q0, norm, posterior = self._target_terms(parameters)
            denominator = len(self.source)+len(self.target)
            value = (self._source_conditional_nll-norm.sum()-len(self.target)*log_scale.sum())/denominator
            marginal_score = score+(posterior[:, 1]-q0[:, 1])[:, None]*self.posterior_normal
            scale_grad = -(marginal_score*scaled_x).sum(axis=0)-len(self.target)
            shift_grad = -marginal_score.sum(axis=0)
            prior_grad = self._prior_gradient(-posterior.sum(axis=0)/prior, latent)
            gradient = np.r_[scale_grad, shift_grad, prior_grad]/denominator
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite posterior-anchored likelihood/gradient")
        return float(value), gradient

    def _separation(self, parameters, correction):
        _, _, latent, prior, scaled_x, _, _, _, _, posterior = self._target_terms(parameters)
        influence = posterior[:, 0]*posterior[:, 1]*(correction[:, 1]-correction[:, 0])/len(self.target)
        scores = influence[:, None]*self.posterior_normal
        prior_gradient = self._prior_gradient(np.array([-influence.sum(), influence.sum()])/prior, latent)
        gradient = np.r_[(scores*scaled_x).sum(axis=0), scores.sum(axis=0), prior_gradient]
        value = float(np.mean(np.sum(posterior*correction, axis=1)))
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite posterior-anchored separation/gradient")
        return value, gradient

    def _coordinate_integrals(self, cutoffs, coordinate, order):
        """Return source marginal CDF and joint class-1 truncated integral."""
        marginal, joint = np.zeros(len(cutoffs)), np.zeros(len(cutoffs))
        kinds = set()
        for k, mass in enumerate(self.pooled_component_weights):
            mu, variance = self._component_means[k, coordinate], self._pooled_variances[k, coordinate]
            sigma = np.sqrt(variance)
            f = ndtr((cutoffs-mu)/sigma)
            a = self.posterior_normal[coordinate]
            total_variance = self._margin_variances[k]
            other_variance = float(np.sum(np.delete(self.posterior_normal**2*self._pooled_variances[k], coordinate)))
            if a == 0 or total_variance == 0:
                h = f*self._component_posterior_prior[k, 1]
                kinds.add("independent_coordinate_analytic")
            elif other_variance == 0:
                # Integrate the smaller normal-probability tail. No discrete
                # GH-node CDF and no interpolation are used in this case.
                nodes, weights = _unit_quadrature(order)
                lower = f <= .5
                tail = np.where(lower, f, 1-f)
                h = np.empty_like(f)
                for start in range(0, len(f), 256):
                    sl = slice(start, start+256)
                    probability = tail[sl, None]*nodes
                    sign = np.where(lower[sl], 1., -1.)
                    points = mu+sigma*sign[:, None]*ndtri(probability)
                    margin = a*points+(self._margin_means[k]-a*mu)
                    integrated = tail[sl]*(expit(margin)@weights)
                    h[sl] = np.where(lower[sl], integrated, self._component_posterior_prior[k, 1]-integrated)
                kinds.add("Gauss_Legendre_normal_probability_tail_integral")
            else:
                nodes, weights = _normal_quadrature(order)
                margin = self._margin_means[k]+np.sqrt(total_variance)*nodes
                conditional_mean = mu+a*variance/np.sqrt(total_variance)*nodes
                conditional_sd = np.sqrt(variance*other_variance/total_variance)
                h = np.empty_like(f)
                for start in range(0, len(f), 256):
                    sl = slice(start, start+256)
                    conditional = ndtr((cutoffs[sl, None]-conditional_mean)/conditional_sd)
                    h[sl] = conditional@(weights*expit(margin))
                kinds.add("Gauss_Hermite_margin_conditional_Gaussian_CDF_integral")
            h = np.where(f == 0, 0., np.where(f == 1, self._component_posterior_prior[k, 1], h))
            marginal += mass*f
            joint += mass*h
        return marginal, joint, sorted(kinds)

    def coordinate_cdf(self, parameters, coordinate, values):
        if (isinstance(coordinate, (bool, np.bool_)) or int(coordinate) != coordinate
                or not 0 <= coordinate < self.dimension):
            raise ValueError("Invalid density coordinate")
        values = np.asarray(values, dtype=np.float64)
        if np.isnan(values).any():
            raise ValueError("CDF values contain NaN")
        log_scale, shift, _, prior = self._reduced_parameters(parameters)
        cutoffs = np.exp(log_scale[coordinate])*values.reshape(-1)+shift[coordinate]
        ratio = prior/self.source_prior_normalizer
        previous, history, converged = None, [], False
        for order in self.cdf_orders:
            f, h, kinds = self._coordinate_integrals(cutoffs, coordinate, order)
            answer = ratio[0]*f+(ratio[1]-ratio[0])*h
            change = None if previous is None else float(np.max(np.abs(answer-previous), initial=0.))
            history.append(dict(order=order, maximum_change=change))
            if change is not None and change <= self.cdf_tolerance:
                converged = True
                break
            previous = answer
        answer = np.where(np.isneginf(cutoffs), 0., np.where(np.isposinf(cutoffs), 1., answer))
        bounds_error = float(max(0., -answer.min(initial=0.), answer.max(initial=1.)-1.))
        ordered = answer[np.argsort(cutoffs)]
        monotonic_error = float(max(0., -np.min(np.diff(ordered), initial=0.)))
        report = dict(coordinate=int(coordinate), methods=kinds, approximation="deterministic numerical integration; no interpolation",
                      tolerance=self.cdf_tolerance, order=order, refinements=history,
                      converged=converged and bounds_error <= self.cdf_tolerance and monotonic_error <= self.cdf_tolerance,
                      boundary_values_validated=True, preclip_bounds_error=bounds_error,
                      monotonicity_error=monotonic_error, samples=len(cutoffs),
                      error_scope="successive-order differences, not rigorous bounds")
        self.cdf_reports[str(coordinate)] = report
        if not np.isfinite(answer).all() or bounds_error > 10*self.cdf_tolerance:
            raise FloatingPointError("Numerically invalid conditional marginal CDF")
        return np.clip(answer, 0., 1.).reshape(values.shape)

    def fit(self):
        reference = super().fit()
        report = deepcopy(self.fit_report)
        report.update(family="posterior_anchored_pooled_source_GMM_affine_v1",
            source_templates_role="pooled source marginal only; source component classes do not define q",
            pooled_component_weights=self.pooled_component_weights.tolist(),
            source_empirical_prior=self.source_empirical_prior.tolist(),
            source_prior_normalizer=self.source_prior_normalizer.tolist(),
            posterior_normal=self.posterior_normal.tolist(), posterior_bias=self.posterior_bias,
            source_head_bias=self.source_head_bias, normalization=deepcopy(self.normalization_report),
            identity_parameters=self.identity_parameters.tolist(), identity_posterior="frozen_B0_exact_up_to_roundoff",
            active_posterior_coordinates=np.flatnonzero(self.posterior_normal).tolist(),
            second_axis_changes_ranking=bool(self.dimension > 1 and self.posterior_normal[1] != 0.),
            positive_diagonal_cannot_activate_zero_coefficients=True,
            CDF_representation="logistic-tilted source mixture; use coordinate_cdf numerical integration",
            cdf_orders=list(self.cdf_orders), cdf_tolerance=self.cdf_tolerance)
        self.fit_report = report
        return reference


def check_posterior_anchored_explanations():
    """Generated-data engineering checks; no real inputs or target labels."""
    import json
    from scipy.integrate import quad
    from tsba.shared_update.config import SharedUpdateConfig

    rng = np.random.default_rng(812)
    cfg = SharedUpdateConfig(seed=43, inner_starts=2, inner_maxiter=150, tau=.02)
    labels = np.repeat([0, 1], 22)
    cases, max_error = [], 0.
    for dimension in [2, 5]:
        source = rng.normal(size=(44, dimension))*.7
        source[:, 0] += (2*labels-1)*.6
        target = rng.normal(size=(13, dimension))
        normal = np.linspace(1., .3, dimension)
        family = PosteriorAnchoredExplanationSet(source, labels, target, cfg,
            source_head_normal=normal, source_head_bias=.17, component_counts=(1,), mixture_n_init=1)
        identity = family.identity_parameters
        expected = expit(target@normal+.17)
        np.testing.assert_allclose(family._target_terms(identity)[-1][:, 1], expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(family.class_prior(identity), family.source_prior_normalizer, atol=1e-14, rtol=0)
        # Independent scalar quadrature validates the GH normalizers.
        numeric_prior = 0.
        for mass, mean, variance in zip(family.pooled_component_weights, family._margin_means, family._margin_variances):
            integral = quad(lambda x: np.exp(-x*x/2)/np.sqrt(2*np.pi)*expit(mean+np.sqrt(variance)*x),
                            -np.inf, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
            numeric_prior += mass*integral
        np.testing.assert_allclose(numeric_prior, family.source_prior_normalizer[1], atol=2e-10, rtol=0)
        theta = identity+np.array([.07, -.04, .03, -.02, .11])
        correction = rng.normal(size=(len(target), 2))
        for name, extra in [("_observation", ()), ("_separation", (correction,))]:
            function = getattr(family, name)
            _, analytic = function(theta, *extra)
            numeric = [(function(theta+1e-6*axis, *extra)[0]-function(theta-1e-6*axis, *extra)[0])/2e-6
                       for axis in np.eye(len(theta))]
            np.testing.assert_allclose(analytic, numeric, atol=5e-7, rtol=2e-5)
            max_error = max(max_error, float(np.max(np.abs(analytic-numeric))))
        if dimension > 2:
            assert family.posterior_normal[1] == 0
            assert family._separation(theta, correction)[1][1] == 0
            assert family._separation(theta, correction)[1][3] == 0
        for coordinate in range(2):
            grid = np.r_[-np.inf, np.linspace(-4, 4, 25), np.inf]
            identity_cdf = family.coordinate_cdf(identity, coordinate, grid)
            gaussian = ndtr((grid[:, None]-family._component_means[:, coordinate])
                            /np.sqrt(family._pooled_variances[:, coordinate]))@family.pooled_component_weights
            np.testing.assert_allclose(identity_cdf, gaussian, atol=1e-12, rtol=0)
            cdf = family.coordinate_cdf(theta, coordinate, grid)
            assert cdf[0] == 0 and cdf[-1] == 1 and np.min(np.diff(cdf)) >= -1e-6
            assert family.cdf_reports[str(coordinate)]["converged"]
        # Independently integrate one nonidentity coordinate CDF.
        coordinate, cutoff = 0, .4
        logscale, shift, _, prior = family._reduced_parameters(theta)
        transformed_cutoff = np.exp(logscale[coordinate])*cutoff+shift[coordinate]
        direct = 0.
        nodes, weights = _normal_quadrature(128)
        for k, mass in enumerate(family.pooled_component_weights):
            mu, sd = family._component_means[k, 0], np.sqrt(family._pooled_variances[k, 0])
            other_mean = family.posterior_bias+family.posterior_normal[1]*family._component_means[k, 1]
            other_sd = abs(family.posterior_normal[1])*np.sqrt(family._pooled_variances[k, 1])
            def integrand(x):
                margin = family.posterior_normal[0]*(mu+sd*x)+other_mean+other_sd*nodes
                q = float(expit(margin)@weights)
                ratio = prior/family.source_prior_normalizer
                return np.exp(-x*x/2)/np.sqrt(2*np.pi)*(ratio[0]*(1-q)+ratio[1]*q)
            direct += mass*quad(integrand, -np.inf, (transformed_cutoff-mu)/sd, epsabs=1e-9)[0]
        np.testing.assert_allclose(family.coordinate_cdf(theta, 0, np.array([cutoff])), [direct], atol=3e-6, rtol=0)
        reference = family.fit()
        candidate = family.search(correction, [reference])
        assert candidate is not None and candidate.observation_loss <= family.threshold+cfg.feasibility_tol
        assert candidate.statistic.shape == (2, dimension+1)
        json.dumps(family.fit_report, allow_nan=False)
        json.dumps(family.cdf_reports, allow_nan=False)
        cases.append(dict(full_dimension=dimension, normalization_order=family.normalization_report["order"],
                          posterior_coordinates=np.flatnonzero(family.posterior_normal).tolist(),
                          search_status=family.last_search["status"]))
    return dict(passed=True, identity_B0_invariant=True, maximum_gradient_error=max_error,
                normalizer_independent_quad_verified=True, conditional_CDF_independent_quad_verified=True,
                full_moments=True, no_second_axis_ranking_when_zero=True, cases=cases)
