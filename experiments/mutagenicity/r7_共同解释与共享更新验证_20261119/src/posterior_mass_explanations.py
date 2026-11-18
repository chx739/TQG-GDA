"""Posterior-anchored explanations with a fixed, label-free mass penalty.

The observation score is raw joint NLL plus
    n_A / (n_S + n_A) * weight * KL(smooth(mean_A(q_B0)) || smooth(mean_A(q_theta))).
For binary probabilities, smooth(p) = eps + (1 - 2*eps)*p, componentwise.
This is an aggregate fractional-class-count penalty with one pseudo class
observation per adaptation graph at weight=1, up to its fixed entropy term.
It is not per-graph pseudo-label cross entropy, an independent likelihood, or
evidence that B0's target class mass is correct. All target inputs are unlabeled.

Only compatibility/fitting changes. Separation retains the unpenalized risk
and full original update statistics; independent search therefore uses the
same augmented constraint without contaminating its raw risk objective.
"""

from copy import deepcopy

import numpy as np
from scipy.special import rel_entr

from .posterior_anchored_explanations import PosteriorAnchoredExplanationSet
from .projected_explanations import _readonly


class PosteriorMassExplanationSet(PosteriorAnchoredExplanationSet):
    """Same family parameters and observer APIs, augmented observation score.

    ``mass_weight=0`` takes the exact parent observation route. The fixed mass
    anchor is the frozen B0 posterior averaged on this instance's adaptation
    inputs; it is distinct from the source-density normalizer and from the
    candidate generative prior. Replay for a new query set must retain the
    original adaptation instance when evaluating compatibility.
    """

    def __init__(self, source_x, source_y, target_x, config, *, mass_weight=1.,
                 mass_floor=1e-8, **posterior_anchor_kwargs):
        if (isinstance(mass_weight, (bool, np.bool_)) or not np.isscalar(mass_weight)
                or not np.isfinite(mass_weight) or mass_weight < 0):
            raise ValueError("mass_weight must be finite and nonnegative")
        if (isinstance(mass_floor, (bool, np.bool_)) or not np.isscalar(mass_floor)
                or not np.isfinite(mass_floor) or not 0 < mass_floor < .5):
            raise ValueError("mass_floor must be finite and strictly between zero and one half")
        self.mass_weight, self.mass_floor = float(mass_weight), float(mass_floor)
        super().__init__(source_x, source_y, target_x, config, **posterior_anchor_kwargs)
        class_one = float(self.source_posterior(self.target)[:, 1].mean())
        self.mass_reference_probabilities = _readonly([1-class_one, class_one])
        self.mass_reference_smoothed_probabilities = _readonly(self._smooth(self.mass_reference_probabilities))
        self.mass_coefficient = float(len(self.target)/(len(self.source)+len(self.target))*self.mass_weight)
        correction = np.zeros((len(self.target), self.classes), dtype=np.float64)
        correction[:, 1] = 1.
        self._mass_correction = _readonly(correction)

    def _smooth(self, probabilities):
        return self.mass_floor+(1-2*self.mass_floor)*probabilities

    def _mass_penalty(self, parameters):
        # This calls the parent's pure linear-risk functional, never an
        # augmented separation objective. Its gradient is d(mean(q_1))/dtheta.
        mean_one, mean_gradient = super()._separation(parameters, self._mass_correction)
        probabilities = np.array([1-mean_one, mean_one])
        smoothed = self._smooth(probabilities)
        base = self.mass_reference_smoothed_probabilities
        if np.any(smoothed <= 0) or not np.isfinite(smoothed).all():
            raise FloatingPointError("Non-finite or nonpositive smoothed posterior mass")
        divergence = float(np.sum(rel_entr(base, smoothed)))
        # The true divergence is nonnegative; remove only cancellation at
        # equality. Its analytic derivative remains that of the smooth KL.
        if divergence < -1e-12:
            raise FloatingPointError("Numerically negative posterior mass KL")
        divergence = max(0., divergence)
        derivative = ((smoothed[1]-base[1])/(smoothed[0]*smoothed[1])
                      *(1-2*self.mass_floor))
        gradient = derivative*mean_gradient
        if not np.isfinite(divergence) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite posterior mass KL or gradient")
        return divergence, gradient, probabilities, smoothed

    def _observation(self, parameters):
        raw, raw_gradient = super()._observation(parameters)
        if self.mass_weight == 0:
            return raw, raw_gradient
        divergence, gradient, _, _ = self._mass_penalty(parameters)
        return raw+self.mass_coefficient*divergence, raw_gradient+self.mass_coefficient*gradient

    def observation_components(self, parameters):
        """Auditable raw/augmented scores; no mutation or target-label input."""
        raw, _ = super()._observation(parameters)
        divergence, _, probabilities, smoothed = self._mass_penalty(parameters)
        penalty = self.mass_coefficient*divergence
        return dict(raw_observation_loss=float(raw), augmented_observation_loss=float(raw+penalty),
                    mass_kl=float(divergence), weighted_mass_penalty=float(penalty),
                    posterior_mean=probabilities.tolist(), smoothed_posterior_mean=smoothed.tolist(),
                    class_prior=self.class_prior(parameters).tolist())

    def fit(self):
        reference = super().fit()
        report = deepcopy(self.fit_report)
        components = self.observation_components(reference.parameters)
        identity = self.observation_components(self.identity_parameters)
        report.update(
            family="posterior_anchored_pooled_source_GMM_mass_affine_v1",
            observation_loss_kind="augmented_raw_joint_NLL_plus_fixed_B0_target_mass_KL",
            compatibility_loss="augmented_observation_loss",
            raw_reference_loss=components["raw_observation_loss"],
            augmented_reference_loss=components["augmented_observation_loss"],
            reference_observation_components=components,
            mass_penalty=dict(
                weight=self.mass_weight, floor=self.mass_floor, coefficient=self.mass_coefficient,
                target_count=len(self.target), source_count=len(self.source),
                effective_pseudo_observation_count=self.mass_weight*len(self.target),
                smoothing="eps + (1 - 2*eps) * probability",
                reference_probabilities=self.mass_reference_probabilities.tolist(),
                reference_smoothed_probabilities=self.mass_reference_smoothed_probabilities.tolist(),
                reference_origin="frozen_B0_posterior_mean_on_unlabeled_adaptation_inputs",
                interpretation="aggregate fractional counts; fixed entropy omitted; not independent data",
                identity_kl=identity["mass_kl"],
                target_labels_used=False,
                separation_objective="unchanged unpenalized linear risk",
                threshold="augmented reference loss + unchanged tau",
                raw_likelihood_has_no_separate_compatibility_threshold=True,
                zero_weight_exact_parent_observation=self.mass_weight == 0))
        self.fit_report = report
        return reference


def check_posterior_mass_explanations():
    """Small generated-data regression/gradient/search test; no target labels."""
    import json
    from tsba.shared_update.config import SharedUpdateConfig
    from .separation_audit import independent_search

    rng = np.random.default_rng(611)
    labels = np.repeat([0, 1], 24)
    source = rng.normal(size=(48, 3))*.8
    source[:, 0] += (2*labels-1)*.55
    target = rng.normal(size=(15, 3))
    normal = np.array([.8, -.2, .4])
    config = SharedUpdateConfig(seed=27, inner_starts=2, inner_maxiter=100, tau=.015)
    options = dict(source_head_normal=normal, source_head_bias=.1,
                   component_counts=(1,), mixture_n_init=1)
    parent = PosteriorAnchoredExplanationSet(source, labels, target, config, **options)
    family = PosteriorMassExplanationSet(source, labels, target, config, **options)
    zero = PosteriorMassExplanationSet(source, labels, target, config, mass_weight=0., **options)
    theta = family.identity_parameters+np.array([.09, -.04, .06, -.03, .23])
    correction = rng.normal(size=(len(target), 2))
    maximum_error = 0.
    for parameters in (family.identity_parameters, theta):
        np.testing.assert_array_equal(zero._observation(parameters)[0], parent._observation(parameters)[0])
        np.testing.assert_array_equal(zero._observation(parameters)[1], parent._observation(parameters)[1])
        np.testing.assert_array_equal(family._separation(parameters, correction)[0], parent._separation(parameters, correction)[0])
        np.testing.assert_array_equal(family._separation(parameters, correction)[1], parent._separation(parameters, correction)[1])
        for function in (family._observation, lambda p: family._mass_penalty(p)[:2]):
            _, gradient = function(parameters)
            finite_difference = np.array([(function(parameters+1e-6*axis)[0]-function(parameters-1e-6*axis)[0])/2e-6
                                          for axis in np.eye(len(parameters))])
            np.testing.assert_allclose(gradient, finite_difference, atol=5e-7, rtol=2e-5)
            maximum_error = max(maximum_error, float(np.max(np.abs(gradient-finite_difference))))
    identity = family.observation_components(family.identity_parameters)
    assert identity["mass_kl"] <= 1e-14
    np.testing.assert_allclose(family._mass_penalty(family.identity_parameters)[1], 0., atol=1e-14, rtol=0)
    assert family.observation_components(theta)["mass_kl"] > 0.
    # Verify the declared fractional-count/denominator coefficient directly.
    np.testing.assert_allclose(family._observation(theta)[0]-parent._observation(theta)[0],
        len(target)/(len(source)+len(target))*family._mass_penalty(theta)[0], atol=5e-16, rtol=1e-12)
    reference = family.fit()
    assert reference.observation_loss == family.reference_loss
    assert family.threshold == family.reference_loss+config.tau
    assert family.observation_components(reference.parameters)["augmented_observation_loss"] == family.reference_loss
    candidate = family.search(correction, [reference])
    assert candidate is not None and candidate.observation_loss <= family.threshold+config.feasibility_tol
    assert candidate.statistic.shape == (2, source.shape[1]+1)
    # The actual independent audit deep-copies the subclass and constrains
    # its dynamic _observation; it must not silently revert to raw NLL.
    frozen = (family.reference_loss, family.threshold, reference.parameters.copy())
    candidates, audit = independent_search(family, correction, seed=7028, starts=2, maxiter=80, known=[reference])
    assert candidates and audit["fresh_optimizer_launches"] == 2
    for item in candidates:
        np.testing.assert_allclose(item.observation_loss, family._observation(item.parameters)[0], atol=1e-14, rtol=0)
        assert item.observation_loss <= family.threshold+config.feasibility_tol
    assert (family.reference_loss, family.threshold) == frozen[:2]
    np.testing.assert_array_equal(reference.parameters, frozen[2])
    json.dumps(family.fit_report, allow_nan=False)
    return dict(passed=True, maximum_gradient_error=maximum_error,
                zero_weight_exact_parent_observation=True, identity_KL_zero=True,
                separation_unchanged=True, full_moments=True,
                independent_augmented_search_verified=True,
                independent_fresh_launches=audit["fresh_optimizer_launches"],
                reference_raw_loss=family.fit_report["raw_reference_loss"],
                reference_augmented_loss=family.reference_loss)
