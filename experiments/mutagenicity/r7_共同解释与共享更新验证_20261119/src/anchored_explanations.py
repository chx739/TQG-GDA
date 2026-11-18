"""Optional projected family with source-only Gaussian class anchors.

Class means and diagonal log variances are fixed at the projected source MLE
(population variance, with the original variance floor).  Only common log
scales, shifts and target prior logits are fitted/searched.  The observation
loss retains the original (n_source + n_target) denominator, and posterior
statistics retain all original update coordinates.  This plug-in restriction
does not account for uncertainty in the estimated source class parameters.
"""

from copy import deepcopy

import numpy as np

from .projected_explanations import ProjectedExplanationSet, _readonly


class AnchoredExplanationSet(ProjectedExplanationSet):
    """Same constructor/projection as ProjectedExplanationSet, smaller theta.

    Optimized ``parameters`` contain [log_scales, shifts, prior_logits], with
    C-1 prior logits and the last class logit fixed at zero as before.
    ``fixed_class_parameters`` packs [class_means, class_log_variances], each
    in row-major class-by-projected-coordinate order.  ``expand_parameters``
    reconstructs the unrestricted family's parameter layout for diagnostics.

    Fit, candidate validation, endpoint restoration and local constrained
    search are inherited.  All such operations see only the reduced vector.
    """

    @property
    def fixed_class_parameters(self):
        return self._fixed_class_parameters

    def _initial_parameters(self):
        # Called by ExplanationSet.__init__ after projected source statistics
        # are available, before any fit. The original initializer supplies the
        # exact source-only means, floored variances, prior and tail bounds.
        initial, lower, upper = super()._initial_parameters()
        self._fixed_size = 2*self.classes*self.dimension
        self._full_parameter_size = len(initial)
        self._fixed_class_parameters = _readonly(initial[:self._fixed_size])
        self._size = len(initial)-self._fixed_size
        return (initial[self._fixed_size:].copy(), lower[self._fixed_size:].copy(),
                upper[self._fixed_size:].copy())

    def expand_parameters(self, parameters):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (self._size,) or not np.isfinite(parameters).all():
            raise ValueError("Expected finite reduced anchored parameters")
        return np.concatenate((self.fixed_class_parameters, parameters))

    def _unpack(self, parameters):
        return super()._unpack(self.expand_parameters(parameters))

    def _observation(self, parameters):
        # The inherited expression obtains fixed means/variances through our
        # _unpack, and builds the full analytic gradient. Keep only free slots.
        value, gradient = super()._observation(parameters)
        return value, gradient[self._fixed_size:]

    def _separation(self, parameters, correction):
        value, gradient = super()._separation(parameters, correction)
        return value, gradient[self._fixed_size:]

    def fit(self):
        reference = super().fit()
        cd = self.classes*self.dimension
        means = self.fixed_class_parameters[:cd].reshape(self.classes, self.dimension)
        log_variances = self.fixed_class_parameters[cd:].reshape(self.classes, self.dimension)
        source_diff = self.source-means[self.labels]
        source_mean_nll = float(.5*np.mean((np.log(2*np.pi)+log_variances[self.labels]
                                          + source_diff**2*np.exp(-log_variances[self.labels])).sum(1)))
        report = deepcopy(self.fit_report or {})
        # This inherited entry describes free source variances, which this
        # family does not have. Actual optimized bounds are already reduced.
        report.pop("variance_upper_multiplier", None)
        report.update(
            family="source_anchored_projected_gaussian_v1",
            reference_loss=self.reference_loss, threshold=self.threshold,
            parameter_layout=["log_scales", "shifts", "prior_logits_except_last"],
            parameter_sizes=[self.dimension, self.dimension, self.classes-1],
            parameter_dimension=self._size,
            lower_bounds=self._lower.tolist(), upper_bounds=self._upper.tolist(),
            fixed_parameter_layout=["class_means", "class_log_variances"],
            fixed_parameter_sizes=[cd, cd],
            fixed_class_parameters=self.fixed_class_parameters.tolist(),
            fixed_class_means=means.tolist(), fixed_class_log_variances=log_variances.tolist(),
            full_parameter_dimension=self._full_parameter_size,
            fixed_parameter_estimator="source_fit_projected_MLE_ddof0_with_variance_floor",
            fixed_parameters_use_target=False, variance_floor=float(self.config.variance_floor),
            source_class_counts=np.bincount(self.labels, minlength=self.classes).tolist(),
            source_mean_conditional_nll=source_mean_nll,
            source_conditional_nll_constant=True,
            observation_denominator=len(self.source)+len(self.target),
            observation_loss_definition="(sum_source_conditional_NLL + sum_target_marginal_NLL) / "
                                        "(n_source + n_target)",
            density_dimension=self.dimension, correction_dimension=self.full_dimension,
            source_anchor_uncertainty_included=False,
            projection=deepcopy(self.projection_report),
        )
        self.fit_report = report
        return reference


def check_anchored_explanations():
    """Generated engineering data only: parity, gradients, replay and search."""
    import json

    from tsba.shared_update.config import SharedUpdateConfig

    rng = np.random.default_rng(2718)
    cfg = SharedUpdateConfig(seed=19, inner_starts=2, inner_maxiter=150, tau=.02)
    cases, worst_gradient_error = [], 0.
    for full_dimension, classes in [(2, 2), (5, 2), (2, 3)]:
        labels = np.repeat(np.arange(classes), 18)
        source = rng.normal(size=(len(labels), full_dimension))*.7
        source[:, 0] += labels-(classes-1)/2
        target = rng.normal(size=(15, full_dimension))
        normal = np.arange(1., full_dimension+1)
        anchored = AnchoredExplanationSet(source, labels, target, cfg, source_head_normal=normal)
        unrestricted = ProjectedExplanationSet(source, labels, target, cfg, source_head_normal=normal)
        cd = classes*anchored.dimension
        np.testing.assert_array_equal(anchored.fixed_class_parameters, unrestricted._initial[:2*cd])
        assert not anchored.fixed_class_parameters.flags.writeable
        assert anchored._size == 2*anchored.dimension+classes-1
        theta = anchored._initial.copy()+rng.normal(size=anchored._size)*.015
        full_theta = anchored.expand_parameters(theta)
        correction = rng.normal(size=(len(target), classes))
        for name, extra in [("_observation", ()), ("_separation", (correction,))]:
            function, other = getattr(anchored, name), getattr(unrestricted, name)
            value, gradient = function(theta, *extra)
            other_value, other_gradient = other(full_theta, *extra)
            np.testing.assert_array_equal(value, other_value)
            np.testing.assert_array_equal(gradient, other_gradient[2*cd:])
            numerical = []
            for axis in np.eye(len(theta)):
                numerical.append((function(theta+1e-6*axis, *extra)[0]
                                  -function(theta-1e-6*axis, *extra)[0])/2e-6)
            np.testing.assert_allclose(gradient, numerical, atol=3e-7, rtol=2e-5)
            worst_gradient_error = max(worst_gradient_error, float(np.max(np.abs(gradient-numerical))))
        np.testing.assert_array_equal(anchored._target_terms(theta)[-1],
                                      unrestricted._target_terms(full_theta)[-1])
        a, b = anchored._candidate(theta), unrestricted._candidate(full_theta)
        np.testing.assert_array_equal(a.statistic, b.statistic)
        assert a.statistic.shape == (classes, full_dimension+1)
        anchors = anchored.fixed_class_parameters.copy()
        reference = anchored.fit()
        candidate = anchored.search(correction, [reference])
        assert candidate is not None and candidate.observation_loss <= anchored.threshold+cfg.feasibility_tol
        assert anchored.last_search["status"] == "local_search_completed"
        np.testing.assert_array_equal(anchored.fixed_class_parameters, anchors)
        assert len(reference.parameters) == anchored._size
        assert anchored.fit() is reference
        assert sum(anchored.fit_report["parameter_sizes"]) == anchored._size
        assert anchored.fit_report["observation_denominator"] == len(source)+len(target)
        assert "variance_upper_multiplier" not in anchored.fit_report
        json.dumps(anchored.fit_report, allow_nan=False)
        # Changing the unlabeled query rows cannot change source anchors.
        replay = AnchoredExplanationSet(source, labels, rng.normal(size=(7, full_dimension)), cfg,
                                       source_head_normal=normal)
        np.testing.assert_array_equal(replay.fixed_class_parameters, anchors)
        np.testing.assert_array_equal(replay.projection_basis, anchored.projection_basis)
        assert replay._target_terms(reference.parameters)[-1].shape == (7, classes)
        cases.append(dict(full_dimension=full_dimension, classes=classes,
                          optimized_parameters=anchored._size, full_parameters=anchored._full_parameter_size,
                          statistic_shape=list(reference.statistic.shape), search_status=anchored.last_search["status"]))
    # The floor is the original absolute projected-coordinate variance floor.
    constant = AnchoredExplanationSet(np.zeros((8, 2)), np.tile([0, 1], 4), np.zeros((3, 2)), cfg,
                                     source_head_normal=np.zeros(2))
    np.testing.assert_array_equal(constant.fixed_class_parameters[4:], np.full(4, np.log(cfg.variance_floor)))
    return dict(passed=True, cases=cases, maximum_gradient_error=worst_gradient_error,
                unrestricted_same_parameter_parity_exact=True, original_joint_denominator=True,
                full_moment_space=True, source_only_anchors_and_replay=True, variance_floor_verified=True)
