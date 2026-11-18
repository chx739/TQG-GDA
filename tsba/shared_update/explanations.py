"""Observable-data explanation fitting and local constrained separation.

This reference family uses diagonal class Gaussians and one positive diagonal
affine target-to-source map.  It is a modelling restriction, not a coverage
guarantee.  In particular positive scales exclude orientation-reversing maps.
The source class means may move by ``mean_radius`` in each input coordinate;
class variances range from ``variance_floor`` to 100 times their initial
floored source estimates.  These bounds are fixed before fitting.

Only source labels enter this module.  A feasible local separation result is
a lower bound on the worst risk, never a global separation certificate.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


@dataclass(frozen=True)
class Explanation:
    parameters: np.ndarray
    observation_loss: float
    statistic: np.ndarray


class ExplanationSet:
    """A fixed observation-compatible set of joint target explanations.

    Parameters are packed as class means, class log variances, shared log
    scales, shared shifts, then C-1 prior logits (the last logit is zero).
    Priors equal floor + (1-C*floor)*softmax(logits).  Prior logits are bounded
    to [-12, 12], a fixed capacity choice in addition to the prior floor.

    ``inner_starts`` bounds optimizer launches, not the number of previously
    known feasible explanations considered when returning the best candidate.
    ``fit_report`` and ``last_search`` expose solver failures independently of
    whether a feasible candidate was retained.
    """

    _FIELDS = ("tau", "variance_floor", "mean_radius", "scale_min", "scale_max",
               "shift_bound", "prior_floor", "inner_starts", "inner_maxiter",
               "feasibility_tol", "seed")

    def __init__(self, source_x, source_y, target_x, config):
        self.config = SimpleNamespace(**{name: getattr(config, name) for name in self._FIELDS})
        self.source = np.array(source_x, dtype=np.float64, copy=True)
        self.target = np.array(target_x, dtype=np.float64, copy=True)
        labels = np.asarray(source_y)
        if (self.source.ndim != 2 or self.target.ndim != 2 or not len(self.source)
                or not len(self.target) or self.source.shape[1] == 0
                or self.source.shape[1] != self.target.shape[1]):
            raise ValueError("Source and target must be nonempty matrices of equal feature dimension")
        if not np.isfinite(self.source).all() or not np.isfinite(self.target).all():
            raise ValueError("Explanation features must be finite")
        if (labels.ndim != 1 or len(labels) != len(self.source)
                or not np.issubdtype(labels.dtype, np.integer)):
            raise ValueError("Source labels must be one integer per source sample")
        classes = np.unique(labels)
        if len(classes) < 2 or not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("Source labels must contain every class in contiguous range 0..C-1")
        self.labels = np.array(labels, dtype=np.int64, copy=True)
        self.classes, self.dimension = len(classes), self.source.shape[1]
        self._validate_config()
        self.features = np.column_stack((self.target, np.ones(len(self.target))))
        self._size = 2 * self.classes * self.dimension + 2 * self.dimension + self.classes - 1
        self._rng = np.random.default_rng(int(self.config.seed))
        self._reference = None
        self.reference_loss = None
        self.threshold = None
        self.fit_report = None
        self.last_search = None
        self._fit_candidates = []
        self._initial, self._lower, self._upper = self._initial_parameters()

    def _validate_config(self):
        c = self.config
        for name in ("tau", "variance_floor", "mean_radius", "scale_min", "scale_max",
                     "shift_bound", "prior_floor", "feasibility_tol"):
            if not np.isfinite(getattr(c, name)):
                raise ValueError(f"{name} must be finite")
        if (c.tau < 0 or c.variance_floor <= 0 or c.mean_radius < 0
                or c.scale_min <= 0 or c.scale_max < c.scale_min
                or c.shift_bound < 0 or not 0 <= c.prior_floor < 1 / self.classes
                or c.feasibility_tol < 0):
            raise ValueError("Invalid explanation set bounds or tolerances")
        for name in ("inner_starts", "inner_maxiter"):
            value = getattr(c, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(c.seed, (bool, np.bool_)) or int(c.seed) != c.seed or c.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def _initial_parameters(self):
        c, d = self.config, self.dimension
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            means = np.stack([self.source[self.labels == k].mean(0) for k in range(self.classes)])
            variances = np.stack([self.source[self.labels == k].var(0) for k in range(self.classes)])
            variances = np.maximum(variances, c.variance_floor)
            frequency = np.bincount(self.labels, minlength=self.classes) / len(self.labels)
            latent = np.maximum(frequency - c.prior_floor, 1e-8)
            latent /= latent.sum()
            prior_logits = np.clip(np.log(latent[:-1] / latent[-1]), -12., 12.)
            initial = np.concatenate((means.ravel(), np.log(variances).ravel(),
                                      np.zeros(d), np.zeros(d), prior_logits))
            lower = np.concatenate(((means - c.mean_radius).ravel(),
                                    np.full(self.classes * d, np.log(c.variance_floor)),
                                    np.full(d, np.log(c.scale_min)),
                                    np.full(d, -c.shift_bound), np.full(self.classes - 1, -12.)))
            upper = np.concatenate(((means + c.mean_radius).ravel(),
                                    (np.log(variances) + np.log(100.)).ravel(),
                                    np.full(d, np.log(c.scale_max)),
                                    np.full(d, c.shift_bound), np.full(self.classes - 1, 12.)))
        if not np.isfinite(np.concatenate((initial, lower, upper))).all():
            raise ValueError("Source statistics do not define finite explanation bounds")
        return np.clip(initial, lower, upper), lower, upper

    def _unpack(self, parameters):
        c, d = self.classes, self.dimension
        cd = c * d
        mean = parameters[:cd].reshape(c, d)
        log_variance = parameters[cd:2 * cd].reshape(c, d)
        log_scale = parameters[2 * cd:2 * cd + d]
        shift = parameters[2 * cd + d:2 * cd + 2 * d]
        latent_prior = softmax(np.append(parameters[2 * cd + 2 * d:], 0.))
        prior = self.config.prior_floor + (1 - c * self.config.prior_floor) * latent_prior
        return mean, log_variance, log_scale, shift, latent_prior, prior

    def _target_terms(self, parameters):
        mean, log_variance, log_scale, shift, latent_prior, prior = self._unpack(parameters)
        scale_x = self.target * np.exp(log_scale)
        difference = scale_x[:, None, :] + shift - mean
        inverse_variance = np.exp(-log_variance)
        # Factor differences to a reference class in each coordinate before
        # summing. Subtracting two large squared distances can lose a class
        # gap even when the softmax itself is stable.
        coordinate_energy = -0.5 * (log_variance + difference ** 2 * inverse_variance)
        reference = coordinate_energy.argmax(axis=1, keepdims=True)
        coordinate_offset = np.take_along_axis(coordinate_energy, reference, axis=1)
        reference_diff = np.take_along_axis(difference, reference, axis=1)
        reference_mean = np.take_along_axis(mean[None, :, :], reference, axis=1)
        reference_log_var = np.take_along_axis(log_variance[None, :, :], reference, axis=1)
        log_var_gap = log_variance - reference_log_var
        inverse_var_gap = (np.sign(log_var_gap) * np.maximum(inverse_variance, np.exp(-reference_log_var))
                           * np.expm1(-np.abs(log_var_gap)))
        squared_gap = ((reference_mean - mean) * (difference + reference_diff) * inverse_variance
                       + reference_diff ** 2 * inverse_var_gap)
        relative_energy = -0.5 * (log_var_gap + squared_gap).sum(axis=2)
        class_offset = relative_energy.max(axis=1, keepdims=True)
        log_joint = relative_energy - class_offset + np.log(prior)
        log_norm = logsumexp(log_joint, axis=1)
        posterior = np.exp(log_joint - log_norm[:, None])
        normalizer = (coordinate_offset[:, 0, :].sum(axis=1) - 0.5 * self.dimension * np.log(2 * np.pi)
                      + class_offset[:, 0] + log_norm)
        return (mean, log_variance, log_scale, latent_prior, prior, scale_x,
                difference, inverse_variance, normalizer, posterior)

    def _prior_gradient(self, prior_derivative, latent_prior):
        gradient = ((1 - self.classes * self.config.prior_floor) * latent_prior
                    * (prior_derivative - latent_prior @ prior_derivative))
        return gradient[:-1]

    def _observation(self, parameters):
        """Joint mean NLL and its exact analytic parameter gradient."""
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            (mean, log_variance, log_scale, latent, prior, scale_x,
             diff, invvar, normalizer, posterior) = self._target_terms(parameters)
            source_diff = self.source - mean[self.labels]
            source_squared = source_diff ** 2 * invvar[self.labels]
            source_nll = 0.5 * (np.log(2 * np.pi) + log_variance[self.labels] + source_squared).sum()
            # The shared map's Jacobian cancels in posterior, not likelihood.
            target_nll = -normalizer.sum() - len(self.target) * log_scale.sum()
            total = len(self.source) + len(self.target)
            loss = (source_nll + target_nll) / total
            gmean = -(posterior[:, :, None] * diff * invvar).sum(0)
            gvariance = 0.5 * (posterior[:, :, None] * (1 - diff ** 2 * invvar)).sum(0)
            np.add.at(gmean, self.labels, -source_diff * invvar[self.labels])
            np.add.at(gvariance, self.labels, 0.5 * (1 - source_squared))
            gx = (posterior[:, :, None] * diff * invvar).sum(1)
            gscale = (gx * scale_x).sum(0) - len(self.target)
            gshift = gx.sum(0)
            gprior = self._prior_gradient(-posterior.sum(0) / prior, latent)
            gradient = np.concatenate((gmean.ravel(), gvariance.ravel(), gscale, gshift, gprior)) / total
        if not np.isfinite(loss) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite explanation likelihood or gradient")
        return float(loss), gradient

    def _separation(self, parameters, correction):
        """Mean q dot correction, with an analytic posterior derivative."""
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            (_, _, _, latent, prior, scale_x, diff, invvar,
             _, posterior) = self._target_terms(parameters)
            expectation = (posterior * correction).sum(1)
            weight = posterior * (correction - expectation[:, None]) / len(self.target)
            gmean = (weight[:, :, None] * diff * invvar).sum(0)
            gvariance = 0.5 * (weight[:, :, None] * (diff ** 2 * invvar - 1)).sum(0)
            gx = -(weight[:, :, None] * diff * invvar).sum(1)
            gscale = (gx * scale_x).sum(0)
            gshift = gx.sum(0)
            gprior = self._prior_gradient(weight.sum(0) / prior, latent)
            gradient = np.concatenate((gmean.ravel(), gvariance.ravel(), gscale, gshift, gprior))
            value = expectation.mean()
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("Non-finite explanation separation value or gradient")
        return float(value), gradient

    def _candidate(self, parameters, require_compatible=False):
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (self._size,) or not np.isfinite(parameters).all():
            return None
        tol = self.config.feasibility_tol
        if np.any(parameters < self._lower - tol) or np.any(parameters > self._upper + tol):
            return None
        parameters = np.clip(parameters, self._lower, self._upper)
        try:
            loss, _ = self._observation(parameters)
            if require_compatible and loss > self.threshold + tol:
                return None
            posterior = self._target_terms(parameters)[-1]
            statistic = posterior.T @ self.features / len(self.target)
        except (FloatingPointError, OverflowError):
            return None
        if not np.isfinite(statistic).all():
            return None
        parameters, statistic = parameters.copy(), statistic.copy()
        parameters.setflags(write=False)
        statistic.setflags(write=False)
        return Explanation(parameters, loss, statistic)

    def _perturb(self, parameters, factor=0.12):
        return np.clip(parameters + factor * (self._upper - self._lower)
                       * self._rng.normal(size=self._size), self._lower, self._upper)

    @staticmethod
    def _unique(candidates):
        result, seen = [], set()
        for candidate in candidates:
            if candidate is not None:
                key = candidate.parameters.tobytes()
                if key not in seen:
                    seen.add(key)
                    result.append(candidate)
        return result

    def fit(self):
        """Fit once, then freeze the reference likelihood and set threshold."""
        if self._reference is not None:
            return self._reference
        candidates, reports = [], []
        bounds = list(zip(self._lower, self._upper))
        for index in range(int(self.config.inner_starts)):
            initial = self._initial if index == 0 else self._perturb(self._initial)
            start = self._candidate(initial)
            if start is not None:
                candidates.append(start)
            try:
                result = minimize(self._observation, initial, jac=True, method="L-BFGS-B",
                                  bounds=bounds, options={"maxiter": int(self.config.inner_maxiter),
                                                          "ftol": 1e-12, "gtol": 1e-7})
                candidate = self._candidate(result.x)
                if candidate is not None:
                    candidates.append(candidate)
                reports.append({"success": bool(result.success), "status": int(result.status),
                                "message": str(result.message), "iterations": int(result.nit),
                                "finite_candidate": candidate is not None})
            except (FloatingPointError, OverflowError) as error:
                reports.append({"success": False, "message": str(error), "finite_candidate": False})
        if not candidates:
            raise RuntimeError("No finite in-bounds explanation found during reference fitting")
        self._fit_candidates = self._unique(candidates)
        self._reference = min(self._fit_candidates, key=lambda candidate: candidate.observation_loss)
        self.reference_loss = self._reference.observation_loss
        self.threshold = self.reference_loss + float(self.config.tau)
        if not np.isfinite(self.threshold):
            raise ValueError("Non-finite observation threshold")
        self.fit_report = {"optimizer_runs": reports, "reference_loss": self.reference_loss,
                           "threshold": self.threshold,
                           "successful_runs": sum(row["success"] for row in reports),
                           "parameter_layout": ["class_means", "class_log_variances", "log_scales",
                                                "shifts", "prior_logits_except_last"],
                           "parameter_sizes": [self.classes * self.dimension,
                                               self.classes * self.dimension, self.dimension,
                                               self.dimension, self.classes - 1],
                           "lower_bounds": self._lower.tolist(), "upper_bounds": self._upper.tolist(),
                           "prior_floor": float(self.config.prior_floor),
                           "variance_upper_multiplier": 100.0,
                           "global_optimality_certified": False}
        return self._reference

    def search(self, correction_logits, starts=()):
        """Return the best feasible explanation found, including known starts.

        Failure to find a better explanation never certifies absence of a
        violated constraint.  Even when every SLSQP run fails, a feasible known
        explanation is returned, and ``last_search`` reports the failures.
        """
        if self._reference is None:
            raise RuntimeError("Call fit() before searching the fixed explanation set")
        correction = np.asarray(correction_logits, dtype=np.float64)
        if correction.shape != (len(self.target), self.classes) or not np.isfinite(correction).all():
            raise ValueError("correction_logits must be a finite [n_target, classes] matrix")
        # Class-common shifts are irrelevant. Scaling prevents tiny outer
        # updates from triggering an inner ftol stop solely by their magnitude.
        centered = correction - correction.mean(1, keepdims=True)
        magnitude = float(np.max(np.abs(centered)))
        objective_correction = centered / magnitude if magnitude else centered
        supplied = list(starts)
        known = self._unique([self._candidate(item.parameters, True)
                              for item in [self._reference, *self._fit_candidates, *supplied]])
        if not known:
            self.last_search = {"status": "no_feasible_start", "global_optimality_certified": False}
            return None
        objective = lambda candidate: self._separation(candidate.parameters, objective_correction)[0]
        known.sort(key=objective)
        previous_best = objective(known[0])
        candidates = list(known)
        optimizer_starts = known[:int(self.config.inner_starts)]
        # A short contraction toward the fixed feasible reference provides
        # additional feasible local starts without changing the set itself.
        for _ in range(4 * int(self.config.inner_starts)):
            if len(optimizer_starts) >= self.config.inner_starts:
                break
            proposed = self._perturb(self._reference.parameters)
            for contraction in range(20):
                parameters = self._reference.parameters + 0.5 ** contraction * (proposed - self._reference.parameters)
                candidate = self._candidate(parameters, True)
                if candidate is not None:
                    optimizer_starts = self._unique([*optimizer_starts, candidate])
                    break
        reports = []
        if magnitude:
            bounds = list(zip(self._lower, self._upper))
            constraint = {"type": "ineq",
                          "fun": lambda p: self.threshold - self._observation(p)[0],
                          "jac": lambda p: -self._observation(p)[1]}
            for start in optimizer_starts:
                candidates.append(start)
                try:
                    result = minimize(lambda p: self._separation(p, objective_correction),
                                      start.parameters.copy(), jac=True, method="SLSQP", bounds=bounds,
                                      constraints=[constraint],
                                      options={"maxiter": int(self.config.inner_maxiter), "ftol": 1e-10})
                    candidate = self._candidate(result.x, True)
                    if candidate is not None:
                        candidates.append(candidate)
                    reports.append({"success": bool(result.success), "status": int(result.status),
                                    "message": str(result.message), "iterations": int(result.nit),
                                    "feasible_candidate": candidate is not None})
                except (FloatingPointError, OverflowError) as error:
                    reports.append({"success": False, "message": str(error), "feasible_candidate": False})
        best = min(candidates, key=objective)
        successes = sum(row["success"] and row["feasible_candidate"] for row in reports)
        minimum_loss = min(candidate.observation_loss for candidate in candidates)
        self.last_search = {
            "status": "flat_objective" if not magnitude else (
                "local_search_completed" if successes else "local_search_failed"),
            "known_feasible_candidates": len(known), "supplied_starts": len(supplied),
            "optimizer_runs": reports, "successful_feasible_runs": successes,
            "objective": self._separation(best.parameters, correction)[0],
            "improvement_over_known": magnitude * (previous_best - objective(best)),
            "new_candidate_found": objective(best) < previous_best - 1e-10,
            "observation_violation": max(0., best.observation_loss - self.threshold),
            "min_observation_loss": minimum_loss,
            "better_reference_found": minimum_loss < self.reference_loss - self.config.feasibility_tol,
            "global_optimality_certified": False,
        }
        return best
