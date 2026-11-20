"""Cross-check the R7 label-shift EM using the same bounded likelihood."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import prior_control


def bounded_optimum(scores, source_prior, options, *, constant_validation_score=None):
    floor = options['prior_floor']
    if (not len(scores) or not np.isfinite(scores).all() or not floor <= source_prior <= 1 - floor):
        raise ValueError('Invalid bounded label-shift likelihood inputs')
    log_ratio = scores - logit(source_prior)
    # A constant V and the exact same constant A give a flat objective.
    # logit(mean(expit(V))) need not round back to the constant itself.
    if constant_validation_score is not None and np.all(scores == constant_validation_score):
        log_ratio = np.zeros_like(scores)
    r0, r1 = expit(-log_ratio), expit(log_ratio)
    numerator = np.tanh(log_ratio / 2)

    def derivative(q):
        return float(np.mean(numerator / ((1 - q) * r0 + q * r1)))

    lower, upper = derivative(floor), derivative(1 - floor)
    flat = bool(np.all(log_ratio == 0))
    if flat:
        q = source_prior
    elif lower <= 0:
        q = floor
    elif upper >= 0:
        q = 1 - floor
    else:
        q = brentq(derivative, floor, 1 - floor, xtol=4 * np.finfo(float).eps)
    base0 = -np.logaddexp(0., scores) - np.log1p(-source_prior)
    base1 = -np.logaddexp(0., -scores) - np.log(source_prior)
    likelihood = np.logaddexp(base0 + np.log1p(-q), base1 + np.log(q)).mean()
    return dict(target_prior=float(q), prior_at_bound=bool(q in (floor, 1 - floor)), flat=flat,
                derivative=derivative(q), lower_derivative=lower, upper_derivative=upper,
                final_log_likelihood=float(likelihood))


def estimate(scores, source_prior, options, *, constant_validation_score=None):
    def root(x, p, cfg):
        return bounded_optimum(x, p, cfg, constant_validation_score=constant_validation_score)

    statuses = {}
    for name, fit in (('em', prior_control.fit_prior), ('root', root)):
        try:
            statuses[name] = fit(scores, source_prior, options)
        except (ValueError, RuntimeError) as error:
            statuses[name] = dict(error=str(error))
    return statuses


def decisions(scores, q, source_prior):
    return scores + (logit(q) - logit(source_prior)) > 0


def majority(positives, n):
    return 2 * positives > n
