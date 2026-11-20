"""Binary source calibration and maximum likelihood label shift on frozen scores."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata


def binary_logits(margins):
    return np.column_stack((np.zeros_like(margins), margins))


def margin(logits):
    return logits[:, 1] - logits[:, 0]


def fit_calibrator(margins, labels, options):
    if not np.isfinite(margins).all() or margins.shape != labels.shape or set(labels) != {0, 1}:
        raise ValueError('Calibration requires finite binary validation scores with both classes')
    t_min, t_max = options['temperature_bounds']
    lower, upper = 1/t_max, 1/t_min

    def objective(parameters):
        slope, intercept = parameters
        scores = slope * margins + intercept
        signed = (1 - 2 * labels) * scores
        loss = np.logaddexp(0., signed).mean()
        residual = (1 - 2 * labels) * expit(signed)
        return loss, np.array([(residual * margins).mean(), residual.mean()])

    initial = [options['initial_inverse_temperature'], options['initial_intercept']]
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      bounds=[(lower, upper), (None, None)],
                      options=dict(ftol=options['optimizer_ftol'], gtol=options['optimizer_gtol'],
                                   maxiter=options['max_iterations']))
    slope, intercept = result.x
    value, gradient = objective(result.x)
    projected = abs(slope - np.clip(slope - gradient[0], lower, upper))
    error = max(projected, abs(gradient[1]))
    if (not result.success or not np.isfinite([slope, intercept, value, error]).all()
            or error > options['projected_gradient_tolerance'] or value > objective(initial)[0] + 1e-12):
        raise ValueError(f'Source calibration did not converge: {result.message}; residual={error}')
    return dict(temperature=float(1/slope), inverse_temperature=float(slope), intercept=float(intercept),
                source_validation_nll_before=float(objective(initial)[0]), source_validation_nll=float(value),
                projected_gradient_error=float(error), iterations=int(result.nit))


def calibrate(margins, fitted):
    return fitted['inverse_temperature'] * margins + fitted['intercept']


def fit_prior(margins, source_prior, options):
    floor = options['prior_floor']
    if (not np.isfinite(margins).all() or not len(margins) or not floor <= source_prior <= 1-floor
            or options['initial_prior'] != 'source_prior'):
        raise ValueError('MLLS requires finite unlabeled A scores and usable source priors')
    source_log_odds = logit(source_prior)
    base_log0 = -np.logaddexp(0., margins)
    base_log1 = -np.logaddexp(0., -margins)

    def likelihood(prior):
        log0 = base_log0 + np.log1p(-prior) - np.log1p(-source_prior)
        log1 = base_log1 + np.log(prior) - np.log(source_prior)
        return np.logaddexp(log0, log1).mean()

    prior = source_prior
    initial = likelihood(prior)
    for iteration in range(1, options['max_iterations'] + 1):
        update = np.clip(expit(margins + logit(prior) - source_log_odds).mean(), floor, 1-floor)
        if abs(update - prior) <= options['tolerance']:
            final = likelihood(update)
            if not np.isfinite(final) or final + 1e-12 < initial:
                raise ValueError('MLLS likelihood decreased')
            return dict(source_prior=float(source_prior), target_prior=float(update),
                        log_odds_shift=float(logit(update) - source_log_odds), em_iterations=iteration,
                        last_prior_step=float(abs(update-prior)), initial_log_likelihood=float(initial),
                        final_log_likelihood=float(final), prior_at_bound=bool(update in (floor, 1-floor)))
        prior = update
    raise ValueError('MLLS exceeded the registered iteration limit')


def adjusted_scores(source_margin, fitted, prior):
    calibrated = calibrate(source_margin, fitted)
    shifted = calibrated + prior['log_odds_shift']
    source_balanced = calibrated - logit(prior['source_prior'])
    target_balanced = shifted - logit(prior['target_prior'])
    return calibrated, shifted, source_balanced, target_balanced


def verify_rank_preservation(reference, transformed, atol):
    expected, actual = rankdata(reference), rankdata(transformed)
    if np.array_equal(expected, actual):
        return 0
    # A positive affine map cannot reverse the mathematical rank. A finite
    # precision tie can only collapse gaps of the order of the replay tolerance.
    order = np.argsort(reference, kind='stable')
    same_input = np.diff(reference[order]) == 0
    gaps = np.diff(transformed[order])
    if (gaps < 0).any() or (gaps[same_input] != 0).any():
        raise ValueError('Calibration or label-shift correction reordered scores')
    ties = (gaps == 0) & ~same_input
    if (np.diff(reference[order])[ties] > atol).any():
        raise ValueError('Score ranking collapsed beyond the replay tolerance')
    return int(ties.sum())
