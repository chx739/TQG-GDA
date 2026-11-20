"""Independent source-screen estimator for offset logistic models and calibration."""
import numpy as np
from scipy.special import expit, logit


def pair_features(mi, mj, sim, size_abs):
    mi, mj = np.asarray(mi, dtype=np.float64), np.asarray(mj, dtype=np.float64)
    product = mi * mj
    return logit((1. + product) / 2.), np.column_stack(
        (mi + mj, product, np.abs(mi - mj), sim, size_abs))


def _newton(design, offset, y, weights, ridge, prior):
    """Unconstrained strictly convex ridge model with fixed offset and prior center."""
    theta = prior.copy()
    sign = 1. - 2. * y
    eye = ridge * np.eye(len(theta))

    def objective(par):
        return (weights @ np.logaddexp(0., sign * (offset + design @ par))
                + .5 * ridge * np.dot(par - prior, par - prior))

    for _ in range(100):
        eta = offset + design @ theta
        residual = sign * expit(sign * eta)
        gradient = design.T @ (weights * residual) + ridge * (theta - prior)
        norm = float(np.max(np.abs(gradient)))
        if norm < 3e-12:
            return theta, norm
        curvature = weights * expit(eta) * expit(-eta)
        hessian = design.T @ (curvature[:, None] * design) + eye
        direction = np.linalg.solve(hessian, gradient)
        decrement = np.dot(gradient, direction)
        if decrement < 1e-15:
            # At numerical precision, an objective-based test can reject a
            # full Newton step even though it still reduces stationarity.
            theta = theta - direction
            continue
        old = objective(theta)
        scale = 1.
        while objective(theta - scale * direction) > old - 1.e-4 * scale * decrement:
            scale *= .5
            if scale < 1e-12:
                raise ArithmeticError('Newton line search did not decrease the objective')
        theta = theta - scale * direction
    raise ArithmeticError('Newton iteration did not converge')


def calib(a, y):
    a, y = np.asarray(a, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n = len(a)
    ones = np.ones(n, dtype=np.float64)
    z = np.zeros(n)
    w = ones / n
    design = np.column_stack((ones, a))
    theta, error = _newton(design, z, y, w, 1./n, np.array([0., 1.]))
    if theta[1] >= 0.:
        return float(theta[0]), float(theta[1]), error

    # Strict convexity puts the optimum at t=0 if the unconstrained
    # solution has t<0. Refit the intercept on this face.
    b, error_b = _newton(ones[:, None], z, y, w, 1./n, np.zeros(1))
    g_t = np.mean(a * (expit(b[0]) - y)) - 1./n
    return float(b[0]), 0., max(error_b, float(max(0., -g_t)))


def head_fit(x, offset, y, w, n):
    x = np.asarray(x, dtype=np.float64)
    y, offset, w = (np.asarray(a, dtype=np.float64) for a in (y, offset, w))
    if not len(x):
        return lambda features, offsets: expit(offsets), 0.
    w = w / w.sum()
    mu = w @ x
    const = np.ptp(x, axis=0) == 0.
    mu[const] = x[0, const]
    scale = np.sqrt(w @ np.square(x - mu))
    scale[const] = 1.
    design = np.column_stack((np.ones(len(x)), (x - mu) / scale))
    theta, error = _newton(design, offset, y, w, 1./n, np.zeros(design.shape[1]))

    def predict(features, offsets):
        return expit(offsets + theta[0] + ((features - mu) / scale) @ theta[1:])

    return predict, error
