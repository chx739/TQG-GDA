"""Read-only risk attribution for R7's existing finite explanation bank."""
from fractions import Fraction
from math import fsum

import numpy as np


def risk_vector(margin_b0, margin_w):
    """Per-graph class-0 loss change and positive log-odds change."""
    return np.logaddexp(0., margin_w) - np.logaddexp(0., margin_b0), margin_w - margin_b0


def soft_risks(normalizer, delta, q):
    return (normalizer[None, :] - q*delta).mean(axis=1)


def fixed_mass_bounds(normalizer, delta, mass):
    """Sharp finite-sample bounds conditional on sum(q); ordering is by delta."""
    n = len(delta)
    if n == 0 or not 0 <= mass <= n or normalizer.shape != delta.shape:
        raise ValueError('Soft class mass or query shape is invalid')
    k = int(mass)
    fraction = float(mass-k)
    z = np.sort(delta)

    def dot(order):
        if k == n:
            return fsum(order)
        return fsum([*order[:k], fraction*order[k]])

    shift = fsum(normalizer)
    return (shift-dot(z[::-1]))/n, (shift-dot(z))/n


def most_adverse(risks, tolerance):
    if risks.ndim != 1 or not len(risks) or not np.isfinite(risks).all() or tolerance < 0:
        raise ValueError('Invalid retained candidate risks or tie tolerance')
    ids = np.flatnonzero(risks.max()-risks <= tolerance).tolist()
    return ids[0], ids


def sum_mass(q):
    if q.ndim != 1 or not np.isfinite(q).all() or np.any((q<0.) | (q>1.)):
        raise ValueError('Soft class mass requires finite binary probabilities')
    return sum((Fraction.from_float(float(v)) for v in q), Fraction())


def decomposition(normalizer, delta, q, y):
    actual = float(np.mean(normalizer-y*delta))
    expected = float(soft_risks(normalizer, delta, q[None, :])[0])
    error = q-y
    mass = float(np.mean(error)*np.mean(delta))
    assignment = float(np.mean((error-error.mean())*(delta-delta.mean())))
    return dict(actual=actual, expected=expected, error_gap=actual-expected,
                candidate_prior=float(q.mean()), true_prior=float(y.mean()), mean_delta=float(delta.mean()),
                mass_term=mass, assignment_term=assignment,
                mass_only=expected+mass, assignment_only=expected+assignment,
                identity_error=abs((actual-expected)-(mass+assignment)))
