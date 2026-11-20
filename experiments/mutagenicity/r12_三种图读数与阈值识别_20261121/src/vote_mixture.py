"""Exact hard actions from a smoothed mixture of joint vote templates."""
from fractions import Fraction

import numpy as np
from scipy.optimize import brentq


def encode(b0, wl, size):
    if (b0.ndim != 1 or b0.shape != wl.shape or b0.shape != size.shape
            or any(not np.isin(v, [0, 1]).all() for v in (b0, wl, size))):
        raise ValueError('Expected three binary votes for each graph')
    return 4 * size.astype(np.int64) + 2 * b0.astype(np.int64) + wl.astype(np.int64)


def source_counts(bins, labels):
    if bins.ndim != 1 or labels.shape != bins.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected one binary V label per joint vote')
    return np.bincount(8 * labels.astype(np.int64) + bins, minlength=16).reshape(2, 2, 4)


def template(counts):
    n0, n1 = (int(r.sum()) for r in counts)
    width = counts.shape[-1]
    g0, g1 = [tuple(Fraction(2 * int(v) + 1, 2 * n + width) for v in row)
              for n, row in zip((n0, n1), counts, strict=True)]
    p0 = Fraction(2 * n1 + width, 2 * (n0 + n1 + width))
    return g0, g1, p0


def derivative(a, g0, delta, p0, p):
    return sum(c * d / (g + p*d) for c,g,d in zip(a,g0,delta,strict=True)) + p0/p - (1-p0)/(1-p)


def fit_row(source, observed):
    """Fit one prior. Bracketing solves for diagnostics; exact signs choose all actions."""
    if (source.ndim != 2 or source.shape[0] != 2 or source.shape[1] not in (4, 8)
            or observed.shape != (source.shape[1],)
            or any(not np.issubdtype(x.dtype,np.integer) or np.any(x < 0) for x in (source,observed))):
        raise ValueError('Expected nonnegative V class-template counts and A vote counts')
    g0, g1, p0 = template(source)
    delta = tuple(b-a for a,b in zip(g0,g1,strict=True))
    count = tuple(map(int, observed))
    source_n = source.sum(axis=1).tolist()
    n = sum(count)
    static = (source[1] > source[0]).astype(np.int64)
    action = static
    critical = None
    if n == 0:
        status = 'empty_A'
    elif min(source_n) == 0:
        status = 'missing_V_class'
    elif g0 == g1:
        status = 'equal_templates'
    else:
        status = 'fit'
        critical = [derivative(count,g0,delta,p0,a/(a+b)) for a,b in zip(g0,g1,strict=True)]
        action = np.asarray([value > 0 for value in critical],dtype=np.int64)
    p = float(p0)
    if status == 'fit':
        a, g, d, prior = np.asarray(count,dtype=float), np.asarray(g0,dtype=float), np.asarray(delta,dtype=float), float(p0)

        def numerical_derivative(q):
            if q == 0.:
                return np.inf
            if q == 1.:
                return -np.inf
            return float(np.dot(a,d/(g+q*d)) + prior/q - (1-prior)/(1-q))

        p = brentq(numerical_derivative, 0., 1., xtol=1e-15)
    mixture = (1-p)*np.asarray(g0,dtype=float) + p*np.asarray(g1,dtype=float)
    return dict(status=status, source_classes=source_n, a_n=n, a_counts=list(count),
                p0_exact=str(p0), estimated_p=p, template_tv=float(sum(map(abs,delta))/2),
                residual_tv=float(np.abs(np.asarray(count)/n-mixture).sum()/2) if n else None,
                static_actions=static.tolist(), actions=action.tolist(),
                static_cells_changed=int((static!=action).sum()),
                derivative_signs=None if critical is None else [(v>0)-(v<0) for v in critical])


def rules(source, observations):
    if source.shape != (2,2,4) or observations.shape != (2,4):
        raise ValueError('Expected the two registered Size strata and four joint outcomes')
    conditional = [fit_row(source[:,z],observations[z]) for z in range(2)]
    global_prior = fit_row(source.reshape(2,8),observations.ravel())
    return dict(conditional=conditional, global_prior=global_prior)


def predict(local, others, bins):
    if len(others) != 2:
        raise ValueError('Expected the two same-source foreign pools')
    static = np.asarray([v for r in local['conditional'] for v in r['static_actions']],dtype=np.int64)

    def table(fitted):
        return np.asarray([v for row in fitted['conditional'] for v in row['actions']],dtype=np.int64)

    own = table(local)[bins]
    b0, wl = (bins//2)%2, bins%2
    return dict(Static=static[bins], Global=np.asarray(local['global_prior']['actions'])[bins],
                Own=own, Other1=table(others[0])[bins], Other2=table(others[1])[bins],
                OwnLock=np.where(b0 == wl,b0,own))
