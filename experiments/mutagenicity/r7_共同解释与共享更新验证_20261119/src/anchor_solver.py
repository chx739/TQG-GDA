"""Solve the binary contrast form of R7's single-moment master."""
import numpy as np
from scipy.linalg import norm, solve
from scipy.optimize import brentq
from scipy.special import expit, logit


def reference_moment(theta, stats, reference, u):
    dim = u.shape[1]
    if (theta.ndim!=2 or reference.shape!=(theta.shape[1],) or stats.shape!=(len(theta),2,dim)
            or not all(np.isfinite(a).all() for a in (theta,stats,reference,u))
            or not np.all(u[:,-1]==1.)):
        raise ValueError('Malformed retained R7 moments or source features')
    hits = np.flatnonzero(np.all(theta==reference,axis=1))
    if len(hits)!=1:
        raise ValueError('Expected exactly one original reference interpretation')
    chosen = stats[hits[0]]
    np.testing.assert_allclose(chosen.sum(axis=0),u.mean(axis=0),rtol=0,atol=1e-12)
    if not 0.<=chosen[1,-1]<=1.:
        raise ValueError('The original positive moment is not a probability measure')
    return chosen[1],int(hits[0])


def score_basis(v0):
    dim = v0.shape[1]
    intercept = np.eye(dim)[:,-1]
    direction = v0[1]-v0[0]
    direction[-1] = 0.
    scale = norm(direction)
    return intercept[:,None] if scale==0. else np.column_stack((intercept,direction/scale))


def terms(c, u, s, lam, d):
    step = u@d
    z = c+step
    p = expit(z)
    # softplus(t) = max(t, 0) + log1p(exp(-abs(t))). Two large positive
    # margins have a linear difference of exactly step, even if c+step rounds to c.
    linear = np.where(c >= 0., np.where(z >= 0., step, -c), np.maximum(z, 0.))
    tails = np.log1p(np.exp(-np.abs(z))) - np.log1p(np.exp(-np.abs(c)))
    # The contrast d is represented by the two changes (-d/2, +d/2).
    value = np.mean(linear + tails) - s@d + lam*(d@d)/4
    gradient = u.T@p/len(c) - s + (lam/2)*d
    return float(value), gradient, p*expit(-z)


def solve_binary(c, u, s, lam, *, maxiter, tolerance, q=None):
    """Return the full-coordinate update and projected stationarity receipt."""
    if (c.ndim!=1 or u.ndim!=2 or len(c)!=len(u) or not len(c) or s.shape!=(u.shape[1],)
            or not all(np.isfinite(a).all() for a in (c,u,s)) or not np.isfinite(lam)
            or not np.isfinite(tolerance) or lam<=0 or tolerance<=0 or maxiter<1):
        raise ValueError('Invalid binary master inputs')
    if q is not None:
        if q.ndim!=2 or q.shape[0]!=u.shape[1] or not q.shape[1] or not np.isfinite(q).all():
            raise ValueError('Invalid score-only basis')
        np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),rtol=0,atol=1e-13)
    v,m = (u,s) if q is None else (u@q,s@q)
    x = np.zeros(v.shape[1])
    penalty = lam/2*np.eye(len(x))
    target = min(1e-10,tolerance/10)
    for step in range(maxiter+1):
        value,grad,curv = terms(c,v,m,lam,x)
        if norm(grad,ord=np.inf)<=target:
            break
        if step==maxiter:
            raise ArithmeticError('Binary master exceeded the original iteration budget')
        hessian = (v.T*curv)@v/len(c)+penalty
        delta = solve(hessian,grad,assume_a='pos')
        slope = grad@delta
        for exponent in range(40):
            a = 2.**-exponent
            trial = x-a*delta
            val,g,_ = terms(c,v,m,lam,trial)
            if (val<=value-1e-4*a*slope or
                    (norm(g,ord=np.inf)<target and val<=value+1e-14)):
                x = trial
                break
        else:
            raise ArithmeticError('Binary master failed its backtracking check')
    d = x if q is None else q@x
    objective,grad,_ = terms(c,u,s,lam,d)
    feasible_grad = grad if q is None else q.T@grad
    residual = float(norm(feasible_grad,ord=np.inf))
    if not np.isfinite(objective) or objective>1e-12 or residual>tolerance:
        raise ArithmeticError('Binary master failed certification in original coordinates')
    return d,dict(iterations=step,objective=objective,kkt_inf=residual,
                  full_gradient_inf=float(norm(grad,ord=np.inf)),subspace_dimension=len(x))


def calibrated_teacher(margin, u, s, v0):
    """Match positive mass and raw GNN score moment with a two-parameter teacher."""
    mass = float(s[-1])
    if not 0.<mass<1.:
        raise ArithmeticError('A finite calibrated teacher requires interior positive mass')
    # Center before taking score moments. The shared intercept cancels
    # analytically and can be orders of magnitude larger than the score spread.
    contrast = (v0[1]-v0[0])[:-1]
    centered_moment = (s[:-1] - mass*u[:,:-1].mean(axis=0)) @ contrast
    origin = float(margin[0])
    shift = margin-origin
    mean_shift = float(shift.mean())
    mean = origin+mean_shift
    centered = shift-mean_shift
    sd = float(norm(centered)/np.sqrt(len(centered)))
    coordinates = dict(score_origin=origin, mean_offset=mean_shift, mean=mean, sd=sd)
    offset = float(logit(mass))
    if np.ptp(margin)==0.:
        if abs(centered_moment)>1e-12:
            raise ArithmeticError('Constant B0 score is incompatible with the teacher moment')
        prob = np.full(len(margin),mass)
        v = u.T@prob/len(prob)
        mass_error = float(v[-1]-mass)
        return v,dict(intercept=offset,slope=0.,**coordinates,mass_error=mass_error,
                      standard_score_error=0.,raw_score_error=mean*mass_error-centered_moment)

    z = centered/sd
    target = centered_moment/sd
    wanted = mass*len(z)
    order = np.sort(z)
    k = int(wanted)
    fraction = wanted-k

    def attainable(sorted_z):
        return (sorted_z[:k].sum()+fraction*sorted_z[min(k,len(z)-1)])/len(z)

    if not attainable(order)<target<attainable(order[::-1]):
        raise ArithmeticError('Calibrated score moment does not admit a finite interior solution')

    def at_slope(b):
        shift = b*z
        a = brentq(lambda a: float(expit(a+shift).mean()-mass),
                   offset-shift.max()-1.,offset-shift.min()+1.,xtol=1e-14)
        return float(a),expit(a+shift)

    def residual(b):
        return float(z@at_slope(b)[1]/len(z)-target)

    bound = 1.
    while residual(-bound)>0. or residual(bound)<0.:
        bound *= 2.
        if bound>2.**24:
            raise ArithmeticError('Could not bracket a numerically finite calibrated score slope')
    b = float(brentq(residual,-bound,bound,xtol=1e-13))
    a,prob = at_slope(b)
    v = u.T@prob/len(prob)
    mass_error = float(v[-1]-mass)
    score_error = float(z@prob/len(prob)-target)
    errors = dict(mass_error=mass_error,standard_score_error=score_error,
                  raw_score_error=mean*mass_error+sd*score_error)
    if max(abs(errors[k]) for k in ('mass_error','standard_score_error'))>1e-11:
        raise ArithmeticError('Calibrated teacher failed its two-moment check')
    return v,dict(intercept=a,slope=b,**coordinates,**errors)
