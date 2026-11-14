"""Restricted fixed-slope threshold oracle; not the nonlinear TSBA head.

Raw log odds = 8*(x1-b). Parameter beta=(b-mean1)/source_scale.
The normalized-margin change is -(b_new-b)/scale, so source/target
tradeoffs retain the archived coordinate convention, lambda and damping.
"""
import math
import numpy as np
from scipy.special import expit, roots_hermitenorm
from scipy.optimize import brentq
from experiments.common.synthetic.analytic import mixture_risk


def source_quadrature(order=1024):
    nodes, weights = roots_hermitenorm(order)
    x = np.concatenate([-.5+.5*nodes, 1.5+.5*nodes])
    return x, np.tile(weights/(2*math.sqrt(2*math.pi)), 2)


def source_terms(b, x, weights):
    q = expit(8*(x-.5)); z = 8*(x-b); p = expit(z)
    ce = np.sum(weights*(np.logaddexp(0., z)-q*z))
    gradient = np.sum(weights*8*(q-p))
    curvature = np.sum(weights*64*p*(1-p))
    return float(ce), float(gradient), float(curvature)


def population_score(b, raw_sigma):
    variance = .25+raw_sigma**2
    return (math.tanh(b/variance)-b)/variance


def request(b, raw_sigma, cfg):
    return -float(np.clip(cfg.tau*raw_sigma**2*population_score(b, raw_sigma),
                          -cfg.request_multiplier*raw_sigma, cfg.request_multiplier*raw_sigma))


def balance(b, x, weights, scale, raw_sigma, cfg):
    return source_terms(b, x, weights)[1]-cfg.lambda_boundary*request(b, raw_sigma, cfg)/scale**2


def simulate(x, weights, scale, raw_sigma, cfg, density_only=False, limit=10000):
    b = .5; rows = []
    for step in range(limit+1):
        ce, gradient, curvature = source_terms(b, x, weights)
        nu = request(b, raw_sigma, cfg)
        rows.append(dict(step=step, b=b, target_ba=1-mixture_risk(b), source_ce=ce,
                         source_kl=ce-source_terms(.5,x,weights)[0], request_raw=nu,
                         balance_gradient=gradient-cfg.lambda_boundary*nu/scale**2))
        if step == limit:
            reason = 'budget'; break
        if density_only:
            delta = nu
            if abs(delta) < 1e-12:
                reason = 'stationary_tolerance'; break
            b += delta; rows[-1].update(delta_raw=delta, alpha=1., trials=1)
            continue
        delta = (-scale**2*gradient+cfg.lambda_boundary*nu)/(scale**2*curvature+cfg.lambda_boundary+cfg.damping)
        if abs(delta) < 1e-12:
            reason = 'stationary_tolerance'; break
        before = ce+cfg.lambda_boundary*.5*(nu/scale)**2
        directional = (gradient-cfg.lambda_boundary*nu/scale**2)*delta
        accepted = False
        for k in range(cfg.backtracks):
            alpha = 2.**(-k); candidate = b+alpha*delta
            after = source_terms(candidate,x,weights)[0]+cfg.lambda_boundary*.5*((candidate-b-nu)/scale)**2
            if abs(candidate-b)<=cfg.execution_multiplier*raw_sigma and after<=before+cfg.armijo*alpha*directional:
                rows[-1].update(delta_raw=candidate-b, alpha=alpha, trials=k+1,
                                objective_before=before, objective_after=after)
                b = candidate; accepted = True; break
        if not accepted:
            reason = 'backtrack_rejection'; break
    return dict(history=rows, stop_reason=reason, final=rows[-1],
                at30=rows[min(30,len(rows)-1)], density_only=density_only)


def analyze(scale, raw_sigma, cfg, finite_source):
    pop_x, pop_w = source_quadrature()
    small_x, small_w = source_quadrature(512)
    grid = np.linspace(0,.5,1001)
    comparison = max(abs(source_terms(float(b),pop_x,pop_w)[i]-source_terms(float(b),small_x,small_w)[i])
                     for b in np.linspace(0,.5,21) for i in (0,1,2))
    if comparison>1e-9:raise ValueError('source quadrature did not converge')
    results = {}
    for label,x,w in [('population_source',pop_x,pop_w),
                      ('finite512_posterior',finite_source,np.full(len(finite_source),1/len(finite_source)))]:
        root = brentq(lambda b:balance(b,x,w,scale,raw_sigma,cfg),0,.5,xtol=1e-14)
        trajectory = simulate(x,w,scale,raw_sigma,cfg)
        required = []
        for goal in (.4,.25,.1):
            force = -request(goal,raw_sigma,cfg)/scale**2
            required.append(dict(b=goal, lambda_for_stationary_balance=-source_terms(goal,x,w)[1]/force,
                                 target_ba=1-mixture_risk(goal)))
        results[label] = dict(equilibrium_b=root, equilibrium_ba=1-mixture_risk(root),
                              equilibrium_residual=balance(root,x,w,scale,raw_sigma,cfg),
                              equilibrium_source_kl=source_terms(root,x,w)[0]-source_terms(.5,x,w)[0],
                              trajectory=trajectory, required_lambda=required)
    results['density_only'] = simulate(pop_x,pop_w,scale,raw_sigma,cfg,density_only=True)
    results['reference'] = dict(initial_ba=1-mixture_risk(.5), target_bayes_ba=1-mixture_risk(0),
                                raw_sigma=raw_sigma, scale=scale, quadrature_max_error=comparison,
                                scope='fixed slope, exact boundary, no projection failure; finite source uses analytic posterior; not nonlinear TSBA or a performance bound')
    curves = dict(b=grid, population_source_gradient=np.array([source_terms(b,pop_x,pop_w)[1] for b in grid]),
                  request_raw=np.array([request(b,raw_sigma,cfg) for b in grid]),
                  target_ba=np.array([1-mixture_risk(b) for b in grid]))
    return results, curves, dict(source_x=pop_x, source_weights=pop_w, finite_source=finite_source)
