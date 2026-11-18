"""Label-blind independent queries and finite-bank re-solves for R7 pilot blocks."""
from copy import deepcopy

import numpy as np
from scipy.special import softmax

from tsba.shared_update.solver import relative_risks

from .controls import solve_pointwise
from .fit import master, predict


def frozen_family(base, cfg):
    family = deepcopy(base)
    family.config.tau = cfg.tau
    family.threshold = family.reference_loss + cfg.tau
    family.fit_report['threshold'] = family.threshold
    return family


def audit(plan, cfg, base, z, u, ids, arrays, records):
    from .separation_audit import independent_search

    options = plan['phases']['P']['independent_audit']
    family = frozen_family(base, cfg)
    original = [family._candidate(theta, True) for theta in arrays['theta']]
    if any(candidate is None for candidate in original):
        raise ValueError('Infeasible candidate in saved joint bank')
    margin = u @ (arrays['w_J'][1] - arrays['w_J'][0])
    rows = np.lexsort((ids, -np.abs(margin)))[:options['individual_graph_count']]
    directions = {name: u @ arrays[f'w_{name}'].T for name in options['aggregate_fixed_updates']}
    for row in rows:
        for sign in options['individual_query_signs']:
            correction = np.zeros_like(z)
            correction[row, options['individual_class']] = sign
            directions[f'graph_{ids[row]}_{sign:+d}'] = correction
    seed = cfg.seed + options['seed_offset']
    known = list(original)
    log = {}
    complete = len(rows) == options['individual_graph_count'] and all(
        records[name]['valid'] for name in options['aggregate_fixed_updates'])
    for query, value in directions.items():
        found, detail = independent_search(family, value, seed=seed, starts=options['inner_starts'],
                                           maxiter=options['inner_maxiter'], known=original)
        if not isinstance(found, (list, tuple)):
            raise TypeError('Independent search must return the list of feasible explanations')
        for candidate in found:
            checked = family._candidate(candidate.parameters, True)
            if checked is None:
                raise ValueError('Independent search returned an infeasible explanation')
            known.append(checked)
        complete &= bool(detail.get('successful_feasible_runs', 0) or detail.get('status') == 'flat_objective')
        log[query] = detail
    bank = family._unique(known)
    stats = np.stack([explanation.statistic for explanation in bank])
    risk = relative_risks(z, u, arrays['w_J'], stats)
    witness_at = int(np.argmax(risk))
    found_harm = risk[witness_at] > options['positive_observed_risk_witness_threshold']
    witness = dict(risk=float(risk[witness_at]), observation_loss=bank[witness_at].observation_loss,
                   parameter=bank[witness_at].parameters.tolist()) if found_harm else None
    corrections, results = {}, {}
    corrections['J_plus'], results['J_plus'] = master(z, u, stats, cfg)
    posteriors = np.stack([family._target_terms(explanation.parameters)[-1] for explanation in bank])
    for name, mass in [('P_plus', False), ('P_mass_plus', True)]:
        arm = 'pointwise_mass' if mass else 'pointwise'
        corrections[name], results[name] = solve_pointwise(
            z, u, posteriors, cfg, plan['controls'][arm]['exact_separation_limit'], mass=mass, ids=ids)
    out = dict(audit_theta=np.stack([explanation.parameters for explanation in bank]), audit_stats=stats)
    for name, w in corrections.items():
        value = predict(name, z, u, w)
        out.update({f'w_{name}': w, f'adapt_logits_{name}': value, f'adapt_probability_{name}': softmax(value, axis=1)})
    results['J_plus']['independent_audit'] = dict(complete=complete, witness=witness, seed=seed, queries=log,
                                                  original_candidate_count=len(original), expanded_candidate_count=len(bank),
                                                  minimum_observation_loss=min(c.observation_loss for c in bank),
                                                  frozen_reference_loss=family.reference_loss,
                                                  fixed_J_worst_risk=float(risk[witness_at]))
    return out, results
