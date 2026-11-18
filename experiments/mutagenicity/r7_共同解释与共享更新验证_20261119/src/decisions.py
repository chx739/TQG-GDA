"""Explicit classification actions; probability logits and NLL never change.

The registered action uses each arm's own frozen, unlabeled adaptation mean
probability. Query rows, labels, and bootstrap samples cannot refit this prior.
These decisions have no finite-NLL-risk guarantee beyond the probability model.
"""
from copy import deepcopy

import numpy as np
from scipy.special import softmax


VERSION = 'adaptation_prior_balanced_v1'
ALIASES = {'J_argmax': 'J', 'B0_balanced': 'B0', 'C05_balanced': 'C05'}
BALANCED = {'J', 'B0_balanced', 'C05_balanced'}


def decision_policy(plan):
    value = plan.get('decision_policy')
    if value is None:
        return None
    if (not isinstance(value, dict) or set(value) != {'version', 'floor'}
            or value['version'] != VERSION or type(value['floor']) not in (int, float)
            or value['floor'] != 1e-8):
        raise ValueError('Unregistered decision_policy; expected fixed adaptation-prior rule and floor 1e-8')
    return {'version': VERSION, 'floor': 1e-8}


def policy_metadata(plan):
    policy = decision_policy(plan)
    return {} if policy is None else {'decision_policy': policy}


def probability_method(name):
    return ALIASES.get(name, name)


def decision_contrasts(plan, existing):
    result = list(existing)
    if decision_policy(plan) is not None:
        for pair in [('J', 'J_argmax'), ('J', 'B0_balanced'), ('J', 'C05_balanced'),
                     ('B0_balanced', 'B0'), ('C05_balanced', 'C05')]:
            if pair not in result:
                result.append(pair)
    return result


def _logits(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Classification decisions require finite binary logits')
    return values


def _rule(name, arrays):
    key = 'decision_rule_' + name
    if key not in arrays:
        if name in ALIASES:
            raise ValueError('Decision-control arm lacks frozen decision metadata: ' + name)
        return 'argmax'
    value = np.asarray(arrays[key])
    if value.shape != () or value.item() not in ('argmax', VERSION):
        raise ValueError('Invalid frozen classification rule')
    return value.item()


def decision_for(name, logits, arrays):
    values = _logits(logits)
    if _rule(name, arrays) == 'argmax':
        return values.argmax(axis=1)
    prior = np.asarray(arrays['decision_prior_' + name], dtype=np.float64)
    if prior.shape != (2,) or not np.isfinite(prior).all() or np.any(prior <= 0) or not np.isclose(prior.sum(), 1., rtol=0, atol=1e-14):
        raise ValueError('Invalid frozen adaptation decision prior')
    # Remove a class-common offset before adjustment to preserve small margins.
    return ((values-values.max(axis=1, keepdims=True))-np.log(prior)).argmax(axis=1)


def decision_metadata(name, arrays):
    if 'decision_rule_' + name not in arrays:
        return {}
    result = dict(decision_rule=_rule(name, arrays), probability_method=probability_method(name))
    if result['decision_rule'] == VERSION:
        prior = arrays['decision_prior_' + name]
        result.update(decision_prior_0=float(prior[0]), decision_prior_1=float(prior[1]))
    return result


def freeze_decisions(plan, arrays, records):
    """In-place OBS-only augmentation from frozen A probabilities; no labels."""
    policy = decision_policy(plan)
    if policy is None:
        return
    if any(key.startswith('decision_rule_') for key in arrays) or any(name in records for name in ALIASES):
        raise ValueError('Classification actions may be frozen only once')
    for alias, original in ALIASES.items():
        if original not in records:
            raise ValueError('Missing probability model for classification control: ' + original)
        for prefix in ('w_', 'adapt_logits_', 'adapt_probability_'):
            arrays[prefix+alias] = np.array(arrays[prefix+original], copy=True)
        records[alias] = {key: deepcopy(records[original][key]) for key in
                         ('valid', 'status', 'stop_reason', 'found_worst_risk') if key in records[original]}
    for name, record in records.items():
        logits = _logits(arrays['adapt_logits_' + name])
        probabilities = np.asarray(arrays['adapt_probability_' + name], dtype=np.float64)
        if not np.array_equal(probabilities, softmax(logits, axis=1)):
            raise ValueError('Frozen A probabilities disagree with logits: ' + name)
        rule = VERSION if name in BALANCED else 'argmax'
        arrays['decision_rule_' + name] = np.asarray(rule)
        if rule == VERSION:
            raw_prior = probabilities.mean(axis=0)
            prior = np.maximum(raw_prior, policy['floor'])
            prior /= prior.sum()
            arrays['decision_prior_raw_' + name] = raw_prior
            arrays['decision_prior_' + name] = prior
            arrays['decision_floor_' + name] = np.asarray(policy['floor'])
        arrays['adapt_decision_' + name] = decision_for(name, logits, arrays)
        record['decision'] = decision_metadata(name, arrays) | dict(
            probability_logits_unchanged=True, prior_fit_pool='unlabeled_adaptation_only' if rule == VERSION else None,
            target_labels_used=False, risk_scope='unshifted_probability_NLL; classification_action_not_risk_certified')


def verify_frozen_decisions(arrays):
    """Verify A provenance before replay; never derive a threshold from H."""
    names = [key[2:] for key in arrays if key.startswith('w_')]
    active = any(name in ALIASES for name in names)
    if not active:
        if any(key.startswith('decision_rule_') for key in arrays):
            raise ValueError('Frozen decision policy lacks its explicit comparison arms')
        return
    if not set(ALIASES) <= set(names):
        raise ValueError('Incomplete frozen classification comparison arms')
    for name in names:
        if 'decision_rule_' + name not in arrays or 'adapt_decision_' + name not in arrays:
            raise ValueError('Missing sealed A classification actions: ' + name)
        expected_rule = VERSION if name in BALANCED else 'argmax'
        if _rule(name, arrays) != expected_rule:
            raise ValueError('Frozen decision rule differs from its registered arm')
        if name in ALIASES:
            for prefix in ('w_', 'adapt_logits_', 'adapt_probability_'):
                np.testing.assert_array_equal(arrays[prefix+name], arrays[prefix+ALIASES[name]])
        if expected_rule == VERSION:
            if np.asarray(arrays['decision_floor_'+name]).item() != 1e-8:
                raise ValueError('Frozen decision prior floor changed')
            raw = arrays['adapt_probability_'+name].mean(axis=0)
            prior = np.maximum(raw, 1e-8)
            prior /= prior.sum()
            np.testing.assert_array_equal(arrays['decision_prior_raw_'+name], raw)
            np.testing.assert_array_equal(arrays['decision_prior_'+name], prior)
        np.testing.assert_array_equal(arrays['adapt_decision_'+name],
                                      decision_for(name, arrays['adapt_logits_'+name], arrays))


def cohort_decisions(arrays, records, pool='holdout'):
    """Explicit actions for the paired bootstrap; legacy callers return {}."""
    if not any(key.startswith('decision_rule_') for key in arrays):
        return {}
    result = {}
    for name, record in records.items():
        if not record['valid']:
            result[name] = None
            continue
        values = decision_for(name, arrays[pool+'_logits_'+name], arrays)
        if 'decision_rule_'+name in arrays:
            np.testing.assert_array_equal(arrays[pool+'_decision_'+name], values)
        result[name] = values
    return result
