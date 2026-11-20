"""Paired graph-level scoring on both frozen directions."""
from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy.special import expit


MAIN = 'correspondence'
BASE = 'B'
SWAP = 'swap'
RANK = 'B_same_positive_count'
ARMS = (BASE, MAIN, SWAP, RANK)
REFERENCES = (BASE, SWAP, RANK)


def actions(margin):
    return (margin > 0.).astype(np.int64)


def match_positive_count(ids, margin, count):
    order = np.lexsort((ids, -margin))
    predictions = np.zeros(len(ids), dtype=np.int64)
    predictions[order[:count]] = 1
    return predictions


def predictions(ids, margin, delta, swapped_delta):
    np.testing.assert_array_equal(swapped_delta, -delta)
    actual = margin + delta
    swapped = margin + swapped_delta
    result = dict(graph_ids=ids, B=margin, delta=delta, swapped_delta=swapped_delta,
                  correspondence=actual, swap=swapped,
                  action_B=actions(margin), action_correspondence=actions(actual),
                  action_swap=actions(swapped))
    result['action_' + RANK] = match_positive_count(ids, margin, int(result['action_' + MAIN].sum()))
    if not all(np.isfinite(column).all() for column in result.values()):
        raise ValueError('Nonfinite frozen prediction')
    return result


def support(role, dictionary, prediction, parity):
    delta = prediction['delta']
    disagreements = {f'{a}_vs_{b}': int(np.count_nonzero(prediction['action_' + a] != prediction['action_' + b]))
                     for a, b in combinations(ARMS, 2)}
    votes = [row['n'] for row in dictionary['words'].values()]
    voting_ids = {vote['graph_id'] for row in dictionary['words'].values() for vote in row['votes']}
    return dict(
        D_graphs=len(role['D']), U_graphs=len(role['U']), U_distinct_fingerprints=len(role['U_representatives']),
        E_graphs=len(role['E']), candidate_words=len(votes),
        candidate_word_support_histogram={str(n): votes.count(n) for n in sorted(set(votes))},
        voting_U_fingerprints=len(voting_ids), total_word_fingerprint_votes=sum(votes),
        pairs=len(dictionary['pairs']), paired_words=2 * len(dictionary['pairs']),
        nonzero_delta_E_graphs=int(np.count_nonzero(delta)),
        nonzero_delta_E_fraction=float(np.count_nonzero(delta) / len(delta)),
        E_predicted_positive_counts={arm: int(prediction['action_' + arm].sum()) for arm in ARMS},
        action_disagreements=disagreements,
        accuracy_gain_upper_bounds_pp={pair: 100 * n / len(delta) for pair, n in disagreements.items()},
        r9_B_prediction_parity=parity,
    )


def identifiability(supports):
    reasons = {}
    for name, row in supports.items():
        failed = []
        if not row['pairs']:
            failed.append('no_pairs')
        if not row['nonzero_delta_E_graphs']:
            failed.append('all_E_deltas_zero')
        if failed:
            reasons[name] = failed
    return dict(scoring_permitted=not reasons, unlabeled_stop_reasons=reasons)


def metrics(y, action, margin, indices):
    labels, action = y[indices], action[indices]
    correct = action == labels
    tn = np.sum(correct & (labels == 0), axis=1)
    tp = np.sum(correct & (labels == 1), axis=1)
    fp_fn = np.sum(~correct, axis=1)
    support0, support1 = np.sum(labels == 0, axis=1), np.sum(labels == 1, axis=1)
    f0_denominator, f1_denominator = 2*tn + fp_fn, 2*tp + fp_fn
    f0 = 2*tn / np.maximum(f0_denominator, 1)
    f1 = 2*tp / np.maximum(f1_denominator, 1)
    ba = (tn / support0 + tp / support1) / 2 if {0, 1} <= set(y) else None
    result = dict(accuracy=correct.mean(axis=1), balanced_accuracy=ba, macro_f1=(f0 + f1) / 2)
    if margin is not None:
        result['nll'] = np.logaddexp(0., (1 - 2 * labels) * margin[indices]).mean(axis=1)
    return result


def summary(vector):
    if vector is None:
        return dict(value=None, percentile_95_interval=None)
    return dict(value=float(vector[0]), percentile_95_interval=np.percentile(vector[1:], [2.5, 97.5]).tolist())


def score_direction(p, labels, draws, rng):
    cohorts = [np.flatnonzero(labels == c) for c in (0, 1)]
    bootstrap = np.concatenate([rng.choice(ids, (draws, len(ids)), replace=True) for ids in cohorts if len(ids)], axis=1)
    indices = np.vstack((np.arange(len(labels)), bootstrap))
    values = {arm: metrics(labels, p['action_' + arm], p.get(arm), indices) for arm in ARMS}
    outcomes = {arm: {key: summary(value) for key, value in values[arm].items()} for arm in ARMS}
    slope = ((labels - expit(p[BASE])) * p['delta'])[indices].mean(axis=1)
    comparisons = {}
    main_correct = p['action_' + MAIN] == labels
    for arm in REFERENCES:
        other_correct = p['action_' + arm] == labels
        corrected, harmed = main_correct & ~other_correct, other_correct & ~main_correct
        gains = {}
        for key, other in values[arm].items():
            main = values[MAIN][key]
            improvement = None if other is None else (other - main if key == 'nll' else main - other)
            gains[key] = summary(improvement)
        comparisons[arm] = dict(corrected=int(corrected.sum()), harmed=int(harmed.sum()),
                                net=int(corrected.sum()) - int(harmed.sum()), gains=gains)
    return dict(n=len(labels), class_counts={str(c): len(ids) for c, ids in enumerate(cohorts)},
                arms=outcomes, correspondence_comparisons=comparisons, nll_improvement_slope=summary(slope))


def promotion(score, plan):
    cutoff = Fraction(*plan['minimum_accuracy_gain_fraction'])
    by_arm = {}
    for arm in REFERENCES:
        gains = {name: Fraction(row['correspondence_comparisons'][arm]['net'], row['n'])
                 for name, row in score.items()}
        average = sum(gains.values()) / len(gains)
        meets = (average > 0 if arm == RANK
                 else average >= cutoff and all(gain >= 0 for gain in gains.values()))
        by_arm[arm] = dict(pass_accuracy=meets, equal_direction_gain_fraction=str(average),
                           equal_direction_gain_pp=float(100 * average),
                           per_direction_gain_fractions={name: str(gain) for name, gain in gains.items()})
    positive_slopes = {name: row['nll_improvement_slope']['value'] > 0. for name, row in score.items()}
    return dict(passed=all(positive_slopes.values()) and all(row['pass_accuracy'] for row in by_arm.values()),
                threshold_fraction=str(cutoff), positive_slopes=positive_slopes, accuracy=by_arm)
