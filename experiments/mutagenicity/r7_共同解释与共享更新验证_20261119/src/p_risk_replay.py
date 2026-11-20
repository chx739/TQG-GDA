"""Read-only R7 P replay and log-loss attribution on the sealed A/H cohorts."""
import csv
import io
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, log_softmax
from scipy.stats import rankdata

from .decisions import decision_for, verify_frozen_decisions
from .r5_inputs import evaluation_labels
from .seals import verified_seal_chain


def read_npz(scope, relative, stage):
    with np.load(io.BytesIO(scope.read_bytes(relative, stage)), allow_pickle=False) as arrays:
        return dict(arrays)


def auc(margins, labels):
    positives = labels.sum()
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    positive_ranks = rankdata(margins)[labels == 1].sum()
    return float((positive_ranks - positives * (positives + 1) / 2) / (positives * negatives))


def replay_block(scope, plan, seed):
    key = f"blocks/{plan['setting']}-M{plan['source']}_to_M{plan['target']}-seed-{seed}"
    width = f"tau-{plan['tau']:.3f}"
    atol = plan['atol']
    manifest = json.loads(scope.read_bytes(f'{key}/manifest.json', 'R7_INPUT'))
    for name, value in dict(setting=plan['setting'], source=plan['source'], target=plan['target'], seed=seed).items():
        if manifest[name] != value:
            raise ValueError(f'{key}: inconsistent manifest {name}')
    fit = read_npz(scope, f'{key}/fit_input.npz', 'R7_INPUT')
    holdout = read_npz(scope, f'{key}/holdout_input.npz', 'R7_INPUT')
    standard = read_npz(scope, f'{key}/standardization.npz', 'R7_OBS')
    held_standard = read_npz(scope, f'{key}/holdout_standardization.npz', 'R7_OBS')
    observed = read_npz(scope, f'{key}/{width}-observed.npz', 'R7_OBS')
    held = read_npz(scope, f'{key}/{width}-heldout.npz', 'R7_OBS')
    records = json.loads(scope.read_bytes(f'{key}/{width}-observed.json', 'R7_OBS'))
    if not {'B0_balanced', 'J_argmax'} <= set(records):
        raise ValueError('Expected the original sealed balanced and argmax actions')
    verify_frozen_decisions(observed)
    outputs, errors = {}, []
    for pool, inputs, frozen, std in (('adapt', fit, observed, standard),
                                       ('holdout', holdout, held, held_standard)):
        ids = inputs[f'{pool}_ids']
        np.testing.assert_array_equal(ids, manifest['ids'][pool])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f'{key}: duplicate {pool} IDs')
        h = inputs[f'{pool}_h']
        u = np.column_stack(((h - standard['mean']) / standard['scale'], np.ones(len(h))))
        np.testing.assert_allclose(u, std[f'{pool}_u'], atol=atol, rtol=0)
        z0 = h @ fit['source_head_weight'].T + fit['source_head_bias']
        np.testing.assert_allclose(z0, inputs[f'{pool}_z0'], atol=atol, rtol=0)
        np.testing.assert_array_equal(inputs[f'{pool}_z0'], std[f'{pool}_z0'])
        methods = {}
        for name in plan['methods']:
            if not records[name]['valid']:
                raise ValueError(f'{key}: {name} was unavailable')
            w = observed[f'w_{name}']
            margin = z0[:, 1] - z0[:, 0] + u @ (w[1] - w[0])
            # Recompute normalized probabilities from the weight difference,
            # independently of the original compensated Torch prediction path.
            replayed = np.column_stack((-np.logaddexp(0., margin), -np.logaddexp(0., -margin)))
            logits = frozen[f'{pool}_logits_{name}']
            logp = log_softmax(logits, axis=1)
            np.testing.assert_allclose(replayed, logp, atol=atol, rtol=0)
            np.testing.assert_allclose(np.exp(logp), frozen[f'{pool}_probability_{name}'], atol=atol, rtol=0)
            if not np.array_equal((margin > 0).astype(int), logits.argmax(1)):
                raise ValueError(f'{key}: unadjusted class replay failed for {name}')
            decision = decision_for(name, logits, observed)
            np.testing.assert_array_equal(decision, frozen[f'{pool}_decision_{name}'])
            methods[name] = dict(logp=logp, decision=decision, margin=logits[:, 1]-logits[:, 0])
            errors.append(float(np.max(np.abs(replayed-logp))))
        outputs[pool] = dict(ids=ids, features=u, methods=methods)
    if np.intersect1d(outputs['adapt']['ids'], outputs['holdout']['ids']).size:
        raise ValueError(f'{key}: A and H overlap')
    return key, outputs, observed, records, max(errors)


def paired_intervals(base, updated, labels, delta, options, seed):
    rng = np.random.default_rng(np.random.SeedSequence([options['seed'], seed]))
    nll, auc_delta = [], []
    for _ in range(options['replicates']):
        selected = rng.integers(len(labels), size=len(labels))
        nll.append(delta[selected].mean())
        y = labels[selected]
        b, j = auc(base[selected], y), auc(updated[selected], y)
        if b is not None:
            auc_delta.append(j-b)
    if not auc_delta:
        raise ValueError('No bootstrap draw with both classes')
    nll_lo, nll_hi = np.quantile(nll, [.025, .975])
    auc_lo, auc_hi = np.quantile(auc_delta, [.025, .975])
    return dict(h_nll_ci_low=nll_lo, h_nll_ci_high=nll_hi, h_auc_ci_low=auc_lo,
                h_auc_ci_high=auc_hi, h_auc_bootstrap_draws=len(auc_delta))


def score_pool(pool, labels, threshold):
    ids, methods = pool['ids'], pool['methods']
    row = np.arange(len(ids))
    by_method = {}
    for name, result in methods.items():
        result['nll'] = -result['logp'][row, labels]
        correct = result['decision'] == labels
        by_method[name] = dict(nll=result['nll'].mean(), accuracy=correct.mean(),
                              balanced_accuracy=np.mean([correct[labels == c].mean() for c in (0, 1)]),
                              auroc=auc(result['margin'], labels), predicted_class1=result['decision'].mean(),
                              mean_p1=expit(result['margin']).mean(), n=len(ids))
    base, j = methods['B0'], methods['J']
    np.testing.assert_array_equal(methods['B0_balanced']['margin'], base['margin'])
    delta = j['nll'] - base['nll']
    j_raw_pred = (j['margin'] > 0).astype(int)
    j_confidence = expit(np.abs(j['margin']))
    high_conf_wrong = (j_raw_pred != labels) & (j_confidence >= threshold)
    per_graph = [dict(graph_id=gid, label=y, b0_nll=base['nll'][i], j_nll=j['nll'][i], delta_nll=delta[i],
                      contribution=delta[i]/len(ids), b0_p1=expit(base['margin'][i]), j_p1=expit(j['margin'][i]),
                      b0_decision=base['decision'][i], b0_balanced_decision=methods['B0_balanced']['decision'][i],
                      j_balanced_decision=j['decision'][i], j_raw_prediction=j_raw_pred[i],
                      j_raw_confidence=j_confidence[i], j_raw_high_conf_wrong=int(high_conf_wrong[i]))
                 for i, (gid, y) in enumerate(zip(ids, labels))]
    group_rows = []
    bwrong = base['decision'] != labels
    jwrong = j_raw_pred != labels
    masks = [('class_0', labels == 0), ('class_1', labels == 1),
             ('wrong_stays_wrong_raw', bwrong & jwrong), ('correct_to_wrong_raw', ~bwrong & jwrong),
             ('wrong_to_correct_raw', bwrong & ~jwrong), ('correct_stays_correct_raw', ~bwrong & ~jwrong),
             ('high_conf_wrong_raw', high_conf_wrong), ('other_than_high_conf_wrong_raw', ~high_conf_wrong)]
    for name, mask in masks:
        group_rows.append(dict(group=name, n=int(mask.sum()), worsened=int((delta[mask] > 0).sum()),
                               mean_delta_nll=delta[mask].mean() if mask.any() else None,
                               contribution=delta[mask].sum()/len(ids)))
    positive = np.sort(delta[delta > 0])[::-1]
    positive_total = positive.sum()
    mass = dict(n=len(ids), class1_count=int(labels.sum()), nll_worsened=len(positive),
                nll_improved=int((delta < 0).sum()), delta_nll=delta.mean(), median_delta=np.median(delta),
                positive_contribution=positive_total/len(ids),
                negative_contribution=delta[delta < 0].sum()/len(ids),
                high_conf_wrong_count=int(high_conf_wrong.sum()),
                high_conf_wrong_contribution=delta[high_conf_wrong].sum()/len(ids),
                high_conf_wrong_positive_share=np.maximum(delta[high_conf_wrong], 0.).sum()/positive_total)
    for n in (1, 5, 10, 20):
        mass[f'top{n}_positive_share'] = positive[:n].sum()/positive_total
        mask = np.ones(len(ids), dtype=bool)
        mask[np.argsort(-delta, kind='stable')[:n]] = False
        mass[f'drop_top{n}_delta_nll'] = delta[mask].mean()
    return by_method, delta, group_rows, per_graph, mass


def run(source, plan, repo):
    tables = {key: [] for key in ('seeds', 'endpoints', 'groups', 'graphs', 'candidate_risks', 'concentration')}
    with verified_seal_chain(source, 'R7_FINAL') as scope:
        # Bind the verified chain to the registered attempt before opening labels.
        sealed = json.loads((Path(source)/'seals/R7_FINAL.json').read_bytes())
        if sealed['digest'] != plan['source_final_digest']:
            raise ValueError('R7_FINAL differs from registered source')
        provenance = json.loads(scope.read_bytes('provenance.json', 'R7_INPUT'))
        if provenance['git_commit'] != plan['source_execution_commit']:
            raise ValueError('Historical execution code identity differs')
        original_plan = json.loads(scope.read_bytes('executed-plan.json', 'R7_INPUT'))
        if original_plan['shared_update']['primary_tau'] != plan['tau']:
            raise ValueError('Replay width differs from historical primary width')
        blocks = {seed: replay_block(scope, plan, seed) for seed in plan['seeds']}

        def original_rows(name, fields):
            text = scope.read_bytes(f'{name}.csv', 'R7_FINAL').decode()
            return {tuple(row[field] for field in fields): row for row in csv.DictReader(io.StringIO(text))
                    if row['setting'] == plan['setting'] and int(row['source']) == plan['source']
                    and int(row['target']) == plan['target'] and float(row['tau']) == plan['tau']}

        moments = original_rows('moments', ('seed', 'pool'))
        endpoints = original_rows('endpoints', ('seed', 'pool', 'method'))
        with evaluation_labels(source, repository=repo) as reader:
            for seed, (key, pools, obs, records, parity) in blocks.items():
                identity = dict(seed=seed)
                a = pools['adapt']['methods']
                delta_logp = a['J']['logp'] - a['B0']['logp']
                # With class 0 as the reference, risk = normalizer change minus
                # the dot product between the class 1 moment and logit-margin update.
                update = obs['w_J'][1] - obs['w_J'][0]
                normalizer = -delta_logp[:, 0].mean()
                banks = {}
                for name, field in (('retained', 'stats'), ('independent_union', 'audit_stats')):
                    stats = obs[field]
                    np.testing.assert_allclose(stats.sum(axis=1),
                                               np.broadcast_to(pools['adapt']['features'].mean(0), stats[:, 0].shape),
                                               atol=plan['atol'], rtol=0)
                    risks = normalizer - stats[:, 1] @ update
                    banks[name] = risks
                    tables['candidate_risks'].extend(dict(seed=seed, bank=name, candidate_index=i,
                                                          fixed_j_a_delta_nll=risk)
                                                      for i, risk in enumerate(risks))
                archive_a = moments[str(seed), 'adapt']
                audit = records['J_plus']['independent_audit']
                original, expanded = banks['retained'], banks['independent_union']
                if (len(original) != audit['original_candidate_count'] or
                        len(expanded) != audit['expanded_candidate_count']):
                    raise ValueError(f'{key}: audit bank count differs')
                # The saved expanded bank is deduplicated. Check the maximum,
                # allowing for the archive's roundoff and equivalence tolerance.
                np.testing.assert_allclose(
                    [original.max(), expanded.max()],
                    [float(archive_a['retained_worst_change']), audit['fixed_J_worst_risk']],
                    atol=plan['atol'], rtol=0)
                if expanded.max() + plan['atol'] < original.max():
                    raise ValueError(f'{key}: candidate union decreased worst risk')
                seed_row = dict(**identity, retained_candidates=len(original), independent_union_candidates=len(expanded),
                                retained_worst_a_delta_nll=original.max(), union_worst_a_delta_nll=expanded.max(),
                                independent_audit_complete=audit['complete'], independent_queries=len(audit['queries']),
                                positive_harm_witness=audit['witness'] is not None,
                                replay_max_logprob_error=parity)
                for pool, prefix in (('adapt', 'a'), ('holdout', 'h')):
                    labels = reader.load(f'{key}/manifest.json', pool, authenticated_snapshot=True)
                    if pool == 'adapt':
                        oracle = read_npz(scope, f"{key}/tau-{plan['tau']:.3f}-oracle.npz", 'R7_ORACLE')
                        np.testing.assert_array_equal(labels, oracle['adapt_labels'])
                    metrics, delta, groups, graphs, mass = score_pool(
                        pools[pool], labels, plan['high_confidence_probability'])
                    np.testing.assert_allclose(delta.mean(), float(moments[str(seed), pool]['empirical_risk_change']),
                                               atol=plan['atol'], rtol=0)
                    for method, row in metrics.items():
                        old = endpoints[str(seed), pool, method]
                        for new_name, old_name in (('nll', 'sampled_nll'), ('accuracy', 'accuracy'),
                                                   ('balanced_accuracy', 'balanced_accuracy')):
                            np.testing.assert_allclose(row[new_name], float(old[old_name]), atol=plan['atol'], rtol=0)
                        tables['endpoints'].append(dict(**identity, pool=pool, method=method, **row))
                    for name, rows in (('groups', groups), ('graphs', graphs), ('concentration', [mass])):
                        tables[name].extend(dict(**identity, pool=pool, **row) for row in rows)
                    seed_row.update({f'{prefix}_true_delta_nll': delta.mean(),
                                     f'{prefix}_j_plus_delta_nll': metrics['J_plus']['nll']-metrics['B0']['nll'],
                                     f'{prefix}_j_minus_b0_balanced_auc': metrics['J']['auroc']-metrics['B0_balanced']['auroc']})
                    if pool == 'holdout':
                        frozen = pools[pool]['methods']
                        seed_row.update(paired_intervals(frozen['B0']['margin'], frozen['J']['margin'], labels,
                                                         delta, plan['holdout_paired_bootstrap'], seed))
                seed_row['a_true_minus_union_worst'] = seed_row['a_true_delta_nll']-expanded.max()
                tables['seeds'].append(seed_row)
    return tables
