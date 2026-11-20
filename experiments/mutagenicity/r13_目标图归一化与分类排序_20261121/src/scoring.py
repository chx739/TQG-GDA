"""Role-gated scoring and exact registered pilot decisions for R13."""
from fractions import Fraction
import json

import numpy as np
from scipy.stats import rankdata

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11_inputs
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src.run import summarize_pairs
from . import historical, observation

MEANS = {'Other_node': ('Other_node_1', 'Other_node_2'), 'Other_graph': ('Other_graph_1', 'Other_graph_2')}
NODE_GATES = ('B0', 'SourceCal', 'S_node', 'Other_node')
GRAPH_GATES = ('B0', 'SourceCal', 'S_graph', 'Other_graph', 'A_node', 'B0_equal_positive')
INTERACTION = ('A_graph-A_node', 'S_graph-S_node')


def comparisons(graph):
    basic = tuple((m, 'B0') for m in observation.METHODS_NODE if m not in ('B0', 'A_node'))
    basic += tuple(('A_node', m) for m in (*NODE_GATES, *MEANS['Other_node']))
    optional = tuple(('A_node', m) for m in observation.HISTORICAL)
    optional += tuple((m, 'B0') for m in observation.HISTORICAL)
    if graph:
        basic += tuple(('A_graph', m) for m in (*GRAPH_GATES, *MEANS['Other_graph']))
        basic += (('S_graph', 'B0'), ('S_graph', 'S_node'), ('Other_graph_1', 'B0'), ('Other_graph_2', 'B0'),
                  ('B0_equal_positive', 'B0'), INTERACTION)
        optional += tuple(('A_graph', m) for m in observation.HISTORICAL)
    return basic, optional


def coefficients(left, right):
    if (left, right) == INTERACTION:
        return {'A_graph': 1, 'A_node': -1, 'S_graph': -1, 'S_node': 1}
    targets = MEANS.get(right, (right,))
    return {left: Fraction(1)} | {name: Fraction(-1, len(targets)) for name in targets}


def compare(left, right, actions, correct, resampled):
    terms = coefficients(left, right)
    if not set(terms) <= set(correct):
        return None
    net = sum(weight * int(correct[name].sum()) for name, weight in terms.items())
    draws = sum(float(weight) * resampled[name] for name, weight in terms.items()) if resampled else None
    result = dict(net_exact=str(net), corrected_exact=None, harmed_exact=None, changed_exact=None)
    if (left, right) != INTERACTION:
        targets = MEANS.get(right, (right,))
        c, h, d = (sum(int(mask(name).sum()) for name in targets) for mask in (
            lambda name: correct[left] & ~correct[name],
            lambda name: ~correct[left] & correct[name],
            lambda name: actions[left] != actions[name],
        ))
        corrected, harmed, changed = (Fraction(x, len(targets)) for x in (c, h, d))
        if net != corrected - harmed:
            raise AssertionError('Paired exact net correction arithmetic differs')
        result.update(corrected_exact=str(corrected), harmed_exact=str(harmed), changed_exact=str(changed))
    return Fraction(net, len(next(iter(correct.values())))), draws, result


def auc(logits, y):
    positive = int(y.sum())
    negative = len(y) - positive
    if not positive or not negative:
        return None
    ranks = rankdata(logits[:,1] - logits[:,0])
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def gate(plan, pair_values):
    """Only the pilot's registered seed controls investment decisions."""
    directions = [(s, t) for s in plan['domains'] for t in plan['domains'] if s != t]
    seed = plan['phase1_seed']
    rows = {}
    for setting in plan['settings']:
        row = {}
        for left, controls, direction_controls in (('A_graph', GRAPH_GATES, ('B0', 'A_node')),
                                                    ('A_node', NODE_GATES, ('B0',))):
            per_pair = {right: [pair_values.get((setting, s, t, seed, left, right)) for s, t in directions]
                        for right in controls}
            means = {right: None if None in values else sum(values)/len(values) for right, values in per_pair.items()}
            positive = sum(all(per_pair[c][i] is not None and per_pair[c][i] > 0 for c in direction_controls)
                           for i in range(len(directions)))
            passed = (all(x is not None and x > 0 for x in means.values())
                      and positive >= plan['minimum_positive_directions'])
            row[left] = dict(passed=passed, jointly_positive_directions=positive,
                             mean_pp={right: None if value is None else float(100*value) for right, value in means.items()})
        rows[setting] = row
    required = plan['minimum_positive_settings']
    n_graph = sum(row['A_graph']['passed'] for row in rows.values())
    n_node = sum(row['A_node']['passed'] for row in rows.values())
    branch = 'graph' if n_graph >= required else 'node' if n_node >= required else 'stop'
    return dict(branch=branch, settings=rows, graph_settings=n_graph, node_settings=n_node)


def summary(plan, seeds, endpoints):
    results = []
    names = sorted({row['method'] for row in endpoints})
    metrics = ('accuracy', 'balanced_accuracy', 'recall_0', 'recall_1', 'auroc',
               'sampled_nll', 'predicted_class_1')
    expected = len(seeds) * len(plan['domains']) * (len(plan['domains']) - 1)
    for setting in plan['settings']:
        for arm in names:
            rows = [row for row in endpoints if row['setting'] == setting and row['method'] == arm]
            if not rows:
                continue
            present = len(rows)
            averages = {name: float(np.mean([r[name] for r in rows]))
                        if all(r[name] is not None for r in rows) else None for name in metrics}
            results.append(dict(setting=setting, method=arm, coverage=present, grid=expected, **averages))
    return results


def _verified(output, sha, plan, seeds):
    members = prior_run.verified_members(output, 'OBS', sha, observation.required_members(plan, seeds),
                                          {'r7_final_digest': plan['r7_final_digest']})
    roles = json.loads(members['roles.json'])
    _, expected = observation.observations_expected(plan, seeds)
    if set(roles) != set(expected):
        raise PermissionError('The sealed task inventory differs from the registered grid')
    return roles, members


def score(output, repo, plan, seeds, weights, obs_sha, *, pilot):
    roles, members = _verified(output, obs_sha, plan, seeds)
    graph = 'graph' in weights
    baseline, optional = comparisons(graph)
    history = json.loads(members['historical-plan.json'])
    c = historical.CArchive(plan)
    if history != c.plan:
        raise PermissionError('Historical C configuration differs before scoring')
    verified, grants = {}, {}
    # Authenticate the entire sealed prediction grid and every H authorization
    # before constructing the first label reader.
    for key, role in roles.items():
        task = {k: role[k] for k in observation.IDENTITY}
        payload = r11_inputs.npz_member(members[f'predictions/{key}.npz'])
        methods = observation.registered_predictions(payload, role, graph)
        manifest, sha, context = historical.context(c, task, repo)
        if (r11_inputs.key(**task) != key or sha != role['manifest_sha256']
                or manifest['ids']['holdout'] != role['held_ids']):
            raise PermissionError('Sealed H ID and task manifest do not agree')
        origin = json.loads(members[f"stats/{role['group_key']}.json"])
        if ({k: origin[k] for k in ('setting', 'source', 'seed')} != {k: task[k] for k in ('setting','source','seed')}
                or origin['original']['sha256'] != manifest['provenance']['r5_files'][origin['original']['path']]):
            raise PermissionError('The sealed stats and checkpoint lineage do not agree')
        y_role = task['setting'], task['target']
        label_port = r11_inputs.raw_grant(context, manifest, 'holdout')
        if grants.setdefault(y_role, label_port) != label_port:
            raise PermissionError('Shared H label grants differ across directions')
        verified[key] = (methods, payload)
    reps, seed, minimum, _ = paired._settings(history)
    cohorts, accesses = {}, {}
    endpoints, paired_rows = [], []
    point, curves = {}, {}
    with r11_inputs.GraphLabelSession('holdout') as labels:
        for key, (methods, arrays) in verified.items():
            task = {k: roles[key][k] for k in observation.IDENTITY}
            setting, source, target, model_seed = (task[k] for k in observation.IDENTITY)
            group = setting, target
            ids = arrays['ids']
            if group not in cohorts:
                y = labels.load(grants[group], roles[key]['held_ids'])
                order = np.argsort(ids, kind='stable')
                cohort = paired._cohort(setting, target, ids[order], y[order], reps, seed, minimum)
                cohorts[group] = (y, order, cohort)
                accesses[key] = roles[key]['held_ids']
            y, order, cohort = cohorts[group]
            np.testing.assert_array_equal(ids[order], cohort['ids'])
            actions = {arm: arrays[f'decision_{arm}'] for arm in methods}
            correct = {arm: z == y for arm, z in actions.items()}
            draw_arms = {}
            if cohort['weights'] is not None:
                draws = cohort['weights'] @ np.column_stack([correct[arm][order] for arm in methods])
                draw_arms = dict(zip(methods, draws.T))
            for arm in methods:
                z = arrays[f'logits_{arm}']
                metrics = evaluate.classification(z, y, actions['B0'], decisions=actions[arm])
                if arm == 'B0_equal_positive':
                    metrics['sampled_nll'] = None
                endpoints.append(dict(**task, method=arm, auroc=auc(z,y), **metrics))
            for left, right in (*baseline, *optional):
                result = compare(left, right, actions, correct, draw_arms)
                if result is None:
                    continue
                value, draws, extra = result
                k = (setting, source, target, model_seed, left, right)
                point[k] = value
                if draws is not None:
                    curves[k] = draws
                rank_diff = None
                if left in actions and right in actions:
                    a = auc(arrays[f'logits_{left}'], y)
                    b = auc(arrays[f'logits_{right}'], y)
                    rank_diff = None if a is None or b is None else a - b
                paired_rows.append(dict(**task, left=left, right=right, n=len(y), accuracy_pp=float(100*value),
                                        auroc_difference=rank_diff, **extra))
    _verified(output, obs_sha, plan, seeds)
    write_table(output / 'endpoints.csv', endpoints)
    write_table(output / 'endpoint-summary.csv', summary(plan, seeds, endpoints))
    write_table(output / 'paired-blocks.csv', paired_rows)
    pair_summaries = summarize_pairs(plan | {'seeds': seeds}, point, curves, baseline)
    write_table(output / 'paired-summary.csv', pair_summaries)
    write_json(output / 'label-access.json', dict(obs_sha256=obs_sha, actual_h_reads=accesses))
    write_json(output / 'cohort-audit.json', [v[2]['audit'] for v in cohorts.values()])
    decision = gate(plan, point) if pilot else None
    write_json(output / 'gate.json', {'pilot': pilot, 'decision': decision})
    files = ['OBS.json', 'endpoints.csv', 'endpoint-summary.csv', 'paired-blocks.csv', 'paired-summary.csv',
             'label-access.json', 'cohort-audit.json', 'gate.json']
    final = prior_run.seal(output, 'FINAL', files, {'obs_sha256': obs_sha})
    return final, decision
