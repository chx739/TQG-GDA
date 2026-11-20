"""Verify and score the complete R10 structural OBS before opening exact H labels."""
from fractions import Fraction
import json
from statistics import mean, stdev

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.inputs import sha
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src.run import summarize_pairs
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src.scoring import auc
from . import structural, structural_observation as observation


PRIMARY = (('WL_selected', 'B0'), ('WL_selected', 'SourceCal'), ('WL_selected', 'Size'),
           ('WL_fixed', 'B0'), ('WL_selected', 'WL_fixed'))
R10_PAIRS = tuple(('WL_selected', method) for method in observation.R10_ARMS)
R13_PAIRS = tuple((f'R13_{method}', 'WL_selected') for method in observation.R13_ARMS)
PAIRS = PRIMARY + R10_PAIRS + R13_PAIRS
METRICS = ('accuracy', 'balanced_accuracy', 'recall_0', 'recall_1', 'auroc', 'sampled_nll', 'corrected', 'harmed')


def verify_before_label_access(output, repo, plan, fit_sha, obs_sha):
    fit, fit_roles, access = observation.verify_fit(output, plan, fit_sha)
    observed = prior_run.verified_members(output, 'OBS', obs_sha, observation.observation_members(plan),
                                           {'fit_sha256': fit_sha})
    roles = json.loads(observed['roles.json'])
    grid = observation.tasks(plan)
    if set(roles) != set(grid) or sha(observed['FIT.json']) != fit_sha:
        raise PermissionError('OBS does not bind the full registered grid and FIT')
    c = observation.open_c(plan)
    if json.loads(fit['historical-plan.json']) != c.plan:
        raise PermissionError('OBS historical plan differs from C')
    # Preflight the entire grid before constructing the label session.
    blocks, grants = {}, {}
    models = {group: json.loads(fit[f'fits/{group}.json']) for group in access}
    for key, task in grid.items():
        role = roles[key]
        group = observation.group_key(*(task[k] for k in structural.GROUP))
        model = models[group]
        if (role['group'] != group or role['fit_sha256'] != sha(fit[f'fits/{group}.json'])
                or any(model[k] != task[k] for k in structural.GROUP)
                or any(role[k] != value for k, value in fit_roles[key].items())):
            raise PermissionError('OBS differs from sealed FIT model identity or role')
        manifest, context = observation.bind(c, repo, task, role, access[group])
        cohort = task['setting'], task['target']
        grant = inputs.raw_grant(context, manifest, 'holdout')
        if grants.setdefault(cohort, grant) != grant:
            raise PermissionError('H grants differ across directions or model seeds')
        arrays = inputs.npz_member(observed[f'predictions/{key}.npz'])
        methods = observation.verify_predictions(arrays, role)
        if model['wl']['selected_C'] == 1.:
            np.testing.assert_array_equal(arrays['logits_WL_selected'], arrays['logits_WL_fixed'])
        blocks[key] = task, role, methods, arrays
    return c.plan, blocks, grants


def endpoint_summary(plan, endpoints):
    index = {(row['setting'], row['source'], row['target'], row['seed'], row['method']): row for row in endpoints}
    directions = [(s, t) for s in plan['domains'] for t in plan['domains'] if s != t]
    for setting in plan['settings']:
        for subset in [[direction] for direction in directions] + [directions]:
            s, t = subset[0] if len(subset) == 1 else (None, None)
            for method in (*observation.REQUIRED_ARMS, *observation.OPTIONAL):
                seeds = plan['seeds'][:1] if method.startswith('R13_') else plan['seeds']
                expected = len(subset) * len(seeds)
                rows = [[index.get((setting, src, tgt, seed, method)) for src, tgt in subset] for seed in seeds]
                present = sum(row is not None for group in rows for row in group)
                if not present:
                    continue
                result = dict(setting=setting, scope='direction' if len(subset) == 1 else 'setting',
                              source=s, target=t, method=method, blocks=present, matching_grid=expected,
                              full_grid=len(subset) * len(plan['seeds']))
                for metric in METRICS:
                    values = ([mean(row[metric] for row in group) for group in rows]
                              if present == expected and all(row[metric] is not None for group in rows for row in group)
                              else [])
                    result[f'{metric}_mean'] = mean(values) if values else None
                    result[f'{metric}_sd'] = stdev(values) if len(values) > 1 else None
                yield result


def score(output, repo, plan, fit_sha, obs_sha):
    original, blocks, grants = verify_before_label_access(output, repo, plan, fit_sha, obs_sha)
    reps, seed, minimum, _ = paired._settings(original)
    cohorts, h_access = {}, {}
    endpoints, comparisons, values, curves = [], [], {}, {}
    with inputs.GraphLabelSession('holdout') as labels:
        for key, (task, role, methods, arrays) in blocks.items():
            setting, source, target, model_seed = (task[k] for k in structural.IDENTITY)
            cohort_key = setting, target
            ids = role['holdout_ids']
            if cohort_key not in cohorts:
                y = labels.load(grants[cohort_key], ids)
                order = np.argsort(ids, kind='stable')
                cohorts[cohort_key] = y, order, paired._cohort(
                    setting, target, arrays['ids'][order], y[order], reps, seed, minimum)
                h_access[key] = ids
            y, order, cohort = cohorts[cohort_key]
            np.testing.assert_array_equal(arrays['ids'][order], cohort['ids'])
            actions = {name: arrays[f'action_{name}'] for name in methods}
            correct = {name: action == y for name, action in actions.items()}
            weights = cohort['weights']
            sampled = (dict(zip(methods, (weights @ np.column_stack(
                       [correct[arm][order] for arm in methods])).T)) if weights is not None else {})
            for name, action in actions.items():
                z = arrays[f'logits_{name}']
                metrics = evaluate.classification(z, y, actions['B0'], decisions=action)
                endpoints.append(dict(**task, method=name, auroc=auc(z, y), **metrics))
            for left, right in PAIRS:
                if left not in correct or right not in correct:
                    continue
                gained = int(np.sum(correct[left] & ~correct[right]))
                harmed = int(np.sum(~correct[left] & correct[right]))
                delta = Fraction(gained - harmed, len(y))
                identity = (setting, source, target, model_seed, left, right)
                values[identity] = delta
                if sampled:
                    curves[identity] = sampled[left] - sampled[right]
                comparisons.append(dict(**task, left=left, right=right, n=len(y), corrected=gained,
                                        harmed=harmed, net=gained - harmed, accuracy_pp=float(100*delta)))
    if any(sum(row['method'] == name for row in endpoints) != len(blocks) for name in observation.REQUIRED_ARMS):
        raise ValueError('Primary endpoint coverage differs from the sealed prediction grid')
    if any(sum((row['left'], row['right']) == pair for row in comparisons) != len(blocks) for pair in PRIMARY):
        raise ValueError('Primary pairing coverage differs from the sealed prediction grid')
    summaries = summarize_pairs(plan, values, curves, PRIMARY + R10_PAIRS)
    summaries.extend(summarize_pairs(dict(plan, seeds=plan['seeds'][:1]), values, curves, R13_PAIRS))
    write_table(output / 'endpoints.csv', endpoints)
    write_table(output / 'paired-blocks.csv', comparisons)
    write_table(output / 'endpoint-summary.csv', list(endpoint_summary(plan, endpoints)))
    write_table(output / 'paired-summary.csv', summaries)
    write_json(output / 'label-access.json', dict(obs_sha256=obs_sha, actual_h_reads=h_access))
    write_json(output / 'cohort-audit.json', [cohort[2]['audit'] for cohort in cohorts.values()])
    write_json(output / 'coverage.json', dict(blocks=len(blocks), source_groups=len(observation.fit_members(plan))-4,
        h_cohorts=len(cohorts), methods={name: sum(row['method'] == name for row in endpoints)
                                      for name in (*observation.REQUIRED_ARMS, *observation.OPTIONAL)},
        comparisons=[dict(left=left, right=right,
                          blocks=sum((row['left'], row['right']) == (left, right) for row in comparisons))
                     for left, right in PAIRS]))
    files = ['OBS.json', 'endpoints.csv', 'endpoint-summary.csv', 'paired-blocks.csv', 'paired-summary.csv',
             'label-access.json', 'cohort-audit.json', 'coverage.json']
    return prior_run.seal(output, 'FINAL', files, {'obs_sha256': obs_sha})
