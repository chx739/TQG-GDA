"""Read-only replay of fixed R8 slopes at one common A probability mean."""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import rankdata

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    evaluate, fit, prior_control, r5_inputs, seals,
)
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.p_risk_replay import auc
from . import inputs, pilot

ADJUSTED = ('SourceCal', 'SourceSlope', 'TargetSlope', 'SwapSlope')
HISTORICAL = ('SourceFusion', 'JointExact')
METHODS = ADJUSTED + HISTORICAL
COMPARISONS = tuple(('TargetSlope', ref) for ref in (*ADJUSTED, *HISTORICAL) if ref != 'TargetSlope')


def fit_intercept(base, v, slope, tolerance):
    if (base.shape != v.shape or base.ndim != 1 or not len(base)
            or not np.isfinite(base).all() or not np.isfinite(v).all() or not np.isfinite(slope)):
        raise ValueError('Mass matching requires finite scores on one common A cohort')
    target = float(expit(base).mean())
    if slope == 0:
        return dict(intercept=0., slope=0., target=target, bracket=[0., 0.], residual=0.)
    bv = slope*v
    lower, upper = -float(bv.max()), -float(bv.min())

    def residual(a):
        return float(expit(base + (a + bv)).mean() - target)

    lv, uv = residual(lower), residual(upper)
    if lv > tolerance or uv < -tolerance:
        raise ValueError('A mass root fell outside its registered bracket')
    if abs(lv) <= 2*np.finfo(float).eps:
        a = lower
    elif abs(uv) <= 2*np.finfo(float).eps:
        a = upper
    else:
        a = float(brentq(residual, lower, upper, xtol=1e-14))
    error = abs(residual(a))
    if error > tolerance:
        raise ValueError(f'A mass replay error {error:g} exceeds {tolerance:g}')
    return dict(intercept=a, slope=float(slope), target=target, bracket=[lower, upper], residual=error)


def apply(base, v, params):
    if params['slope'] == params['intercept'] == 0.:
        return base.copy()
    return base + (params['intercept'] + params['slope']*v)


def compare_shift(before, after, amount, tolerance):
    delta = after - before - amount
    error = float(np.max(np.abs(delta)))
    if error > tolerance:
        raise ValueError('Mass matching changed more than a constant intercept')
    # A constant shift preserves the theoretical rank. Expose any numerical
    # tie changes instead of claiming rank equality from the formula alone.
    rank_changes = int((rankdata(before) != rankdata(after)).sum())
    ties_before = len(before) - len(np.unique(before))
    ties_after = len(after) - len(np.unique(after))
    return dict(constant_shift_max_error=error, changed_average_ranks=rank_changes,
                tied_excess_rows_before=ties_before, tied_excess_rows_after=ties_after)


def read_parent(root, plan):
    if digest(root/'FINAL.json') != plan['parent_final_sha256']:
        raise PermissionError('R8 parent FINAL identity differs')
    final = json.loads((root/'FINAL.json').read_bytes())
    if final['observation_sha256'] != plan['parent_obs_sha256']:
        raise PermissionError('R8 parent FINAL references a different observation')
    for relative, expected in final['files'].items():
        if digest(root/relative) != expected:
            raise PermissionError(f'Changed R8 parent FINAL member: {relative}')
    config = json.loads((root/'executed-config.json').read_bytes())['plan']
    for key in ('setting', 'directions', 'seeds'):
        if config[key] != plan[key]:
            raise ValueError(f'R8 fixed-mass scope differs at {key}')
    return config, pilot.verified_observations(root, plan['parent_obs_sha256'], config)


def freeze(parent, output, plan):
    config, observed = read_parent(parent, plan)
    sources, statuses = [json.loads(observed[name]) for name in ('source-statistics.json', 'block-status.json')]
    keys = {(task['seed'], task['target']): key for key, task in inputs.task_grid(plan)}
    parameters, records = {}, {}
    (output/'predictions').mkdir()
    for key, task in inputs.task_grid(plan):
        block = statuses[key]
        other_targets = [target for source, target in plan['directions'] if target != task['target']]
        if len(other_targets) != 1:
            raise ValueError('Swap control requires exactly two registered directions')
        swapped_key = keys[task['seed'], other_targets[0]]
        exact, swapped = [statuses[item]['solutions']['JointExact'] for item in (key, swapped_key)]
        if not exact['valid'] or not swapped['valid']:
            raise ValueError(f'{key}: a registered exact slope was unavailable')
        source = sources[block['source_key']]['fusion']
        slopes = {'SourceCal': 0., 'SourceSlope': source['weights'][0],
                  'TargetSlope': exact['slope'], 'SwapSlope': swapped['slope']}
        base = inputs.npz_member(observed[f'predictions/{key}-A.npz'])
        parameters[key] = {arm: fit_intercept(base['z_SourceCal'], base['v'], slope, plan['a_mean_error_max'])
                           for arm, slope in slopes.items()}
        records[key] = dict(task=task, swapped_from_target=other_targets[0], manifest=block['manifest'],
                            manifest_sha256=block['manifest_sha256'])
    # All A intercepts are now fixed. H is used only for forward predictions.
    for key, _ in inputs.task_grid(plan):
        check = {}
        old_intercepts = {'SourceFusion': sources[statuses[key]['source_key']]['fusion']['intercept'],
                          'JointExact': statuses[key]['solutions']['JointExact']['intercept']}
        for pool in ('A', 'H'):
            before = inputs.npz_member(observed[f'predictions/{key}-{pool}.npz'])
            data = {name: before['z_'+name] for name in HISTORICAL}
            for name in (*HISTORICAL, 'SourceCal'):
                np.testing.assert_array_equal(before['z_'+name] > 0, before['decision_'+name])
            data.update({arm: apply(before['z_SourceCal'], before['v'], par)
                         for arm, par in parameters[key].items()})
            np.testing.assert_array_equal(data['SourceCal'], before['z_SourceCal'])
            if pool == 'H':
                for relocated, orig in (('SourceSlope', 'SourceFusion'), ('TargetSlope', 'JointExact')):
                    amount = parameters[key][relocated]['intercept'] - old_intercepts[orig]
                    check[relocated] = compare_shift(data[orig], data[relocated], amount,
                                                     plan['logit_shift_replay_atol'])
            fit.write_npz(output/f'predictions/{key}-{pool}.npz', ids=before['ids'],
                          **{f'z_{arm}': data[arm] for arm in METHODS},
                          **{f'decision_{arm}': (data[arm] > 0).astype(np.int64) for arm in METHODS})
        records[key]['holdout_rank_replay'] = check
    write_json(output/'intercepts.json', parameters)
    write_json(output/'replay.json', records)
    files = ('executed-config.json', 'intercepts.json', 'replay.json',
             *(f'predictions/{key}-{pool}.npz' for key in records for pool in ('A', 'H')))
    write_json(output/'OBS.json', dict(stage='R8_FIXED_MASS_OBS', parent_final_sha256=plan['parent_final_sha256'],
                                      files={name: digest(output/name) for name in files}))
    return digest(output/'OBS.json'), config


def read_frozen(output, sha, plan):
    blob = (output/'OBS.json').read_bytes()
    if inputs.sha(blob) != sha:
        raise PermissionError('Fixed-mass OBS identity changed')
    seal = json.loads(blob)
    if seal['stage'] != 'R8_FIXED_MASS_OBS' or seal['parent_final_sha256'] != plan['parent_final_sha256']:
        raise PermissionError('Unexpected fixed-mass evidence chain')
    expected = {'executed-config.json', 'intercepts.json', 'replay.json'}
    expected.update(f'predictions/{key}-{pool}.npz' for key, _ in inputs.task_grid(plan) for pool in ('A', 'H'))
    if seal['files'].keys() != expected:
        raise PermissionError('Incomplete fixed-mass OBS')
    files = {}
    for relative, expected_sha in seal['files'].items():
        payload = (output/relative).read_bytes()
        if inputs.sha(payload) != expected_sha:
            raise PermissionError(f'Fixed-mass OBS member changed: {relative}')
        files[relative] = payload
    return files


def score(output, obs_sha, plan, r8_config, repo):
    records = read_frozen(output, obs_sha, plan)
    replay = json.loads(records['replay.json'])
    original_root = Path(r8_config['r7_archive'])
    endpoint_rows, pair_rows, cohorts = [], [], []
    with seals.verified_seal_chain(original_root, 'R7_FINAL') as scope:
        if json.loads((original_root/'seals/R7_FINAL.json').read_bytes())['digest'] != r8_config['r7_final_digest']:
            raise PermissionError('Fixed-mass R7 cohort chain changed')
        old_plan = json.loads(scope.read_bytes('executed-plan.json', 'R7_INPUT'))
        with r5_inputs.evaluation_labels(original_root, repository=repo) as labels:
            for key, task in inputs.task_grid(plan):
                item = replay[key]
                manifest = scope.read_bytes(item['manifest'], 'R7_INPUT')
                if inputs.sha(manifest) != item['manifest_sha256']:
                    raise PermissionError('Fixed-mass manifest no longer matches R8 roles')
                a = inputs.npz_member(records[f'predictions/{key}-A.npz'])
                h = inputs.npz_member(records[f'predictions/{key}-H.npz'])
                roles = json.loads(manifest)['ids']
                np.testing.assert_array_equal(a['ids'], roles['adapt'])
                np.testing.assert_array_equal(h['ids'], roles['holdout'])
                y = labels.load(item['manifest'], 'holdout')
                logits = {arm: prior_control.binary_logits(h[f'z_{arm}']) for arm in METHODS}
                actions = {arm: h[f'decision_{arm}'] for arm in METHODS}
                outcomes = {arm: actions[arm] == y for arm in METHODS}
                cohorts.append(dict(**task, ids=h['ids'], y=y, logits=logits, decisions=actions))
                metrics = {}
                for arm in METHODS:
                    row = evaluate.classification(logits[arm], y, actions['SourceCal'], decisions=actions[arm])
                    row.update(auroc=auc(h[f'z_{arm}'], y), a_mean_predicted_probability=expit(a[f'z_{arm}']).mean(),
                               a_predicted_positive_ratio=a[f'decision_{arm}'].mean(),
                               h_mean_predicted_probability=expit(h[f'z_{arm}']).mean())
                    metrics[arm] = row
                    endpoint_rows.append(dict(**task, method=arm, **row))
                for relocated, orig in (('SourceSlope', 'SourceFusion'), ('TargetSlope', 'JointExact')):
                    if not item['holdout_rank_replay'][relocated]['changed_average_ranks']:
                        if metrics[relocated]['auroc'] != metrics[orig]['auroc']:
                            raise ValueError(f'{key}: identical H ranks must give identical AUROC')
                for left, right in COMPARISONS:
                    corrected = int(np.sum(outcomes[left] & ~outcomes[right]))
                    harmed = int(np.sum(~outcomes[left] & outcomes[right]))
                    pair_rows.append(dict(**task, left=left, right=right, n=len(y), corrected=corrected,
                                          harmed=harmed, net=corrected-harmed, accuracy_pp=100*(corrected-harmed)/len(y),
                                          auroc_pp=100*(metrics[left]['auroc']-metrics[right]['auroc']),
                                          nll_delta=metrics[left]['sampled_nll']-metrics[right]['sampled_nll']))
    read_frozen(output, obs_sha, plan)
    intervals = pilot.classification_intervals(old_plan, cohorts, COMPARISONS, r8_config['r7_tau'], seed_intervals=True)
    add_exact_summary(output, pair_rows, plan)
    pilot.write_table(output/'endpoints.csv', endpoint_rows)
    pilot.write_table(output/'paired-seeds.csv', pair_rows)
    write_json(output/'paired-intervals.json', intervals)
    files = ('OBS.json', 'endpoints.csv', 'paired-seeds.csv', 'paired-intervals.json', 'decision.json')
    write_json(output/'FINAL.json', dict(stage='R8_FIXED_MASS_FINAL', observation_sha256=obs_sha,
                                        completed_blocks=len(cohorts), label_pools=['holdout'],
                                        files={name: digest(output/name) for name in files}))


def add_exact_summary(output, rows, plan):
    exact = {}
    for left, right in COMPARISONS:
        group = [r for r in rows if (r['left'], r['right']) == (left, right)]
        exact[right] = {}
        for _, target in plan['directions']:
            subset = [r for r in group if r['target'] == target]
            if {r['seed'] for r in subset} != set(plan['seeds']) or len(subset) != len(plan['seeds']):
                raise ValueError('Fixed-mass comparison has an incomplete seed grid')
            average = sum(Fraction(row['net'], row['n']) for row in subset)/len(subset)
            exact[right][f'M{target}'] = str(average)
        exact[right]['equal_directions'] = str(sum(Fraction(row['net'], row['n']) for row in group)/len(group))
    alternatives = ('SourceCal', 'SourceSlope', 'SwapSlope')
    all_positive = all(Fraction(exact[name][f'M{target}']) > 0 for name in alternatives for _, target in plan['directions'])
    new_development_proposal_supported = all_positive and Fraction(exact['SourceFusion']['equal_directions']) > 0
    write_json(output/'decision.json', dict(comparison='TargetSlope minus comparator, exact accuracy fractions',
                                          net_accuracy=exact, next_development_proposal_supported=new_development_proposal_supported))
