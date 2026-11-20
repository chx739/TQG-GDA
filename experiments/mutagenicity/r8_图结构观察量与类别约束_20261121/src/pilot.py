"""Run the registered R8 pilot, with H labels restricted to post-OBS evaluation."""
from collections import Counter
import csv
import io
import json

import numpy as np
from scipy.special import expit
from scipy.stats import t
import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    evaluate, fit, paired, prior_control, r5_inputs,
)
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.p_risk_replay import auc
from . import composition, inputs, projection

BASELINES = ('B0', 'SourceCal', 'MLLS', 'v', 'SourceFusion', 'J_argmax', 'J')
PROJECTED = ('Joint', 'SamePrior', 'JointExact', 'SamePriorExact')
METHODS = BASELINES + PROJECTED
COMPARISONS = (
    ('Joint', 'SourceCal'), ('Joint', 'SamePrior'), ('Joint', 'SourceFusion'),
    ('Joint', 'MLLS'), ('Joint', 'B0'), ('Joint', 'v'), ('Joint', 'J_argmax'), ('Joint', 'J'),
    ('SamePrior', 'SourceCal'), ('JointExact', 'SourceCal'), ('JointExact', 'SamePriorExact'),
    ('JointExact', 'SourceFusion'), ('SamePriorExact', 'SourceCal'), ('SourceFusion', 'SourceCal'),
    ('v', 'SourceCal'), ('MLLS', 'SourceCal'), ('SourceCal', 'B0'),
)
ATOL = 1e-12


def group_stats(v):
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)))


def take_counts(counts, ids):
    return [counts[i] for i in ids]


def source_fit(scope, reader, key, entry, counts, historical, plan):
    path, _, manifest, context = entry
    ids = manifest['ids']
    original = inputs.npz_member(scope.read_bytes(f'blocks/{key}/fit_input.npz', 'R7_INPUT'),
                                 ['source_ids', 'source_y'])
    np.testing.assert_array_equal(original['source_ids'], ids['source'])
    relative = f"source_labels/M{manifest['source']}/seed{manifest['seed']}_fit.pt"
    labels = torch.load(io.BytesIO(context['read'](relative, 'INPUT')), weights_only=True, map_location='cpu').numpy()
    np.testing.assert_array_equal(labels, original['source_y'])
    validation_labels = reader.load(path, 'validation')
    source = take_counts(counts, ids['source'])
    vocabulary = composition.Vocabulary.fit(source)
    fitted = composition.fit_composition(vocabulary, source, labels, plan['composition'], plan['source_solver'])
    validation_v = expit(composition.composition_scores(vocabulary, take_counts(counts, ids['validation']), fitted))
    validation = inputs.npz_member(scope.read_bytes(f'blocks/{key}/validation_input.npz', 'R7_INPUT'),
                                   ['validation_z0', 'validation_ids'])
    np.testing.assert_array_equal(validation['validation_ids'], ids['validation'])
    calibrated = prior_control.calibrate(prior_control.margin(validation['validation_z0']), historical['calibrator'])
    fusion = composition.fit_logistic(validation_v[:, None], validation_labels, calibrated, plan['source_solver'])
    stats = [group_stats(validation_v[validation_labels == c]) for c in (0, 1)]
    return vocabulary, dict(composition=fitted, fusion=fusion, validation_v_by_class=stats,
                            source_fit_n=len(labels), validation_ids=ids['validation'],
                            source_fit_ids=ids['source'], calibrator=historical['calibrator'])


def fit_adaptation(scope, key, entry, counts, fitted, historical, plan):
    _, _, manifest, _ = entry
    ids = manifest['ids']['adapt']
    vocabulary, source = fitted
    value = composition.composition_scores(vocabulary, take_counts(counts, ids), source['composition'])
    v = expit(value)
    width = f"tau-{plan['r7_tau']:.3f}"
    block = inputs.npz_member(scope.read_bytes(f'blocks/{key}/{width}-observed.npz', 'R7_OBS'),
                              ['adapt_logits_B0', 'adapt_logits_J', 'adapt_decision_J'])
    raw = prior_control.margin(block['adapt_logits_B0'])
    calibrated = prior_control.calibrate(raw, source['calibrator'])
    mu = np.asarray([row['mean'] for row in source['validation_v_by_class']])
    ns = [row['n'] for row in source['validation_v_by_class']]
    eps_s, eps_a = projection.band_widths(ns, len(v), plan['projection']['alpha'])
    solutions, intervals = {}, {}
    for exact, joint, prior in ((False, 'Joint', 'SamePrior'), (True, 'JointExact', 'SamePriorExact')):
        bands = projection.Bands.build(v, mu, np.zeros(2) if exact else eps_s, 0. if exact else eps_a)
        interval = projection.prior_interval(bands, plan['projection'])
        intervals[joint] = interval
        solutions[joint] = projection.joint_projection(calibrated, bands, interval, plan['projection'], exact=exact)
        solutions[prior] = projection.prior_projection(calibrated, interval, plan['projection'])
    arrays = dict(ids=np.asarray(ids), v=v, z_B0=raw, z_SourceCal=calibrated, z_v=value,
                  z_MLLS=calibrated+historical['prior']['log_odds_shift'],
                  z_SourceFusion=calibrated+composition.scores(v[:, None], source['fusion']),
                  z_J=prior_control.margin(block['adapt_logits_J']), decision_J=block['adapt_decision_J'])
    arrays['z_J_argmax'] = arrays['z_J']
    return arrays, dict(intervals=intervals, solutions=solutions, eps_source=eps_s.tolist(),
                        eps_adapt=eps_a, adapt_n=len(v), mu=mu.tolist())


def predict_holdout(scope, donor, key, entry, counts, fitted, historical, status):
    _, _, manifest, _ = entry
    ids = manifest['ids']['holdout']
    vocabulary, source = fitted
    value = composition.composition_scores(vocabulary, take_counts(counts, ids), source['composition'])
    v = expit(value)
    historical_h = inputs.npz_member(donor(f'predictions/{key}.npz'))
    np.testing.assert_array_equal(historical_h['ids'], ids)
    archived = inputs.npz_member(scope.read_bytes(f'blocks/{key}/holdout_input.npz', 'R7_INPUT'),
                                 ['holdout_z0', 'holdout_ids'])
    np.testing.assert_array_equal(archived['holdout_ids'], ids)
    old_predictions = inputs.npz_member(
        scope.read_bytes(f'blocks/{key}/tau-{status["tau"]:.3f}-heldout.npz', 'R7_OBS'),
        [f'holdout_{field}_{name}' for name in ('B0', 'J', 'J_argmax') for field in ('logits', 'decision')])
    for name in ('B0', 'J', 'J_argmax'):
        for field in ('logits', 'decision'):
            np.testing.assert_array_equal(historical_h[f'{field}_{name}'], old_predictions[f'holdout_{field}_{name}'])
    baseline = {name: prior_control.margin(historical_h[f'logits_{name}'])
                for name in ('B0', 'SourceCal', 'MLLS', 'J_argmax', 'J')}
    np.testing.assert_allclose(baseline['B0'], prior_control.margin(archived['holdout_z0']), atol=ATOL, rtol=0)
    replay = prior_control.calibrate(baseline['B0'], source['calibrator'])
    np.testing.assert_array_equal(replay, baseline['SourceCal'])
    np.testing.assert_array_equal(replay+historical['prior']['log_odds_shift'], baseline['MLLS'])
    arrays = dict(ids=np.asarray(ids), v=v, z_v=value,
                  z_SourceFusion=replay+composition.scores(v[:, None], source['fusion']))
    for name, z in baseline.items():
        arrays[f'z_{name}'] = z
        arrays[f'decision_{name}'] = historical_h[f'decision_{name}']
        if name != 'J':
            np.testing.assert_array_equal(arrays[f'decision_{name}'], (z > 0).astype(int))
    return predictions(arrays, status)


def predictions(arrays, status):
    base, v = arrays['z_SourceCal'], arrays['v']
    for name, solution in status['solutions'].items():
        new = projection.projected_logits(base, v, solution)
        if new is not None:
            arrays[f'z_{name}'] = new
    for name in METHODS:
        z = arrays.get(f'z_{name}')
        if z is not None:
            if not np.isfinite(z).all():
                raise ValueError(f'Nonfinite predictions for {name}')
            arrays.setdefault(f'decision_{name}', (z > 0).astype(np.int64))
    return arrays


def make_observation(output, plan, repo):
    (output/'predictions').mkdir()
    with inputs.registered_inputs(plan, repo) as (archive, scope, donor, original, contexts):
        counts, topology = inputs.graph_preflight(next(iter(contexts.values()))[3], plan)
        historical = json.loads(donor('parameters.json'))
        if set(historical) != set(contexts):
            raise ValueError('R7 controls must cover the exact R8 ten-block grid')
        fitted, source_status, block_status = {}, {}, {}
        # This scope reads source validation labels only. All A fits complete
        # before any H graph is passed to the prediction phase.
        with r5_inputs.evaluation_labels(archive, repository=repo) as reader:
            for key, task in inputs.task_grid(plan):
                source_key = f"M{task['source']}-seed-{task['seed']}"
                entry = contexts[key]
                if source_key not in fitted:
                    fitted[source_key] = source_fit(scope, reader, key, entry, counts, historical[key], plan)
                    source_status[source_key] = fitted[source_key][1]
                source = source_status[source_key]
                manifest = entry[2]
                if (source['source_fit_ids'] != manifest['ids']['source']
                        or source['validation_ids'] != manifest['ids']['validation']
                        or source['calibrator'] != historical[key]['calibrator']):
                    raise ValueError('Source model or source role changed by target direction')
                arrays, status = fit_adaptation(scope, key, entry, counts, fitted[source_key], historical[key], plan)
                block_status[key] = dict(task=task, source_key=source_key, manifest=entry[0],
                                         manifest_sha256=entry[1], tau=plan['r7_tau'], **status)
                fit.write_npz(output/f'predictions/{key}-A.npz', **predictions(arrays, status))
        for key, status in block_status.items():
            arrays = predict_holdout(scope, donor, key, contexts[key], counts, fitted[status['source_key']],
                                     historical[key], status)
            fit.write_npz(output/f'predictions/{key}-H.npz', **arrays)
    write_json(output/'source-statistics.json', source_status)
    write_json(output/'block-status.json', block_status)
    write_json(output/'input-checks.json', topology)
    names = ['executed-config.json', 'source-statistics.json', 'block-status.json', 'input-checks.json']
    names.extend(f'predictions/{key}-{pool}.npz' for key in block_status for pool in ('A', 'H'))
    files = {name: digest(output/name) for name in names}
    write_json(output/'OBS.json', dict(stage='R8_OBS', parent=plan['r7_final_digest'],
                                      r7_prior_final_sha256=plan['r7_prior_control_final_sha256'], files=files))
    return digest(output/'OBS.json'), original


def verified_observations(output, obs_sha, plan):
    blob = (output/'OBS.json').read_bytes()
    if inputs.sha(blob) != obs_sha:
        raise PermissionError('R8 observation seal changed')
    obs = json.loads(blob)
    if obs['parent'] != plan['r7_final_digest']:
        raise PermissionError('R8 observation belongs to another archive')
    expected = {'executed-config.json', 'source-statistics.json', 'block-status.json', 'input-checks.json'}
    expected.update(f'predictions/{key}-{pool}.npz' for key, _ in inputs.task_grid(plan) for pool in ('A', 'H'))
    if set(obs['files']) != expected:
        raise PermissionError('R8 observation files are incomplete')
    frozen = {}
    for relative, sha in obs['files'].items():
        payload = (output/relative).read_bytes()
        if inputs.sha(payload) != sha:
            raise PermissionError(f'R8 sealed observation changed: {relative}')
        frozen[relative] = payload
    return frozen


def ci_difference(a, b):
    # Welch interval for two per-class means, conditional on their trained v.
    av, bv = a['sd']**2/a['n'], b['sd']**2/b['n']
    delta = a['mean'] - b['mean']
    variance = av+bv
    df = variance**2/(av**2/(a['n']-1)+bv**2/(b['n']-1)) if variance else 1.
    radius = float(t.ppf(.975, df)*np.sqrt(variance))
    return dict(delta=delta, lower=delta-radius, upper=delta+radius)


def pool_diagnostics(task, pool, arrays, y, status, source, plan):
    v = arrays['v']
    val_stats = source['validation_v_by_class']
    sample_stats = [group_stats(v[y == c]) for c in (0, 1)]
    separation = ci_difference(sample_stats[1], sample_stats[0])
    source_separation = ci_difference(val_stats[1], val_stats[0])
    class_rows = [dict(**task, pool=pool, label=c, source_n=val_stats[c]['n'], source_mean=val_stats[c]['mean'],
                       source_sd=val_stats[c]['sd'], target_n=sample_stats[c]['n'], target_mean=sample_stats[c]['mean'],
                       target_sd=sample_stats[c]['sd'], **ci_difference(sample_stats[c], val_stats[c]),
                       target_separation=separation['delta'], target_separation_low=separation['lower'],
                       target_separation_high=separation['upper'], source_separation=source_separation['delta'],
                       source_separation_low=source_separation['lower'], source_separation_high=source_separation['upper'])
                  for c in (0, 1)]
    eps_s, eps = projection.band_widths([row['n'] for row in val_stats], len(y), plan['projection']['alpha'])
    moments, scores = [], []
    for name in (*METHODS, 'true_labels'):
        z = arrays.get('z_'+name)
        if z is None and name != 'true_labels':
            scores.append(dict(**task, pool=pool, method=name, n=len(y), valid=False,
                               status=status['solutions'][name]['status']))
            continue
        q = y if name == 'true_labels' else expit(z)
        mean_q = q.mean()
        if name != 'true_labels':
            accuracy = evaluate.classification(prior_control.binary_logits(z), y, arrays['decision_B0'],
                                               decisions=arrays['decision_'+name])
            method_status = status['solutions'].get(name, {}).get('status', 'fixed')
            scores.append(dict(**task, pool=pool, method=name, valid=True, status=method_status,
                               **accuracy, auroc=auc(z, y), mean_p1=mean_q, class_prior_abs_error=abs(mean_q-y.mean())))
        for c, weights in enumerate((1-q, q)):
            residual = np.mean(weights*(v-status['mu'][c]))
            bound = eps_s[c]*weights.mean()+eps
            moments.append(dict(**task, pool=pool, method=name, label=c, n=len(y), residual=residual,
                                source_bandwidth=eps_s[c], pool_bandwidth=eps, bound=bound,
                                absolute_excess=abs(residual)-bound, feasible=bool(abs(residual)<=bound),
                                mean_p1=mean_q, true_p1=y.mean(), class_prior_abs_error=abs(mean_q-y.mean())))
    return class_rows, moments, scores


def write_table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def evaluate_observation(output, obs_sha, original, plan, repo):
    frozen = verified_observations(output, obs_sha, plan)
    status = json.loads(frozen['block-status.json'])
    sources = json.loads(frozen['source-statistics.json'])
    class_rows, moment_rows, endpoint_rows, pair_rows, h_cohorts = [], [], [], [], []
    with r5_inputs.evaluation_labels(Path(plan['r7_archive']), repository=repo) as reader:
        for key, task in inputs.task_grid(plan):
            block = status[key]
            for pool, suffix in (('adapt', 'A'), ('holdout', 'H')):
                arrays = inputs.npz_member(frozen[f'predictions/{key}-{suffix}.npz'])
                y = reader.load(block['manifest'], pool, authenticated_snapshot=True)
                cls, moments, scores = pool_diagnostics(task, pool, arrays, y, block,
                                                        sources[block['source_key']], plan)
                class_rows.extend(cls)
                moment_rows.extend(moments)
                endpoint_rows.extend(scores)
                if pool != 'holdout':
                    continue
                logits, actions = {}, {}
                by_method = {row['method']: row for row in scores}
                for name in METHODS:
                    if by_method[name]['valid']:
                        logits[name] = prior_control.binary_logits(arrays['z_'+name])
                        actions[name] = arrays['decision_'+name]
                    else:
                        logits[name] = None
                h_cohorts.append(dict(**task, ids=arrays['ids'], y=y, logits=logits, decisions=actions))
                for left, right in COMPARISONS:
                    row = dict(**task, left=left, right=right, n=len(y), available=left in actions and right in actions)
                    if row['available']:
                        l_correct = actions[left] == y
                        r_correct = actions[right] == y
                        corrected = int((l_correct & ~r_correct).sum())
                        harmed = int((r_correct & ~l_correct).sum())
                        row.update(corrected=corrected, harmed=harmed, net=corrected-harmed,
                                   accuracy_pp=100*(corrected-harmed)/len(y))
                        for metric in ('balanced_accuracy', 'auroc', 'sampled_nll'):
                            row[metric] = by_method[left][metric]-by_method[right][metric]
                    pair_rows.append(row)
    verified_observations(output, obs_sha, plan)
    paired_summary = paired.paired_bootstrap(original, h_cohorts, phase='P', comparisons=COMPARISONS, tau=plan['r7_tau'])
    paired_summary['interval_rows'] = [r for r in paired_summary['interval_rows'] if r['metric'] != 'directional_risk_gap']
    gate = decide(status, pair_rows, original)
    for name, rows in (('conditional-means', class_rows), ('moment-residuals', moment_rows),
                       ('endpoints', endpoint_rows), ('paired-seeds', pair_rows)):
        write_table(output/f'{name}.csv', rows)
    write_json(output/'paired-intervals.json', paired_summary)
    write_json(output/'gate.json', gate)
    outputs = ('OBS.json', 'conditional-means.csv', 'moment-residuals.csv', 'endpoints.csv',
               'paired-seeds.csv', 'paired-intervals.json', 'gate.json')
    write_json(output/'FINAL.json', dict(stage='R8_FINAL', observation_sha256=obs_sha, target_label_pools=['adapt', 'holdout'],
                                        files={relative: digest(output/relative) for relative in outputs}))
    return gate


def decide(status, pair_rows, original):
    def count(left, right, direction=None):
        rows = [r for r in pair_rows if (r['left'], r['right']) == (left, right)
                and (direction is None or [r['source'], r['target']] == direction)]
        valid = [r for r in rows if r['available']]
        return dict(available=len(valid), planned=len(rows),
                    accuracy_pp=float(np.mean([r['accuracy_pp'] for r in valid])) if rows and len(valid) == len(rows) else None)

    comparisons = {f'M{source}->M{target}': {
        name: count('Joint', name, [source, target]) for name in ('SourceCal', 'SamePrior')
    } for source, target in original['phases']['P']['directions']}
    extra = count('Joint', 'SourceFusion')
    grad = {name: dict(Counter(row['solutions'][name]['status'] for row in status.values()))
            for name in PROJECTED}
    required = [item['accuracy_pp'] for row in comparisons.values() for item in row.values()]
    required.append(extra['accuracy_pp'])
    passed = all(delta is not None and delta > 0. for delta in required)
    return dict(passed=passed, directions=comparisons, pooled_vs_source_fusion=extra, arm_status_counts=grad,
                next_phase='larger_development_validation' if passed else None,
                classification_endpoint='holdout Accuracy, equal directions within seed')
