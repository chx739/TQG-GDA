"""Freeze source-calibrated and MLLS predictions before scoring the R7 P holdout."""
import csv
import json

import numpy as np
from scipy.special import expit

from experiments.common.io import digest, write_json

from .decisions import decision_for, verify_frozen_decisions
from .evaluate import classification
from .fit import read_npz as read_local_npz, write_npz
from .p_risk_replay import auc, read_npz
from .paired import paired_bootstrap
from .prior_control import adjusted_scores, binary_logits, calibrate, fit_calibrator, fit_prior, margin, verify_rank_preservation
from .r5_inputs import evaluation_labels
from .seals import verified_seal_chain

EXISTING = ('B0', 'B0_balanced', 'J_argmax', 'J')
NEW = ('SourceCal', 'MLLS', 'SourceCal_balanced', 'MLLS_balanced')
METHODS = EXISTING + NEW


def task_grid(plan):
    for source, target in plan['directions']:
        for seed in plan['seeds']:
            key = f"{plan['setting']}-M{source}_to_M{target}-seed-{seed}"
            yield key, dict(setting=plan['setting'], source=source, target=target, seed=seed)


def fit_blocks(archive, scope, plan, repo):
    fitted, fitted_source = {}, {}
    width = f"tau-{plan['tau']:.3f}"
    atol = plan['checks_atol']
    with evaluation_labels(archive, repository=repo) as labels:
        for key, task in task_grid(plan):
            relative = 'blocks/' + key
            path = f'{relative}/manifest.json'
            manifest = json.loads(scope.read_bytes(path, 'R7_INPUT'))
            if any(manifest[name] != value for name, value in task.items()):
                raise ValueError(f'{key}: manifest differs from the fixed task grid')
            validation = read_npz(scope, f'{relative}/validation_input.npz', 'R7_INPUT')
            val_ids, z_val = validation['validation_ids'], validation['validation_z0']
            np.testing.assert_array_equal(val_ids, manifest['ids']['validation'])
            fit_data = read_npz(scope, f'{relative}/fit_input.npz', 'R7_INPUT')
            np.testing.assert_array_equal(fit_data['adapt_ids'], manifest['ids']['adapt'])
            np.testing.assert_allclose(validation['validation_h'] @ fit_data['source_head_weight'].T
                                       + fit_data['source_head_bias'], z_val, rtol=0, atol=atol)
            observed = read_npz(scope, f'{relative}/{width}-observed.npz', 'R7_OBS')
            valid = json.loads(scope.read_bytes(f'{relative}/{width}-observed.json', 'R7_OBS'))
            if any(valid[name]['valid'] is not True for name in EXISTING):
                raise ValueError(f'{key}: a registered comparator is invalid')
            verify_frozen_decisions(observed)
            a_margins = margin(observed['adapt_logits_B0'])
            np.testing.assert_allclose(a_margins, margin(fit_data['adapt_z0']), rtol=0, atol=atol)
            source_key = task['source'], task['seed']
            if source_key not in fitted_source:
                y = labels.load(path, 'validation')
                calibrator = fit_calibrator(margin(z_val), y, plan['calibration'])
                source_prior = expit(calibrate(margin(z_val), calibrator)).mean()
                fitted_source[source_key] = (val_ids, z_val, calibrator, source_prior)
            ref_ids, ref_z, calibrator, source_prior = fitted_source[source_key]
            np.testing.assert_array_equal(val_ids, ref_ids)
            np.testing.assert_array_equal(z_val, ref_z)
            prior = fit_prior(calibrate(a_margins, calibrator), source_prior, plan['mlls'])
            fitted[key] = dict(task=task, calibrator=calibrator, prior=prior, source_validation_n=len(ref_ids),
                               adaptation_n=len(a_margins), source_selected_epoch=manifest['provenance']['selected_source_epoch'],
                               source_validation_used_for_checkpoint_selection=True)
    return fitted


def predict_block(h, observed, parameters, atol):
    arrays = {}
    for name in EXISTING:
        logits = h[f'holdout_logits_{name}']
        decision = h[f'holdout_decision_{name}']
        np.testing.assert_array_equal(decision, decision_for(name, logits, observed))
        arrays[f'logits_{name}'] = logits
        arrays[f'decision_{name}'] = decision
    source = margin(arrays['logits_B0'])
    np.testing.assert_array_equal(arrays['logits_B0_balanced'], arrays['logits_B0'])
    np.testing.assert_array_equal(arrays['logits_J_argmax'], arrays['logits_J'])
    cal, shifted, source_balanced, target_balanced = adjusted_scores(
        source, parameters['calibrator'], parameters['prior'])
    np.testing.assert_allclose(source_balanced, target_balanced, rtol=0, atol=atol)
    source_ties = verify_rank_preservation(source, cal, atol)
    mlls_ties = verify_rank_preservation(source, shifted, atol)
    base_actions = (source_balanced > 0).astype(int)
    target_actions = (target_balanced > 0).astype(int)
    different = base_actions != target_actions
    if np.any(np.abs(source_balanced[different]) > atol):
        raise ValueError('Balanced MLLS and source calibration actions differ beyond boundary roundoff')
    # Canonical algebraically cancelled score resolves a roundoff-only tie by
    # the same fixed argmax convention in both balanced arms.
    for name, score, decisions in (('SourceCal', cal, cal > 0), ('MLLS', shifted, shifted > 0),
                                   ('SourceCal_balanced', cal, base_actions), ('MLLS_balanced', shifted, base_actions)):
        arrays[f'logits_{name}'] = binary_logits(score)
        arrays[f'decision_{name}'] = decisions.astype(int)
    checks = dict(rank_ties_source_cal=source_ties, rank_ties_mlls=mlls_ties,
                  balanced_raw_roundoff_action_disagreements=int(different.sum()),
                  balanced_cancel_max_error=float(np.max(np.abs(source_balanced-target_balanced))))
    return arrays, checks


def freeze(archive, plan, repo, output):
    width = f"tau-{plan['tau']:.3f}"
    atol = plan['checks_atol']
    with verified_seal_chain(archive, 'R7_FINAL') as scope:
        record = json.loads((archive/'seals/R7_FINAL.json').read_bytes())
        if record['digest'] != plan['source_final_digest']:
            raise ValueError('Original FINAL differs from the registered source')
        original = json.loads(scope.read_bytes('executed-plan.json', 'R7_INPUT'))
        if (original['shared_update']['primary_tau'] != plan['tau']
                or original['phases']['P']['directions'] != plan['directions']
                or original['phases']['P']['setting'] != plan['setting']
                or original['real_model_seeds'] != plan['seeds']):
            raise ValueError('Original P task grid or primary width differs')
        parameters = fit_blocks(archive, scope, plan, repo)
        write_json(output/'parameters.json', parameters)
        children = output/'predictions'
        children.mkdir()
        checks = {}
        for key, values in parameters.items():
            relative = f'blocks/{key}'
            observed = read_npz(scope, f'{relative}/{width}-observed.npz', 'R7_OBS')
            held = read_npz(scope, f'{relative}/{width}-heldout.npz', 'R7_OBS')
            inputs = read_npz(scope, f'{relative}/holdout_input.npz', 'R7_INPUT')
            manifest = json.loads(scope.read_bytes(f'{relative}/manifest.json', 'R7_INPUT'))
            ids = inputs['holdout_ids']
            np.testing.assert_array_equal(ids, manifest['ids']['holdout'])
            if np.intersect1d(ids, manifest['ids']['adapt']).size:
                raise ValueError(f'{key}: target A/H are not disjoint')
            np.testing.assert_allclose(margin(held['holdout_logits_B0']), margin(inputs['holdout_z0']),
                                       rtol=0, atol=atol)
            arrays, checks[key] = predict_block(held, observed, values, atol)
            write_npz(children/f'{key}.npz', ids=ids, **arrays)
        write_json(output/'prediction-checks.json', checks)
    paths = ['executed-config.json', 'parameters.json', 'prediction-checks.json']
    paths.extend(f'predictions/{key}.npz' for key in parameters)
    obs = dict(stage='PRIOR_CONTROL_OBS', parent=plan['source_final_digest'],
               files={name: digest(output/name) for name in paths})
    write_json(output/'OBS.json', obs)
    return digest(output/'OBS.json'), original


def verify_obs(output, sha):
    if digest(output/'OBS.json') != sha:
        raise PermissionError('Prior-control observation seal changed')
    record = json.loads((output/'OBS.json').read_bytes())
    for name, expected in record['files'].items():
        if digest(output/name) != expected:
            raise PermissionError(f'Sealed prior-control artifact changed: {name}')


def score(archive, plan, repo, output, obs_sha, original):
    verify_obs(output, obs_sha)
    endpoint_rows, paired_rows, cohorts = [], [], []
    with evaluation_labels(archive, repository=repo) as labels:
        for key, task in task_grid(plan):
            arrays = read_local_npz(output/f'predictions/{key}.npz')
            y = labels.load(f'blocks/{key}/manifest.json', 'holdout')
            logits = {name: arrays[f'logits_{name}'] for name in METHODS}
            decisions = {name: arrays[f'decision_{name}'] for name in METHODS}
            cohorts.append(dict(**task, ids=arrays['ids'], y=y, logits=logits, decisions=decisions))
            by_method = {}
            for name in METHODS:
                result = classification(logits[name], y, decisions['B0'], decisions=decisions[name])
                result['auroc'] = auc(margin(logits[name]), y)
                result['mean_p1'] = expit(margin(logits[name])).mean()
                endpoint_rows.append(dict(**task, method=name, **result))
                by_method[name] = result
            for left, right in plan['comparisons']:
                left_correct, right_correct = decisions[left] == y, decisions[right] == y
                corrected = int(np.sum(left_correct & ~right_correct))
                harmed = int(np.sum(~left_correct & right_correct))
                l, r = by_method[left], by_method[right]
                paired_rows.append(dict(**task, left=left, right=right, corrected=corrected, harmed=harmed,
                                        net_corrections=corrected-harmed, accuracy_pp=100*(corrected-harmed)/len(y),
                                        ba_pp=100*(l['balanced_accuracy']-r['balanced_accuracy']),
                                        auc_pp=100*(l['auroc']-r['auroc']), nll=l['sampled_nll']-r['sampled_nll']))
    verify_obs(output, obs_sha)
    intervals = paired_bootstrap(original, cohorts, phase='P', comparisons=plan['comparisons'], tau=plan['tau'])
    # Prior controls have no explanation banks. Retain the registered paired
    # classification/NLL resamples and record AUROC by seed as a separate diagnostic.
    intervals['metadata'].pop('directional_gap', None)
    intervals['metadata']['candidate_max_recomputed_each_replicate'] = False
    intervals['interval_rows'] = [row for row in intervals['interval_rows'] if row['metric'] != 'directional_risk_gap']
    for name, rows in (('endpoints', endpoint_rows), ('paired-seeds', paired_rows)):
        with (output/f'{name}.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    write_json(output/'paired-intervals.json', intervals)
    write_json(output/'FINAL.json', dict(stage='PRIOR_CONTROL_FINAL', observation_sha256=obs_sha,
        labels_read=['holdout'], endpoints=len(endpoint_rows), paired_seed_rows=len(paired_rows),
        files={name: digest(output/name) for name in ('OBS.json', 'endpoints.csv', 'paired-seeds.csv', 'paired-intervals.json')}))
