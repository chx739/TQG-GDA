"""Sealed R11 prior controls, from source-only fitting to independent H scoring."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import prior_control, r5_inputs, seals
from . import inputs, prior_estimation, thresholds

ID_FIELDS = ('setting', 'source', 'target', 'seed')
UNLABELED = ('B0', 'SourceV', 'SourceCal', 'MLLS', 'Swap1', 'Swap2', 'MLLSRoot', 'SwapRoot1', 'SwapRoot2')
SUPERVISED = ('OwnA', 'PriorA', 'MajorityA')


def inventory(keys, prefix):
    return [f'{prefix}/{key}.npz' for key in keys]


def save_predictions(output, relative, arrays):
    with (output / relative).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def seal(output, stage, members, predecessors):
    value = dict(stage=stage, predecessors=predecessors, files={name: digest(output / name) for name in members})
    write_json(output / f'{stage}.json', value)
    return digest(output / f'{stage}.json')


def verified_members(output, stage, checksum, required, predecessors):
    payload = (output / f'{stage}.json').read_bytes()
    record = json.loads(payload)
    if (inputs.sha(payload) != checksum or record['stage'] != stage or record['predecessors'] != predecessors
            or set(record['files']) != set(required)):
        raise PermissionError(f'Mismatched {stage} stage, chain or members')
    return {name: read_member(output, record['files'], name) for name in required}


def read_member(root, checksums, name):
    payload = (root / name).read_bytes()
    if inputs.sha(payload) != checksums[name]:
        raise PermissionError(f'Archived member changed: {name}')
    return payload


def main_registry(registered, repo):
    root = repo / registered['main_attempt']
    payload = (root / 'OBS.json').read_bytes()
    obs = json.loads(payload)
    if inputs.sha(payload) != registered['main_obs_sha256'] or obs['stage'] != 'R11_OBS':
        raise PermissionError('R11 main OBS identity differs')
    members = obs['files']
    main_plan = json.loads(read_member(root, members, 'executed-config.json'))['plan']
    if obs['r7_final_digest'] != main_plan['r7_final_digest']:
        raise PermissionError('Main R11 refers to a different R7 archive')
    records = json.loads(read_member(root, members, 'fit-label-access.json'))
    roles = json.loads(read_member(root, members, 'roles.json'))
    groups = {(item['setting'], item['source'], item['seed']): item for item in records}
    expected = group_keys(main_plan)
    if (set(groups) != set(expected) or len(records) != len(groups)
            or set(roles) != {key for keys in expected.values() for key in keys}):
        raise ValueError('Main R11 is missing registered groups or blocks')
    return root, members, main_plan, groups, roles


def group_keys(plan):
    return {(s, src, seed): [inputs.key(s, src, t, seed) for t in targets]
            for s, src, seed, targets in inputs.registered(plan)}


def verify_group(setting, validation, group, receipt, roles):
    if receipt['validation_ids'] != validation['ids'] or receipt['source_checkpoint'] != validation['lineage']:
        raise PermissionError('Source validation or checkpoint differ from main R11')
    a_ids = receipt['selected_adaptation_ids']
    n = min(len(item['ids']['adapt']) for item in group.values())
    if set(a_ids) != {str(t) for t in group}:
        raise PermissionError('Main R11 A target set differs')
    for target, item in group.items():
        task = roles[item['key']]
        if (any(task[k] != item['task'][k] for k in ID_FIELDS)
                or task['manifest_sha256'] != item['manifest_sha256']
                or task['holdout_ids'] != item['ids']['holdout']
                or task['other_targets'] != sorted(t for t in group if t != target)
                or a_ids[str(target)] != thresholds.subset_ids(setting, item['ids']['adapt'], n)):
            raise PermissionError('Historical A/H roles or selected A budget differ from main R11')
    return a_ids


def fit_scores(validation, group, selected, options):
    """No target label reader is available to this phase."""
    v, y = validation['margin'], validation['v_port'].load()
    threshold = thresholds.choose(v, y, 0.)['threshold']
    try:
        fitted = prior_control.fit_calibrator(v, y, options['calibration'])
    except ValueError as error:
        return threshold, dict(calibration_error=str(error), source_prior=None, targets={})
    v_scores = prior_control.calibrate(v, fitted)
    p_s = float(expit(v_scores).mean())
    constant = float(v_scores[0]) if np.all(v_scores == v_scores[0]) else None
    fits = {}
    for target, item in group.items():
        rows = {gid: row for row, gid in enumerate(item['ids']['adapt'])}
        margins = item['margins']['adapt'][[rows[gid] for gid in selected[str(target)]]]
        scores = prior_control.calibrate(margins, fitted)
        fits[str(target)] = prior_estimation.estimate(scores, p_s, options['mlls'],
                                                      constant_validation_score=constant)
    return threshold, dict(calibration=fitted, source_prior=p_s, targets=fits)


def predict(item, threshold, fit, others, atol):
    margin = item['margins']['holdout']
    actions = dict(ids=np.asarray(item['ids']['holdout']), margin=margin,
                   B0=thresholds.decisions(margin, 0.), SourceV=thresholds.decisions(margin, threshold))
    if fit['source_prior'] is None:
        return actions
    scores = prior_control.calibrate(margin, fit['calibration'])
    prior_control.verify_rank_preservation(margin, scores, atol)
    actions.update(calibrated=scores, SourceCal=scores > 0)
    targets = [item['task']['target'], *others]
    for solver, arms in (('em', ('MLLS', 'Swap1', 'Swap2')), ('root', ('MLLSRoot', 'SwapRoot1', 'SwapRoot2'))):
        for target, arm in zip(targets, arms, strict=True):
            result = fit['targets'][str(target)][solver]
            if 'target_prior' in result:
                actions[arm] = prior_estimation.decisions(scores, result['target_prior'], fit['source_prior'])
    return actions


def freeze(output, repo, registered, options):
    _, _, plan, main_groups, main_roles = main_registry(registered, repo)
    archive = Path(plan['archive'])
    roots = dict(main_obs_sha256=registered['main_obs_sha256'], r7_final_digest=plan['r7_final_digest'])
    fits, roles, input_pools = {}, {}, {}
    (output / 'unlabeled').mkdir()
    with seals.verified_seal_chain(archive, 'R7_FINAL') as scope:
        final = json.loads((archive / 'seals/R7_FINAL.json').read_bytes())
        if final['digest'] != plan['r7_final_digest']:
            raise PermissionError('R11 historical FINAL differs')
        original = json.loads(scope.read_bytes('executed-plan.json', 'R7_INPUT'))
        for setting, source, seed, targets in inputs.registered(plan):
            validation, group = inputs.prepare_group(scope, repo, plan, setting, source, seed, targets)
            receipt = main_groups[setting, source, seed]
            selected = verify_group(setting, validation, group, receipt, main_roles)
            threshold, fit = fit_scores(validation, group, selected, options)
            for target, item in group.items():
                key = item['key']
                cohort = input_pools.setdefault(item['dataset'], {'adapt': set(), 'holdout': set()})
                for role, pool in cohort.items():
                    pool.update(item['ids'][role])
                task = main_roles[key]
                roles[key] = dict(**task, selected_a_ids=selected[str(target)])
                fits[key] = dict(**item['task'], source_prior=fit['source_prior'],
                                 calibration=fit.get('calibration'), calibration_error=fit.get('calibration_error'),
                                 **fit['targets'].get(str(target), {}))
                rows = predict(item, threshold, fit, task['other_targets'], plan['replay_atol'])
                save_predictions(output, f'unlabeled/{key}.npz', rows)
    if any(pool['adapt'] & pool['holdout'] for pool in input_pools.values()):
        raise PermissionError('A and H are not globally disjoint within a dataset')
    write_json(output / 'roles.json', roles)
    write_json(output / 'fits.json', fits)
    write_json(output / 'r7-executed-plan.json', original)
    checksum = seal(output, 'UNLABELED', unlabeled_members(roles), roots)
    return checksum, plan, roots


def unlabeled_members(keys):
    return ['executed-config.json', 'roles.json', 'fits.json', 'r7-executed-plan.json', *inventory(keys, 'unlabeled')]


def registered_keys(plan):
    return [key for keys in group_keys(plan).values() for key in keys]


def verified_predictions(arrays, required_ids, methods, required):
    np.testing.assert_array_equal(arrays['ids'], required_ids)
    if not set(required) <= set(arrays):
        raise ValueError('Missing baseline predictions from sealed block')
    for method in methods:
        if method in arrays and (arrays[method].dtype != np.bool_ or arrays[method].shape != (len(required_ids),)):
            raise ValueError(f'Unexpected decision representation for {method}')


def label_manifest(scope, key, task, role):
    blob = scope.read_bytes(f'blocks/{key}/manifest.json', 'R7_INPUT')
    manifest = json.loads(blob)
    if inputs.sha(blob) != task['manifest_sha256'] or any(manifest[k] != task[k] for k in ID_FIELDS):
        raise PermissionError('Sealed label request refers to a different task')
    expected = task['holdout_ids'] if role == 'holdout' else task['selected_a_ids']
    ids = manifest['ids'][role]
    if (role == 'holdout' and expected != ids) or not set(expected) <= set(ids) or len(set(expected)) != len(expected):
        raise PermissionError('Sealed label request differs from the registered graph role')
    return manifest


def label_port(manifest, task, role, repo):
    identity = {k: task[k] for k in ID_FIELDS}
    context = r5_inputs._context(manifest['r7_plan'], **identity, repository=repo)
    return inputs.raw_grant(context, manifest, role)


def supervise(output, repo, registered, options, plan, roots, unlabeled_sha):
    preliminary = verified_members(output, 'UNLABELED', unlabeled_sha,
                                   unlabeled_members(registered_keys(plan)), roots)
    main_root, main_members, _, _, main_roles = main_registry(registered, repo)
    roles = json.loads(preliminary['roles.json'])
    fits = json.loads(preliminary['fits.json'])
    if set(roles) != set(main_roles) or set(fits) != set(main_roles):
        raise PermissionError('Unlabeled inventory no longer matches the registered grid')
    archive = Path(plan['archive'])
    (output / 'supervised').mkdir()
    prior_a, receipts, label_cache = {}, {}, {}
    with seals.verified_seal_chain(archive, 'R7_FINAL') as scope, inputs.GraphLabelSession('adapt') as labels:
        for key, task in roles.items():
            x = inputs.npz_member(preliminary[f'unlabeled/{key}.npz'])
            archived = inputs.npz_member(read_member(main_root, main_members, f'predictions/{key}.npz'))
            verified_predictions(x, task['holdout_ids'], UNLABELED, ('B0', 'SourceV'))
            verified_predictions(archived, task['holdout_ids'], ('B0', 'SourceV', 'OwnA'), ('B0', 'SourceV', 'OwnA'))
            for field in ('margin', 'B0', 'SourceV'):
                np.testing.assert_array_equal(x[field], archived[field])
            chosen = task['selected_a_ids']
            manifest = label_manifest(scope, key, task, 'adapt')
            if chosen != thresholds.subset_ids(task['setting'], manifest['ids']['adapt'], len(chosen)):
                raise PermissionError('Selected A order changed')
            cache_key = (task['setting'], task['target'], tuple(chosen))
            if cache_key not in label_cache:
                y = labels.load(label_port(manifest, task, 'adapt', repo), chosen)
                if y.shape != (len(chosen),) or not np.isin(y, [0, 1]).all():
                    raise ValueError('Expected binary labels for the authenticated A subset')
                label_cache[cache_key] = int(y.sum())
                receipts[key] = chosen
            count = label_cache[cache_key]
            n = len(chosen)
            q = count / n
            floor = options['mlls']['prior_floor']
            clipped = float(np.clip(q, floor, 1 - floor))
            constant = prior_estimation.majority(count, n)
            prior_a[key] = dict(positive=count, n=n, sample_prior=q, clipped_prior=clipped, majority=int(constant))
            rows = dict(ids=x['ids'], OwnA=archived['OwnA'], MajorityA=np.full(len(x['ids']), constant, dtype=bool))
            if 'calibrated' in x:
                rows['PriorA'] = prior_estimation.decisions(x['calibrated'], clipped, fits[key]['source_prior'])
            save_predictions(output, f'supervised/{key}.npz', rows)
    write_json(output / 'a-counts.json', prior_a)
    write_json(output / 'a-label-access.json', receipts)
    return seal(output, 'OBS', supervised_members(roles), {'unlabeled_sha256': unlabeled_sha})


def supervised_members(keys):
    return ['UNLABELED.json', 'a-counts.json', 'a-label-access.json', *inventory(keys, 'supervised')]
