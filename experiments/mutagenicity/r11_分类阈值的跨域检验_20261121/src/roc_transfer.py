"""Bind historical Size and B0 inputs, freeze deployment, then score A-count diagnostics."""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from . import inputs, prior_estimation, prior_run, roc_rate, thresholds


STRONG = ('B0', 'SourceV', 'SourceCal', 'WL', 'Size')
HISTORICAL = (*STRONG, 'MLLS')
ROC_ARMS = ('RateROC', 'NativeV', 'SourcePrior', 'OtherPrior1', 'OtherPrior2', 'OtherQuantile1', 'OtherQuantile2')
UNLABELED = (*HISTORICAL, *ROC_ARMS, 'DirectP', 'SamePriorCal', 'PoolMajority')
SUPERVISED = ('KnownRateROC', 'KnownNativeV', 'PriorA', 'MajorityA', 'OwnA')
MEAN_ARMS = {'OtherPriorMean': ('OtherPrior1', 'OtherPrior2'),
             'OtherQuantileMean': ('OtherQuantile1', 'OtherQuantile2')}


def stage(root, name, sha):
    payload = (root / f'{name}.json').read_bytes()
    rec = json.loads(payload)
    if inputs.sha(payload) != sha or rec['stage'] != name:
        raise PermissionError(f'Unexpected historical stage: {root}/{name}')
    return rec


def historical_inputs(repo, here, cfg):
    original_config = json.loads((here / cfg['base_config']).read_bytes())
    _, _, full, groups, roles = prior_run.main_registry(original_config, repo)
    if full['seeds'] != cfg['pilot_seeds'] + cfg['reserve_seeds']:
        raise PermissionError('Pilot and reserve seeds differ from the original five seed grid')
    plan = dict(full, seeds=cfg['pilot_seeds'])
    prior = here / cfg['prior_attempt']
    unlabeled = stage(prior, 'UNLABELED', cfg['prior_unlabeled_sha256'])
    root = Path(cfg['structural_attempt'])
    fit = stage(root, 'FIT', cfg['structural_fit_sha256'])
    obs = stage(root, 'OBS', cfg['structural_obs_sha256'])
    r7 = plan['r7_final_digest']
    if (unlabeled['predecessors'] != dict(main_obs_sha256=original_config['main_obs_sha256'], r7_final_digest=r7)
            or fit['predecessors'] != dict(r7_final_digest=r7)
            or obs['predecessors'] != dict(fit_sha256=cfg['structural_fit_sha256'])):
        raise PermissionError('Historical R10 and R11 inputs do not share the registered source')
    return plan, groups, roles, prior, unlabeled['files'], root, fit['files'], obs['files']


def model_for(setting, source, seed, root, fit, fit_access, validation):
    group = structural_observation.group_key(setting, source, seed)
    path = f'fits/{group}.json'
    fitted = json.loads(prior_run.read_member(root, fit, path))
    access = fit_access[group]
    if (tuple(fitted[k] for k in structural.GROUP) != (setting, source, seed)
            or access['validation_ids'] != validation['ids']
            or access['v_label_sha256'] != inputs.sha(validation['v_port'].payload)):
        raise PermissionError('Frozen Size model and the source V cohort differ')
    return fitted['size'], fit[path], access


def verify_references(u, s, item, archived_role, s_role, fit_sha, selected):
    ids = item['ids']['holdout']
    identity = item['task']
    if (any(archived_role[k] != v or s_role[k] != v for k, v in identity.items())
            or any(role['manifest_sha256'] != item['manifest_sha256'] or role['holdout_ids'] != ids
                   for role in (archived_role, s_role))
            or s_role['fit_sha256'] != fit_sha or archived_role['selected_a_ids'] != selected):
        raise PermissionError('Frozen R10/R11 arms are not bound to the matched cohort and Size model')
    prior_run.verified_predictions(u, ids, ('B0', 'SourceV', 'SourceCal', 'MLLS'), ('B0', 'SourceV', 'SourceCal', 'MLLS'))
    structural_observation.verify_predictions(s, s_role)
    np.testing.assert_array_equal(u['margin'], item['margins']['holdout'])
    np.testing.assert_array_equal(u['B0'], u['margin'] > 0.)
    np.testing.assert_array_equal(u['B0'], s['action_B0'])
    np.testing.assert_array_equal(u['SourceCal'], s['action_SourceCal'])
    np.testing.assert_allclose(u['margin'], inputs.margin(s['logits_B0']), rtol=0., atol=1e-12)
    return {k: u[k] for k in ('B0', 'SourceV', 'SourceCal', 'MLLS')} | {
        'WL': s['action_WL_selected'].astype(bool), 'Size': s['action_Size'].astype(bool)}


def decisions(roc, p, source_prior, a, others, h, calibrated, calibrated_prior, floor):
    a_cut, npos = roc_rate.quantile(a, p)
    result = dict(DirectP=roc_rate.classify(h, a_cut), PoolMajority=np.full(len(h), p > Fraction(1, 2), dtype=bool),
                  SamePriorCal=prior_estimation.decisions(calibrated, float(np.clip(float(p), floor, 1-floor)),
                                                          calibrated_prior))
    diagnostics = dict(p=str(p), direct_p_cut=thresholds.encode_threshold(a_cut), direct_p_a_positive=npos)
    if roc is None:
        result.update({arm: h > 0. for arm in ROC_ARMS})
        diagnostics['status'] = 'source_V_missing_class'
        return result, diagnostics

    diagnostics['status'] = 'ok'
    result['RateROC'], diagnostics['RateROC'] = roc_rate.transfer(roc, p, a, h)
    result['NativeV'] = roc_rate.classify(h, diagnostics['RateROC']['selected']['cut'])
    result['SourcePrior'], diagnostics['SourcePrior'] = roc_rate.transfer(roc, source_prior, a, h)
    q = Fraction(diagnostics['RateROC']['q'])
    for index, (other_target, other_p, other_a) in enumerate(others, 1):
        prior_arm, quantile_arm = f'OtherPrior{index}', f'OtherQuantile{index}'
        result[prior_arm], diagnostics[prior_arm] = roc_rate.transfer(roc, other_p, a, h)
        cut, count = roc_rate.quantile(other_a, q)
        result[quantile_arm] = roc_rate.classify(h, cut)
        diagnostics[quantile_arm] = dict(a_cut=thresholds.encode_threshold(cut), a_positive=count, q=str(q))
        diagnostics[prior_arm]['target'] = diagnostics[quantile_arm]['target'] = other_target
    return result, diagnostics


def unlabeled_members(keys):
    return ['executed-config.json', 'roles.json', 'workpoints.json', 'deploy-details.json',
            'historical-plan.json', *prior_run.inventory(keys, 'deployment')]


def labeled_members(keys):
    return ['UNLABELED.json', 'a-audit.json', *prior_run.inventory(keys, 'a-diagnostic')]


def freeze(output, repo, cfg, bound, parents):
    plan, main_groups, main_roles, prior, pf, structural_root, sf, so = bound
    prior_roles = json.loads(prior_run.read_member(prior, pf, 'roles.json'))
    priors = json.loads(prior_run.read_member(prior, pf, 'fits.json'))
    floor = json.loads(prior_run.read_member(prior, pf, 'executed-config.json'))['r7_options']['mlls']['prior_floor']
    structural_roles = json.loads(prior_run.read_member(structural_root, so, 'roles.json'))
    fit_access = json.loads(prior_run.read_member(structural_root, sf, 'fit-access.json'))
    (output / 'deployment').mkdir()
    roles, details, workpoints, stores = {}, {}, {}, {}
    archive = Path(plan['archive'])
    access_archive = historical.CArchive(plan | dict(r7_archive=plan['archive'], phase1_seed=cfg['pilot_seeds'][0],
                                                   reserve_seeds=cfg['reserve_seeds']))
    with seals.verified_seal_chain(archive, 'R7_OBS') as scope:
        if json.loads((archive / 'seals/R7_FINAL.json').read_bytes())['digest'] != plan['r7_final_digest']:
            raise PermissionError('Historical C does not match R11')
        original = json.loads(scope.read_bytes('executed-plan.json', 'R7_INPUT'))
        for setting, source, seed, targets in inputs.registered(plan):
            validation, items = inputs.prepare_group(scope, repo, plan, setting, source, seed, targets)
            main_group = main_groups[setting, source, seed]
            chosen = prior_run.verify_group(setting, validation, items, main_group, main_roles)
            size, size_sha, size_access = model_for(setting, source, seed, structural_root, sf, fit_access, validation)
            opt = structural.select_model(size)
            y = validation['v_port'].load()
            roc = roc_rate.source_roc(validation['margin'], y)
            p_s = Fraction(int(y.sum()), len(y))
            name = structural_observation.group_key(setting, source, seed)
            workpoints[name] = dict(roc=roc, source_prior=str(p_s), v_ids=validation['ids'],
                                    source_checkpoint=validation['lineage'], size_fit_sha256=size_sha,
                                    size_C=size['selected_C'])
            p_by_target, a_by_target, size_by_target = {}, {}, {}
            for target, item in items.items():
                a_ids = chosen[str(target)]
                task = item['task']
                manifest, checksum, ctx = historical.context(access_archive, task, repo)
                if (manifest['ids'] != item['ids'] or checksum != item['manifest_sha256']
                        or size_access['source_ids'] != item['ids']['source']
                        or size_access['class_mapping'] != ctx['label_meta']['mapping']):
                    raise PermissionError('R10 Size S and R11 A roles differ')
                if setting not in stores:
                    stores[setting] = historical.GraphStore(ctx)
                store = stores[setting]
                if store.graph_sha != size_access['graph_container_sha256'] or not set(a_ids) <= set(manifest['ids']['adapt']):
                    raise PermissionError('Invalid Size graph container or adaptation read selection')
                raw = np.asarray([structural.size(graph) for graph in store.take(a_ids)])
                margin = composition.scores((raw-size['mean'])/size['scale'], opt)
                if not np.isfinite(margin).all():
                    raise ValueError('Nonfinite Size predictions on the frozen A')
                index = {gid: i for i, gid in enumerate(item['ids']['adapt'])}
                a_by_target[target] = item['margins']['adapt'][[index[gid] for gid in a_ids]]
                size_by_target[target] = margin
                p_by_target[target] = Fraction.from_float(float(expit(margin).mean()))

            for target, item in items.items():
                key = item['key']
                upath, spath = f'unlabeled/{key}.npz', f'predictions/{key}.npz'
                u = inputs.npz_member(prior_run.read_member(prior, pf, upath))
                s = inputs.npz_member(prior_run.read_member(structural_root, so, spath))
                a_ids = chosen[str(target)]
                old = verify_references(u, s, item, prior_roles[key], structural_roles[key], size_sha, a_ids)
                h = item['margins']['holdout']
                a = a_by_target[target]
                foreign = [(t, p_by_target[t], a_by_target[t]) for t in prior_roles[key]['other_targets']]
                new, detail = decisions(roc, p_by_target[target], p_s, a, foreign, h, u['calibrated'],
                                        priors[key]['source_prior'], floor)
                details[key] = detail
                roles[key] = dict(**prior_roles[key], group=name, manifest_sha256_checked=item['manifest_sha256'],
                                  r11_unlabeled_member_sha256=pf[upath], r10_h_member_sha256=so[spath],
                                  size_fit_sha256=size_sha, a_graph_container_sha256=store.graph_sha)
                prior_run.save_predictions(output, f'deployment/{key}.npz', dict(ids=u['ids'], a_margin=a, h_margin=h,
                                           a_size_margin=size_by_target[target], **old, **new))
            print(f'R11 ROC deployment {name}', flush=True)
    keys = prior_run.registered_keys(plan)
    if set(roles) != set(keys):
        raise PermissionError('Incomplete deployment prediction grid')
    for name, value in (('roles', roles), ('workpoints', workpoints), ('deploy-details', details), ('historical-plan', original)):
        write_json(output / f'{name}.json', value)
    return prior_run.seal(output, 'UNLABELED', unlabeled_members(keys), parents)


def verify_deployment(output, plan, parents, sha):
    keys = prior_run.registered_keys(plan)
    frozen = prior_run.verified_members(output, 'UNLABELED', sha, unlabeled_members(keys), parents)
    roles = json.loads(frozen['roles.json'])
    if set(roles) != set(keys):
        raise PermissionError('Incomplete deployment role grid')
    rows = {}
    for key in keys:
        row = inputs.npz_member(frozen[f'deployment/{key}.npz'])
        role = roles[key]
        if set(row) != {'ids', 'a_margin', 'a_size_margin', 'h_margin', *UNLABELED}:
            raise PermissionError(f'Wrong deployment member set: {key}')
        prior_run.verified_predictions(row, role['holdout_ids'], UNLABELED, UNLABELED)
        for field, ids in (('h_margin', role['holdout_ids']), ('a_margin', role['selected_a_ids']),
                           ('a_size_margin', role['selected_a_ids'])):
            values = row[field]
            if values.shape != (len(ids),) or values.dtype != np.float64 or not np.isfinite(values).all():
                raise PermissionError(f'Wrong deployment score role: {key}/{field}')
        np.testing.assert_array_equal(row['B0'], row['h_margin'] > 0.)
        rows[key] = row
    return frozen, roles, rows


def diagnose(output, cfg, bound, parents, unlabeled_sha):
    plan, _, _, prior, *_ = bound
    frozen, roles, rows = verify_deployment(output, plan, parents, unlabeled_sha)
    # Open the already published aggregate A counts only after validating all
    # new label-free predictions. No A label reader is created here.
    old = stage(prior, 'OBS', cfg['prior_obs_sha256'])
    if old['predecessors'] != dict(unlabeled_sha256=cfg['prior_unlabeled_sha256']):
        raise PermissionError('Published A counts are not from the matching R11 arm')
    a_counts = json.loads(prior_run.read_member(prior, old['files'], 'a-counts.json'))
    v = json.loads(frozen['workpoints.json'])
    details = json.loads(frozen['deploy-details.json'])
    audit = {}
    (output / 'a-diagnostic').mkdir()
    for key, role in roles.items():
        x = rows[key]
        record = a_counts[key]
        pos, n = record['positive'], record['n']
        if type(pos) is not int or type(n) is not int or n != len(role['selected_a_ids']) or not 0 <= pos <= n:
            raise PermissionError(f'Published A count differs from the selected A budget: {key}')
        p = Fraction(pos, n)
        np.testing.assert_equal(float(p), record['sample_prior'])
        archived = inputs.npz_member(prior_run.read_member(prior, old['files'], f'supervised/{key}.npz'))
        prior_run.verified_predictions(archived, role['holdout_ids'], prior_run.SUPERVISED, prior_run.SUPERVISED)
        np.testing.assert_array_equal(archived['MajorityA'], np.full(len(x['ids']), p > Fraction(1, 2), dtype=bool))
        roc = v[role['group']]['roc']
        point = dict(status='source_V_missing_class')
        action = native = x['B0']
        if roc is not None:
            action, point = roc_rate.transfer(roc, p, x['a_margin'], x['h_margin'])
            native = roc_rate.classify(x['h_margin'], point['selected']['cut'])
        audit[key] = dict(positive=pos, n=n, observed_prior=str(p), source_prior=v[role['group']]['source_prior'],
                          size_prior=details[key]['p'], known=point)
        prior_run.save_predictions(output, f'a-diagnostic/{key}.npz', dict(
            KnownRateROC=action, KnownNativeV=native, **archived))
    write_json(output / 'a-audit.json', audit)
    return prior_run.seal(output, 'OBS', labeled_members(roles),
                          dict(unlabeled_sha256=unlabeled_sha, prior_obs_sha256=cfg['prior_obs_sha256']))
