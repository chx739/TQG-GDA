"""Replay a background change on the frozen WL margin without fitting any parameters."""
import csv
from fractions import Fraction
import io
import json

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    feature_dropout_transfer as dropout, residual_inputs as data)
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer
from . import path_closure as closure, screen_inputs, source_screen


CONTINUOUS = ('WL', 'Delta', 'Intercept', 'MixtureDelta', 'Reverse', 'SourceBg', 'TargetBg', 'Raw')
ARMS = (*CONTINUOUS, 'Count')
BLIND = dict(Delta='main', WL='static', Count='count', Intercept='offset')


def supplement_ids(roles, r9, r19):
    pools = {name: set(ids) for name, ids in roles.items()}
    if len(pools) != 4 or len(set.union(*pools.values())) != sum(map(len, roles.values())):
        raise PermissionError('Original pools overlap or contain repeated IDs')
    if any(len(ids) != len(set(ids)) or not set(ids) <= pools['U'] for ids in (r9, r19)):
        raise PermissionError('U reference is repeated or outside its directional pool')
    return sorted(pools['U'] - set(r9) - set(r19))


def decisions(counts, ids, static, model):
    pred = closure.actions(counts, ids, static, model)
    bg = model['background']
    p_s = list(map(Fraction, model['pS']))
    opportunity = closure.per_node([counts[i] for i in ids])[1]
    reference_o = closure.per_node([counts[i] for i in model['references']['U']])[1]

    def slope(name):
        d = np.asarray([float(s - Fraction(v)) for s, v in zip(p_s, model[name], strict=True)])
        return closure.design(d[None, :], bg['rms'])[0] * bg['coef']

    target = slope('pU')
    delta = opportunity @ target
    intercept = float((reference_o @ target).mean())
    mixture = opportunity @ slope('pMix')
    margin = dict(WL=static, Delta=static + delta, Intercept=static + intercept,
                  MixtureDelta=static + mixture, Reverse=static - delta,
                  **{k: pred[f'margin_{k}'] for k in ('SourceBg', 'TargetBg', 'Raw')})
    flags = {name: z > 0 for name, z in margin.items()}
    flags['Count'] = dropout.count_matched(ids, static, int(flags['Delta'].sum())).astype(bool)
    return dict(ids=ids, delta=delta, offset=np.asarray(intercept), mixture_delta=mixture,
                **flags, **{f'margin_{name}': z for name, z in margin.items()})


class Frozen:
    def __init__(self, archive, here, cfg):
        self.archive, self.cfg = archive, cfg
        original = here / 'results/attempt-001-source-screen'
        seal = roc_transfer.stage(original, 'OBS', cfg['source_obs_sha256'])
        files = seal['files']
        if set(files) != set(source_screen.members(archive.roles)) or seal['predecessors'] != screen_inputs.parents(archive.cfg):
            raise PermissionError('Original R19 OBS members or input differ')
        saved = lambda name: json.loads(prior_run.read_member(original, files, f'{name}.json'))
        if saved('roles') != archive.roles or saved('executed-config')['plan'] != archive.cfg:
            raise PermissionError('Original R19 plan or roles differ')
        self.parents = dict(source_obs_sha256=cfg['source_obs_sha256'], static_obs_sha256=archive.cfg['static_obs_sha256'],
                            structure_sha256=archive.cfg['structure_sha256'])
        self.fits = saved('fits')
        self.roles, self.model_records = {}, {}
        for key, row in archive.roles.items():
            roles = row['roles']
            r9, r19 = archive.static_roles[key], self.fits[key]
            if r9['roles'] != roles or r19['references'] != closure.reference_ids(key, roles):
                raise PermissionError(f'Original direction or R19 reference differs: {key}')
            m = min(len(roles['C']), len(roles['U']))
            ref9 = dropout.sample(roles['U'], m, key)
            if r9['target_reference'] != ref9 or r9['supplement'] != sorted(set(roles['U']) - set(ref9)):
                raise PermissionError(f'Original R9 reference partition differs: {key}')
            ids = supplement_ids(roles, ref9, r19['references']['U'])
            if not ids:
                raise ValueError('No evaluation graphs after excluding both U reference pools')
            self.roles[key] = row | dict(supplement=ids, r9_reference=ref9, r19_reference=r19['references']['U'])
            self.model_records[key] = dict(r19_fits_sha256=files['fits.json'],
                r9_prediction_sha256=archive.static_members[f'predictions/{key}.npz'])
        all_ids = [gid for row in self.roles.values() for gid in row['supplement']]
        old_e = {gid for row in archive.roles.values() for gid in row['roles']['E']}
        if len(all_ids) != cfg['evaluation_count'] or len(set(all_ids)) != len(all_ids) or set(all_ids) & old_e:
            raise PermissionError('Evaluation cohort differs from the registered 1,961 graphs')

    def arms(self, key):
        role = self.roles[key]
        ids = np.asarray(role['supplement'], dtype=np.int64)
        raw = self.archive.static_member(f'predictions/{key}.npz')
        f = inputs.npz_member(raw)
        history = self.archive.static_roles[key]['supplement']
        static = f['margin_Static']
        if (not np.array_equal(f['ids'], history) or static.shape != f['ids'].shape or static.dtype != np.float64
                or not np.isfinite(static).all() or not np.array_equal(static > 0, f['Static'])):
            raise PermissionError(f'Frozen static margin or decision differs: {key}')
        loc = np.searchsorted(f['ids'], ids)
        np.testing.assert_array_equal(f['ids'][loc], ids)
        return decisions(self.archive.counts, ids, static[loc], self.fits[key])


def members(roles):
    return ['executed-config.json', 'roles.json', 'models.json', *prior_run.inventory(roles, 'predictions')]


def freeze(out, frozen):
    (out / 'predictions').mkdir()
    for key, row in frozen.roles.items():
        arrays = frozen.arms(key)
        if any(not np.isfinite(x).all() for x in arrays.values()):
            raise ArithmeticError(f'Invalid R19 delta actions: {key}')
        prior_run.save_predictions(out, f'predictions/{key}.npz', arrays)
        print(f'{key} supplement={len(row["supplement"])} offset={arrays["offset"]:.12g}', flush=True)
    for name, value in (('roles', frozen.roles), ('models', frozen.model_records)):
        write_json(out / f'{name}.json', value)
    return prior_run.seal(out, 'OBS', members(frozen.roles), frozen.parents)


def observed(out, frozen, obs_sha):
    obs = prior_run.verified_members(out, 'OBS', obs_sha, members(frozen.roles), frozen.parents)
    for name, expected in (('roles', frozen.roles), ('models', frozen.model_records)):
        if json.loads(obs[f'{name}.json']) != expected:
            raise PermissionError(f'R19 supplement {name} differ from their source')
    if json.loads(obs['executed-config.json'])['plan'] != frozen.cfg:
        raise PermissionError('R19 supplement plan differs')
    result = {}
    for key in frozen.roles:
        pred = inputs.npz_member(obs[f'predictions/{key}.npz'])
        expected = frozen.arms(key)
        if set(pred) != set(expected) or any(
                pred[k].dtype != v.dtype or not np.array_equal(pred[k], v) for k, v in expected.items()):
            raise PermissionError(f'R19 supplement predictions do not reproduce: {key}')
        result[key] = pred
    return result


def verify_blind(pred, directory, cfg, obs_sha):
    manifest_path = directory / 'manifest.json'
    if digest(manifest_path) != cfg['blind_manifest_sha256']:
        raise PermissionError('Independent pre-score manifest differs')
    manifest = json.loads(manifest_path.read_bytes())
    records = {}
    for name in ('unlabeled-decisions.csv', 'roles.json'):
        records[name] = prior_run.read_member(directory, manifest['files'], name)
    blind_roles = json.loads(records['roles.json'])
    grouped = {key: [] for key in pred}
    for row in csv.DictReader(io.StringIO(records['unlabeled-decisions.csv'].decode('ascii'))):
        grouped[row['key']].append(row)
    if set(blind_roles) != set(pred):
        raise PermissionError('Independent direction keys differ')
    totals = dict.fromkeys(cfg['registered_disagreements'], 0)
    absolute = dict.fromkeys(('static_z', 'delta', 'offset_value'), 0.)
    checked = 0
    for key, x in pred.items():
        rows = grouped[key]
        np.testing.assert_array_equal(x['ids'], blind_roles[key]['evaluation'])
        np.testing.assert_array_equal(x['ids'], [int(r['id']) for r in rows])
        checked += len(rows)
        for name, col in BLIND.items():
            np.testing.assert_array_equal(x[name], [int(r[col]) for r in rows])
            if name != 'Delta':
                totals[name] += int(np.count_nonzero(x[name] != x['Delta']))
        for name, col in (('margin_WL', 'static_z'), ('delta', 'delta'), ('offset', 'offset_value')):
            other = np.array([float(r[col]) for r in rows])
            np.testing.assert_allclose(x[name], other, atol=2e-12, rtol=2e-12)
            absolute[col] = max(absolute[col], float(np.max(np.abs(x[name] - other))))
    if checked != cfg['evaluation_count'] or totals != cfg['registered_disagreements']:
        raise PermissionError('Disagreement counts differ from the registered no-label check')
    return dict(obs_sha256=obs_sha, manifest_sha256=cfg['blind_manifest_sha256'],
                prediction_csv_sha256=manifest['files']['unlabeled-decisions.csv'],
                directions=len(pred), graphs=checked, exact_actions=checked*len(BLIND),
                disagreements=totals, max_abs_score_error=absolute)


def evaluation_roles(roles):
    # The existing exact-offset port names its authorized evaluation slot E.
    return {key: row | dict(roles={'E': row['supplement']}) for key, row in roles.items()}


def validate_score_receipt(roles, actual):
    data.validate_receipt(evaluation_roles(roles), actual, 'replay_E')
