"""Replay frozen WL coefficients with mean-preserving radius-specific scale alignment."""
from hashlib import sha256
import json
from zipfile import ZipFile

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    feature_dropout_transfer as dropout, reference_geometry, wl)
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import (
    inputs, prior_run, roc_transfer, size_exposure as curve)
from experiments.mutagenicity.r17_目标图外围与分类一致性_20261121.src import screen as paired_readout


R9 = 'r9_目标近邻与源标签传递_20261121'
R17 = 'r17_目标图外围与分类一致性_20261121'
R9_PLAN = 'configs/feature-dropout-transfer-check.json'
GEOMETRY = 'configs/reference-geometry.json'
R17_PLAN = 'configs/paired-readout.json'
R17_ARMS = ('WL_C1', 'WL_selected', 'Size', 'Base')
CURVE_ARMS = ('Curve', 'CurvePool')
MAIN = 'Target'
FLOOR = 1e-12


def methods(setting):
    return (MAIN, 'Static', 'Source', 'Common', 'Count', *R17_ARMS,
            *(CURVE_ARMS if setting == 'proteins_edge' else ()))


def d_reference(key, ids, m):
    if not 0 < m <= len(ids):
        raise ValueError('D must support the original U/C size')
    return sorted(ids, key=lambda i: (sha256(f'channel-v1|{key}|{i}'.encode('utf-8')).digest(), i))[:m]


def align(features, reference, donor, original):
    d, u = donor.std(axis=0, ddof=0), reference.std(axis=0, ddof=0)
    rho = np.divide(d, u, out=np.ones_like(d), where=(d > FLOOR) & (u > FLOOR))
    center = reference.mean(axis=0)
    return original + (features-center) @ (rho-1), dict(ratio=rho.tolist(), center=center.tolist(),
                                                      donor_sd=d.tolist(), reference_sd=u.tolist())


def scale_predictions(ids, static, d, u, c, e, intercept):
    target, t = align(e, u, d, static)
    source, s = align(e, c, d, static)
    total = lambda h: h.sum(axis=1, keepdims=True) + intercept
    common, one = align(static[:, None], total(u), total(d), static)
    margins = dict(Static=static, Target=target, Source=source, Common=common)
    hard = {name: z > 0 for name, z in margins.items()}
    hard['Count'] = dropout.count_matched(ids, static, int(hard['Target'].sum())).astype(bool)
    return (dict(ids=ids, channel_E=e, **hard, **{f'margin_{a}': z for a, z in margins.items()}),
            dict(target=t, source=s, common=one))


def load_npz(root, files, name):
    return inputs.npz_member(prior_run.read_member(root, files, name))


class Frozen:
    def __init__(self, repo, here, cfg):
        self.cfg = cfg
        r9, r17 = (here.parent / name for name in (R9, R17))
        r9_cfg = json.loads((r9 / R9_PLAN).read_bytes())
        geometry = json.loads((r9 / GEOMETRY).read_bytes())
        reference = reference_geometry.authenticate(r9, geometry)
        self.screen = dropout.HistoricalScreen(repo, r9, r9_cfg, geometry, reference)
        self.roles = self.screen.roles
        self.static = r9 / 'results/feature-dropout-transfer-001'
        self.strong = r17 / 'results/attempt-001-paired-readout'
        static_obs = roc_transfer.stage(self.static, 'OBS', cfg['r9_static_obs_sha256'])
        strong_obs = roc_transfer.stage(self.strong, 'OBS', cfg['r17_obs_sha256'])
        strong_cfg = json.loads((r17 / R17_PLAN).read_bytes())
        self.static_files, self.strong_files = static_obs['files'], strong_obs['files']
        if (static_obs['predecessors'] != dropout.parents(r9_cfg)
                or set(self.static_files) != set(dropout.members(self.roles))
                or set(self.strong_files) != set(paired_readout.members(reference.roles))
                or strong_obs['predecessors'] != paired_readout.parents(strong_cfg)
                or json.loads(prior_run.read_member(self.static, self.static_files, 'roles.json')) != self.roles
                or json.loads(prior_run.read_member(self.strong, self.strong_files, 'roles.json')) != reference.roles):
            raise PermissionError('R9 or R17 model and role provenance differs')
        self.historical = paired_readout.load_baselines(r17, strong_cfg, reference)
        self.fitted = json.loads(prior_run.read_member(self.static, self.static_files, 'fitted.json'))
        counts = {s: sum(len(r['roles']['E']) for r in self.roles.values() if r['setting']==s) for s in cfg['counts']}
        directions = {s: sum(r['setting']==s for r in self.roles.values()) for s in cfg['counts']}
        if counts != cfg['counts'] or set(directions.values()) != {8}:
            raise PermissionError('Unexpected R13 directional counts')
        self.parents = {k: cfg[k] for k in ('r9_static_obs_sha256', 'r17_obs_sha256')}
        self.parents.update(geometry_obs_sha256=r9_cfg['geometry_obs_sha256'])

    def strong_arms(self, key, row, ids):
        name = f'predictions/{key}.npz'
        x = load_npz(self.strong, self.strong_files, name)
        paired_readout.validate_arrays(x, ids.tolist())
        fixed, prior, _ = self.historical
        expected = dict(WL_C1=fixed[key][1]['wl_e'], **{a: prior[key][f'margin_{a}'] for a in R17_ARMS[1:]})
        if any(not np.array_equal(x[f'margin_{a}'], z) for a, z in expected.items()):
            raise PermissionError(f'R17 strong baselines do not reproduce archived R9 margins: {key}')
        margins = expected
        if row['setting'] == 'proteins_edge':
            model = self.fitted[key]['protein']
            historical = load_npz(self.static, self.static_files, name)
            np.testing.assert_array_equal(historical['ids'], row['supplement'])
            nodes = lambda ids: self.screen.nodes(row['setting'], ids)
            prob = float(curve.exposure_prior(nodes(row['target_reference']), model))
            old_curve = curve.predict(nodes(row['supplement']), model)-.5
            old_pool = np.full(len(old_curve), prob-.5)
            for a, v in zip(CURVE_ARMS, (old_curve, old_pool), strict=True):
                np.testing.assert_array_equal(historical[f'margin_{a}'], v)
                np.testing.assert_array_equal(historical[a], v > 0)
            margins |= dict(Curve=curve.predict(nodes(ids), model)-.5, CurvePool=np.full(len(ids), prob-.5))
        return dict(**{a: z > 0 for a, z in margins.items()}, **{f'margin_{a}': z for a, z in margins.items()})

    def prediction(self, key, old_package):
        row = self.roles[key]
        pool = row['roles']
        features = self.screen.features[row['setting'], row['source']]
        lex = wl.vocabulary(features, pool['D'])
        fit = self.fitted[key]
        model_file = f'models/{key}.npz'
        arrays = load_npz(self.static, self.static_files, model_file)
        w, intercepts = arrays['Static'], arrays['intercepts']
        old = self.screen.original_arrays(old_package, key)
        registration = self.screen.initial[key]
        if (dropout.lex_hash(lex) != fit['vocabulary_sha256'] or w.shape != (len(lex),)
                or w.dtype != np.float64 or intercepts.shape != (len(dropout.ORIGINAL),) or intercepts.dtype != np.float64
                or not np.isfinite(w).all() or not np.isfinite(intercepts).all()
                or fit['initial_member_sha256'] != registration['prediction_sha']
                or fit['params']['Static'] != registration['params']['Static']):
            raise PermissionError(f'Historical Static WL fit differs: {key}')
        b = float(intercepts[dropout.ORIGINAL.index('Static')])
        m = row['m']
        dr, ur, cr = d_reference(key, pool['D'], m), row['target_reference'], row['source_reference']
        order = dr + ur + cr + pool['E']
        x = wl.sparse_rows(features, order, lex)
        rad = np.asarray([q for q, _ in lex])
        contrib = np.stack([x[:, rad == j] @ w[rad == j] for j in range(3)], axis=1)
        d, u, c, e = np.split(contrib, [m, 2*m, 3*m])
        static = old['Static']
        replay = b + e.sum(axis=1)
        np.testing.assert_allclose(replay, static, rtol=0, atol=FLOOR)
        np.testing.assert_array_equal(replay > 0, static > 0)
        ids = old['e']
        arrays, cert = scale_predictions(ids, static, d, u, c, e, b)
        arrays |= self.strong_arms(key, row, ids)
        cert |= dict(m=m, D_reference=dr, U_reference=ur, C_reference=cr, intercept=b,
                     replay_error=float(np.max(np.abs(replay-static))),
                     static_model_sha256=self.static_files[model_file], historical_e_sha256=registration['prediction_sha'],
                     r17_e_sha256=self.strong_files[f'predictions/{key}.npz'],
                     r9_input_baselines=self.historical[2][key])
        return arrays, cert


def members(roles):
    return ['executed-config.json', 'roles.json', 'scales.json', *prior_run.inventory(roles, 'predictions')]


def freeze(out, frozen):
    (out / 'predictions').mkdir()
    certs = {}
    with ZipFile(frozen.screen.package) as package:
        for key, row in frozen.roles.items():
            arrays, cert = frozen.prediction(key, package)
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ArithmeticError(f'Nonfinite R13 WL-channel prediction: {key}')
            certs[key] = cert
            prior_run.save_predictions(out, f'predictions/{key}.npz', arrays)
            print(f'{key} m={cert["m"]} E={len(arrays["ids"])} original_error={cert["replay_error"]:.3g}', flush=True)
    write_json(out / 'roles.json', frozen.roles)
    write_json(out / 'scales.json', certs)
    return prior_run.seal(out, 'OBS', members(frozen.roles), frozen.parents)


def observed(out, frozen, sha):
    raw = prior_run.verified_members(out, 'OBS', sha, members(frozen.roles), frozen.parents)
    certs = json.loads(raw['scales.json'])
    if (json.loads(raw['roles.json']) != frozen.roles or set(certs) != set(frozen.roles)
            or json.loads(raw['executed-config.json'])['plan'] != frozen.cfg):
        raise PermissionError('R13 WL-channel plan and roster do not replay')
    preds = {}
    with ZipFile(frozen.screen.package) as package:
        for key in frozen.roles:
            cur = inputs.npz_member(raw[f'predictions/{key}.npz'])
            exp, cert = frozen.prediction(key, package)
            if set(cur) != set(exp) or cert != certs[key] or any(
                    v.dtype != cur[n].dtype or not np.array_equal(v, cur[n]) for n, v in exp.items()):
                raise PermissionError(f'R13 WL-channel prediction does not replay: {key}')
            preds[key] = cur
    return preds, certs


def verify_blind(preds, certs, roles, blind, cfg, obs_sha):
    paths = {k: blind / name for k, name in (
        ('blind_predictions_sha256', 'decisions.npz'), ('blind_references_sha256', 'directions.json'))}
    if any(digest(p) != cfg[k] for k, p in paths.items()):
        raise PermissionError('Independent R13 predictions or references differ')
    ref = {r['key']: r for r in json.loads(paths['blind_references_sha256'].read_bytes())}
    independent = inputs.npz_member(paths['blind_predictions_sha256'].read_bytes())
    if set(ref) != set(preds) or set(independent) != set(preds):
        raise PermissionError('Independent R13 directions differ')
    diffs = {s: dict.fromkeys(v, 0) for s, v in cfg['registered_disagreements'].items()}
    max_error = dict.fromkeys(('Static', 'Target', 'Source'), 0.)
    checks = 0
    for key, x in preds.items():
        y, model, recorded = independent[key], certs[key], ref[key]
        if (recorded['D_reference'] != model['D_reference'] or recorded['m'] != model['m']
                or recorded['n'] != len(x['ids'])):
            raise PermissionError(f'Independent D reference differs: {key}')
        np.testing.assert_array_equal(y[:, 0], x['ids'])
        np.testing.assert_allclose(recorded['ratio'], model['target']['ratio'], rtol=FLOOR, atol=FLOOR)
        np.testing.assert_allclose(recorded['source_ratio'], model['source']['ratio'], rtol=FLOOR, atol=FLOOR)
        for i, name in enumerate(('Static', 'Target', 'Source'), start=1):
            np.testing.assert_allclose(y[:, i], x[f'margin_{name}'], atol=FLOOR, rtol=FLOOR)
            np.testing.assert_array_equal(y[:, i] > 0, x[name])
            max_error[name] = max(max_error[name], float(np.max(np.abs(y[:, i]-x[f'margin_{name}']))))
        np.testing.assert_array_equal(y[:, 4], x['Count'])
        checks += len(x['ids'])
        for name in diffs[roles[key]['setting']]:
            diffs[roles[key]['setting']][name] += int(np.count_nonzero(x[MAIN] != x[name]))
    if diffs != cfg['registered_disagreements'] or checks != sum(cfg['counts'].values()):
        raise PermissionError('Independent R13 graph coverage or disagreement totals changed')
    return dict(obs_sha256=obs_sha, directions=len(ref), graphs=checks, exact_actions=4*checks,
                max_abs_margin_error=max_error, disagreement=diffs, **{k: cfg[k] for k in paths})
