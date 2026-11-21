"""Freeze all eight directional predictions before using any direction's E labels."""
from fractions import Fraction
import json
from zipfile import ZipFile

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import residual_inputs as data, true_rate_replay
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_score
from . import path_closure as closure, screen_inputs


MAIN = 'TargetBg'
PAIRS = [(MAIN, arm) for arm in closure.METHODS if arm != MAIN] + [('Raw', 'WL')]
STRONG = ('WL', 'Intercept', 'SourceBg', 'Raw')


def members(roles):
    return ['executed-config.json', 'roles.json', 'fit-label-access.json', 'fits.json',
            *prior_run.inventory(roles, 'predictions')]


def freeze(out, archive):
    (out / 'predictions').mkdir()
    features = archive.graph_features()
    fits = {}
    with ZipFile(archive.package) as previous, data.ScopedLabels(archive.by_setting, archive.roles, 'fit') as port:
        for key, role in archive.roles.items():
            ids = role['roles']
            f_c, f_e, identity = archive.static_margins(key, features[role['source']], previous)
            with port.direction(key):
                y_d = port.load(key, 'D', ids['D'])
                y_c = port.load(key, 'C', ids['C'])
            model = closure.fit(key, archive.counts, ids, y_d, y_c, f_c)
            pred = closure.actions(archive.counts, np.asarray(ids['E'], dtype=np.int64), f_e, model)
            if any(not np.isfinite(z).all() for z in pred.values()):
                raise ArithmeticError(f'Nonfinite R19 prediction: {key}')
            fits[key] = identity | model
            prior_run.save_predictions(out, f'predictions/{key}.npz', pred | {'static_C_margin': f_c})
            print(f'{key} m={model["m"]} E={len(f_e)} with_paths={pred["has_paths"].sum()} '
                  f'old_margin_max_error={identity["replay_max_abs"]:.3g}', flush=True)
    data.validate_receipt(archive.roles, port.actual, 'fit')
    for name, value in (('roles', archive.roles), ('fit-label-access', port.actual), ('fits', fits)):
        write_json(out / f'{name}.json', value)
    return prior_run.seal(out, 'OBS', members(archive.roles), screen_inputs.parents(archive.cfg))


def verified_actions(out, archive, sha):
    raw = prior_run.verified_members(out, 'OBS', sha, members(archive.roles), screen_inputs.parents(archive.cfg))
    fit = json.loads(raw['fits.json'])
    if (json.loads(raw['roles.json']) != archive.roles or set(fit) != set(archive.roles)
            or json.loads(raw['executed-config.json'])['plan'] != archive.cfg):
        raise PermissionError('Sealed R19 roles or plan differ')
    data.validate_receipt(archive.roles, json.loads(raw['fit-label-access.json']), 'fit')
    result = {}
    with ZipFile(archive.package) as previous:
        for key, role in archive.roles.items():
            ids = np.asarray(role['roles']['E'], dtype=np.int64)
            model = fit[key]
            prediction = inputs.npz_member(raw[f'predictions/{key}.npz'])
            old = inputs.npz_member(previous.read(f'{key}.npz'))
            np.testing.assert_array_equal(old['e'], ids)
            f_e = prediction['margin_WL']
            np.testing.assert_allclose(f_e, old['Static'], rtol=0, atol=archive.cfg['absolute_replay_tolerance'])
            np.testing.assert_array_equal(f_e > 0, old['Static'] > 0)
            if model['references'] != closure.reference_ids(key, role['roles']):
                raise PermissionError(f'Sealed D/U reference sampling differs: {key}')
            f_c = prediction.pop('static_C_margin')
            if f_c.shape != (len(role['roles']['C']),) or not np.isfinite(f_c).all():
                raise PermissionError(f'Malformed C source margins: {key}')
            expected = closure.actions(archive.counts, ids, f_e, model)
            if set(expected) != set(prediction) or any(v.dtype != prediction[k].dtype or not np.array_equal(
                    v, prediction[k]) or not np.isfinite(v).all() for k, v in expected.items()):
                raise PermissionError(f'Sealed R19 decisions do not reproduce: {key}')
            result[key] = prediction
    return result


def gate(means, by_source):
    top = max(STRONG, key=means.__getitem__)
    margin = means[MAIN] - means[top]
    wins = sum(delta > 0 for delta in by_source.values())
    tests = dict(strong_gain=margin >= Fraction(1, 200), beats_mixture=means[MAIN] > means['Mixture'],
                 beats_count=means[MAIN] > means['Count'], source_groups=wins >= 3)
    return dict(advance=all(tests.values()), checks=tests, strongest_fixed=top,
                gain_over_strongest_pp=float(100*margin), gain_over_strongest_exact=str(margin),
                source_groups_positive=wins,
                target_minus_source_by_group_pp={str(k): float(100*v) for k, v in by_source.items()})


def score(out, scored, archive, obs_sha, execution):
    predictions = verified_actions(out, archive, obs_sha)
    endpoints, pairs = [], []
    by_key, pair_counts = {}, {}
    with data.ScopedLabels(archive.by_setting, archive.roles, 'replay_E') as port:
        for key, role in archive.roles.items():
            pred = predictions[key]
            with port.direction(key):
                y = port.load(key, 'E', role['roles']['E'])
            trace = {k: role[k] for k in data.IDENTITY} | dict(key=key, n=len(y))
            for method in closure.METHODS:
                tn, fp, fn, tp = roc_score.confusion(pred[method], y)
                acc, f1 = Fraction(tn + tp, len(y)), roc_score.f1((tn, fp, fn, tp))
                by_key[key, method] = (acc, f1)
                endpoints.append(trace | dict(
                    method=method, tn=tn, fp=fp, fn=fn, tp=tp, with_paths=int(pred['has_paths'].sum()),
                    accuracy=float(acc), macro_f1=float(f1)))
            for left, right in PAIRS:
                counts = true_rate_replay.paired_counts(pred[left], pred[right], y)
                pair_counts[key, left, right] = counts
                l, r = by_key[key, left], by_key[key, right]
                if Fraction(counts['net'], len(y)) != l[0]-r[0]:
                    raise ArithmeticError('Same-graph correction ledger disagrees with Accuracy')
                pairs.append(trace | dict(left=left, right=right, **counts,
                                          accuracy_pp=float(100*(l[0]-r[0])), macro_f1_pp=float(100*(l[1]-r[1]))))
    data.validate_receipt(archive.roles, port.actual, 'replay_E')

    def summarize(keys, name, source):
        metrics = {m: tuple(sum(by_key[k, m][j] for k in keys)/len(keys) for j in (0, 1)) for m in closure.METHODS}
        scope = dict(scope=name, source=source, directions=len(keys),
                     n=sum(len(archive.roles[k]['roles']['E']) for k in keys))
        summary = [scope | dict(method=m, accuracy=float(v[0]), macro_f1=float(v[1])) for m, v in metrics.items()]
        contrasts = []
        for left, right in PAIRS:
            l, r = metrics[left], metrics[right]
            counts = {c: sum(pair_counts[k, left, right][c] for k in keys)
                      for c in ('corrected', 'harmed', 'net', 'changed')}
            contrasts.append(scope | dict(left=left, right=right, **counts,
                                          accuracy_pp=float(100*(l[0]-r[0])), macro_f1_pp=float(100*(l[1]-r[1]))))
        return summary, contrasts, {m: v[0] for m, v in metrics.items()}

    summary, contrasts, means = summarize(list(archive.roles), 'all', None)
    by_source = {}
    for src in archive.cfg['domains']:
        for include, name in ((True, 'source'), (False, 'leave_one_source_out')):
            keys = [key for key, role in archive.roles.items() if (role['source'] == src) == include]
            rows, diff, group_means = summarize(keys, name, src)
            summary.extend(rows)
            contrasts.extend(diff)
            if include:
                by_source[src] = group_means[MAIN] - group_means['SourceBg']
    decision = gate(means, by_source)
    tables = dict(endpoints=endpoints, pairs=pairs, summary=summary, contrasts=contrasts)
    records = {'e-label-access': port.actual, 'score-execution': execution, 'decision': decision}
    for name, rows in tables.items():
        write_table(scored / f'{name}.csv', rows)
    for name, value in records.items():
        write_json(scored / f'{name}.json', value)
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return prior_run.seal(scored, 'FINAL', [*(f'{k}.csv' for k in tables), *(f'{k}.json' for k in records)],
                          {'obs_sha256': obs_sha})
