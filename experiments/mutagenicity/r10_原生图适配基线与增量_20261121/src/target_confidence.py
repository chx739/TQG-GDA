"""ATC model selection on frozen V/A predictions with exact tie credit."""
from fractions import Fraction
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import (
    conditional_votes as historical_votes, reverse_vote_observation as reverse, vote_mixture)
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from . import structural


MODELS = ('B0', 'WL', 'Size')
PRIORITY = ('WL', 'Size', 'B0')
CHOOSERS = ('SourceV', 'OwnATC', 'OtherATC1', 'OtherATC2')
BITS = {name: [(i >> b) & 1 for i in range(8)] for name, b in zip(MODELS, (1, 0, 2), strict=True)}
ID = structural.IDENTITY
R12 = 'r12_三种图读数与阈值识别_20261121'
R11 = 'r11_分类阈值的跨域检验_20261121'


def confidence(margins):
    x = np.abs(np.asarray(margins, dtype=np.float64))
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Confidence needs a nonempty finite margin vector')
    return x


def estimate(margins, fit):
    x = confidence(margins)
    threshold = fit['threshold']
    if threshold is None:
        return Fraction(fit['k'], fit['n'])
    return (int(np.count_nonzero(x > threshold)) +
            Fraction(fit['tie_credit']) * int(np.count_nonzero(x == threshold))) / len(x)


def fit_threshold(margins, correct):
    x = confidence(margins)
    n = len(x)
    if not 0 <= correct <= n:
        raise ValueError('V correct count outside 0..n')
    threshold, tie = None, Fraction(0)
    if 0 < correct < n:
        threshold = float(np.sort(x)[n-correct])
        higher = np.count_nonzero(x > threshold)
        tied = np.count_nonzero(x == threshold)
        tie = Fraction(correct - int(higher), int(tied))
    record = dict(k=correct, n=n, threshold=threshold, tie_credit=str(tie))
    if estimate(margins, record) != Fraction(correct, n):
        raise ArithmeticError('ATC does not reproduce the source Accuracy')
    return record


def select(estimates, fit):
    return max(PRIORITY, key=lambda m: (estimates[m], Fraction(fit[m]['k'], fit[m]['n'])))


def encode(matrix):
    return vote_mixture.encode(*(matrix.T > 0))


def v_fit(margins, counts):
    x = np.asarray(margins)
    c = np.asarray(counts)
    if c.shape != (2, 2, 4) or c.dtype.kind not in 'iu' or (c < 0).any() or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError('V margins and joint two-class counts are malformed')
    matrix = c.reshape(2, 8)
    np.testing.assert_array_equal(np.bincount(encode(x), minlength=8), matrix.sum(axis=0))
    result = {}
    for i, method in enumerate(MODELS):
        bit = np.asarray(BITS[method])
        correct = int(matrix[0] @ (1-bit) + matrix[1] @ bit)
        result[method] = fit_threshold(x[:, i], correct)
    return result


class Archive:
    def __init__(self, repo, here, cfg, reference, original):
        self.repo, self.here, self.cfg, self.reference, self.original = repo, here, cfg, reference, original
        self.r12 = here.parent / R12 / reference['historical_attempt']
        self.blobs, self.roles, self.sources = reverse.historical_obs(self.r12, reference, original)
        self.history = historical_votes.Archive(repo, original)
        self.parents = dict(r12_obs_sha256=reference['historical_obs_sha256'],
                            r10_fit_sha256=original['r10_fit_sha256'], r10_obs_sha256=original['r10_obs_sha256'],
                            r11_unlabeled_sha256=original['r11_unlabeled_sha256'], r16_fit_sha256=original['r16_fit_sha256'])

    def block(self, key, v_rows):
        role = self.roles[key]
        task = {k: role[k] for k in ID}
        group = role['group']
        name = f'fits/{group}.json'
        receipt = self.sources[group]
        m, identity, ctx = historical.context(self.history.c, task, self.repo)
        if (identity != role['manifest_sha256'] or m['ids'] != role['ids']
                or m['ids']['validation'] != receipt['v_ids'] or m['ids']['holdout'] != role['holdout_ids']):
            raise PermissionError(f'Historical R12 task differs: {key}')
        x, _, _ = reverse.historical_votes(self.blobs[f'predictions/{key}.npz'], role)
        raw_a = inputs.npz_member(self.history.member('u', f'deployment/{key}.npz'))
        r11_role = self.history.roles['u'][key]
        if (r11_role['selected_a_ids'] != role['selected_a_ids'] or r11_role['holdout_ids'] != role['holdout_ids']
                or r11_role['manifest_sha256'] != identity
                or role['evidence']['r11_sha256'] != self.history.files['u'][f'deployment/{key}.npz']):
            raise PermissionError(f'Historical R11 A/H role differs: {key}')
        np.testing.assert_array_equal(raw_a['ids'], x['ids'])
        access = self.history.fit_access[group]
        model = json.loads(self.history.member('sf', name))
        if (receipt['structure_sha256'] != self.history.files['sf'][name]
                or receipt['v_ids'] != access['validation_ids']
                or receipt['v_label_sha256'] != access['v_label_sha256']
                or access['class_mapping'] != ctx['label_meta']['mapping']
                or any(model[k] != role[k] for k in structural.GROUP)):
            raise PermissionError(f'Original WL/Size source certificate differs: {key}')

        def structural_scores(ids, allowed):
            points = self.history.features_for(role['setting'], ctx, ids, allowed, access['graph_container_sha256'])
            z = structural.predict(model, points)
            return z['WL_selected'], z['Size']

        path = f'blocks/{key}'
        v_ids = receipt['v_ids']
        original_v = inputs.npz_member(self.history.c.read('R7_INPUT', f'{path}/validation_input.npz'),
                                       ['validation_ids', 'validation_z0'])
        np.testing.assert_array_equal(original_v['validation_ids'], v_ids)
        b0_v = inputs.margin(original_v['validation_z0'])
        if group not in v_rows:
            v = np.column_stack((b0_v, *structural_scores(v_ids, v_ids)))
            old = json.loads(self.history.member('f', name))
            if (old['validation_ids'] != v_ids or self.history.files['f'][name] != receipt['r16_fit_member_sha256']
                    or old['lineage']['model_sha256'] != receipt['structure_sha256']
                    or old['lineage']['graph_sha256'] != access['graph_container_sha256']
                    or old['source_label_sha256']['V'] != receipt['v_label_sha256']):
                raise PermissionError(f'Original R16 V certificate differs: {group}')
            np.testing.assert_array_equal(v, old['validation_margins'])
            v_rows[group] = dict(ids=np.asarray(v_ids, dtype=np.int64), margins=v)
        np.testing.assert_array_equal(v_rows[group]['margins'][:,0], b0_v)

        original_a = inputs.npz_member(self.history.c.read('R7_INPUT', f'{path}/fit_input.npz'),
                                       ['adapt_ids', 'adapt_z0'])
        observed_a = inputs.npz_member(self.history.c.read('R7_OBS', f'{path}/tau-{self.history.c.tau:.3f}-observed.npz'),
                                       ['adapt_logits_B0', 'w_B0'])
        all_a, selected = m['ids']['adapt'], role['selected_a_ids']
        np.testing.assert_array_equal(original_a['adapt_ids'], all_a)
        if np.any(observed_a['w_B0'] != 0):
            raise PermissionError(f'Historical B0 was updated on A: {key}')
        np.testing.assert_allclose(inputs.margin(original_a['adapt_z0']), inputs.margin(observed_a['adapt_logits_B0']),
                                   rtol=0., atol=1e-12)
        index = {g: j for j, g in enumerate(all_a)}
        b0_a = inputs.margin(observed_a['adapt_logits_B0'])[[index[g] for g in selected]]
        wl_a, size_a = structural_scores(selected, all_a)
        np.testing.assert_array_equal(b0_a, raw_a['a_margin'])
        np.testing.assert_array_equal(size_a, raw_a['a_size_margin'])
        margins = np.column_stack((b0_a, wl_a, size_a))
        np.testing.assert_array_equal(encode(margins), x['a_bins'])

        holdout = inputs.npz_member(self.history.member('s', f'predictions/{key}.npz'))
        if self.history.files['s'][f'predictions/{key}.npz'] != role['evidence']['r10_sha256']:
            raise PermissionError(f'Historical R10 H artifact differs: {key}')
        np.testing.assert_array_equal(holdout['ids'], x['ids'])
        for name, r10_name in zip(MODELS, ('B0', 'WL_selected', 'Size'), strict=True):
            np.testing.assert_array_equal(holdout[f'action_{r10_name}'], x[name])
        if set(selected) & set(x['ids']):
            raise PermissionError(f'Historical A/H overlap: {key}')
        return dict(a_ids=x['a_ids'], a_margin=margins, h_ids=x['ids'], h_bins=x['h_bins'])


def selections(roles, sources, v, a):
    fits = {group: v_fit(row['margins'], sources[group]['counts']) for group,row in v.items()}
    own = {}
    for key, r in roles.items():
        group = r['group']
        if v[group]['ids'].tolist() != sources[group]['v_ids'] or a[key]['a_ids'].tolist() != r['selected_a_ids']:
            raise PermissionError(f'V or A order differs: {key}')
        own[key] = {m: estimate(a[key]['a_margin'][:,i], fits[group][m]) for i,m in enumerate(MODELS)}
    lookup = {(r['group'],r['target']):key for key,r in roles.items()}
    decisions = {}
    for key, row in roles.items():
        group = row['group']
        fit = fits[group]
        v_rate = {m: Fraction(k['k'], k['n']) for m,k in fit.items()}
        donors = sorted(t for g,t in lookup if g==group and t!=row['target'])
        if donors != row['other_targets'] or len(donors) != 2:
            raise PermissionError(f'Historical donor targets differ: {key}')
        donor_keys = [lookup[group,t] for t in donors]
        if any(len(a[k]['a_ids']) != len(a[key]['a_ids']) for k in donor_keys):
            raise PermissionError(f'Mismatched historical adaptation budgets: {key}')
        estimators = dict(SourceV=v_rate, OwnATC=own[key], OtherATC1=own[donor_keys[0]], OtherATC2=own[donor_keys[1]])
        decisions[key] = dict(choices={name: select(row_fit, fit) for name,row_fit in estimators.items()},
                              estimates={name: {m:str(s) for m,s in row_fit.items()} for name,row_fit in estimators.items()},
                              foreign_targets=donors)
    return fits,decisions


def members(archive):
    return ['executed-config.json', 'roles.json', 'thresholds.json', 'selections.json',
            *prior_run.inventory(archive.sources, 'validation'), *prior_run.inventory(archive.roles, 'directions')]


def freeze(out, archive):
    a, v = {}, {}
    for name in ('validation', 'directions'):
        (out/name).mkdir()
    for key in archive.roles:
        a[key] = archive.block(key, v)
        prior_run.save_predictions(out, f'directions/{key}.npz', a[key])
    for key, values in v.items():
        prior_run.save_predictions(out, f'validation/{key}.npz', values)
    fits, decisions = selections(archive.roles, archive.sources, v, a)
    for name, value in (('roles', archive.roles), ('thresholds', fits), ('selections', decisions)):
        write_json(out/f'{name}.json', value)
    return prior_run.seal(out, 'OBS', members(archive), archive.parents)


def observed(out, archive, obs_sha):
    blobs = prior_run.verified_members(out, 'OBS', obs_sha, members(archive), archive.parents)
    if json.loads(blobs['roles.json']) != archive.roles or json.loads(blobs['executed-config.json'])['plan'] != archive.cfg:
        raise PermissionError('ATC frozen scope or plan differs')
    v = {k: inputs.npz_member(blobs[f'validation/{k}.npz']) for k in archive.sources}
    a = {k: inputs.npz_member(blobs[f'directions/{k}.npz']) for k in archive.roles}
    fits, decisions = selections(archive.roles, archive.sources, v, a)
    if json.loads(blobs['thresholds.json']) != fits or json.loads(blobs['selections.json']) != decisions:
        raise PermissionError('Frozen ATC threshold or selection cannot reproduce')
    for key, role in archive.roles.items():
        history, _, _ = reverse.historical_votes(archive.blobs[f'predictions/{key}.npz'], role)
        for name, original in (('h_ids', 'ids'), ('h_bins','h_bins')):
            np.testing.assert_array_equal(a[key][name], history[original])
        np.testing.assert_array_equal(encode(a[key]['a_margin']), history['a_bins'])
    return a, decisions
