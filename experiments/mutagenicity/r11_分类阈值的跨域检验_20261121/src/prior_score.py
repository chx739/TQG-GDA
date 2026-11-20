"""Score only the sealed decisions using the original H cohorts and shared draws."""
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import paired, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from . import inputs, prior_run, run

METHODS = prior_run.UNLABELED + prior_run.SUPERVISED
GATE_CONTROLS = ('B0', 'SourceCal', 'SourceV', 'SwapMean')
PAIRS = tuple(('MLLS', arm) for arm in (*GATE_CONTROLS, 'Swap1', 'Swap2', 'OwnA'))
PAIRS += (('SourceCal', 'B0'), ('SourceCal', 'SourceV'), ('OwnA', 'B0'),
          ('PriorA', 'MLLS'), ('PriorA', 'SourceCal'), ('PriorA', 'OwnA'), ('MLLSRoot', 'MLLS'),
          ('MLLSRoot', 'SourceCal'), ('MLLSRoot', 'SwapRootMean'))
PAIRS += tuple((arm, 'MajorityA') for arm in METHODS if arm != 'MajorityA')


def compare(correct, weights):
    for (left, right) in PAIRS:
        if left not in correct:
            continue
        controls = {'SwapMean': ('Swap1', 'Swap2'), 'SwapRootMean': ('SwapRoot1', 'SwapRoot2')}.get(right, (right,))
        if any(arm not in correct for arm in controls):
            continue
        value = correct[left]
        gains = sum(int((value & ~correct[arm]).sum()) for arm in controls)
        harms = sum(int((~value & correct[arm]).sum()) for arm in controls)
        correction = Fraction(gains, len(controls))
        loss = Fraction(harms, len(controls))
        net = correction - loss
        delta = value.astype(float) - np.mean([correct[arm] for arm in controls], axis=0)
        yield (left, right), correction, loss, net, None if weights is None else weights @ delta


def gate(plan, registered, summaries, pairs):
    directions = [(s, t) for s in plan['domains'] for t in plan['domains'] if s != t]
    cutoff = Fraction(*registered['minimum_mean_accuracy_fraction'])
    by_setting = {}
    for setting in plan['settings']:
        values = {r['right']: None if r['exact_fraction'] is None else Fraction(r['exact_fraction'])
                  for r in summaries if r['setting'] == setting and r['scope'] == 'setting' and r['left'] == 'MLLS'}
        positive = 0
        for source, target in directions:
            both = [[pairs.get((setting, source, target, seed, 'MLLS', arm)) for seed in plan['seeds']]
                    for arm in ('SourceCal', 'SwapMean')]
            positive += all(None not in xs and sum(xs) > 0 for xs in both)
        complete = all(values.get(arm) is not None for arm in GATE_CONTROLS)
        passed = (complete and all(values[arm] > 0 for arm in GATE_CONTROLS)
                  and all(values[arm] >= cutoff for arm in ('SourceCal', 'SwapMean'))
                  and positive >= registered['positive_directions_per_setting'])
        by_setting[setting] = dict(passed=passed, complete=complete, simultaneously_positive_directions=positive,
                                   comparison_pp={arm: None if values.get(arm) is None else float(100 * values[arm])
                                                  for arm in GATE_CONTROLS})
    return dict(settings=by_setting, passed=sum(x['passed'] for x in by_setting.values()) >= registered['settings_required'])


def endpoint_metrics(predictions, y):
    for arm in METHODS:
        if arm not in predictions:
            continue
        predicted = predictions[arm]
        correct = predicted == y
        recalls = [float(correct[y == c].mean()) if np.any(y == c) else None for c in (0, 1)]
        yield arm, correct, dict(method=arm, n=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
                                 ba=None if None in recalls else sum(recalls) / 2, recall_0=recalls[0], recall_1=recalls[1],
                                 predicted_positive=int(predicted.sum()), constant=bool(predicted.min() == predicted.max()))


def endpoint_summary(plan, endpoints):
    rows = []
    expected = len(plan['seeds']) * len(plan['domains']) * (len(plan['domains']) - 1)
    metrics = ('accuracy', 'ba', 'recall_0', 'recall_1')
    for setting in plan['settings']:
        for arm in METHODS:
            values = [row for row in endpoints if row['setting'] == setting and row['method'] == arm]
            observed = len(values)
            means = {m: float(np.mean([r[m] for r in values]))
                     if observed == expected and all(r[m] is not None for r in values) else None for m in metrics}
            rows.append(dict(setting=setting, method=arm, available=observed, expected=expected,
                             constant_blocks=sum(r['constant'] for r in values), **means))
    return rows


def prior_error_rows(key, task, fits, a_counts):
    count = a_counts[key]
    for name in ('em', 'root'):
        entry = fits[key].get(name, {})
        q = entry.get('target_prior')
        yield dict(**{k: task[k] for k in prior_run.ID_FIELDS}, solver=name,
                   a_positives=count['positive'], a_n=count['n'], a_sample_prior=count['sample_prior'],
                   estimated_prior=q, signed_error=None if q is None else q - count['sample_prior'],
                   at_bound=entry.get('prior_at_bound'),
                   error=fits[key]['calibration_error'] or entry.get('error'))


def prior_error_summary(plan, errors):
    for setting in plan['settings']:
        for solver in ('em', 'root'):
            selected = [row for row in errors if row['setting'] == setting and row['solver'] == solver]
            available = [row['signed_error'] for row in selected if row['signed_error'] is not None]
            yield dict(setting=setting, solver=solver, expected=len(selected), available=len(available),
                       failed=len(selected)-len(available), at_bound=sum(row['at_bound'] is True for row in selected),
                       mean_signed_error=float(np.mean(available)) if available else None,
                       mean_absolute_error=float(np.mean(np.abs(available))) if available else None)


def score(output, repo, registered, plan, roots, unlabeled_sha, obs_sha):
    keys = prior_run.registered_keys(plan)
    observed = prior_run.verified_members(output, 'OBS', obs_sha, prior_run.supervised_members(keys),
                                          {'unlabeled_sha256': unlabeled_sha})
    unlabeled = prior_run.verified_members(output, 'UNLABELED', unlabeled_sha, prior_run.unlabeled_members(keys), roots)
    roles = json.loads(unlabeled['roles.json'])
    a_counts = json.loads(observed['a-counts.json'])
    fits = json.loads(unlabeled['fits.json'])
    if any(set(data) != set(keys) for data in (roles, a_counts, fits)):
        raise PermissionError('Missing sealed scored blocks')
    original = json.loads(unlabeled['r7-executed-plan.json'])
    reps, seed, minimum, _ = paired._settings(original)
    endpoints, comparisons, prior_errors, solver_parity = [], [], [], []
    values, curves, cohorts, h_access = {}, {}, {}, {}
    archive = Path(plan['archive'])
    with seals.verified_seal_chain(archive, 'R7_FINAL') as scope, inputs.GraphLabelSession('holdout') as labels:
        for key in keys:
            task = roles[key]
            identity = {k: task[k] for k in prior_run.ID_FIELDS}
            setting, source, target, model_seed = identity.values()
            u = inputs.npz_member(unlabeled[f'unlabeled/{key}.npz'])
            a = inputs.npz_member(observed[f'supervised/{key}.npz'])
            ids = task['holdout_ids']
            prior_run.verified_predictions(u, ids, prior_run.UNLABELED, ('B0', 'SourceV'))
            prior_run.verified_predictions(a, ids, prior_run.SUPERVISED, ('OwnA', 'MajorityA'))
            manifest = prior_run.label_manifest(scope, key, task, 'holdout')
            cohort_key = setting, target
            if cohort_key not in cohorts:
                y = labels.load(prior_run.label_port(manifest, task, 'holdout', repo), ids)
                if y.shape != (len(ids),) or not np.isin(y, [0, 1]).all():
                    raise ValueError('Expected binary labels for the authenticated H cohort')
                order = np.argsort(ids, kind='stable')
                bootstrap = paired._cohort(setting, target, u['ids'][order], y[order], reps, seed, minimum)
                cohorts[cohort_key] = y, order, bootstrap
                h_access[key] = ids
            y, order, cohort = cohorts[cohort_key]
            np.testing.assert_array_equal(u['ids'][order], cohort['ids'])
            correct = {}
            for arm, flags, row in endpoint_metrics(u | a, y):
                endpoints.append(dict(**identity, **row))
                correct[arm] = flags[order]
            for (left, right), correction, harm, net, draws in compare(correct, cohort['weights']):
                pairing = (setting, source, target, model_seed, left, right)
                values[pairing] = net / len(y)
                if draws is not None:
                    curves[pairing] = draws
                comparisons.append(dict(**identity, left=left, right=right, n=len(y), corrected_exact=str(correction),
                                        harmed_exact=str(harm), net_exact=str(net), accuracy_pp=float(100 * values[pairing])))
            prior_errors.extend(prior_error_rows(key, task, fits, a_counts))
            em, root = fits[key].get('em', {}), fits[key].get('root', {})
            if 'target_prior' in em and 'target_prior' in root:
                solver_parity.append(dict(**identity, em_minus_root_prior=em['target_prior'] - root['target_prior'],
                                          root_minus_em_log_likelihood=root['final_log_likelihood'] - em['final_log_likelihood'],
                                          prediction_disagreements=int((u['MLLS'] != u['MLLSRoot']).sum())))
    summaries = run.summarize_pairs(plan, values, curves, PAIRS)
    result = gate(plan, registered, summaries, values)
    tables = {'endpoints': endpoints, 'endpoint-summary': endpoint_summary(plan, endpoints),
              'paired-blocks': comparisons, 'paired-summary': summaries,
              'prior-errors': prior_errors, 'prior-error-summary': list(prior_error_summary(plan, prior_errors)),
              'solver-parity': solver_parity}
    for name, rows in tables.items():
        write_table(output / f'{name}.csv', rows)
    write_json(output / 'gate.json', result)
    write_json(output / 'h-label-access.json', dict(obs_sha256=obs_sha, actual_reads=h_access))
    write_json(output / 'cohort-audit.json', [entry[2]['audit'] for entry in cohorts.values()])
    files = ['OBS.json', 'gate.json', 'h-label-access.json', 'cohort-audit.json', *[f'{name}.csv' for name in tables]]
    final = prior_run.seal(output, 'FINAL', files, {'obs_sha256': obs_sha})
    return dict(final_sha256=final, gate=result)
