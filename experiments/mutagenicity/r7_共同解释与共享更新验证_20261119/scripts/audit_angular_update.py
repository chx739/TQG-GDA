"""Compare V8/V9 frozen P predictions after BOTH R7_FINAL seals verify.

RMS matching and the calibration/direction decomposition use unlabeled A
only. H labels enter endpoint reporting after all paired input checks. No
study model is retrained, selected, written back, or supplied to a phase gate.
"""
import argparse
import hashlib
import io
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from experiments.common.io import write_json

from ..src.evaluate import classification
from ..src.paired import paired_bootstrap
from ..src.r5_inputs import load_labels
from ..src.seals import verify_seal


TAU = .005
WIDTH = 'tau-0.005'
ARMS = ('B0', 'C05', 'C075', 'V8_J', 'V8_J_rms_matched', 'V9_J', 'V9_directional')
COMPARISONS = [(name, 'B0') for name in ARMS if name != 'B0'] + [
    ('V9_J', 'V8_J'), ('V9_J', 'V8_J_rms_matched'), ('V9_J', 'C05'), ('V9_J', 'C075'),
    ('V9_directional', 'V9_J'), ('V9_directional', 'V8_J_rms_matched'),
]


def _logits(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f'{name} must contain finite binary logits')
    return values


def _margin(values):
    return values[:, 1] - values[:, 0]


def _rms(values):
    maximum = float(np.max(np.abs(values)))
    return maximum * float(np.sqrt(np.mean((values / maximum) ** 2))) if maximum else 0.


def _add_margin(baseline, delta):
    # Class-common components of a binary correction do not affect prediction.
    return baseline + np.column_stack((-delta / 2, delta / 2))


def derive_arms(b0_a, b0_h, v8_a, v8_h, v9_a, v9_h, *, valid_v8=True, valid_v9=True):
    """Pure label-free A calibration and application to already frozen H logits."""
    b0_a, b0_h = _logits(b0_a, 'A B0'), _logits(b0_h, 'H B0')
    arms = dict(B0=b0_h, C05=.5 * b0_h, C075=.75 * b0_h,
                V8_J=None, V8_J_rms_matched=None, V9_J=None, V9_directional=None)
    detail = dict(fitting_pool='unlabeled_A_only', rms_definition='sqrt(mean(delta_binary_margin**2))',
                  calibration_columns=['intercept', 'frozen_B0_binary_margin'],
                  least_squares_rcond=1e-12, reasons={}, rms_scale=None)
    pairs = {}
    for version, valid, a, h in (('V8', valid_v8, v8_a, v8_h), ('V9', valid_v9, v9_a, v9_h)):
        if not valid:
            detail['reasons'][version + '_J'] = 'invalid_sealed_J'
            continue
        a, h = _logits(a, version + ' A J'), _logits(h, version + ' H J')
        if a.shape != b0_a.shape or h.shape != b0_h.shape:
            raise ValueError('Paired J and B0 shapes differ')
        da, dh = _margin(a) - _margin(b0_a), _margin(h) - _margin(b0_h)
        if not np.isfinite(da).all() or not np.isfinite(dh).all():
            raise ValueError('Nonfinite binary correction')
        pairs[version] = da, dh
        arms[version + '_J'] = h
        detail[version + '_A_margin_rms'] = _rms(da)
    if len(pairs) == 2:
        old, new = detail['V8_A_margin_rms'], detail['V9_A_margin_rms']
        if old == 0:
            detail['reasons']['V8_J_rms_matched'] = 'zero_V8_A_RMS_no_identifiable_matching_scale'
        else:
            alpha = new / old
            with np.errstate(over='ignore', invalid='ignore'):
                scaled = _add_margin(b0_h, alpha * pairs['V8'][1])
            if np.isfinite(alpha) and np.isfinite(scaled).all():
                arms['V8_J_rms_matched'] = scaled
                detail.update(rms_scale=alpha, matched_V8_A_margin_rms=_rms(alpha * pairs['V8'][0]))
            else:
                detail['reasons']['V8_J_rms_matched'] = 'nonfinite_RMS_matched_correction'
    else:
        detail['reasons']['V8_J_rms_matched'] = 'requires_both_valid_J_arms'
    if 'V9' in pairs:
        design = np.column_stack((np.ones(len(b0_a)), _margin(b0_a)))
        coefficient, _, rank, singular = np.linalg.lstsq(design, pairs['V9'][0], rcond=1e-12)
        detail.update(calibration_coefficients=coefficient.tolist(), calibration_rank=int(rank),
                      calibration_singular_values=singular.tolist())
        if rank < 2:
            detail['reasons']['V9_directional'] = 'rank_deficient_A_calibration_design'
        else:
            residual_a = pairs['V9'][0] - design @ coefficient
            calibration_h = coefficient[0] + coefficient[1] * _margin(b0_h)
            residual_h = pairs['V9'][1] - calibration_h
            directional = _add_margin(b0_h, residual_h)
            if not np.isfinite(directional).all():
                detail['reasons']['V9_directional'] = 'nonfinite_directional_correction'
            else:
                arms['V9_directional'] = directional
                detail.update(directional_A_margin_rms=_rms(residual_a),
                              A_residual_normal_equation_max_abs=float(np.max(np.abs(design.T @ residual_a))),
                              H_direction_rule='delta9_H - beta_intercept - beta_margin * B0_margin_H')
    else:
        detail['reasons']['V9_directional'] = 'requires_valid_V9_J'
    return arms, detail


def _read(root, relative, record, *, keys=None):
    path = root / relative
    if (Path(relative).is_absolute() or '..' in Path(relative).parts
            or not path.resolve().is_relative_to(root) or relative not in record['files']):
        raise PermissionError(f'Artifact is outside the verified seal: {relative}')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record['files'][relative]:
        raise PermissionError(f'Sealed artifact changed: {relative}')
    if keys is None:
        return json.loads(payload)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: archive[name] for name in keys}


def _grid(plan):
    phase = plan['phases']['P']
    keys = [(phase['setting'], source, target, seed)
            for (source, target), seed in product(phase['directions'], plan['real_model_seeds'])]
    if (len(keys) != len(set(keys)) or len(keys) != phase['expected_blocks']
            or plan['shared_update']['primary_tau'] != TAU or TAU not in phase['width_grid']):
        raise ValueError('Incomplete/duplicated pilot grid or width differs from .005')
    return keys


def _block_name(key):
    setting, source, target, seed = key
    return f'{setting}-M{source}_to_M{target}-seed-{seed}'


def _pair_inputs(roots, records, plans, grid):
    """Complete every identity/B0 check before any raw target-label request."""
    expected_names = {_block_name(key) for key in grid}
    result = []
    for root in roots:
        if {p.name for p in (root / 'blocks').iterdir() if p.is_dir()} != expected_names:
            raise ValueError('Pilot directories do not contain exactly the complete registered blocks')
    for key in grid:
        relative = f'blocks/{_block_name(key)}'
        sides = []
        for root, rec, plan in zip(roots, records, plans):
            manifest = _read(root, relative + '/manifest.json', rec['INPUT'])
            if (tuple(manifest[k] for k in ('setting', 'source', 'target', 'seed')) != key
                    or manifest['r7_plan'] != plan):
                raise ValueError('Block identity or its frozen plan differs from pilot registration')
            a = _read(root, relative + '/fit_input.npz', rec['INPUT'], keys=['adapt_ids', 'adapt_z0'])
            h = _read(root, relative + '/holdout_input.npz', rec['INPUT'], keys=['holdout_ids', 'holdout_z0'])
            for pool, values in (('adapt', a), ('holdout', h)):
                ids = values[pool + '_ids']
                if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
                        or len(np.unique(ids)) != len(ids) or ids.tolist() != manifest['ids'][pool]):
                    raise ValueError('A/H array order differs from the authenticated manifest')
            if set(a['adapt_ids']) & set(h['holdout_ids']):
                raise ValueError('A and H overlap')
            obs = _read(root, relative + f'/{WIDTH}-observed.npz', rec['OBS'],
                        keys=['adapt_logits_' + name for name in ('B0', 'C05', 'C075', 'J')])
            held = _read(root, relative + f'/{WIDTH}-heldout.npz', rec['OBS'],
                         keys=['holdout_logits_' + name for name in ('B0', 'C05', 'C075', 'J')])
            diagnostics = _read(root, relative + f'/{WIDTH}-observed.json', rec['OBS'])
            for name in ('B0', 'C05', 'C075', 'J'):
                if type(diagnostics[name]['valid']) is not bool:
                    raise ValueError('Sealed method validity must be an explicit boolean')
                if name != 'J' and not diagnostics[name]['valid']:
                    raise ValueError('Fixed B0/temperature controls are invalid')
            for pool, values, predictions in (('adapt', a, obs), ('holdout', h, held)):
                z0 = _logits(values[pool + '_z0'], pool + ' raw B0')
                frozen = _logits(predictions[pool + '_logits_B0'], pool + ' frozen B0')
                if z0.shape != frozen.shape or z0.shape[0] != len(values[pool + '_ids']):
                    raise ValueError('Frozen B0 shape differs from its graph IDs')
                np.testing.assert_allclose(_margin(frozen), _margin(z0), rtol=1e-12, atol=1e-12)
                for name, multiplier in (('C05', .5), ('C075', .75)):
                    control = _logits(predictions[pool + '_logits_' + name], name)
                    np.testing.assert_allclose(_margin(control), multiplier * _margin(z0), rtol=1e-12, atol=1e-12)
            sides.append(dict(manifest=manifest, a=a, h=h, obs=obs, held=held, diagnostic=diagnostics['J']))
        left, right = sides
        for pool, field in (('a', 'adapt'), ('h', 'holdout')):
            for suffix in ('ids', 'z0'):
                np.testing.assert_array_equal(left[pool][field + '_' + suffix], right[pool][field + '_' + suffix])
            kind = 'obs' if field == 'adapt' else 'held'
            np.testing.assert_array_equal(left[kind][field + '_logits_B0'], right[kind][field + '_logits_B0'])
        if (left['manifest']['ids'] != right['manifest']['ids']
                or left['manifest']['provenance'] != right['manifest']['provenance']):
            raise ValueError('Two pilots do not share the complete source/fold provenance')
        arms, detail = derive_arms(
            right['a']['adapt_z0'], right['h']['holdout_z0'],
            left['obs']['adapt_logits_J'], left['held']['holdout_logits_J'],
            right['obs']['adapt_logits_J'], right['held']['holdout_logits_J'],
            valid_v8=left['diagnostic']['valid'], valid_v9=right['diagnostic']['valid'])
        detail['sealed_J_status'] = {name: {k: side['diagnostic'].get(k) for k in ('valid', 'status', 'stop_reason')}
                                     for name, side in zip(('V8', 'V9'), sides)}
        result.append(dict(key=key, relative=relative, ids=right['h']['holdout_ids'],
                           arms=arms, diagnostics=detail))
    return result


def _endpoint(logits, y, b0):
    values = classification(logits, y, b0.argmax(axis=1))
    counts = np.bincount(y, minlength=2)
    ranks = rankdata(_margin(logits), method='average')
    auc = float((ranks[y == 1].sum() - counts[1] * (counts[1] + 1) / 2) / (counts[0] * counts[1])) if counts.all() else None
    return dict(nll=values['sampled_nll'], ba_pp=None if values['balanced_accuracy'] is None else 100 * values['balanced_accuracy'],
                accuracy_pp=100 * values['accuracy'], net_corrections=values['net_corrections'],
                corrected=values['corrected'], harmed=values['harmed'], auc=auc)


def _summaries(rows, plan):
    result = []
    keys = ('nll', 'ba_pp', 'accuracy_pp', 'net_corrections', 'auc')
    for source, target in plan['phases']['P']['directions']:
        for arm in ARMS:
            selected = [row for row in rows if row['source'] == source and row['target'] == target and row['arm'] == arm]
            expected = len(plan['real_model_seeds'])
            complete = len(selected) == expected and all(row['valid'] for row in selected)
            result.append(dict(source=source, target=target, arm=arm, expected_seeds=expected,
                               valid_seeds=sum(row['valid'] for row in selected), complete=complete,
                               reason=None if complete else 'incomplete_valid_method_pairs',
                               **{key: float(np.mean([row[key] for row in selected]))
                                  if complete and all(row.get(key) is not None for row in selected) else None for key in keys}))
    return result


def audit(reference, angular, *, repository=None):
    """Return a JSON report; never writes into either run or changes a method."""
    roots = [Path(path).resolve() for path in (reference, angular)]
    if roots[0] == roots[1]:
        raise ValueError('Reference and angular pilots must be distinct')
    for root in roots:
        if not (root / 'seals/R7_FINAL.json').is_file():
            raise FileNotFoundError(f'Pilot is not FINAL-sealed: {root}')
    # Do not inspect either plan/prediction or request labels until both finals pass.
    finals = [verify_seal(root, 'R7_FINAL') for root in roots]
    records = [{stage: verify_seal(root, 'R7_' + stage) for stage in ('INPUT', 'OBS')} for root in roots]
    plans = [_read(root, 'executed-plan.json', rec['INPUT']) for root, rec in zip(roots, records)]
    for root, rec, plan, iteration in zip(roots, records, plans, (8, 9)):
        execution = _read(root, 'executed-configs.json', rec['INPUT'])
        if execution.get('phase') != 'P' or plan.get('development_iteration') != iteration:
            raise ValueError('Expected the frozen V8 P and V9 P in that order')
    options = plans[1]['diagnostics_iteration09']
    if (options['primary_tau'] != TAU or not options['same_blocks_and_B0_required']
            or not options['adaptation_rms_matched_reference'] or not options['a_only_intercept_and_margin_residual']
            or not options['target_labels_after_R7_FINAL'] or options['affects_p_gate']):
        raise ValueError('V9 diagnostic registration differs from this implementation')
    if Path(options['reference']).parts != roots[0].parts[-len(Path(options['reference']).parts):]:
        raise ValueError('Reference pilot location differs from the preregistered V8 attempt')
    for name in ('shared_update', 'target_split', 'real_settings', 'source_mixture_options', 'posterior_mass_options'):
        if plans[0][name] != plans[1][name]:
            raise ValueError(f'V8/V9 differ in a supposedly fixed configuration: {name}')
    grids = [_grid(plan) for plan in plans]
    if set(grids[0]) != set(grids[1]):
        raise ValueError('V8/V9 registered complete block identities differ')
    paired_inputs = _pair_inputs(roots, records, plans, grids[1])
    rows, contrasts, cohorts, diagnostics = [], [], [], []
    for block in paired_inputs:
        identity = dict(zip(('setting', 'source', 'target', 'seed'), block['key']))
        relative = block['relative'] + '/manifest.json'
        labels = [load_labels(root, relative, 'holdout', repository=repository) for root in roots]
        np.testing.assert_array_equal(labels[0], labels[1])
        y = labels[1]
        scores = {}
        for arm, prediction in block['arms'].items():
            valid = prediction is not None
            row = dict(**identity, arm=arm, n=len(y), valid=valid,
                       reason=block['diagnostics']['reasons'].get(arm))
            if valid:
                scores[arm] = _endpoint(prediction, y, block['arms']['B0'])
                row.update(scores[arm])
            rows.append(row)
        for left, right in COMPARISONS:
            valid = left in scores and right in scores
            contrasts.append(dict(**identity, left=left, right=right, valid=valid,
                                  reason=None if valid else 'invalid_or_unavailable_paired_arm',
                                  **{'delta_' + metric: scores[left][metric] - scores[right][metric]
                                     if valid and scores[left][metric] is not None and scores[right][metric] is not None else None
                                     for metric in ('nll', 'ba_pp', 'net_corrections', 'auc')}))
        cohorts.append(dict(**identity, ids=block['ids'], y=y, logits=block['arms']))
        diagnostics.append(identity | block['diagnostics'])
    bootstrap = paired_bootstrap(plans[1], cohorts, phase='P', comparisons=COMPARISONS, tau=TAU)
    counts = {(row['setting'], row['target']): len(row['ids']) for row in cohorts}
    intervals = []
    for row in bootstrap['interval_rows']:
        if row['scope'] != 'direction' or row['metric'] not in ('NLL', 'BA', 'Accuracy'):
            continue
        row = dict(row)
        if row['metric'] == 'Accuracy':
            factor = counts[row['setting'], row['target']] / 100
            row.update(metric='net_corrections', unit='graphs')
            for field in ('estimate', 'lower', 'upper', 'seed_sample_sd'):
                if row[field] is not None:
                    row[field] *= factor
            row['seed_estimates'] = [dict(item, estimate=None if item['estimate'] is None else item['estimate'] * factor)
                                     for item in row['seed_estimates']]
        intervals.append(row)
    return dict(format='r7-v9-angular-update-audit-v1', tau=TAU,
                scope='fixed diagnostic on shared sealed development H; no fitting/selection on H; does not affect P gate',
                final_seals={name: dict(root=str(root), digest=final['digest'])
                             for name, root, final in zip(('V8', 'V9'), roots, finals)},
                expected_blocks=len(grids[1]), complete_block_identities=True,
                conventions=dict(binary_margin='logit_1_minus_logit_0', differences='left_minus_right',
                                 direction_aggregation='equal_weight_all_registered_model_seeds; incomplete estimates remain null',
                                 net_corrections='corrected_minus_harmed_relative_to_shared_B0; contrasts subtract these counts',
                                 auc='class_1 rank AUC with average ties; point estimates only'),
                block_diagnostics=diagnostics, block_endpoints=rows, block_contrasts=contrasts,
                direction_endpoints=_summaries(rows, plans[1]), direction_contrasts=intervals,
                cohort_audit=bootstrap['cohort_audit'], bootstrap=bootstrap['metadata'],
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy_version=np.__version__)


def check_calculations():
    """Pure generated arrays: no experiment directory or target label reader."""
    a_margin = np.array([-2., -1., 0., 1., 2.])
    h_margin = np.array([-3., -.5, .5, 3.])
    z_a = np.column_stack((-a_margin / 2, a_margin / 2))
    z_h = np.column_stack((-h_margin / 2, h_margin / 2))
    nonlinear_a = np.array([1., -2., 2., -2., 1.])
    nonlinear_h = np.array([.7, -.4, .2, -.5])
    delta_a = .3 + .2 * a_margin + nonlinear_a
    delta_h = .3 + .2 * h_margin + nonlinear_h
    old_a, old_h = _add_margin(z_a, delta_a / 2), _add_margin(z_h, delta_h / 2)
    new_a, new_h = _add_margin(z_a, delta_a), _add_margin(z_h, delta_h)
    arms, detail = derive_arms(z_a, z_h, old_a, old_h, new_a, new_h)
    np.testing.assert_allclose(detail['calibration_coefficients'], [.3, .2], rtol=0, atol=1e-14)
    np.testing.assert_allclose(detail['rms_scale'], 2., rtol=0, atol=1e-14)
    np.testing.assert_allclose(_margin(arms['V8_J_rms_matched']), _margin(new_h), rtol=0, atol=1e-14)
    np.testing.assert_allclose(_margin(arms['V9_directional']), h_margin + nonlinear_h, rtol=0, atol=1e-14)
    # Altering H does not change any coefficient selected on A.
    _, changed = derive_arms(z_a, z_h * 7, old_a, old_h * 7, new_a, new_h * 7)
    for key in ('rms_scale', 'calibration_coefficients', 'V8_A_margin_rms', 'V9_A_margin_rms'):
        assert detail[key] == changed[key]
    labels = np.array([0, 1, 0, 1])
    base = _endpoint(z_h, labels, z_h)
    for name in ('C05', 'C075'):
        values = _endpoint(arms[name], labels, z_h)
        assert values['ba_pp'] == base['ba_pp'] and values['auc'] == base['auc'] and values['net_corrections'] == 0
    invalid, _ = derive_arms(z_a, z_h, None, None, new_a, new_h, valid_v8=False)
    assert invalid['V8_J'] is None and invalid['V8_J_rms_matched'] is None and invalid['V9_J'] is not None
    zero, report = derive_arms(z_a, z_h, z_a, z_h, new_a, new_h)
    assert zero['V8_J_rms_matched'] is None and report['reasons']['V8_J_rms_matched'].startswith('zero_')
    rank_zero, report = derive_arms(np.zeros_like(z_a), z_h, z_a, z_h, new_a, new_h)
    assert rank_zero['V9_directional'] is None and report['calibration_rank'] == 1
    return dict(status='passed', A_only_scaling_and_residual=True, H_does_not_select_coefficients=True,
                temperature_preserves_BA_and_AUC=True, invalid_and_degenerate_cases_explicit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', '--v8', type=Path)
    parser.add_argument('--angular', '--v9', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true', help='Run only generated pure calculation fixtures')
    args = parser.parse_args()
    if args.check:
        if any(value is not None for value in (args.reference, args.angular, args.output)):
            parser.error('--check cannot be combined with experiment paths')
        print(json.dumps(check_calculations(), allow_nan=False))
        return
    if any(value is None for value in (args.reference, args.angular, args.output)):
        parser.error('--reference, --angular, and --output are required unless --check is used')
    output = args.output.resolve()
    if any(output.is_relative_to(root.resolve()) for root in (args.reference, args.angular)):
        raise ValueError('Write the diagnostic outside both immutable pilot directories')
    if output.exists():
        raise FileExistsError(output)
    write_json(output, audit(args.reference, args.angular))


if __name__ == '__main__':
    main()
