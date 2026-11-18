"""Compare V10 sum/mean four-arm predictions only after BOTH P FINAL seals.

This descriptive, fixed-model diagnostic never changes either phase gate.
Graph-bootstrap intervals condition on the fitted models; difference-in-
differences is a point estimate, not an independent confirmation experiment.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.common.io import write_json

from .audit_angular_update import TAU, WIDTH, _block_name, _grid, _logits, _margin, _read
from ..src.evaluate import classification
from ..src.paired import paired_bootstrap
from ..src.r5_inputs import load_labels
from ..src.seals import verify_seal


ARMS = ('B0_sum', 'J_sum', 'B0_mean', 'J_mean')
COMPARISONS = [('B0_mean', 'B0_sum'), ('J_mean', 'B0_mean'),
               ('J_sum', 'B0_sum'), ('J_mean', 'J_sum')]
METRICS = ('NLL', 'BA', 'Accuracy', 'net_corrections')


def _plans(plans):
    normalized = []
    for pooling, original in zip(('sum', 'mean'), plans):
        plan = deepcopy(original)
        if (plan.get('development_iteration') != 10
                or plan['source_refit']['pooling'] != pooling
                or plan['phases']['D']['source_model']['pooling'] != pooling):
            raise ValueError('Expected V10 sum and mean plans, in that order')
        plan['source_refit']['pooling'] = plan['phases']['D']['source_model']['pooling'] = '<paired>'
        normalized.append(plan)
    if len(normalized) != 2 or normalized[0] != normalized[1]:
        raise ValueError('Branch plans differ beyond the two registered pooling fields')
    options = plans[1]['paired_pooling_diagnostic']
    if (options['primary'] != 'mean' or options['reference'] != 'sum'
            or not options['same_roles_and_graph_ids']
            or options['contrasts'] != [list(pair) for pair in COMPARISONS]
            or options['paired_effect'] != '(J_mean - B0_mean) - (J_sum - B0_sum)'
            or options['effect_metrics'] != list(METRICS)
            or options['source_selection'] != 'minimum_source_validation_CE_per_pooling_and_seed'
            or options['affects_primary_p_gate'] is not False
            or options['final_seal_required_in_both_p_branches'] is not True):
        raise ValueError('Pooling diagnostic differs from its registered readout')
    grids = [_grid(plan) for plan in plans]
    if grids[0] != grids[1]:
        raise ValueError('Branches do not have the identical ordered complete P grid')
    return grids[0]


def _refit(root, manifest, record, pooling):
    refit = manifest['provenance']['source_refit']
    if (refit['pooling'] != pooling or refit['b0_origin'] != 'fresh_source_refit'
            or refit['recipe'] != manifest['r7_plan']['source_refit']['recipe']):
        raise ValueError('Source-refit provenance differs from its branch registration')
    artifact = Path(refit['artifact_root']).resolve()
    if not artifact.is_relative_to(root / 'source-refit'):
        raise ValueError('P source artifact must be inside its sealed source-refit directory')
    prefix = artifact.relative_to(root).as_posix()
    for name, expected in refit['files'].items():
        if record['files'].get(prefix + '/' + name) != expected:
            raise ValueError('Source artifact hashes differ from R7_INPUT')
    prepared = _read(root, prefix + '/prepared.json', record)
    complete = _read(root, prefix + '/complete.json', record)
    initial = refit['initialization_state_sha256']
    if (not isinstance(initial, str) or len(initial) != 64
            or any(char not in '0123456789abcdef' for char in initial)
            or complete['initialization_state_sha256'] != initial
            or complete['digest'] != refit['artifact_digest']
            or prepared['digest'] != refit['preparation_digest']
            or complete['preparation_digest'] != prepared['digest']
            or 'initialization.pt' not in complete['files']
            or complete['files'] != {k: v for k, v in refit['files'].items() if k != 'complete.json'}
            or complete['pooling'] != pooling or prepared['pooling'] != pooling
            or tuple(prepared[k] for k in ('setting', 'source', 'seed'))
            != tuple(manifest[k] for k in ('setting', 'source', 'seed'))
            or prepared['ids'] != {key: manifest['ids'][key] for key in ('source', 'validation')}
            or complete['selected_epoch'] != manifest['provenance']['selected_source_epoch']
            or prepared['source_config_hash'] != manifest['provenance']['source_config_hash']
            or complete['target_labels_loaded'] is not False
            or complete['target_forward_pass_before_freezing'] is not False
            or prepared['target_labels_loaded'] is not False
            or prepared['target_forward_pass_before_freezing'] is not False):
        raise ValueError('Source initialization, roles, or frozen-artifact identity disagrees')
    for key in ('code', 'environment', 'source_roles', 'r5_input_files'):
        if prepared[key] != refit[key]:
            raise ValueError('Source preparation and manifest disagree: ' + key)
    prepared = deepcopy(prepared)
    prepared.pop('digest')
    # Pooling is separate from the legacy Config, whose tensors/init stay identical.
    prepared['pooling'] = '<paired>'
    return dict(initialization=initial, preparation=prepared,
                artifact_digest=complete['digest'], selected_epoch=complete['selected_epoch'])


def _match_inputs(sides):
    left, right = sides
    if left['manifest']['ids'] != right['manifest']['ids']:
        raise ValueError('Source/validation/A/H IDs or order differ between branches')
    if (left['refit']['initialization'] != right['refit']['initialization']
            or left['refit']['preparation'] != right['refit']['preparation']):
        raise ValueError('Source initialization or source-only recipe differs beyond pooling')
    # Selected epochs and all fitted model values may differ under pooling.
    fixed = [{k: v for k, v in side['manifest']['provenance'].items()
              if k not in ('source_refit', 'selected_source_epoch')} for side in sides]
    if fixed[0] != fixed[1] or left['manifest']['label_records'] != right['manifest']['label_records']:
        raise ValueError('Paired branches do not share their original R5 data provenance')


def _inputs(roots, records, plans, grid):
    expected = {_block_name(key) for key in grid}
    for root in roots:
        if {p.name for p in (root / 'blocks').iterdir() if p.is_dir()} != expected:
            raise ValueError('Pilot does not contain exactly the complete registered blocks')
    result, source_artifacts = [], {}
    for key in grid:
        relative = 'blocks/' + _block_name(key)
        sides, arms = [], {}
        for pooling, root, rec, plan in zip(('sum', 'mean'), roots, records, plans):
            manifest = _read(root, relative + '/manifest.json', rec['INPUT'])
            if (tuple(manifest[k] for k in ('setting', 'source', 'target', 'seed')) != key
                    or manifest['r7_plan'] != plan or manifest['format'] != 'r7-r5-inputs-v1'):
                raise ValueError('Frozen block identity/plan differs from its registration')
            fit = _read(root, relative + '/fit_input.npz', rec['INPUT'],
                        keys=['source_ids', 'adapt_ids', 'adapt_z0'])
            held = _read(root, relative + '/holdout_input.npz', rec['INPUT'], keys=['holdout_ids', 'holdout_z0'])
            validation = _read(root, relative + '/validation_input.npz', rec['INPUT'], keys=['validation_ids'])
            for pool, arrays in (('source', fit), ('adapt', fit), ('holdout', held), ('validation', validation)):
                ids = arrays[pool + '_ids']
                if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
                        or len(np.unique(ids)) != len(ids) or ids.tolist() != manifest['ids'][pool]):
                    raise ValueError('Input graph IDs differ from their authenticated manifest')
            pools = [set(manifest['ids'][pool]) for pool in ('source', 'validation', 'adapt', 'holdout')]
            if any(a & b for i, a in enumerate(pools) for b in pools[i + 1:]):
                raise ValueError('Source/validation/A/H graph roles overlap')
            obs = _read(root, relative + '/' + WIDTH + '-observed.npz', rec['OBS'],
                        keys=['adapt_logits_B0', 'adapt_logits_J'])
            predictions = _read(root, relative + '/' + WIDTH + '-heldout.npz', rec['OBS'],
                                keys=['holdout_logits_B0', 'holdout_logits_J'])
            status = _read(root, relative + '/' + WIDTH + '-observed.json', rec['OBS'])
            if type(status['J']['valid']) is not bool or status['B0']['valid'] is not True:
                raise ValueError('Method validity is missing or the frozen baseline is invalid')
            for pool, arrays, frozen in (('adapt', fit, obs), ('holdout', held, predictions)):
                raw = _logits(arrays[pool + '_z0'], 'raw ' + pool)
                baseline = _logits(frozen[pool + '_logits_B0'], 'frozen ' + pool)
                if raw.shape != baseline.shape or len(raw) != len(arrays[pool + '_ids']):
                    raise ValueError('Frozen B0 shape disagrees with its input graph IDs')
                np.testing.assert_allclose(_margin(raw), _margin(baseline), rtol=1e-12, atol=1e-12)
                if status['J']['valid'] and _logits(frozen[pool + '_logits_J'], pool + ' J').shape != raw.shape:
                    raise ValueError('Valid J shape disagrees with its graph IDs')
            arms['B0_' + pooling] = predictions['holdout_logits_B0']
            arms['J_' + pooling] = predictions['holdout_logits_J'] if status['J']['valid'] else None
            refit = _refit(root, manifest, rec['INPUT'], pooling)
            source_key = pooling, key[0], key[1], key[3]
            if source_artifacts.setdefault(source_key, refit['artifact_digest']) != refit['artifact_digest']:
                raise ValueError('One source/seed uses different fitted artifacts across target directions')
            sides.append(dict(manifest=manifest, refit=refit,
                              status={k: status['J'].get(k) for k in ('valid', 'status', 'stop_reason')}))
        _match_inputs(sides)
        diagnostics = {pooling: side['refit'] | {'J': side['status']} for pooling, side in zip(('sum', 'mean'), sides)}
        for value in diagnostics.values():
            value.pop('preparation')
        result.append(dict(key=key, relative=relative, ids=held['holdout_ids'], arms=arms, diagnostics=diagnostics))
    return result


def _endpoints(arms, y):
    result = {}
    for name, logits in arms.items():
        if logits is None:
            result[name] = {metric: None for metric in METRICS} | dict(corrected=None, harmed=None, valid=False)
            continue
        baseline = arms['B0_' + name.rsplit('_', 1)[1]]
        score = classification(logits, y, baseline.argmax(axis=1))
        result[name] = dict(NLL=score['sampled_nll'], BA=None if score['balanced_accuracy'] is None
                            else 100 * score['balanced_accuracy'], Accuracy=100 * score['accuracy'],
                            net_corrections=score['net_corrections'], corrected=score['corrected'],
                            harmed=score['harmed'], valid=True)
    return result


def _did(scores):
    return {metric: scores['J_mean'][metric] - scores['B0_mean'][metric]
            - (scores['J_sum'][metric] - scores['B0_sum'][metric])
            if all(scores[arm][metric] is not None for arm in ARMS) else None for metric in METRICS}


def _did_summary(rows, plan):
    output = []
    directions = [tuple(pair) for pair in plan['phases']['P']['directions']]
    for scope, pairs in [('direction', [pair]) for pair in directions] + [('setting', directions)]:
        for metric in METRICS:
            seeds = []
            for seed in plan['real_model_seeds']:
                selected = [r[metric] for r in rows if r['seed'] == seed and (r['source'], r['target']) in pairs]
                complete = len(selected) == len(pairs) and all(v is not None for v in selected)
                seeds.append(dict(seed=seed, estimate=float(np.mean(selected)) if complete else None,
                                  complete_pairs=sum(v is not None for v in selected), expected_pairs=len(pairs)))
            complete = all(item['estimate'] is not None for item in seeds)
            output.append(dict(scope=scope,
                               setting=plan['phases']['P']['setting'],
                               source=pairs[0][0] if scope == 'direction' else None,
                               target=pairs[0][1] if scope == 'direction' else None,
                               metric=metric, unit='nats' if metric == 'NLL' else 'graphs' if metric == 'net_corrections' else 'pp',
                               estimate=float(np.mean([s['estimate'] for s in seeds])) if complete else None,
                               seed_estimates=seeds, complete_pairs=sum(s['complete_pairs'] for s in seeds),
                               expected_pairs=len(pairs) * len(seeds),
                               reason=None if complete else 'incomplete_four_arm_pairs', interval=None))
    return output


def audit(sum_output, mean_output, *, repository=None):
    roots = [Path(p).resolve() for p in (sum_output, mean_output)]
    if roots[0] == roots[1] or any(not (root / 'seals/R7_FINAL.json').is_file() for root in roots):
        raise ValueError('Two distinct FINAL-sealed P branches are required')
    # Complete BOTH chain checks before opening either predictions or raw labels.
    finals = [verify_seal(root, 'R7_FINAL') for root in roots]
    records = [{stage: verify_seal(root, 'R7_' + stage) for stage in ('INPUT', 'OBS')} for root in roots]
    plans = [_read(root, 'executed-plan.json', rec['INPUT']) for root, rec in zip(roots, records)]
    grid = _plans(plans)
    provenance = [_read(root, 'provenance.json', rec['INPUT']) for root, rec in zip(roots, records)]
    if any(provenance[0][key] != provenance[1][key] for key in ('git_commit', 'file_hashes')):
        raise ValueError('Branch execution commits or pinned source files differ')
    for root, rec, plan in zip(roots, records, plans):
        execution = _read(root, 'executed-configs.json', rec['INPUT'])
        if (execution['phase'] != 'P' or execution.get('source_refit_applied') is not True
                or execution.get('source_refit') != plan['source_refit']):
            raise ValueError('Pilot did not execute its registered fresh source branch')
    blocks = _inputs(roots, records, plans, grid)
    # Every block/ID/initialization/prediction check is done before this label boundary.
    cohorts, endpoints, contrasts, did, diagnostics = [], [], [], [], []
    for block in blocks:
        identity = dict(zip(('setting', 'source', 'target', 'seed'), block['key']))
        labels = [load_labels(root, block['relative'] + '/manifest.json', 'holdout', repository=repository) for root in roots]
        np.testing.assert_array_equal(labels[0], labels[1])
        y = labels[0]
        scores = _endpoints(block['arms'], y)
        endpoints += [identity | dict(arm=arm, n=len(y)) | values for arm, values in scores.items()]
        for left, right in COMPARISONS:
            contrasts.append(identity | dict(left=left, right=right,
                             **{metric: scores[left][metric] - scores[right][metric]
                                if scores[left][metric] is not None and scores[right][metric] is not None else None
                                for metric in ('NLL', 'BA', 'Accuracy')}))
        did.append(identity | _did(scores))
        cohorts.append(identity | dict(ids=block['ids'], y=y, logits=block['arms']))
        diagnostics.append(identity | block['diagnostics'])
    paired = paired_bootstrap(plans[1], cohorts, phase='P', comparisons=COMPARISONS, tau=TAU)
    net_intervals = []
    counts = {(row['setting'], row['target']): len(row['ids']) for row in cohorts}
    for row in paired['interval_rows']:
        if (row['scope'] == 'direction' and row['metric'] == 'Accuracy'
                and (row['left'], row['right']) in [('J_sum', 'B0_sum'), ('J_mean', 'B0_mean')]):
            item = deepcopy(row)
            item.update(metric='net_corrections', unit='graphs')
            factor = counts[row['setting'], row['target']] / 100
            for field in ('estimate', 'lower', 'upper', 'seed_sample_sd'):
                item[field] = None if item[field] is None else item[field] * factor
            for seed in item['seed_estimates']:
                seed['estimate'] = None if seed['estimate'] is None else seed['estimate'] * factor
            net_intervals.append(item)
    return dict(format='r7-v10-pooling-pair-audit-v1', tau=TAU,
                scope='fixed-model post-seal diagnostic on development H; no method selection or phase-gate changes; not independent confirmation',
                final_seals={pooling: dict(root=str(root), digest=final['digest'],
                                          plan_sha256=rec['INPUT']['files']['executed-plan.json'])
                             for pooling, root, final, rec in zip(('sum', 'mean'), roots, finals, records)},
                expected_blocks=len(grid), complete_block_identities=True,
                common_execution_commit=provenance[0]['git_commit'], exact_execution_file_hashes_matched=True,
                conventions=dict(differences='left_minus_right', BA_and_Accuracy='percentage_points',
                                 NLL='nats', correction_counts='J corrected/harmed relative to its OWN pooling B0',
                                 did='(J_mean - B0_mean) - (J_sum - B0_sum); point only',
                                 aggregation='equal directions within seed, then equal registered seeds; incomplete pairs remain null',
                                 source_selection='source-validation CE separately per branch; epochs and B0 may differ'),
                block_identity_checks=diagnostics, block_endpoints=endpoints, block_contrasts=contrasts,
                paired_contrasts=[row for row in paired['interval_rows'] if row['metric'] in ('NLL', 'BA', 'Accuracy')],
                adaptation_net_corrections=net_intervals,
                block_difference_in_differences=did, difference_in_differences=_did_summary(did, plans[1]),
                cohort_audit=paired['cohort_audit'], bootstrap=paired['metadata'],
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), numpy_version=np.__version__)


def check_calculations():
    """Generated fixtures only; no research archive or label broker access."""
    from unittest.mock import patch
    plan = dict(development_iteration=10, source_refit=dict(pooling='sum'), real_model_seeds=[1, 2],
                shared_update=dict(primary_tau=TAU),
                phases=dict(P=dict(setting='toy', directions=[[0, 1]], expected_blocks=2, width_grid=[TAU]),
                            D=dict(source_model=dict(pooling='sum'))),
                paired_pooling_diagnostic=dict(primary='mean', reference='sum', same_roles_and_graph_ids=True,
                    contrasts=[list(pair) for pair in COMPARISONS], effect_metrics=list(METRICS),
                    paired_effect='(J_mean - B0_mean) - (J_sum - B0_sum)',
                    source_selection='minimum_source_validation_CE_per_pooling_and_seed',
                    affects_primary_p_gate=False, final_seal_required_in_both_p_branches=True),
                evaluation=dict(cohort_bootstrap_replicates=20, cohort_bootstrap_seed=123,
                    paired_interval=dict(strata=['target_domain', 'class'],
                        reuse_draws_across_methods_seeds_and_source_directions=True, conditional_on_fitted_models=True,
                        minimum_graphs_per_class_for_interval=2, percentiles=[2.5, 97.5], level=.95)))
    mean_plan = deepcopy(plan)
    mean_plan['source_refit']['pooling'] = mean_plan['phases']['D']['source_model']['pooling'] = 'mean'
    assert len(_plans([plan, mean_plan])) == 2
    bad = deepcopy(mean_plan)
    bad['shared_update']['regularization'] = .4
    try:
        _plans([plan, bad])
    except ValueError:
        pass
    else:
        raise AssertionError('Unregistered plan difference accepted')
    side = dict(manifest=dict(ids={'source': [0], 'validation': [1], 'adapt': [2], 'holdout': [3]},
                             provenance=dict(source_refit={}, selected_source_epoch=1), label_records={}),
                refit=dict(initialization='a' * 64, preparation={}))
    other = deepcopy(side)
    other['manifest']['provenance']['selected_source_epoch'] = 2
    _match_inputs([side, other])
    for section, field, value in (('refit', 'initialization', 'b' * 64),
                                  ('manifest', 'ids', {'holdout': [4]})):
        bad = deepcopy(other)
        bad[section][field] = value
        try:
            _match_inputs([side, bad])
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched source initialization/IDs accepted')
    y = np.array([0, 0, 1, 1])
    make = lambda margin: np.column_stack((np.zeros(4), margin))
    arms = dict(B0_sum=make([-2., 1., -1., 2.]), J_sum=make([-2., -1., -1., 2.]),
                B0_mean=make([-1., -2., -2., 1.]), J_mean=make([-1., -2., 2., 1.]))
    scores = _endpoints(arms, y)
    assert scores['J_sum']['net_corrections'] == scores['J_mean']['net_corrections'] == 1
    assert scores['B0_sum']['Accuracy'] == 50 and scores['B0_mean']['Accuracy'] == 75
    assert _did(scores)['Accuracy'] == 0
    blocks = [dict(setting='toy', source=0, target=1, seed=seed, ids=np.arange(4), y=y, logits=arms) for seed in (1, 2)]
    paired = paired_bootstrap(plan, blocks, phase='P', comparisons=COMPARISONS, tau=TAU)
    estimate = next(row['estimate'] for row in paired['interval_rows']
                    if row['scope'] == 'direction' and row['left'] == 'B0_mean' and row['metric'] == 'Accuracy')
    assert estimate == 25
    # Exercise audit's ordering with generated arrays and mocked chain/broker IO.
    # A failing second FINAL must prevent even plan reads; identity failure must
    # prevent raw labels; a successful run must preserve four differing arms.
    events = []
    def fake_verify(root, stage):
        events.append((root.name, stage))
        return dict(digest='a' * 64, files={'executed-plan.json': 'b' * 64})
    def fake_read(root, relative, record):
        selected = plan if root.name == 'sum' else mean_plan
        if relative == 'provenance.json':
            return dict(git_commit='generated', file_hashes={})
        return selected if relative == 'executed-plan.json' else dict(
            phase='P', source_refit_applied=True, source_refit=selected['source_refit'])
    def fake_inputs(*args):
        assert events[:2] == [('sum', 'R7_FINAL'), ('mean', 'R7_FINAL')]
        events.append(('both', 'identities_checked'))
        return [dict(key=('toy', 0, 1, seed), relative=f'blocks/toy-{seed}', ids=np.arange(4),
                     arms=arms, diagnostics={}) for seed in (1, 2)]
    def fake_labels(*args, **kwargs):
        assert ('both', 'identities_checked') in events
        return y
    with patch.object(Path, 'is_file', return_value=True), \
            patch(__name__ + '.verify_seal', side_effect=fake_verify) as verifier, \
            patch(__name__ + '._read', side_effect=fake_read) as reader, \
            patch(__name__ + '._inputs', side_effect=fake_inputs) as inputs, \
            patch(__name__ + '.load_labels', side_effect=fake_labels) as broker:
        report = audit('/generated-only/sum', '/generated-only/mean')
        assert broker.call_count == 4 and len(report['block_endpoints']) == 8
        broker.reset_mock()
        inputs.side_effect = ValueError('generated ID mismatch')
        try:
            audit('/generated-only/sum', '/generated-only/mean')
        except ValueError:
            pass
        else:
            raise AssertionError('Identity mismatch accepted')
        broker.assert_not_called()
        reader.reset_mock()
        verifier.side_effect = [dict(digest='a'), ValueError('generated second FINAL failure')]
        try:
            audit('/generated-only/sum', '/generated-only/mean')
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid second FINAL accepted')
        reader.assert_not_called()
        broker.assert_not_called()
    rows = [dict(seed=seed, source=0, target=1) | _did(scores) for seed in (1, 2)]
    assert all(row['complete_pairs'] == 2 for row in _did_summary(rows, plan))
    arms['J_mean'] = None
    assert all(value is None for value in _did(_endpoints(arms, y)).values())
    rows[1].update(_did(_endpoints(arms, y)))
    assert all(row['estimate'] is None for row in _did_summary(rows, plan))
    return dict(status='passed', distinct_B0_and_epochs_allowed=True, exact_plan_and_initialization_checks=True,
                own_B0_net_corrections=True, paired_bootstrap=True, incomplete_four_arm_pairs_remain_null=True,
                both_FINAL_and_all_identities_before_label_broker=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sum-output', type=Path)
    parser.add_argument('--mean-output', type=Path)
    parser.add_argument('--output', type=Path, help='New JSON report outside both immutable raw P directories')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    paths = (args.sum_output, args.mean_output, args.output)
    if args.check:
        if any(path is not None for path in paths):
            parser.error('--check cannot be combined with research paths')
        print(json.dumps(check_calculations(), allow_nan=False))
        return
    if any(path is None for path in paths):
        parser.error('--sum-output, --mean-output, and --output are required')
    output = args.output.resolve()
    if any(output.is_relative_to(root.resolve()) for root in paths[:2]):
        raise ValueError('Diagnostic output must be outside both immutable raw P directories')
    if output.exists():
        raise FileExistsError(output)
    write_json(output, audit(args.sum_output, args.mean_output))


if __name__ == '__main__':
    main()
