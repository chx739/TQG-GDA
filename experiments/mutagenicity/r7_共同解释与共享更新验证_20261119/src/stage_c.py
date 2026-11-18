"""R7 all-direction development evaluation using the sealed pilot and matched A0/B1."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import permutations, product
import json
import multiprocessing
from pathlib import Path
import traceback

import numpy as np
from scipy.special import softmax
import torch

from experiments.common.io import digest, provenance, save_tensor, write_json

from .decisions import cohort_decisions, decision_contrasts, decision_policy, policy_metadata
from .evaluate import classification, exact_methods, methods, replay_heldout, replay_predictions
from .fit import (assert_density_execution, assert_fitter_sources, config, fit_block,
                  read_npz, width_key, write_npz)
from .matched_tsba import blind_inputs, fit_matched_tsba, predict_matched
from .matched_reuse import open_donor
from .paired import paired_bootstrap
from .r5_inputs import evaluation_labels, load_inputs
from . import source_refit_stage
from .seals import _signature, seal, verify_seal
from .stage_p import _bank_diagnostics, _endpoints, _FIT_ARRAYS, _oracles
from .stage_s import _assert_pinned_method, csv_write, json_read


_INPUT_FILES = ('fit_input.npz', 'holdout_input.npz', 'validation_input.npz', 'manifest.json')
_PRIMARY_CONTRASTS = [('J', 'B0'), ('J', 'A0'), ('J', 'B1'), ('A0', 'B0'), ('B1', 'B0'),
                      ('J', 'E'), ('J', 'P'), ('J', 'P_mass'), ('J', 'J_bias'), ('J', 'J_cal'),
                      ('J', 'O'), ('O', 'B0'), ('J', 'C05'), ('J', 'C075')]
_AUDIT_CONTRASTS = [('J_plus', 'P_plus'), ('J_plus', 'P_mass_plus')]
_EXPANDED_ARMS = {name for pair in _AUDIT_CONTRASTS for name in pair}
_MATCHED_ARMS = ('B1', 'A0')


def _registered_blocks(plan):
    pilot, coverage = (plan['phases'][phase] for phase in ('P', 'C'))
    seeds = plan['real_model_seeds']
    tau = plan['shared_update']['primary_tau']
    pairs = list(permutations(coverage['domains'], 2))
    grid = [(setting, source, target, seed) for setting, (source, target), seed
            in product(coverage['settings'], pairs, seeds)]
    pilot_keys = {(pilot['setting'], source, target, seed) for (source, target), seed
                  in product(pilot['directions'], seeds)}
    repeat = coverage['independent_audit_repeat']
    audit_keys = {(repeat['setting'], source, target, seed) for (source, target), seed
                  in product(repeat['directions'], repeat['seeds'])}
    if (not coverage['enable_only_if_P_passes'] or coverage['directions'] != 'ordered_cross_domain_pairs'
            or tau != coverage['primary_tau'] or tau not in pilot['width_grid']
            or tau != pilot['independent_audit']['tau'] or repeat['inherits'] != 'P.independent_audit'
            or len(pairs) != coverage['directions_per_setting'] or len(pairs) != len(set(pairs))
            or len(grid) != len(set(grid)) or len(grid) != coverage['primary_blocks_total']
            or len(pilot_keys) != pilot['expected_blocks'] or len(pilot_keys) != coverage['reuse_same_R7_P_blocks']
            or len(grid) - len(pilot_keys) != coverage['new_blocks']
            or len(audit_keys) != len(repeat['directions']) * len(repeat['seeds'])
            or not pilot_keys <= set(grid) or not audit_keys <= set(grid) - pilot_keys):
        raise ValueError('The registered pilot reuse or all-direction coverage grid is inconsistent')
    return tau, grid, pilot_keys, audit_keys


def _block_name(task):
    setting, source, target, seed = task
    return f'{setting}-M{source}_to_M{target}-seed-{seed}'


def _shared_observations(tau):
    key = width_key(tau)
    return ('standardization.npz', 'holdout_standardization.npz',
            f'{key}-observed.npz', f'{key}-observed.json', f'{key}-heldout.npz')


def _oracle_files(tau):
    key = width_key(tau)
    return (f'{key}-oracle.npz', f'{key}-oracle.json')


def _sealed_json(root, name, members):
    payload = (root / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != members[name]:
        raise ValueError(f'Sealed pilot JSON changed: {name}')
    return json.loads(payload)


def _pilot_seal_record(root, stage, parent):
    # The old oracle files contain adaptation labels. We can verify their
    # manifest and its position in the chain before opening those payloads.
    record = json_read(root / 'seals' / f'{stage}.json')
    checksum = record.pop('digest')
    if (record['stage'] != stage or record['parent'] != parent or not record['files']
            or _signature(record) != checksum):
        raise ValueError(f'Invalid pilot seal: {stage}')
    return record | {'digest': checksum}


def _verify_pilot(plan_path, plan, p_output, repo, tau, pilot_keys):
    stage_members = {'INPUT': _INPUT_FILES, 'OBS': _shared_observations(tau), 'ORACLE': _oracle_files(tau)}
    observed = verify_seal(p_output, 'R7_OBS')
    records = {'INPUT': verify_seal(p_output, 'R7_INPUT'), 'OBS': observed,
               'ORACLE': _pilot_seal_record(p_output, 'R7_ORACLE', observed['digest'])}
    final = _pilot_seal_record(p_output, 'R7_FINAL', records['ORACLE']['digest'])
    required = {'executed-plan.json', 'executed-configs.json', 'provenance.json'}
    if 'gate.json' not in final['files'] or not required <= records['INPUT']['files'].keys():
        raise ValueError('Pilot verdict or configuration is outside the verified seals')
    gate = _sealed_json(p_output, 'gate.json', final['files'])
    execution = _sealed_json(p_output, 'executed-configs.json', records['INPUT']['files'])
    assert_density_execution(plan, execution)
    if plan.get('source_refit') and execution.get('source_refit_applied') is not True:
        raise ValueError('The sealed pilot did not use the registered source refit')
    if gate['passed'] is not True or gate['next_phase'] != 'C' or execution['phase'] != 'P':
        raise ValueError('The sealed P decision did not authorize all-direction evaluation')
    if digest(plan_path) != records['INPUT']['files']['executed-plan.json']:
        raise ValueError('C must use the identical plan and parameters from the passing pilot')
    policy = decision_policy(plan)
    if (execution['separation_variant'] != plan.get('separation_variant', 'stock')
            or execution['independent_audit'] != plan['phases']['P']['independent_audit']
            or any(config(plan, seed, tau).to_dict() not in execution['common']
                   for seed in plan['real_model_seeds'])):
        raise ValueError('The sealed pilot used different primary fitting or audit parameters')
    frozen = _sealed_json(p_output, 'provenance.json', records['INPUT']['files'])['file_hashes']
    additional = ['evaluate.py', 'real_audit.py', 'separation_audit.py', 'r5_inputs.py']
    if 'matched_tsba_reuse' in plan:
        additional.extend(['stage_c.py', 'matched_reuse.py'])
    if policy is not None:
        additional.extend(['decisions.py', 'paired.py', 'stage_p.py', 'stage_c.py'])
    assert_fitter_sources(plan, repo, frozen, additional=additional)
    for task in pilot_keys:
        block = Path('blocks') / _block_name(task)
        for stage, members in stage_members.items():
            for member in members:
                if str(block / member) not in records[stage]['files']:
                    raise ValueError(f'Missing sealed pilot artifact: {block / member}')
    return final['digest'], {name: record['files'] for name, record in records.items()}


def _copy_verified(root, p_output, block, names, members):
    for name in names:
        relative = block.relative_to(root) / name
        payload = (p_output / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != members[str(relative)]:
            raise ValueError(f'Pilot evidence changed before copying: {relative}')
        with (block / name).open('xb') as stream:
            stream.write(payload)


def _input_arrays(loaded):
    return {'fit_input.npz': {key: loaded[key] for key in _FIT_ARRAYS}} | {
        f'{pool}_input.npz': {f'{pool}_{key}': loaded[f'{pool}_{key}'] for key in ('h', 'z0', 'ids')}
        for pool in ('holdout', 'validation')
    }


def _prepare_block(plan, repo, root, block, task, reused_input=None, *, refit_root=None):
    loaded = (load_inputs(plan, *task, repository=repo, include_tsba=True) if refit_root is None
              else source_refit_stage.load(plan, task, refit_root, repo, include_tsba=True))
    block.mkdir()
    inputs = _input_arrays(loaded)
    if reused_input is None:
        for name, values in inputs.items():
            write_npz(block / name, **values)
        write_json(block / 'manifest.json', loaded['manifest'])
    else:
        p_output, members = reused_input
        _copy_verified(root, p_output, block, _INPUT_FILES, members)
        if loaded['manifest'] != json_read(block / 'manifest.json'):
            raise ValueError(f'{block.name}: pilot and matched reference have different source or fold identity')
        for name, values in inputs.items():
            saved = read_npz(block / name)
            if set(saved) != set(values):
                raise ValueError(f'{block.name}: pilot input layout differs in {name}')
            for key, value in values.items():
                np.testing.assert_array_equal(saved[key], value, err_msg=f'{block.name}/{name}/{key}')
    blind = {key: torch.from_numpy(np.array(value, copy=True)) if isinstance(value, np.ndarray) else value
             for key, value in blind_inputs(loaded).items()}
    save_tensor(block / 'matched_input.pt', blind)


def _fit_view(plan, tau):
    return dict(shared_update=plan['shared_update'], controls=plan['controls'],
                separation_variant=plan.get('separation_variant', 'stock'),
                source_mixture_options=plan.get('source_mixture_options', {}),
                density_coordinates=plan.get('density_coordinates', 'source_standardized_v1'),
                target_posterior_power=plan.get('target_posterior_power', 1.),
                posterior_mass_options=plan.get('posterior_mass_options', {}), phases={
                    'S': {'width_grid': [tau]},
                    'P': {'width_grid': [tau], 'independent_audit': plan['phases']['P']['independent_audit']},
                    'C': plan['phases']['C'],
                }, real_model_seeds=plan['real_model_seeds'], **policy_metadata(plan))


def _fit_blind(view, block, seed, reused, audited, reuse_matched=False):
    torch.set_num_threads(1)
    if not reused:
        # The existing fitter's P branch adds the registered audit; its S branch
        # is the same one-width fit without the extra queries.
        fit_block(view, seed, block, phase='P' if audited else 'S')
    if not reuse_matched:
        inputs = torch.load(block / 'matched_input.pt', map_location='cpu', weights_only=True)
        fit_matched_tsba(view, inputs, seed, output_directory=block / 'matched')


def _fit_all(view, root, tasks, workers, pilot_keys, audit_keys, *, reuse_matched=False):
    verify_seal(root, 'R7_INPUT')
    errors = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        waiting = {pool.submit(_fit_blind, view, block, task[-1], task in pilot_keys, task in audit_keys,
                               reuse_matched): block.name
                   for block, task in tasks}
        for future in as_completed(waiting):
            name = waiting[future]
            try:
                future.result()
                print(f'C fit frozen {name}', flush=True)
            except Exception:
                errors.append(dict(block=name, traceback=traceback.format_exc()))
    if errors:
        write_json(root / 'fit_errors.json', errors)
        raise RuntimeError(f'{len(errors)} C block(s) failed before observations were sealed')


def _matched_observations(view, block):
    budget = view['phases']['C']['matched_tsba']
    complete = json_read(block / 'matched' / 'complete.json')
    attempts = budget['attempts_per_group']
    if (complete['valid'] is not True or complete['updates'] != budget['score_exact_updates']
            or set(complete['arms']) != set(_MATCHED_ARMS)
            or any(arm['attempts'] != attempts for arm in complete['arms'].values())):
        raise ValueError(f'{block.name}: matched baselines did not complete the registered budget')
    result = torch.load(block / 'matched' / 'predictor.pt', map_location='cpu', weights_only=True)
    fit, holdout, validation = (read_npz(block / name) for name in _INPUT_FILES[:3])
    predictions = {}
    for pool, raw in [('source_fit', fit['source_h']), ('source_validation', validation['validation_h']),
                      ('adapt', fit['adapt_h']), ('holdout', holdout['holdout_h'])]:
        for name, logits in predict_matched(result, raw).items():
            if pool == 'adapt':
                np.testing.assert_array_equal(logits, np.asarray(result['arms'][name]['adapt_logits']),
                                              err_msg=f'{block.name}/{name}: matched A prediction replay')
            predictions[f'{pool}_logits_{name}'] = logits
    if set(predictions) != {f'{pool}_logits_{name}' for pool, name in product(
            ('source_fit', 'source_validation', 'adapt', 'holdout'), _MATCHED_ARMS)}:
        raise ValueError(f'{block.name}: incomplete matched predictions')
    return predictions, result


def _matched_files(attempts):
    names = ['matched/metadata.json', 'matched/schedule.pt', 'matched/score.pt', 'matched/predictor.pt',
             'matched/complete.json', 'matched-predictions.npz']
    return names + [f'matched/{name}/round{iteration:02d}.pt'
                    for name, iteration in product(_MATCHED_ARMS, range(attempts + 1))]


def _freeze_observations(plan, view, root, p_output, tasks, tau, pilot_keys, audit_keys, pilot_members, *, donor=None):
    files = []
    key = width_key(tau)
    matched_files = _matched_files(view['phases']['C']['matched_tsba']['attempts_per_group'])
    for block, task in tasks:
        reused = task in pilot_keys
        if reused:
            _copy_verified(root, p_output, block, _shared_observations(tau), pilot_members)
        else:
            replay_heldout(view, block, phase='P' if task in audit_keys else 'S')
        expected = exact_methods(plan, tau)
        if reused or task in audit_keys:
            expected |= _EXPANDED_ARMS
        records = json_read(block / f'{key}-observed.json')
        arrays = read_npz(block / f'{key}-observed.npz')
        if (set(records) != expected or set(methods(arrays)) != expected
                or any(type(record['valid']) is not bool for record in records.values())):
            raise ValueError(f'{block.name}: incomplete primary shared-update arms')
        if donor is not None:
            donor.copy_observations(block, block.relative_to(root), matched_files)
        predictions, state = _matched_observations(view, block)
        if donor is None:
            write_npz(block / 'matched-predictions.npz', **predictions)
        else:
            donor.verify_replay(block, predictions, state)
        names = list(_shared_observations(tau)) + matched_files
        files.extend(str(block.relative_to(root) / name) for name in names)
    return files


def _freeze_oracles(view, root, repo, tasks, p_output, tau, pilot_keys, pilot_members):
    files = []
    with evaluation_labels(root, repository=repo, stage='R7_OBS') as label_reader:
        def read_adaptation_labels(_root, manifest, pool, *, repository):
            return label_reader.load(manifest, pool)

        for block, task in tasks:
            if task in pilot_keys:
                _copy_verified(root, p_output, block, _oracle_files(tau), pilot_members)
            else:
                _oracles(view, root, block, task[-1], repo, label_reader=read_adaptation_labels)
            files.extend(str(block.relative_to(root) / name) for name in _oracle_files(tau))
    return files


def _matched_endpoints(record, name, key, logits, labels, reference):
    rows = []
    for pool, y in labels.items():
        z, z0 = logits[f'{pool}_logits_{name}'], reference[pool]
        delta = z - z0
        delta -= delta.mean(axis=1, keepdims=True)
        rows.append(dict(**key, pool=pool, method=name, valid=record['valid'], status=record['status'],
                         stop_reason=record['stop_reason'], centered_rms=float(np.sqrt(np.mean(delta ** 2))),
                         mean_predicted_p1=float(softmax(z, axis=1)[:, 1].mean()),
                         **classification(z, y, z0.argmax(axis=1))))
    return rows


def _evaluate(plan, view, root, repo, tasks, tau, pilot_keys):
    with evaluation_labels(root, repository=repo) as label_reader:
        tables = _evaluate_frozen(plan, view, root, tasks, tau, pilot_keys, label_reader)
    return tables


def _evaluate_frozen(plan, view, root, tasks, tau, pilot_keys, label_reader):
    key = width_key(tau)
    endpoints, moments, coverage, separation, cohorts = [], [], [], [], []
    for block, task in tasks:
        setting, source, target, seed = task
        metadata = dict(setting=setting, source=source, target=target, seed=seed, tau=tau,
                        block_origin='reused_P' if task in pilot_keys else 'new_C')
        manifest = str((block / 'manifest.json').relative_to(root))
        a_y = label_reader.load(manifest, 'adapt', authenticated_snapshot=True)
        h_y = label_reader.load(manifest, 'holdout')
        v_y = label_reader.load(manifest, 'validation')
        fit, std, held = (read_npz(block / name) for name in
                          ('fit_input.npz', 'standardization.npz', 'holdout_standardization.npz'))
        source_pairs = read_npz(block / 'validation_input.npz') | {'source_z0': fit['source_z0']}
        observed, heldout, oracle = (read_npz(block / f'{key}-{kind}.npz')
                                     for kind in ('observed', 'heldout', 'oracle'))
        np.testing.assert_array_equal(oracle['adapt_labels'], a_y, err_msg=f'{block.name}: oracle labels changed')
        for inputs, saved in ((observed, heldout), (oracle, oracle)):
            for name, replay in replay_predictions(std, held, inputs).items():
                np.testing.assert_array_equal(replay, saved[name], err_msg=f'{block.name}/{name}')
        arrays = observed | heldout | oracle
        labels = dict(adapt=a_y, holdout=h_y, source_fit=fit['source_y'], source_validation=v_y)
        references = {pool: arrays[f'{pool}_logits_B0'] for pool in ('adapt', 'holdout')}
        records = json_read(block / f'{key}-observed.json') | {'O': json_read(block / f'{key}-oracle.json')}
        for name, record in records.items():
            endpoints.extend(_endpoints(name, record, metadata, std, arrays, labels, source_pairs, references))
            separation.append(dict(**metadata, method=name, valid=record['valid'], stop_reason=record['stop_reason'],
                                   known_worst_risk=record.get('found_worst_risk'),
                                   exact_separation_violation=record.get('exact_separation_violation'),
                                   independent_audit=record.get('independent_audit')))
        holdout_logits = {name: arrays[f'holdout_logits_{name}'] if record['valid'] else None
                          for name, record in records.items()}
        matched, state = _matched_observations(view, block)
        saved = read_npz(block / 'matched-predictions.npz')
        if set(saved) != set(matched):
            raise ValueError(f'{block.name}: sealed matched predictions have unexpected members')
        for name, prediction in matched.items():
            np.testing.assert_array_equal(prediction, saved[name], err_msg=f'{block.name}/{name}')
        baseline = references | {'source_fit': fit['source_z0'], 'source_validation': source_pairs['validation_z0']}
        for name, arm in state['arms'].items():
            endpoints.extend(_matched_endpoints(arm, name, metadata, matched, labels, baseline))
            holdout_logits[name] = matched[f'holdout_logits_{name}']
        block_moments, fit_quality, original_q, expanded_q = _bank_diagnostics(
            plan, config(plan, seed, tau), fit, std, held, arrays, a_y, h_y)
        moments.extend(metadata | row for row in block_moments)
        coverage.append(metadata | fit_quality)
        cohort = dict(**metadata, ids=held['holdout_ids'], y=h_y, logits=holdout_logits,
                      candidate_posteriors=original_q, expanded_candidate_posteriors=expanded_q)
        actions = cohort_decisions(arrays, records)
        if actions:
            cohort['decisions'] = actions
        cohorts.append(cohort)
    return endpoints, moments, coverage, separation, cohorts


def _pair_subset(plan, cohorts, keys, comparisons, tau):
    subset = [row for row in cohorts if tuple(row[name] for name in ('setting', 'source', 'target', 'seed')) in keys]
    reports = {}
    for setting in plan['phases']['C']['settings']:
        blocks = [row for row in subset if row['setting'] == setting]
        if not blocks:
            continue
        directions = sorted({(source, target) for name, source, target, _ in keys if name == setting})
        seeds = sorted({seed for name, _, _, seed in keys if name == setting})
        # A single-setting direction grid uses the paired module's explicit-grid
        # interface. Its resampling streams are keyed by setting and target.
        local = plan | dict(real_model_seeds=seeds, phases={'P': {'setting': setting, 'directions': directions}})
        report = paired_bootstrap(local, blocks, phase='P', comparisons=comparisons, tau=tau)
        reports[setting] = report | {'phase': 'C'}
    return reports


def _seed_outcomes(rows, tie):
    for row in rows:
        scores = [item['estimate'] for item in row['seed_estimates']]
        values = np.array([value for value in scores if value is not None])
        if row['metric'] in ('NLL', 'BA', 'Accuracy'):
            signed = -values if row['metric'] == 'NLL' else values
            tolerance = tie if row['metric'] == 'NLL' else 1e-12
            row.update(seed_wins=int(sum(signed > tolerance)), seed_losses=int(sum(signed < -tolerance)),
                       seed_ties=int(sum(np.abs(signed) <= tolerance)), seed_missing=len(scores) - len(values))
    return rows


def _block_contrasts(endpoints, contrasts):
    grouped = {}
    fields = ('setting', 'source', 'target', 'seed', 'tau', 'block_origin')
    for row in endpoints:
        if row['pool'] == 'holdout':
            key = tuple(row[field] for field in fields)
            grouped.setdefault(key, {})[row['method']] = row
    results = []
    for key, arms in grouped.items():
        for left, right in contrasts:
            if left not in arms or right not in arms:
                continue
            a, b = arms[left], arms[right]
            delta = {}
            if a['valid'] and b['valid']:
                for metric, unit, output in [('sampled_nll', 1, 'nll'), ('balanced_accuracy', 100, 'ba_pp'),
                                             ('accuracy', 100, 'accuracy_pp'), ('net_corrections', 1, 'net_corrections')]:
                    delta[output] = (unit * (a[metric] - b[metric])
                                     if a[metric] is not None and b[metric] is not None else None)
            results.append(dict(zip(fields, key)) | dict(left=left, right=right, valid=bool(delta), **delta))
    return results


def _coverage_gate(plan, rows):
    settings = plan['phases']['C']['settings']
    criteria = plan['phases']['C']['classification_criteria']
    datasets = {item['id']: item['tu_dataset'] for item in plan['real_settings'] if item['id'] in settings}
    stats = {(r['setting'], r['left'], r['right'], r['metric']): r for r in rows if r['scope'] == 'setting'}
    decisions = {}
    for left, right in criteria['comparisons_evaluated_separately']:
        summary = {}
        for setting in settings:
            nll, ba = [stats[(setting, left, right, metric)] for metric in ('NLL', 'BA')]
            summary[setting] = dict(dataset=datasets[setting], nll_delta=nll['estimate'], ba_pp_delta=ba['estimate'],
                                    complete=all(row['complete_pairs'] == row['expected_pairs'] for row in (nll, ba)),
                                    nll_ci=[nll['lower'], nll['upper']], ba_pp_ci=[ba['lower'], ba['upper']],
                                    favorable_intervals=(nll['upper'] is not None and nll['upper'] < 0
                                                         and ba['lower'] is not None and ba['lower'] > 0))
        complete = all(row['complete'] for row in summary.values())
        minimum_ba = criteria['minimum_mean_ba_gain_for_PROTEINS_and_one_Mutagenicity_split_pp']
        gains = {row['dataset'] for row in summary.values()
                 if row['ba_pp_delta'] is not None and row['ba_pp_delta'] >= minimum_ba}
        nll_reduced = not criteria['average_nll_improves_each_setting'] or all(
            row['nll_delta'] is not None and row['nll_delta'] < -plan['evaluation']['nll_tie_tolerance']
            for row in summary.values())
        ba_non_decreasing = all(row['ba_pp_delta'] is not None and
                                row['ba_pp_delta'] >= criteria['minimum_mean_ba_gain_each_setting_pp']
                                for row in summary.values())
        favorable_datasets = sorted({row['dataset'] for row in summary.values() if row['favorable_intervals']})
        point_pass = complete and nll_reduced and ba_non_decreasing and {'PROTEINS', 'Mutagenicity'} <= gains
        intervals_pass = (not criteria['require_complete_favorable_intervals_for_clear_cross_dataset_development_claim']
                          or {'PROTEINS', 'Mutagenicity'} <= set(favorable_datasets))
        decisions[f'{left}-{right}'] = dict(complete_primary_pairs=complete, point_estimate_pass=point_pass,
                                         clear_development_replication=point_pass and intervals_pass,
                                         favorable_interval_datasets=favorable_datasets, settings=summary)
    allow_d = decisions['J-B0']['point_estimate_pass'] and decisions['J-B0']['complete_primary_pairs']
    return dict(passed=allow_d, next_phase='D' if allow_d else None, contrasts=decisions,
                scope='R7 development evaluation on two datasets; relations are conditional on fixed models')


def run(plan_path, root, p_output, workers, repo, *, matched_from=None):
    """Run C only from a verified P pass, including exact pilot-artifact reuse."""
    plan_path, root, p_output, repo = (Path(path).resolve() for path in (plan_path, root, p_output, repo))
    plan = json_read(plan_path)
    tau, grid, pilot_keys, audit_keys = _registered_blocks(plan)
    prior, pilot_members = _verify_pilot(plan_path, plan, p_output, repo, tau, pilot_keys)
    if not 1 <= workers <= plan['execution']['maximum_workers']:
        raise ValueError('Worker count exceeds the registered CPU budget')
    _assert_pinned_method(plan, repo)
    torch.set_num_threads(1)
    matched_files = _matched_files(plan['phases']['C']['matched_tsba']['attempts_per_group'])
    donor = open_donor(plan, matched_from, repo,
                       [(str(Path('blocks') / _block_name(task)), task) for task in grid], matched_files)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'blocks').mkdir()
    tasks = [(root / 'blocks' / _block_name(task), task) for task in grid]
    prior_sources = {key: p_output / 'source-refit' for key in source_refit_stage.source_keys(pilot_keys)}
    source_roots = source_refit_stage.prepare(plan, grid, root, workers, repo, reuse=prior_sources)
    view = _fit_view(plan, tau)
    input_names = ['executed-plan.json', 'executed-configs.json', 'provenance.json', 'blocks.json']
    (root / 'executed-plan.json').write_bytes(plan_path.read_bytes())
    execution = dict(phase='C', primary_tau=tau, p_final_digest=prior, p_output=str(p_output),
                     workers=workers, threads_per_worker=1, separation_variant=view['separation_variant'],
                     density_coordinates=view['density_coordinates'], target_posterior_power=view['target_posterior_power'],
                     target_posterior_power_effective=view['target_posterior_power'], source_refit=plan.get('source_refit'),
                     source_refit_applied=bool(source_roots),
                     common=[config(plan, seed, tau).to_dict() for seed in plan['real_model_seeds']],
                     matched_tsba=view['phases']['C']['matched_tsba'], audit=view['phases']['P']['independent_audit'],
                     **policy_metadata(view))
    if donor is not None:
        execution['matched_tsba_reuse'] = donor.lock
    write_json(root / 'executed-configs.json', execution)
    write_json(root / 'provenance.json', provenance(repo))
    write_json(root / 'blocks.json', [dict(path=str(block.relative_to(root)), setting=task[0], source=task[1],
                                         target=task[2], seed=task[3], reused_p=task in pilot_keys,
                                         repeat_independent_audit=task in audit_keys) for block, task in tasks])
    for block, task in tasks:
        reused_input = (p_output, pilot_members['INPUT']) if task in pilot_keys else None
        if source_roots:
            setting, source, _, seed = task
            _prepare_block(plan, repo, root, block, task, reused_input,
                           refit_root=source_roots[setting, source, seed])
        else:
            _prepare_block(plan, repo, root, block, task, reused_input)
        if donor is not None:
            donor.verify_inputs(block, block.relative_to(root), task, view)
        input_names.extend(str(block.relative_to(root) / name) for name in (*_INPUT_FILES, 'matched_input.pt'))
    input_names += source_refit_stage.local_files(root)
    seal(root, 'R7_INPUT', input_names, expected=input_names)
    _fit_all(view, root, tasks, workers, pilot_keys, audit_keys, reuse_matched=donor is not None)
    observed_names = _freeze_observations(plan, view, root, p_output, tasks, tau, pilot_keys, audit_keys,
                                          pilot_members['OBS'], donor=donor)
    seal(root, 'R7_OBS', observed_names, expected=observed_names)
    oracle_names = _freeze_oracles(view, root, repo, tasks, p_output, tau, pilot_keys, pilot_members['ORACLE'])
    seal(root, 'R7_ORACLE', oracle_names, expected=oracle_names)
    if verify_seal(p_output, 'R7_FINAL')['digest'] != prior:
        raise ValueError('Pilot evidence changed after C verified its verdict')
    endpoints, moments, coverage, separation, cohorts = _evaluate(plan, view, root, repo, tasks, tau, pilot_keys)
    contrasts = decision_contrasts(plan, _PRIMARY_CONTRASTS)
    paired = paired_bootstrap(plan, cohorts, phase='C', comparisons=contrasts, tau=tau)
    new = _pair_subset(plan, cohorts, set(grid) - pilot_keys, contrasts, tau)
    audit = _pair_subset(plan, cohorts, pilot_keys | audit_keys, _AUDIT_CONTRASTS, tau)
    tie = plan['evaluation']['nll_tie_tolerance']
    paired_rows = _seed_outcomes(paired['interval_rows'], tie)
    new_rows = _seed_outcomes([row for report in new.values() for row in report['interval_rows']], tie)
    audit_rows = _seed_outcomes([row for report in audit.values() for row in report['interval_rows']], tie)
    tables = dict(endpoints=endpoints, moments=moments, coverage=coverage, separation=separation,
                  paired=paired_rows, paired_new=new_rows, paired_audit=audit_rows,
                  block_contrasts=_block_contrasts(endpoints, [*contrasts, *_AUDIT_CONTRASTS]))
    final_names = []
    for name, rows in tables.items():
        csv_write(root / f'{name}.csv', rows)
        final_names.append(f'{name}.csv')
    write_json(root / 'paired-audit.json', dict(all_blocks=paired, new_blocks=new, expanded_banks=audit))
    gate = _coverage_gate(plan, paired_rows)
    gate.update(reused_p_blocks=len(pilot_keys), new_c_blocks=len(grid) - len(pilot_keys),
                repeated_audit_blocks=len(audit_keys))
    write_json(root / 'gate.json', gate)
    final_names += ['gate.json', 'paired-audit.json']
    seal(root, 'R7_FINAL', final_names, expected=final_names)
    verify_seal(root, 'R7_FINAL')
    print(gate, flush=True)
    return gate
