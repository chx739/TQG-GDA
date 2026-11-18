"""External NCI1 confirmation, authorized exclusively by a sealed C pass."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from itertools import product
import json
import multiprocessing
from pathlib import Path
import traceback

import numpy as np
from scipy.special import softmax
import torch

from experiments.common.io import digest, provenance, write_json

from . import nci1_inputs
from .evaluate import exact_methods, methods, replay_predictions
from .fit import config, fitter_source_files, master, moment, predict, read_npz, width_key, write_npz
from .paired import paired_bootstrap
from .seals import seal, verify_seal
from .stage_c import _block_name, _input_arrays, _oracle_files, _seed_outcomes, _shared_observations
from .stage_p import _bank_diagnostics, _endpoints, _source_template_diagnostics
from .stage_s import _assert_pinned_method, _fit, _leaf_paths, csv_write, json_read


_CONTRASTS = [('J', 'B0'), ('J', 'P'), ('J', 'E'), ('J', 'P_mass'), ('J', 'J_bias'), ('J', 'J_cal'),
              ('J', 'O'), ('O', 'B0'), ('J', 'C05'), ('J', 'C075')]


def _locked_files(plan, repo):
    local = Path(__file__).resolve().parent
    module_names = [*fitter_source_files(plan), 'stage_d.py', 'nci1_inputs.py', 'r5_inputs.py',
                    'stage_s.py', 'stage_p.py', 'stage_c.py', 'evaluate.py', 'coverage.py',
                    'paired.py', 'seals.py']
    if 'pooling' in plan['phases']['D']['source_model']:
        module_names.append('source_refit_inputs.py')
    files = [local / name for name in module_names] + [local.parent / 'scripts' / 'run.py',
             local.parent / 'docs' / 'r7-protocol.md']
    files.extend(repo / name for name in nci1_inputs.SOURCE_FIT_FILES)
    if plan.get('source_refit') is not None:
        from .source_refit_inputs import _DEPENDENCIES

        files.extend(repo / name for name in _DEPENDENCIES)
    files.extend((repo / 'tsba' / 'shared_update').glob('*.py'))
    return {str(path.relative_to(repo)): digest(path) for path in sorted(set(files))}


def _registered_blocks(plan):
    external = nci1_inputs._registered(plan)
    phase = plan['phases']['D']
    tau = phase['primary_tau']
    if (tau != plan['shared_update']['primary_tau'] or tau != plan['evaluation']['primary_tau']
            or tau != plan['phases']['C']['primary_tau'] or plan['evaluation']['primary_pool'] != 'holdout'):
        raise ValueError('D endpoints must use the primary width and holdout pool fixed in C')
    tasks = [(external['id'], source, target, seed) for (source, target), seed
             in product(phase['directions'], plan['real_model_seeds'])]
    if len(tasks) != len(set(tasks)) or len(tasks) != phase['confirmation']['complete_blocks']:
        raise ValueError('D confirmation requires the complete registered direction and seed grid')
    return tau, tasks


def _train_sources(plan, root, prepared, workers):
    errors = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {pool.submit(nci1_inputs.train_source, plan, prepared, seed): seed
                for seed in plan['real_model_seeds']}
        for future in as_completed(jobs):
            seed = jobs[future]
            try:
                future.result()
                print(f'D source frozen M0 seed {seed}', flush=True)
            except Exception:
                errors.append(dict(seed=seed, traceback=traceback.format_exc()))
    if errors:
        write_json(root / 'source_fit_errors.json', errors)
        raise RuntimeError(f'{len(errors)} D source checkpoint(s) failed before target fitting')


def _source_members(plan, prepared, root):
    preparation = json_read(prepared / 'prepared.json')
    seeds = plan['real_model_seeds']
    role_names = ('roles/seed{seed}.json', 'vocabulary/seed{seed}.json',
                  'source_labels/seed{seed}_fit.pt', 'source_labels/seed{seed}_validation.pt')
    prepared_names = {'topology.pt', 'domains.json', 'graph_sizes.json', 'label_records.json'}
    prepared_names.update(name.format(seed=seed) for name, seed in product(role_names, seeds))
    if set(preparation['files']) != prepared_names:
        raise ValueError('Incomplete frozen D source input archive')
    files = [prepared / 'prepared.json', *(prepared / name for name in preparation['files'])]
    expected = {'source.pt', 'frozen.pt'} | {f'epochs/epoch{i:03d}.pt'
                                             for i in range(plan['phases']['D']['source_model']['source_epochs'])}
    for seed in seeds:
        source = prepared / 'source' / f'seed{seed}'
        complete = json_read(source / 'complete.json')
        if set(complete['files']) != expected:
            raise ValueError(f'Incomplete source training archive for D seed {seed}')
        # The epoch granularity is recorded in complete.json. Verify the full
        # archive once; only selected state/cache and the index enter the seal
        # read on every label request.
        for name, checksum in complete['files'].items():
            if digest(source / name) != checksum:
                raise ValueError(f'Changed D source training artifact: {source / name}')
        files.extend(source / name for name in ('complete.json', 'source.pt', 'frozen.pt'))
    return [str(path.relative_to(root)) for path in files]


def _initialize(plan, repo, block, task, prepared):
    loaded = nci1_inputs.load_inputs(plan, *task, prepared_root=prepared, repository=repo)
    block.mkdir()
    for name, arrays in _input_arrays(loaded).items():
        write_npz(block / name, **arrays)
    write_json(block / 'manifest.json', loaded['manifest'])


def _fit_oracle(plan, root, block, repo, seed, tau):
    rel = str((block / 'manifest.json').relative_to(root))
    labels = nci1_inputs.load_labels(root, rel, 'adapt', repository=repo)
    std = read_npz(block / 'standardization.npz')
    held = read_npz(block / 'holdout_standardization.npz')
    z, u = std['adapt_z0'], std['adapt_u']
    statistic = moment(np.eye(z.shape[1])[labels], u)
    w, record = master(z, u, [statistic], config(plan, seed, tau))
    prediction = predict('O', z, u, w)
    arrays = dict(w_O=w, oracle_stat=statistic, adapt_labels=labels, adapt_logits_O=prediction,
                  adapt_probability_O=softmax(prediction, axis=1))
    arrays.update(replay_predictions(std, held, arrays))
    key = width_key(tau)
    write_npz(block / f'{key}-oracle.npz', **arrays)
    write_json(block / f'{key}-oracle.json', record)


def _evaluate(plan, root, repo, tasks, tau):
    verify_seal(root, 'R7_ORACLE')
    endpoints, moments, coverage, separation, cohorts = [], [], [], [], []
    key = width_key(tau)
    for block, (setting, source, target, seed) in tasks:
        metadata = dict(setting=setting, source=source, target=target, seed=seed, tau=tau)
        rel = str((block / 'manifest.json').relative_to(root))
        a_y = nci1_inputs.load_labels(root, rel, 'adapt', repository=repo, authenticated_snapshot=True)
        h_y = nci1_inputs.load_labels(root, rel, 'holdout', repository=repo)
        v_y = nci1_inputs.load_labels(root, rel, 'validation', repository=repo)
        fit, std, held = (read_npz(block / name) for name in
                          ('fit_input.npz', 'standardization.npz', 'holdout_standardization.npz'))
        validation = read_npz(block / 'validation_input.npz')
        source_pairs = validation | {'source_z0': fit['source_z0']}
        observed, heldout, oracle = (read_npz(block / f'{key}-{name}.npz') for name in ('observed', 'heldout', 'oracle'))
        np.testing.assert_array_equal(oracle['adapt_labels'], a_y,
                                      err_msg=f'{block.name}: oracle labels changed in the authenticated snapshot')
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
                                   exact_separation_violation=record.get('exact_separation_violation')))
        cfg = config(plan, seed, tau)
        block_moments, fit_quality, posterior, _ = _bank_diagnostics(plan, cfg, fit, std, held, arrays, a_y, h_y)
        fit_quality['source_template_validation'] = _source_template_diagnostics(
            plan, cfg, fit, std, validation, v_y, arrays['ref_theta'])
        moments.extend(metadata | row for row in block_moments)
        coverage.append(metadata | fit_quality)
        cohorts.append(dict(**metadata, ids=held['holdout_ids'], y=h_y, candidate_posteriors=posterior,
                            logits={name: arrays[f'holdout_logits_{name}'] if record['valid'] else None
                                    for name, record in records.items()}))
    return endpoints, moments, coverage, separation, cohorts


def _confirmation(plan, rows):
    phase = plan['phases']['D']
    limits = phase['confirmation']
    by_key = {(row['source'], row['target'], row['metric']): row for row in rows
              if row['setting'] == phase['setting'] and (row['left'], row['right']) == ('J', 'B0')
              and row['metric'] in ('NLL', 'BA')}

    def summary(source, target, expected):
        nll = by_key[(source, target, 'NLL')]
        ba = by_key[(source, target, 'BA')]
        complete = all(row['complete_pairs'] == row['expected_pairs'] == expected and row['estimate'] is not None
                       for row in (nll, ba))
        return dict(nll_delta=nll['estimate'], ba_pp_delta=ba['estimate'], complete=complete,
                    nll_ci=[nll['lower'], nll['upper']], ba_pp_ci=[ba['lower'], ba['upper']],
                    expected_pairs=expected, nll_complete_pairs=nll['complete_pairs'], ba_complete_pairs=ba['complete_pairs'])

    overall = summary(None, None, limits['complete_blocks'])
    directions = {f'M{source}->M{target}': summary(source, target, len(plan['real_model_seeds']))
                  for source, target in phase['directions']}
    complete = overall['complete'] and all(row['complete'] for row in directions.values())
    nll, ba = overall['nll_delta'], overall['ba_pp_delta']
    probability_effect = nll is not None and nll <= -limits['minimum_mean_nll_decrease_vs_B0']
    classification_effect = ba is not None and ba >= limits['minimum_mean_ba_gain_vs_B0_pp']
    nll_lower, nll_upper = overall['nll_ci']
    ba_lower, ba_upper = overall['ba_pp_ci']
    intervals_favorable = nll_upper is not None and nll_upper < 0 and ba_lower is not None and ba_lower > 0
    directions_ba_positive = sum(row['ba_pp_delta'] is not None and row['ba_pp_delta'] > 0
                                 for row in directions.values())
    direction_limits = complete and all(row['nll_delta'] <= limits['maximum_nll_increase_any_direction']
                                        and row['ba_pp_delta'] >= -limits['maximum_ba_decrease_any_direction_pp']
                                        for row in directions.values())
    passed = (complete and probability_effect and classification_effect and direction_limits
              and directions_ba_positive >= limits['minimum_directions_with_positive_mean_ba']
              and (intervals_favorable or not limits['require_favorable_paired_intervals_for_both']))
    if passed:
        assessment = 'confirmed'
    elif not complete:
        assessment = 'incomplete'
    elif (nll_lower is not None and nll_lower > 0) or (ba_upper is not None and ba_upper < 0):
        assessment = 'adverse_effect'
    elif probability_effect and nll_upper is not None and nll_upper < 0 and not classification_effect:
        assessment = 'probability_quality_only'
    elif ba is not None and 0 <= ba < limits['minimum_mean_ba_gain_vs_B0_pp']:
        assessment = 'classification_effect_below_minimum'
    else:
        assessment = 'not_confirmed'
    return dict(passed=passed, assessment=assessment, complete_primary_pairs=complete, overall=overall,
                directions=directions, direction_limits_met=direction_limits, directions_ba_positive=directions_ba_positive,
                probability_effect_met=probability_effect, classification_effect_met=classification_effect,
                favorable_paired_intervals=intervals_favorable,
                scope='NCI1 edge-count domains M0 to M1/M2/M3; freshly trained M0 source checkpoints')


def check_gate(plan):
    """Exercise the D decision using generated paired predictions only."""
    tau, grid = _registered_blocks(plan)
    if 'pooling' in plan['phases']['D']['source_model']:
        nci1_inputs.check_source_pooling()
    toy = plan | {'evaluation': plan['evaluation'] | {'cohort_bootstrap_replicates': 32}}
    cohorts = []
    labels = np.tile(np.arange(2), 20)
    for setting, source, target, seed in grid:
        logits = {'B0': np.broadcast_to([.2, 0.], (len(labels), 2)), 'J': np.eye(2)[labels]}
        cohorts.append(dict(setting=setting, source=source, target=target, seed=seed,
                            ids=np.arange(len(labels)) + 100 * target, y=labels, logits=logits))

    def decision(data):
        paired = paired_bootstrap(toy, data, phase='D', comparisons=[('J', 'B0')], tau=tau)
        return _confirmation(toy, paired['interval_rows'])

    assert decision(cohorts)['passed']
    assert decision(cohorts[:-1])['assessment'] == 'incomplete'
    adverse_target = toy['phases']['D']['directions'][-1][1]
    adverse = [row | {'logits': row['logits'] | {'J': np.eye(2)[1 - labels]}}
               if row['target'] == adverse_target else row for row in cohorts]
    rejected = decision(adverse)
    assert not rejected['passed'] and not rejected['direction_limits_met']


def run(plan_path, root, c_output, acquisition, workers, repo):
    """Run the registered external confirmation using local authenticated inputs."""
    plan_path, root, c_output, acquisition, repo = (Path(path).resolve() for path in
                                                   (plan_path, root, c_output, acquisition, repo))
    plan = json_read(plan_path)
    tau, grid = _registered_blocks(plan)
    source_spec = nci1_inputs._source_recipe(plan)
    if not 1 <= workers <= plan['execution']['maximum_workers']:
        raise ValueError('Worker count exceeds the registered CPU budget')
    locked = _locked_files(plan, repo)
    authorization = nci1_inputs._verify_access(plan, c_output, repo, locked)
    plan_bytes = plan_path.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != authorization['c_plan_sha256']:
        raise ValueError('D must use the identical executed C plan')
    _assert_pinned_method(plan, repo)
    acquisition_bytes = acquisition.read_bytes()
    raw = json.loads(acquisition_bytes)
    if set(raw) != {'zip_sha256', 'raw_sha256'}:
        raise ValueError('Acquisition record requires zip_sha256 and raw_sha256 fields')
    torch.set_num_threads(plan['execution']['threads_per_worker'])
    root.mkdir(parents=True, exist_ok=False)
    lock = root / 'D_LOCK.json'
    write_json(lock, authorization | dict(primary_tau=tau, opened_raw_inputs=False,
                                          acquisition_sha256=hashlib.sha256(acquisition_bytes).hexdigest(),
                                          source_model=source_spec, source_refit=plan.get('source_refit')))
    lock_sha256 = digest(lock)
    (root / 'executed-plan.json').write_bytes(plan_bytes)
    (root / 'acquisition.json').write_bytes(acquisition_bytes)
    write_json(root / 'provenance.json', provenance(repo))
    write_json(root / 'executed-configs.json', dict(phase='D', workers=workers, threads_per_worker=torch.get_num_threads(),
                                                   primary_tau=tau, c_final_digest=authorization['c_final_digest'],
                                                   separation_variant=plan.get('separation_variant', 'stock'),
                                                   density_coordinates=plan.get('density_coordinates', 'source_standardized_v1'),
                                                   target_posterior_power=plan.get('target_posterior_power', 1.),
                                                   source_pooling=source_spec.get('pooling', 'sum'),
                                                   source_refit=plan.get('source_refit'),
                                                   common=[config(plan, seed, tau).to_dict()
                                                           for seed in plan['real_model_seeds']]))
    prepared = root / 'external'
    try:
        prepared_record = nci1_inputs.prepare_inputs(plan, prepared, repository=repo, raw_hashes=raw['raw_sha256'],
                                                    zip_sha256=raw['zip_sha256'], c_output=c_output, locked_files=locked)
    except (FileNotFoundError, PermissionError, ValueError) as error:
        write_json(root / 'input_failure.json', dict(phase='D', status='not_executed', reason=str(error)))
        raise
    if prepared_record['authorization'] != authorization or digest(lock) != lock_sha256:
        raise ValueError('D authorization or pre-access lock changed before source training')
    _train_sources(plan, root, prepared, min(workers, len(plan['real_model_seeds'])))
    (root / 'blocks').mkdir()
    tasks = [(root / 'blocks' / _block_name(task), task) for task in grid]
    for block, task in tasks:
        _initialize(plan, repo, block, task, prepared)
    specs = [(block, '', task[-1]) for block, task in tasks]
    input_names = ['D_LOCK.json', 'executed-plan.json', 'acquisition.json', 'provenance.json', 'executed-configs.json']
    input_names += _source_members(plan, prepared, root)
    input_names += _leaf_paths(root, specs, ['fit_input.npz', 'validation_input.npz', 'holdout_input.npz', 'manifest.json'])
    if digest(lock) != lock_sha256:
        raise ValueError('D pre-access lock changed before input sealing')
    seal(root, 'R7_INPUT', input_names, expected=input_names)
    view = plan | {'phases': plan['phases'] | {'D': plan['phases']['D'] | {'width_grid': [tau]}}}
    _fit(view, root, specs, workers, phase='D')
    key = width_key(tau)
    arms = exact_methods(plan, tau)
    for block, _ in tasks:
        records = json_read(block / f'{key}-observed.json')
        if (set(records) != arms or set(methods(read_npz(block / f'{key}-observed.npz'))) != arms
                or any(type(record['valid']) is not bool for record in records.values())):
            raise ValueError(f'{block.name}: missing or invalid D observations')
    observed_names = _leaf_paths(root, specs, _shared_observations(tau))
    seal(root, 'R7_OBS', observed_names, expected=observed_names)
    for block, _, seed in specs:
        _fit_oracle(plan, root, block, repo, seed, tau)
    oracle_names = _leaf_paths(root, specs, _oracle_files(tau))
    seal(root, 'R7_ORACLE', oracle_names, expected=oracle_names)
    endpoints, moments, coverage, separation, cohorts = _evaluate(plan, root, repo, tasks, tau)
    paired = paired_bootstrap(plan, cohorts, phase='D', comparisons=_CONTRASTS, tau=tau)
    rows = _seed_outcomes(paired['interval_rows'], plan['evaluation']['nll_tie_tolerance'])
    tables = dict(endpoints=endpoints, moments=moments, coverage=coverage, separation=separation, paired=rows)
    for name, values in tables.items():
        csv_write(root / f'{name}.csv', values)
    gate = _confirmation(plan, rows)
    write_json(root / 'paired-audit.json', paired)
    write_json(root / 'gate.json', gate)
    final_names = [f'{name}.csv' for name in tables] + ['gate.json', 'paired-audit.json']
    seal(root, 'R7_FINAL', final_names, expected=final_names)
    verify_seal(root, 'R7_FINAL')
    print(gate, flush=True)
    return gate
