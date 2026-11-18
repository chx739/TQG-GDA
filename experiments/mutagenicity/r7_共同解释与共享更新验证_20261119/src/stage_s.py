"""Registered S experiment with append-only observations and oracle labels."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from itertools import product
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback

import numpy as np
import scipy
import torch

from experiments.common.io import digest, provenance, write_json
from tsba.shared_update.solver import relative_risks

from . import controls, coverage
from .evaluate import (_committed_truth, endpoint, exact_methods, fit_oracles, methods, replay_heldout,
                       replay_predictions, vault)
from .fit import config, fit_block, read_npz, width_key, write_npz
from .seals import seal, verify_seal
from .synthetic import check_known_moment, generate, synthetic_gate


def json_read(path):
    return json.loads(Path(path).read_text())


def csv_write(path, rows):
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _log(message):
    print(message, flush=True)


def _initialize_block(plan, block, world_index, seed):
    block.mkdir()
    config_s = plan['phases']['S']
    inputs, truth = generate(plan, world_index, seed)
    w0, b0 = (truth.pop(key) for key in ('source_head_weight', 'source_head_bias'))
    n = config_s['adaptation_count']
    write_npz(block / 'fit_input.npz', source_h=inputs['source_h'], source_y=inputs['source_y'],
              source_ids=np.arange(config_s['source_count']), adapt_h=inputs['adapt_h'],
              adapt_z0=inputs['adapt_z0'], adapt_ids=np.arange(n), source_head_weight=w0, source_head_bias=b0)
    write_npz(block / 'holdout_input.npz', holdout_h=inputs['holdout_h'], holdout_z0=inputs['holdout_z0'],
              holdout_ids=np.arange(n, n + config_s['holdout_count']))
    held_truth = {key: truth.pop(key) for key in ('holdout_y', 'holdout_q')}
    write_npz(block / 'secret_adapt.npz', **truth)
    write_npz(block / 'secret_holdout.npz', **held_truth)
    write_json(block / 'commitments.json', {f'secret_{pool}.npz': digest(block / f'secret_{pool}.npz')
                                            for pool in ('adapt', 'holdout')})


def _leaf_paths(root, specs, filenames):
    return [str((block / name).relative_to(root)) for block, _, _ in specs for name in filenames]


def _fit(plan, root, specs, workers, *, phase='S'):
    verify_seal(root, 'R7_INPUT')
    errors = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        worker_plan = {key: plan[key] for key in ('shared_update', 'controls')}
        worker_plan['separation_variant'] = plan.get('separation_variant', 'stock')
        worker_plan['phases'] = {phase: {key: plan['phases'][phase][key]
                                       for key in ('width_grid', 'independent_audit')
                                       if key in plan['phases'][phase]}}
        pending = {pool.submit(fit_block, worker_plan, seed, block, phase=phase): block.name
                   for block, _, seed in specs}
        for future in as_completed(pending):
            try:
                _log(f'fit frozen {future.result()}')
            except Exception:
                error = dict(block=pending[future], traceback=traceback.format_exc())
                errors.append(error)
                _log(f'fit error {error["block"]}: {error["traceback"]}')
    if errors:
        write_json(root / 'fit_errors.json', errors)
        raise RuntimeError(f'{len(errors)} block(s) failed; sealed truth remains closed')
    for block, _, _ in specs:
        replay_heldout(plan, block, phase=phase)


def _evaluate(plan, root, specs):
    verify_seal(root, 'R7_ORACLE')
    results, comparisons, cov_rows, moments, separation = [], [], [], [], []
    for block, world, seed in specs:
        truth = _committed_truth(block, 'adapt') | _committed_truth(block, 'holdout')
        fit = read_npz(block / 'fit_input.npz')
        standard = read_npz(block / 'standardization.npz')
        held = read_npz(block / 'holdout_standardization.npz')
        for tau in plan['phases']['S']['width_grid']:
            key = width_key(tau)
            width = dict(world=world, seed=seed, tau=tau)
            observed = read_npz(block / f'{key}-observed.npz')
            oracle = read_npz(block / f'{key}-oracle.npz')
            records = json_read(block / f'{key}-observed.json') | json_read(block / f'{key}-oracle.json')
            heldout = read_npz(block / f'{key}-heldout.npz')
            for arrays, saved in ((observed, heldout), (oracle, oracle)):
                for name, value in replay_predictions(standard, held, arrays).items():
                    if not np.array_equal(value, saved[name]):
                        raise ValueError(f'{block.name}/{key}/{name}: holdout replay mismatch')
            cov = coverage.diagnostics(config(plan, seed, tau), fit, standard, held, truth, observed, records['J'])
            cov_rows.append(width | cov)
            moments.append(width | {k: v for k, v in cov.items() if 'moment' in k or 'simplex' in k})
            arrays = observed | oracle | heldout
            if set(methods(arrays)) != set(records) or set(methods(oracle)) != exact_methods(plan, tau, oracle=True):
                raise ValueError(f'{block.name}/{key}: incomplete evaluation arms')
            b0a, b0h = (arrays[f'{pool}_logits_B0'] for pool in ('adapt', 'holdout'))
            metrics = {}
            for name in methods(arrays):
                result = records[name]
                # Each contrast against the retained joint set uses exactly those candidates.
                worst = float(relative_risks(standard['adapt_z0'], standard['adapt_u'], arrays[f'w_{name}'],
                                             observed['stats']).max())
                if name in ('C05', 'C075'):
                    multiplier = {'C05': .5, 'C075': .75}[name]
                    worst = float(relative_risks(standard['adapt_z0'], standard['adapt_u'],
                                                 (multiplier - 1) * standard['v0'], observed['stats']).max())
                metrics[name] = endpoint(result, arrays[f'adapt_logits_{name}'], arrays[f'holdout_logits_{name}'],
                                         b0a, b0h, truth, worst, method=name, **width)
                if name in ('J', 'J_cal'):
                    metrics[name]['boundary_normal_error_degrees'] = coverage.boundary_angle(
                        standard['v0'], arrays[f'w_{name}'], truth, standard)
                separation.append(width | dict(method=name, valid=result['valid'], status=result['status'],
                                               stop_reason=result['stop_reason'],
                                               solver_worst_risk=result.get('found_worst_risk'),
                                               joint_bank_worst_risk=worst,
                                               true_directional_risk_gap=metrics[name]['true_directional_risk_gap'],
                                               candidate_count=records['J']['candidate_count']))
            results.extend(metrics.values())
            for other in sorted(metrics.keys() - {'J'}):
                left, right = metrics['J'], metrics[other]
                contrasts = {}
                if left['valid'] and right['valid']:
                    contrasts = {f'{k}_delta': left[k] - right[k] for k in (
                        'holdout_expected_nll', 'sampled_nll', 'balanced_accuracy', 'accuracy', 'net_corrections')
                        if left[k] is not None and right[k] is not None}
                comparisons.append(width | dict(left='J', right=other, valid=bool(contrasts)) | contrasts)
    return results, comparisons, cov_rows, moments, separation


def _assert_pinned_method(plan, repo):
    method = plan['method_commit']
    subprocess.run(['git', 'diff', '--quiet', method, '--', 'tsba/shared_update'], cwd=repo, check=True)
    workspace = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)
    if workspace:
        raise ValueError('Use a clean committed worktree to start a research attempt')


def run(plan_path, root, workers, repo):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    plan = json_read(plan_path)
    if not 1 <= workers <= plan['execution']['maximum_workers']:
        raise ValueError('Worker count exceeds the registered CPU budget')
    torch.set_num_threads(1)
    _assert_pinned_method(plan, repo)
    known_moment = check_known_moment()
    if not known_moment['passed']:
        raise ValueError('Known-moment engineering fixture failed')
    root.mkdir(parents=True, exist_ok=False)
    controls.check_controls()
    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        check_seals(temporary / 'seals')
        check_replay(plan, temporary / 'replay')
    regression_command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/tsba',
                          '-p', 'test_shared_update.py', '-v']
    regression = subprocess.run(regression_command, cwd=repo, capture_output=True, text=True)
    (root / 'engineering-regression.txt').write_text(regression.stdout + regression.stderr)
    regression.check_returncode()
    write_json(root / 'engineering.json', dict(passed=True, command=regression_command,
                                               exact_controls=True, sealed_label_access=True,
                                               small_pipeline_replay=True, known_moment=True))
    stage_config = plan['phases']['S']
    specs = [(root / 'blocks' / f'{world["id"]}-seed-{seed}', world['id'], seed)
             for world, seed in product(stage_config['worlds'], stage_config['seeds'])]
    (root / 'blocks').mkdir()
    for index, world in enumerate(stage_config['worlds']):
        for seed in stage_config['seeds']:
            _initialize_block(plan, root / 'blocks' / f'{world["id"]}-seed-{seed}', index, seed)
    (root / 'executed-plan.json').write_bytes(Path(plan_path).read_bytes())
    widths = stage_config['width_grid']
    common = [dict(seed=seed, configs=[config(plan, seed, tau).to_dict() for tau in widths]) for seed in stage_config['seeds']]
    write_json(root / 'executed-configs.json', dict(phase='S', plan_sha256=digest(plan_path), workers=workers,
                                                   threads_per_worker=torch.get_num_threads(), dtype='float64',
                                                   separation_variant=plan.get('separation_variant', 'stock'),
                                                   world_ids=[world['id'] for world in stage_config['worlds']],
                                                   common=common, numpy=np.__version__, scipy=scipy.__version__,
                                                   hostname=platform.node(), executable=sys.executable))
    write_json(root / 'provenance.json', provenance(repo))
    write_json(root / 'known-moment.json', known_moment)
    files = ['executed-plan.json', 'executed-configs.json', 'provenance.json', 'known-moment.json',
             'engineering.json', 'engineering-regression.txt']
    files += _leaf_paths(root, specs, ['fit_input.npz', 'holdout_input.npz', 'commitments.json'])
    seal(root, 'R7_INPUT', files, expected=files)
    _fit(plan, root, specs, workers)
    files = _leaf_paths(root, specs, ['standardization.npz', 'holdout_standardization.npz'] + [
        f'{width_key(tau)}-{suffix}' for tau, suffix in product(widths, ('observed.npz', 'observed.json', 'heldout.npz'))])
    seal(root, 'R7_OBS', files, expected=files)
    _log('R7_OBS sealed. Fitting the two target-supervised diagnostics.')
    fit_oracles(plan, root, specs)
    files = _leaf_paths(root, specs, [f'{width_key(tau)}-{suffix}' for tau, suffix in product(widths, ('oracle.npz', 'oracle.json'))])
    seal(root, 'R7_ORACLE', files, expected=files)
    _log('R7_ORACLE sealed. Opening holdout labels only for evaluation.')
    tables = _evaluate(plan, root, specs)
    names = ['endpoints.csv', 'paired.csv', 'coverage.csv', 'moments.csv', 'separation.csv']
    for name, rows in zip(names, tables):
        csv_write(root / name, rows)
    decision = synthetic_gate(plan, tables[0])
    # Missing numerical controls remain visible but do not replace the registered S science endpoints.
    decision['expected_observation_blocks'] = len(specs)
    decision['widths'] = widths
    stock_comparator = plan.get('development_iteration') and plan.get('separation_variant') == 'stock'
    decision['next_phase'] = 'P' if decision['passed'] and not stock_comparator else None
    decision['engineering_regression_passed'] = True
    decision['known_moment_passed'] = known_moment['passed']
    decision['input_and_frozen_replay_passed'] = True
    write_json(root / 'gate.json', decision)
    files = names + ['gate.json']
    seal(root, 'R7_FINAL', files, expected=files)
    verify_seal(root, 'R7_FINAL')
    _log(json.dumps(decision, ensure_ascii=False))
    return decision


def check_replay(plan, root):
    """Small end-to-end fixture with the same artifact and numerical path as S."""
    plan = deepcopy(plan)
    phase = plan['phases']['S']
    phase.update(source_count=48, adaptation_count=60, holdout_count=92,
                 width_grid=[plan['shared_update']['primary_tau']])
    root.mkdir()
    block = root / 'fixture'
    _initialize_block(plan, block, 0, 0)
    fit_data = read_npz(block / 'fit_input.npz')
    held_data = read_npz(block / 'holdout_input.npz')
    assert set(fit_data) == {'source_h', 'source_y', 'source_ids', 'source_head_weight', 'source_head_bias',
                             'adapt_h', 'adapt_z0', 'adapt_ids'}
    assert set(held_data) == {'holdout_h', 'holdout_z0', 'holdout_ids'}
    fit_block(plan, 0, block)
    replay_heldout(plan, block)
    key = width_key(phase['width_grid'][0])
    files = ['fixture/commitments.json']
    seal(root, 'R7_INPUT', files, expected=files)
    files = [f'fixture/{key}-heldout.npz', f'fixture/{key}-observed.npz', f'fixture/{key}-observed.json']
    seal(root, 'R7_OBS', files, expected=files)
    specs = [(block, phase['worlds'][0]['id'], 0)]
    fit_oracles(plan, root, specs)
    files = [f'fixture/{key}-oracle.npz', f'fixture/{key}-oracle.json']
    seal(root, 'R7_ORACLE', files, expected=files)
    endpoints, _, cov, _, _ = _evaluate(plan, root, specs)
    assert {r['method'] for r in endpoints} == exact_methods(plan, phase['width_grid'][0]) | {'O', 'O_analytic'}
    assert len(cov) == 1


def check_seals(root):
    """Small engineering fixture for append-only commits and gated truth."""
    root.mkdir()
    block = root / 'block'
    block.mkdir()
    for pool in ('adapt', 'holdout'):
        write_npz(block / f'secret_{pool}.npz', labels=np.array([0, 1]))
    write_json(block / 'commitments.json', {f'secret_{pool}.npz': digest(block / f'secret_{pool}.npz')
                                           for pool in ('adapt', 'holdout')})
    for stage, filename in [('R7_INPUT', 'block/commitments.json'), ('R7_OBS', 'obs'), ('R7_ORACLE', 'oracle')]:
        if stage == 'R7_INPUT':
            pool = 'adapt'
        else:
            pool = {'R7_OBS': 'adapt', 'R7_ORACLE': 'holdout'}[stage]
            (root / filename).touch()
        try:
            vault(root, block, pool)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError(f'Unsealed labels opened: {pool}')
        seal(root, stage, [filename], expected=[filename])
        if stage != 'R7_INPUT':
            np.testing.assert_array_equal(vault(root, block, pool)['labels'], [0, 1])
    (root / 'obs').write_text('tampered')
    try:
        vault(root, block, 'holdout')
    except ValueError:
        return
    raise AssertionError('Tampered predecessor accepted')
