"""Real pilot with phased label access, fixed cohorts, and an independent audit."""
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch

from experiments.common.io import digest, provenance, write_json
from tsba.shared_update.solver import relative_risks

from .coverage import _cdf_diagnostic, hull_distance
from .evaluate import classification, exact_methods, methods, predict, replay_predictions
from .fit import config, explanation_family, features, master, moment, read_npz, width_key, write_npz
from .r5_inputs import load_inputs, load_labels
from .seals import seal, verify_seal
from .stage_s import _assert_pinned_method, _fit, _leaf_paths, csv_write, json_read


_FIT_ARRAYS = ('source_h', 'source_y', 'source_ids', 'source_z0', 'source_head_weight', 'source_head_bias',
               'adapt_h', 'adapt_ids', 'adapt_z0')
_WIDTH_OUTPUTS = ('observed.npz', 'observed.json', 'heldout.npz')
_CONTRASTS = [('J', 'B0'), ('J', 'P'), ('J', 'P_mass'), ('J', 'E'), ('J', 'J_bias'), ('J', 'J_cal'),
              ('J', 'O'), ('O', 'B0'), ('J', 'C05'), ('J', 'C075'), ('J_plus', 'P_plus'), ('J_plus', 'P_mass_plus')]


def _initialize(plan, repo, block, setting, source, target, seed):
    block.mkdir()
    loaded = load_inputs(plan, setting, source, target, seed, repository=repo)
    write_npz(block / 'fit_input.npz', **{key: loaded[key] for key in _FIT_ARRAYS})
    for pool in ('holdout', 'validation'):
        write_npz(block / f'{pool}_input.npz', **{f'{pool}_{key}': loaded[f'{pool}_{key}']
                                                for key in ('h', 'z0', 'ids')})
    write_json(block / 'manifest.json', loaded['manifest'])


def _oracles(plan, root, block, seed, repo):
    labels = load_labels(root, str((block / 'manifest.json').relative_to(root)), 'adapt', repository=repo)
    std = read_npz(block / 'standardization.npz')
    held = read_npz(block / 'holdout_standardization.npz')
    z, u = std['adapt_z0'], std['adapt_u']
    statistic = moment(np.eye(z.shape[1])[labels], u)
    for tau in plan['phases']['P']['width_grid']:
        cfg = config(plan, seed, tau)
        w, record = master(z, u, [statistic], cfg)
        pred = predict('O', z, u, w)
        arrays = dict(w_O=w, oracle_stat=statistic, adapt_labels=labels, adapt_logits_O=pred,
                      adapt_probability_O=softmax(pred, axis=1))
        arrays.update(replay_predictions(std, held, arrays))
        if set(methods(arrays)) != {'O'}:
            raise ValueError('Missing frozen adaptation-label oracle')
        key = width_key(tau)
        write_npz(block / f'{key}-oracle.npz', **arrays)
        write_json(block / f'{key}-oracle.json', record)


def _bank_diagnostics(plan, cfg, fit, std, held, arrays, adapt_y, holdout_y):
    src = std['source_u'][:, :-1]
    adapt = explanation_family(plan, src, fit['source_y'], std['adapt_u'][:, :-1], cfg, std['v0'])
    holdout = explanation_family(plan, src, fit['source_y'], held['holdout_u'][:, :-1], cfg, std['v0'])
    h_posterior = np.stack([holdout._target_terms(theta)[-1] for theta in arrays['theta']])
    h_stats = np.stack([moment(q, held['holdout_u']) for q in h_posterior])
    pool_moments = []
    tolerance = dict(feasibility_tol=cfg.feasibility_tol, optimality_tol=cfg.optimality_tol)
    for pool, z, u, y, bank in (
        ('adapt', std['adapt_z0'], std['adapt_u'], adapt_y, arrays['stats']),
        ('holdout', held['holdout_z0'], held['holdout_u'], holdout_y, h_stats),
    ):
        sy = moment(np.eye(z.shape[1])[y], u)
        ss = moment(softmax(z, axis=1), u)
        w = arrays['w_J']
        risk = float(relative_risks(z, u, w, sy[None])[0])
        worst = float(relative_risks(z, u, w, bank).max())
        pool_moments.append(dict(pool=pool, candidate_count=len(bank),
                                 observed_moment=hull_distance(sy, bank, **tolerance),
                                 source_prediction_moment=hull_distance(ss, bank, **tolerance),
                                 empirical_risk_change=risk, retained_worst_change=worst, gap=risk - worst))
    fit_quality = _cdf_diagnostic(adapt, arrays['ref_theta'], holdout.target)
    fit_quality['density_dimension'] = adapt.dimension
    fit_quality['correction_dimension'] = std['adapt_u'].shape[1] - 1
    expanded = None
    if 'audit_theta' in arrays:
        expanded = np.stack([holdout._target_terms(theta)[-1] for theta in arrays['audit_theta']])
    return pool_moments, fit_quality, h_posterior, expanded


def _endpoints(name, record, key, std, arrays, labels, source_pairs, reference):
    val_h, val_z = source_pairs['validation_h'], source_pairs['validation_z0']
    val_u = features(val_h, std['mean'], std['scale'])
    vol = [
        ('source_fit', predict(name, source_pairs['source_z0'], std['source_u'], arrays[f'w_{name}']),
         labels['source_fit'], source_pairs['source_z0']),
        ('source_validation', predict(name, val_z, val_u, arrays[f'w_{name}']), labels['source_validation'], val_z),
        ('adapt', arrays[f'adapt_logits_{name}'], labels['adapt'], reference['adapt']),
        ('holdout', arrays[f'holdout_logits_{name}'], labels['holdout'], reference['holdout']),
    ]
    rows = []
    for pool, z, y, b0 in vol:
        delta = z - b0
        delta -= delta.mean(1, keepdims=True)
        row = dict(**key, pool=pool, method=name, valid=record['valid'], status=record['status'],
                   stop_reason=record['stop_reason'], n=len(y))
        if record['valid']:
            row.update(classification(z, y, b0.argmax(1)))
            row['centered_rms'] = float(np.sqrt(np.mean(delta ** 2)))
            row['mean_predicted_p1'] = float(softmax(z, axis=1)[:, 1].mean())
        rows.append(row)
    return rows


def _evaluate(plan, root, repo, tasks):
    from .paired import paired_bootstrap

    endpoints, moments, coverage, separation, cohorts = [], [], [], [], []
    primary = plan['shared_update']['primary_tau']
    for block, setting, source, target, seed in tasks:
        manifest = str((block / 'manifest.json').relative_to(root))
        a_y = load_labels(root, manifest, 'adapt', repository=repo, authenticated_snapshot=True)
        h_y = load_labels(root, manifest, 'holdout', repository=repo)
        v_y = load_labels(root, manifest, 'validation', repository=repo)
        fit = read_npz(block / 'fit_input.npz')
        std = read_npz(block / 'standardization.npz')
        held = read_npz(block / 'holdout_standardization.npz')
        source_pairs = read_npz(block / 'validation_input.npz') | {'source_z0': fit['source_z0']}
        for tau in plan['phases']['P']['width_grid']:
            key = width_key(tau)
            metadata = dict(setting=setting, source=source, target=target, seed=seed, tau=tau)
            observed = read_npz(block / f'{key}-observed.npz')
            oracle = read_npz(block / f'{key}-oracle.npz')
            np.testing.assert_array_equal(oracle['adapt_labels'], a_y,
                                          err_msg='Oracle labels differ from the authenticated raw snapshot')
            observed_record = json_read(block / f'{key}-observed.json')
            oracle_record = json_read(block / f'{key}-oracle.json')
            holdout = read_npz(block / f'{key}-heldout.npz')
            for inputs, replay in ((observed, holdout), (oracle, oracle)):
                for name, result in replay_predictions(std, held, inputs).items():
                    np.testing.assert_array_equal(result, replay[name])
            arrays = observed | holdout | oracle
            labels = {'adapt': oracle['adapt_labels'], 'holdout': h_y,
                      'source_fit': fit['source_y'], 'source_validation': v_y}
            results = observed_record | {'O': oracle_record}
            references = {pool: arrays[f'{pool}_logits_B0'] for pool in ('adapt', 'holdout')}
            for name, record in results.items():
                endpoints.extend(_endpoints(name, record, metadata, std, arrays, labels, source_pairs, references))
                separation.append(dict(**metadata, method=name, valid=record['valid'], stop_reason=record['stop_reason'],
                                       known_worst_risk=record.get('found_worst_risk'),
                                       exact_separation_violation=record.get('exact_separation_violation'),
                                       independent_audit=record.get('independent_audit')))
            cfg = config(plan, seed, tau)
            pool_moments, fit_quality, h_q, expanded_q = _bank_diagnostics(plan, cfg, fit, std, held, arrays, labels['adapt'], h_y)
            moments.extend(metadata | item for item in pool_moments)
            coverage.append(metadata | fit_quality)
            if tau == primary:
                cohorts.append(dict(**metadata, ids=held['holdout_ids'], y=h_y, candidate_posteriors=h_q,
                                    expanded_candidate_posteriors=expanded_q,
                                    logits={name: arrays[f'holdout_logits_{name}'] if record['valid'] else None
                                            for name, record in results.items()}))
    paired = paired_bootstrap(plan, cohorts, phase='P', comparisons=_CONTRASTS, tau=primary)
    return endpoints, moments, coverage, separation, paired


def _pilot_gate(plan, endpoints, separation):
    p = plan['phases']['P']
    limits = p['gate']
    primary = plan['shared_update']['primary_tau']
    h = {(row['source'], row['target'], row['seed'], row['method']): row for row in endpoints
         if row['pool'] == 'holdout' and row['tau'] == primary}
    audit = {(row['source'], row['target'], row['seed']): row['independent_audit'] for row in separation
             if row['tau'] == primary and row['method'] == 'J_plus'}
    tie = plan['evaluation']['nll_tie_tolerance']
    directions = {}
    for source, target in p['directions']:
        pairs = [(h.get((source, target, seed, 'B0')), h.get((source, target, seed, 'J')))
                 for seed in plan['real_model_seeds']]
        valid = [(a, b) for a, b in pairs if a and b and a['valid'] and b['valid']
                 and a['balanced_accuracy'] is not None and b['balanced_accuracy'] is not None]
        dnll = [j['sampled_nll'] - b['sampled_nll'] for b, j in valid]
        dba = [100. * (j['balanced_accuracy'] - b['balanced_accuracy']) for b, j in valid]
        rms = sum(j['centered_rms'] > limits['centered_logit_rms_nonzero'] for _, j in valid)
        wins = sum(delta < -tie for delta in dnll)
        corrections = sum(j['net_corrections'] > 0 for _, j in valid)
        mean_nll, mean_ba = (float(np.mean(x)) if x else None for x in (dnll, dba))
        directions[f'M{source}->M{target}'] = dict(valid_pairs=len(valid), expected_pairs=len(pairs),
                                                 dnll=mean_nll, dba=mean_ba, nll_wins=wins, nonzero=rms,
                                                 positive_net_corrections=corrections,
                                                 nll_pass=len(valid) == len(pairs) and
                                                 mean_nll <= -limits['mean_nll_decrease_vs_B0_for_J'] and
                                                 wins >= limits['minimum_seed_improvement_count_per_direction'],
                                                 nonzero_pass=rms >= limits['minimum_nonzero_J_seeds_per_direction'])
    complete = sum(row['valid_pairs'] for row in directions.values()) == limits['complete_blocks']
    audits_complete = len(audit) == limits['complete_blocks'] and all(record and record['complete'] for record in audit.values())
    found_harm = any(record and record['witness'] is not None for record in audit.values())
    both_ba = complete and all(row['dba'] >= limits['minimum_mean_ba_gain_each_direction_pp'] for row in directions.values())
    practically_helpful = any(row['dba'] is not None and row['dba'] >= limits['minimum_mean_ba_gain_either_direction_pp']
                              and row['positive_net_corrections'] >= limits['minimum_positive_net_corrections_seeds_in_BA_gain_direction']
                              for row in directions.values())
    passed = complete and all(row['nll_pass'] and row['nonzero_pass'] for row in directions.values())
    passed = passed and audits_complete and not found_harm and both_ba and practically_helpful
    return dict(passed=passed, next_phase='C' if passed else None, complete_primary_pairs=complete,
                audits_complete=audits_complete, observed_harm_witness_found=found_harm,
                both_ba_non_decreasing=both_ba, practically_helpful=practically_helpful, directions=directions)


def run(plan_path, root, s_output, workers, repo):
    root, s_output, repo = (Path(p).resolve() for p in (root, s_output, repo))
    prior = verify_seal(s_output, 'R7_FINAL')
    input_record = verify_seal(s_output, 'R7_INPUT')
    if 'gate.json' not in prior['files'] or not {'executed-plan.json', 'provenance.json'} <= input_record['files'].keys():
        raise ValueError('The prior gate or configuration is outside the verified seals')
    decision = json_read(s_output / 'gate.json')
    if not decision['passed'] or decision['next_phase'] != 'P':
        raise ValueError('The sealed S decision did not authorize this pilot')
    if digest(plan_path) != digest(s_output / 'executed-plan.json'):
        raise ValueError('The pilot and passed S must use the same frozen plan')
    plan = json_read(plan_path)
    if plan.get('development_iteration') and plan.get('separation_variant') == 'stock':
        raise ValueError('This stock plan is a synthetic comparator only')
    fit_sources = ['fit.py', 'controls.py']
    variant = plan.get('separation_variant')
    if variant in ('restore_trust_v1', 'projected_restore_v1'):
        fit_sources.append('restored_separation.py')
    if variant == 'projected_restore_v1':
        fit_sources.append('projected_explanations.py')
    frozen = json_read(s_output / 'provenance.json')['file_hashes']
    for name in fit_sources:
        path = Path(__file__).parent / name
        if frozen.get(str(path.relative_to(repo))) != digest(path):
            raise ValueError(f'The pilot fitter differs from the passed synthetic implementation: {name}')
    if not 1 <= workers <= plan['execution']['maximum_workers']:
        raise ValueError('Worker count exceeds the registered CPU budget')
    _assert_pinned_method(plan, repo)
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'blocks').mkdir()
    p = plan['phases']['P']
    tasks = []
    for (source, target), seed in product(p['directions'], plan['real_model_seeds']):
        block = root / 'blocks' / f'{p["setting"]}-M{source}_to_M{target}-seed-{seed}'
        tasks.append((block, p['setting'], source, target, seed))
        _initialize(plan, repo, *tasks[-1])
    if len(tasks) != p['expected_blocks'] or len({task[0] for task in tasks}) != len(tasks):
        raise ValueError('Registered pilot block keys are incomplete or duplicated')
    specs = [(block, '', seed) for block, _, _, _, seed in tasks]
    widths = p['width_grid']
    (root / 'executed-plan.json').write_bytes(Path(plan_path).read_bytes())
    write_json(root / 'executed-configs.json', dict(phase='P', s_final_digest=prior['digest'], s_output=str(s_output),
                                                   workers=workers, threads_per_worker=torch.get_num_threads(),
                                                   independent_audit=p['independent_audit'],
                                                   separation_variant=plan.get('separation_variant', 'stock'),
                                                   common=[config(plan, seed, tau).to_dict()
                                                           for seed, tau in product(plan['real_model_seeds'], widths)]))
    write_json(root / 'provenance.json', provenance(repo))
    files = ['executed-plan.json', 'executed-configs.json', 'provenance.json']
    files += _leaf_paths(root, specs, ['fit_input.npz', 'holdout_input.npz', 'validation_input.npz', 'manifest.json'])
    seal(root, 'R7_INPUT', files, expected=files)
    _fit(plan, root, specs, workers, phase='P')
    for block, _, _ in specs:
        for tau in widths:
            key = width_key(tau)
            expected = exact_methods(plan, tau)
            if tau == p['independent_audit']['tau']:
                expected |= {'J_plus', 'P_plus', 'P_mass_plus'}
            records = json_read(block / f'{key}-observed.json')
            if set(records) != expected or any(type(row['valid']) is not bool for row in records.values()):
                raise ValueError(f'{block.name}/{key}: incomplete frozen diagnostics')
    files = _leaf_paths(root, specs, ['standardization.npz', 'holdout_standardization.npz'] +
                        [f'{width_key(tau)}-{suffix}' for tau, suffix in product(widths, _WIDTH_OUTPUTS)])
    seal(root, 'R7_OBS', files, expected=files)
    for block, _, seed in specs:
        _oracles(plan, root, block, seed, repo)
    files = _leaf_paths(root, specs, [f'{width_key(tau)}-{suffix}' for tau, suffix in
                                    product(widths, ('oracle.npz', 'oracle.json'))])
    seal(root, 'R7_ORACLE', files, expected=files)
    endpoints, moments, coverage, separation, paired = _evaluate(plan, root, repo, tasks)
    table = dict(endpoints=endpoints, moments=moments, coverage=coverage, separation=separation,
                 paired=paired['interval_rows'])
    for name, rows in table.items():
        csv_write(root / f'{name}.csv', rows)
    write_json(root / 'paired-audit.json', paired)
    gate = _pilot_gate(plan, endpoints, separation)
    gate['mechanism'] = _mechanism(plan, paired['interval_rows'])
    write_json(root / 'gate.json', gate)
    files = [f'{name}.csv' for name in table] + ['gate.json', 'paired-audit.json']
    seal(root, 'R7_FINAL', files, expected=files)
    verify_seal(root, 'R7_FINAL')
    print(gate, flush=True)
    return gate


def _mechanism(plan, intervals):
    criteria = plan['phases']['P']['mechanism_criteria']

    def passes(left, right):
        means = [row['estimate'] for row in intervals if row['left'] == left and row['right'] == right
                 and row['scope'] == 'direction' and row['metric'] == 'NLL']
        if len(means) != len(plan['phases']['P']['directions']) or any(value is None for value in means):
            return None
        return (min(means) <= -criteria['minimum_mean_nll_decrease_at_least_one_direction'] and
                max(means) <= criteria['maximum_mean_nll_increase_other_direction'])

    return [{'contrast': [left, right], 'primary_point_criteria': passes(left, right),
             'expanded_point_criteria': passes(f'{left}_plus', f'{right}_plus')}
            for left, right in criteria['contrasts']]


def check_gate(plan):
    """A positive fixture and a feasible-witness stop with identical metrics."""
    p = plan['phases']['P']
    primary = plan['shared_update']['primary_tau']
    endpoints, separation = [], []
    for (source, target), seed in product(p['directions'], plan['real_model_seeds']):
        key = dict(source=source, target=target, seed=seed, tau=primary, valid=True)
        endpoints.extend([key | dict(pool='holdout', method='B0', sampled_nll=.5, balanced_accuracy=.6),
                          key | dict(pool='holdout', method='J', sampled_nll=.49, balanced_accuracy=.61,
                                     centered_rms=.1, net_corrections=1)])
        separation.append(key | dict(method='J_plus', independent_audit=dict(complete=True, witness=None)))
    assert _pilot_gate(plan, endpoints, separation)['passed']
    separation[0]['independent_audit']['witness'] = {'risk': .001}
    assert not _pilot_gate(plan, endpoints, separation)['passed']
