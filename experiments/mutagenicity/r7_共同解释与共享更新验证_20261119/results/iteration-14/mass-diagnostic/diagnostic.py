"""One preregistered paired diagnostic; all new A fits seal before truth access."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[1] / 'S'
SNAPSHOT = ROOT / 'code_snapshot'
sys.path.insert(0, str(SNAPSHOT))
PREFIX = 'experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.'
import numpy as np
import scipy
from scipy.special import softmax, log_softmax
import torch
torch.set_num_threads(1)
fit = importlib.import_module(PREFIX+'fit')
decisions = importlib.import_module(PREFIX+'decisions')
seals = importlib.import_module(PREFIX+'seals')
solver = importlib.import_module('tsba.shared_update.solver')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def signed(paths, name, parent=None):
    record = dict(parent=parent, files={str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)})
    record['digest'] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write(ROOT/name, record)
    return record


def verify(record):
    for path, checksum in record['files'].items():
        if sha(ROOT/path) != checksum:
            raise ValueError('Diagnostic artifact changed: '+path)


def family(plan, data, std, cfg, query):
    return fit.explanation_family(plan, std['source_u'][:, :-1], data['source_y'], query, cfg, std['v0'],
                                  source_mean=std['mean'], source_scale=std['scale'])


def run_one(task):
    world, seed = task
    started = time.monotonic()
    name = f'{world}-seed-{seed}'
    original = ORIGINAL/'blocks'/name
    out = ROOT/'blocks'/name
    out.mkdir()
    plan = json.loads((ORIGINAL/'executed-plan.json').read_text())
    data = fit.read_npz(original/'fit_input.npz')
    std = fit.read_npz(original/'standardization.npz')
    old = json.loads((original/'tau-0.005-observed.json').read_text())['J']
    cfg = fit.config(plan, seed, .005)
    new_plan = deepcopy(plan)
    new_plan['posterior_mass_options']['mass_weight'] = 1/len(std['adapt_u'])
    base = family(new_plan, data, std, cfg, std['adapt_u'][:, :-1])
    for actual, saved in zip(base.source_templates, old['reference_fit_diagnostics']['source_templates']):
        for key in actual:
            np.testing.assert_array_equal(actual[key], saved[key])
    reference = base.fit()
    z, u = std['adapt_z0'], std['adapt_u']
    w, detail = solver.solve_shared_update(z, u, base, cfg)
    detail['valid'] = detail['status'] == 'completed' and detail['stop_reason'] not in (
        'local_search_incomplete', 'search_returned_no_candidate')
    if not detail['candidates']:
        raise ValueError('New fit has no retained candidates')
    we, e = fit.master(z, u, [reference.statistic], cfg)
    theta = np.asarray([row['parameters'] for row in detail['candidates']])
    stats = np.asarray([row['statistic'] for row in detail['candidates']])
    q = np.stack([base._target_terms(p)[-1] for p in theta])
    np.testing.assert_allclose(stats, np.einsum('knc,nd->kcd', q, u)/len(u), rtol=0, atol=1e-12)
    arrays = dict(theta=theta, stats=stats, ref_theta=reference.parameters,
                  adapt_q_reference=base._target_terms(reference.parameters)[-1], adapt_q_bank=q)
    records = dict(J=detail, E=e)
    for name_, correction in [('J', w), ('E', we), ('B0', np.zeros_like(w)),
                               ('C05', np.zeros_like(w)), ('C075', np.zeros_like(w))]:
        arrays['w_'+name_] = correction
        arrays['adapt_logits_'+name_] = fit.predict(name_, z, u, correction)
        arrays['adapt_probability_'+name_] = softmax(arrays['adapt_logits_'+name_], axis=1)
        records.setdefault(name_, dict(valid=True, status='fixed', stop_reason='fixed'))
    decisions.freeze_decisions(new_plan, arrays, records)
    fit.write_npz(out/'a.npz', **arrays)
    write(out/'fit.json', dict(world=world, seed=seed, records=records, fit_report=base.fit_report,
                              mass_weight=1/len(u), mass_coefficient=base.mass_coefficient,
                              effective_pseudo_count=1, source_templates_exactly_reused=True,
                              input_hashes={name_: sha(original/name_) for name_ in (
                                  'fit_input.npz','standardization.npz','tau-0.005-observed.json','tau-0.005-observed.npz')},
                              seconds=time.monotonic()-started, truth_read=False, holdout_read=False))
    return dict(world=world, seed=seed, valid=detail['valid'], stop_reason=detail['stop_reason'],
                seconds=round(time.monotonic()-started, 2), candidates=len(theta))


def mean_loss(z, q):
    return -np.sum(q*log_softmax(z, axis=1), axis=1).mean()


def evaluate(config, obs_seal):
    verify(obs_seal)
    plan = json.loads((ORIGINAL/'executed-plan.json').read_text())
    rows = []
    for world, seed in config['tasks']:
        name = f'{world}-seed-{seed}'
        original, out = ORIGINAL/'blocks'/name, ROOT/'blocks'/name
        std = fit.read_npz(original/'standardization.npz')
        data = fit.read_npz(original/'fit_input.npz')
        held = fit.read_npz(original/'holdout_standardization.npz')
        # First truth read occurs only after EVERY new A fit and decision is sealed.
        commitments = json.loads((original/'commitments.json').read_text())
        for pool in ('adapt', 'holdout'):
            if sha(original/f'secret_{pool}.npz') != commitments[f'secret_{pool}.npz']:
                raise ValueError('Synthetic truth commitment changed')
        truth = fit.read_npz(original/'secret_adapt.npz') | fit.read_npz(original/'secret_holdout.npz')
        original_arrays = fit.read_npz(original/'tau-0.005-observed.npz')
        original_h = fit.read_npz(original/'tau-0.005-heldout.npz')
        original_records = json.loads((original/'tau-0.005-observed.json').read_text())
        new_arrays = fit.read_npz(out/'a.npz')
        new_record = json.loads((out/'fit.json').read_text())
        cfg = fit.config(plan, seed, .005)
        ah, z0 = std['adapt_u'], std['adapt_z0']
        hu, h0 = held['holdout_u'], held['holdout_z0']
        for arm, arrays, records in [('old', original_arrays, original_records),
                                      ('fixed_pseudocount', new_arrays, new_record['records'])]:
            local_plan = deepcopy(plan)
            if arm != 'old':
                local_plan['posterior_mass_options']['mass_weight'] = 1/len(ah)
            query = family(local_plan, data, std, cfg, hu[:, :-1])
            hq = np.stack([query._target_terms(p)[-1] for p in arrays['theta']])
            hstats = np.einsum('knc,nd->kcd', hq, hu)/len(hu)
            aq = arrays.get('adapt_q_bank')
            if aq is None:
                a_family = family(local_plan, data, std, cfg, ah[:, :-1])
                aq = np.stack([a_family._target_terms(p)[-1] for p in arrays['theta']])
            output = {}
            for method in ('J', 'E'):
                w = arrays['w_'+method]
                z = arrays['adapt_logits_'+method]
                h = fit.predict(method, h0, hu, w)
                if arm == 'old':
                    np.testing.assert_array_equal(h, original_h['holdout_logits_'+method])
                action = decisions.decision_for(method, h, arrays)
                output['holdout_logits_'+method] = h
                output['holdout_decision_'+method] = action
                ar = float(mean_loss(z, truth['adapt_q'])-mean_loss(z0, truth['adapt_q']))
                hr = float(mean_loss(h, truth['holdout_q'])-mean_loss(h0, truth['holdout_q']))
                worst = float(solver.relative_risks(z0, ah, w, arrays['stats']).max())
                hworst = float(solver.relative_risks(h0, hu, w, hstats).max())
                delta = h-h0
                delta -= delta.mean(1, keepdims=True)
                rms = float(np.sqrt(np.mean(delta**2)))
                valid = bool(records[method]['valid'])
                rows.append(dict(world=world, seed=seed, arm=arm, method=method, valid=valid,
                    holdout_expected_nll_delta=hr, adapt_true_risk_delta=ar, retained_A_worst_risk=worst,
                    true_directional_risk_gap=ar-worst, retained_H_worst_risk=hworst,
                    holdout_directional_risk_gap=hr-hworst, centered_rms=rms,
                    joint_seed_pass=bool(valid and hr < 0 and rms > 1e-6 and ar-worst <= .005),
                    reference_A_qmass=float(aq[0,:,1].mean()), bank_A_qmass_min=float(aq[:,:,1].mean(1).min()),
                    bank_A_qmass_max=float(aq[:,:,1].mean(1).max()), true_A_qmass=float(truth['adapt_q'][:,1].mean()),
                    B0_A_qmass=float(softmax(z0, axis=1)[:,1].mean())))
            if arm != 'old':
                fit.write_npz(out/'holdout.npz', **output)
    write(ROOT/'endpoints.json', rows)
    with (ROOT/'endpoints.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summaries = {}
    for world in config['worlds']:
        summaries[world] = {}
        for arm in ('old','fixed_pseudocount'):
            selected = [r for r in rows if r['world']==world and r['arm']==arm and r['method']=='J']
            complete = len(selected)==15 and all(r['valid'] for r in selected)
            vals = [r['holdout_expected_nll_delta'] for r in selected]
            summaries[world][arm] = dict(complete=complete, passing_seeds=[r['seed'] for r in selected if r['joint_seed_pass']],
                valid_count=sum(r['valid'] for r in selected), mean_delta=float(np.mean(vals)) if complete else None,
                max_delta=max(vals) if complete else None, gate=bool(complete and (sum(r['joint_seed_pass'] for r in selected)>=12
                    if world=='well_specified' else np.mean(vals)<=.005 and max(vals)<=.02)),
                maximum_true_risk_gap=max(r['true_directional_risk_gap'] for r in selected),
                mean_reference_qmass=float(np.mean([r['reference_A_qmass'] for r in selected])))
    write(ROOT/'summary.json', dict(scope='paired post-seal mechanism diagnostic; not complete S or permission for P',
        reused_source_A=True, only_change='effective mass pseudo-count n_A to 1', summaries=summaries))
    final_paths = [ROOT/'endpoints.json',ROOT/'endpoints.csv',ROOT/'summary.json',*sorted((ROOT/'blocks').glob('*/holdout.npz'))]
    signed(final_paths, 'DIAGNOSTIC_FINAL.json', obs_seal['digest'])
    print(json.dumps(summaries), flush=True)


def main():
    final = seals.verify_seal(ORIGINAL, 'R7_FINAL')
    plan = json.loads((ORIGINAL/'executed-plan.json').read_text())
    provenance = json.loads((ORIGINAL/'provenance.json').read_text())
    for path in SNAPSHOT.rglob('*.py'):
        relative = path.relative_to(SNAPSHOT).as_posix()
        if relative in provenance['file_hashes'] and sha(path) != provenance['file_hashes'][relative]:
            raise ValueError('Pinned execution source mismatch: '+relative)
    seeds = list(range(65,80))
    assert plan['phases']['S']['seeds']==seeds
    config = dict(worlds=['well_specified','no_shift'], seeds=seeds,
        tasks=[(world,seed) for world in ('well_specified','no_shift') for seed in seeds], workers=4,
        primary_tau=.005, effective_pseudo_count=1, original_mass_weight=1,
        exact_change='posterior_mass_options.mass_weight=1/n_adaptation',
        parent_final=final['digest'], execution_commit=provenance['git_commit'],
        original_plan_sha256=sha(ORIGINAL/'executed-plan.json'), script_sha256=sha(__file__),
        target_truth_before_observation_seal=False, diagnostic_only=True,
        original_gate_thresholds_unchanged=plan['phases']['S']['gate'],
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=str(torch.__version__)))
    write(ROOT/'config.json',config)
    write(ROOT/'code-hashes.json',{str(p.relative_to(SNAPSHOT)):sha(p) for p in sorted(SNAPSHOT.rglob('*.py'))})
    (ROOT/'blocks').mkdir()
    initial = signed([ROOT/'config.json',ROOT/'code-hashes.json',ROOT/'diagnostic.py'],'DIAGNOSTIC_INPUT.json')
    completed = []
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_one,task) for task in config['tasks']]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(f'A_ONLY {len(completed)}/30 '+json.dumps(result),flush=True)
    write(ROOT/'fit-progress.json',completed)
    verify(initial)
    paths = [ROOT/'fit-progress.json',*sorted((ROOT/'blocks').glob('*/a.npz')),*sorted((ROOT/'blocks').glob('*/fit.json'))]
    observed = signed(paths,'DIAGNOSTIC_OBS.json',initial['digest'])
    print('ALL_A_FITS_AND_ACTIONS_SEALED '+observed['digest'],flush=True)
    evaluate(config,observed)


if __name__=='__main__':
    try:
        main()
    except Exception:
        write(ROOT/'FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
