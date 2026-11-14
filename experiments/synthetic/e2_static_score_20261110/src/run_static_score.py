"""Run five one-factor static score settings. Never adapt a classifier."""
from experiments.common.paths import repository_root
import argparse
import json
from pathlib import Path
import resource
import signal
import time
import traceback
import torch
from tsba.classifier import Coordinates
from tsba.config import Config
from experiments.common.io import digest, manifest, provenance, save_tensor, write_json
from experiments.synthetic.e1_e2_20261109.src.distributions import oracles
from experiments.synthetic.e2_controls_20261109.src.execution import make_head
from experiments.synthetic.e2_full_source_20261110.src.execution import verify_archive
from experiments.synthetic.e2_static_score_20261110.src.estimation import SETTINGS, audit_scores, enlarged_pool, evaluate, fit, new_model, schedules, tensor_hash


def read(p):
    return json.loads(p.read_text())


def run_cell(parent, root, dim, seed, log):
    begin = time.perf_counter()
    directory = root/f'd{dim}_seed{seed}'
    directory.mkdir()
    old = parent/directory.name
    inputs = torch.load(old/'inputs.pt', weights_only=True)
    cfg = Config(**read(old/'config.json'))
    if (cfg.score_epochs, cfg.score_batch, cfg.score_validation) != (100, 128, True):
        raise ValueError('parent training budget changed')
    coords = Coordinates(**inputs['coordinates'])
    head = make_head(inputs).requires_grad_(False).eval()
    frozen = {k: tensor_hash(v) for k, v in head.state_dict().items()}
    # Do not load evaluation_labels.pt or any trajectory, and do not rebuild queries.
    old_queries = torch.load(old/'queries.pt', weights_only=True)
    queries = {k: {kk: vv for kk, vv in b.items() if not kk.startswith('score_')}
               for k, b in old_queries.items()}
    target = enlarged_pool(inputs['target'], coords, seed)
    schedule = schedules(dim, seed)
    save_tensor(directory/'fixed-inputs.pt', dict(target=target, coordinates=inputs['coordinates'],
                initial_head=inputs['initial_head'], baseline_sigma=inputs['sigma'], queries=queries))
    save_tensor(directory/'schedule.pt', schedule)
    write_json(directory/'config.json', cfg.to_dict())
    # One common eligibility mask across ALL sigma settings prevents denominator drift.
    masks = {}
    for name, bank in queries.items():
        if 'normals' not in bank:
            continue
        mask = torch.ones(len(bank['points']), dtype=torch.bool)
        for multiplier in (.5, 1., 2.):
            sigma = multiplier*inputs['sigma']
            pop, _ = oracles(target[:512], coords, sigma)
            gamma = (evaluate(pop, bank['points'])*bank['normals']).sum(-1)
            mask &= (sigma*gamma).abs() > 1e-3
        masks[name] = mask
    save_tensor(directory/'common-masks.pt', masks)
    results = []
    for setting, n, mult in SETTINGS:
        start = time.perf_counter()
        dest = directory/setting
        dest.mkdir()
        sigma = inputs['sigma']*mult
        def save_checkpoint(payload):
            save_tensor(dest/'checkpoint.pt', dict(**payload, dim=dim, sigma=sigma, hidden=cfg.score_hidden))
        learned, training, best = fit(target[:n], sigma, cfg, schedule,
            log=lambda m: log(f'd={dim} seed={seed} {setting} {m}'), checkpoint=save_checkpoint)
        write_json(dest/'training.json', training)
        selected = new_model(dim, cfg).requires_grad_(False).eval()
        selected.load_state_dict(best)
        checkpoint = torch.load(dest/'checkpoint.pt', weights_only=True)
        restored = new_model(dim, cfg).requires_grad_(False).eval()
        restored.load_state_dict(checkpoint['final_state'])
        pop, emp = oracles(target[:n], coords, sigma)
        save_tensor(dest/'oracle.pt', dict(population=pop.state_dict(), empirical=emp.state_dict()))
        measured, per_query = {}, {}
        for name, bank in queries.items():
            points, normals = bank['points'], bank.get('normals')
            values = dict(learned=evaluate(learned, points), empirical=evaluate(emp, points),
                          population=evaluate(pop, points), selected=evaluate(selected, points), zero=torch.zeros_like(points))
            torch.testing.assert_close(evaluate(restored, points), values['learned'], atol=0, rtol=0)
            raw_x1 = points[:, 0]*coords.scale+coords.mean[0]
            measured[name], per_query[name] = audit_scores(points, normals, values, sigma, float(coords.scale), cfg,
                                                         masks.get(name), raw_x1)
        if setting == 'n512_s1':
            previous = torch.load(old/'score.pt', weights_only=True)['state_dict']
            for name in best:
                torch.testing.assert_close(best[name], previous[name], atol=0, rtol=0)
            old_training = read(old/'score_training.json')
            assert training['selected_update'] == 4*(old_training['selected_epoch']+1)
        save_tensor(dest/'per-query.pt', per_query)
        write_json(dest/'metrics.json', measured)
        row = dict(dim=dim, seed=seed, setting=setting, samples=n, sigma_multiplier=mult, sigma=sigma,
                   raw_sigma=sigma*float(coords.scale), updates=training['updates'],
                   selected_update=training['selected_update'], seconds=time.perf_counter()-start,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   pool_hash=tensor_hash(target[:n]), noise_hash=tensor_hash(schedule['noise']),
                   initial_weights_hash={k: tensor_hash(v) for k, v in checkpoint['initial_state'].items()},
                   query_hashes={k: {kk: tensor_hash(vv) for kk, vv in b.items() if isinstance(vv, torch.Tensor)} for k, b in queries.items()},
                   checkpoint_predictions_bitwise=True, historical_checkpoint_bitwise=setting == 'n512_s1')
        if row['peak_rss_bytes'] > 4294967296:
            raise MemoryError('4 GiB RSS budget exceeded')
        write_json(dest/'summary.json', row)
        results.append(row)
        log(f'SETTING COMPLETE d={dim} seed={seed} {setting} seconds={row["seconds"]:.1f}')
    assert frozen == {k: tensor_hash(v) for k, v in head.state_dict().items()}
    write_json(directory/'cell-summary.json', dict(dim=dim, seed=seed, seconds=time.perf_counter()-begin,
               settings=results, classifier_unchanged=True, query_parent_sha256=digest(old/'queries.pt'),
               inputs_parent_sha256=digest(old/'inputs.pt')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = repository_root()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    write_json(root/'provenance.json', provenance(repo))
    protocol = repo/'experiments/synthetic/e2_static_score_20261110/configs/static-score-plan.json'
    (root/'plan.json').write_bytes(protocol.read_bytes())
    integrity = verify_archive(args.parent, repo/'experiments/synthetic/e1_e2_20261109/results/run-manifest.json')
    write_json(root/'parent-integrity-before.json', integrity)
    failures, completed, start = [], [], time.perf_counter()
    def timeout(signum, frame):
        raise TimeoutError('1200 second cell budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    with (root/'run.log').open('x', buffering=1) as file:
        def log(s):
            line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())+' '+s
            print(line, flush=True)
            file.write(line+'\n')
        for dim in (2, 16):
            for seed in range(5):
                signal.alarm(1200)
                try:
                    if time.perf_counter()-start > 14400:
                        raise TimeoutError('4 hour global budget exceeded')
                    run_cell(args.parent, root, dim, seed, log)
                    completed.append([dim, seed])
                except Exception as exc:
                    failures.append(dict(dim=dim, seed=seed, error=repr(exc), traceback=traceback.format_exc()))
                    log(f'CELL FAILED d={dim} seed={seed}: {exc}')
                finally:
                    signal.alarm(0)
        log(f'COMPLETE cells={len(completed)}/10 failures={len(failures)} seconds={time.perf_counter()-start:.1f}')
    write_json(root/'failures.json', failures)
    write_json(root/'completion.json', dict(completed=completed, failures=len(failures), seconds=time.perf_counter()-start,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    write_json(root/'parent-integrity-after.json', verify_archive(args.parent, repo/'experiments/synthetic/e1_e2_20261109/results/run-manifest.json'))
    write_json(root/'label-access.json', dict(real_target_labels=False, synthetic_evaluation_labels=False,
               synthetic_component_ids='generated only by input sampler, discarded immediately; never passed to trainer'))
    write_json(root/'manifest.json', manifest(root))
    if failures:
        raise SystemExit(1)
