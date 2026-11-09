"""Run the frozen post-E2 controls using the original archive's exact inputs/IDs."""
import argparse
import copy
import json
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback
import torch
from torch.func import jacrev
from torch.nn import functional as F
from tsba.boundary import BoundaryBatch
from tsba.classifier import ParameterMap, ResidualClassifier
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.subproblem import GGNSystem
from .archive import digest, manifest, provenance, save_tensor, write_json
from .controls import ARMS, SourceGGN, SourceProblem, adapt_control
from .metrics import classification

PARENT_ARMS = ('learned', 'empirical', 'population', 'zero')


def load_json(path):
    return json.loads(path.read_text())


def verify_parent(parent, repo):
    expected = repo/'docs/tsba/experiments/e1-e2-20261109/run-manifest.json'
    if digest(parent/'manifest.json') != digest(expected):
        raise ValueError('parent manifest differs from the published archive')
    items = load_json(expected)
    for relative, record in items.items():
        p = parent/relative
        if p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise ValueError(f'parent archive changed: {relative}')
    return dict(parent_path=str(parent), manifest_sha256=digest(expected), files_verified=len(items))


def make_head(inputs):
    d = inputs['source'].shape[1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        head = ResidualClassifier(d, 2, min(d, 16), dtype=inputs['source'].dtype).eval()
    head.load_state_dict(inputs['initial_head'])
    return head


def prepare_cell(parent, destination):
    destination.mkdir()
    # Opaque copy only: evaluator labels are not decoded during adaptation.
    for name in ('inputs.pt', 'config.json', 'evaluation_labels.pt', 'initial_predictions.pt'):
        shutil.copy2(parent/name, destination/name)
    shutil.copy2(parent/'cell_summary.json', destination/'parent_cell_summary.json')
    records = torch.load(parent/'learned'/'rounds.pt', weights_only=True)
    schedule = [dict(source_ids=r['source_ids'], target_ids=r['target_ids']) for r in records]
    save_tensor(destination/'schedule.pt', schedule)
    masks = {}
    for arm in PARENT_ARMS:
        old = torch.load(parent/arm/'rounds.pt', weights_only=True)
        if len(old) != len(records):
            raise ValueError('parent arm has an incomplete schedule')
        for a,b in zip(old,records):
            for key in ('source_ids','target_ids'):
                if not torch.equal(a[key],b[key]):raise ValueError('parent batch mismatch')
        cfg = Config(**load_json(parent/'config.json'))
        sigma = torch.load(parent/'inputs.pt', weights_only=True)['sigma']
        masks[arm] = [r['boundaries'] is not None and len(r['boundaries']['points']) > 0
                      and bool((r['boundaries']['zeta'].abs() > cfg.movement_zero*sigma).any()) for r in old]
        shutil.copy2(parent/arm/'predictions.pt', destination/f'parent_{arm}_predictions.pt')
    write_json(destination/'parent_attempt_masks.json', masks)
    save_tensor(destination/'parent_first_round.pt', records[0])
    return schedule


def first_round_audit(inputs, cfg, parent_event, events):
    zero_event = events['zero_zeta_geometry'][0]
    source_event = events['source_ce_ggn'][0]
    for event in (zero_event, source_event):
        torch.testing.assert_close(event['before'],parent_event['before'],rtol=0,atol=0)
    for key in ('points','pairs','normal_norm','normals','old_margin','input_indices'):
        torch.testing.assert_close(zero_event['boundaries'][key],parent_event['boundaries'][key],rtol=0,atol=0)
    assert torch.count_nonzero(zero_event['boundaries']['zeta']) == 0
    model = make_head(inputs)
    ids = parent_event['source_ids']
    b = BoundaryBatch(**parent_event['boundaries'])
    original = TsbaProblem(model,inputs['source'][ids],inputs['source_y'][ids],b,cfg)
    system = GGNSystem(original)
    matrix = system.dense()
    zero_source = torch.linalg.solve(matrix,-system.gs)
    torch.testing.assert_close(zero_event['delta'],zero_source,atol=1e-12,rtol=1e-12)
    d = jacrev(original.normalized_margin)(original.base).detach()
    source = SourceGGN(SourceProblem(model,inputs['source'][ids],inputs['source_y'][ids],cfg))
    curvature_error = (matrix-source.dense()-cfg.lambda_boundary/len(b)*d.T@d).abs().max()
    assert float(curvature_error) < 1e-12
    torch.testing.assert_close(source_event['delta'],torch.linalg.solve(source.dense(),-system.gs),atol=1e-12,rtol=1e-12)
    return dict(first_reference_bitwise_equal=True, zero_request_exactly_zero=True,
                zero_request_delta_max_error=float((zero_event['delta']-zero_source).abs().max()),
                removed_curvature_identity_max_error=float(curvature_error),
                later_round_matrices_identical_claim=False)


def run_cell(parent, directory, dim, seed, log):
    started = time.perf_counter()
    schedule = prepare_cell(parent, directory)
    inputs = torch.load(directory/'inputs.pt',weights_only=True)
    cfg = Config(**load_json(directory/'config.json'))
    if cfg.seed != seed or inputs['source'].shape[1] != dim:raise ValueError('cell identity mismatch')
    initial = make_head(inputs)
    with torch.no_grad():
        torch.testing.assert_close(initial(inputs['evaluation']),torch.load(directory/'initial_predictions.pt',weights_only=True),atol=0,rtol=0)
        source_initial = float(F.cross_entropy(initial(inputs['source']), inputs['source_y']))
    summary = dict(dim=dim,seed=seed,source_initial_ce=source_initial,arms={})
    all_events = {}
    for arm in ARMS:
        head=copy.deepcopy(initial);events=[];arm_dir=directory/arm;arm_dir.mkdir()
        def observe(payload):
            events.append(payload)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4294967296:
                raise MemoryError('4 GiB process RSS budget exceeded')
        arm_start=time.perf_counter()
        try:
            trace=adapt_control(head,inputs['source'],inputs['source_y'],inputs['target'],inputs['sigma'],
                                cfg,schedule,arm,observer=observe,
                                logger=lambda s:log(f'd={dim} seed={seed} arm={arm} {s}'))
        finally:
            save_tensor(arm_dir/'rounds.pt',events)
            save_tensor(arm_dir/'classifier.pt',head.state_dict())
        seconds=time.perf_counter()-arm_start
        with torch.no_grad():
            save_tensor(arm_dir/'predictions.pt',head(inputs['evaluation']))
            source_ce=float(F.cross_entropy(head(inputs['source']),inputs['source_y']))
        write_json(arm_dir/'adaptation.json',trace)
        summary['arms'][arm]=dict(accepted=trace['accepted'],stop_reason=trace['stop_reason'],
                                 rounds=len(trace['history']),seconds=seconds,source_ce=source_ce)
        all_events[arm]=events
    audit=first_round_audit(inputs,cfg,torch.load(directory/'parent_first_round.pt',weights_only=True),all_events)
    write_json(directory/'first_round_audit.json',audit)
    summary['seconds']=time.perf_counter()-started
    summary['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write_json(directory/'cell_summary.json',summary)
    log(f'CELL COMPLETE d={dim} seed={seed} seconds={summary["seconds"]:.1f}')


def evaluate(root, cells):
    seal=manifest(root,exclude=('run.log',))
    write_json(root/'prediction-seal.json',seal)
    evaluations=[];access=[]
    for dim,seed in cells:
        directory=root/f'd{dim}_seed{seed}'
        labels=torch.load(directory/'evaluation_labels.pt',weights_only=True)
        access.append(dict(path=str((directory/'evaluation_labels.pt').relative_to(root)),
                           time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                           purpose='post-hoc synthetic paired evaluation after sealing all new predictions'))
        initial=classification(torch.load(directory/'initial_predictions.pt',weights_only=True),labels)
        for arm in (*PARENT_ARMS,*ARMS):
            path=(directory/f'parent_{arm}_predictions.pt') if arm in PARENT_ARMS else directory/arm/'predictions.pt'
            if digest(path)!=seal[str(path.relative_to(root))]['sha256']:raise ValueError('prediction seal mismatch')
            final=classification(torch.load(path,weights_only=True),labels)
            evaluations.append(dict(dim=dim,seed=seed,arm=arm,initial=initial,final=final,
                                    delta_ba_pp=100*(final['balanced_accuracy']-initial['balanced_accuracy']),
                                    delta_nll=final['nll']-initial['nll']))
    write_json(root/'label-access.json',access)
    write_json(root/'evaluation.json',evaluations)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    repo=Path(__file__).resolve().parents[3]
    parent=args.parent.resolve()
    audit=verify_parent(parent,repo)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    write_json(root/'parent-integrity.json',audit)
    write_json(root/'provenance.json',provenance(repo))
    shutil.copy2(repo/'docs/tsba/experiments/controls-plan.json',root/'controls-plan.json')
    shutil.copy2(repo/'docs/tsba/experiments/validation-plan.md',root/'validation-plan.md')
    failures=[];completed=[];started=time.perf_counter()
    def timeout(signum,frame):raise TimeoutError('predeclared 1200 second cell budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    with (root/'run.log').open('x',buffering=1) as logfile:
        def log(message):
            line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+message
            print(line,flush=True);logfile.write(line+'\n')
        for dim in (2,16):
            for seed in range(5):
                signal.alarm(1200)
                try:
                    if time.perf_counter()-started>14400:raise TimeoutError('global budget exceeded')
                    run_cell(parent/f'd{dim}_seed{seed}',root/f'd{dim}_seed{seed}',dim,seed,log)
                    completed.append((dim,seed))
                except Exception as exc:
                    failures.append(dict(dim=dim,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
                    log(f'CELL FAILED d={dim} seed={seed}: {exc}')
                finally:signal.alarm(0)
        write_json(root/'failures.json',failures)
        evaluate(root,completed)
        log(f'COMPLETE cells={len(completed)}/10 failures={len(failures)} seconds={time.perf_counter()-started:.1f}')
    write_json(root/'manifest.json',manifest(root))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
