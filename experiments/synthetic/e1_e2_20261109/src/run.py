"""Run frozen E1/E2, then seal every prediction before reading synthetic eval labels.

python -m experiments.synthetic.e1_e2_20261109.src.run --output /absolute/new/run-directory
No raw graph data or real target labels are loaded by this entrypoint.
"""
from experiments.common.paths import repository_root
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import resource
import signal
import time
import traceback
import torch
from torch.func import jacrev, jvp
from torch.nn import functional as F
from tsba.adapt import adapt_classifier, backtrack
from tsba.boundary import BoundaryBatch, build_boundaries
from tsba.classifier import ParameterMap
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.score import bandwidth, train_score
from tsba.subproblem import GGNSystem, solve
from experiments.synthetic.e1_e2_20261109.src.analytic import run_analytic
from experiments.common.io import digest, manifest, provenance, save_tensor, write_json
from experiments.synthetic.e1_e2_20261109.src.distributions import ZeroScore, generator, oracles, query_banks, setup_cell
from experiments.common.geometry import compare_geometry
from experiments.common.metrics import classification, score_audit

ARMS=('learned','empirical','population','zero')


def boundary_subset(b, limit):
    return BoundaryBatch(**{k:(v[:limit] if isinstance(v,torch.Tensor) else v) for k,v in vars(b).items()})


def nonlinear_probe(cell, pop, sigma, cfg):
    head=copy.deepcopy(cell['head'])
    with torch.no_grad():
        head.U2.weight.copy_(.1*torch.randn(head.U2.weight.shape,generator=generator(cfg.seed+6000),dtype=head.U2.weight.dtype))
    b=build_boundaries(head,cell['target'][:128],pop,sigma,cfg)
    saved=dict(head=head.state_dict(),boundaries=vars(b))
    summary=dict(anchors=len(b),rejected=b.rejected,steps=[])
    if not len(b):
        summary['reason']='no_valid_boundary';return summary,saved
    problem=TsbaProblem(head,cell['source'][:128],cell['source_y'][:128],b,cfg)
    delta,info=solve(problem)
    summary['solver']=info
    if delta is None or not info['valid']:return summary,saved
    saved.update(base=problem.base,delta=delta,measurements={})
    # Unique roots only: ascending alpha follows the sole available local branch.
    for alpha in [.0625,.125,.25,.5,1.]:
        metrics, arrays=compare_geometry(head,b,problem.base,problem.base+alpha*delta,sigma)
        summary['steps'].append(dict(alpha=alpha,**metrics));saved['measurements'][alpha]=arrays
    return summary,saved


def decompose(model, payload, cell, cfg, sigma):
    """Matched zero-request candidate before any nonlinear line search."""
    params=ParameterMap(model);original=params.vector()
    try:
        params.commit(payload['before'])
        b=BoundaryBatch(**payload['boundaries'])
        ids=payload['source_ids']
        problem=TsbaProblem(model,cell['source'][ids],cell['source_y'][ids],b,cfg)
        system=GGNSystem(problem);matrix=system.dense()
        d=jacrev(problem.normalized_margin)(problem.base).detach()
        delta_s=torch.linalg.solve(matrix,-system.gs)
        delta_t=torch.linalg.solve(matrix,system.coefficient*d.T@b.zeta)
        delta=payload['delta']
        error=float((delta_s+delta_t-delta).norm())
        zproblem=TsbaProblem(model,cell['source'][ids],cell['source_y'][ids],replace(b,zeta=torch.zeros_like(b.zeta)),cfg)
        zcandidate,ztrials=backtrack(zproblem,delta_s,float(system.gs@delta_s),sigma)
        metrics=dict(round=payload['record']['round'],delta_source_norm=float(delta_s.norm()),
                     delta_request_norm=float(delta_t.norm()),decomposition_error=error,
                     source_response_rms=float((d@delta_s).square().mean().sqrt()),
                     request_response_rms=float((d@delta_t).square().mean().sqrt()),
                     zero_zeta_accepted=zcandidate is not None,zero_zeta_trials=ztrials)
        return metrics,dict(delta_source=delta_s,delta_request=delta_t,D=d,g_source=system.gs,
                            A=matrix,zero_zeta_candidate=zcandidate)
    finally:params.commit(original)


def run_cell(dim,seed,root,log):
    start=time.perf_counter();cfg=Config(seed=seed,emb_dim=dim)
    directory=root/f'd{dim}_seed{seed}';directory.mkdir()
    write_json(directory/'config.json',cfg.to_dict())
    cell=setup_cell(dim,seed,cfg)
    # Evaluator label file is deliberately excluded from the training payload.
    save_tensor(directory/'evaluation_labels.pt',cell.pop('evaluation_y'))
    sigma=bandwidth(cell['target'],cfg.kappa_sigma)
    pop,emp=oracles(cell['target'],cell['coordinates'],sigma)
    tensors={k:v for k,v in cell.items() if isinstance(v,torch.Tensor)}
    save_tensor(directory/'inputs.pt',dict(**tensors,coordinates=cell['coordinates'].state_dict(),
                                          initial_head=cell['head'].state_dict(),sigma=sigma))
    initial=cell['head']
    with torch.no_grad():save_tensor(directory/'initial_predictions.pt',initial(cell['evaluation']))
    score_start=time.perf_counter()
    learned,score_history=train_score(cell['target'],sigma,cfg)
    score_seconds=time.perf_counter()-score_start
    save_tensor(directory/'score.pt',dict(state_dict=learned.state_dict(),dim=dim,hidden=cfg.score_hidden,sigma=sigma))
    write_json(directory/'score_training.json',score_history)
    scores=dict(learned=learned,empirical=emp,population=pop,zero=ZeroScore())
    probe,probe_arrays=nonlinear_probe(cell,pop,sigma,cfg)
    write_json(directory/'E1_nonlinear.json',probe);save_tensor(directory/'E1_nonlinear.pt',probe_arrays)
    initial_b=build_boundaries(initial,cell['target'][:128],pop,sigma,cfg)
    banks=query_banks(cell,sigma,seed)
    banks['initial_boundaries']=dict(points=initial_b.points,normals=initial_b.normals,pairs=initial_b.pairs,
                                     input_indices=initial_b.input_indices)
    bank_metrics={}
    for name,bank in banks.items():
        metric,arrays=score_audit(bank['points'],bank.get('normals'),scores,sigma)
        bank_metrics[name]=metric;bank.update(arrays)
    save_tensor(directory/'queries.pt',banks);write_json(directory/'score_queries.json',bank_metrics)
    summary=dict(dim=dim,seed=seed,sigma=sigma,coordinate_scale=float(cell['coordinates'].scale),
                 raw_sigma=sigma*float(cell['coordinates'].scale),score_seconds=score_seconds,
                 restricted_translation_cap_raw=30*.1*sigma*float(cell['coordinates'].scale),
                 score_selected_epoch=score_history['selected_epoch'],
                 score_validation_loss=score_history['noise_validation_loss'],arms={})
    for arm in ARMS:
        head=copy.deepcopy(initial);events=[]
        def observe(payload):
            events.append(payload)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>4294967296:
                raise MemoryError('RSS exceeded 4 GiB budget')
        arm_start=time.perf_counter()
        arm_dir=directory/arm;arm_dir.mkdir()
        try:
            trace=adapt_classifier(head,cell['source'],cell['source_y'],cell['target'],scores[arm],sigma,cfg,
                                   logger=lambda s:log(f'd={dim} seed={seed} arm={arm} {s}'),observer=observe)
        finally:
            # Preserve partial attempts and the last live state on timeout/error.
            save_tensor(arm_dir/'rounds.pt',events)
            save_tensor(arm_dir/'classifier.pt',head.state_dict())
        seconds=time.perf_counter()-arm_start
        with torch.no_grad():
            logits=head(cell['evaluation']);source_ce=float(F.cross_entropy(head(cell['source']),cell['source_y']))
        save_tensor(arm_dir/'predictions.pt',logits)
        write_json(arm_dir/'adaptation.json',trace)
        if arm=='zero':
            torch.testing.assert_close(ParameterMap(head).vector(),ParameterMap(initial).vector(),atol=0,rtol=0)
        summary['arms'][arm]=dict(accepted=trace['accepted'],stop_reason=trace['stop_reason'],
                                 adaptation_seconds=seconds,source_ce=source_ce,attempted_rounds=len(trace['history']))
    # Sealed trajectories: all following diagnostics are offline and cannot feed updates.
    write_json(directory/'trajectory-seal.json',{arm:digest(directory/arm/'rounds.pt') for arm in ARMS})
    reference=torch.load(directory/'learned'/'rounds.pt',weights_only=True)
    for arm in ARMS[1:]:
        events=torch.load(directory/arm/'rounds.pt',weights_only=True)
        for left,right in zip(reference,events):
            for key in ('source_ids','target_ids'):
                assert torch.equal(left[key],right[key]),'paired batch schedule mismatch'
    observation_start=time.perf_counter()
    for arm in ARMS:
        arm_dir=directory/arm
        events=torch.load(arm_dir/'rounds.pt',weights_only=True)
        rows=[];geometry_arrays=[];response_arrays=[];decomposed=False
        head=copy.deepcopy(initial)
        for event in events:
            rec=event['record'];bdata=event['boundaries']
            row=dict(round=rec['round'],accepted=rec['accepted'],reason=rec['reason'])
            if bdata is not None:
                b=BoundaryBatch(**bdata)
                audit,_=score_audit(b.points,b.normals,scores,sigma)
                row.update(anchors=len(b),unique_anchors=len(torch.unique(b.points,dim=0)),score=audit)
                if len(b):
                    with torch.no_grad():gamma=(scores[arm](b.points)*b.normals).sum(-1)
                    quantiles=torch.quantile((sigma*gamma).abs(),torch.tensor([0.,.25,.5,.75,1.],dtype=gamma.dtype))
                    row.update(single_anchor_weight=1/len(b),abs_sigma_gamma_quantiles=quantiles.tolist(),
                               clipped_fraction=float((cfg.tau*sigma**2*gamma.abs()>=cfg.request_multiplier*sigma).double().mean()),
                               zero_request_fraction=float((b.zeta.abs()<=cfg.movement_zero*sigma).double().mean()),
                               request_abs_quantiles=torch.quantile(b.zeta.abs()/sigma,torch.tensor([0.,.5,1.],dtype=gamma.dtype)).tolist())
                    params=ParameterMap(head)
                    def normalized(vector):
                        logits=params.logits(vector,b.points).gather(1,b.pairs)
                        return (logits[:,0]-logits[:,1])/b.normal_norm
                    with torch.no_grad():
                        response=normalized(event['after'])-b.old_margin/b.normal_norm
                    linear=jvp(normalized,(event['before'],),(event['after']-event['before'],))[1].detach()
                    row['E_lin_all_anchors']=float(((response-linear)/sigma).square().mean().sqrt())
                    row['response_request_rmse_sigma']=float(((response-b.zeta)/sigma).square().mean().sqrt())
                    response_arrays.append(dict(round=rec['round'],v=response,linear=linear))
                    metrics,arrays=compare_geometry(head,boundary_subset(b,8),event['before'],event['after'],sigma)
                    row['geometry_first8']=metrics
                    geometry_arrays.append(dict(round=rec['round'],boundary_indices=torch.arange(min(8,len(b))),**arrays))
                    if event['delta'] is not None and not decomposed:
                        decomp,data=decompose(head,event,cell,cfg,sigma)
                        write_json(arm_dir/'first_active_decomposition.json',decomp)
                        save_tensor(arm_dir/'first_active_decomposition.pt',data);decomposed=True
            rows.append(row)
        write_json(arm_dir/'observations.json',rows)
        save_tensor(arm_dir/'geometry.pt',geometry_arrays)
        save_tensor(arm_dir/'responses.pt',response_arrays)
    summary['observation_seconds']=time.perf_counter()-observation_start
    summary['total_seconds']=time.perf_counter()-start
    summary['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    write_json(directory/'cell_summary.json',summary)
    log(f'CELL COMPLETE d={dim} seed={seed} sigma={sigma:.8g} seconds={summary["total_seconds"]:.1f}')
    return summary


def evaluate_sealed(root,completed):
    # Verify all predictions, checkpoints and observations against a fixed seal.
    seal=manifest(root,exclude=('run.log',))
    write_json(root/'prediction-seal.json',seal)
    rows=[];access=[]
    for dim,seed in completed:
        directory=root/f'd{dim}_seed{seed}'
        label_path=directory/'evaluation_labels.pt'
        labels=torch.load(label_path,weights_only=True)
        access.append(dict(path=str(label_path.relative_to(root)),purpose='synthetic final evaluation after global seal',
                           time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        initial=torch.load(directory/'initial_predictions.pt',weights_only=True)
        initial_metrics=classification(initial,labels)
        for arm in ARMS:
            path=directory/arm/'predictions.pt'
            assert digest(path)==seal[str(path.relative_to(root))]['sha256']
            metrics=classification(torch.load(path,weights_only=True),labels)
            rows.append(dict(dim=dim,seed=seed,arm=arm,initial=initial_metrics,final=metrics,
                             delta_ba_pp=100*(metrics['balanced_accuracy']-initial_metrics['balanced_accuracy']),
                             delta_nll=metrics['nll']-initial_metrics['nll']))
    write_json(root/'label-access.json',access)
    write_json(root/'evaluation.json',rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    repo=repository_root()
    write_json(root/'provenance.json',provenance(repo))
    for name in ['plan.json','validation-plan.md']:
        (root/name).write_bytes((repo/'experiments'/('synthetic/e1_e2_20261109/configs' if name == 'plan.json' else '')/name).read_bytes())
    write_json(root/'E1_analytic.json',run_analytic())
    failures=[];completed=[];start=time.perf_counter()
    def timeout(signum,frame):raise TimeoutError('predeclared 1200 second cell budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    with open(root/'run.log','x',buffering=1) as logfile:
        def log(s):
            line=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+' '+s
            print(line,flush=True);logfile.write(line+'\n')
        for dim in [2,16]:
            for seed in range(5):
                signal.alarm(1200)
                try:
                    if time.perf_counter()-start>14400:raise TimeoutError('global budget exceeded')
                    summary=run_cell(dim,seed,root,log)
                    if summary['process_peak_rss_bytes']>4294967296:raise MemoryError('RSS exceeded 4 GiB budget')
                    completed.append((dim,seed))
                except Exception as exc:
                    log(f'CELL FAILED d={dim} seed={seed}: {exc}')
                    failures.append(dict(dim=dim,seed=seed,error=repr(exc),traceback=traceback.format_exc()))
                finally:signal.alarm(0)
        write_json(root/'failures.json',failures)
        evaluate_sealed(root,completed)
        log(f'COMPLETE cells={len(completed)}/10 failures={len(failures)} seconds={time.perf_counter()-start:.1f}')
    write_json(root/'manifest.json',manifest(root))
