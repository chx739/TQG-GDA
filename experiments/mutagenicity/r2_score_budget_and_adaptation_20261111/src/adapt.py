"""Production 30-attempt dynamics, with an explicit geometric-source control.

Only the research control bypasses the production all-zero request guard.
All normal/reverse paths call the production boundary, solve and backtrack.
"""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch
from torch.func import jacrev

from tsba.adapt import backtrack
from tsba.boundary import build_boundaries
from tsba.classifier import ParameterMap
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.subproblem import GGNSystem, solve
from tsba.utils import NumericalError, finite
from experiments.common.runtime import rss_bytes
from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.steps import EmpiricalScore
from experiments.mutagenicity.r2_score_budget_and_adaptation_20261111.src.common import write_json, load, atomic_tensor, model_from_frozen, job_key, pools, predictions, tensor_sha
from experiments.mutagenicity.r2_score_budget_and_adaptation_20261111.src.score import comparison, restore


class ZeroScore(torch.nn.Module):
    def forward(self,x):return torch.zeros_like(x)


def settings(seed,lam,rounds=30):
    return Config(seed=seed,adapt_rounds=rounds,source_adapt_batch=128,target_batch=128,
        lambda_boundary=float(lam),damping=.01,tau=.1,request_multiplier=.1,
        execution_multiplier=.1,radius_multiplier=1.,adaptation_dtype='float64')


def score_for(root,spec,x,target_ids,sigma0):
    if spec['kind']=='geometry':return ZeroScore()
    if spec['kind']=='kde':return EmpiricalScore(x[target_ids],spec['bandwidth']*sigma0)
    path=Path(root)/f"scores/seed{spec['seed']}_domain{spec['domain']}/b{spec['bandwidth']}/checkpoints/u{spec['updates']:05d}.pt"
    model,_,_=restore(load(path));return model.requires_grad_(False).eval()


def one_attempt(model,source,labels,target,score,sigma0,cfg,*,sign=1,force_geometry=False,empirical=None):
    """No randomness; inputs are the pre-generated production batches."""
    before=ParameterMap(model).vector();started=time.monotonic()
    rec=dict(accepted=False,alpha=0.,status='skipped');payload=dict(before_vector=before)
    try:
        b=build_boundaries(model,target,score,sigma0,cfg)
        # Reversal changes only the clipped request; geometry remains identical
        # for identical model and batch inputs.
        b=replace(b,zeta=(torch.zeros_like(b.zeta) if force_geometry else sign*b.zeta))
        payload['boundaries']=asdict(b)
        rec.update(anchors=len(b),unique_generating_batch_positions=int(b.input_indices.unique().numel()),
            projection_rejections=b.rejected)
        if not len(b):rec['reason']='no_valid_boundary'
        else:
            raw=(score(b.points).detach()*b.normals).sum(-1)*cfg.tau*sigma0**2
            rec.update(request_rms_sigma=float((b.zeta/sigma0).square().mean().sqrt()),
                max_request_sigma=float(b.zeta.abs().max()/sigma0),
                request_clipping_rate=float((raw.abs()>=cfg.request_multiplier*sigma0).double().mean()))
            if empirical is not None:
                with torch.no_grad():payload['current_score_comparison']=comparison(empirical(b.points),score(b.points),b.normals,sigma0)
            problem=TsbaProblem(model,source,labels,b,cfg)
            rec['before']=problem.evaluate(problem.base)
            # The extra algebra is passive and records the current system, never
            # a system cached from a prior round or a different trajectory.
            system=GGNSystem(problem);matrix=system.dense()
            d=jacrev(problem.normalized_margin)(problem.base).detach()
            payload.update(source_gradient=system.gs,D=d,matrix_sha256=tensor_sha(matrix))
            rec['matrix_sha256']=payload['matrix_sha256']
            if sign==1 and not force_geometry:
                rhs_t=system.coefficient*system.margin_pullback(b.zeta)[0].detach()
                ds=torch.linalg.solve(matrix,-system.gs);dt=torch.linalg.solve(matrix,rhs_t)
                finite(ds,dt);payload.update(delta_source=ds,delta_target=dt)
                ns,nt=float(ds.norm()),float(dt.norm())
                rec.update(source_proposal_norm=ns,target_proposal_norm=nt,
                    proposal_cosine=float((ds@dt)/(ds.norm()*dt.norm())) if ns and nt else None,
                    target_source_norm_ratio=nt/ns if ns else None,
                    source_margin_rms_sigma=float((d@ds/sigma0).square().mean().sqrt()),
                    target_margin_rms_sigma=float((d@dt/sigma0).square().mean().sqrt()))
            if not force_geometry and bool((b.zeta.abs()<=cfg.movement_zero*sigma0).all()):
                rec['reason']='zero_requests';payload['delta']=torch.zeros_like(before)
            else:
                delta,info=solve(problem);payload['delta']=delta;rec['solver']=info
                if not info['valid']:
                    rec.update(reason=info['reason'],status='skipped' if info['reason']=='stationary' else 'rejected')
                else:
                    if sign==1 and not force_geometry:
                        torch.testing.assert_close(delta,ds+dt,rtol=1e-7,atol=1e-9)
                    candidate,trials=backtrack(problem,delta,info['directional_derivative'],sigma0)
                    rec['trials']=trials
                    if candidate is None:rec.update(reason='backtracking_exhausted',status='rejected')
                    else:
                        movement=problem.movement(candidate).detach();finite(movement)
                        payload['actual_margin_movement']=movement
                        rec.update(accepted=True,reason='accepted',status='accepted',alpha=trials[-1]['alpha'],
                            after=trials[-1],actual_margin_rms_sigma=float((movement/sigma0).square().mean().sqrt()),
                            actual_margin_max_sigma=float(movement.abs().max()/sigma0),
                            movement_request_cosine=(float((movement@b.zeta)/(movement.norm()*b.zeta.norm()))
                                if float(movement.norm()) and float(b.zeta.norm()) else None))
                        if sign==1 and not force_geometry:
                            rec['executed_target_margin_rms_sigma']=rec['alpha']*rec['target_margin_rms_sigma']
                        problem.params.commit(candidate)
            if not rec['accepted']:
                payload['actual_margin_movement']=torch.zeros_like(b.zeta)
                rec.update(actual_margin_rms_sigma=0.,actual_margin_max_sigma=0.,executed_target_margin_rms_sigma=0.)
    except (NumericalError,torch.linalg.LinAlgError) as exc:
        ParameterMap(model).commit(before)
        rec.update(accepted=False,alpha=0.,reason='numerical_error',status='failed',error=str(exc))
    rec['elapsed_seconds']=time.monotonic()-started
    payload.update(record=rec,after_vector=ParameterMap(model).vector(),model_state=copy.deepcopy(model.state_dict()))
    return payload


def run_trajectory(root,spec,plan,*,output_root=None,stop_after=None):
    root=Path(root);out=Path(output_root) if output_root is not None else root/'trajectories'
    directory=out/job_key(spec);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return json.loads((directory/'complete.json').read_text())
    if (root/'seals/OBS.json').exists():raise PermissionError('Training closed after OBS')
    started=time.monotonic();seed,domain=spec['seed'],spec['domain']
    f=load(root/f'deps/source/seed{seed}/frozen.pt');model=model_from_frozen(f);x=f['x']
    bank=load(root/f'deps/geometry/seed{seed}_domain{domain}/banks/source.pt');sigma0=bank['sigma']
    schedule=load(root/f'batches/seed{seed}_domain{domain}.pt')
    labels=json.loads((root/f'source-labels/seed{seed}.json').read_text())
    score=score_for(root,spec,x,bank['target_ids'],sigma0)
    empirical=EmpiricalScore(x[bank['target_ids']],spec['bandwidth']*sigma0) if spec['kind']=='learned' and spec['sign']==1 else None
    cfg=settings(seed,spec['lam']);ids=pools(root,seed,domain)
    if not (directory/'round000.pt').exists():atomic_tensor(directory/'round000.pt',dict(
        spec=spec,attempt=0,model_state=copy.deepcopy(model.state_dict()),predictions=predictions(model,x,ids),sigma0=sigma0))
    existing=sorted(directory.glob('round[0-9][0-9][0-9].pt'))
    latest=load(existing[-1]);model.load_state_dict(latest['model_state']);start=latest['attempt']
    assert [int(p.stem[5:]) for p in existing]==list(range(start+1))
    failed=latest.get('record',{}).get('status')=='failed'
    for row in schedule[start:]:
        if failed:break
        s,t=row['source_ids'],row['target_ids'];y=torch.tensor([labels[str(i)] for i in s])
        payload=one_attempt(model,x[s],y,x[t],score,sigma0,cfg,sign=spec['sign'],
            force_geometry=spec['kind']=='geometry',empirical=empirical)
        payload.update(spec=spec,attempt=row['attempt'],batch=row,sigma0=sigma0)
        b=payload.get('boundaries')
        if b is not None:payload['origin_ids']=[t[i] for i in b['input_indices'].tolist()]
        if row['attempt'] in plan['prediction_attempts'] or payload['record']['status']=='failed':
            payload['predictions']=predictions(model,x,ids)
        atomic_tensor(directory/f"round{row['attempt']:03d}.pt",payload)
        # Every complete round is a resumable unit; no stochastic state is hidden.
        failed=payload['record']['status']=='failed'
        if rss_bytes()>plan['resources']['max_worker_rss_bytes']:raise MemoryError('adaptation RSS limit')
        if stop_after is not None and row['attempt']>=stop_after:return dict(status='interrupted_for_resume_test')
    files=sorted(directory.glob('round[0-9][0-9][0-9].pt'))[1:]
    records=[load(p)['record'] for p in files]
    result=dict(**spec,status='failed' if failed else 'complete',attempts=len(records),
        counts={s:sum(r['status']==s for r in records) for s in ('accepted','skipped','rejected','failed')},
        elapsed_seconds=sum(r['elapsed_seconds'] for r in records),worker_seconds=time.monotonic()-started,
        peak_rss_bytes=rss_bytes(),path=str(directory))
    if not failed:assert len(records)==30
    write_json(directory/'complete.json',result);return result
