"""R6 §3–4: fixed s/u decomposition, no Armijo or candidate commits."""
import copy
from dataclasses import asdict
import traceback
import torch
from torch.func import jacrev
from tsba.classifier import ParameterMap
from tsba.boundary import BoundaryBatch
from tsba.problem import TsbaProblem
from tsba.subproblem import GGNSystem
from tsba.utils import finite
from .shared import io, t3, d3, normalized_margin, assert_nested_equal
from .protocol import restore, block_path


def predictions(params,vector,pools):
    with torch.no_grad():
        result={pool:params.logits(vector,x).log_softmax(-1).detach() for pool,x in pools.items()}
    finite(*result.values())
    return result


def compare_predictions(actual,expected,plan):
    for pool,wanted in expected.items():
        torch.testing.assert_close(actual[pool],wanted,atol=plan['numerics']['replay_logprob_atol'],rtol=plan['numerics']['replay_logprob_rtol'])
        if not torch.equal(actual[pool].argmax(1),wanted.argmax(1)):
            raise ValueError('Historical class prediction mismatch')


def make_candidates(model,pools,labels,record,previous,sigma,cfg,old,plan):
    original=ParameterMap(model).vector();rng=torch.random.get_rng_state()
    # Load historical theta into an isolated work copy, never into the caller's model.
    work=copy.deepcopy(model).eval()
    params=ParameterMap(work);params.commit(record['before'])
    theta=params.vector()
    before=predictions(params,theta,pools)
    compare_predictions(before,previous['predictions'],plan)
    result=dict(status=record['side']['status'],theta=theta,baseline=before,batch=record['batch'],sigma=sigma,
                parameter_order=[(name,list(shape),n) for name,shape,n in params.entries])
    try:
        if record['side']['status']!='available':return result
        bounds=BoundaryBatch(**record['bounds'])
        if not len(bounds):raise ValueError('Available historical block has no boundaries')
        problem=TsbaProblem(work,pools['source_batch'],labels[record['batch']['source_positions']],bounds,cfg)
        system=GGNSystem(problem);matrix=system.dense()
        response=jacrev(problem.normalized_margin)(theta).detach()
        finite(matrix,response,system.rhs.norm())
        old_candidates=record['side']['candidates']
        s=old_candidates[0]['delta'].detach().clone()
        u=(old_candidates[1]['delta']-s).detach()
        rhs_u=system.coefficient*system.margin_pullback(bounds.zeta)[0].detach()
        residuals={}
        for name,direction,rhs in (('s',s,-system.gs),('u',u,rhs_u),('reverse',old_candidates[-1]['delta'], -system.gs-rhs_u)):
            norm=(matrix@direction-rhs).norm()
            threshold=10*cfg.cg_tolerance*max(float(system.rhs.norm()),float(rhs.norm()),1e-12)
            finite(norm)
            if float(norm)>threshold:raise ValueError(f'Historical {name} linear residual exceeds tolerance')
            residuals[name]=dict(residual=float(norm),threshold=threshold)
        torch.testing.assert_close(old_candidates[-1]['delta'],s-u,
                                   atol=plan['numerics']['replay_logprob_atol'],rtol=plan['numerics']['replay_logprob_rtol'])
        alpha=record['side']['alpha']
        if not 0<alpha<=old['side_alpha_start']:raise ValueError('Invalid historical alpha')
        for sign,direction in ((0,s),(1,s+u),(-1,s-u)):
            compare_predictions(predictions(params,theta+alpha*direction,pools),old_candidates[sign]['predictions'],plan)
        origin_positions=torch.tensor(record['batch']['target_positions'])[bounds.input_indices].unique()
        origin_ids=torch.tensor(record['batch']['target_ids'])[bounds.input_indices].unique()
        finite(s.norm(),u.norm())
        norm_s,norm_u=float(s.norm()),float(u.norm())
        result.update(status='available',s=s,u=u,alpha=alpha,origins=origin_positions,origin_ids=origin_ids,
            residuals=residuals,bounds=record['bounds'],norm_s=norm_s,norm_u=norm_u,
            cosine_s_u=float(s@u)/(norm_s*norm_u) if norm_s*norm_u>0 else None,
            baseline_normalized_margin=normalized_margin(params,theta,pools['target'],cfg.min_normal),candidates=[])
        for k in plan['steps']['multipliers']:
            beta=k*alpha
            for arm in plan['factorial']['arms']:
                a,q=arm['source_gradient_coefficient'],arm['request_sign']
                delta=(a*s+q*u).detach();vector=(theta+beta*delta).detach()
                finite(vector)
                movement=problem.movement(vector).detach();finite(movement)
                geometry=normalized_margin(params,vector,pools['target'],cfg.min_normal)
                relocation=d3.relocate(params,vector,bounds,sigma,cfg,old)
                single=[(i,r['displacement']) for i,r in enumerate(relocation['intersections']) if r['status']=='single_root']
                error=None
                if single:
                    indices,displacements=zip(*single)
                    error=float(((movement[list(indices)]+movement.new_tensor(displacements))/sigma).square().mean().sqrt())
                result['candidates'].append(dict(k=k,arm=arm['id'],a=a,q=q,beta=beta,vector=vector,
                    predictions=predictions(params,vector,pools),movement=movement,
                    motion_feasible=bool(movement.abs().max()<=cfg.execution_multiplier*sigma),
                    max_movement=float(movement.abs().max()),movement_rms_sigma=float(movement.square().mean().sqrt()/sigma),
                    source_linear_rms_sigma=float((beta*response@s).square().mean().sqrt()/sigma),
                    request_linear_rms_sigma=float((beta*response@u).square().mean().sqrt()/sigma),
                    combined_linear_rms_sigma=float((beta*response@delta).square().mean().sqrt()/sigma),
                    normalized_margin=geometry,relocation=relocation,single_root_count=len(single),root_rmse_vs_proxy_sigma=error))
            current=[c for c in result['candidates'] if c['k']==k]
            jointly=all(c['motion_feasible'] for c in current)
            for c in current:c['common_motion_feasible']=jointly
        return result
    finally:
        assert_nested_equal(ParameterMap(model).vector(),original)
        assert_nested_equal(params.vector(),theta)
        assert_nested_equal(torch.random.get_rng_state(),rng)


def worker(root,plan,item):
    task,step=item;path=block_path(root,task,step)
    if path.exists():return
    identity=dict(setting=plan['setting']['id'],**asdict(task),attempt=step)
    try:
        model,pools,y,record,previous,sigma,cfg,old=restore(root,task,step)
        result=make_candidates(model,pools,y,record,previous,sigma,cfg,old,plan)
        result.update(identity)
    except t3.NUMERICAL_ERRORS:
        result=dict(**identity,status='numerical_failure',error=traceback.format_exc())
    except (ValueError,AssertionError,KeyError):
        result=dict(**identity,status='engineering_failure',error=traceback.format_exc())
    io.atomic_tensor(path,result)
