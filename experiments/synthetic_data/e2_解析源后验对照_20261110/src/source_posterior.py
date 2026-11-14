"""Explicit diagnostic: replace hard source CE with known conditional CE.

Production TSBA remains unchanged. GGN, projection, damping, line search and
execution bounds are imported unchanged; only the source loss is replaced.
"""
import copy
from dataclasses import asdict
import torch
from torch.func import grad
from torch.nn import functional as F
from tsba.adapt import backtrack
from tsba.boundary import build_boundaries
from tsba.classifier import ParameterMap
from tsba.problem import TsbaProblem
from tsba.subproblem import solve
from tsba.utils import finite,NumericalError
from experiments.synthetic_data.e2_目标请求对照_20261109.src.controls import SourceProblem, solve_source, backtrack_source
from experiments.synthetic_data.e1_e2_符号几何与适配_20261109.src.distributions import ZeroScore
from experiments.synthetic_data.e2_适配轨迹回放_20261110.src.diagnostics import posterior_log_prob

ARMS=('source_ce_ggn','zero_zeta_geometry','population')


def analytic_source_posterior(source,coordinates):
    # Invert the frozen scalar coordinate transform. Known source means are
    # -.5 and 1.5; variance on x1=.25 and equal priors give log odds 8(x1-.5).
    raw=source*coordinates.scale+coordinates.mean
    logp=posterior_log_prob(raw,.5)
    return logp.exp().detach(),logp.detach()


def validate_posterior(source,posterior):
    finite(source,posterior)
    if posterior.shape!=(len(source),2) or not posterior.is_floating_point():raise ValueError('expected [N,2] source probabilities')
    if bool((posterior<0).any()) or float((posterior.sum(-1)-1).abs().max())>1e-12:raise ValueError('invalid posterior probabilities')


def expected_ce(logits,posterior):
    # Conditional expectation over the SOURCE label; no target pseudo-labels.
    return -(posterior*F.log_softmax(logits,dim=-1)).sum(-1)


def class_conditional_ce(logits,posterior):
    loss=-F.log_softmax(logits,dim=-1)
    return {str(c):float((posterior[:,c]*loss[:,c]).sum()/posterior[:,c].sum())
            for c in range(posterior.shape[1]) if float(posterior[:,c].sum())>0}


class PosteriorSourceProblem(SourceProblem):
    def __init__(self,model,source,posterior,cfg):
        validate_posterior(source,posterior)
        super().__init__(model,source,posterior,cfg)

    def loss(self,vector):return expected_ce(self.logits(vector),self.labels).mean()

    @torch.no_grad()
    def evaluate(self,vector):
        logits=self.logits(vector);ce=expected_ce(logits,self.labels);finite(logits,ce)
        return dict(objective=float(ce.mean()),source_ce=float(ce.mean()),
                    source_class_ce=class_conditional_ce(logits,self.labels))


class PosteriorTsbaProblem(TsbaProblem):
    def __init__(self,model,source,posterior,boundaries,cfg):
        validate_posterior(source,posterior)
        super().__init__(model,source,posterior,boundaries,cfg)

    def source_loss(self,vector):return expected_ce(self.source_logits(vector),self.labels).mean()

    @torch.no_grad()
    def evaluate(self,vector):
        logits=self.source_logits(vector);ce=expected_ce(logits,self.labels)
        movement=self.movement(vector);move_loss=(movement-self.boundaries.zeta).square().mean()/2
        objective=ce.mean()+self.cfg.lambda_boundary*move_loss
        finite(logits,ce,movement,objective)
        return dict(objective=float(objective),source_ce=float(ce.mean()),
                    source_class_ce=class_conditional_ce(logits,self.labels),move_loss=float(move_loss),
                    max_movement=float(movement.abs().max()),movement_rmse=float((movement-self.boundaries.zeta).square().mean().sqrt()))


def adapt_posterior(model,source,posterior,target,population,sigma,cfg,schedule,arm,observer=None,logger=None):
    """Same branch-specific activity rules as the archived hard-label runs."""
    if arm not in ARMS:raise ValueError(arm)
    if len(schedule)!=cfg.adapt_rounds:raise ValueError('incomplete schedule')
    validate_posterior(source,posterior)
    model.eval();model.zero_grad(set_to_none=True)
    source,posterior,target=source.detach(),posterior.detach(),target.detach()
    population.requires_grad_(False).eval();zero=ZeroScore()
    params=ParameterMap(model);history=[];accepted=0;stop_reason='round_budget'
    for round_id,ids in enumerate(schedule):
        source_ids,target_ids=ids['source_ids'],ids['target_ids']
        before=params.vector();boundaries=None;delta=None;source_gradient=None
        record=dict(round=round_id,accepted=False)
        try:
            if arm=='source_ce_ggn':
                problem=PosteriorSourceProblem(model,source[source_ids],posterior[source_ids],cfg)
            else:
                boundaries=build_boundaries(model,target[target_ids],population if arm=='population' else zero,sigma,cfg)
                record.update(anchors=len(boundaries),projection_rejections=boundaries.rejected)
                problem=None
                if not len(boundaries):record['reason']='no_valid_boundary'
                elif arm=='population' and bool((boundaries.zeta.abs()<=cfg.movement_zero*sigma).all()):record['reason']='zero_requests'
                else:
                    problem=PosteriorTsbaProblem(model,source[source_ids],posterior[source_ids],boundaries,cfg)
                    if arm=='population':record['max_request']=float(boundaries.zeta.abs().max())
            if problem is not None:
                record['before']=problem.evaluate(problem.base)
                source_gradient=grad(problem.loss if arm=='source_ce_ggn' else problem.source_loss)(problem.base).detach()
                record['source_gradient_norm']=float(source_gradient.norm())
                delta,info=solve_source(problem) if arm=='source_ce_ggn' else solve(problem)
                record['solver']=info
                if not info['valid']:record['reason']=info['reason']
                else:
                    candidate,trials=(backtrack_source(problem,delta,info['directional_derivative']) if arm=='source_ce_ggn'
                                      else backtrack(problem,delta,info['directional_derivative'],sigma))
                    record['trials']=trials
                    if candidate is None:record['reason']='backtracking_exhausted'
                    else:
                        params.commit(candidate);accepted+=1;record.update(accepted=True,reason='accepted')
        except NumericalError as exc:
            record.update(reason='numerical_error',error=str(exc));stop_reason='numerical_error'
        history.append(record)
        if observer is not None:
            payload=dict(record=copy.deepcopy(record),before=before,after=params.vector(),
                         source_ids=source_ids.detach().clone(),target_ids=target_ids.detach().clone(),
                         boundaries=asdict(boundaries) if boundaries is not None else None,
                         delta=delta.detach().clone() if delta is not None else None,
                         source_gradient=source_gradient)
            with torch.random.fork_rng(devices=[source.device] if source.is_cuda else []):observer(payload)
        if logger:logger(f'round {round_id}: {record["reason"]}; anchors={record.get("anchors","not_used")}')
        if stop_reason=='numerical_error':break
    return dict(arm=arm,source_objective='analytic_source_posterior_expected_ce',accepted=accepted,stop_reason=stop_reason,history=history)


def audit_posterior_event(head,inputs,posterior,schedule_row,event):
    from torch.func import vjp
    for name in ('source_ids','target_ids'):
        if not torch.equal(event[name],schedule_row[name]):raise ValueError('schedule changed: '+name)
    if not torch.equal(event['source_ids'].sort().values,torch.arange(len(inputs['source']))):raise ValueError('source set incomplete')
    params=ParameterMap(head);record=event['record'];before=event['before'];errors=[]
    # Independent formula using logsumexp and probability-weighted logits.
    def ce(vector):
        logits=params.logits(vector,inputs['source'])
        return float((posterior.sum(-1)*logits.logsumexp(-1)-(posterior*logits).sum(-1)).mean())
    with torch.no_grad():
        before_ce=ce(before);after_ce=ce(event['after'])
        if 'before' in record:errors.append(abs(before_ce-record['before']['source_ce']))
        for trial in record.get('trials',[]):errors.append(abs(ce(before+trial['alpha']*event['delta'])-trial['source_ce']))
    gradient_error=0.
    if event['source_gradient'] is not None:
        logits,pullback=vjp(lambda vector:params.logits(vector,inputs['source']),before)
        analytic=pullback((logits.softmax(-1)*posterior.sum(-1,keepdim=True)-posterior)/len(posterior))[0]
        gradient_error=float((analytic-event['source_gradient']).abs().max())
    max_error=max(errors,default=0.)
    if max_error>1e-11 or gradient_error>1e-11:raise ValueError('posterior CE/gradient audit failed')
    event['posterior_audit']=dict(before=before_ce,after=after_ce,max_trial_ce_error=max_error,
                                  trials_checked=len(record.get('trials',[])),gradient_max_error=gradient_error)
