"""Post-E2 diagnostic controls. These do not change production TSBA skip rules."""
import copy
from dataclasses import asdict
import torch
from torch.func import grad, jacrev, jvp, vjp
from torch.nn import functional as F
from tsba.adapt import backtrack
from tsba.boundary import build_boundaries
from tsba.classifier import ParameterMap
from tsba.problem import TsbaProblem
from tsba.subproblem import solve
from tsba.utils import finite, NumericalError
from experiments.common.synthetic.distributions import ZeroScore

ARMS = ('source_ce_ggn', 'zero_zeta_geometry')


class SourceProblem:
    """Only source CE. No boundary regularizer and no boundary execution bound."""
    def __init__(self, model, source, labels, cfg):
        self.params = ParameterMap(model)
        self.base = self.params.vector()
        self.source, self.labels = source.detach(), labels.detach()
        self.cfg = cfg

    def logits(self, vector):
        return self.params.logits(vector, self.source)

    def loss(self, vector):
        return F.cross_entropy(self.logits(vector), self.labels)

    @torch.no_grad()
    def evaluate(self, vector):
        logits = self.logits(vector)
        losses = F.cross_entropy(logits, self.labels, reduction='none')
        finite(logits, losses)
        return dict(objective=float(losses.mean()), source_ce=float(losses.mean()),
                    source_class_ce={str(c):float(losses[self.labels == c].mean())
                                     for c in self.labels.unique().tolist()})


class SourceGGN:
    """A=F_S+μI, rhs=−g_S; same CE GGN convention as tsba.subproblem."""
    def __init__(self, problem):
        self.problem = problem
        self.p = problem.logits(problem.base).softmax(-1).detach()
        self.gs = grad(problem.loss)(problem.base).detach()
        self.rhs = -self.gs
        _, self.pullback = vjp(problem.logits, problem.base)
        finite(self.p, self.gs)

    def dense(self):
        j = jacrev(self.problem.logits)(self.problem.base).detach()
        hj = self.p[..., None] * (j - (self.p[..., None]*j).sum(1, keepdim=True))
        fs = torch.einsum('bcp,bcq->pq', j, hj) / len(self.p)
        a = (fs+fs.T)/2 + self.problem.cfg.damping*torch.eye(
            len(self.problem.base), dtype=fs.dtype, device=fs.device)
        finite(a)
        return a

    def matvec(self, vector):
        jv = jvp(self.problem.logits, (self.problem.base,), (vector,))[1]
        hjv = self.p*(jv-(self.p*jv).sum(-1, keepdim=True))
        result = self.pullback(hjv/len(self.p))[0] + self.problem.cfg.damping*vector
        finite(result)
        return result.detach()


def solve_source(problem):
    # Predeclared d=2/16 residual heads fit the production direct-solver budget.
    if problem.cfg.solver == 'cg' or len(problem.base) > problem.cfg.direct_max_params:
        raise ValueError('this diagnostic source driver supports the frozen direct-solver matrix only')
    system = SourceGGN(problem)
    rhs_norm = system.rhs.norm()
    finite(rhs_norm)
    if float(rhs_norm) == 0:
        return torch.zeros_like(system.rhs), dict(valid=False, reason='stationary', solver='direct')
    try:
        delta = torch.linalg.solve(system.dense(), system.rhs)
    except torch.linalg.LinAlgError as exc:
        return None, dict(valid=False, reason=str(exc), solver='direct')
    relative = (system.matvec(delta)-system.rhs).norm()/rhs_norm
    directional = system.gs@delta
    finite(delta, relative, directional)
    valid = float(relative) <= problem.cfg.cg_tolerance and float(directional) < 0
    return delta.detach(), dict(valid=valid, reason='solved' if valid else 'inaccurate_or_non_descent',
                               solver='direct', iterations=1, relative_residual=float(relative),
                               directional_derivative=float(directional))


def backtrack_source(problem, delta, directional):
    before = problem.evaluate(problem.base)
    trials = []
    for trial in range(problem.cfg.backtracks):
        alpha = .5**trial
        candidate = problem.base+alpha*delta
        result = problem.evaluate(candidate)
        armijo = result['objective'] <= before['objective']+problem.cfg.armijo*alpha*directional
        trials.append(dict(alpha=alpha, armijo=armijo, **result))
        if armijo:
            return candidate.detach(), trials
    return None, trials


def adapt_control(model, source, labels, target, sigma, cfg, schedule, arm, observer=None, logger=None):
    """Replay archived batch IDs, keep every attempt, accept before committing.

    zero_zeta_geometry rebuilds references from ITS OWN current head. It retains
    DᵀD, movement loss and execution bound but deliberately bypasses production's
    all-zero-request skip. Empty boundaries still skip, never source fallback.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    if len(schedule) != cfg.adapt_rounds:
        raise ValueError('incomplete archived batch schedule')
    model.eval();model.zero_grad(set_to_none=True)
    source, labels, target = source.detach(), labels.detach(), target.detach()
    params = ParameterMap(model)
    history = [];accepted = 0;stop_reason = 'round_budget'
    for round_id, ids in enumerate(schedule):
        source_ids, target_ids = ids['source_ids'], ids['target_ids']
        before = params.vector()
        record = dict(round=round_id, accepted=False)
        boundaries = None;delta = None
        try:
            if arm == 'source_ce_ggn':
                problem = SourceProblem(model, source[source_ids], labels[source_ids], cfg)
            else:
                boundaries = build_boundaries(model, target[target_ids], ZeroScore(), sigma, cfg)
                record.update(anchors=len(boundaries), projection_rejections=boundaries.rejected)
                if len(boundaries):
                    problem = TsbaProblem(model, source[source_ids], labels[source_ids], boundaries, cfg)
                else:
                    problem = None
                    record['reason'] = 'no_valid_boundary'
            if problem is not None:
                record['before'] = problem.evaluate(problem.base)
                delta, info = solve_source(problem) if arm == 'source_ce_ggn' else solve(problem)
                record['solver'] = info
                if not info['valid']:
                    record['reason'] = info['reason']
                else:
                    candidate, trials = (backtrack_source(problem, delta, info['directional_derivative'])
                                         if arm == 'source_ce_ggn' else
                                         backtrack(problem, delta, info['directional_derivative'], sigma))
                    record['trials'] = trials
                    if candidate is None:
                        record['reason'] = 'backtracking_exhausted'
                    else:
                        params.commit(candidate);accepted += 1
                        record.update(accepted=True, reason='accepted')
        except NumericalError as exc:
            record.update(reason='numerical_error', error=str(exc));stop_reason='numerical_error'
        history.append(record)
        if observer is not None:
            payload=dict(record=copy.deepcopy(record), before=before, after=params.vector(),
                         source_ids=source_ids.detach().clone(), target_ids=target_ids.detach().clone(),
                         boundaries=asdict(boundaries) if boundaries is not None else None,
                         delta=delta.detach().clone() if delta is not None else None)
            with torch.random.fork_rng(devices=[source.device] if source.is_cuda else []):
                observer(payload)
        if logger:logger(f'round {round_id}: {record["reason"]}; anchors={record.get("anchors", "not_used")}')
        if stop_reason == 'numerical_error':break
    return dict(arm=arm, accepted=accepted, stop_reason=stop_reason, history=history)
