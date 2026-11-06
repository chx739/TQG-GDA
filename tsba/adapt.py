"""SEGA-style propose -> nonlinear re-evaluate -> accept/backtrack, with TSBA objectives."""
import torch
from .boundary import build_boundaries
from .problem import TsbaProblem
from .subproblem import solve
from .utils import NumericalError


def backtrack(problem, delta, directional, sigma):
    cfg = problem.cfg
    initial = problem.evaluate(problem.base)
    trials = []
    for trial in range(cfg.backtracks):
        alpha = 0.5 ** trial
        candidate = problem.base + alpha * delta
        result = problem.evaluate(candidate)  # nonfinite is an error, never silent rejection
        armijo = result["objective"] <= initial["objective"] + cfg.armijo * alpha * directional
        bounded = result["max_movement"] <= cfg.execution_multiplier * sigma
        trials.append(dict(alpha=alpha, armijo=armijo, bounded=bounded, **result))
        if armijo and bounded:
            return candidate.detach(), trials
    return None, trials


def adapt_classifier(model, source, labels, target, score, sigma, cfg, logger=None):
    model.eval()
    model.zero_grad(set_to_none=True)
    if sigma is None:
        return dict(accepted=0, stop_reason="fewer_than_two_distinct_target_points", history=[])
    source, labels, target = source.detach(), labels.detach(), target.detach()
    score.requires_grad_(False).eval()
    rng = torch.Generator(device=source.device).manual_seed(cfg.seed + 201)
    accepted, history, stop_reason = 0, [], "round_budget"
    for round_id in range(cfg.adapt_rounds):
        rec = dict(round=round_id, accepted=False)
        try:
            source_ids = torch.randperm(len(source), generator=rng, device=source.device)[:cfg.source_adapt_batch]
            target_ids = torch.randperm(len(target), generator=rng, device=target.device)[:cfg.target_batch]
            boundaries = build_boundaries(model, target[target_ids], score, sigma, cfg)
            rec.update(anchors=len(boundaries), projection_rejections=boundaries.rejected)
            if not len(boundaries):
                rec["reason"] = "no_valid_boundary"
            elif bool((boundaries.zeta.abs() <= cfg.movement_zero * sigma).all()):
                rec["reason"] = "zero_requests"
            else:
                problem = TsbaProblem(model, source[source_ids], labels[source_ids], boundaries, cfg)
                rec["before"] = problem.evaluate(problem.base)
                rec["max_request"] = float(boundaries.zeta.abs().max())
                delta, info = solve(problem)
                rec["solver"] = info
                if not info["valid"]:
                    rec["reason"] = info["reason"]
                else:
                    candidate, trials = backtrack(problem, delta, info["directional_derivative"], sigma)
                    rec["trials"] = trials
                    if candidate is None:
                        rec["reason"] = "backtracking_exhausted"
                    else:
                        problem.params.commit(candidate)
                        accepted += 1
                        rec.update(accepted=True, reason="accepted")
        except NumericalError as exc:
            rec.update(reason="numerical_error", error=str(exc))
            stop_reason = "numerical_error"
        history.append(rec)
        if logger:
            logger(f"round {round_id}: {rec['reason']}; anchors={rec.get('anchors', 0)}")
        if stop_reason == "numerical_error":
            break
    return dict(accepted=accepted, stop_reason=stop_reason, history=history)
