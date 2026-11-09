"""SEGA-style propose -> nonlinear re-evaluate -> accept/backtrack, with TSBA objectives."""
import copy
from dataclasses import asdict
import torch
from .classifier import ParameterMap
from .boundary import build_boundaries
from .problem import TsbaProblem
from .subproblem import solve
from .utils import NumericalError


# §3.12：固定本轮参考，尝试 ψ_candidate=ψ_bar+αδ，α=1,1/2,1/4,...。
# 候选通过函数式前向求值；此函数不会修改真实模型参数。
def backtrack(problem, delta, directional, sigma):
    cfg = problem.cfg
    initial = problem.evaluate(problem.base)
    trials = []
    for trial in range(cfg.backtracks):
        alpha = 0.5 ** trial
        candidate = problem.base + alpha * delta
        result = problem.evaluate(candidate)  # nonfinite is an error, never silent rejection
        # Armijo：J_t(ψ_bar+αδ)≤J_t(ψ_bar)+cA·α·∇J_t(ψ_bar)ᵀδ。
        armijo = result["objective"] <= initial["objective"] + cfg.armijo * alpha * directional
        # 第二项验收：max_j|v_j(ψ_candidate)|≤Δexec。
        # 这里控制的是归一化 margin 变化，不是重新定位后测得的精确边界距离。
        bounded = result["max_movement"] <= cfg.execution_multiplier * sigma
        trials.append(dict(alpha=alpha, armijo=armijo, bounded=bounded, **result))
        if armijo and bounded:
            return candidate.detach(), trials
    return None, trials


# §3.6–3.14：每轮抽批次、构造边界、求候选、真实验收、接受才提交。
# 原位更新 model；返回日志；源坐标、目标坐标和 score 始终固定。
def adapt_classifier(model, source, labels, target, score, sigma, cfg, logger=None, observer=None):
    """Optional observer receives detached copies after each attempt, never live state.

    Research observers must be passive; CPU/CUDA torch RNG is restored around the
    callback. Observer failures propagate to the caller, not solver diagnostics.
    """
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
        before = ParameterMap(model).vector() if observer is not None else None
        boundaries, delta = None, None
        try:
            source_ids = torch.randperm(len(source), generator=rng, device=source.device)[:cfg.source_adapt_batch]
            target_ids = torch.randperm(len(target), generator=rng, device=target.device)[:cfg.target_batch]
            # 本轮构造参考时模型尚未变化；求解与回溯期间不重算这些参考。
            boundaries = build_boundaries(model, target[target_ids], score, sigma, cfg)
            rec.update(anchors=len(boundaries), projection_rejections=boundaries.rejected)
            # §3.14：M=0 或请求全零时跳过，不能自动转成继续源训练。
            if not len(boundaries):
                rec["reason"] = "no_valid_boundary"
            elif bool((boundaries.zeta.abs() <= cfg.movement_zero * sigma).all()):
                rec["reason"] = "zero_requests"
            else:
                # 保存 ψ_bar 并建立固定目标；其响应 D 和源 GGN 都在该快照计算。
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
                        # §3.12：唯一提交点。下一轮重新定位边界，不在旧参考上连续训练多次。
                        problem.params.commit(candidate)
                        accepted += 1
                        rec.update(accepted=True, reason="accepted")
        # §3.14：非有限异常停止本次适配，保留最后接受状态；
        # 有限但不合格的候选只是拒绝本轮，与数值异常区分记录。
        except NumericalError as exc:
            rec.update(reason="numerical_error", error=str(exc))
            stop_reason = "numerical_error"
        history.append(rec)
        if observer is not None:
            payload = dict(record=copy.deepcopy(rec), before=before,
                           after=ParameterMap(model).vector(),
                           source_ids=source_ids.detach().clone(), target_ids=target_ids.detach().clone(),
                           boundaries=asdict(boundaries) if boundaries is not None else None,
                           delta=delta.detach().clone() if delta is not None else None)
            devices = [source.device] if source.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                observer(payload)
        if logger:
            logger(f"round {round_id}: {rec['reason']}; anchors={rec.get('anchors', 0)}")
        if stop_reason == "numerical_error":
            break
    return dict(accepted=accepted, stop_reason=stop_reason, history=history)
