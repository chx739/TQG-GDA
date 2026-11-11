"""Atomic attempt records, fixed-state side comparisons and production M00."""
from dataclasses import asdict, replace
import time

import numpy as np
import torch

from tsba.boundary import build_boundaries
from tsba.classifier import ParameterMap
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.utils import NumericalError
from ..next_round.steps import EmpiricalScore
from ..r2.adapt import ZeroScore
from ..r2.common import atomic_tensor, load, model_from_frozen, write_json
from .constraints import Budgets, accept, actual_objective, assess, propose
from .diagnostics import density_audit, relocate
from .protocol import domain_ids, roles_for
from .training import score_model, source_dir


NUMERICAL_ERRORS = (NumericalError, torch.linalg.LinAlgError, np.linalg.LinAlgError, FloatingPointError)


@torch.no_grad()
def predictions(params, vector, pools):
    return {name: params.logits(vector, x).log_softmax(-1) for name, x in pools.items()}


def common_side(problem, normal_request, sigma, budgets, plan, pools, existing):
    proposals = {}
    for direction in (1, 0, -1):
        paired = TsbaProblem(problem.params.model, problem.source, problem.labels,
                             replace(problem.boundaries, zeta=direction * normal_request), problem.cfg)
        delta, info = existing[direction] if direction in existing else propose(paired, budgets, plan["solver"])
        proposals[direction] = (paired, delta, info)
    if any(not info["valid"] and info["reason"] != "stationary" for _, _, info in proposals.values()):
        return dict(status="proposal_unavailable", solvers={d: info for d, (_, _, info) in proposals.items()})
    initial = {d: actual_objective(p, p.base, budgets)["objective"] for d, (p, _, _) in proposals.items()}
    for attempt in range(problem.cfg.backtracks):
        alpha = plan["side_alpha_start"] * .5 ** attempt
        candidates = {}
        for direction, (p, delta, info) in proposals.items():
            vector, result, valid = assess(p, delta, info.get("directional_derivative", 0.),
                                           alpha, sigma, budgets, initial[direction])
            if not valid:
                break
            candidates[direction] = dict(vector=vector, delta=delta, trial=result, solver=info,
                                          movement=p.movement(vector).detach(),
                                          predictions=predictions(p.params, vector, pools))
        if len(candidates) == 3:
            return dict(status="available", alpha=alpha, candidates=candidates)
    return dict(status="no_common_feasible_step")


def one_attempt(model, batch, source, labels, target, estimator, sigma, budgets,
                job, cfg, plan, pools, side=False, side_estimator=None):
    params = ParameterMap(model)
    base = params.vector()
    bounds = build_boundaries(model, target[batch["target_positions"]], estimator, sigma, cfg)
    normal_request = bounds.zeta
    bounds = replace(bounds, zeta=normal_request * job.direction)
    payload = dict(before=base, after=base, batch=batch, bounds=asdict(bounds),
                   status="no_valid_boundary", accepted=False, trials=[])
    if not len(bounds):
        if side:
            payload["side"] = dict(status="no_valid_boundary")
        return payload
    positions = torch.tensor(batch["target_positions"])[bounds.input_indices]
    local_ids = batch["source_positions"]
    problem = TsbaProblem(model, source[local_ids], labels[local_ids], bounds, cfg)
    if job.direction and not bool((bounds.zeta.abs() > cfg.movement_zero * sigma).any()):
        delta = torch.zeros_like(base)
        info = dict(valid=False, reason="zero_requests")
    else:
        delta, info = propose(problem, budgets, plan["solver"])
    payload.update(solver=info, delta=delta, status=info["reason"])
    if info["valid"]:
        vector, trials = accept(problem, delta, info["directional_derivative"], sigma, budgets)
        payload["trials"] = trials
        payload["status"] = "backtracking_exhausted" if vector is None else "accepted"
        if vector is not None:
            payload.update(after=vector, accepted=True)
    if side:
        try:
            with torch.random.fork_rng(devices=[]):
                if side_estimator is not None:
                    with torch.no_grad():
                        gamma = (side_estimator(bounds.points) * bounds.normals).sum(1)
                        cap = cfg.request_multiplier * sigma
                        normal_request = (cfg.tau * sigma ** 2 * gamma).clamp(-cap, cap)
                existing = {job.direction: (delta, info)} if info["reason"] != "zero_requests" else {}
                payload["side"] = common_side(problem, normal_request, sigma, budgets, plan, pools, existing)
        except NUMERICAL_ERRORS as error:
            payload["side"] = dict(status="numerical_failure", error=str(error))
    bandwidth = (job.bandwidth or 1) * sigma
    payload["margin_proxy"] = problem.movement(payload["after"]).detach()
    payload["true_budgets"] = budgets.check(params, payload["after"])
    try:
        payload["density_audit"] = density_audit(bounds, target, positions, bandwidth, sigma,
                                                  estimator if job.density == "learned" else None)
    except NUMERICAL_ERRORS as error:
        payload["density_error"] = str(error)
    try:
        payload["relocation"] = relocate(params, payload["after"], bounds, sigma, cfg, plan)
    except NUMERICAL_ERRORS as error:
        payload["relocation_error"] = str(error)
    if payload["accepted"]:
        params.commit(payload["after"])
    return payload


def trajectory(root, job, plan, stop_after=None):
    out = root / "trajectories" / job.key
    if (out / "complete.json").exists():
        return
    started = time.monotonic()
    frozen = load(source_dir(root, job.seed) / "frozen.pt")
    x = frozen["x"]
    model = model_from_frozen(frozen)
    params = ParameterMap(model)
    roles = roles_for(root, job.seed)
    source = x[roles["fit"]]
    labels = load(root / "source_labels" / f"seed{job.seed}_fit.pt")
    target = x[domain_ids(root, job.domain)]
    pools = dict(source_fit=source, source_validation=x[roles["validation"]], target=target)
    sigma = load(source_dir(root, job.seed) / f"geometry_d{job.domain}.pt")["sigma"]
    cfg = job.config(plan)
    budgets = Budgets(model, source, labels, target, job, plan)
    initial = params.vector()
    checks = budgets.check(params, initial)
    if not checks["risk_ok"] or not checks["confidence_ok"]:
        raise ValueError("Initial classifier exceeds its budgets")

    kde = EmpiricalScore(target, (job.bandwidth or 1) * sigma)
    if job.density == "learned":
        estimator = score_model(root, job, plan, x.shape[1])
    elif job.density == "kde":
        estimator = kde
    else:
        estimator = ZeroScore()
    side_estimator = None
    if job.density == "zero":
        side_estimator = score_model(root, replace(job, bandwidth=1), plan, x.shape[1])
    batches = load(root / "batches" / f"seed{job.seed}_d{job.domain}.pt")
    files = sorted(out.glob("round[0-9][0-9].pt"))
    if files:
        if [p.name for p in files] != [f"round{i:02d}.pt" for i in range(len(files))]:
            raise ValueError("Trajectory attempt sequence is incomplete")
        params.commit(load(files[-1])["after"])
    else:
        atomic_tensor(out / "round00.pt", dict(after=initial, predictions=predictions(params, initial, pools)))
        files = [out / "round00.pt"]
    if stop_after is not None and len(files) - 1 >= stop_after:
        return

    for batch in batches[len(files) - 1:]:
        attempt = batch["attempt"]
        base = params.vector()
        try:
            record = one_attempt(model, batch, source, labels, target, estimator, sigma,
                                 budgets, job, cfg, plan, pools,
                                 side=attempt in plan["side_attempts"], side_estimator=side_estimator)
        except NUMERICAL_ERRORS as error:
            params.commit(base)
            record = dict(before=base, after=base, batch=batch, status="numerical_failure",
                          accepted=False, trials=[], error=str(error))
            if attempt in plan["side_attempts"]:
                record["side"] = dict(status="main_numerical_failure")
        if job.protect and job.direction == 0:
            torch.testing.assert_close(record["after"], initial, rtol=0, atol=0)
        if attempt in plan["endpoints"]:
            record["predictions"] = predictions(params, record["after"], pools)
        record["attempt"] = attempt
        atomic_tensor(out / f"round{attempt:02d}.pt", record)
        if stop_after == attempt:
            return
    write_json(out / "complete.json", dict(job=asdict(job), attempts=len(batches),
                                         elapsed_seconds=time.monotonic() - started))


def check_sensitivity_zero(root, seed, domain, plan):
    from .protocol import Job
    x_and_state = load(source_dir(root, seed) / "frozen.pt")
    x = x_and_state["x"]
    model = model_from_frozen(x_and_state)
    fit = roles_for(root, seed)["fit"]
    source, target = x[fit], x[domain_ids(root, domain)]
    labels = load(root / "source_labels" / f"seed{seed}_fit.pt")
    geometry = load(source_dir(root, seed) / f"geometry_d{domain}.pt")
    from tsba.boundary import BoundaryBatch
    bounds = BoundaryBatch(**geometry["boundaries"])
    batches = load(root / "batches" / f"seed{seed}_d{domain}.pt")
    batch = batches[0]
    if not len(bounds):
        cfg = Config(seed=seed, **plan["adapt"])
        for batch in batches[1:]:
            bounds = build_boundaries(model, target[batch["target_positions"]], ZeroScore(),
                                      geometry["sigma"], cfg)
            if len(bounds):
                break
    if not len(bounds):
        return dict(seed=seed, domain=domain, verified_combinations=0, status="no_initial_anchors")
    for weight in plan["sensitivity_lambda"]:
        for epsilon in plan["sensitivity_epsilon"]:
            job = Job(seed, domain, "M11", "zero", 0, 0, weight, epsilon)
            budgets = Budgets(model, source, labels, target, job, plan)
            positions = batch["source_positions"]
            problem = TsbaProblem(model, source[positions], labels[positions], bounds, job.config(plan))
            delta, info = propose(problem, budgets, plan["solver"])
            if info["reason"] != "stationary" or bool(delta.any()):
                raise AssertionError(f"M11 zero-step mismatch: {job.key}")
    return dict(seed=seed, domain=domain, verified_combinations=9, batch_attempt=batch["attempt"])
