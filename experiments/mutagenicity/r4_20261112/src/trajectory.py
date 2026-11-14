"""Apply the R3 numerical kernel to isolated R4 task pools."""
from dataclasses import asdict, replace

import torch

from tsba.boundary import BoundaryBatch, build_boundaries
from tsba.classifier import ParameterMap
from tsba.problem import TsbaProblem
from experiments.mutagenicity.r1b_20261111.src.steps import EmpiricalScore
from experiments.mutagenicity.r2_20261111.src.adapt import ZeroScore
from experiments.mutagenicity.r2_20261111.src.common import atomic_tensor, load, model_from_frozen, write_json
from experiments.mutagenicity.r3_20261112.src.constraints import Budgets, propose
from experiments.mutagenicity.r3_20261112.src.trajectory import NUMERICAL_ERRORS, one_attempt, predictions
from experiments.mutagenicity.r4_20261112.src.protocol import batch_path, config, domain, job_dir, labels_path, r3_job, role, source_dir
from experiments.mutagenicity.r4_20261112.src.training import geometry, learned_score


def inputs(root, plan, task):
    frozen = load(source_dir(root, plan, task.source, task.seed) / "frozen.pt")
    x = frozen["x"]
    selected = role(root, task.source, task.seed)
    if set(selected["fit"] + selected["validation"]) & set(domain(root, task.target)):
        raise ValueError("Task source overlaps target inputs")
    return model_from_frozen(frozen), dict(source_fit=x[selected["fit"]],
        source_validation=x[selected["validation"]], target=x[domain(root, task.target)])


def check_zero(root, plan, task):
    model, pools = inputs(root, plan, task)
    y = load(labels_path(root, task.source, task.seed, "fit"))
    initial = ParameterMap(model).vector()
    spec = geometry(root, plan, task)
    boundaries = BoundaryBatch(**spec["boundaries"])
    job = replace(r3_job(plan, task, "A2"), density="zero", direction=0, bandwidth=0)
    budget = Budgets(model, pools["source_fit"], y, pools["target"], job, plan)
    checks = budget.check(ParameterMap(model), initial)
    if not checks["risk_ok"] or not checks["confidence_ok"]:
        raise ValueError("Initial head violates M11 budgets")
    schedule = load(batch_path(root, task))
    selected = schedule[0]
    cfg = config(plan, task.seed)
    if not len(boundaries):
        for selected in schedule[1:]:
            boundaries = build_boundaries(model, pools["target"][selected["target_positions"]],
                                          ZeroScore(), spec["sigma"], cfg)
            if len(boundaries):
                break
    if not len(boundaries):
        return dict(asdict(task), status="no_valid_boundary", anchors=0, batches_checked=len(schedule))
    positions = selected["source_positions"]
    problem = TsbaProblem(model, pools["source_fit"][positions], y[positions], boundaries, cfg)
    delta, info = propose(problem, budget, plan["solver"])
    if info["reason"] != "stationary" or bool(delta.any()):
        raise ValueError(f"Protected zero request moves the initial classifier: {task.key}")
    return dict(asdict(task), status="stationary", anchors=len(boundaries), attempt=selected["attempt"])


def adapt(root, plan, item, stop_after=None, *, reference=None):
    task, method = item
    out = job_dir(root, plan, task, method)
    if (out / "complete.json").exists():
        return
    job = r3_job(plan, task, method)
    input_root, input_plan = reference if reference is not None else (root, plan)
    model, pools = inputs(input_root, input_plan, task)
    y = load(labels_path(input_root, task.source, task.seed, "fit"))
    source, target = pools["source_fit"], pools["target"]
    spec = geometry(input_root, input_plan, task)
    sigma = spec["sigma"]
    cfg = config(plan, task.seed)
    budget = Budgets(model, source, y, target, job, plan)
    params = ParameterMap(model)
    initial = params.vector()
    checks = budget.check(params, initial)
    if not checks["risk_ok"] or not checks["confidence_ok"]:
        raise ValueError("Initial head exceeds registered budgets")
    if job.density == "kde":
        estimator, side_estimator = EmpiricalScore(target, sigma), None
    elif job.density == "learned":
        estimator, side_estimator = learned_score(input_root, input_plan, task), None
    else:
        estimator, side_estimator = ZeroScore(), learned_score(input_root, input_plan, task)
    schedule = load(batch_path(input_root, task))
    files = sorted(out.glob("round[0-9][0-9].pt"))
    if files:
        if [p.name for p in files] != [f"round{i:02d}.pt" for i in range(len(files))] or len(files) > len(schedule) + 1:
            raise ValueError(f"Invalid round sequence in {out}")
        params.commit(load(files[-1])["after"])
    else:
        atomic_tensor(out / "round00.pt", dict(after=initial, predictions=predictions(params, initial, pools)))
        files = [out / "round00.pt"]
    if stop_after is not None and len(files) - 1 >= stop_after:
        return
    for batch in schedule[len(files) - 1:]:
        step = batch["attempt"]
        before = params.vector()
        side = step in plan["side_attempts"]
        try:
            record = one_attempt(model, batch, source, y, target, estimator, sigma, budget,
                                 job, cfg, plan, pools, side=side, side_estimator=side_estimator)
        except NUMERICAL_ERRORS as error:
            params.commit(before)
            record = dict(before=before, after=before, batch=batch, status="numerical_failure",
                          error=str(error), accepted=False, trials=[])
            if side:
                record["side"] = dict(status="main_numerical_failure")
        if "true_budgets" not in record:
            record["true_budgets"] = budget.check(params, record["after"])
        record["predictions"] = predictions(params, record["after"], pools)
        record["attempt"] = step
        atomic_tensor(out / f"round{step:02d}.pt", record)
        if step == stop_after:
            return
    write_json(out / "complete.json", dict(asdict(task), method=method, attempts=len(schedule)))
