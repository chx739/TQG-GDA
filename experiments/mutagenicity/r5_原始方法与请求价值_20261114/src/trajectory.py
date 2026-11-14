"""R5 M00 adaptation uses the R3 kernel and stops on numerical exceptions."""
from tsba.classifier import ParameterMap
from .shared import io, p4, t4, t3, constraints, training
from .diagnostics import side_diagnostics
from experiments.mutagenicity.r2_分数训练预算与多轮适配_20261111.src.adapt import ZeroScore
from dataclasses import asdict
load, atomic_tensor, write_json = io.load, io.atomic_tensor, io.write_json
job_dir, r3_job, labels_path, config, batch_path = p4.job_dir, p4.r3_job, p4.labels_path, p4.config, p4.batch_path
inputs = t4.inputs
geometry, learned_score = training.geometry, training.learned_score
Budgets = constraints.Budgets
one_attempt, predictions, NUMERICAL_ERRORS = t3.one_attempt, t3.predictions, t3.NUMERICAL_ERRORS

def adapt(root, plan, item, stop_after=None):
    task, method = item
    if method not in ("B1", "A0"):
        raise ValueError("R5 permits only B1 and A0 adaptation")
    out = job_dir(root, plan, task, method)
    if list(out.glob("numerical-failure-*.json")):
        raise RuntimeError("Numerically failed trajectory requires a new attempt directory")
    if (out / "complete.json").exists():
        return
    job = r3_job(plan, task, method)
    input_root, input_plan = root, plan
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
    if job.density == "learned":
        estimator, side_estimator = learned_score(input_root, input_plan, task), None
    else:
        estimator, side_estimator = ZeroScore(), None
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
        side = method == "A0" and step in plan["side_attempts"]
        try:
            record = one_attempt(model, batch, source, y, target, estimator, sigma, budget,
                                 job, cfg, plan, pools, side=side, side_estimator=side_estimator)
        except NUMERICAL_ERRORS as error:
            params.commit(before)
            write_json(out / f"numerical-failure-{step:02d}.json", dict(attempt=step, error=str(error)))
            raise
        if record.get("side", {}).get("status") == "numerical_failure" or record.get("density_error") or record.get("relocation_error"):
            write_json(out / f"numerical-failure-{step:02d}.json", dict(attempt=step, error="Numerical diagnostic failure"))
            raise FloatingPointError("Numerical failure requires a new attempt after repair")
        if side:
            try:
                record["r5_side"] = side_diagnostics(model, pools, y, record, sigma, cfg, plan)
            except NUMERICAL_ERRORS as error:
                write_json(out / f"numerical-failure-{step:02d}.json", dict(attempt=step, error=str(error)))
                raise
        if "true_budgets" not in record:
            record["true_budgets"] = budget.check(params, record["after"])
        record["predictions"] = predictions(params, record["after"], pools)
        record["attempt"] = step
        atomic_tensor(out / f"round{step:02d}.pt", record)
        if step == stop_after:
            return
    write_json(out / "complete.json", dict(asdict(task), method=method, attempts=len(schedule)))
