"""Registered, post-S frozen-W diagnostic; never rewrites the original run.

The stock and independent arms each launch at most ten SLSQP solves (300
iterations each).  The latter reserves *all* launches for seeded new feasible
starts, while preserving every original explanation as a candidate.  Neither
arm refits the reference or changes the observation set.  The full blind phase
is sealed before this process decodes either target truth archive.
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax


PIN = "fea693841b6da36af7d77f92d564f5410ade1808"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def verify_original(root):
    parent = None
    for stage in ("R7_INPUT", "R7_OBS", "R7_ORACLE", "R7_FINAL"):
        record = json.loads((root / "seals" / f"{stage}.json").read_text())
        checksum = record.pop("digest")
        if record["parent"] != parent or signature(record) != checksum:
            raise ValueError(f"Original seal mismatch: {stage}")
        for member, expected in record["files"].items():
            path = root / member
            if not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
                raise ValueError(f"Original member mismatch: {member}")
        parent = checksum
    return parent


def seal(root, stage, names, parent):
    record = dict(stage=stage, parent=parent,
                  files={name: sha(root / name) for name in sorted(names)})
    record["digest"] = signature(record)
    write_json(root / f"{stage}.json", record)
    return record["digest"]


def verify_audit(root, stage):
    record = json.loads((root / f"{stage}.json").read_text())
    checksum = record.pop("digest")
    if checksum != signature(record):
        raise ValueError("Audit seal mismatch")
    for name, expected in record["files"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Audit member mismatch: {name}")
    return checksum


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_family(module, fit, standard, observed, detail, cfg):
    family = module.ExplanationSet(standard["source_u"][:, :-1], fit["source_y"],
                                   standard["adapt_u"][:, :-1], cfg)
    family.reference_loss = float(detail["reference_loss"])
    family.threshold = float(detail["threshold"])
    family._reference = family._candidate(observed["ref_theta"], True)
    if family._reference is None:
        raise ValueError("Frozen reference is infeasible")
    if abs(family._reference.observation_loss - family.reference_loss) > 1e-12:
        raise ValueError("Frozen reference likelihood failed replay")
    original_bounds = detail["reference_fit_diagnostics"]
    if not (np.array_equal(family._lower, original_bounds["lower_bounds"])
            and np.array_equal(family._upper, original_bounds["upper_bounds"])):
        raise ValueError("Frozen parameter bounds failed replay")
    known = [family._candidate(theta, True) for theta in observed["theta"]]
    if any(candidate is None for candidate in known):
        raise ValueError("Original explanation is infeasible")
    if not np.allclose([candidate.statistic for candidate in known], observed["stats"], atol=1e-12, rtol=0):
        raise ValueError("Frozen explanation statistics failed replay")
    # Historical fit trial parameters were not archived. We use the archived
    # reference and retained explanations, never reconstruct an unlogged fit.
    family._fit_candidates = []
    return family, known


def run_search(module, family, known, correction, *, independent):
    """Use unchanged stock objective/constraint/Jacobians in both arms."""
    capture, additions = [], []
    centered = correction - correction.mean(1, keepdims=True)
    magnitude = float(np.max(np.abs(centered)))
    objective_correction = centered / magnitude if magnitude else centered

    def recorded_minimize(fun, initial, *args, **kwargs):
        result = minimize(fun, initial, *args, **kwargs)
        candidate = family._candidate(result.x, True)
        start = family._candidate(initial, True)
        if start is not None:
            additions.append(start)
        if candidate is not None:
            additions.append(candidate)
        raw = family._candidate(result.x)
        capture.append(dict(initial_parameters=np.array(initial).tolist(),
                            result_parameters=result.x.tolist(), success=bool(result.success),
                            status=int(result.status), message=str(result.message), iterations=int(result.nit),
                            feasible_candidate=candidate is not None,
                            observation_loss=None if raw is None else raw.observation_loss,
                            observation_violation=None if raw is None else max(0., raw.observation_loss-family.threshold)))
        return result

    if not independent:
        original_minimize = module.minimize
        module.minimize = recorded_minimize
        try:
            best = family.search(correction, known)
        finally:
            module.minimize = original_minimize
        info = copy.deepcopy(family.last_search)
    else:
        starts, contractions = [], []
        original_bytes = {candidate.parameters.tobytes() for candidate in known}
        for _ in range(40 * family.config.inner_starts):
            proposed = family._perturb(family._reference.parameters)
            for contraction in range(20):
                theta = family._reference.parameters + .5 ** contraction * (proposed-family._reference.parameters)
                candidate = family._candidate(theta, True)
                if candidate is not None and candidate.parameters.tobytes() not in original_bytes:
                    original_bytes.add(candidate.parameters.tobytes())
                    starts.append(candidate)
                    contractions.append(contraction)
                    break
            if len(starts) == family.config.inner_starts:
                break
        if len(starts) != family.config.inner_starts:
            raise RuntimeError("Could not construct the registered independent start count")
        constraint = {"type": "ineq", "fun": lambda theta: family.threshold-family._observation(theta)[0],
                      "jac": lambda theta: -family._observation(theta)[1]}
        for start in starts:
            recorded_minimize(lambda theta: family._separation(theta, objective_correction),
                              start.parameters.copy(), jac=True, method="SLSQP",
                              bounds=list(zip(family._lower, family._upper)), constraints=[constraint],
                              options={"maxiter": family.config.inner_maxiter, "ftol": 1e-10})
        best = min([*known, *additions], key=lambda candidate: family._separation(candidate.parameters, objective_correction)[0])
        successes = sum(row["success"] and row["feasible_candidate"] for row in capture)
        info = dict(status="local_search_completed" if successes else "local_search_failed",
                    successful_feasible_runs=successes, contractions=contractions,
                    strategy="10 new seeded feasible contractions; originals retained only as candidates",
                    global_optimality_certified=False)
    if best is not None:
        additions.append(best)
    known_bytes = {candidate.parameters.tobytes() for candidate in known}
    fresh_launches = sum(np.asarray(row["initial_parameters"], dtype=np.float64).tobytes() not in known_bytes for row in capture)
    info.update(optimizer_runs=capture, fresh_optimizer_launches=fresh_launches,
                total_optimizer_launches=len(capture), fixed_reference_loss=family.reference_loss,
                fixed_threshold=family.threshold, seed=family.config.seed,
                historical_unarchived_fit_candidates_reconstructed=False)
    return family._unique(additions), info


def independent_search(family, correction_logits, *, seed, starts, maxiter, known=()):
    """Return all feasible launch/result candidates without mutating ``family``.

    All optimizer launches are seeded new feasible contractions.  Previously
    known candidates remain in the returned pool, but never consume launches.
    This is a local audit, not a certificate or a replacement reference fit.
    """
    local = copy.deepcopy(family)
    local.config.seed = int(seed)
    local.config.inner_starts = int(starts)
    local.config.inner_maxiter = int(maxiter)
    local._rng = np.random.default_rng(int(seed))
    local._validate_config()
    correction = np.asarray(correction_logits, dtype=np.float64)
    if correction.shape != (len(local.target), local.classes) or not np.isfinite(correction).all():
        raise ValueError("Invalid independent-search correction logits")
    if local._reference is None:
        raise ValueError("Independent search requires a frozen reference")
    candidates = local._unique([local._candidate(item.parameters, True)
                                for item in [local._reference, *known]])
    additions, info = run_search(None, local, candidates, correction, independent=True)
    return local._unique([*candidates, *additions]), info


def master(solver, standard, bank, cfg):
    w, result = solver.solve_master(standard["adapt_z0"], standard["adapt_u"],
                                   np.stack([c.statistic for c in bank]), regularization=cfg.regularization,
                                   feasibility_tol=cfg.feasibility_tol, optimality_tol=cfg.optimality_tol,
                                   master_maxiter=cfg.master_maxiter)
    return w, result


def blind_block(root, output, seed, config, module, solver):
    block = root / "blocks" / f"well_specified-seed-{seed}"
    fit = read_npz(block / "fit_input.npz")
    standard = read_npz(block / "standardization.npz")
    observed = read_npz(block / "tau-0.010-observed.npz")
    detail = json.loads((block / "tau-0.010-observed.json").read_text())["J"]
    cfg = SimpleNamespace(**config, seed=seed+7001)
    family, known = frozen_family(module, fit, standard, observed, detail, cfg)
    w = observed["w_J"]
    risk = lambda bank, update=w: solver.relative_risks(standard["adapt_z0"], standard["adapt_u"], update,
                                                      np.stack([candidate.statistic for candidate in bank]))
    original_worst = float(risk(known).max())
    if abs(original_worst-detail["found_worst_risk"]) > 1e-12:
        raise ValueError("Original fixed-W worst risk failed replay")
    stock, stock_info = run_search(module, copy.deepcopy(family), known, standard["adapt_u"] @ w.T, independent=False)
    random, random_info = independent_search(family, standard["adapt_u"] @ w.T, seed=seed+7001,
                                            starts=10, maxiter=300, known=known)
    if random_info["fresh_optimizer_launches"] != 10:
        raise ValueError("Independent seed failed to contribute all ten launches")
    banks = dict(original=known, stock=family._unique([*known, *stock]),
                 independent=family._unique([*known, *random]), union=family._unique([*known, *stock, *random]))
    corrections, masters = {"fixed_original": w}, {}
    for name, bank in banks.items():
        corrections[f"resolved_{name}"], masters[name] = master(solver, standard, bank, cfg)
        if not masters[name]["valid"]:
            masters[name]["failed_trial_w"] = corrections[f"resolved_{name}"].tolist()
            corrections[f"resolved_{name}"] = np.zeros_like(w)
    # Every newly accepted candidate receives a fixed-W re-risk and a prefix
    # finite-master re-solve. This is diagnostic; no result chooses later starts.
    trace, prefix = [], list(known)
    for candidate in banks["union"][len(known):]:
        prefix.append(candidate)
        update, record = master(solver, standard, prefix, cfg)
        trace.append(dict(added_index=len(prefix)-1, parameters=candidate.parameters,
                          observation_loss=candidate.observation_loss, fixed_W_risk=float(risk([candidate])[0]),
                          correction=update, master=record))
    arrays = {f"theta_{name}": np.stack([c.parameters for c in bank]) for name, bank in banks.items()}
    arrays.update({f"stats_{name}": np.stack([c.statistic for c in bank]) for name, bank in banks.items()})
    arrays.update({f"w_{name}": update for name, update in corrections.items()})
    # H features are accessed only after all searches and master solves above.
    held = read_npz(block / "holdout_standardization.npz")
    for name, update in corrections.items():
        arrays[f"adapt_logits_{name}"] = standard["adapt_z0"] + standard["adapt_u"] @ update.T
        arrays[f"holdout_logits_{name}"] = held["holdout_z0"] + held["holdout_u"] @ update.T
    risk_report = {name: dict(count=len(bank), candidate_losses=[c.observation_loss for c in bank],
                             candidate_fixed_W_risks=risk(bank), worst_fixed_W_risk=float(risk(bank).max()),
                             exceeds_original_by=float(risk(bank).max()-original_worst),
                             invalidates_original_worst=bool(risk(bank).max()>original_worst+1e-6),
                             harmful_fixed_W_witness=bool(risk(bank).max()>1e-6)) for name, bank in banks.items()}
    prefix = f"seed-{seed}"
    write_npz(output / f"{prefix}-blind.npz", **arrays)
    write_json(output / f"{prefix}-blind.json", dict(seed=seed, original_valid=detail["valid"],
               original_stop_reason=detail["stop_reason"], reference_loss=family.reference_loss, threshold=family.threshold,
               stock=stock_info, independent=random_info, banks=risk_report, masters=masters, candidate_trace=trace))
    print(json.dumps(dict(seed=seed, phase="blind_complete", stock_successes=stock_info["successful_feasible_runs"],
                         independent_successes=random_info["successful_feasible_runs"],
                         stock_fresh_launches=stock_info["fresh_optimizer_launches"],
                         independent_fresh_launches=random_info["fresh_optimizer_launches"])), flush=True)
    return [f"{prefix}-blind.npz", f"{prefix}-blind.json"]


def hull_distance(statistic, bank):
    matrix = (bank-statistic).reshape(len(bank), -1)
    scale = float(np.max(np.abs(matrix)))
    if not scale:
        return 0.
    normalized = matrix / scale
    def objective(weights):
        residual = weights @ normalized
        return .5*float(residual@residual), normalized@residual
    candidates = []
    for initial in [np.ones(len(bank))/len(bank), np.eye(len(bank))[np.argmin((matrix**2).sum(1))]]:
        result = minimize(objective, initial, jac=True, method="SLSQP", bounds=[(0., 1.)]*len(bank),
                          constraints={"type":"eq", "fun":lambda x:x.sum()-1., "jac":lambda x:np.ones_like(x)},
                          options={"ftol":1e-14, "maxiter":1000})
        _, gradient = objective(result.x)
        if max(abs(result.x.sum()-1), max(0., -result.x.min()), float(result.x@gradient-gradient.min())) <= 1e-6:
            candidates.append(float(np.linalg.norm(result.x@matrix)))
    return min(candidates) if candidates else None


def evaluate(root, output, seed, module, solver, config):
    verify_audit(output, "AUDIT_OBS")
    block = root / "blocks" / f"well_specified-seed-{seed}"
    commitments = json.loads((block / "commitments.json").read_text())
    truth = {}
    for pool in ["adapt", "holdout"]:
        name = f"secret_{pool}.npz"
        if sha(block/name) != commitments[name]:
            raise ValueError("Target truth commitment changed")
        truth.update(read_npz(block/name))
    standard, held = read_npz(block/"standardization.npz"), read_npz(block/"holdout_standardization.npz")
    arrays = read_npz(output/f"seed-{seed}-blind.npz")
    records = json.loads((output/f"seed-{seed}-blind.json").read_text())
    a, h = standard["adapt_u"], held["holdout_u"]
    az, hz = standard["adapt_z0"], held["holdout_z0"]
    b0 = hz.argmax(1)
    y = truth["holdout_y"]
    ba = lambda pred: float(np.mean([np.mean(pred[y==c]==c) for c in range(2)]))
    base_ba = ba(b0)
    true_stat = truth["adapt_q"].T @ a / len(a)
    source_stat = softmax(az, axis=1).T @ a / len(a)
    rows = []
    for method in ["fixed_original", "resolved_original", "resolved_stock", "resolved_independent", "resolved_union"]:
        w = arrays[f"w_{method}"]
        bank_name = "original" if method=="fixed_original" else method.removeprefix("resolved_")
        bank = arrays[f"stats_{bank_name}"]
        ahat, hhat = arrays[f"adapt_logits_{method}"], arrays[f"holdout_logits_{method}"]
        delta_h = -np.sum(truth["holdout_q"]*(log_softmax(hhat,axis=1)-log_softmax(hz,axis=1)),axis=1)
        true_risk = float(solver.relative_risks(az,a,w,np.array([true_stat]))[0])
        worst = float(solver.relative_risks(az,a,w,bank).max())
        change = hhat-hz
        change -= change.mean(1,keepdims=True)
        pred = hhat.argmax(1)
        rows.append(dict(seed=seed,method=method,finite_master_valid=(True if method=="fixed_original" else
                                                                 records["masters"][bank_name]["valid"]),
                         original_algorithm_valid=records["original_valid"],
                         search_successes=(None if bank_name=="original" else
                             records[bank_name]["successful_feasible_runs"] if bank_name in ("stock","independent") else
                             records["stock"]["successful_feasible_runs"]+records["independent"]["successful_feasible_runs"]),
                         holdout_expected_nll_delta=float(delta_h.mean()),holdout_ba_delta_pp=100*(ba(pred)-base_ba),
                         centered_rms=float(np.sqrt(np.mean(change**2))),true_directional_risk=true_risk,
                         retained_worst_risk=worst,true_directional_risk_gap=true_risk-worst,
                         source_moment_distance=hull_distance(source_stat,bank),true_moment_distance=hull_distance(true_stat,bank),
                         corrected=int(np.sum((b0!=y)&(pred==y))),harmed=int(np.sum((b0==y)&(pred!=y))),
                         candidate_count=len(bank),scope="finite bank; not full-set verified"))
    write_json(output/f"seed-{seed}-evaluation.json",rows)
    return rows


def run(original, output, repository):
    original, output, repository = map(lambda p: Path(p).resolve(), (original, output, repository))
    original_digest = verify_original(original)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/"snapshot"
    snapshot.mkdir()
    provenance = json.loads((original/"provenance.json").read_text())
    files = []
    for filename in ["explanations.py", "solver.py"]:
        relative = f"tsba/shared_update/{filename}"
        content = subprocess.check_output(["git","show",f"{PIN}:{relative}"],cwd=repository)
        if hashlib.sha256(content).hexdigest()!=provenance["file_hashes"][relative]:
            raise ValueError(f"Pinned module differs from original execution: {relative}")
        with (snapshot/filename).open("xb") as stream:
            stream.write(content)
        files.append(f"snapshot/{filename}")
    module = load_module("r7_audit_stock_explanations",snapshot/"explanations.py")
    solver = load_module("r7_audit_stock_solver",snapshot/"solver.py")
    plan = json.loads((original/"executed-plan.json").read_text())
    config = dict(json.loads((original/"executed-configs.json").read_text())["common"][0]["configs"][1])
    config.pop("seed")
    config.update(inner_starts=10,inner_maxiter=300)
    if config["tau"]!=.01 or plan["method_commit"]!=PIN:
        raise ValueError("Unexpected original primary configuration")
    write_json(output/"audit-plan.json",dict(original=str(original),original_final_digest=original_digest,
               method_commit=PIN,config=config,seeds=list(range(5)),seed_offset=7001,
               stock_arm="unchanged stock search with archived reference/retained candidates; 10 launches maximum",
               independent_arm="all 10 launches are new seed+7001 feasible perturbations; known candidates retained",
               reference_refitted=False,target_truth_access="after AUDIT_OBS for all five seeds",
               original_data_previously_evaluated=True,confirmation=False,
               source_sha256=sha(__file__),python=sys.version,numpy=np.__version__))
    with (snapshot/"separation_audit.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    files.extend(["audit-plan.json","snapshot/separation_audit.py"])
    input_digest = seal(output,"AUDIT_INPUT",files,original_digest)
    blind = []
    for seed in range(5):
        blind.extend(blind_block(original,output,seed,config,module,solver))
    obs_digest = seal(output,"AUDIT_OBS",blind,input_digest)
    rows = []
    for seed in range(5):
        rows.extend(evaluate(original,output,seed,module,solver,config))
    with (output/"summary.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    final_files=[f"seed-{seed}-evaluation.json" for seed in range(5)]+["summary.csv"]
    final_digest=seal(output,"AUDIT_FINAL",final_files,obs_digest)
    if verify_original(original)!=original_digest:
        raise ValueError("Original archive changed during audit")
    print(json.dumps(dict(phase="final",output=str(output),digest=final_digest)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--repository",default=str(Path(__file__).resolve().parents[4]))
    args=parser.parse_args()
    run(args.original,args.output,args.repository)
