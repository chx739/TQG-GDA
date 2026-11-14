"""Prepare, train, replay and evaluate one registered R5 setting."""
import argparse
from collections import Counter
import fcntl
import importlib.metadata
from pathlib import Path
import os
import shutil
import subprocess
import traceback
from uuid import uuid4

import torch
from tsba.classifier import ParameterMap
from .shared import io, output, p3, p4, training, t4, t3, e4, run_stage, digest, provenance, assert_nested_equal
from .protocol import resolve, prepare, reuse
from .trajectory import adapt
from .diagnostics import side_diagnostics


def check_failure(root):
    if list((root / "terminal_failures").glob("*.json")):
        raise RuntimeError("Numerical failure: retain this archive and register a new attempt after repair")


def guarded(root, plan, item, operation):
    check_failure(root)
    try:
        return operation(root, plan, item)
    except t3.NUMERICAL_ERRORS:
        io.write_json(root / "terminal_failures" / f"{uuid4().hex}.json",
                      dict(stage=operation.__name__, item=str(item), error=traceback.format_exc()))
        raise


def source_worker(root, plan, item):
    return guarded(root, plan, item, training.train_source)


def score_worker(root, plan, item):
    return guarded(root, plan, item, training.train_score)


def adapt_worker(root, plan, item):
    return guarded(root, plan, item, adapt)


def replay(root, plan):
    counts, sides = Counter(), Counter()
    for source in plan["domains"]:
        for seed in plan["seeds"]:
            e4.verify_source(root, plan, source, seed)
            counts["sources"] += 1
    for task in p4.tasks(plan):
        directory = p4.score_dir(root, plan, task)
        cfg = p4.config(plan, task.seed)
        steps = (len(p4.domain(root, task.target)) + cfg.score_batch - 1) // cfg.score_batch
        for updates in plan["score"]["endpoints"]:
            checkpoint = io.load(directory / f"epoch{updates // steps:04d}.pt")
            if checkpoint["updates"] != updates or not {"optimizer_state", "training_rng_state",
                    "torch_rng_state", "validation_rng_state"} <= checkpoint.keys():
                raise ValueError("Incomplete score checkpoint")
            counts["score_endpoints"] += 1
        counts["scores"] += 1
        model, pools = t4.inputs(root, plan, task)
        params = ParameterMap(model)
        initial = params.vector()
        baseline = t3.predictions(params, initial, pools)
        path = root / "baselines" / f"{task.key}.pt"
        if path.exists():
            assert_nested_equal(io.load(path), baseline)
        else:
            io.atomic_tensor(path, baseline)
        schedule = io.load(p4.batch_path(root, task))
        for method in ("B1", "A0"):
            local_plan = dict(plan, side_attempts=plan["side_attempts"] if method == "A0" else [])
            directory = p4.job_dir(root, plan, task, method)
            e4.replay_trajectory(directory, local_plan, params, pools, initial, schedule, counts, sides)
            for step in plan["side_attempts"] if method == "A0" else []:
                record = io.load(directory / f"round{step:02d}.pt")
                y = io.load(p4.labels_path(root, task.source, task.seed, "fit"))
                sigma = training.geometry(root, plan, task)["sigma"]
                detail = side_diagnostics(model, pools, y, record, sigma, cfg, plan)
                if "r5_side" in record:
                    assert_nested_equal(detail, record["r5_side"])
                output_path = root / "diagnostics" / task.key / f"side{step:02d}.pt"
                if output_path.exists():
                    assert_nested_equal(io.load(output_path), detail)
                else:
                    io.atomic_tensor(output_path, detail)
    expected = plan["counts"]
    if any(counts[k] != expected[k] for k in expected) or counts["predictions_replayed"] != 3720:
        raise ValueError(f"Replay count mismatch {counts}")
    if sum(sides.values()) != 120 or counts["side_predictions_replayed"] != 3 * sides["available"]:
        raise ValueError("Side coverage mismatch")
    counts.setdefault("side_predictions_replayed", 0)
    return dict(counts, side_statuses=dict(sides))


def initialize(root, repo, requested, plan):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("Commit reviewed implementation before research execution")
    stage = root.parent / f".{root.name}.prepare-{uuid4().hex}"
    stage.mkdir(parents=True)
    io.write_json(stage / "protocol/requested-plan.json", requested)
    io.write_json(stage / "protocol/plan.json", plan)
    io.write_json(stage / "protocol/provenance.json", provenance(repo))
    io.write_json(stage / "protocol/environment.json", {
        name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "torch-geometric", "matplotlib")})
    with (stage / "protocol/source.tar").open("xb") as stream:
        subprocess.run(["git", "archive", "HEAD"], cwd=repo, stdout=stream, check=True)
    try:
        prepare(stage, repo, plan)
        if plan["setting"]["action"] == "reuse":
            reuse(stage, plan)
        io.seal(stage, "R5_INPUT", ["protocol", "inputs", "roles", "source_labels", "batches"])
    except Exception:
        io.write_json(stage / "preparation-failure.json", dict(error=traceback.format_exc()))
        raise
    stage.rename(root)


def execute(args):
    repo = Path(__file__).resolve().parents[4]
    requested = p3.read_json(args.plan)
    plan = resolve(requested, args.setting)
    root = args.root.resolve()
    if not root.exists():
        initialize(root, repo, requested, plan)
    check_failure(root)
    io.verify(root, "R5_INPUT")
    if p3.read_json(root / "protocol/plan.json") != plan:
        raise ValueError("Registered plan changed")
    for path, expected in p3.read_json(root / "protocol/provenance.json")["file_hashes"].items():
        if path.endswith(".py") and digest(repo / path) != expected:
            raise ValueError(f"Registered implementation changed: {path}")
    if args.stage == "prepare":
        return
    if (root / "seals/R5_FINAL.json").exists():
        io.verify(root, "R5_OBS")
        io.verify(root, "R5_FINAL")
        return
    if not (root / "seals/R5_OBS.json").exists():
        if args.stage == "evaluate":
            raise PermissionError("Evaluation requires this setting's R5_OBS")
        tasks = p4.tasks(plan)
        if plan["setting"]["action"] == "fresh" and args.stage != "replay":
            sources = [(s, seed) for s in plan["domains"] for seed in plan["seeds"]]
            run_stage(root, plan, "source", source_worker, sources)
            run_stage(root, plan, "score", score_worker, tasks)
            run_stage(root, plan, "adapt", adapt_worker, [(t, m) for t in tasks for m in ("B1", "A0")])
        result = replay(root, plan)
        output.write_json(root / "replay.json", result)
        io.seal(root, "R5_OBS", ["protocol", "inputs", "roles", "source_labels", "batches",
                               "source", "scores", "trajectories", "baselines", "diagnostics",
                               "replay.json", "seals/R5_INPUT.json"])
        print("R5_OBS sealed", plan["setting"]["id"], result, flush=True)
    if args.stage == "replay":
        return
    from .evaluate import evaluate
    evaluate(root, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "run", "replay", "evaluate"), default="run")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError(f"Set {key}=1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.root.parent.mkdir(parents=True, exist_ok=True)
    with args.root.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args)
        except t3.NUMERICAL_ERRORS:
            if args.root.exists():
                io.write_json(args.root / "terminal_failures" / f"{uuid4().hex}.json",
                              dict(stage=args.stage, error=traceback.format_exc()))
            raise
