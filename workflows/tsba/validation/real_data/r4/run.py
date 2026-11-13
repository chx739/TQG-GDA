"""Run the fixed Mutagenicity R4 plan from an atomic input registration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import importlib.metadata
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import torch

from ...archive import digest, provenance
from ..r2.common import seal, verify, write_json
from ..r2.evaluate import write_json as verified_json
from ..r3.run import configure, stamp
from .evaluate import evaluate, replay
from .protocol import config_hash, prepare, read_json, tasks
from .training import train_score, train_source
from .trajectory import adapt, check_zero


def run_stage(root, plan, stage, operation, items):
    marker = root / "stages" / (stage + ".json")
    if marker.exists():
        if read_json(marker)["count"] != len(items):
            raise ValueError(f"Stage marker count differs for {stage}")
        return
    start = time.monotonic()
    failures = []
    workers = min(plan["workers"], len(items))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=configure) as executor:
        pending = {executor.submit(operation, root, plan, item): item for item in items}
        for done, future in enumerate(as_completed(pending), 1):
            try:
                future.result()
            except Exception:
                failures.append(dict(item=str(pending[future]), error=traceback.format_exc()))
                print(f"{stamp()} {stage} failure {failures[-1]}", flush=True)
            if done % 5 == 0 or done == len(items):
                print(f"{stamp()} {stage} {done}/{len(items)} processed", flush=True)
    if failures:
        write_json(root / "failures" / f"{stage}-{uuid4().hex}.json", failures)
        raise RuntimeError(f"{len(failures)} {stage} worker failures; prior checkpoints are retained")
    write_json(marker, dict(count=len(items), workers=workers, elapsed_seconds=time.monotonic()-start))


def initialize(root, repo, plan_path, reference, request, verification_log):
    staging = root.parent / f".{root.name}.input-{uuid4().hex}"
    p = staging / "protocol"
    p.mkdir(parents=True)
    for path in (plan_path, plan_path.with_name("r4-protocol.md")):
        shutil.copyfile(path, p / path.name)
    if request:
        shutil.copyfile(request, p / "requested-design.txt")
    if verification_log:
        shutil.copyfile(verification_log, p / "preflight-tests.log")
    write_json(p / "provenance.json", provenance(repo))
    write_json(p / "environment.json", dict(
        python=sys.version, torch=torch.__version__, deterministic_algorithms=True,
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "torch-geometric", "matplotlib")},
        threads={name: os.environ.get(name) for name in
                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}))
    with (p / "source.tar").open("xb") as stream:
        subprocess.run(["git", "archive", "HEAD"], cwd=repo, stdout=stream, check=True)
    plan = read_json(plan_path)
    prepare(staging, repo, plan, reference)
    write_json(p / "config-identity.json", dict(config_hash=config_hash(plan)))
    seal(staging, "R4_INPUT", ["protocol", "inputs", "roles", "source_labels", "batches"])
    staging.rename(root)


def execute(root, repo, plan_path, args):
    if not root.exists():
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
            raise ValueError("Commit the reviewed registration before starting R4")
        initialize(root, repo, plan_path, args.r3, args.request, args.verification_log)
    verify(root, "R4_INPUT")
    registered = root / "protocol/r4-plan.json"
    if digest(plan_path) != digest(registered):
        raise ValueError("R4 plan differs from the sealed registration")
    for path, expected in read_json(root / "protocol/provenance.json")["file_hashes"].items():
        if path.endswith(".py") and digest(repo / path) != expected:
            raise ValueError(f"Registered code changed: {path}")

    if (root / "seals/R4_FINAL.json").exists():
        verify(root, "R4_OBS")
        verify(root, "R4_FINAL")
        print(f"{stamp()} R4 complete at {root}", flush=True)
        return
    plan = read_json(registered)
    if not (root / "seals/R4_OBS.json").exists():
        registered_tasks = tasks(plan)
        sources = [(s, seed) for s in plan["domains"] for seed in plan["seeds"]]
        jobs = [(task, m) for task in registered_tasks for m in plan["methods"] if m != "B0"]
        if dict(sources=len(sources), scores=len(registered_tasks), trajectories=len(jobs),
                attempts=len(jobs) * plan["model"]["adapt_rounds"]) != plan["counts"]:
            raise ValueError("R4 work count mismatch")
        run_stage(root, plan, "source", train_source, sources)
        run_stage(root, plan, "scores", train_score, registered_tasks)
        zero = root / "stages/zero.json"
        if not zero.exists():
            write_json(zero, [check_zero(root, plan, task) for task in registered_tasks])
        pilot = root / "stages/pilot.json"
        if not pilot.exists():
            records = []
            for source, target, method in ((3, 2, "A2"), (2, 3, "B1")):
                task = next(t for t in registered_tasks if (t.source, t.target) == (source, target))
                start = time.monotonic()
                adapt(root, plan, (task, method), stop_after=3)
                records.append(dict(task=task.key, method=method, attempts=3,
                                    elapsed_seconds=time.monotonic()-start))
            write_json(pilot, records)
            print(f"{stamp()} pilots complete {records}", flush=True)
        run_stage(root, plan, "adapt", adapt, jobs)
        result = replay(root, plan)
        verified_json(root / "replay.json", result)
        seal(root, "R4_OBS", ["seals/R4_INPUT.json", "protocol", "inputs", "roles", "source_labels",
                             "batches", "source", "scores", "trajectories", "baselines", "stages", "replay.json"])
        print(f"{stamp()} sealed observation {result}", flush=True)
    evaluate(root, plan)
    print(f"{stamp()} R4 complete at {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--r3", type=Path, required=True, help="Archived R3 input reference")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--verification-log", type=Path)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(var) != "1":
            raise ValueError(f"Set {var}=1 for deterministic per-process execution")
    configure()
    repo = Path(__file__).resolve().parents[5]
    plan_path = repo / "docs/tsba/experiments/mutagenicity/r4-plan.json"
    root = args.root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(root, repo, plan_path, args)


if __name__ == "__main__":
    main()
