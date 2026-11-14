"""Run the registered R3 study and resume from its published attempt records."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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

from experiments.common.io import digest, provenance
from experiments.mutagenicity.score_budget_and_adaptation_20261111.src.common import seal, verify, write_json
from experiments.mutagenicity.risk_and_confidence_constraints_20261112.src.evaluate import evaluate, replay
from experiments.mutagenicity.risk_and_confidence_constraints_20261112.src.protocol import grid, prepare, read_json
from experiments.mutagenicity.risk_and_confidence_constraints_20261112.src.training import score_fit, source_fit
from experiments.mutagenicity.risk_and_confidence_constraints_20261112.src.trajectory import check_sensitivity_zero, trajectory


def configure():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def worker(stage, root, item, plan):
    if stage == "source":
        source_fit(root, item, plan)
    elif stage == "scores":
        score_fit(root, item, plan)
    elif stage == "adapt":
        trajectory(root, item, plan)
    else:
        raise ValueError(stage)


def parallel(stage, root, items, plan, workers):
    marker = root / "stages" / f"{stage}.json"
    if marker.exists():
        return
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(workers, len(items)),
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=configure) as executor:
        pending = {executor.submit(worker, stage, root, item, plan): item for item in items}
        failures = []
        for count, future in enumerate(as_completed(pending), start=1):
            try:
                future.result()
            except Exception:
                failures.append(dict(job=str(pending[future]), error=traceback.format_exc()))
                print(f"{stamp()} {stage} failed {pending[future]}\n{failures[-1]['error']}", flush=True)
            if stage != "adapt" or count % 10 == 0 or count == len(items):
                print(f"{stamp()} {stage} {count}/{len(items)} processed", flush=True)
    if failures:
        write_json(root / "failures" / f"{stage}-{uuid4().hex}.json", failures)
        raise RuntimeError(f"{stage}: {len(failures)} worker failures, see logs and failures/")
    write_json(marker, dict(count=len(items), elapsed_seconds=time.monotonic() - started,
                            workers=workers, completed_utc=stamp()))


def initialize(root, repo, plan_path, request):
    root.mkdir(parents=True, exist_ok=False)
    p = root / "protocol"
    p.mkdir()
    for file in (plan_path, plan_path.parent.parent / "docs/r3-protocol.md",
                 plan_path.parent.parent.parent / "stage1-inputs.json"):
        shutil.copyfile(file, p / file.name)
    if request is not None:
        shutil.copyfile(request, p / "requested-design.txt")
    git = provenance(repo)
    write_json(p / "provenance.json", git)
    write_json(p / "environment.json", dict(
        python=sys.version, torch=torch.__version__,
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "torch-geometric", "matplotlib")},
        threads={name: os.environ.get(name) for name in
                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        deterministic_algorithms=True))
    with (p / "source.tar").open("xb") as stream:
        subprocess.run(["git", "archive", "HEAD"], cwd=repo, stdout=stream, check=True)
    prepare(root, repo, read_json(plan_path), read_json(p / "stage1-inputs.json"))
    seal(root, "R3_INPUT", ["inputs", "roles", "source_labels", "batches", "protocol"])


def execution(root, repo, plan_path, args):
    if not root.exists():
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
            raise ValueError("Commit the reviewed implementation and plan before initializing the run")
        staging = root.parent / f".{root.name}.input-{uuid4().hex}"
        initialize(staging, repo, plan_path, args.request)
        staging.rename(root)
    verify(root, "R3_INPUT")
    registered = root / "protocol/r3-plan.json"
    if digest(plan_path) != digest(registered):
        raise ValueError("Registered plan differs from the running plan")
    for path, expected in read_json(root / "protocol/provenance.json")["file_hashes"].items():
        if path.endswith(".py") and digest(repo / path) != expected:
            raise ValueError(f"Registered code changed: {path}")

    plan = read_json(registered)
    jobs = grid(plan)
    workers = args.workers or plan["workers"]
    if workers < 1:
        raise ValueError("Positive worker count required")
    if (root / "seals/R3_FINAL.json").exists():
        verify(root, "R3_OBS")
        verify(root, "R3_FINAL")
        print(f"{stamp()} R3 complete at {root}", flush=True)
        return
    if not (root / "seals/R3_OBS.json").exists():
        parallel("source", root, plan["seeds"], plan, workers)
        specs = [(seed, domain, h) for seed in plan["seeds"] for domain in plan["domains"]
                 for h in plan["score"]["bandwidths"]]
        parallel("scores", root, specs, plan, workers)
        marker = root / "stages/sensitivity_zero.json"
        if not marker.exists():
            rows = [check_sensitivity_zero(root, s, d, plan)
                    for s in plan["seeds"] for d in plan["domains"]]
            write_json(marker, rows)
        pilot = root / "stages/pilot.json"
        if not pilot.exists():
            times = []
            for group in plan["groups"]:
                job = next(j for j in jobs if j.group == group and j.density == "learned"
                           and j.bandwidth == 1 and j.direction == 1)
                start = time.monotonic()
                trajectory(root, job, plan, stop_after=3)
                times.append(dict(job=asdict(job), attempts=3, elapsed_seconds=time.monotonic() - start))
            write_json(pilot, times)
            print(f"{stamp()} pilot complete {times}", flush=True)
        parallel("adapt", root, jobs, plan, workers)
        result = replay(root, plan)
        from experiments.mutagenicity.score_budget_and_adaptation_20261111.src.evaluate import write_json as verified_write_json
        verified_write_json(root / "replay.json", result)
        paths = ["seals/R3_INPUT.json", "inputs", "roles", "source_labels", "batches",
                 "protocol", "source", "scores", "trajectories", "baselines", "stages", "replay.json"]
        seal(root, "R3_OBS", paths)
        print(f"{stamp()} sealed observation {result}", flush=True)
    evaluate(root, plan)
    print(f"{stamp()} R3 complete at {root}", flush=True)
