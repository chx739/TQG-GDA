"""Command-line entry for this experiment."""
import argparse
import fcntl
import os
from pathlib import Path
from experiments.mutagenicity.r3_risk_and_confidence_constraints_20261112.src.execution import configure, execution
from experiments.common.paths import repository_root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--workers", type=int, help="CPU scheduling only")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"Set {variable}=1 for reproducible per-process CPU execution")
    configure()
    repo = repository_root()
    plan_path = repo / "experiments/mutagenicity/r3_risk_and_confidence_constraints_20261112/configs/r3-plan.json"
    root = args.root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    # A single orchestrator owns a run directory even across terminal sessions.
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execution(root, repo, plan_path, args)
