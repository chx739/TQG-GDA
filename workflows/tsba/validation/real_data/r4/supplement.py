"""Run and evaluate the post hoc A0 control against the sealed Mutagenicity R4."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
from datetime import datetime
import fcntl
from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4
from zoneinfo import ZoneInfo

import torch

from tsba.classifier import ParameterMap
from ...archive import digest, provenance
from ..data import canonical_sha
from ..r2.common import load, seal, verify, write_json
from ..r2.evaluate import write_csv, write_json as verified_json
from ..r2.score import assert_nested_equal
from ..r3.protocol import read_json
from ..r3.run import configure, stamp
from ..r3.training import score_dir as r3_score_dir, source_dir as r3_source_dir
from ..r3.trajectory import predictions
from .evaluate import METRICS, replay_trajectory, summarize, trajectory_rows
from .protocol import (
    batch_path, config, config_hash, domain, evaluation_labels, job_dir, labels_path,
    r3_job, role, score_dir, source_dir, tasks,
)
from .report import report
from .run import run_stage
from .trajectory import adapt, inputs

METHOD = "A0"
COUNTS = dict(trajectories=60, attempts=1800, predictions_replayed=1860)
PAIRS = (("A0", "B0"), ("A0", "B1"), ("A1", "A0"), ("A2", "A0"))


def supplement_plan(base):
    plan = {**base, "methods": dict(base["methods"]), "pairs": [*base["pairs"], *map(list, PAIRS)]}
    plan["methods"][METHOD] = dict(group="M00", density="learned")
    plan["counts"] = {**base["counts"],
                      "trajectories": base["counts"]["trajectories"] + COUNTS["trajectories"],
                      "attempts": base["counts"]["attempts"] + COUNTS["attempts"]}
    plan["reuse"] = "sealed_r4_inputs_for_a0_only"
    return plan


def references(base, plan, obs):
    records = []
    for task in tasks(plan):
        source = source_dir(base, plan, task.source, task.seed)
        score = score_dir(base, plan, task)
        paths = dict(
            model=source / "source.pt", frozen=source / "frozen.pt",
            geometry=source / f"geometry_M{task.target}.pt",
            score=score / f"epoch{config(plan, task.seed).score_epochs:04d}.pt",
            batches=batch_path(base, task), roles=base / "roles" / f"{task.source_key}.json",
            source_fit_labels=labels_path(base, task.source, task.seed, "fit"),
            baseline=base / "baselines" / f"{task.key}.pt",
        )
        files = {}
        for name, path in paths.items():
            relative = str(path.relative_to(base))
            files[name] = dict(path=relative, sha256=obs["files"][relative])
        records.append(dict(**asdict(task), method=METHOD, files=files))
    shared = {name: obs["files"][name] for name in
              ("inputs/domains.json", "inputs/label_records.json", "protocol/r4-plan.json")}
    return dict(r4_root=str(base), observation=obs["digest"], shared=shared, tasks=records)


def register(root, base, r3, repo, plan, refs, r3_digest):
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("Commit the supplement implementation before running it")
    staging = root.parent / f".{root.name}.input-{uuid4().hex}"
    protocol = staging / "protocol"
    write_json(protocol / "plan.json", plan)
    write_json(protocol / "references.json", refs)
    write_json(protocol / "registration.json", dict(
        post_hoc=True, method=METHOD, counts=COUNTS, r3_root=str(r3), r3_observation=r3_digest,
        date=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        config_hash=config_hash(plan), side_attempts=plan["side_attempts"],
        jobs=[str(job_dir(staging, plan, task, METHOD).relative_to(staging)) for task in tasks(plan)],
    ))
    write_json(protocol / "provenance.json", provenance(repo))
    with (protocol / "source.tar").open("xb") as stream:
        subprocess.run(["git", "archive", "HEAD"], cwd=repo, stdout=stream, check=True)
    seal(staging, "A0_INPUT", ["protocol"])
    staging.rename(root)


def historical_parity(root, base, r3, plan, base_plan, r3_digest):
    """Check the sealed R3 comparison inputs and all 31 A0 states on historical tasks."""
    obs = read_json(r3 / "seals/R3_OBS.json")
    if (obs["digest"] != r3_digest or obs["stage"] != "R3_OBS"
            or canonical_sha({k: obs[k] for k in ("stage", "files")}) != r3_digest):
        raise ValueError("R3 comparison seal identity changed")

    def checked(path):
        expected = obs["files"][str(path.relative_to(r3))]
        if digest(path) != expected:
            raise ValueError(f"R3 comparison file changed: {path}")
        return path

    r3_plan = read_json(checked(r3 / "protocol/r3-plan.json"))
    records = []
    for task in tasks(plan):
        if [task.source, task.target] not in plan["historical_directions"]:
            continue
        job = r3_job(plan, task, METHOD)
        source = source_dir(base, base_plan, task.source, task.seed)
        old_source = r3_source_dir(r3, task.seed)
        old_roles = read_json(checked(r3 / "roles" / f"seed{task.seed}.json"))
        new_roles = role(base, task.source, task.seed)
        for split in ("fit", "validation"):
            assert_nested_equal(old_roles[split], new_roles[split])
        assert_nested_equal(read_json(checked(r3 / "inputs/domains.json"))[str(task.target)],
                            domain(base, task.target))
        old_frozen = load(checked(old_source / "frozen.pt"))
        new_frozen = load(source / "frozen.pt")
        for key in ("x", "initial_head", "mean", "scale"):
            assert_nested_equal(old_frozen[key], new_frozen[key])
        assert_nested_equal(load(checked(r3 / "source_labels" / f"seed{task.seed}_fit.pt")),
                            load(labels_path(base, task.source, task.seed, "fit")))
        assert_nested_equal(load(checked(r3 / "batches" / f"seed{task.seed}_d{task.target}.pt")),
                            load(batch_path(base, task)))
        assert_nested_equal(load(checked(old_source / f"geometry_d{task.target}.pt"))["sigma"],
                            load(source / f"geometry_M{task.target}.pt")["sigma"])
        epoch = f"epoch{config(plan, task.seed).score_epochs:04d}.pt"
        old_score = load(checked(r3_score_dir(r3, task.seed, task.target, job.bandwidth) / epoch))
        new_score = load(score_dir(base, base_plan, task) / epoch)
        for key in ("updates", "model_state"):
            assert_nested_equal(old_score[key], new_score[key])
        old_cfg = asdict(job.config(r3_plan))
        new_cfg = asdict(config(plan, task.seed))
        # R3 reconstructs the width from the frozen head and loads the fixed score checkpoint.
        differences = {key: [old_cfg[key], value] for key, value in new_cfg.items() if old_cfg[key] != value}
        if differences.keys() - {"source_epochs", "score_epochs", "adapter_width"}:
            raise ValueError(f"R3/R4 adaptation config differs: {differences}")
        for key in ("solver", "constraint_tolerance", "side_alpha_start", "side_attempts"):
            assert_nested_equal(r3_plan[key], plan[key])

        out = job_dir(root, plan, task, METHOD)
        historical = r3 / "trajectories" / job.key
        stored_predictions = 0
        for step in range(plan["model"]["adapt_rounds"] + 1):
            name = f"round{step:02d}.pt"
            old = load(checked(historical / name))
            new = load(out / name)
            assert_nested_equal(old["after"], new["after"])
            if step in r3_plan["endpoints"]:
                assert_nested_equal(old["predictions"], new["predictions"])
                stored_predictions += 1
            if step:
                assert_nested_equal(old["status"], new["status"])
        records.append(dict(**asdict(task), r3_job=job.key, states_exact=31, attempts_exact=30,
                            saved_predictions_exact=stored_predictions,
                            config_fields_superseded_by_frozen_artifacts=differences))
    if len(records) != 10:
        raise ValueError("R3 compatibility audit requires both historical directions")
    return records


def replay(root, base, plan, base_plan):
    counts, sides = Counter(), Counter()
    for task in tasks(plan):
        model, pools = inputs(base, base_plan, task)
        params = ParameterMap(model)
        initial = params.vector()
        baseline = load(base / "baselines" / f"{task.key}.pt")
        assert_nested_equal(predictions(params, initial, pools), baseline)
        out = job_dir(root, plan, task, METHOD)
        complete = dict(**asdict(task), method=METHOD, attempts=plan["model"]["adapt_rounds"])
        assert_nested_equal(read_json(out / "complete.json"), complete)
        schedule = load(batch_path(base, task))
        replay_trajectory(out, plan, params, pools, initial, schedule, counts, sides)
    if (any(counts[k] != v for k, v in COUNTS.items())
            or sum(sides.values()) != COUNTS["trajectories"] * len(plan["side_attempts"])
            or counts["side_predictions_replayed"] != 3 * sides["available"]):
        raise ValueError(f"A0 replay coverage mismatch: {counts}, {sides}")
    return dict(counts, side_statuses=dict(sides))


def rows_from_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numeric_endpoints(rows):
    integers = ("attempt", "source", "target", "seed", "n", "correct")
    return [{**row, **{k: int(row[k]) for k in integers},
             **{k: float(row[k]) for k in METRICS}} for row in rows if row["pool"] == "target"]


def numeric_rounds(rows):
    integers = ("attempt", "net_correct", "anchors", "confidence_rejections", "risk_rejections")
    floats = ("accuracy", "mean_ess", "mean_origin_weight", "mean_maximum_weight")
    return [{**row, **{k: int(row[k] or 0) for k in integers},
             **{k: float(row[k]) if row[k] else None for k in floats},
             "accepted": row["accepted"] == "True"} for row in rows]


def evaluate(root, base, plan, base_plan, historical):
    obs = verify(root, "A0_OBS")
    replay_info = read_json(root / "replay.json")
    required = {"protocol/registration.json", "replay.json", "historical-parity.json"}
    for task in tasks(plan):
        path = job_dir(root, plan, task, METHOD).relative_to(root)
        required.add(str(path / "complete.json"))
        required.update(str(path / f"round{step:02d}.pt") for step in range(plan["model"]["adapt_rounds"] + 1))
    if not required <= obs["files"].keys() or any(replay_info[k] != v for k, v in COUNTS.items()):
        raise PermissionError("A0 evaluation requires complete sealed predictions")
    # Label access starts after both observation seals have been verified.
    base_obs, labels_by_domain = evaluation_labels(base)
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    verified_json(out / "label_access.json", dict(a0_observation=obs["digest"], r4_observation=base_obs,
                                                target_domains=plan["domains"]))
    base_dir = base / "evaluation"
    old_endpoints = rows_from_csv(base_dir / "endpoints.csv")
    old_rounds = rows_from_csv(base_dir / "rounds.csv")
    old_side = rows_from_csv(base_dir / "side.csv")
    old_pairs = rows_from_csv(base_dir / "paired.csv")
    endpoints, rounds, sides = [], [], []
    for task in tasks(plan):
        labels = dict(target=labels_by_domain[task.target], **{
            f"source_{split}": load(labels_path(base, task.source, task.seed, split))
            for split in ("fit", "validation")
        })
        task_row = dict(**asdict(task), method=METHOD,
                        historical=[task.source, task.target] in plan["historical_directions"])
        baseline = load(base / "baselines" / f"{task.key}.pt")
        e, r, d = trajectory_rows(job_dir(root, plan, task, METHOD), plan, task_row, labels, baseline)
        endpoints.extend(e)
        rounds.extend(r)
        sides.extend(d)
    pairs, summary = summarize(plan, numeric_endpoints(old_endpoints) + endpoints)
    old_summary = read_json(base_dir / "summary.json")
    old_identities = {row["method_or_contrast"] for row in old_summary}
    assert_nested_equal([row for row in summary if row["method_or_contrast"] in old_identities], old_summary)
    new_pairs = [row for row in pairs if row["contrast"] not in old_identities]
    for name, rows in (("endpoints", old_endpoints + endpoints), ("rounds", old_rounds + rounds),
                       ("side", old_side + sides), ("paired", old_pairs + new_pairs), ("summary", summary)):
        write_csv(out / f"{name}.csv", rows)
    verified_json(out / "summary.json", summary)
    report(base, plan, out, summary, numeric_rounds(old_rounds) + rounds,
           supplement=dict(replay=replay_info, historical=historical,
                           date=read_json(root / "protocol/registration.json")["date"]))
    seal(root, "A0_FINAL", ["seals/A0_OBS.json", "evaluation"])


def execute(root, base, r3, repo):
    docs = repo / "experiments/mutagenicity"
    index = read_json(docs / "r4-20261112/archive-index.json")
    r3_digest = read_json(docs / "r3-20261112/archive-index.json")["observation_digest"]
    obs = verify(base, "R4_OBS")
    if obs["digest"] != index["audit"]["seals"]["OBS"]:
        raise ValueError("Reference differs from published R4")
    base_plan = read_json(base / "protocol/r4-plan.json")
    plan = supplement_plan(base_plan)
    registered = tasks(plan)
    cfg = plan["model"]
    if (len(registered) != 60 or cfg["adapt_rounds"] != 30 or plan["weight"] != 1
            or plan["score"]["endpoints"][-1] != 7200
            or plan["seeds"] != list(range(3405, 3410))):
        raise ValueError("R4 settings differ from the A0 protocol")
    job = r3_job(plan, registered[0], METHOD)
    if (job.group, job.density, job.direction, job.bandwidth, job.protect, job.epsilon) != (
            "M00", "learned", 1, 1, False, None):
        raise ValueError("A0 must be the R3 M00 normal request")
    refs = references(base, base_plan, obs)
    if not root.exists():
        register(root, base, r3, repo, plan, refs, r3_digest)
    verify(root, "A0_INPUT")
    protocol = root / "protocol"
    registration = read_json(protocol / "registration.json")
    if (read_json(protocol / "plan.json") != plan or read_json(protocol / "references.json") != refs
            or registration["r3_root"] != str(r3) or registration["r3_observation"] != r3_digest):
        raise ValueError("A0 dependencies differ from the sealed registration")
    for path, expected in read_json(protocol / "provenance.json")["file_hashes"].items():
        if path.endswith(".py") and digest(repo / path) != expected:
            raise ValueError(f"Registered code changed: {path}")
    if not (root / "seals/A0_OBS.json").exists():
        print(f"{stamp()} A0 uses sealed R4 observation {obs['digest']}", flush=True)
        run_stage(root, plan, "adapt", partial(adapt, reference=(base, base_plan)),
                  [(task, METHOD) for task in registered])
        replay_info = replay(root, base, plan, base_plan)
        verified_json(root / "replay.json", replay_info)
        historical = historical_parity(root, base, r3, plan, base_plan, r3_digest)
        verified_json(root / "historical-parity.json", historical)
        seal(root, "A0_OBS", ["seals/A0_INPUT.json", "protocol", "trajectories", "stages",
                             "replay.json", "historical-parity.json"])
        print(f"{stamp()} A0 observation sealed {replay_info}", flush=True)
    if not (root / "seals/A0_FINAL.json").exists():
        base_final = verify(base, "R4_FINAL")
        if base_final["digest"] != index["audit"]["seals"]["FINAL"]:
            raise ValueError("Historical R4 evaluation differs from the published seal")
        evaluate(root, base, plan, base_plan, read_json(root / "historical-parity.json"))
    verify(root, "A0_FINAL")
    print(f"{stamp()} A0 supplement complete at {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--r4", type=Path, required=True)
    parser.add_argument("--r3", type=Path, required=True)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(var) != "1":
            raise ValueError(f"Set {var}=1 for deterministic per-process execution")
    configure()
    root, base, r3 = (path.resolve() for path in (args.root, args.r4, args.r3))
    if root.is_relative_to(base) or root.is_relative_to(r3):
        raise ValueError("Supplement outputs must stay outside the historical experiments")
    root.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[5]
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(root, base, r3, repo)


if __name__ == "__main__":
    main()
