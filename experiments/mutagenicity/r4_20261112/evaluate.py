"""Replay all saved predictions before target scoring, then report paired effects."""
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
import math

import numpy as np
import torch

from tsba.classifier import Coordinates, ParameterMap
from experiments.common.real_data.r2.common import atomic_tensor, load, seal
from experiments.common.real_data.r2.evaluate import write_csv, write_json
from experiments.common.real_data.r2.score import assert_nested_equal
from experiments.common.real_data.r3.trajectory import predictions
from experiments.common.real_data.r3.evaluate import margin_summary, metrics, round_summary
from experiments.mutagenicity.r4_20261112.protocol import batch_path, config, domain, evaluation_labels, job_dir, labels_path, role, score_dir, source_dir, tasks
from experiments.mutagenicity.r4_20261112.trajectory import inputs


def verify_source(root, plan, source, seed):
    directory = source_dir(root, plan, source, seed)
    complete = load(directory / "source.pt")
    cfg = config(plan, seed)
    history = complete["history"]
    if len(history) != cfg.source_epochs:
        raise ValueError("Incomplete source training")
    best = min(history, key=lambda r: r["source_validation"]["ce"])["epoch"]
    if best != complete["selected_epoch"]:
        raise ValueError("Source selection differs from earliest minimum validation CE")
    files = sorted((directory / "epochs").glob("epoch*.pt"))
    if [p.name for p in files] != [f"epoch{i:03d}.pt" for i in range(cfg.source_epochs)]:
        raise ValueError("Missing source epoch checkpoints")
    selected = load(files[best])
    assert_nested_equal(complete["state"], selected["state"])
    frozen = load(directory / "frozen.pt")
    coords = Coordinates.fit(frozen["h"][role(root, source, seed)["fit"]], cfg.coordinate_eps)
    for actual, expected in ((coords.mean, frozen["mean"]), (coords.scale, frozen["scale"]),
                             (coords(frozen["h"]), frozen["x"])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def replay(root, plan):
    summaries = Counter()
    side_statuses = Counter()
    for source in plan["domains"]:
        for seed in plan["seeds"]:
            verify_source(root, plan, source, seed)
            summaries["sources"] += 1
    for task in tasks(plan):
        cfg = config(plan, task.seed)
        score_path = score_dir(root, plan, task)
        for updates in plan["score"]["endpoints"]:
            epoch = updates // math.ceil(len(domain(root, task.target)) / cfg.score_batch)
            checkpoint = load(score_path / f"epoch{epoch:04d}.pt")
            if checkpoint["updates"] != updates or not {
                "optimizer_state", "training_rng_state", "torch_rng_state", "validation_rng_state"
            } <= checkpoint.keys():
                raise ValueError("Incomplete score checkpoint")
            summaries["score_endpoints"] += 1
        summaries["scores"] += 1
        model, pools = inputs(root, plan, task)
        params = ParameterMap(model)
        initial = params.vector()
        baseline = predictions(params, initial, pools)
        destination = root / "baselines" / (task.key + ".pt")
        if destination.exists():
            assert_nested_equal(load(destination), baseline)
        else:
            atomic_tensor(destination, baseline)
        schedule = load(batch_path(root, task))
        for method in plan["methods"]:
            if method == "B0":
                continue
            path = job_dir(root, plan, task, method)
            replay_trajectory(path, plan, params, pools, initial, schedule, summaries, side_statuses)
    if any(summaries[key] != plan["counts"][key] for key in plan["counts"]):
        raise ValueError(f"R4 count mismatch: {summaries}")
    return dict(summaries, side_statuses=dict(side_statuses))


def replay_trajectory(path, plan, params, pools, initial, schedule, summaries, side_statuses):
    if not (path / "complete.json").is_file():
        raise ValueError(f"Incomplete trajectory {path}")
    records = sorted(path.glob("round[0-9][0-9].pt"))
    if [p.name for p in records] != [f"round{i:02d}.pt" for i in range(plan['model']['adapt_rounds'] + 1)]:
        raise ValueError("Round coverage differs from registration")
    previous = initial
    for step, record_path in enumerate(records):
        r = load(record_path)
        if step:
            if r["attempt"] != step:
                raise ValueError("Attempt identity mismatch")
            assert_nested_equal(r["batch"], schedule[step - 1])
            torch.testing.assert_close(r["before"], previous, rtol=0, atol=0)
            if not r["accepted"]:
                torch.testing.assert_close(r["after"], previous, rtol=0, atol=0)
            summaries["attempts"] += 1
        else:
            torch.testing.assert_close(r["after"], initial, rtol=0, atol=0)
        assert_nested_equal(predictions(params, r["after"], pools), r["predictions"])
        previous = r["after"]
        summaries["predictions_replayed"] += 1
        if step in plan["side_attempts"]:
            side = r["side"]
            side_statuses[side["status"]] += 1
            if side["status"] == "available":
                if set(side["candidates"]) != {-1, 0, 1}:
                    raise ValueError("Incomplete available side comparison")
                for candidate in side["candidates"].values():
                    assert_nested_equal(predictions(params, candidate["vector"], pools),
                                        candidate["predictions"])
                    summaries["side_predictions_replayed"] += 1
    summaries["trajectories"] += 1


def classify(logp, labels):
    row = metrics(logp, labels)
    cm = row["confusion"]
    denom = [sum(cm[c]) + sum(line[c] for line in cm) for c in (0, 1)]
    row["macro_f1"] = sum(cm[c][c] / denom[c] for c in (0, 1))
    row["correct"] = sum(cm[c][c] for c in (0, 1))
    return row


def correction(logp, baseline, labels):
    previous = baseline.argmax(1) == labels
    current = logp.argmax(1) == labels
    fixed = ~previous & current
    lost = previous & ~current
    result = {}
    for name, mask in (("", torch.ones_like(current)), ("_class0", labels == 0), ("_class1", labels == 1)):
        gains, losses = int((fixed & mask).sum()), int((lost & mask).sum())
        result.update({"wrong_to_right" + name: gains, "right_to_wrong" + name: losses,
                       "net_correct" + name: gains - losses})
    return result


def statistic(values, paired, lower_better=False):
    result = dict(mean=float(sum(values) / len(values)),
                  sd=float(np.std([float(v) for v in values], ddof=1)))
    if paired:
        signed = [-v for v in values] if lower_better else values
        result.update(wins=sum(v > 0 for v in signed), ties=sum(v == 0 for v in signed),
                      losses=sum(v < 0 for v in signed))
    return result


METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "nll", "recall0", "recall1")


def summarize(plan, rows):
    """Average directions inside each seed, then measure seed variation."""
    buckets = defaultdict(dict)
    endpoint = {}
    for row in rows:
        if row["attempt"] != plan["model"]["adapt_rounds"] or row["pool"] != "target":
            continue
        s, t, seed, method = (row[k] for k in ("source", "target", "seed", "method"))
        endpoint[s, t, seed, method] = row
    paired_rows = []
    for task in tasks(plan):
        for left, right in plan["pairs"]:
            before = endpoint[task.source, task.target, task.seed, right]
            after = endpoint[task.source, task.target, task.seed, left]
            if after["n"] != before["n"]:
                raise ValueError("Paired predictions cover different targets")
            delta = {key: after[key] - before[key] for key in METRICS}
            # Use integer correct-count differences for exact per-task ties.
            correct_difference = after["correct"] - before["correct"]
            delta["accuracy"] = correct_difference / after["n"]
            paired_rows.append(dict(asdict(task), contrast=f"{left}−{right}", **delta,
                                    correct_difference=correct_difference, n=after["n"]))

    for row in (*endpoint.values(), *paired_rows):
        s, t, seed = row["source"], row["target"], row["seed"]
        identifier = row["method"] if "method" in row else row["contrast"]
        scopes = [f"M{s}→M{t}", f"source M{s}", f"target M{t}", "Avg"]
        if [s, t] not in plan["historical_directions"]:
            scopes.append("New coverage")
        for scope in scopes:
            buckets[identifier, scope].setdefault(seed, []).append(row)
    summary = []
    for (identifier, scope), by_seed in sorted(buckets.items()):
        if sorted(by_seed) != sorted(plan["seeds"]):
            raise ValueError("Missing seed for summary")
        values = {}
        for key in METRICS:
            if key == "accuracy":
                counts = "correct" if "−" not in identifier else "correct_difference"
                per_seed = [sum(Fraction(r[counts], r["n"]) for r in rows) / len(rows)
                            for rows in (by_seed[seed] for seed in plan["seeds"])]
            else:
                per_seed = [math.fsum(r[key] for r in by_seed[seed]) / len(by_seed[seed])
                            for seed in plan["seeds"]]
            values[key] = statistic(per_seed, paired="−" in identifier, lower_better=key == "nll")
        summary.append(dict(method_or_contrast=identifier, scope=scope, **values))
    return paired_rows, summary



def trajectory_rows(path, plan, row, labels, baseline):
    endpoints, rounds, sides = [], [], []
    previous = baseline["target"]
    for step in range(plan["model"]["adapt_rounds"] + 1):
        record = load(path / f"round{step:02d}.pt")
        saved = record["predictions"]
        result = {pool: classify(value, labels[pool]) for pool, value in saved.items()}
        if step in plan["endpoints"]:
            endpoints.extend(dict(**row, attempt=step, pool=pool, **value)
                             for pool, value in result.items())
        if step:
            detail = round_summary(record)
            detail["numerical_error"] = record.get("error")
            if "bounds" in record:
                detail["anchors"] = len(record["bounds"]["points"])
                detail["projection_rejections"] = record["bounds"]["rejected"]
            if "relocation" in record:
                displacement = [r["displacement"] for r in record["relocation"]["intersections"]
                                if r["status"] == "single_root"]
                detail["single_root_count"] = len(displacement)
                detail["normal_displacement_rms"] = (
                    math.sqrt(math.fsum(v*v for v in displacement) / len(displacement))
                    if displacement else None
                )
        else:
            detail = dict(attempt=0, status="initial", accepted=False)
        result_row = dict(**row, **detail, **result["target"], **margin_summary(saved["target"]))
        result_row.update({f"{pool}_{key}": result[pool][key]
                           for pool in ("source_fit", "source_validation")
                           for key in ("nll", "accuracy", "class0_ce", "class1_ce")})
        result_row.update(correction(saved["target"], baseline["target"], labels["target"]))
        result_row.update({"previous_" + key: value for key, value in
                           correction(saved["target"], previous, labels["target"]).items()})
        if result_row["net_correct"] != result["target"]["correct"] - int(
            (baseline["target"].argmax(1) == labels["target"]).sum()
        ):
            raise ValueError("Correction counts do not match the accuracy change")
        rounds.append(result_row)
        previous = saved["target"]
        if step in plan["side_attempts"]:
            side = record["side"]
            for direction in (-1, 0, 1):
                item = dict(**row, attempt=step, side_direction=direction,
                            status=side["status"], alpha=side.get("alpha"))
                if side["status"] == "available":
                    item.update(classify(side["candidates"][direction]["predictions"]["target"],
                                         labels["target"]))
                sides.append(item)
    return endpoints, rounds, sides

def evaluate(root, plan):
    digest, target_labels = evaluation_labels(root)
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    write_json(out / "label_access.json", dict(observation=digest, target_domains=plan["domains"],
               sizes={str(d): len(y) for d, y in target_labels.items()}))
    write_json(out / "target_labels.json", {str(d): y.tolist() for d, y in target_labels.items()})
    endpoints, rounds, sides, score_rows, source_rows = [], [], [], [], []
    for s in plan["domains"]:
        for seed in plan["seeds"]:
            summary = load(source_dir(root, plan, s, seed) / "source.pt")
            source_rows.extend(dict(source=s, seed=seed, selected=summary["selected_epoch"] == r["epoch"],
                                    **r) for r in summary["history"])
    for task in tasks(plan):
        task_row = dict(asdict(task), historical=[task.source, task.target] in plan["historical_directions"])
        labels = dict(target=target_labels[task.target],
                      source_fit=load(labels_path(root, task.source, task.seed, "fit")),
                      source_validation=load(labels_path(root, task.source, task.seed, "validation")))
        baseline = load(root / "baselines" / (task.key + ".pt"))
        score_rows.extend(dict(**task_row, updates=r["step"], dsm=r["validation"], **r["all"])
                          for r in load(score_dir(root, plan, task) / "diagnostics.pt"))
        for method in plan["methods"]:
            row = dict(**task_row, method=method)
            if method == "B0":
                for pool, value in baseline.items():
                    endpoints.append(dict(**row, attempt=0, pool=pool, **classify(value, labels[pool])))
                    endpoints.append(dict(**row, attempt=plan["model"]["adapt_rounds"], pool=pool,
                                          **classify(value, labels[pool])))
                continue
            path = job_dir(root, plan, task, method)
            e, r, d = trajectory_rows(path, plan, row, labels, baseline)
            endpoints.extend(e)
            rounds.extend(r)
            sides.extend(d)
    pairs, summary = summarize(plan, endpoints)
    for name, data in (("endpoints", endpoints), ("paired", pairs), ("rounds", rounds), ("side", sides),
                       ("score_endpoints", score_rows), ("source_curves", source_rows), ("summary", summary)):
        write_csv(out / (name + ".csv"), data)
    write_json(out / "summary.json", summary)
    from experiments.mutagenicity.r4_20261112.report import report
    report(root, plan, out, summary, rounds)
    seal(root, "R4_FINAL", ["seals/R4_OBS.json", "evaluation"])
