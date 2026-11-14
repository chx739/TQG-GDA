"""Post-observation metrics for each independent R5 setting."""
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
import math
import statistics
import torch
from .shared import io, output, p3, p4, e4, assert_nested_equal
from experiments.mutagenicity.r3_源风险与置信度约束_20261112.src.evaluate import metrics


def target_labels(root, plan):
    seal = io.verify(root, "R5_OBS")
    replay = p3.read_json(root / "replay.json")
    required = {"protocol/plan.json", "replay.json", "inputs/label_records.json",
                "seals/R5_INPUT.json", "inputs/domains.json", "inputs/graphs.pt"}
    io.verify(root, "R5_INPUT")
    for source in plan["domains"]:
        for seed in plan["seeds"]:
            directory = p4.source_dir(root, plan, source, seed).relative_to(root)
            required.update(str(directory / name) for name in ("source.pt", "frozen.pt", "complete.json"))
    for task in p4.tasks(plan):
        required.add(f"baselines/{task.key}.pt")
        required.add(str(p4.batch_path(root, task).relative_to(root)))
        directory = p4.score_dir(root, plan, task).relative_to(root)
        steps = (len(p4.domain(root, task.target)) + plan["model"]["score_batch"] - 1) // plan["model"]["score_batch"]
        required.update(str(directory / f"epoch{updates // steps:04d}.pt") for updates in plan["score"]["endpoints"])
        required.update(str(directory / name) for name in ("complete.json", "diagnostics.pt"))
        for method in ("B1", "A0"):
            directory = p4.job_dir(root, plan, task, method).relative_to(root)
            required.add(str(directory / "complete.json"))
            required.update(str(directory / f"round{s:02d}.pt") for s in range(31))
        required.update(str((root / "diagnostics" / task.key / f"side{s:02d}.pt").relative_to(root))
                        for s in plan["side_attempts"])
    if not required <= seal["files"].keys() or any(replay[k] != v for k, v in plan["counts"].items()):
        raise PermissionError("R5 observation is incomplete")
    if (replay["predictions_replayed"] != 3720 or sum(replay["side_statuses"].values()) != 120
            or replay["score_endpoints"] != 240
            or replay["side_predictions_replayed"] != 3 * replay["side_statuses"].get("available", 0)):
        raise PermissionError("R5 observation omits required replay")
    meta = p3.read_json(root / "inputs/label_records.json")
    result = {}
    for number in plan["domains"]:
        values = p3.read_values(meta["path"], meta["offsets"], p4.domain(root, number), meta["sha256"])
        result[number] = torch.tensor([meta["mapping"][str(y)] for y in values])
    return seal["digest"], result



def subset_metrics(predictions, labels):
    if not len(labels):
        return dict(n=0)
    result = metrics(predictions, labels)
    cm = result["confusion"]
    denominators = [sum(cm[c]) + sum(r[c] for r in cm) for c in (0, 1)]
    result["macro_f1"] = sum(cm[c][c] / denominators[c] if denominators[c] else 0. for c in (0, 1))
    result["correct"] = cm[0][0] + cm[1][1]
    return result


def side_rows(root, plan, task, labels, identity):
    rows = []
    path = p4.job_dir(root, plan, task, "A0")
    for attempt in plan["side_attempts"]:
        record = io.load(path / f"round{attempt:02d}.pt")
        before = io.load(path / f"round{attempt-1:02d}.pt")["predictions"]["target"]
        side = record["side"]
        detail = io.load(root / "diagnostics" / task.key / f"side{attempt:02d}.pt")
        row = dict(**identity, method="A0", attempt=attempt, status=side["status"])
        if side["status"] != "available":
            rows.append(row)
            continue
        origin_mask = torch.zeros(len(labels), dtype=torch.bool)
        origin_mask[detail["origins"]] = True
        correct = before.argmax(1) == labels
        pools = dict(target=torch.ones_like(origin_mask), generators=origin_mask,
                     complement=~origin_mask, initially_correct=correct, initially_wrong=~correct)
        for pool, mask in pools.items():
            for direction in (-1, 0, 1):
                candidate = side["candidates"][direction]
                extra = detail["candidates"][direction]
                predictions = candidate["predictions"]["target"]
                measured = subset_metrics(predictions[mask], labels[mask])
                measured.update(e4.correction(predictions[mask], before[mask], labels[mask]))
                for other in (0, -1):
                    corrections = e4.correction(predictions[mask], side["candidates"][other]["predictions"]["target"][mask], labels[mask])
                    measured.update({f"vs_{other}_{key}": value for key, value in corrections.items()})
                geometric = extra["normalized_margin"]
                reference = detail["baseline"]
                valid = mask & geometric["valid"] & reference["valid"]
                rows.append(dict(**row, pool=pool, direction=direction, alpha=side["alpha"], **measured,
                    full_target_n=len(labels), request_rms_sigma=detail["request_rms_sigma"],
                    increment_motion_rms_sigma=extra["increment_motion_rms_sigma"],
                    total_motion_rms_sigma=extra["total_motion_rms_sigma"],
                    actual_vs_zero_rms_sigma=extra["actual_vs_zero_rms_sigma"],
                    increment_residual=extra["increment_residual"],
                    normalized_margin_valid=int(valid.sum()),
                    normalized_margin_change_rms=float((geometric["value"][valid] -
                        reference["value"][valid]).square().mean().sqrt()) if valid.any() else None,
                    relocation_statuses=dict(Counter(r["status"] for r in extra["relocation"]["intersections"])),
                    single_root_count=extra["single_root_count"], root_rmse_vs_proxy_sigma=extra["root_rmse_vs_proxy_sigma"],
                    relocation_roots=len(extra["relocation"]["intersections"])))
    return rows


def side_pairs(rows):
    indexed = {(r["source"], r["target"], r["seed"], r["attempt"], r["pool"], r["direction"]): r
               for r in rows if r["status"] == "available"}
    result = []
    for key, normal in indexed.items():
        if key[-1] != 1:
            continue
        for direction in (0, -1):
            other = indexed[(*key[:-1], direction)]
            row = {k: normal[k] for k in ("dataset", "split", "source", "target", "seed", "attempt", "pool", "origin")}
            row.update(contrast=f"normal_minus_{direction}", n=normal["n"], full_target_n=normal["full_target_n"])
            if normal["n"]:
                for metric in ("accuracy", "balanced_accuracy", "macro_f1", "nll"):
                    row[metric] = normal[metric] - other[metric] if normal.get(metric) is not None and other.get(metric) is not None else None
                row["net_correct"] = normal["correct"] - other["correct"]
                row["accuracy"] = row["net_correct"] / row["n"]
                for correction_name in ("wrong_to_right", "right_to_wrong"):
                    row[correction_name] = normal[f"vs_{direction}_{correction_name}"]
                row["nll_contribution"] = row["nll"] * row["n"] / row["full_target_n"]
                for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
                    row[metric_name + "_pp"] = 100 * row[metric_name] if row[metric_name] is not None else None
            result.append(row)
    return result


def aggregate_pairs(plan, pairs, *, side=False):
    """Keep all planned seeds. Full means are NA when a constituent pair is absent."""
    scopes = [("Avg", plan["directions"])] + [(f"M{s}→M{t}", [[s, t]]) for s, t in plan["directions"]]
    metrics = ("accuracy", "balanced_accuracy", "macro_f1", "nll")
    if side:
        metrics += ("net_correct", "wrong_to_right", "right_to_wrong", "nll_contribution")
    groups = [(c, attempt, pool) for c in ("normal_minus_0", "normal_minus_-1")
              for attempt in plan["side_attempts"]
              for pool in ("target", "generators", "complement", "initially_correct", "initially_wrong")] if side else [
                  (f"{a}−{b}", 30, "target") for a, b in plan["pairs"]]
    seed_rows, summaries = [], []
    for contrast, attempt, pool in groups:
        selected = [r for r in pairs if r["contrast"] == contrast
                    and (not side or (r["attempt"], r["pool"]) == (attempt, pool))]
        for scope, directions in scopes:
            scope_rows = []
            for seed in plan["seeds"]:
                found = [r for r in selected if r["seed"] == seed and [r["source"], r["target"]] in directions and r["n"]]
                record = dict(contrast=contrast, attempt=attempt, pool=pool, scope=scope, seed=seed,
                              planned_pairs=len(directions), available_pairs=len(found))
                for metric in metrics:
                    values = [Fraction(r["net_correct" if side else "correct_difference"], r["n"])
                              if metric == "accuracy" else r.get(metric) for r in found]
                    valid = [v for v in values if v is not None]
                    complete = len(valid) == len(directions)
                    mean = sum(valid) / len(valid) if valid else None
                    record[metric] = float(mean) if complete else None
                    record[f"{metric}_available_mean"] = float(mean) if mean is not None else None
                    record[f"{metric}_available_pairs"] = len(valid)
                    record[f"{metric}_sign"] = ((mean > 0) - (mean < 0)) if complete else None
                for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                    record[metric + "_pp"] = 100*record[metric] if record[metric] is not None else None
                scope_rows.append(record)
            seed_rows.extend(scope_rows)
            for metric in metrics:
                values = [r[metric] for r in scope_rows if r[metric] is not None]
                signs = [r[f"{metric}_sign"] * (-1 if metric in ("nll", "nll_contribution", "right_to_wrong") else 1)
                         for r in scope_rows if r[metric] is not None]
                complete = len(values) == len(plan["seeds"])
                summaries.append(dict(contrast=contrast, attempt=attempt, pool=pool, scope=scope, metric=metric,
                    available_pairs=sum(r[f"{metric}_available_pairs"] for r in scope_rows),
                    planned_pairs=len(directions)*len(plan["seeds"]), complete_seeds=len(values),
                    mean=statistics.mean(values) if complete else None,
                    sd=statistics.stdev(values) if complete and len(values)>1 else None,
                    wins=signs.count(1), ties=signs.count(0), losses=signs.count(-1), missing_seeds=len(plan["seeds"])-len(values)))
    return seed_rows, summaries


def evaluate(root, plan):
    seal, target = target_labels(root, plan)
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    output.write_json(out / "label_access.json", dict(observation=seal, domains=plan["domains"]))
    endpoints, rounds, sides, scales, source_curves, score_rows = [], [], [], [], [], []
    common = dict(dataset=plan["dataset"], split=plan["setting"]["id"], origin=plan["setting"]["action"])
    for source in plan["domains"]:
        for seed in plan["seeds"]:
            state = io.load(p4.source_dir(root, plan, source, seed) / "source.pt")
            source_curves.extend(dict(**common, source=source, seed=seed, selected=r["epoch"] == state["selected_epoch"], **r)
                                 for r in state["history"])
    for task in p4.tasks(plan):
        identity = dict(**common, **asdict(task), historical=plan["setting"]["action"] == "reuse")
        labels = dict(target=target[task.target], **{f"source_{s}": io.load(p4.labels_path(root, task.source, task.seed, s))
                                                    for s in ("fit", "validation")})
        baseline = io.load(root / "baselines" / f"{task.key}.pt")
        score_rows.extend(dict(**identity, updates=r["step"], dsm=r["validation"], **r["all"])
                          for r in io.load(p4.score_dir(root, plan, task) / "diagnostics.pt"))
        for pool, prediction in baseline.items():
            for step in (0, 30):
                endpoints.append(dict(**identity, method="B0", attempt=step, pool=pool,
                                      **e4.classify(prediction, labels[pool])))
            for scale in plan["evaluation"]["fixed_B0_logit_scales"]:
                logp = (scale * prediction).log_softmax(1)
                assert_nested_equal(logp.argmax(1), prediction.argmax(1))
                scales.append(dict(**identity, method=f"B0_scale_{scale}", attempt=0, pool=pool,
                                   **e4.classify(logp, labels[pool])))
        for method in ("B1", "A0"):
            local = dict(plan, side_attempts=plan["side_attempts"] if method == "A0" else [])
            e, r, _ = e4.trajectory_rows(p4.job_dir(root, plan, task, method), local,
                                       dict(**identity, method=method), labels, baseline)
            endpoints.extend(e)
            rounds.extend(r)
        sides.extend(side_rows(root, plan, task, target[task.target], identity))
    pairs, summary = e4.summarize(plan, endpoints)
    summary = [row for row in summary if row["scope"] != "New coverage"]
    for row in summary:
        row.update(common)
        for metric in e4.METRICS:
            info = row[metric]
            radius = plan["evaluation"]["t_interval"]["critical_value"] * info["sd"] / math.sqrt(len(plan["seeds"]))
            info["ci95_seed_low"], info["ci95_seed_high"] = info["mean"] - radius, info["mean"] + radius
    for row in pairs:
        row.update(common)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            row[metric + "_pp"] = 100 * row[metric]
    paired_sides = side_pairs(sides)
    seed_rows, _ = aggregate_pairs(plan, pairs)
    side_seeds, side_summary = aggregate_pairs(plan, paired_sides, side=True)
    for row in (*seed_rows, *side_seeds, *side_summary):
        row.update(common)
    diagnostics = []
    for method in ("B1", "A0"):
        values = [r for r in rounds if r["method"] == method and r["attempt"]]
        diagnostics.append(dict(**common, method=method, statuses=dict(Counter(r["status"] for r in values)),
                                attempts=len(values), anchors=sum(r.get("anchors", 0) for r in values)))
    for name, values in (("endpoints", endpoints), ("rounds", rounds), ("side", sides),
                         ("side_paired", paired_sides), ("paired", pairs), ("summary", summary),
                         ("paired_seeds", seed_rows), ("side_seeds", side_seeds), ("side_summary", side_summary),
                         ("calibration", scales), ("diagnostics", diagnostics), ("source_curves", source_curves),
                         ("score_endpoints", score_rows)):
        output.write_csv(out / f"{name}.csv", values)
    output.write_json(out / "summary.json", summary)
    from .report import report
    report(out, plan, summary, rounds, sides)
    io.seal(root, "R5_FINAL", ["seals/R5_OBS.json", "evaluation"])
