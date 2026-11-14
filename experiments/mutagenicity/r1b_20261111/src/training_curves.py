"""Publish pre-registered warm training curves from sealed saved histories."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.mutagenicity.r1b_20261111.src.common import verify_seal


SEEDS = (3405, 3406, 3407, 3408, 3409)
DOMAINS = (1, 3)
RATES = (.001, .0001)
STEPS = tuple(range(0, 801, 20))
OUTPUTS = ("warm-training-curves.json", "warm-training-curves.png",
           "warm-training-curves-metadata.json")


def _sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values, expected):
    available = [float(value) for value in values if _finite(value)]
    return statistics.mean(available) if len(available) == expected else None


def _summary(values):
    available = [float(value) for value in values if _finite(value)]
    return dict(mean=_mean(values, len(SEEDS)),
                sample_sd=statistics.stdev(available) if len(available) == len(SEEDS) else None,
                available_count=len(available), expected_count=len(SEEDS))


def _read(root, relative, obs, inputs):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise PermissionError("Warm history path escapes archive")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if obs["files"].get(relative) != digest:
        raise PermissionError(f"Warm history or metadata is unsealed or changed: {relative}")
    inputs[relative] = dict(sha256=digest, bytes=len(raw))
    return json.loads(raw)


def _history(rows, field):
    result = {}
    for row in rows:
        step = row["step"]
        if step in result:
            raise ValueError("Repeated step in a saved training history")
        result[step] = row.get(field) if _finite(row.get(field)) else None
    return result


def _collect(root, obs, inputs):
    histories, selected = {}, {}
    for seed in SEEDS:
        for domain in DOMAINS:
            for fold in range(4):
                for rate in RATES:
                    base = f"warm/seed{seed}/domain{domain}/fold{fold}/lr{rate:.0e}"
                    metadata = _read(root, base + "/metadata.json", obs, inputs)
                    history = _read(root, base + "/history.json", obs, inputs)
                    if tuple(metadata.get(key) for key in ("seed", "domain", "fold", "lr")) != (seed, domain, fold, rate):
                        raise ValueError("Saved warm identity disagrees with registered grid")
                    histories[seed, domain, fold, rate] = (
                        _history(history["training"], "train_ce"),
                        _history(history["validation"], "validation_ce"))
                    step = metadata.get("selected_step") if metadata.get("status") == "success" else None
                    if step is not None and step not in STEPS:
                        raise ValueError("Selected warm step is outside the registered cadence")
                    selected[seed, domain, fold, rate] = step
    by_seed, lookup = [], {}
    for seed in SEEDS:
        for domain in DOMAINS:
            for rate in RATES:
                for step in STEPS:
                    train = [histories[seed, domain, fold, rate][0].get(step) for fold in range(4)]
                    validation = [histories[seed, domain, fold, rate][1].get(step) for fold in range(4)]
                    row = dict(seed=seed, domain=domain, learning_rate=rate, step=step,
                        train_ce=_mean(train, 4), validation_ce=_mean(validation, 4),
                        train_available_folds=sum(_finite(value) for value in train),
                        validation_available_folds=sum(_finite(value) for value in validation), expected_folds=4)
                    by_seed.append(row)
                    lookup[seed, domain, rate, step] = row
    curves, selections = [], []
    for domain in DOMAINS:
        for rate in RATES:
            for step in STEPS:
                curves.append(dict(domain=domain, learning_rate=rate, step=step,
                    **{field: _summary([lookup[seed, domain, rate, step][field] for seed in SEEDS])
                       for field in ("train_ce", "validation_ce")}))
            values = [selected[seed, domain, fold, rate] for seed in SEEDS for fold in range(4)]
            available = [value for value in values if value is not None]
            complete = len(available) == 20
            selections.append(dict(domain=domain, learning_rate=rate, values=values,
                available_count=len(available), expected_count=20,
                mean=statistics.mean(available) if complete else None,
                median=statistics.median(available) if complete else None,
                minimum=min(available) if complete else None, maximum=max(available) if complete else None,
                range=[min(available), max(available)] if complete else None,
                zero_count=available.count(0)))
    return dict(by_seed=by_seed, across_seeds=curves, selected_step_distributions=selections)


def _plot(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(row["domain"], row["learning_rate"], row["step"]): row for row in data["across_seeds"]}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
        for col, domain in enumerate(DOMAINS):
            for line, (field, title) in enumerate((("train_ce", "A training cross-entropy"),
                                                   ("validation_ce", "A inner-validation cross-entropy"))):
                ax = axes[line, col]
                for rate, color in zip(RATES, ("#0072B2", "#D55E00")):
                    stats = [lookup[domain, rate, step][field] for step in STEPS]
                    means = [row["mean"] if _finite(row["mean"]) else math.nan for row in stats]
                    lower = [row["mean"]-row["sample_sd"] if _finite(row["mean"]) and _finite(row["sample_sd"])
                             else math.nan for row in stats]
                    upper = [row["mean"]+row["sample_sd"] if _finite(row["mean"]) and _finite(row["sample_sd"])
                             else math.nan for row in stats]
                    ax.plot(STEPS, means, color=color, linewidth=2, label=f"LR {rate:.0e}")
                    ax.fill_between(STEPS, lower, upper, color=color, alpha=.17, linewidth=0)
                ax.set_title(f"Target {domain}: {title}")
                ax.set_xlabel("Completed optimizer updates")
                ax.set_ylabel("Cross-entropy")
                ax.grid(alpha=.2)
                ax.set_xlim(0, 800)
                ax.set_xticks((0, 200, 400, 600, 800))
                ax.legend(frameon=False)
        fig.suptitle("Source-initialized warm heads: saved training and validation curves", fontsize=14)
        fig.text(.5, .018, "Four folds averaged within each seed; curves are five-seed means and shading is sample SD, not a confidence interval.\n"
                 "Steps 0, 20, …, 800 are retained for both registered learning rates; no B-driven selection.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .075, 1, .95))
        with path.open("xb") as stream:
            fig.savefig(stream, format="png", dpi=180, bbox_inches="tight", facecolor="white",
                        metadata={"Title": "Warm training curves", "Description": "Five-seed mean plus/minus sample SD after within-seed fold averaging"})
        plt.close(fig)


def _write_json(path, value, compact=False):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=None if compact else 2,
                  separators=(",", ":") if compact else None)
        stream.write("\n")


def publish(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise PermissionError("Training-curve output must be separate from the immutable archive")
    if any((output/name).exists() for name in OUTPUTS):
        raise FileExistsError("Refusing to overwrite training-curve publication")
    final, obs = verify_seal(root, "FINAL"), verify_seal(root, "OBS")
    if final["files"].get("seals/OBS.json") != _sha(root / "seals/OBS.json"):
        raise PermissionError("FINAL does not authenticate the current OBS identity")
    inputs = {}
    data = _collect(root, obs, inputs)
    metadata = dict(schema_version=1, FINAL_digest=final["digest"], OBS_digest=obs["digest"],
        input_files=inputs, helper_file="experiments/mutagenicity/r1b_20261111/src/training_curves.py",
        helper_sha256=_sha(Path(__file__)), expected_fits=80, registered_seeds=list(SEEDS),
        domains=list(DOMAINS), learning_rates=list(RATES), sampled_steps=list(STEPS),
        aggregation="Four fold CE values averaged within each seed, then mean and sample SD across five seeds; all expected folds/seeds required",
        selected_values_order="Seeds 3405,3406,3407,3408,3409; folds 0,1,2,3 within each seed",
        selected_step_distribution_scope="20 overlapping fold fits; descriptive mean, median and range only, not independent replicates",
        source_label_values_parsed=False, target_label_values_parsed=False,
        restricted_label_bytes_hashed_for_seal_verification=True, tensor_files_loaded=False,
        fits_performed=False, queries_changed=False, B_based_selection=False, archive_written=False,
        missing_values="null; incomplete fold/seed sets never replace strict means with available-only means")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output/OUTPUTS[0], dict(metadata=metadata, **data), compact=True)
    _plot(data, output/OUTPUTS[1])
    result = dict(metadata, outputs={name: dict(bytes=(output/name).stat().st_size, sha256=_sha(output/name))
                                    for name in OUTPUTS[:2]},
                  selected_step_distributions=data["selected_step_distributions"])
    _write_json(output/OUTPUTS[2], result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = publish(args.root, args.output)
    print(json.dumps({key: result[key] for key in ("outputs", "selected_step_distributions")}, indent=2))
