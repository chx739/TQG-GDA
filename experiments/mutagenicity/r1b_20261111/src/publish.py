"""Export sealed R1b aggregates and scientific figures outside the run archive.

This module never loads tensors, labels, per-graph predictions or per-query
records. Every plotted setting is fixed by the registration, never ranked by B.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from experiments.mutagenicity.r1b_20261111.src.common import verify_seal


AGGREGATES = (
    "evaluation/warm_summary.json", "evaluation/warm_by_seed.json",
    "evaluation/warm_paired.json", "evaluation/steps_summary.json",
    "evaluation/steps_comparisons.json", "evaluation/steps_paired.json",
    "evaluation/scales_summary.json", "evaluation/scales_groups.json",
    "evaluation/scales_seed_aggregates.json", "evaluation/scales_banks.json",
    "scales/index.json", "scores/index.json", "completion.json",
    "phase-outcomes.json", "prediction-replay.json", "evaluation/audit.json",
)
METHODS = ("source_initial", "random_corrected", "warm_lr1e-03",
           "warm_lr1e-04", "old_random_mlp")
METHOD_LABELS = ("Source initial", "Random corrected", "Warm LR 1e−3",
                 "Warm LR 1e−4", "Old random MLP")
SETTINGS = ("r1_b1", "r1.5_b1", "r2_b1", "r1_b1.5", "r1_b2", "r2_b2")
SCORE_FIELDS = {
    "status", "epochs_requested", "updates_completed", "epochs_completed",
    "selected_epoch", "seed", "bandwidth", "hidden", "lr", "batch",
    "initialization_seed", "training_rng_seed", "validation_rng_seed",
    "validation_scope", "noise_validation_loss", "elapsed_seconds",
    "peak_rss_bytes", "domain", "bandwidth_multiplier", "sigma0", "h",
}
STATIC_FIELDS = {
    "reference", "n", "vector_rmse", "vector_rmse_sigma", "normal_score_rmse",
    "normal_score_rmse_sigma", "normal_sign_agreement", "normal_sign_threshold_sigma",
    "normal_sign_threshold_denominator", "normal_sign_agreement_above_threshold",
    "request_rmse_sigma", "request_sign_agreement", "seed", "domain",
    "bandwidth_multiplier", "sigma0", "h",
}


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _write_csv(path, value):
    rows = value if isinstance(value, list) else [value]
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(item, separators=(",", ":"))
                             if isinstance(item, (dict, list)) else item
                             for key, item in row.items()})


def _check_public(value):
    """Fail closed if an aggregate schema unexpectedly gains graph-level data."""
    if isinstance(value, dict):
        for key, item in value.items():
            if (key in {"id", "ids", "origin_id", "cluster_id", "labels", "probabilities",
                        "log_probabilities", "model_state", "optimizer_state", "anchors",
                        "empirical_gamma", "learned_gamma", "empirical_request", "learned_request"}
                    or key.endswith("_ids")):
                raise ValueError(f"Nonpublic field in aggregate schema: {key}")
            _check_public(item)
    elif isinstance(value, list):
        for item in value:
            _check_public(item)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ValueError("Only JSON aggregate scalars may be published")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite aggregate; missing values must be JSON null")


def _score_metadata(rows):
    result = []
    for row in rows:
        metadata = {key: value for key, value in row.items() if key in SCORE_FIELDS}
        static = row.get("static_comparison", {})
        metadata["static_comparison"] = {key: value for key, value in static.items()
                                         if key in STATIC_FIELDS}
        if any(isinstance(value, (dict, list)) for key, value in metadata.items()
               if key != "static_comparison") or any(
                isinstance(value, (dict, list)) for value in metadata["static_comparison"].values()):
            raise ValueError("Score metadata must contain only scalar fields")
        metadata["failure_recorded"] = bool(row.get("failure"))
        result.append(metadata)
    return result


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _missing(ax, x, label="NA"):
    ax.text(x, .04, label, transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=8, color="0.45")


def _save(fig, path, title):
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight",
                metadata={"Title": title, "Description": "Sealed R1b public aggregate data"})


def _warm_plot(plt, data, output):
    rows = data.get("evaluation/warm_summary.json", [])
    if not rows:
        return {"status": "skipped", "reason": "warm_summary is missing or empty"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), squeeze=False)
    plotted = 0
    for col, domain in enumerate((1, 3)):
        selected = {row["method"]: row for row in rows if row["domain"] == domain
                    and row["endpoint"] == "selected" and row["pool"] == f"B{domain}"}
        for line, (metric, label) in enumerate((("balanced_accuracy", "Balanced accuracy"),
                                               ("nll", "Negative log-likelihood"))):
            ax = axes[line, col]
            for x, method in enumerate(METHODS):
                stats = selected.get(method, {}).get(metric, {})
                mean, sd = stats.get("mean"), stats.get("sample_sd")
                if _finite(mean) and _finite(sd):
                    ax.errorbar(x, mean, yerr=sd, fmt="o", capsize=4,
                                color=plt.get_cmap("tab10")(x), markersize=6)
                    plotted += 1
                else:
                    _missing(ax, x, f"NA ({stats.get('available_count', 0)}/5)")
            ax.set_xticks(range(len(METHODS)), METHOD_LABELS, rotation=24, ha="right")
            ax.set_xlim(-.5, len(METHODS)-.5)
            ax.set_ylabel(label)
            ax.set_title(f"Source 0 → target {domain}")
            ax.grid(axis="y", alpha=.25)
    fig.suptitle("Selected reference heads on B: mean ± seed sample SD", fontsize=14)
    fig.text(.5, .005, "Four overlapping folds are averaged within each seed; five seeds share B. "
             "Warm LR 1e−4 is a sensitivity setting.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    path = output / "warm-reference-heads.png"
    _save(fig, path, "Warm reference head comparison")
    plt.close(fig)
    return {"status": "created", "file": path.name, "plotted_mean_sd_pairs": plotted,
            "uncertainty": "sample SD across five seed means; not confidence intervals"}


def _step_plot(plt, data, output):
    rows = data.get("evaluation/steps_comparisons.json", [])
    if not rows:
        return {"status": "skipped", "reason": "steps_comparisons is missing or empty"}
    combinations = [(estimator, bandwidth, view)
                    for estimator in ("empirical", "learned") for bandwidth in (1., 2.)
                    for view in ("small_step", "production_backtrack")]
    labels = [f"{estimator}\nb={bandwidth:g}, {'small' if view == 'small_step' else 'backtrack'}"
              for estimator, bandwidth, view in combinations]
    seeds = sorted({row["seed"] for row in rows})
    markers = ("o", "s", "^", "D", "v", "P", "X")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    plotted, absent = 0, 0
    for col, domain in enumerate((1, 3)):
        for line, control in enumerate(("matched_source", "reverse")):
            ax = axes[line, col]
            lookup = {(row["estimator"], row["bandwidth_multiplier"], row["view"], row["seed"]): row
                      for row in rows if row["domain"] == domain and row["control"] == control}
            for x, combination in enumerate(combinations):
                missing = 0
                for i, seed in enumerate(seeds):
                    value = lookup.get((*combination, seed), {}).get("nll_difference")
                    if _finite(value):
                        ax.scatter(x + (i - (len(seeds)-1)/2) * .075, value,
                                   color=plt.get_cmap("tab10")(i), marker=markers[i % len(markers)],
                                   s=32, label=f"Seed {seed}" if x == 0 else None, zorder=3)
                        plotted += 1
                    else:
                        missing += 1
                if missing:
                    _missing(ax, x, f"{missing} NA")
                    absent += missing
            ax.axhline(0, color="0.25", lw=1)
            ax.set_xticks(range(len(combinations)), labels, rotation=25, ha="right", fontsize=8)
            ax.set_title(f"Target {domain}: normal − {control}")
            ax.set_ylabel("Paired B NLL difference (lower favors normal)")
            ax.grid(axis="y", alpha=.2)
    handles = [plt.Line2D([], [], color=plt.get_cmap("tab10")(i), marker=markers[i % len(markers)],
                         linestyle="", label=f"Seed {seed}") for i, seed in enumerate(seeds)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=max(1, len(seeds)))
    fig.suptitle("Single-step controls: every registered estimator, bandwidth and execution view", fontsize=14)
    fig.tight_layout(rect=(0, .02, 1, .91))
    path = output / "single-step-paired-nll.png"
    _save(fig, path, "Single-step paired B NLL differences")
    plt.close(fig)
    return {"status": "created", "file": path.name, "seed_ids": seeds,
            "plotted_seed_points": plotted, "missing_seed_points": absent,
            "difference": "normal minus paired control; no acceptance-subset filtering"}


def _score_plot(plt, data, output):
    rows = data.get("scores/index.json", [])
    if not rows:
        return {"status": "skipped", "reason": "scores/index is missing or empty"}
    seeds = sorted({row["seed"] for row in rows})
    metrics = (("normal_sign_agreement", "Normal sign agreement"),
               ("normal_sign_agreement_above_threshold", "Sign agreement: |σ₀ normal KDE score| ≥ 10⁻³"),
               ("normal_score_rmse_sigma", "Normal score RMSE × σ₀"),
               ("vector_rmse_sigma", "Vector score RMSE × σ₀"),
               ("request_rmse_sigma", "Request RMSE / σ₀"))
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 15), squeeze=False)
    plotted = 0
    for col, domain in enumerate((1, 3)):
        lookup = {(row["bandwidth_multiplier"], row["seed"]): row.get("static_comparison", {})
                  for row in rows if row["domain"] == domain}
        for line, (metric, title) in enumerate(metrics):
            ax = axes[line, col]
            for x, bandwidth in enumerate((1., 2.)):
                missing = 0
                for i, seed in enumerate(seeds):
                    value = lookup.get((bandwidth, seed), {}).get(metric)
                    if _finite(value):
                        ax.scatter(x + (i-(len(seeds)-1)/2)*.06, value,
                                   color=plt.get_cmap("tab10")(i), s=35,
                                   label=f"Seed {seed}" if x == 0 else None)
                        plotted += 1
                    else:
                        missing += 1
                if missing:
                    _missing(ax, x, f"{missing} NA")
            ax.set_xticks((0, 1), ("h = σ₀", "h = 2σ₀"))
            ax.set_xlim(-.4, 1.4)
            ax.set_title(f"Target {domain}: {title}", fontsize=10)
            ax.grid(axis="y", alpha=.2)
            if "agreement" in metric:
                ax.set_ylim(-.03, 1.03)
            else:
                ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=plt.get_cmap("tab10")(i), marker="o", linestyle="",
                         label=f"Seed {seed}") for i, seed in enumerate(seeds)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .975), ncol=max(1, len(seeds)))
    fig.suptitle("Learned score versus exact finite-pool KDE at frozen source anchors", fontsize=13)
    fig.text(.5, .008, "Each point is one seed; KDE is a finite-pool reference. "
             "The thresholded panel excludes near-zero reference scores.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .95))
    path = output / "static-score-comparison.png"
    _save(fig, path, "Static learned score comparison")
    plt.close(fig)
    return {"status": "created", "file": path.name, "seed_ids": seeds,
            "plotted_seed_points": plotted, "normalization": "original frozen sigma0, not fitted bandwidth"}


def _density_plot(plt, data, output):
    observations = data.get("scales/index.json", [])
    banks = data.get("evaluation/scales_banks.json", [])
    if not observations or not banks:
        return {"status": "skipped", "reason": "aggregate scales/index or scales_banks missing"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), squeeze=False)
    plotted = 0
    for col, domain in enumerate((1, 3)):
        for line, (field, title) in enumerate((
                ("unlabeled_supported", "Both sides have ≥20 independent graphs"),
                ("ess_supported", "All three KDE locations have ESS ≥5 (main pool)"),
                ("distinct_B_covered", "Distinct B coverage among eligible anchors (main pool)"))):
            ax = axes[line, col]
            for group, offset, color in (("source", -.14, "#0072B2"), ("reference", .14, "#D55E00")):
                for x, setting in enumerate(SETTINGS):
                    source = observations if field != "distinct_B_covered" else banks
                    selected = [row for row in source if row["domain"] == domain
                                and row["setting_id"] == setting
                                and (row["head_key"] == "source") == (group == "source")
                                and (field != "distinct_B_covered" or row["pool"] == "main")]
                    # All folds stay within seed. These are seed distributions,
                    # never confidence intervals over overlapping reference banks.
                    by_seed = {}
                    for row in selected:
                        denominator = 271 if field == "distinct_B_covered" else row["denominator"]
                        if _finite(row.get(field)) and denominator > 0:
                            by_seed.setdefault(row["seed"], []).append(row[field]/denominator)
                    values = [sum(v)/len(v) for _, v in sorted(by_seed.items())]
                    for i, value in enumerate(values):
                        ax.scatter(x+offset+(i-(len(values)-1)/2)*.028, value, s=25,
                                   color=color, marker="o" if group == "source" else "s", alpha=.85)
                        plotted += 1
                    if not values:
                        _missing(ax, x+offset)
            ax.set_xticks(range(len(SETTINGS)), SETTINGS, rotation=25, ha="right")
            ax.set_ylim(-.03, 1.03)
            ax.set_ylabel("Fraction")
            ax.set_title(f"Target {domain}: {title}", fontsize=10)
            ax.grid(axis="y", alpha=.2)
    handles = [plt.Line2D([], [], color=color, marker=marker, linestyle="", label=label)
               for color, marker, label in (("#0072B2", "o", "Source bank"),
                                             ("#D55E00", "s", "Reference banks: within-seed mean"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .96), ncol=2)
    fig.suptitle("All six fixed scales: distributions of aggregate support, ESS gates and B coverage", fontsize=13)
    fig.text(.5, .008, "Points are seeds. Coverage uses the 271-graph B pool and eligible anchors; "
             "it is not full target neighborhood coverage.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .93))
    path = output / "density-scale-aggregates.png"
    _save(fig, path, "Fixed density scale aggregate diagnostics")
    plt.close(fig)
    return {"status": "created", "file": path.name, "plotted_seed_points": plotted,
            "raw_ess_distribution": "unavailable in public aggregate schemas",
            "target_neighborhood_coverage_distribution": "unavailable in public aggregate schemas",
            "coverage_shown": "distinct B graphs covered by eligible anchors / 271, main pool",
            "aggregation": "reference banks averaged within seed; no setting selected"}


def publish(root, output):
    """Verify FINAL and OBS, then create a new public directory outside root."""
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise PermissionError("Publication must be separate from the immutable run archive")
    if output.exists():
        raise FileExistsError(output)
    seals = {stage: verify_seal(root, stage) for stage in ("FINAL", "OBS")}
    seal_hashes = {stage: _sha(root / "seals" / f"{stage}.json") for stage in seals}
    data, sources, missing = {}, {}, []
    for name in AGGREGATES:
        path = root / name
        if not path.exists():
            missing.append(name)
            continue
        if not path.resolve().is_relative_to(root):
            raise PermissionError(f"Aggregate escapes archive: {name}")
        expected = {record["files"][name] for record in seals.values() if name in record["files"]}
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if expected != {checksum}:
            raise PermissionError(f"Aggregate is unsealed or changed: {name}")
        value = json.loads(content)
        if name == "scores/index.json":
            value = _score_metadata(value)
        _check_public(value)
        if not isinstance(value, (list, dict)) or (isinstance(value, list) and
                any(not isinstance(row, dict) for row in value)):
            raise ValueError(f"Expected aggregate record(s): {name}")
        data[name] = value
        sources[name] = {"sha256": checksum, "bytes": len(content)}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in data.items():
        target = output / name
        _write_json(target, value)
        _write_csv(target.with_suffix(".csv"), value)
    manifest = {"schema_version": 1, "seals": {
        stage: {"digest": record["digest"], "seal_file_sha256": seal_hashes[stage]}
        for stage, record in seals.items()}, "aggregate_inputs": sources,
        "missing_optional_aggregates": missing, "labels_parsed": False,
        "tensors_loaded": False, "per_graph_or_query_records_read": False}
    _write_json(output / "input-hash-manifest.json", manifest)
    plots = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        plots["all"] = {"status": "skipped", "reason": f"matplotlib unavailable: {exc}"}
    else:
        with plt.rc_context({"font.family": "DejaVu Sans", "axes.spines.top": False,
                             "axes.spines.right": False, "font.size": 10}):
            for name, renderer in (("warm", _warm_plot), ("paired_steps", _step_plot),
                                   ("static_score", _score_plot), ("density_scales", _density_plot)):
                plots[name] = renderer(plt, data, output)
    for stage, expected in seal_hashes.items():
        if _sha(root / "seals" / f"{stage}.json") != expected:
            raise PermissionError("Seal changed during publication")
    result = {"schema_version": 1, "status": "published", "aggregate_files": len(data),
              "missing_optional_aggregates": missing, "figures": plots,
              "setting_selection_performed": False, "archive_writes": False,
              "figure_inputs": "only exported aggregate JSON; all settings retained"}
    _write_json(output / "publication.json", result)
    _write_json(output / "public-file-manifest.json", {
        str(path.relative_to(output)): {"sha256": _sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.output), indent=2))
