"""Render fixed-scale locality figures from the post-FINAL scalar summary only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


METRICS = ("min_both_sides_unlabeled_n", "min_both_sides_B_n", "ess_center",
           "ess_min_three", "max_kernel_weight_center", "origin_kernel_weight_center",
           "coverage_fraction_target_pool")
IDENTITY = ("domain", "group", "pool", "setting_id", "support_multiplier", "bandwidth_multiplier")
COLUMNS = ((1, "source"), (1, "reference_all"), (3, "source"), (3, "reference_all"))
OUTPUTS = ("neighborhood-locality.png", "kernel-concentration.png", "density-summary.json",
           "density-summary.csv", "locality-figures-metadata.json")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _lookup(summary):
    rows = {}
    for row in summary["pooled_descriptive"]:
        key = tuple(row[field] for field in ("domain", "group", "pool", "setting_id"))
        if key in rows:
            raise ValueError("Repeated pooled aggregate identity")
        rows[key] = row
    return rows


def _values(rows, domain, group, pool, settings, metric, statistic, multiplier=1.):
    result = []
    for setting in settings:
        value = rows.get((domain, group, pool, setting), {}).get("distributions", {}).get(metric, {}).get(statistic)
        result.append(multiplier * value if _finite(value) else math.nan)
    return result


def _axes_style(ax, col, ylabel, xlabel):
    ax.grid(axis="y", alpha=.2)
    ax.set_xlabel(xlabel)
    if col == 0:
        ax.set_ylabel(ylabel)
    ax.set_xlim(.9, 2.1)
    ax.set_xticks((1., 1.5, 2.))
    ax.set_ylim(bottom=0)


def _save(fig, path, title):
    # Exclusive creation also protects against output files appearing after the
    # initial preflight. No existing publication or archive artifact is changed.
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=180, facecolor="white", bbox_inches="tight",
                    metadata={"Title": title, "Description": "Descriptive pooled correlated query records; no confidence intervals"})


def _neighbors(plt, rows, output):
    settings = ("r1_b1", "r1.5_b1", "r2_b1")
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), squeeze=False, sharex=True)
    metrics = (("min_both_sides_unlabeled_n", "Minimum count across support sides", 1.),
               ("min_both_sides_B_n", "Minimum B count across support sides", 1.),
               ("coverage_fraction_target_pool", "Full target pool covered (%)", 100.))
    styles = (("median", "#0072B2", "o", "-"), ("p90", "#D55E00", "s", "--"),
              ("max", "#009E73", "^", ":"))
    for col, (domain, group) in enumerate(COLUMNS):
        title = f"Target {domain} · {'Source bank' if group == 'source' else 'Reference banks'}"
        for line, (metric, ylabel, multiplier) in enumerate(metrics):
            ax = axes[line, col]
            for statistic, color, marker, linestyle in styles:
                values = _values(rows, domain, group, "main", settings, metric, statistic, multiplier)
                ax.plot((1., 1.5, 2.), values, color=color, marker=marker,
                        linestyle=linestyle, linewidth=1.6, markersize=5, label=statistic)
                for x, value in zip((1., 1.5, 2.), values):
                    if not math.isfinite(value):
                        ax.text(x, .04, "NA", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
            _axes_style(ax, col, ylabel, "Support radius multiplier r")
            if line == 0:
                ax.set_title(title, fontsize=11)
            if col:
                ax.tick_params(labelleft=True)
    handles = [plt.Line2D([], [], color=color, marker=marker, linestyle=linestyle, label=statistic)
               for statistic, color, marker, linestyle in styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .947), ncol=3)
    fig.suptitle("Frozen support neighborhoods: radius scan at fixed density bandwidth b = 1", fontsize=15)
    fig.text(.5, .018, "Descriptive quantiles over available correlated query records; all attempted-query denominators remain in the data.\n"
             "Support excludes the originating graph. Coverage is the union of both sides / entire target pool. Main and LOO share support.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .915), h_pad=2.0)
    _save(fig, output / OUTPUTS[0], "Fixed support radius and full-target locality")
    plt.close(fig)
    return dict(file=OUTPUTS[0], settings=list(settings), pool="main (support identical in LOO)",
                metrics=[metric for metric, _, _ in metrics], statistics=[style[0] for style in styles],
                coverage_display_multiplier=100)


def _kernels(plt, rows, output):
    settings = ("r1_b1", "r1_b1.5", "r1_b2")
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), squeeze=False, sharex=True)
    metrics = (("ess_min_three", "Minimum ESS across three locations", 1.),
               ("max_kernel_weight_center", "Largest graph weight at center (%)", 100.),
               ("origin_kernel_weight_center", "Origin graph weight at center (%)", 100.))
    styles = (("main", "median", "#0072B2", "o", "-"),
              ("main", "p90", "#0072B2", "^", "--"),
              ("loo", "median", "#D55E00", "s", "-"),
              ("loo", "p90", "#D55E00", "D", "--"))
    for col, (domain, group) in enumerate(COLUMNS):
        title = f"Target {domain} · {'Source bank' if group == 'source' else 'Reference banks'}"
        for line, (metric, ylabel, multiplier) in enumerate(metrics):
            ax = axes[line, col]
            for pool, statistic, color, marker, linestyle in styles:
                values = _values(rows, domain, group, pool, settings, metric, statistic, multiplier)
                ax.plot((1., 1.5, 2.), values, color=color, marker=marker,
                        linestyle=linestyle, linewidth=1.6, markersize=5)
                for x, value in zip((1., 1.5, 2.), values):
                    if not math.isfinite(value):
                        ax.text(x, .04, "NA", transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
            _axes_style(ax, col, ylabel, "Density bandwidth multiplier b")
            if line == 0:
                ax.set_title(title, fontsize=11)
                ax.axhline(5, color=".45", linewidth=.9, linestyle=":")
                ax.text(1.02, 5, " ESS gate 5", color=".4", fontsize=8, va="bottom")
            else:
                ax.set_ylim(-2, 102)
    handles = [plt.Line2D([], [], color=color, marker=marker, linestyle=linestyle,
                         label=f"{'Main pool' if pool == 'main' else 'Origin excluded (LOO)'} · {statistic}")
               for pool, statistic, color, marker, linestyle in styles]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .948), ncol=4, fontsize=9)
    fig.suptitle("Frozen kernel concentration: bandwidth scan at fixed support radius r = 1", fontsize=15)
    fig.text(.5, .018, "Descriptive quantiles over available correlated query records; no confidence intervals or setting selection.\n"
             "Center and shoulders remain fixed. LOO removes the originating graph before kernel normalization; its LOO weight is exactly zero.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .915), h_pad=2.0)
    _save(fig, output / OUTPUTS[1], "Fixed bandwidth and kernel concentration")
    plt.close(fig)
    return dict(file=OUTPUTS[1], settings=list(settings), pools=["main", "loo"],
                metrics=[metric for metric, _, _ in metrics], statistics=["median", "p90"],
                weight_display_multiplier=100)


def _compact(summary):
    across = []
    for row in summary["across_seeds"]:
        across.append(dict({key: row[key] for key in IDENTITY}, count_per_bank=row["count_per_bank"],
            equal_bank_mean_statistics={metric: row["equal_bank_mean_statistics"][metric] for metric in METRICS}))
    pooled = []
    for row in summary["pooled_descriptive"]:
        pooled.append(dict({key: row[key] for key in IDENTITY}, counts=row["counts"],
            distributions={metric: row["distributions"][metric] for metric in METRICS}))
    return dict(metadata=summary["metadata"], across_seeds=across, pooled_descriptive=pooled)


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        key = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, key))
        else:
            result[key] = item
    return result


def _write_json(path, value, compact=False):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=None if compact else 2,
                  separators=(",", ":") if compact else None)
        stream.write("\n")


def render(summary_path, output):
    """Create two PNGs and compact scalar exports; never open the run archive."""
    summary_path, output = Path(summary_path).resolve(strict=True), Path(output).resolve()
    if any((output / name).exists() for name in OUTPUTS):
        raise FileExistsError("Refusing to overwrite a locality publication output")
    # The caller supplies the external analysis summary, not the experiment.
    # Detect archive destinations, including paths nested below a sealed root.
    if any((parent / "seals/OBS.json").exists() or (parent / "seals/FINAL.json").exists()
           for parent in (output, *output.parents)):
        raise PermissionError("Figures must be written outside every sealed run archive")
    raw = summary_path.read_bytes()
    summary_hash = hashlib.sha256(raw).hexdigest()
    summary = json.loads(raw)
    del raw
    meta = summary["metadata"]
    if not meta.get("FINAL_digest") or not meta.get("OBS_digest"):
        raise ValueError("Expected a post-FINAL verified scalar observation summary")
    if meta.get("setting_selection_performed") is not False:
        raise ValueError("Only the registered unselected scale grid may be rendered")
    compact = _compact(summary)
    provenance = dict(schema_version=1, input_file=summary_path.name, input_sha256=summary_hash,
                      helper_file="experiments/mutagenicity/r1b_密度与单步适配诊断_20261111/src/locality_figures.py",
                      helper_sha256=_sha(Path(__file__)), retained_metric_keys=list(METRICS),
                      compact_scope="All six settings, main and LOO, source/linear/mlp/reference_all; across-seed and pooled scalar summaries only",
                      raw_queries_read=False, labels_parsed=False, experiment_archive_written=False,
                      fitting_or_setting_selection_performed=False)
    compact["publication_provenance"] = provenance
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / OUTPUTS[2], compact, compact=True)
    csv_rows = [dict(table=name, **_flatten(row)) for name in ("across_seeds", "pooled_descriptive")
                for row in compact[name]]
    fields = sorted({key for row in csv_rows for key in row})
    with (output / OUTPUTS[3]).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _lookup(summary)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figures = [_neighbors(plt, rows, output), _kernels(plt, rows, output)]
    result = dict(provenance, figures=figures, outputs={name: dict(bytes=(output/name).stat().st_size,
                    sha256=_sha(output/name)) for name in OUTPUTS[:-1]},
                  figure_statistics="Pooled descriptive quantiles of correlated records; no confidence intervals",
                  compact_JSON_preserves_source_metadata=True)
    _write_json(output / OUTPUTS[4], result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.summary, args.output), indent=2))
