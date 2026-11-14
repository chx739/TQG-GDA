"""Post-FINAL aggregate support, ESS and locality from frozen query records.

No new observations, neighbors, density estimates, fits or setting selections
are made. Query identifiers and label-related payloads are never exported.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.common import verify_seal


SEEDS = (3405, 3406, 3407, 3408, 3409)
DOMAINS = (1, 3)
POOLS = ("main", "loo")
SETTINGS = (("r1_b1", 1., 1.), ("r1.5_b1", 1.5, 1.), ("r2_b1", 2., 1.),
            ("r1_b1.5", 1., 1.5), ("r1_b2", 1., 2.), ("r2_b2", 2., 2.))
BANKS = ("source",) + tuple(f"{head}_fold{fold}" for head in
                          ("linear", "hidden32_silu_mlp") for fold in range(4))
GROUPS = {"source": ("source",), "linear": BANKS[1:5], "mlp": BANKS[5:9],
          "reference_all": BANKS[1:]}
QUERIES_PER_BANK = 64
SOURCE = "evaluation/scales_queries.json"
DIRECT_METRICS = (
    "positive_unlabeled_n", "negative_unlabeled_n", "positive_B_n", "negative_B_n",
    "coverage_fraction_target_pool", "coverage_fraction_independent_pool",
    "positive_coverage_fraction_target_pool", "negative_coverage_fraction_target_pool",
    "positive_coverage_fraction_independent_pool", "negative_coverage_fraction_independent_pool",
) + tuple(f"{side}_{distance}_{statistic}" for side in ("positive", "negative")
          for distance in ("distance", "tangent") for statistic in ("median", "max"))
DERIVED_METRICS = ("min_both_sides_unlabeled_n", "min_both_sides_B_n", "ess_center",
                   "ess_min_three", "max_kernel_weight_center", "origin_kernel_weight_center")
METRICS = DIRECT_METRICS + DERIVED_METRICS
STATISTICS = ("median", "p10", "p90", "max")
COUNTS = ("attempted", "records_present", "missing_records", "projected", "support_observed",
          "observed", "eligible", "joint_positive", "semantic_joint_positive",
          "unlabeled_supported", "B_supported", "ess_supported", "failed_records")


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _read_final_queries(root):
    """Verify FINAL while reading the potentially large query file only once."""
    seal = json.loads((root / "seals/FINAL.json").read_text())
    content = {key: seal[key] for key in ("stage", "files")}
    checksum = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if seal["stage"] != "FINAL" or checksum != seal["digest"] or not seal["files"]:
        raise PermissionError("Invalid FINAL seal")
    if SOURCE not in seal["files"] or "seals/OBS.json" not in seal["files"]:
        raise PermissionError("FINAL must seal both observations identity and query evaluation")
    raw = None
    for name, expected in seal["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise PermissionError("Sealed path escapes the archive")
        if name == SOURCE:
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
        else:
            actual = _digest(path)
        if actual != expected:
            raise PermissionError(f"Sealed evidence changed: {name}")
    return seal, raw


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pair_min(row, first, second):
    values = (row.get(first), row.get(second))
    return min(values) if all(_finite(value) for value in values) else None


def _center(row, key):
    value = row.get(key)
    return value[0] if isinstance(value, list) and len(value) == 3 and _finite(value[0]) else None


def _metric_values(row):
    values = {key: row.get(key) if _finite(row.get(key)) else None for key in DIRECT_METRICS}
    ess = row.get("ess")
    values.update(
        min_both_sides_unlabeled_n=_pair_min(row, "positive_unlabeled_n", "negative_unlabeled_n"),
        min_both_sides_B_n=_pair_min(row, "positive_B_n", "negative_B_n"),
        ess_center=_center(row, "ess"),
        ess_min_three=min(ess) if isinstance(ess, list) and len(ess) == 3 and
            all(_finite(value) for value in ess) else None,
        max_kernel_weight_center=_center(row, "max_kernel_weight"),
        origin_kernel_weight_center=_center(row, "origin_kernel_weight"))
    return values


def _quantile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values, denominator):
    values = sorted(float(value) for value in values if _finite(value))
    return dict(available_count=len(values), attempted_denominator=denominator,
                missing_count=denominator-len(values),
                median=_quantile(values, .5) if values else None,
                p10=_quantile(values, .1) if values else None,
                p90=_quantile(values, .9) if values else None,
                max=values[-1] if values else None)


def _mean(values, expected):
    available = [float(value) for value in values if _finite(value)]
    return dict(mean=statistics.mean(available) if len(available) == expected else None,
                available_mean=statistics.mean(available) if available else None,
                available_count=len(available), expected_count=expected)


def _seed_summary(values):
    result = _mean(values, len(SEEDS))
    result["sample_sd"] = statistics.stdev(values) if result["available_count"] == len(SEEDS) else None
    return result


def _bank_summary(rows):
    values = {key: [] for key in METRICS}
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(attempted=QUERIES_PER_BANK, records_present=len(rows),
                  missing_records=QUERIES_PER_BANK-len(rows))
    for row in rows:
        extracted = _metric_values(row)
        for key, value in extracted.items():
            values[key].append(value)
        counts["projected"] += bool(row.get("success"))
        counts["support_observed"] += extracted["min_both_sides_unlabeled_n"] is not None
        counts["observed"] += extracted["ess_min_three"] is not None
        counts["failed_records"] += bool(row.get("failure")) or not bool(row.get("success"))
        for key in ("eligible", "joint_positive", "semantic_joint_positive", "unlabeled_supported",
                    "B_supported", "ess_supported"):
            counts[key] += bool(row.get(key))
    return {"counts": counts, "distributions": {
        key: _distribution(items, QUERIES_PER_BANK) for key, items in values.items()}}, values


def _aggregate(rows):
    """Retain the registered bank grid even for absent/failed query records."""
    by_bank = defaultdict(list)
    allowed_settings = {setting: (radius, bandwidth) for setting, radius, bandwidth in SETTINGS}
    for row in rows:
        identity = tuple(row.get(key) for key in ("seed", "domain", "head_key", "pool", "setting_id"))
        seed, domain, head, pool, setting = identity
        if seed not in SEEDS or domain not in DOMAINS or head not in BANKS or pool not in POOLS or setting not in allowed_settings:
            raise ValueError("Query row is outside the registered grid")
        if (row.get("support_multiplier"), row.get("bandwidth_multiplier")) != allowed_settings[setting]:
            raise ValueError("Query setting identity and fixed multipliers disagree")
        by_bank[identity].append(row)
        if len(by_bank[identity]) > QUERIES_PER_BANK:
            raise ValueError("More than 64 attempted queries in a registered bank")
    banks, bank_lookup, bank_values = [], {}, {}
    for seed in SEEDS:
        for domain in DOMAINS:
            for head in BANKS:
                for pool in POOLS:
                    for setting, radius, bandwidth in SETTINGS:
                        key = (seed, domain, head, pool, setting)
                        summary, values = _bank_summary(by_bank[key])
                        identity = dict(seed=seed, domain=domain, head_key=head, pool=pool, setting_id=setting,
                                        support_multiplier=radius, bandwidth_multiplier=bandwidth)
                        item = dict(identity, **summary)
                        banks.append(item)
                        bank_lookup[key], bank_values[key] = item, values
    seed_rows, seed_lookup, pooled_rows = [], {}, []
    for domain in DOMAINS:
        for group, heads in GROUPS.items():
            for pool in POOLS:
                for setting, radius, bandwidth in SETTINGS:
                    identity = dict(domain=domain, group=group, pool=pool, setting_id=setting,
                                    support_multiplier=radius, bandwidth_multiplier=bandwidth)
                    pooled = {metric: [] for metric in METRICS}
                    total_counts = dict.fromkeys(COUNTS, 0)
                    for seed in SEEDS:
                        selected = [bank_lookup[(seed, domain, head, pool, setting)] for head in heads]
                        counts = {key: sum(bank["counts"][key] for bank in selected) for key in COUNTS}
                        for key in COUNTS:
                            total_counts[key] += counts[key]
                        distributions = {}
                        for metric in METRICS:
                            distributions[metric] = {
                                statistic: _mean([bank["distributions"][metric][statistic]
                                                  for bank in selected], len(heads))
                                for statistic in STATISTICS}
                            distributions[metric]["available_query_records"] = sum(
                                bank["distributions"][metric]["available_count"] for bank in selected)
                            distributions[metric]["attempted_query_records"] = counts["attempted"]
                            for head in heads:
                                pooled[metric].extend(bank_values[(seed, domain, head, pool, setting)][metric])
                        item = dict(identity, seed=seed, expected_banks=len(heads), counts=counts,
                                    count_bank_means={key: counts[key]/len(heads) for key in COUNTS},
                                    equal_bank_mean_statistics=distributions)
                        seed_rows.append(item)
                        seed_lookup[(seed, domain, group, pool, setting)] = item
                    pooled_rows.append(dict(identity, counts=total_counts,
                        distributions={metric: _distribution(values, total_counts["attempted"])
                                       for metric, values in pooled.items()}))
    across_seeds = []
    for pooled in pooled_rows:
        identity = {key: pooled[key] for key in ("domain", "group", "pool", "setting_id",
                                               "support_multiplier", "bandwidth_multiplier")}
        selected = [seed_lookup[(seed, identity["domain"], identity["group"], identity["pool"],
                                 identity["setting_id"])] for seed in SEEDS]
        across_seeds.append(dict(identity,
            count_per_bank={key: _seed_summary([row["count_bank_means"][key] for row in selected])
                            for key in COUNTS},
            equal_bank_mean_statistics={metric: {
                statistic: _seed_summary([row["equal_bank_mean_statistics"][metric][statistic]["mean"]
                                          for row in selected]) for statistic in STATISTICS} for metric in METRICS}))
    return dict(bank_statistics=banks, within_seed=seed_rows, across_seeds=across_seeds,
                pooled_descriptive=pooled_rows)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _write_csv(path, tables):
    rows = [dict(table=name, **_flatten(row)) for name, values in tables.items() for row in values]
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(root, outputfile):
    """Write JSON, CSV and scope metadata outside the sealed run, without overwrite."""
    root, outputfile = Path(root).resolve(strict=True), Path(outputfile).resolve()
    if outputfile.suffix.lower() != ".json":
        raise ValueError("outputfile must have a .json suffix")
    paths = (outputfile, outputfile.with_suffix(".csv"), outputfile.with_suffix(".metadata.json"))
    if any(path.is_relative_to(root) for path in paths):
        raise PermissionError("Summary output must be outside the immutable archive")
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite summary output")
    obs = verify_seal(root, "OBS")
    final, raw = _read_final_queries(root)
    if final["files"]["seals/OBS.json"] != _digest(root / "seals/OBS.json"):
        raise PermissionError("Observation identity changed during verification")
    raw_size = len(raw)
    raw_sha = hashlib.sha256(raw).hexdigest()
    records = json.loads(raw)
    del raw
    if not isinstance(records, list):
        raise ValueError("Expected a list of sealed query records")
    record_count = len(records)
    tables = _aggregate(records)
    del records
    metadata = dict(schema_version=1, FINAL_digest=final["digest"], OBS_digest=obs["digest"],
        input={"path": SOURCE, "sha256": raw_sha, "bytes": raw_size,
               "query_records": record_count, "reads": 1, "json_parses": 1},
        registered_seeds=list(SEEDS), domains=list(DOMAINS), pools=list(POOLS),
        settings=[dict(setting_id=s, support_multiplier=r, bandwidth_multiplier=b) for s, r, b in SETTINGS],
        groups={key: list(value) for key, value in GROUPS.items()},
        attempted_queries_per_bank=QUERIES_PER_BANK,
        expected_query_records=len(SEEDS)*len(DOMAINS)*len(BANKS)*len(POOLS)*len(SETTINGS)*QUERIES_PER_BANK,
        observed_definition="All three saved ESS values finite; support_observed counts available side counts separately",
        missing_values="null / empty CSV cell; absent bank statistics remain NA, never replaced by zero",
        count_denominators="All 64 registered attempts retained, including projection/observation failures; counts report observed events",
        distribution_quantiles="Linear interpolation at (n-1)*p over finite saved query scalars; no eligibility filtering",
        bank_to_seed="Each statistic averaged equally across every registered bank in the group; incomplete bank sets yield null mean and separate available_mean",
        seed_to_summary="Mean and sample SD of the five within-seed equal-bank statistics; requires all five complete seed means",
        pooled_descriptive="Quantiles of all available correlated query records; no independence assumption, standard error or confidence interval",
        overlapping_groups="reference_all contains the same four linear and four MLP banks; groups are not disjoint replicates",
        center_index=0, query_locations=["center", "positive shoulder", "negative shoulder"],
        distance_units="Frozen embedding coordinates; saved side distance/tangent medians and maxima, not graph-distance samples",
        locality="coverage_fraction_target_pool is the union of both frozen support sides divided by the entire 1084/1085 target input pool, excluding origin from support",
        B_scope="Previously observed development B; no setting selection or inferential independence claimed",
        restricted_label_values_parsed=False, restricted_label_bytes_hashed_for_seal_verification=True,
        per_graph_values_exported=False,
        new_neighbors=False, new_density_estimates=False, model_fits=False,
        setting_selection_performed=False, immutable_archive_written=False)
    outputfile.parent.mkdir(parents=True, exist_ok=True)
    _write_json(paths[0], dict(metadata=metadata, **tables))
    _write_csv(paths[1], tables)
    _write_json(paths[2], metadata)
    return dict(output_json=str(paths[0]), output_csv=str(paths[1]), metadata_json=str(paths[2]),
                input_sha256=raw_sha, query_records=record_count,
                table_rows={key: len(values) for key, values in tables.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.output), indent=2))
