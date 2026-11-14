"""Independent, post-S2 evaluation of the frozen Mutagenicity R1 observations.

No model fitting, query construction, neighbor selection, or density estimation is
performed here. Labels enter only through the stage-gated broker. Per-graph and
per-query records stay in the local archive; ``public`` contains aggregate data.
The thresholds below are descriptive protocol rules, not significance tests.
"""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.common.io import save_tensor, write_json


from experiments.common.statistics import classification_metrics, clean, difference, finite, ratio, validated_nonnegative_errorbars, wilson, write_csv

ARCHITECTURES = ("linear", "mlp")
METRICS = ("accuracy", "balanced_accuracy", "nll", "recall0", "recall1")


def architecture(name):
    return "mlp" if name in ("mlp", "hidden32_silu_mlp") else name


















def strict_fold_summary(values, expected=4):
    """The available-fold mean is descriptive and never replaces the endpoint."""
    if len(values) != expected:
        raise ValueError("The fixed fold denominator must be retained")
    valid = [float(x) for x in values if finite(x)]
    return dict(mean=float(np.mean(valid)) if len(valid) == expected else None,
                available_mean=float(np.mean(valid)) if valid else None,
                available_count=len(valid), expected_count=expected, values=clean(values))


def seed_summary(values, expected=5):
    valid = [float(x) for x in values if finite(x)]
    return dict(mean=float(np.mean(valid)) if len(valid) == expected else None,
                sample_sd=float(np.std(valid, ddof=1)) if len(valid) == expected and expected > 1 else None,
                available_mean=float(np.mean(valid)) if valid else None,
                available_count=len(valid), expected_count=expected, values=clean(values))


def nll_screen(values, expected=5):
    summary = seed_summary(values, expected)
    positive = sum(finite(value) and value > 0 for value in values)
    complete = summary["available_count"] == expected
    worsening = complete and positive >= 4 and summary["mean"] >= .01
    return dict(mean=summary["mean"], positive_seeds=positive, complete=complete,
                systematic_worsening=worsening, passes=complete and not worsening)


def direction_from_ci(interval, ess):
    if interval is None or len(interval) != 2 or not all(finite(x) for x in interval) or not finite(ess) or ess < 5:
        return None
    if float(interval[0]) > 1e-3:
        return -1
    if float(interval[1]) < -1e-3:
        return 1
    return None


def reference_consensus(roots, sigma):
    """Keep denominator eight, including absent, tangent, and coincident heads."""
    if len(roots) != 8:
        raise ValueError("All eight reference root records must be retained")
    valid = []
    statuses = Counter()
    for root in roots:
        statuses[root.get("status", "missing")] += 1
        t = root.get("t")
        if root.get("status") == "detected_unique" and finite(t) and abs(t) > .05 * sigma:
            valid.append((1 if t > 0 else -1, float(t), architecture(root.get("architecture"))))
    majorities = {}
    for head in ARCHITECTURES:
        signs = [r[0] for r in valid if r[2] == head]
        pos, neg = signs.count(1), signs.count(-1)
        majorities[head] = 1 if pos > neg else -1 if neg > pos else None
    conflict = majorities["linear"] is not None and majorities["mlp"] is not None and majorities["linear"] != majorities["mlp"]
    best = max((-1, 1), key=lambda sign: sum(r[0] == sign for r in valid))
    same = [r[1] for r in valid if r[0] == best]
    iqr = float(np.quantile(same, .75) - np.quantile(same, .25)) / sigma if same else None
    supported = len(same) >= 6 and iqr <= 1 and not conflict
    return dict(status="consensus" if supported else "reference_uncertain", sign=best if supported else None,
                same_direction_count=len(same), usable_count=len(valid), denominator=8,
                iqr_sigma=iqr, architecture_conflict=conflict, architecture_majorities=majorities,
                root_status_counts=dict(statuses))


def evaluate_panel(panel, b_labels):
    support = panel.get("support", {})
    positive = set(map(int, support.get("positive_ids", [])))
    negative = set(map(int, support.get("negative_ids", [])))
    plus = sorted(positive.intersection(b_labels))
    minus = sorted(negative.intersection(b_labels))
    # Verify sealed B subsets independently; never construct a new neighborhood.
    for key, expected in (("positive_B_ids", plus), ("negative_B_ids", minus)):
        if key in support and set(map(int, support[key])) != set(expected):
            raise ValueError("Sealed B-neighborhood IDs disagree with the fixed B role")
    plus0, minus0 = sum(b_labels[i] == 0 for i in plus), sum(b_labels[i] == 0 for i in minus)
    pci, mci = wilson(plus0, len(plus)), wilson(minus0, len(minus))
    counts_ok = min(len(plus), len(minus)) >= 10
    unlabeled_ok = min(len(positive), len(negative)) >= 20
    semantics = None
    if counts_ok:
        if pci[0] > .5 and mci[1] < .5:
            semantics = "correct"
        elif pci[1] < .5 and mci[0] > .5:
            semantics = "reversed"
        elif (pci[0] > .5 and mci[0] > .5) or (pci[1] < .5 and mci[1] < .5):
            semantics = "same_majority"
        else:
            semantics = "label_uncertain"
    ess = list(panel.get("ess", []))
    ess_ok = len(ess) == 3 and all(finite(e) and e >= 5 for e in ess)
    interval = panel.get("V_ci", [None, None])
    valid_interval = len(interval) == 2 and all(finite(x) for x in interval)
    valley = "uncertain"
    if not ess_ok:
        valley = "kernel_concentrated"
    elif valid_interval and interval[0] > math.log(1.1):
        valley = "supported"
    elif valid_interval and interval[1] < 0:
        valley = "opposed"
    eligible = unlabeled_ok and counts_ok and ess_ok and valid_interval
    joint = eligible and semantics in ("correct", "reversed") and valley == "supported"
    return dict(bandwidth_multiplier=panel.get("bandwidth_multiplier"), shoulder_sigma=panel.get("shoulder_sigma"),
                positive_unlabeled_n=len(positive), negative_unlabeled_n=len(negative),
                positive_B_n=len(plus), negative_B_n=len(minus), positive_B_ids=plus, negative_B_ids=minus,
                positive_class0_n=plus0, negative_class0_n=minus0,
                positive_class0_fraction=ratio(plus0, len(plus)), negative_class0_fraction=ratio(minus0, len(minus)),
                positive_wilson95=pci, negative_wilson95=mci, unlabeled_supported=unlabeled_ok,
                B_supported=counts_ok, ess=clean(ess), ess_supported=ess_ok,
                V=panel.get("V"), V_ci=clean(interval), valley_state=valley, label_state=semantics,
                opposite_majorities=semantics in ("correct", "reversed"),
                eligible=eligible, joint_positive=joint, semantic_joint_positive=joint and semantics == "correct")


def cluster_equal_fraction(records, key):
    groups = defaultdict(list)
    for row in records:
        if row.get("cluster_id") is not None:
            groups[row["cluster_id"]].append(float(bool(row[key])))
    return float(np.mean([np.mean(v) for v in groups.values()])) if groups else None


def support_reuse(records):
    counts = Counter()
    for row in records:
        counts.update(set(row.get("positive_B_ids", [])) | set(row.get("negative_B_ids", [])))
    return dict(distinct_B_covered=len(counts), B_reuse_histogram=dict(Counter(counts.values())),
                max_B_reuse=max(counts.values(), default=0),
                max_B_neighborhood_fraction=ratio(max(counts.values(), default=0), len(records)),
                clusters=len({r["cluster_id"] for r in records if r.get("cluster_id") is not None}))


def summarize_bank(records, denominator=64):
    if len(records) != denominator:
        raise ValueError("Failed queries must remain in the fixed 64 denominator")
    eligible = [r for r in records if r.get("eligible", False)]
    reuse = support_reuse(eligible)
    joint = sum(bool(r.get("joint_positive")) for r in eligible)
    semantic = sum(bool(r.get("semantic_joint_positive")) for r in eligible)
    support_ok = len(eligible) >= 16 and reuse["distinct_B_covered"] >= 40 and reuse["clusters"] >= 8
    fraction = ratio(joint, len(eligible))
    reversed_n = sum(r.get("label_state") == "reversed" for r in eligible)
    same_n = sum(r.get("label_state") == "same_majority" for r in eligible)
    return dict(denominator=denominator, projected=sum(bool(r.get("success")) for r in records),
                unlabeled_supported=sum(bool(r.get("unlabeled_supported")) for r in records),
                B_supported=sum(bool(r.get("B_supported")) for r in records), eligible=len(eligible),
                eligible_fraction_total=ratio(len(eligible), denominator), joint_positive=joint,
                joint_positive_fraction_total=ratio(joint, denominator), joint_positive_fraction_eligible=fraction,
                semantic_joint_positive=semantic, semantic_joint_fraction_eligible=ratio(semantic, len(eligible)),
                reversed_semantics=reversed_n, same_majority=same_n,
                valley_supported=sum(r.get("valley_state") == "supported" for r in eligible),
                valley_opposed=sum(r.get("valley_state") == "opposed" for r in eligible),
                label_uncertain=sum(r.get("label_state") == "label_uncertain" for r in eligible),
                enough_support=support_ok, passes=support_ok and fraction >= .5,
                semantically_opposing=support_ok and reversed_n > len(eligible) / 2,
                cluster_equal_joint_fraction=cluster_equal_fraction(eligible, "joint_positive"), **reuse)


def summarize_direction(anchors, scans, sigma, denominator=64):
    if len(anchors) != denominator or len(scans) != denominator:
        raise ValueError("Source queries and scans must retain their fixed denominator")
    scans_by_id = {int(r["origin_id"]): r for r in scans}
    records, original_identifiable, flipped, lost = [], 0, 0, 0
    for anchor in anchors:
        row = dict(anchor)
        scan = scans_by_id[int(anchor["origin_id"])]
        ci = scan.get("sigma_g_ci")
        ess = anchor.get("ess", [None])[0]
        d = direction_from_ci(ci, ess) if scan.get("success") else None
        loo = scan.get("loo") or {}
        loo_ess = loo.get("ess_center", loo.get("ess", [None]))
        if isinstance(loo_ess, (list, tuple, torch.Tensor)):
            loo_ess = loo_ess[0]
        ld = direction_from_ci(loo.get("sigma_g_ci"), loo_ess)
        if d is not None:
            original_identifiable += 1
            flipped += int(ld is not None and ld != d)
            lost += int(ld is None)
        roots = scan.get("roots", [])
        consensus = reference_consensus(roots, sigma) if scan.get("success") else dict(
            status="reference_uncertain", sign=None, architecture_conflict=False,
            denominator=8, usable_count=0, root_status_counts={"source_anchor_or_scan_unavailable": 8})
        eligible = (bool(anchor.get("success")) and bool(anchor.get("unlabeled_supported"))
                    and bool(anchor.get("B_supported")) and d is not None and consensus["status"] == "consensus")
        row.update(density_direction=d, loo_direction=ld, reference_consensus=consensus,
                   g=scan.get("g"), sigma_g_ci=clean(scan.get("sigma_g_ci")),
                   finite_difference_flag=bool(scan.get("fd_flag")), fd_discrepancy=scan.get("fd_discrepancy"),
                   direction_eligible=eligible, compatible=bool(eligible and d == consensus["sign"]),
                   incompatible=bool(eligible and d != consensus["sign"]))
        records.append(row)
    eligible = [r for r in records if r["direction_eligible"]]
    reuse = support_reuse(eligible)
    compatible = sum(r["compatible"] for r in eligible)
    fraction = ratio(compatible, len(eligible))
    support_ok = len(eligible) >= 16 and reuse["distinct_B_covered"] >= 40 and reuse["clusters"] >= 8
    flip_fraction, lost_fraction = ratio(flipped, original_identifiable), ratio(lost, original_identifiable)
    loo_unstable = (finite(flip_fraction) and flip_fraction > .25) or (finite(lost_fraction) and lost_fraction > .25)
    conflict_n = sum(r["reference_consensus"]["architecture_conflict"] for r in records)
    root_statuses = Counter()
    for row in records:
        root_statuses.update(row["reference_consensus"]["root_status_counts"])
    state = "inconclusive"
    if support_ok and not loo_unstable and conflict_n == 0:
        if fraction >= .75:
            state = "support"
        elif fraction <= .25:
            state = "oppose_current_local_configuration"
        else:
            state = "mixed"
    return dict(state=state, denominator=denominator, eligible=len(eligible),
                eligible_fraction_total=ratio(len(eligible), denominator), enough_support=support_ok,
                density_identifiable=original_identifiable, compatible=compatible, incompatible=len(eligible)-compatible,
                compatible_fraction_eligible=fraction, compatible_fraction_total=ratio(compatible, denominator),
                incompatible_fraction_total=ratio(len(eligible)-compatible, denominator),
                reference_consensus=sum(r["reference_consensus"]["status"] == "consensus" for r in records),
                architecture_conflicts=conflict_n, loo_opposite=flipped, loo_unidentifiable=lost,
                finite_difference_flags=sum(r["finite_difference_flag"] for r in records),
                root_status_counts=dict(root_statuses),
                loo_original_identifiable_denominator=original_identifiable,
                loo_opposite_fraction=flip_fraction, loo_unidentifiable_fraction=lost_fraction,
                loo_unstable=loo_unstable, cluster_equal_compatible_fraction=cluster_equal_fraction(eligible, "compatible"),
                **reuse), records


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _metric_row(probabilities, ids, labels_by_id, identity, graph_records, log_probabilities=None):
    ids = list(map(int, ids))
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated evaluation graph IDs")
    labels = torch.tensor([labels_by_id[i] for i in ids], dtype=torch.long)
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    metrics = classification_metrics(p, labels, log_probabilities)
    lp = p.clamp_min(1e-300).log() if log_probabilities is None else torch.as_tensor(log_probabilities, dtype=torch.float64)
    graph_records.append(dict(identity=identity, ids=ids, labels=labels, probabilities=p,
                              predicted=p.argmax(1), nll=-lp[torch.arange(len(ids)), labels]))
    return dict(identity, status="success", **metrics)


def evaluate_predictions(root, plan, b_labels, diagnostic_labels):
    """Evaluate all prespecified fits/endpoints, retaining explicit failed rows."""
    root = Path(root)
    metrics, fit_records, graph_records = [], [], []
    domains = [int(pair[1]) for pair in plan["tasks"]]
    for seed in plan["seeds"]:
        original_path = root / "source" / f"seed{seed}" / "predictions.pt"
        original = _load(original_path) if original_path.exists() else None
        source_ids = sorted(diagnostic_labels[seed])
        for evaluation_domain, ids, labels in [(0, source_ids, diagnostic_labels[seed])] + [
                (domain, sorted(b_labels[domain]), b_labels[domain]) for domain in domains]:
            identity = dict(seed=seed, domain=evaluation_domain, origin="initial", fold=None,
                            feature="H", head="initial", condition="true", endpoint="selected",
                            evaluation="source_diagnostic" if evaluation_domain == 0 else "B")
            if original is None:
                metrics.append(dict(identity, status="missing", **{metric: None for metric in METRICS}))
            else:
                lp = original.get("log_probabilities")
                metrics.append(_metric_row(original["probabilities"][ids], ids, labels, identity, graph_records,
                                           None if lp is None else lp[ids]))
        for origin, origin_domain in [("source", 0)] + [("target", d) for d in domains]:
            origin_path = "source" if origin == "source" else f"target{origin_domain}"
            for fold in range(4):
                for feature in ("H", "SIZE"):
                    for head in ARCHITECTURES:
                        for condition in ("true", "permuted"):
                            path = root / "probes" / origin_path / f"seed{seed}" / f"fold{fold}" / feature / head / condition
                            meta_path = path / "metadata.json"
                            meta = json.loads(meta_path.read_text()) if meta_path.exists() else dict(status="missing")
                            identity = dict(seed=seed, origin=origin, origin_domain=origin_domain,
                                            fold=fold, feature=feature, head=head, condition=condition)
                            fit_records.append(dict(identity, status=meta.get("status"),
                                failure=meta.get("failure", meta.get("reason")), selected_step=meta.get("selected_step"),
                                budget_sensitive=meta.get("budget_sensitive"),
                                train_class_counts=meta.get("train_class_counts"),
                                validation_class_counts=meta.get("validation_class_counts"),
                                prior_probabilities=meta.get("prior_probabilities")))
                            successful = meta.get("status") == "success"
                            predicted = _load(path / "probabilities.pt") if successful else {}
                            evaluation_pools = [(0, "audit", diagnostic_labels[seed])] if origin == "source" else []
                            evaluation_pools += [(d, f"B{d}", b_labels[d]) for d in (domains if origin == "source" else [origin_domain])]
                            for domain, pool, label_map in evaluation_pools:
                                for endpoint in ("selected", "200", "800"):
                                    row_identity = dict(identity, domain=domain, endpoint=endpoint,
                                                        evaluation="source_audit" if domain == 0 else "B")
                                    payload = predicted.get(endpoint, predicted.get(int(endpoint), {}) if endpoint.isdigit() else {}).get(pool)
                                    if payload is None:
                                        metrics.append(dict(row_identity, status=meta.get("status") if not successful else "missing_predictions"))
                                        continue
                                    ids = list(map(int, payload["ids"]))
                                    expected = set(meta["audit_ids"]) if pool == "audit" else set(b_labels[domain])
                                    if set(ids) != expected:
                                        raise ValueError(f"Wrong frozen evaluation IDs: {path}/{pool}/{endpoint}")
                                    metrics.append(_metric_row(payload["probabilities"], ids, label_map, row_identity, graph_records,
                                                               payload.get("log_probabilities")))
                                    if endpoint == "selected":
                                        prior = torch.as_tensor(meta["prior_probabilities"], dtype=torch.float64).clamp(1e-6, 1-1e-6)
                                        if abs(float(prior.sum())-1) > 1e-6:
                                            raise ValueError("Invalid prior probabilities")
                                        prior_identity = dict(row_identity, condition=f"prior_{condition}")
                                        metrics.append(_metric_row(prior.repeat(len(ids), 1), ids, label_map, prior_identity, graph_records))
    families, family_index = [], {}
    grouped = defaultdict(dict)
    keys = ("seed", "domain", "origin", "feature", "head", "condition", "endpoint")
    for row in metrics:
        if row["origin"] != "initial":
            grouped[tuple(row[k] for k in keys)][row["fold"]] = row
    for group_key, fold_rows in sorted(grouped.items()):
        row = dict(zip(keys, group_key))
        for metric in METRICS:
            values = [fold_rows.get(fold, {}).get(metric) for fold in range(4)]
            aggregate = strict_fold_summary(values)
            row[metric] = aggregate["mean"]
            row[f"{metric}_available_mean"] = aggregate["available_mean"]
            row[f"{metric}_available_folds"] = aggregate["available_count"]
            row[f"{metric}_fold_values"] = aggregate["values"]
        families.append(row)
        family_index[group_key] = row
    source_initial = {(r["seed"], r["domain"]): r for r in metrics if r["origin"] == "initial"}
    comparisons = []
    for seed in plan["seeds"]:
        for domain in domains:
            for head in ARCHITECTURES:
                for feature in ("H", "SIZE"):
                    for endpoint in ("selected", "200", "800"):
                        target = family_index.get((seed, domain, "target", feature, head, "true", endpoint), {})
                        baselines = dict(permuted=family_index.get((seed, domain, "target", feature, head, "permuted", endpoint), {}),
                                         source_probe=family_index.get((seed, domain, "source", feature, head, "true", endpoint), {}),
                                         source_initial=source_initial[seed, domain])
                        if endpoint == "selected":
                            baselines["prior"] = family_index.get((seed, domain, "target", feature, head, "prior_true", endpoint), {})
                        if feature == "H":
                            baselines["size"] = family_index.get((seed, domain, "target", "SIZE", head, "true", endpoint), {})
                        for baseline_name, baseline in baselines.items():
                            comparisons.append(dict(seed=seed, domain=domain, head=head, feature=feature,
                                                    endpoint=endpoint, baseline=baseline_name,
                                                    **{f"delta_{m}": difference(target.get(m), baseline.get(m)) for m in METRICS}))
    return dict(metrics=metrics, fits=fit_records, families=families, family_index=family_index,
                source_initial=source_initial, comparisons=comparisons, per_graph=graph_records)


def _bank_rows(bank, b_labels, panel_index):
    rows = []
    for anchor in bank["anchors"]:
        row = dict(origin_id=int(anchor["origin_id"]), success=bool(anchor.get("success")),
                   failure=anchor.get("failure"), cluster_id=anchor.get("cluster_id"))
        if row["success"]:
            if panel_index == "loo":
                panel = anchor.get("loo")
            else:
                panels = anchor.get("panels", [])
                panel = panels[panel_index] if len(panels) > panel_index else None
            if panel is not None:
                row.update(evaluate_panel(panel, b_labels))
        rows.append(row)
    return rows


def evaluate_geometry(root, plan, b_labels):
    root = Path(root)
    summaries, directions, per_query, coverage_summaries = [], [], {}, []
    for seed in plan["seeds"]:
        for _, domain in plan["tasks"]:
            path = root / "geometry" / f"seed{seed}_domain{domain}"
            cell_key = f"seed{seed}_domain{domain}"
            per_query[cell_key] = {}
            coverage_path = path / "coverage.pt"
            if coverage_path.exists():
                covered = _load(coverage_path)
                size_bins = {int(k): int(v) for k, v in covered.get("size_quartile_by_id", {}).items()}
                for head_key, observations in covered["heads"].items():
                    strata = [("all_target", observations)]
                    strata += [(f"B_class{label}", [r for r in observations if b_labels[domain].get(int(r["origin_id"])) == label]) for label in (0, 1)]
                    if size_bins:
                        strata += [(f"size_quartile{q+1}", [r for r in observations if size_bins[int(r["origin_id"])] == q]) for q in range(4)]
                        strata += [(f"B_class{label}_size_quartile{q+1}", [r for r in observations
                            if b_labels[domain].get(int(r["origin_id"])) == label and size_bins[int(r["origin_id"])] == q])
                            for label in (0, 1) for q in range(4)]
                    for stratum, selected in strata:
                        valid = [r for r in selected if r["success"]]
                        distances = [float(r["distance_sigma"]) for r in valid if finite(r.get("distance_sigma"))]
                        residuals = [float(r["residual"]) for r in valid if finite(r.get("residual"))]
                        coverage_summaries.append(dict(seed=seed, domain=domain, bank=head_key, stratum=stratum,
                            denominator=len(selected), projected=len(valid), coverage=ratio(len(valid), len(selected)),
                            failures=dict(Counter(r.get("failure", "unknown") for r in selected if not r["success"])),
                            analytic_numerical_misses=sum(r.get("analytic", {}).get("numerical_miss", False) for r in selected),
                            analytic_outside_radius=sum(r.get("analytic", {}).get("status") == "outside_radius" for r in selected),
                            distance_sigma_median=float(np.median(distances)) if distances else None,
                            distance_sigma_p95=float(np.quantile(distances, .95)) if distances else None,
                            residual_median=float(np.median(residuals)) if residuals else None,
                            size_quartile_edges=clean(covered.get("size_quartile_edges"))))
            source_rows, sigma = None, None
            for key in ["source"] + [f"{arch}_fold{fold}" for arch in ("linear", "hidden32_silu_mlp") for fold in range(4)]:
                file_path = path / "banks" / f"{key}.pt"
                if not file_path.exists():
                    for panel in [0, 1, 2, 3, 4, "loo"]:
                        summaries.append(dict(seed=seed, domain=domain, bank=key, panel=str(panel), status="missing", passes=False))
                    continue
                bank = _load(file_path)
                sigma = bank.get("sigma", sigma)
                for panel in [0, 1, 2, 3, 4, "loo"]:
                    rows = _bank_rows(bank, b_labels[domain], panel)
                    per_query[cell_key][f"{key}_panel{panel}"] = rows
                    identity = dict(seed=seed, domain=domain, bank=key, panel=str(panel),
                                    head=architecture(bank.get("architecture", key.split("_fold")[0])), fold=bank.get("fold"))
                    summaries.append(dict(identity, status="success", **summarize_bank(rows)))
                    if key == "source" and panel == 0:
                        source_rows = rows
            scans_path = path / "source_scans.pt"
            if source_rows is not None and scans_path.exists():
                scans = _load(scans_path)
                sigma = scans.get("sigma", sigma)
                if finite(sigma) and sigma > 0:
                    summary, rows = summarize_direction(source_rows, scans["records"], sigma)
                    directions.append(dict(seed=seed, domain=domain, **summary))
                    per_query[cell_key]["source_directions"] = rows
                else:
                    directions.append(dict(seed=seed, domain=domain, state="inconclusive", reason="degenerate_sigma"))
            else:
                directions.append(dict(seed=seed, domain=domain, state="inconclusive", reason="missing_geometry"))
    return dict(banks=summaries, directions=directions, per_query=per_query, coverage=coverage_summaries)


def screening(plan, predictions, geometry):
    families = predictions["family_index"]
    initial = predictions["source_initial"]
    bank_map = {(r["seed"], r["domain"], r["bank"], r["panel"]): r for r in geometry["banks"]}
    direction_map = {(r["seed"], r["domain"]): r for r in geometry["directions"]}
    cells, tasks = [], []
    for seed in plan["seeds"]:
        for _, domain in plan["tasks"]:
            cell_architectures = []
            for head in ARCHITECTURES:
                true = families.get((seed, domain, "target", "H", head, "true", "selected"), {})
                permuted = families.get((seed, domain, "target", "H", head, "permuted", "selected"), {})
                ba, pba = true.get("balanced_accuracy"), permuted.get("balanced_accuracy")
                true_folds = true.get("balanced_accuracy_fold_values", [None]*4)
                perm_folds = permuted.get("balanced_accuracy_fold_values", [None]*4)
                positive_folds = sum(finite(a) and finite(b) and a > b for a, b in zip(true_folds, perm_folds))
                readability = finite(ba) and finite(pba) and ba >= .55 and ba-pba >= .05 and positive_folds >= 3
                headroom = difference(ba, initial[seed, domain].get("balanced_accuracy"))
                bank_arch = "hidden32_silu_mlp" if head == "mlp" else head
                banks = [bank_map.get((seed, domain, f"{bank_arch}_fold{fold}", "0"), {}) for fold in range(4)]
                passed_banks = sum(bool(b.get("passes")) for b in banks)
                opposing_banks = sum(bool(b.get("semantically_opposing")) for b in banks)
                direction = direction_map[seed, domain]
                budget_sensitive = any(f.get("budget_sensitive") for f in predictions["fits"]
                    if f["seed"] == seed and f["origin"] == "target" and f["origin_domain"] == domain
                    and f["feature"] == "H" and f["head"] == head and f["condition"] == "true")
                cell_architectures.append(dict(seed=seed, domain=domain, head=head,
                    target_BA=ba, source_BA=initial[seed, domain]["balanced_accuracy"], target_permuted_BA=pba,
                    true_minus_permuted_BA=difference(ba, pba), positive_folds=positive_folds,
                    readability=bool(readability), headroom_BA=headroom,
                    observed_headroom=finite(headroom) and headroom >= .01,
                    delta_NLL=difference(true.get("nll"), initial[seed, domain].get("nll")),
                    density_joint_fraction_mean=strict_fold_summary([b.get("joint_positive_fraction_eligible") for b in banks])["mean"],
                    density_joint_fraction_available_mean=strict_fold_summary([b.get("joint_positive_fraction_eligible") for b in banks])["available_mean"],
                    density_passing_banks=passed_banks, density_opposing_banks=opposing_banks,
                    density_positive=bool(readability and passed_banks >= 3),
                    direction_state=direction["state"], budget_sensitive=bool(budget_sensitive)))
            for row in cell_architectures:
                other = next(r for r in cell_architectures if r["head"] != row["head"])
                row["other_architecture_opposing"] = other["density_opposing_banks"] >= 3
                row["qualifies_for_R2_screen"] = (row["readability"] and row["density_positive"]
                    and row["observed_headroom"] and row["direction_state"] == "support"
                    and not row["other_architecture_opposing"])
                row["state"] = "support" if row["qualifies_for_R2_screen"] else (
                    "oppose_current_local_configuration" if row["direction_state"] == "oppose_current_local_configuration" else "inconclusive")
                row["unmet"] = [key for key in ("readability", "density_positive", "observed_headroom") if not row[key]]
                if row["direction_state"] != "support":
                    row["unmet"].append("source_direction_"+row["direction_state"])
                if row["other_architecture_opposing"]:
                    row["unmet"].append("architecture_conflict")
                cells.append(row)
    for _, domain in plan["tasks"]:
        for head in ARCHITECTURES:
            rows = [r for r in cells if r["domain"] == domain and r["head"] == head]
            delta = nll_screen([r["delta_NLL"] for r in rows], len(plan["seeds"]))
            nll_positive = delta["positive_seeds"]
            nll_block = delta["systematic_worsening"]
            passing = sum(r["qualifies_for_R2_screen"] for r in rows)
            tasks.append(dict(domain=domain, head=head, seeds=len(rows), qualifying_seeds=passing,
                readability_seeds=sum(r["readability"] for r in rows), density_positive_seeds=sum(r["density_positive"] for r in rows),
                headroom_seeds=sum(r["observed_headroom"] for r in rows),
                direction_support_seeds=sum(r["direction_state"] == "support" for r in rows),
                direction_opposition_seeds=sum(r["direction_state"] == "oppose_current_local_configuration" for r in rows),
                mean_delta_NLL=delta["mean"], positive_delta_NLL_seeds=nll_positive,
                systematic_NLL_worsening=bool(nll_block), NLL_evidence_complete=delta["complete"],
                recommend_register_R2=passing >= 4 and delta["passes"],
                state="support" if passing >= 4 and delta["passes"] else "inconclusive"))
    return cells, tasks


def aggregate_seeds(plan, predictions):
    rows = []
    grouped = defaultdict(dict)
    keys = ("domain", "origin", "feature", "head", "condition", "endpoint")
    for family in predictions["families"]:
        grouped[tuple(family[k] for k in keys)][family["seed"]] = family
    for group, seed_rows in sorted(grouped.items()):
        row = dict(zip(keys, group))
        for metric in METRICS:
            values = [seed_rows.get(seed, {}).get(metric) for seed in plan["seeds"]]
            summary = seed_summary(values, len(plan["seeds"]))
            for key, value in summary.items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    # Equal-weight task mean WITHIN each shared source-model seed, before seed SD.
    cross = defaultdict(dict)
    domains = [d for _, d in plan["tasks"]]
    for family in predictions["families"]:
        if family["domain"] in domains:
            key = tuple(family[k] for k in keys if k != "domain")
            cross[key][family["seed"], family["domain"]] = family
    for group, members in sorted(cross.items()):
        row = dict(zip([k for k in keys if k != "domain"], group), domain="equal_weight_tasks_within_seed")
        for metric in METRICS:
            values = []
            for seed in plan["seeds"]:
                domain_values = [members.get((seed, d), {}).get(metric) for d in domains]
                values.append(float(np.mean(domain_values)) if all(finite(v) for v in domain_values) else None)
            for key, value in seed_summary(values, len(plan["seeds"])).items():
                row[f"{metric}_{key}"] = value
        rows.append(row)
    return rows


def verify_observations_sealed(root, plan, seal):
    """Require the complete existing observation tree to precede B access."""
    root = Path(root)
    if seal.get("stage") != "S2":
        raise PermissionError("Independent evaluation requires S2")
    files = seal.get("files", {})
    for directory in ("source", "probes", "geometry", "inputs", "roles"):
        for path in (root / directory).rglob("*"):
            if path.is_file() and str(path.relative_to(root)) not in files:
                raise PermissionError(f"Observation not included in S2: {path.relative_to(root)}")
    for required in ("roles.json", "seals/S1.json"):
        if required not in files:
            raise PermissionError(f"S2 missing required dependency: {required}")
    # A runner-level failure can explain absent artifacts, but a successful phase
    # must have one explicit terminal record for each planned model/geometry cell.
    failures = list((root / "failures").glob("*.json"))
    if not failures:
        required = []
        for seed in plan["seeds"]:
            required.append(f"source/seed{seed}/predictions.pt")
            for origin_path in ["source"] + [f"target{d}" for _, d in plan["tasks"]]:
                for fold in range(4):
                    for feature in ("H", "SIZE"):
                        for head in ARCHITECTURES:
                            for condition in ("true", "permuted"):
                                required.append(f"probes/{origin_path}/seed{seed}/fold{fold}/{feature}/{head}/{condition}/metadata.json")
            for _, domain in plan["tasks"]:
                prefix = f"geometry/seed{seed}_domain{domain}"
                required += [f"{prefix}/coverage.pt", f"{prefix}/source_scans.pt"]
                required += [f"{prefix}/banks/{key}.pt" for key in ["source"] + [
                    f"{arch}_fold{fold}" for arch in ("linear", "hidden32_silu_mlp") for fold in range(4)]]
        missing = [path for path in required if path not in files]
        if missing:
            raise PermissionError(f"S2 is incomplete: {len(missing)} required observations missing, first={missing[0]}")


def make_figures(root, plan, predictions, geometry, public):
    """Static report figures; no graph labels or per-graph probabilities exported."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, public = Path(root), Path(public)
    figures = []
    families = predictions["family_index"]
    fig, axes = plt.subplots(len(plan["tasks"]), 2, figsize=(12, 4*len(plan["tasks"])), squeeze=False)
    for ti, (_, domain) in enumerate(plan["tasks"]):
        for ai, head in enumerate(ARCHITECTURES):
            ax = axes[ti, ai]
            series = {"Source initial": [predictions["source_initial"][s, domain].get("balanced_accuracy") for s in plan["seeds"]]}
            for label, origin, feature, condition in [("Target H", "target", "H", "true"),
                    ("Source H probe", "source", "H", "true"), ("Target SIZE", "target", "SIZE", "true"),
                    ("Permuted H", "target", "H", "permuted"), ("Prior", "target", "H", "prior_true")]:
                series[label] = [families.get((s, domain, origin, feature, head, condition, "selected"), {}).get("balanced_accuracy") for s in plan["seeds"]]
            for label, values in series.items():
                ax.plot(plan["seeds"], [v if finite(v) else np.nan for v in values], "o-", label=label, markersize=4)
            ax.axhline(.5, color="gray", linestyle=":")
            ax.set(title=f"0 to {domain}: {head}, four-fold means", ylabel="Balanced accuracy", xlabel="Source split + model seed")
            ax.set_xticks(plan["seeds"])
            ax.legend(fontsize=7)
    fig.tight_layout()
    name = "probe-balanced-accuracy.png"
    fig.savefig(public/name, dpi=150)
    plt.close(fig)
    figures.append(name)

    fig, axes = plt.subplots(2*len(plan["tasks"]), 2, figsize=(12, 7*len(plan["tasks"])), squeeze=False)
    for di, (_, domain) in enumerate(plan["tasks"]):
        for ai, head in enumerate(ARCHITECTURES):
            train_ax, val_ax = axes[2*di, ai], axes[2*di+1, ai]
            for label, origin, condition in [("Target H", f"target{domain}", "true"),
                    ("Permuted target H", f"target{domain}", "permuted"), ("Source H", "source", "true")]:
                by_step = {"training": defaultdict(list), "validation": defaultdict(list)}
                for seed in plan["seeds"]:
                    for fold in range(4):
                        path = root/"probes"/origin/f"seed{seed}"/f"fold{fold}"/"H"/head/condition/"history.json"
                        if path.exists():
                            history = json.loads(path.read_text())
                            for key, ce_key in (("training", "train_ce"), ("validation", "validation_ce")):
                                for item in history.get(key, []):
                                    by_step[key][item["step"]].append(item[ce_key])
                for ax, key in [(train_ax, "training"), (val_ax, "validation")]:
                    steps = sorted(by_step[key])
                    if steps:
                        ax.plot(steps, [np.mean(by_step[key][s]) for s in steps], label=label)
            train_ax.set(title=f"0 to {domain}: {head}, optimization118", ylabel="Mean training CE", xlabel="Update")
            val_ax.set(title=f"0 to {domain}: {head}, inner validation29", ylabel="Mean validation CE", xlabel="Update")
            for ax in (train_ax, val_ax):
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
    fig.suptitle("Descriptive mean learning curves across available folds/seeds; selection remains per fit", y=.999)
    fig.tight_layout()
    name = "probe-learning-curves.png"
    fig.savefig(public/name, dpi=150)
    plt.close(fig)
    figures.append(name)

    for seed in plan["seeds"]:
        for _, domain in plan["tasks"]:
            path = root / "geometry" / f"seed{seed}_domain{domain}" / "source_scans.pt"
            if not path.exists():
                continue
            saved = _load(path)
            sigma = saved.get("sigma")
            evaluated = {r["origin_id"]: r for r in geometry["per_query"].get(f"seed{seed}_domain{domain}", {}).get("source_directions", [])}
            # The first eight fixed hash-order queries are shown, including every
            # failure. No label-dependent display selection or replacement.
            fig, axes = plt.subplots(8, 3, figsize=(15, 23), squeeze=False)
            for rank in range(8):
                axs = axes[rank]
                if rank >= len(saved["records"]):
                    axs[0].text(.1, .5, "Missing frozen query record")
                    continue
                scan = saved["records"][rank]
                if not scan.get("success") or not finite(sigma):
                    for ax in axs:
                        ax.text(.05, .5, f"Query rank {rank+1}: {scan.get('failure', 'missing')}", wrap=True)
                        ax.set_axis_off()
                    continue
                t = torch.as_tensor(scan["grid_t"]).numpy()/sigma
                axs[0].plot(t, torch.as_tensor(scan["source_margin"]).numpy(), "k-", label="Source")
                for key, values in scan.get("reference_margin", {}).items():
                    axs[0].plot(t, torch.as_tensor(values).numpy(), alpha=.65, linewidth=.8,
                                color="tab:blue" if key.startswith("linear") else "tab:orange")
                axs[0].axhline(0, color="gray", linestyle=":")
                axs[0].set_ylabel(f"Query {rank+1}\nMargin (class 0 - 1)")
                axs[1].plot(t, torch.as_tensor(scan["logp"]).numpy(), color="tab:green")
                axs[1].set_ylabel("Empirical log KDE")
                row = evaluated.get(int(scan["origin_id"]), {})
                for coordinate, prefix in [(-1, "negative"), (1, "positive")]:
                    value = row.get(f"{prefix}_class0_fraction")
                    interval = row.get(f"{prefix}_wilson95", [None, None])
                    if finite(value):
                        axs[2].errorbar([coordinate], [value], yerr=validated_nonnegative_errorbars(value, interval), fmt="o", color="tab:purple")
                        axs[2].annotate(f"n={row.get(prefix+'_B_n')}", (coordinate, value), xytext=(4, 5), textcoords="offset points", fontsize=8)
                axs[2].set(xlim=(-4, 4), ylim=(-.03, 1.03), ylabel="Real B-neighbor class-0 fraction")
                axs[2].axhline(.5, color="gray", linestyle=":")
                for ax in axs:
                    ax.axvline(0, color="gray", linestyle=":")
                    ax.set_xlabel("t / sigma along the fixed source normal")
            axes[0,0].set_title("Source (black), LR (blue), MLP (orange)")
            axes[0,1].set_title("Density at fixed virtual scan points")
            axes[0,2].set_title("Labels of real neighbors only; Wilson 95%")
            fig.suptitle(f"Seed {seed}, 0 to {domain}: first 8 preselected queries", y=.999)
            fig.tight_layout()
            name = f"source-scans-seed{seed}-domain{domain}.png"
            fig.savefig(public/name, dpi=120)
            plt.close(fig)
            figures.append(name)
    return figures


def evaluate_run(root, plan):
    """Open B only after complete S2 verification; write immutable S3 inputs."""
    from experiments.mutagenicity.representation_boundary_support_20261111.src.data import LabelBroker
    root = Path(root)
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    public = out / "public"
    public.mkdir(exist_ok=False)
    seal = json.loads((root / "seals/S2.json").read_text())
    verify_observations_sealed(root, plan, seal)
    broker = LabelBroker(root, plan)
    # Explicitly verify every sealed byte before opening the first evaluation
    # label; the broker repeats/caches integrity verification on each read.
    broker._seal("S2", seal["digest"])
    roles = json.loads((root / "roles.json").read_text())
    b_labels, diagnostic_labels = {}, {}
    for _, domain in plan["tasks"]:
        ids = roles["targets"][str(domain)]["B"]
        values = broker.read(ids, role="target_B", purpose="R1 fixed post-S2 evaluation", seal=seal["digest"], domain=domain)
        b_labels[domain] = dict(zip(map(int, ids), map(int, values)))
    for seed in plan["seeds"]:
        ids = json.loads((root / "roles" / f"seed{seed}.json").read_text())["diagnostic_ids"]
        values = broker.read(ids, role="source_diagnostic", purpose="R1 held-out source audit", seal=seal["digest"], seed=seed)
        diagnostic_labels[seed] = dict(zip(map(int, ids), map(int, values)))
    predictions = evaluate_predictions(root, plan, b_labels, diagnostic_labels)
    geometry = evaluate_geometry(root, plan, b_labels)
    cells, tasks = screening(plan, predictions, geometry)
    aggregate = aggregate_seeds(plan, predictions)
    outputs = dict(graph_metrics=predictions["metrics"], probe_fits=predictions["fits"],
                   probe_families=predictions["families"], comparisons=predictions["comparisons"],
                   seed_aggregates=aggregate, density_banks=geometry["banks"],
                   source_directions=geometry["directions"], coverage=geometry["coverage"],
                   screening_cells=cells, screening_tasks=tasks)
    for name, rows in outputs.items():
        write_csv(public / f"{name}.csv", rows)
        write_json(public / f"{name}.json", clean(rows))
    save_tensor(out / "per_graph.pt", predictions["per_graph"])
    save_tensor(out / "per_query.pt", geometry["per_query"])
    source_descriptives = []
    for seed in plan["seeds"]:
        path = root / "source" / f"seed{seed}" / "descriptives.json"
        if path.exists():
            source_descriptives.append(dict(seed=seed, domains=json.loads(path.read_text())))
    write_json(public / "size_descriptives.json", source_descriptives)
    figures = make_figures(root, plan, predictions, geometry, public)
    success = sum(row["status"] == "success" for row in predictions["fits"])
    summary = dict(protocol_id=plan["protocol_id"], S2_digest=seal["digest"],
        source_models=sum((root/"source"/f"seed{s}"/"predictions.pt").exists() for s in plan["seeds"]),
        probes_expected=len(predictions["fits"]), probes_successful=success,
        probes_missing_or_failed=len(predictions["fits"])-success,
        budget_sensitive_fits=sum(bool(row["budget_sensitive"]) for row in predictions["fits"]),
        tasks=tasks, figures=figures, final_labels_read=False, score_trained=False, classifier_adapted=False,
        evaluation_unit="graph; four folds averaged within architecture/seed; five shared source-model seeds",
        bootstrap_interpretation="conditional empirical-density sensitivity, not oracle truth or independent repeats",
        warning="B and A are development data; screening is descriptive, not a significance test",
        public_artifacts="aggregate metrics only; per-graph labels/probabilities and per-query IDs stay in local archive")
    write_json(public / "summary.json", clean(summary))
    write_json(out / "summary.json", clean(summary))
    broker._seal("S2", seal["digest"])
    return summary
