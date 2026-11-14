"""Research-only radius/bandwidth diagnostics on immutable R1 anchors.

The scanner reads only copied frozen coordinates, geometry, role IDs and the old
bootstrap draws. Class labels are accepted solely by the post-OBS evaluator.
No projection, fit, query selection or scale selection takes place here.
"""
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import resource
import time

import torch

from experiments.common.io import save_tensor, write_json
from experiments.mutagenicity.r1_20261111.src.density import GaussianKDE, valley_observation
from experiments.common.statistics import clean, ratio; from experiments.mutagenicity.r1_20261111.src.evaluate import evaluate_panel, seed_summary, summarize_bank


SETTINGS = ((1., 1.), (1.5, 1.), (2., 1.), (1., 1.5), (1., 2.), (2., 2.))
BANKS = ("source",) + tuple(f"{head}_fold{fold}" for head in
                          ("linear", "hidden32_silu_mlp") for fold in range(4))
DENOMINATOR = 64


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _settings(plan):
    settings = plan["scales"]["settings"]
    pairs = tuple((float(s["support_multiplier"]), float(s["bandwidth_multiplier"]))
                  for s in settings)
    if pairs != SETTINGS or plan["scales"]["bootstrap_repeats"] != 200:
        raise ValueError("scales_require_six_registered_settings_and_200_old_draws")
    return [dict(setting_id=f"r{r:g}_b{b:g}", support_multiplier=r,
                 bandwidth_multiplier=b, shoulder_sigma=1.) for r, b in pairs]


def _emit(log, value):
    if callable(log):
        log(value)
    elif log is not None:
        log.write(json.dumps(value) + "\n")
        log.flush()


def _limits(started, plan):
    resources = plan.get("resources", {})
    limit = resources.get("max_scale_cell_seconds",
                          resources.get("max_geometry_density_cell_seconds", 600))
    if time.monotonic() - started > limit:
        raise RuntimeError("scale_cell_time_limit")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > min(
            resources.get("max_rss_bytes", 8 * 1024**3), 8 * 1024**3):
        raise RuntimeError("scale_rss_limit")


def scaled_support(x, ids, origin_id, point, normal, sigma0, support_multiplier, B_ids):
    """Change the tangent radius alone; retain original half-open axial slabs.

    Distances are in frozen embedding coordinates. Counts are distinct graph
    counts, and the originating graph is excluded on both sides at every scale.
    """
    ids = torch.as_tensor(ids, dtype=torch.long)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("duplicate_target_ids")
    delta = x - point
    axial = delta @ normal
    tangent = (delta - axial[:, None] * normal).norm(dim=-1)
    distance = delta.norm(dim=-1)
    eligible = (tangent <= 4 * sigma0 * support_multiplier) & (ids != origin_id)
    masks = (("positive", eligible & (axial >= .5 * sigma0) & (axial < 1.5 * sigma0)),
             ("negative", eligible & (axial > -1.5 * sigma0) & (axial <= -.5 * sigma0)))
    B = set(map(int, B_ids))
    result = dict(target_pool_n=len(ids), independent_pool_n=len(ids) - int(origin_id in ids),
                  tangent_radius=4 * sigma0 * support_multiplier,
                  axial_slab_sigma0=[.5, 1.5], origin_excluded=True)
    union = set()
    for side, mask in masks:
        selected = ids[mask].tolist()
        union.update(selected)
        result[side + "_ids"] = selected
        result[side + "_B_ids"] = [i for i in selected if i in B]
        result[side + "_n"] = len(selected)
        result[side + "_B_n"] = len(result[side + "_B_ids"])
        result[side + "_coverage_fraction_target_pool"] = ratio(len(selected), len(ids))
        result[side + "_coverage_fraction_independent_pool"] = ratio(len(selected), result["independent_pool_n"])
        for name, values in (("distance", distance), ("tangent", tangent), ("axial", axial)):
            result[side + "_" + name] = values[mask].clone()
            if name != "axial":
                result[side + "_" + name + "_median"] = float(values[mask].quantile(.5)) if selected else None
                result[side + "_" + name + "_max"] = float(values[mask].max()) if selected else None
    result["union_n"] = len(union)
    result["coverage_fraction_target_pool"] = ratio(len(union), len(ids))
    result["coverage_fraction_independent_pool"] = ratio(len(union), result["independent_pool_n"])
    result["unlabeled_supported"] = all(result[s + "_n"] >= 20 for s in ("positive", "negative"))
    result["B_supported"] = all(result[s + "_B_n"] >= 10 for s in ("positive", "negative"))
    return result


def density_at_frozen_queries(x, point, normal, sigma0, bandwidth_multiplier, counts,
                              origin_index, leave_origin_out=False):
    """Reuse R1 KDE and valley rules without moving the center or shoulders."""
    kde = GaussianKDE(x, sigma0 * bandwidth_multiplier)
    panel = valley_observation(kde, point, normal, sigma0, 1., counts,
                              origin_index if leave_origin_out else None)
    log_weights = panel["log_kernel"].clone()
    if leave_origin_out:
        log_weights[:, origin_index] = -torch.inf
    weights = torch.softmax(log_weights, dim=-1)
    panel.update(max_kernel_weight=weights.max(-1).values,
                 origin_kernel_weight=weights[:, origin_index].clone(),
                 sigma0=sigma0, bandwidth=sigma0 * bandwidth_multiplier,
                 bandwidth_multiplier=bandwidth_multiplier, shoulder_sigma=1.,
                 pool="leave_origin_out" if leave_origin_out else "full_target_pool")
    # The reusable kernel matrix is transient; all observations and bootstrap
    # values remain archived without duplicating a query-by-pool matrix per arm.
    del panel["log_kernel"]
    return panel


def _parity(actual, previous):
    """Audit the old baseline when it was estimable, without replacing failures."""
    if previous is None:
        return dict(status="old_observation_unavailable")
    errors = {}
    for key in ("logp", "ess", "V", "V_bootstrap", "V_ci", "g", "g_bootstrap", "sigma_g_ci", "query_points"):
        if key not in previous:
            return dict(status="old_field_unavailable", field=key)
        left, right = torch.as_tensor(actual[key]), torch.as_tensor(previous[key])
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
        errors[key] = float((left - right).abs().max())
    return dict(status="matches", max_absolute_errors=errors)


def observe_anchor(x, ids, B_ids, anchor, sigma0, counts, settings, check_limits=None, dependency_failure=None):
    """Observe one frozen anchor, retaining failures independently per setting."""
    row = {k: copy.deepcopy(anchor[k]) for k in (
        "origin_id", "success", "failure", "point", "normal", "normal_norm", "margin", "residual",
        "distance_sigma", "radius_ratio", "cluster_id") if k in anchor}
    row["settings"] = []
    density_cache, support_cache = {}, {}
    for specification in settings:
        observation = dict(specification, support=None, main=None, loo=None,
                           support_failure=None, main_failure=None, loo_failure=None)
        row["settings"].append(observation)
        if dependency_failure or not row.get("success"):
            observation["failure"] = dependency_failure or row.get("failure") or "frozen_projection_failed"
            continue
        try:
            if check_limits is not None:
                check_limits()
            point, normal = row["point"].double(), row["normal"].double()
            if not (torch.isfinite(point).all() and torch.isfinite(normal).all()):
                raise ValueError("nonfinite_frozen_anchor")
            if not math.isclose(float(normal.norm()), 1., rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError("frozen_normal_not_unit")
            origin_index = ids.index(int(row["origin_id"]))
        except Exception as exc:
            observation["failure"] = f"{type(exc).__name__}: {exc}"
            continue
        r, b = specification["support_multiplier"], specification["bandwidth_multiplier"]
        if r not in support_cache:
            try:
                support_cache[r] = (scaled_support(x, ids, row["origin_id"], point, normal, sigma0, r, B_ids), None)
            except Exception as exc:
                support_cache[r] = (None, f"{type(exc).__name__}: {exc}")
        observation["support"], observation["support_failure"] = support_cache[r]
        for pool in ("main", "loo"):
            if (b, pool) not in density_cache:
                try:
                    if check_limits is not None:
                        check_limits()
                    panel = density_at_frozen_queries(x, point, normal, sigma0, b, counts, origin_index,
                                                     leave_origin_out=pool == "loo")
                    if b == 1:
                        old = (anchor.get("loo") if pool == "loo" else
                               next((p for p in anchor.get("panels", []) if
                                     p.get("bandwidth_multiplier") == 1 and p.get("shoulder_sigma") == 1), None))
                        panel["old_baseline_parity"] = _parity(panel, old)
                    density_cache[b, pool] = (panel, None)
                except Exception as exc:
                    density_cache[b, pool] = (None, f"{type(exc).__name__}: {exc}")
            observation[pool], observation[pool + "_failure"] = density_cache[b, pool]
        observation["failure"] = observation["support_failure"] or observation["main_failure"]
    return row


def _failed_anchors(query_ids, failure):
    return [dict(origin_id=i, success=False, failure=failure, point=None, normal=None, cluster_id=None)
            for i in query_ids]


def _bank_summary(bank, settings):
    rows = []
    for index, specification in enumerate(settings):
        observations = [a["settings"][index] for a in bank["anchors"]]
        failures = Counter(o.get("failure") for o in observations if o.get("failure"))
        rows.append(dict(seed=bank["seed"], domain=bank["domain"], head_key=bank["head_key"],
            **specification, denominator=DENOMINATOR, projected=sum(a["success"] for a in bank["anchors"]),
            main_observed=sum(o["main"] is not None for o in observations),
            loo_observed=sum(o["loo"] is not None for o in observations),
            unlabeled_supported=sum(bool((o["support"] or {}).get("unlabeled_supported")) for o in observations),
            B_supported=sum(bool((o["support"] or {}).get("B_supported")) for o in observations),
            ess_supported=sum(bool((o["main"] or {}).get("kernel_supported")) for o in observations),
            dependency_failed=bool(bank.get("failure")),
            observation_failures=sum(bool(a.get("success") and (o.get("failure") or o.get("loo_failure")))
                                     for a,o in zip(bank["anchors"],observations)),
            failure_counts=dict(failures),
            loo_failure_counts=dict(Counter(o["loo_failure"] for o in observations if o["loo_failure"]))))
    return rows


def run_scales(root, plan, log=None):
    """Run the registered label-free scan using only copied immutable deps."""
    root = Path(root)
    if (root / "seals/OBS.json").exists():
        raise PermissionError("scales_are_frozen_after_OBS")
    settings = _settings(plan)
    deps = root / "deps"
    roles = json.loads((deps / "roles.json").read_text())
    summaries = []
    for seed in plan["seeds"]:
        source_error, features = None, None
        try:
            features = _load(deps / "source" / f"seed{seed}" / "frozen.pt")["x"].double()
            if features.ndim != 2 or not torch.isfinite(features).all():
                raise ValueError("invalid_frozen_coordinates")
        except Exception as exc:
            source_error = f"frozen_dependency:{type(exc).__name__}: {exc}"
        for domain in plan.get("domains", (1, 3)):
            started = time.monotonic()
            domain = int(domain)
            target = roles["targets"][str(domain)]
            ids = list(map(int, target["ids"]))
            old_cell = deps / "geometry" / f"seed{seed}_domain{domain}"
            directory = root / "scales" / f"seed{seed}_domain{domain}"
            directory.mkdir(parents=True, exist_ok=False)
            dependency_error, bootstrap, counts = source_error, None, None
            try:
                if source_error:
                    raise ValueError(source_error)
                if len(set(ids)) != len(ids) or not set(target["B"]).issubset(ids):
                    raise ValueError("invalid_target_role_ids")
                bootstrap = _load(old_cell / "bootstrap.pt")
                counts = bootstrap["counts"].double()
                if list(bootstrap["target_ids"]) != ids or counts.shape != (200, len(ids)):
                    raise ValueError("old_bootstrap_shape_or_target_order_mismatch")
                if not torch.isfinite(counts).all() or (counts < 0).any() or not torch.equal(counts, counts.floor()):
                    raise ValueError("invalid_old_bootstrap_counts")
                if not bool((counts.sum(-1) == len(ids)).all()):
                    raise ValueError("old_bootstrap_counts_do_not_resample_entire_pool")
                sigma0 = float(bootstrap["sigma"])
                if not math.isfinite(sigma0) or sigma0 <= 0:
                    raise ValueError("invalid_frozen_sigma0")
                x = features[ids]
            except Exception as exc:
                dependency_error = f"scale_dependency:{type(exc).__name__}: {exc}"
                sigma0, x = None, None
            for key in BANKS:
                old_bank, error, bank_valid = None, dependency_error, False
                query_ids = list(target.get("query_ids", [None] * DENOMINATOR))
                try:
                    old_bank = _load(old_cell / "banks" / f"{key}.pt")
                    query_ids = list(old_bank["origin_ids"])
                    if len(query_ids) != DENOMINATOR or len(set(query_ids)) != DENOMINATOR:
                        raise ValueError("frozen_bank_must_retain_64_distinct_queries")
                    if "query_ids" in target and query_ids != list(target["query_ids"]):
                        raise ValueError("frozen_query_order_differs_from_roles")
                    if not set(query_ids).issubset(target["A"]):
                        raise ValueError("frozen_query_outside_A")
                    if list(old_bank["target_ids"]) != ids:
                        raise ValueError("frozen_bank_target_order_mismatch")
                    if [a["origin_id"] for a in old_bank["anchors"]] != query_ids:
                        raise ValueError("frozen_anchor_order_mismatch")
                    if error is None and float(old_bank["sigma"]) != sigma0:
                        raise ValueError("frozen_bank_sigma_mismatch")
                    bank_valid = True
                except Exception as exc:
                    error = f"bank_dependency:{type(exc).__name__}: {exc}"
                if len(query_ids) != DENOMINATOR:
                    query_ids = [None] * DENOMINATOR
                anchors = (old_bank["anchors"] if bank_valid else _failed_anchors(query_ids, error))
                observed = [observe_anchor(x, ids, target["B"], a, sigma0, counts, settings,
                                            lambda: _limits(started, plan), dependency_failure=error) for a in anchors]
                bank = dict(schema_version=1, seed=seed, domain=domain, head_key=key, sigma0=sigma0,
                    target_ids=ids, origin_ids=query_ids, denominator=DENOMINATOR,
                    source_bank=str((old_cell / "banks" / f"{key}.pt").relative_to(root)),
                    bootstrap_file=str((old_cell / "bootstrap.pt").relative_to(root)),
                    settings=settings, anchors=observed, failure=error,
                    clusters=(old_bank or {}).get("clusters", {}), elapsed_cell_seconds=time.monotonic() - started,
                    bootstrap_repeats=200, frozen_queries=True, labels_used=False)
                save_tensor(directory / f"{key}.pt", bank)
                summary = _bank_summary(bank, settings)
                write_json(directory / f"{key}.json", clean(summary))
                summaries.extend(summary)
                _emit(log, dict(stage="scales", seed=seed, domain=domain, head_key=key,
                                main_observations=sum(s["main_observed"] for s in summary),
                                denominator=DENOMINATOR * len(settings), failure=error))
    write_json(root / "scales/index.json", clean(summaries))
    return summaries


def _verify_sealed(root):
    path = root / "seals/OBS.json"
    if not path.exists():
        raise PermissionError("scale_evaluation_requires_OBS")
    # Reuse the run-level seal verifier, including its canonical digest and
    # traversal checks, before accepting any externally supplied class map.
    from experiments.mutagenicity.r1b_20261111.src.common import verify_seal
    seal = verify_seal(root, "OBS")
    for path in [root / "deps/roles.json", *sorted((root / "scales").rglob("*"))]:
        if path.is_file() and str(path.relative_to(root)) not in seal["files"]:
            raise PermissionError("scale_observation_not_sealed:" + str(path.relative_to(root)))
    return seal


def _groups(summaries):
    """Sum descriptive denominators; average reference banks within each seed.

    No anchor or overlapping reference fold is treated as an independent
    statistical replicate. Across-seed summaries use the within-seed means.
    """
    grouped = defaultdict(list)
    for row in summaries:
        head = row["head_key"]
        groups = ["source"] if head == "source" else ["reference_all", "linear" if head.startswith("linear") else "mlp"]
        for group in groups:
            grouped[row["seed"], row["domain"], group, row["pool"], row["setting_id"]].append(row)
    outputs = []
    count_keys = ("denominator", "projected", "unlabeled_supported", "B_supported", "eligible",
                  "joint_positive", "semantic_joint_positive", "reversed_semantics", "same_majority",
                  "valley_supported", "valley_opposed", "label_uncertain")
    fraction_keys = ("eligible_fraction_total", "joint_positive_fraction_total", "joint_positive_fraction_eligible",
                     "semantic_joint_fraction_eligible", "cluster_equal_joint_fraction")
    for (seed, domain, group, pool, setting_id), banks in sorted(grouped.items()):
        row = dict(seed=seed, domain=domain, group=group, pool=pool, setting_id=setting_id,
                   support_multiplier=banks[0]["support_multiplier"], bandwidth_multiplier=banks[0]["bandwidth_multiplier"],
                   banks=len(banks), failing_banks=sum(bool(b["failure_counts"]) for b in banks))
        row.update({k: sum(b[k] for b in banks) for k in count_keys})
        row["conditional_pooled_denominator"] = row["eligible"]
        row["joint_fraction_pooled_eligible"] = ratio(row["joint_positive"], row["eligible"])
        for key in fraction_keys:
            values = [b[key] for b in banks]
            available = [v for v in values if v is not None]
            row[key + "_bank_mean"] = sum(available) / len(banks) if len(available) == len(banks) else None
            row[key + "_available_bank_mean"] = sum(available) / len(available) if available else None
            row[key + "_available_banks"] = len(available)
        outputs.append(row)
    return outputs


def _seed_aggregates(groups, plan):
    grouped = defaultdict(dict)
    for row in groups:
        grouped[row["domain"], row["group"], row["pool"], row["setting_id"]][row["seed"]] = row
    outputs = []
    for (domain, group, pool, setting_id), cells in sorted(grouped.items()):
        row = dict(domain=domain, group=group, pool=pool, setting_id=setting_id)
        for metric in ("eligible_fraction_total_bank_mean", "joint_positive_fraction_total_bank_mean",
                       "joint_positive_fraction_eligible_bank_mean", "semantic_joint_fraction_eligible_bank_mean"):
            summary = seed_summary([cells.get(seed, {}).get(metric) for seed in plan["seeds"]], len(plan["seeds"]))
            row.update({metric + "_" + key: value for key, value in summary.items()})
        outputs.append(row)
    return outputs


def evaluate_scales(root, plan, b_labels, log=None):
    """Apply existing Wilson/valley rules after OBS; never choose a scale."""
    root = Path(root)
    seal = _verify_sealed(root)
    settings = _settings(plan)
    roles = json.loads((root / "deps/roles.json").read_text())
    domains = list(map(int, plan.get("domains", (1, 3))))
    if set(b_labels) != set(domains):
        raise PermissionError("scale_evaluation_requires_only_declared_B_domains")
    for domain in domains:
        if set(b_labels[domain]) != set(roles["targets"][str(domain)]["B"]):
            raise PermissionError("scale_evaluation_requires_exact_B_role_ids")
        if any(v not in (0, 1) for v in b_labels[domain].values()):
            raise ValueError("scale_evaluation_requires_binary_classes")
    summaries, queries = [], []
    for seed in plan["seeds"]:
        for domain in domains:
            for key in BANKS:
                bank = _load(root / "scales" / f"seed{seed}_domain{domain}" / f"{key}.pt")
                for setting_index, specification in enumerate(settings):
                    for pool in ("main", "loo"):
                        rows = []
                        for anchor in bank["anchors"]:
                            observation = anchor["settings"][setting_index]
                            row = dict(origin_id=anchor["origin_id"], cluster_id=anchor.get("cluster_id"),
                                success=bool(anchor["success"]), failure=(observation.get("support_failure") or
                                    observation.get(pool + "_failure") or (observation.get("failure") if observation.get("support") is None else None)))
                            panel = observation.get(pool)
                            if observation.get("support") is not None:
                                row.update(evaluate_panel(dict(panel or {}, support=observation["support"]), b_labels[domain]))
                                support = observation["support"]
                                row.update({k: support[k] for k in support if k.endswith(("_median", "_max")) or
                                            "coverage_fraction" in k or k in ("positive_ids", "negative_ids")})
                                if panel is not None:
                                    row.update(g=panel["g"], sigma_g_ci=panel["sigma_g_ci"],
                                               max_kernel_weight=panel["max_kernel_weight"],
                                               origin_kernel_weight=panel["origin_kernel_weight"])
                            rows.append(row)
                        identity = dict(seed=seed, domain=domain, head_key=key, pool=pool, **specification)
                        summary = dict(identity, **summarize_bank(rows, DENOMINATOR),
                                       failure_counts=dict(Counter(r["failure"] for r in rows if r["failure"])))
                        summaries.append(summary)
                        queries.extend(dict(r, **identity) for r in rows)
                _emit(log, dict(stage="evaluate_scales", seed=seed, domain=domain, head_key=key))
    groups = _groups(summaries)
    aggregates = _seed_aggregates(groups, plan)
    summary = dict(OBS_digest=seal.get("digest"), bank_settings=len(summaries),
                   banks_expected=len(plan["seeds"]) * len(domains) * len(BANKS),
                   anchors_per_bank=DENOMINATOR, settings=len(settings), pools=["main", "loo"],
                   scale_selection_performed=False, conditional_zero_denominator="null (NA)",
                   inference="descriptive frozen-scale diagnostics; no B-based scale selection")
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    for name, value in (("scales_banks", summaries), ("scales_queries", queries), ("scales_groups", groups),
                        ("scales_seed_aggregates", aggregates), ("scales_summary", summary)):
        write_json(out / f"{name}.json", clean(value))
    return dict(banks=summaries, queries=queries, groups=groups, seed_aggregates=aggregates, summary=summary)
