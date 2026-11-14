"""R1 frozen boundary observations, with no label lookup or adaptation.

Coverage calls the production projector unchanged. Analytic observers diagnose
its failures; they never replace a failed production anchor in the main banks.
"""
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch

from tsba.boundary import margin_and_normal, project
from tsba.classifier import ResidualClassifier
from tsba.config import Config
from tsba.score import bandwidth
from experiments.common.io import save_tensor, write_json
from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.density import GaussianKDE, bootstrap_counts, derived_seed, finite_support, valley_observation


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _log(log, message):
    if callable(log):
        log(message)
    elif log is not None:
        log.write(message + "\n")
        log.flush()


def _cell_path(root, seed, domain):
    return Path(root) / "geometry" / f"seed{seed}_domain{domain}"


def _assert_phase(root, expected):
    present = [stage for stage in ("S1", "S2", "S3") if (Path(root) / "seals" / f"{stage}.json").exists()]
    actual = present[-1] if present else "R0"
    if actual != expected:
        raise PermissionError(f"Geometry requires phase {expected}, actual {actual}; no observations after S2")


def _source(root, seed):
    frozen = _load(Path(root) / "source" / f"seed{seed}" / "frozen.pt")
    x = frozen["x"].to(dtype=torch.float64, device="cpu")
    model = ResidualClassifier(x.shape[1], 2, 16, dtype=torch.float64)
    model.load_state_dict(frozen["initial_head"])
    if bool((model.U2.weight != 0).any()):
        raise ValueError("R1 requires the unadapted linear source head (U2=0)")
    model.eval().requires_grad_(False)
    return x, model


def _roles(root, domain):
    with open(Path(root) / "roles.json") as handle:
        return json.load(handle)["targets"][str(domain)]


def fixed_query_ids(A_ids, protocol_id, domain, count=64):
    return sorted((int(i) for i in A_ids), key=lambda i: (
        hashlib.sha256(f"{protocol_id}|geometry|{domain}|{i}".encode("utf-8")).hexdigest(), i))[:count]


def _config(plan, seed=0):
    geometry = plan["geometry"]
    return Config(seed=seed, emb_dim=16, projection_steps=geometry["projection_steps"],
                  projection_backtracks=geometry["projection_backtracks"],
                  min_normal=geometry["min_normal"], root_tolerance=geometry["root_tolerance_sigma"])


# Count executed production lines without copying or altering the solver.
_PROJECT_LINES, _PROJECT_START = inspect.getsourcelines(project)
_ITERATION_LINE = next(_PROJECT_START + i for i, s in enumerate(_PROJECT_LINES)
                       if "margin, gradient, _ = margin_and_normal" in s)
_BACKTRACK_LINE = next(_PROJECT_START + i for i, s in enumerate(_PROJECT_LINES)
                       if "candidate = point + (0.5 ** back)" in s)


def observed_project(model, x, sigma, cfg):
    counts = {"iterations": 0, "backtrack_attempts": 0, "backtrack_reductions": 0}

    def local(frame, event, arg):
        if event == "line":
            if frame.f_lineno == _ITERATION_LINE:
                counts["iterations"] += 1
            elif frame.f_lineno == _BACKTRACK_LINE:
                counts["backtrack_attempts"] += 1
                counts["backtrack_reductions"] += int(frame.f_locals["back"] > 0)
        return local

    def trace(frame, event, arg):
        return local if frame.f_code is project.__code__ else None

    old_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        result, reason = project(model, x, (0, 1), 4 * sigma, cfg.root_tolerance * sigma, cfg)
    finally:
        sys.settrace(old_trace)
    return result, reason, counts


def affine_margin(model, dim):
    zero = torch.zeros(dim, dtype=torch.float64)
    margin, gradient, _ = margin_and_normal(model, zero, (0, 1))
    return gradient, float(margin)


def analytic_projection(x, w, b, sigma, min_normal=1e-8):
    norm = float(w.norm())
    if norm <= min_normal:
        return {"status": "degenerate", "distance": None, "point": None}
    margin = x @ w + b
    distance = float(margin.abs()) / norm
    status = ("outside_radius" if distance > 4 * sigma + 1e-8 * sigma else
              "inside_radius" if distance < 4 * sigma - 1e-8 * sigma else "radius_edge")
    return {"status": status, "distance": distance, "point": x - margin * w / norm**2}


def coverage(model, x, ids, sigma, cfg, is_linear=False, deadline=None):
    """One record per original graph, preserving all production failures."""
    affine = affine_margin(model, x.shape[1]) if is_linear else None
    rows = []
    for graph_id, value in zip(ids, x):
        row = {"origin_id": int(graph_id), "success": False, "failure": None,
               "point": None, "normal": None, "normal_norm": None, "margin": None,
               "residual": None, "distance_sigma": None, "radius_ratio": None}
        if deadline is not None and time.monotonic() >= deadline:
            row["failure"] = "cell_time_limit"
            rows.append(row)
            continue
        try:
            initial_margin, initial_normal, _ = margin_and_normal(model, value, (0, 1))
            row.update(initial_margin=float(initial_margin), initial_normal_norm=float(initial_normal.norm()))
            projected, reason, counts = observed_project(model, value, sigma, cfg)
            row.update(counts)
            row["failure"] = reason
            if projected is not None:
                point, margin, normal, norm = projected
                row.update(success=True, point=point, normal=normal, normal_norm=float(norm), margin=float(margin),
                           residual=float(margin.abs() / norm), distance_sigma=float((point - value).norm()) / sigma,
                           radius_ratio=float((point - value).norm()) / (4 * sigma))
            if affine is not None:
                analytic = analytic_projection(value, *affine, sigma, cfg.min_normal)
                analytic["numerical_miss"] = not row["success"] and analytic["status"] == "inside_radius"
                analytic["projection_error_sigma"] = (float((row["point"] - analytic["point"]).norm()) / sigma
                                                        if row["success"] and analytic["point"] is not None else None)
                row["analytic"] = analytic
        except (RuntimeError, ValueError, FloatingPointError) as error:
            row["failure"] = "numerical_error"
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def _coverage_summary(rows):
    return {"n": len(rows), "success": sum(r["success"] for r in rows),
            "failures": dict(Counter(r["failure"] for r in rows if not r["success"])),
            "analytic_numerical_misses": sum(r.get("analytic", {}).get("numerical_miss", False) for r in rows)}


def run_source_coverage(root, seed, domain, plan, log=None):
    """Source-only prerequisite for S1; never loads a probe or target label."""
    _assert_phase(root, "R0")
    started = time.monotonic()
    cpu_started = time.process_time()
    path = _cell_path(root, seed, domain)
    path.mkdir(parents=True, exist_ok=True)
    x, model = _source(root, seed)
    roles = _roles(root, domain)
    ids = roles["ids"]
    sigma = bandwidth(x[ids], 1.0)
    queries = fixed_query_ids(roles["A"], plan["protocol_id"], domain, plan["geometry"]["queries_per_bank"])
    if "query_ids" in roles and roles["query_ids"] != queries:
        raise ValueError("materialized fixed geometry IDs disagree with protocol hash")
    limit = plan["resources"]["max_geometry_density_cell_seconds"]
    rows = (coverage(model, x[ids], ids, sigma, _config(plan, seed), True, started + limit)
            if sigma is not None else _missing_rows(ids, 'degenerate_bandwidth'))
    result = {"schema_version": 1, "seed": seed, "domain": domain, "sigma": sigma,
              "target_ids": ids, "query_ids": queries, "rows": rows,
              "elapsed_seconds": time.monotonic() - started, "cpu_seconds": time.process_time() - cpu_started}
    sizes = _load(Path(root) / "inputs" / "size.pt")[ids, 0].double()
    edges = torch.quantile(sizes, sizes.new_tensor([.25, .5, .75]))
    result.update(size_quartile_edges=edges,
                  size_quartile_by_id={int(i): int(q) for i, q in zip(ids, torch.bucketize(sizes, edges, right=False))},
                  size_basis="log1p_nodes")
    save_tensor(path / "source_coverage.pt", result)
    summary = {k: result[k] for k in ("seed", "domain", "sigma", "elapsed_seconds", "cpu_seconds")}
    summary.update(_coverage_summary(rows))
    summary["status"] = ("degenerate_bandwidth" if sigma is None else
                         "cell_time_limit" if any(r["failure"] == "cell_time_limit" for r in rows) else "completed")
    write_json(path / "source_summary.json", summary)
    _log(log, f"source coverage seed={seed} domain={domain}: {summary['success']}/{len(ids)}, sigma={sigma}")
    return summary


def cluster_anchors(anchors, sigma):
    """Greedy distance clusters in ascending origin-ID order, protocol §5.2."""
    representatives, assignments = [], {}
    for record in sorted(anchors, key=lambda r: r["origin_id"]):
        if not record["success"]:
            record["cluster_id"] = None
            continue
        candidates = [(float((record["point"] - point).norm()), rep_id)
                      for rep_id, point in representatives]
        close = [item for item in candidates if item[0] <= .1 * sigma]
        rep = min(close)[1] if close else record["origin_id"]
        if not close:
            representatives.append((rep, record["point"]))
        record["cluster_id"] = rep
        assignments.setdefault(rep, []).append(record["origin_id"])
    return {"n_clusters": len(representatives), "members": assignments,
            "max_cluster_weight": max((len(v) for v in assignments.values()), default=0),
            "n_success": sum(len(v) for v in assignments.values()), "n_predeclared": len(anchors)}


def margins_and_gradients(model, points):
    with torch.enable_grad():
        values = points.detach().requires_grad_(True)
        output = model(values)
        margins = output[:, 0] - output[:, 1]
        gradients, = torch.autograd.grad(margins.sum(), values)
    return margins.detach(), gradients.detach()


def scan_reference(model, point, normal, sigma, plan, is_linear=False):
    """Detect intersections on the registered grid; preserve ambiguity states."""
    geometry = plan["geometry"]
    grid = torch.linspace(-4 * sigma, 4 * sigma, geometry["scan_grid"], dtype=torch.float64)
    points = point + grid[:, None] * normal
    margins, gradients = margins_and_gradients(model, points)
    norms = gradients.norm(dim=-1)
    residuals = margins.abs() / norms.clamp_min(torch.finfo(torch.float64).tiny)
    result = {"grid_margin": margins, "grid_normal_norm": norms,
              "grid_min_residual": float(residuals.min()), "t": None, "roots": [],
              "status": None, "near_coincident": False}
    if not bool(torch.isfinite(margins).all() and torch.isfinite(gradients).all()):
        result["status"] = "nonfinite"
        return result
    if bool((norms <= geometry["min_normal"]).all()):
        result["status"] = "degenerate_normal"
        return result
    if is_linear:
        w, b = affine_margin(model, point.numel())
        norm = float(w.norm())
        if norm <= geometry["min_normal"]:
            result["status"] = "degenerate_normal"
            return result
        slope = float(w @ normal)
        intercept = float(w @ point + b)
        result["analytic_slope_relative"] = abs(slope) / norm
        if abs(slope) / norm <= geometry["lr_ray_parallel_relative_tolerance"]:
            result["status"] = ("line_near_coincident" if abs(intercept) / norm <= 1e-5 * sigma else "parallel")
            return result
        result["analytic_t"] = -intercept / slope
        result["analytic_intersects_scan"] = abs(result["analytic_t"]) <= 4 * sigma
    candidates, brackets = [], []
    for i in range(len(grid) - 1):
        if float(margins[i]) * float(margins[i + 1]) < 0:
            brackets.append((float(grid[i]), float(grid[i + 1]), float(margins[i])))
    near_grid = torch.nonzero(((residuals <= geometry["root_tolerance_sigma"] * sigma)
                             & (norms > geometry["min_normal"])) | (margins == 0)).flatten().tolist()
    crossing_grid = [i for i in near_grid if i > 0 and i + 1 < len(grid)
                     and float(margins[i - 1]) * float(margins[i + 1]) < 0]
    candidates.extend(float(grid[i]) for i in near_grid)
    for left, right, left_value in brackets:
        for _ in range(geometry["bisection_steps"]):
            if right - left <= geometry["bisection_width_sigma"] * sigma:
                break
            mid = (left + right) / 2
            with torch.no_grad():
                logits = model((point + mid * normal).unsqueeze(0))[0]
                value = float(logits[0] - logits[1])
            if value == 0:
                left = right = mid
                break
            if value * left_value > 0:
                left, left_value = mid, value
            else:
                right = mid
        candidates.append((left + right) / 2)
    merged = []
    for candidate in sorted(candidates):
        if not merged or candidate - merged[-1] > geometry["root_merge_distance_sigma"] * sigma:
            merged.append(candidate)
        else:
            # Keep the lower-residual representative, deterministically.
            with torch.no_grad():
                logits = model(torch.stack((point + merged[-1] * normal, point + candidate * normal)))
                if abs(float(logits[1, 0] - logits[1, 1])) < abs(float(logits[0, 0] - logits[0, 1])):
                    merged[-1] = candidate
    result["roots"] = merged
    result["n_brackets"] = len(brackets)
    if not merged:
        result["status"] = "no_detected_root"
        return result
    if len(merged) > 1:
        result["status"] = "multiple_detected_roots"
        return result
    t = merged[0]
    _, gradient, _ = margin_and_normal(model, point + t * normal, (0, 1))
    norm = float(gradient.norm())
    result["root_normal_norm"] = norm
    if norm <= geometry["min_normal"]:
        result["status"] = "degenerate_normal"
        return result
    if not is_linear and not brackets and not crossing_grid:
        result["status"] = "possible_touch"
        return result
    tangent = abs(float(gradient @ normal)) / norm
    result["root_ray_slope_relative"] = tangent
    if not is_linear and tangent <= geometry["mlp_ray_tangent_relative_tolerance"]:
        result["status"] = "near_tangent"
        return result
    result["t"] = t
    result["near_coincident"] = abs(t) <= geometry["near_coincident_sigma"] * sigma
    result["status"] = "near_coincident" if result["near_coincident"] else "detected_unique"
    if is_linear:
        result["analytic_error_sigma"] = abs(t - result["analytic_t"]) / sigma
    return result


def reference_consensus(roots, sigma, plan):
    valid = [r for r in roots if r["status"] == "detected_unique" and r.get("t") is not None]
    architecture_sign = {}
    for architecture in ("linear", "hidden32_silu_mlp"):
        votes = [1 if r["t"] > 0 else -1 for r in valid if r["architecture"] == architecture]
        balance = sum(votes)
        architecture_sign[architecture] = (1 if balance > 0 else -1) if balance else None
    conflict = all(v is not None for v in architecture_sign.values()) and len(set(architecture_sign.values())) > 1
    counts = {sign: sum((1 if r["t"] > 0 else -1) == sign for r in valid) for sign in (-1, 1)}
    sign = max(counts, key=lambda s: (counts[s], s))
    group = [r["t"] for r in valid if (1 if r["t"] > 0 else -1) == sign]
    iqr = float(torch.quantile(torch.tensor(group, dtype=torch.float64), .75)
                - torch.quantile(torch.tensor(group, dtype=torch.float64), .25)) / sigma if group else None
    supported = (not conflict and counts[sign] >= plan["geometry"]["reference_consensus_min_heads"]
                 and iqr is not None and iqr <= plan["geometry"]["reference_root_iqr_sigma"])
    return {"status": "consensus" if supported else "reference_uncertain", "sign": sign if supported else None,
            "count": counts[sign], "counts_by_sign": counts, "denominator": 8,
            "iqr_sigma": iqr, "architecture_conflict": conflict, "architecture_sign": architecture_sign}


def _missing_rows(ids, reason):
    return [{"origin_id": int(i), "success": False, "failure": reason, "point": None,
             "normal": None, "cluster_id": None} for i in ids]


def run_cell(root, seed, domain, plan, log=None):
    """Complete all frozen observations; output tensors contain no labels."""
    _assert_phase(root, "S1")
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.probes import load_reference
    started = time.monotonic()
    cpu_started = time.process_time()
    path = _cell_path(root, seed, domain)
    if not (path / "source_coverage.pt").exists():
        raise FileNotFoundError(f"Pre-S1 source coverage dependency missing: {path / 'source_coverage.pt'}; see initial-coverage failure record")
    source_data = _load(path / "source_coverage.pt")
    sigma = source_data["sigma"]
    source_seconds = source_data["elapsed_seconds"]
    deadline = started + max(0, plan["resources"]["max_geometry_density_cell_seconds"] - source_seconds)
    (path / "banks").mkdir(exist_ok=True)
    x_global, source = _source(root, seed)
    roles = _roles(root, domain)
    ids, queries = roles["ids"], source_data["query_ids"]
    x = x_global[ids]
    models = {"source": source}
    metadata = {"source": {"head_key": "source", "architecture": "source", "fold": None}}
    for head in ("linear", "hidden32_silu_mlp"):
        for fold in range(4):
            key = f"{head}_fold{fold}"
            models[key] = load_reference(root, seed, domain, fold, "mlp" if head == "hidden32_silu_mlp" else head)
            if models[key] is not None:
                models[key].eval().requires_grad_(False)
            metadata[key] = {"head_key": key, "architecture": head, "fold": fold}
    coverage_data = {"source": source_data["rows"]}
    for key, model in models.items():
        if key == "source":
            continue
        if sigma is None or model is None:
            rows = _missing_rows(ids, "degenerate_bandwidth" if sigma is None else "missing_reference_head")
        else:
            rows = coverage(model, x, ids, sigma, _config(plan, seed), metadata[key]["architecture"] == "linear", deadline)
        coverage_data[key] = rows
        _log(log, f"reference coverage seed={seed} domain={domain} head={key}: {sum(r['success'] for r in rows)}/{len(ids)}")
    save_tensor(path / "coverage.pt", {"schema_version": 1, "sigma": sigma, "target_ids": ids,
                                      "heads": coverage_data, "metadata": metadata,
                                      **{k: source_data[k] for k in ("size_quartile_edges", "size_quartile_by_id", "size_basis")}})
    if sigma is None:
        for key in models:
            save_tensor(path / "banks" / f"{key}.pt", {**metadata[key], "sigma": None, "origin_ids": queries,
                        "target_ids": ids, "anchors": _missing_rows(queries, "degenerate_bandwidth"),
                        "clusters": {"n_clusters": 0, "members": {}, "max_cluster_weight": 0}})
        save_tensor(path / "source_scans.pt", {"sigma": None, "records": _missing_rows(queries, "degenerate_bandwidth")})
        summary = {"status": "degenerate_bandwidth", "seed": seed, "domain": domain, "sigma": None}
        write_json(path / "summary.json", summary)
        return summary
    density_seed = derived_seed(plan["protocol_id"], "density_bootstrap", seed, domain)
    counts, draw_indices = bootstrap_counts(len(ids), plan["density"]["bootstrap_repeats"], density_seed, return_indices=True)
    save_tensor(path / "bootstrap.pt", {"seed": density_seed, "counts": counts, "draw_indices": draw_indices, "target_ids": ids,
                                      "shared_across": ["head", "query", "panel"], "sigma": sigma})
    kde_models = {v: GaussianKDE(x, v * sigma, plan["density"]["chunk"]) for v in (.5, 1., 2.)}
    banks, errors = {}, []
    index = {graph_id: position for position, graph_id in enumerate(ids)}
    for key, rows in coverage_data.items():
        lookup = {r["origin_id"]: r for r in rows}
        anchors = [dict(lookup[q]) if q in lookup else _missing_rows([q], "degenerate_bandwidth")[0] for q in queries]
        clusters = cluster_anchors(anchors, sigma)
        for anchor in anchors:
            anchor["panels"] = []
            anchor["loo"] = None
            if not anchor["success"]:
                continue
            if time.monotonic() >= deadline:
                anchor["density_failure"] = "cell_time_limit"
                continue
            try:
                point, normal = anchor["point"], anchor["normal"]
                for specification in plan["density"]["panels"]:
                    multiplier, shoulder = specification["bandwidth_multiplier"], specification["shoulder_sigma"]
                    panel = valley_observation(kde_models[multiplier], point, normal, sigma, shoulder, counts)
                    panel.update(specification)
                    panel["support"] = finite_support(x, ids, anchor["origin_id"], point, normal, sigma, shoulder,
                                                       roles["B"], plan["density"]["min_unlabeled_per_side"],
                                                       plan["density"]["min_B_per_side"])
                    anchor["panels"].append(panel)
                loo = valley_observation(kde_models[1.], point, normal, sigma, 1., counts, index[anchor["origin_id"]])
                loo.update(bandwidth_multiplier=1., shoulder_sigma=1., support=anchor["panels"][0]["support"])
                anchor["loo"] = loo
            except (RuntimeError, ValueError, FloatingPointError) as error:
                anchor["density_failure"] = f"{type(error).__name__}: {error}"
                errors.append({"head": key, "origin_id": anchor["origin_id"], "error": anchor["density_failure"]})
        bank = {**metadata[key], "schema_version": 1, "sigma": sigma, "origin_ids": queries, "target_ids": ids,
                "coverage_file": "../coverage.pt", "anchors": anchors, "clusters": clusters}
        save_tensor(path / "banks" / f"{key}.pt", bank)
        banks[key] = bank
        _log(log, f"density bank seed={seed} domain={domain} head={key}: {sum(bool(a['panels']) for a in anchors)}/64")
    scans = []
    for anchor in banks["source"]["anchors"]:
        row = {"origin_id": anchor["origin_id"], "success": False, "failure": anchor.get("failure"), "roots": []}
        if not anchor["success"] or not anchor["panels"]:
            row["failure"] = row["failure"] or anchor.get("density_failure", "density_missing")
            scans.append(row)
            continue
        if time.monotonic() >= deadline:
            row["failure"] = "cell_time_limit"
            scans.append(row)
            continue
        try:
            point, normal = anchor["point"], anchor["normal"]
            grid = torch.linspace(-4 * sigma, 4 * sigma, plan["geometry"]["scan_grid"], dtype=torch.float64)
            points = point + grid[:, None] * normal
            with torch.no_grad():
                logits = source(points)
                source_margin = logits[:, 0] - logits[:, 1]
            profile = kde_models[1.].evaluate(points)
            roots, reference_margins = [], {}
            for key, model in models.items():
                if key == "source":
                    continue
                if model is None:
                    observation = {"status": "missing_reference_head", "t": None, "roots": []}
                else:
                    observation = scan_reference(model, point, normal, sigma, plan,
                                                 metadata[key]["architecture"] == "linear")
                    reference_margins[key] = observation.pop("grid_margin")
                observation.update(metadata[key])
                roots.append(observation)
            main = anchor["panels"][0]
            epsilon = plan["density"]["finite_difference_epsilon_sigma"] * sigma
            fd_logp = kde_models[1.].evaluate(torch.stack((point + epsilon * normal, point - epsilon * normal)))["logp"][0]
            g_fd = float(fd_logp[0] - fd_logp[1]) / (2 * epsilon)
            difference = abs(g_fd - main["g"])
            fields = ("g", "g_bootstrap", "sigma_g_ci", "direction_identifiable", "density_direction", "ess_center")
            row.update({k: main[k] for k in fields})
            row.update(success=True, failure=None, grid_t=grid, source_margin=source_margin,
                       reference_margin=reference_margins, logp=profile["logp"][0], ess=profile["ess"][0],
                       roots=roots, consensus=reference_consensus(roots, sigma, plan),
                       loo={k: anchor["loo"][k] for k in fields} if anchor["loo"] else None,
                       g_finite_difference=g_fd, fd_discrepancy=difference,
                       fd_flag=difference > .01 * max(abs(main["g"]), 1 / sigma))
            for item in row["roots"]:
                item["alignment"] = ((-main["g"]) * item["t"] > 0
                                     if main["direction_identifiable"] and item["status"] == "detected_unique" else None)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            row["failure"] = f"{type(error).__name__}: {error}"
            errors.append({"phase": "scan", "origin_id": anchor["origin_id"], "error": row["failure"]})
        scans.append(row)
    save_tensor(path / "source_scans.pt", {"schema_version": 1, "sigma": sigma, "records": scans,
                                          "origin_ids": queries, "target_ids": ids})
    summary = {"schema_version": 1, "seed": seed, "domain": domain, "sigma": sigma,
               "status": "cell_time_limit" if time.monotonic() >= deadline else "completed_with_failures" if errors else "completed",
               "elapsed_seconds": time.monotonic() - started + source_seconds,
               "cpu_seconds": time.process_time() - cpu_started + source_data["cpu_seconds"],
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "bootstrap_repeats": len(counts), "kde_query_count": sum(k.query_count for k in kde_models.values()),
               "coverage": {key: _coverage_summary(rows) for key, rows in coverage_data.items()},
               "banks": {key: {"success": sum(a["success"] for a in bank["anchors"]),
                               "density_observed": sum(len(a["panels"]) == 5 for a in bank["anchors"]),
                               "n_clusters": bank["clusters"]["n_clusters"]} for key, bank in banks.items()},
               "source_scans_success": sum(r["success"] for r in scans), "errors": errors}
    write_json(path / "summary.json", summary)
    _log(log, f"geometry/density completed seed={seed} domain={domain}: {summary['status']}, {summary['elapsed_seconds']:.1f}s")
    return summary
