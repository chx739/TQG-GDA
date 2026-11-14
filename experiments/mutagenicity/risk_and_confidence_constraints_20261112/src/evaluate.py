"""Score the frozen R3 predictions and publish the complete paired grid."""
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from tsba.classifier import ParameterMap
from experiments.common.statistics import classification_metrics; from experiments.mutagenicity.representation_boundary_support_20261111.src.evaluate import seed_summary
from experiments.mutagenicity.score_budget_and_adaptation_20261111.src.common import atomic_tensor, load, model_from_frozen, seal
from experiments.mutagenicity.score_budget_and_adaptation_20261111.src.evaluate import write_csv, write_json
from experiments.mutagenicity.risk_and_confidence_constraints_20261112.src.protocol import domain_ids, grid, roles_for, target_values
from experiments.mutagenicity.risk_and_confidence_constraints_20261112.src.training import score_dir, source_dir
from experiments.mutagenicity.risk_and_confidence_constraints_20261112.src.trajectory import predictions


def pools_for(root, seed, domain):
    frozen = load(source_dir(root, seed) / "frozen.pt")
    roles = roles_for(root, seed)
    x = frozen["x"]
    pools = dict(source_fit=x[roles["fit"]], source_validation=x[roles["validation"]],
                 target=x[domain_ids(root, domain)])
    return ParameterMap(model_from_frozen(frozen)), pools


def replay(root, plan):
    completed = 0
    verified = 0
    side_count = 0
    side_statuses = Counter()
    for seed in plan["seeds"]:
        for domain in plan["domains"]:
            params, pools = pools_for(root, seed, domain)
            for job in grid(plan):
                if (job.seed, job.domain) != (seed, domain):
                    continue
                out = root / "trajectories" / job.key
                if not (out / "complete.json").is_file():
                    raise ValueError(f"Incomplete trajectory: {job.key}")
                previous = params.vector()
                for attempt in range(plan["adapt"]["adapt_rounds"] + 1):
                    record = load(out / f"round{attempt:02d}.pt")
                    if attempt:
                        torch.testing.assert_close(record["before"], previous, rtol=0, atol=0)
                        if not record["accepted"]:
                            torch.testing.assert_close(record["after"], previous, rtol=0, atol=0)
                    previous = record["after"]
                    if attempt in plan["endpoints"]:
                        actual = predictions(params, previous, pools)
                        for name, value in actual.items():
                            torch.testing.assert_close(value, record["predictions"][name], rtol=0, atol=0)
                        verified += 1
                    side = record.get("side", {})
                    if attempt in plan["side_attempts"]:
                        if not side:
                            raise ValueError(f"Missing side status: {job.key} attempt {attempt}")
                        side_statuses[side["status"]] += 1
                    if side.get("status") == "available":
                        if set(side["candidates"]) != {-1, 0, 1}:
                            raise ValueError("Incomplete available side comparison")
                        for candidate in side["candidates"].values():
                            actual = predictions(params, candidate["vector"], pools)
                            for name, value in actual.items():
                                torch.testing.assert_close(value, candidate["predictions"][name], rtol=0, atol=0)
                            side_count += 1
                completed += 1
            dest = root / "baselines" / f"seed{seed}_d{domain}.pt"
            with torch.no_grad():
                logits = {name: params.logits(params.vector(), x) for name, x in pools.items()}
                baselines = {scale: {name: (scale * z).log_softmax(-1) for name, z in logits.items()}
                             for scale in [1., *plan["confidence_baselines"]]}
            for scaled in baselines.values():
                for name in logits:
                    if not torch.equal(scaled[name].argmax(1), logits[name].argmax(1)):
                        raise ValueError("Positive scaling changed a baseline class prediction")
            if not dest.exists():
                atomic_tensor(dest, baselines)
    return dict(trajectories=completed, endpoints_replayed=verified, side_predictions_replayed=side_count,
                side_statuses=dict(side_statuses))


def metrics(logp, labels):
    result = classification_metrics(logp.exp(), labels, logp)
    ce = -logp[torch.arange(len(labels)), labels]
    for label in (0, 1):
        selected = labels == label
        result[f"class{label}_ce"] = float(ce[selected].mean()) if selected.any() else None
    return result


def margin_summary(logp):
    absolute = (logp[:, 0] - logp[:, 1]).abs()
    return dict(mean_confidence=float(logp.max(dim=1).values.exp().mean()),
                abs_margin_quantiles=torch.quantile(absolute, absolute.new_tensor([0, .25, .5, .75, 1])).tolist())


def round_summary(record):
    info = dict(attempt=record["attempt"], status=record["status"], accepted=record["accepted"])
    trials = record["trials"]
    info["trials"] = len(trials)
    info["risk_rejections"] = sum(not r.get("risk_ok", True) for r in trials)
    info["confidence_rejections"] = sum(not r.get("confidence_ok", True) for r in trials)
    info["motion_rejections"] = sum(not r["bounded"] for r in trials)
    info["armijo_rejections"] = sum(not r["armijo"] for r in trials)
    info["alpha"] = trials[-1]["alpha"] if record["accepted"] else 0.
    info["solver"] = record.get("solver")
    info["unconstrained_risk_violations"] = (record.get("solver") or {}).get("unconstrained_risk_violations")
    info["unconstrained_margin_violation_fraction"] = (record.get("solver") or {}).get(
        "unconstrained_margin_violation_fraction")
    info["density_error"] = record.get("density_error")
    info["relocation_error"] = record.get("relocation_error")
    delta = record.get("delta")
    info["proposal_norm"] = float(delta.norm()) if delta is not None else None
    info["accepted_parameter_norm"] = float((record["after"] - record["before"]).norm())
    if "true_budgets" in record:
        info.update(record["true_budgets"])
    if "relocation" in record:
        roots = record["relocation"]["intersections"]
        info["intersections"] = dict(Counter(r["status"] for r in roots))
        accepted_roots = [(i, r) for i, r in enumerate(roots) if r["status"] == "single_root"]
        info["projection_rejections"] = record["bounds"]["rejected"]
        info["anchors"] = len(roots)
        info["true_margin_rms"] = float(record["margin_proxy"].square().mean().sqrt())
        if accepted_roots:
            index, roots = zip(*accepted_roots)
            displacements = np.array([r["displacement"] for r in roots])
            proxy = -record["margin_proxy"][list(index)].numpy()
            request = -record["bounds"]["zeta"][list(index)].numpy()
            info.update(root_displacement_rmse_vs_proxy=float(np.mean((displacements - proxy)**2)**.5),
                        root_displacement_rmse_vs_request=float(np.mean((displacements - request)**2)**.5),
                        root_displacement_sign_agreement=float(np.mean(np.sign(displacements) == np.sign(request))),
                        new_to_old_slope_ratio=float(np.mean([r["slope_ratio"] for r in roots])),
                        normal_cosine=float(np.mean([r["normal_cosine"] for r in roots
                                                    if r["normal_cosine"] is not None]))
                        if any(r["normal_cosine"] is not None for r in roots) else None)
    density = record.get("density_audit")
    if density is not None:
        for name in ("ess", "origin_weight", "maximum_weight"):
            info[f"mean_{name}"] = float(density[name].mean())
        info["loo_request_rmse"] = float((density["loo_request"] - density["kde_request"]).square().mean().sqrt())
        if "learned_vs_kde" in density:
            info.update({f"score_{k}": v for k, v in density["learned_vs_kde"]["all"].items()})
    return info


def paired_grid(jobs, endpoint, baselines, plan):
    rows = []
    for job in jobs:
        if job.direction != 1:
            continue
        reverse = replace(job, direction=-1)
        zero = replace(job, density="zero", bandwidth=0, direction=0,
                       weight=plan["main_lambda"],
                       epsilon=plan["main_epsilon"] if job.epsilon is not None else None)
        for attempt in plan["endpoints"]:
            for pool in ("source_fit", "source_validation", "target"):
                result = endpoint[job, attempt, pool]
                refs = dict(reverse=endpoint[reverse, attempt, pool],
                            zero=endpoint[zero, attempt, pool])
                for scale in (1., *plan["confidence_baselines"]):
                    refs["source" if scale == 1 else f"scale{scale}"] = baselines[
                        job.seed, job.domain, scale, pool]
                row = dict(asdict(job), attempt=attempt, pool=pool)
                for name in ("nll", "balanced_accuracy", "accuracy"):
                    row[name] = result[name]
                    for label, ref in refs.items():
                        row[f"{name}_vs_{label}"] = result[name] - ref[name]
                rows.append(row)
    return rows


def aggregate(rows, key_fields, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in key_fields)].append(row)
    result = []
    for key, records in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        by_seed = sorted(records, key=lambda r: r["seed"])
        result.append(dict(zip(key_fields, key), **{
            field: seed_summary([r[field] for r in by_seed]) for field in fields
        }))
    return result


def intervention_contrasts(jobs, endpoints, plan):
    rows = []
    for control in jobs:
        if control.group != "M00":
            continue
        for treatment, reference in (("M10", "M00"), ("M01", "M00"),
                                     ("M11", "M00"), ("M11", "M10"), ("M11", "M01")):
            def group_job(group):
                return replace(control, group=group,
                               epsilon=plan["main_epsilon"] if group[2] == "1" else None)
            for attempt in plan["endpoints"]:
                for pool in ("source_fit", "source_validation", "target"):
                    treated = endpoints[group_job(treatment), attempt, pool]
                    baseline = endpoints[group_job(reference), attempt, pool]
                    values = {metric: treated[metric] - baseline[metric] for metric in
                              ("nll", "balanced_accuracy", "accuracy")}
                    rows.append(dict(asdict(control), treatment=treatment, reference=reference,
                                     attempt=attempt, pool=pool, **values))
    return rows


def evaluate(root, plan):
    digest, target_labels = target_values(root, plan["domains"])
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    write_json(out / "label_access.json", dict(observation=digest, purpose="full-domain R3 development evaluation",
                domains={str(d): domain_ids(root, d) for d in plan["domains"]}))
    write_json(out / "target_labels.json", {str(d): labels.tolist() for d, labels in target_labels.items()})
    jobs = grid(plan)
    rounds, side_rows, endpoints, baselines = [], [], {}, {}
    for seed in plan["seeds"]:
        for domain in plan["domains"]:
            params, pools = pools_for(root, seed, domain)
            labels = {role: load(root / "source_labels" / f"seed{seed}_{role}.pt")
                      for role in ("fit", "validation")}
            labels = dict(source_fit=labels["fit"], source_validation=labels["validation"],
                          target=target_labels[domain])
            for scale, values in load(root / "baselines" / f"seed{seed}_d{domain}.pt").items():
                for pool, logp in values.items():
                    baselines[seed, domain, scale, pool] = metrics(logp, labels[pool])
            for job in jobs:
                if (job.seed, job.domain) != (seed, domain):
                    continue
                directory = root / "trajectories" / job.key
                streak = 0
                for attempt in range(plan["adapt"]["adapt_rounds"] + 1):
                    record = load(directory / f"round{attempt:02d}.pt")
                    values = record.get("predictions")
                    if values is None:
                        values = predictions(params, record["after"], pools)
                    if attempt in plan["endpoints"]:
                        for pool, logp in values.items():
                            endpoints[job, attempt, pool] = metrics(logp, labels[pool])
                    if attempt:
                        row = dict(asdict(job), **round_summary(record))
                        streak = 0 if record["accepted"] else streak + 1
                        row["consecutive_nonacceptance"] = streak
                        for pool, logp in values.items():
                            row.update({f"{pool}_{k}": v for k, v in margin_summary(logp).items()})
                            if pool != "target":
                                row.update({f"{pool}_{k}": v for k, v in metrics(logp, labels[pool]).items()})
                        rounds.append(row)
                    if attempt in plan["side_attempts"]:
                        side = record.get("side", {"status": "no_proposal"})
                        for direction in (1, 0, -1):
                            row = dict(asdict(job), attempt=attempt, side_direction=direction,
                                       status=side["status"], alpha=side.get("alpha"))
                            if side["status"] == "available":
                                candidate = side["candidates"][direction]
                                for pool, logp in candidate["predictions"].items():
                                    row.update({f"{pool}_{k}": v for k, v in metrics(logp, labels[pool]).items()})
                            side_rows.append(row)

    pairs = paired_grid(jobs, endpoints, baselines, plan)
    endpoint_rows = [dict(asdict(job), attempt=attempt, pool=pool, **values)
                     for (job, attempt, pool), values in endpoints.items()]
    baseline_rows = [dict(seed=s, domain=d, scale=c, pool=p, **values)
                     for (s, d, c, p), values in baselines.items()]
    write_csv(out / "endpoints.csv", endpoint_rows)
    write_csv(out / "baselines.csv", baseline_rows)
    write_csv(out / "paired.csv", pairs)
    write_csv(out / "rounds.csv", rounds)
    write_csv(out / "side.csv", side_rows)
    interventions = intervention_contrasts(jobs, endpoints, plan)
    write_csv(out / "interventions.csv", interventions)
    contrast_keys = ("domain", "density", "bandwidth", "direction",
                     "treatment", "reference", "attempt", "pool")
    write_json(out / "intervention_summary.json", aggregate(
        interventions, contrast_keys, ("nll", "balanced_accuracy", "accuracy")))
    keys = ("domain", "group", "density", "bandwidth", "direction", "weight", "epsilon", "attempt", "pool")
    fields = [f for f in pairs[0] if f not in (*keys, "seed")]
    pair_summary = aggregate(pairs, keys, fields)
    group_summary = aggregate(endpoint_rows, keys, ("nll", "balanced_accuracy", "accuracy", "class0_ce", "class1_ce"))
    baseline_summary = aggregate(baseline_rows, ("domain", "scale", "pool"), ("nll", "balanced_accuracy", "accuracy"))
    write_json(out / "paired_summary.json", pair_summary)
    write_json(out / "endpoint_summary.json", group_summary)
    write_json(out / "baseline_summary.json", baseline_summary)
    report(root, out, plan, pair_summary, group_summary, baseline_summary, rounds)
    seal(root, "R3_FINAL", ["seals/R3_OBS.json", "evaluation"])


def report(root, out, plan, pairs, groups, baselines, rounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = [r for r in pairs if r["attempt"] == 30 and r["pool"] == "target"]
    lines = ["# Mutagenicity R3", "", "全目标域开发评价。第30次尝试是主终点，NLL越低越好。"
             "各格先在同一种子内配对，再汇总五个种子的均值和样本标准差。", "",
             "逐种子结果见 [endpoints.csv](endpoints.csv) 和 [paired.csv](paired.csv)。"
             "轮次、置信度、边界和密度诊断见 [rounds.csv](rounds.csv)。"
             "同状态旁路见 [side.csv](side.csv)。"
             "干预组配对见 [interventions.csv](interventions.csv)。", "",
             "## 原始源头与输出缩放", "",
             "| 任务 | 输出倍率 | NLL均值 | BA均值 |", "|---|---:|---:|---:|"]
    for row in baselines:
        if row["pool"] == "target":
            lines.append(f"| 0→{row['domain']} | {row['scale']} | {row['nll']['mean']:.6f} | {row['balanced_accuracy']['mean']:.5f} |")
    lines += ["", "## 完整正常请求网格", "",
              "| 任务 | 组 | 密度 | h/σ0 | λ | ε | NLL均值±SD | BA均值±SD | ΔNLL源头 | ΔNLL零 | ΔNLL反向 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in final:
        n, ba = r["nll"], r["balanced_accuracy"]
        deltas = " | ".join(f"{r['nll_vs_' + ref]['mean']:+.6f}" for ref in ("source", "zero", "reverse"))
        lines.append(f"| 0→{r['domain']} | {r['group']} | {r['density']} | {r['bandwidth']} | "
                     f"{r['weight']:g} | {r['epsilon']} | {n['mean']:.6f} ± {n['sample_sd']:.6f} | "
                     f"{ba['mean']:.5f} ± {ba['sample_sd']:.5f} | {deltas} |")

    lines += ["", "## 正常请求的配对结果", ""]
    for domain in plan["domains"]:
        configs = [r for r in final if r["domain"] == domain]
        all_better = [r for r in configs if all(r["nll_vs_" + ref]["mean"] < 0
                                               for ref in ("source", "zero", "reverse"))]
        consistent = sum(
            sum(all(r["nll_vs_" + ref]["values"][i] < 0 for ref in ("source", "zero", "reverse"))
                for i in range(len(plan["seeds"]))) >= 3
            for r in all_better)
        lines.append(f"- 0→{domain} 有 {len(all_better)}/{len(configs)} 个配置的平均NLL同时优于源头、零请求和反向。"
                     f"其中 {consistent} 格在至少三个种子上同时满足三项比较。")

    lines += ["", "## 尝试状态", ""]
    for group in plan["groups"]:
        statuses = Counter(row["status"] for row in rounds if row["group"] == group)
        lines.append(f"- {group} {dict(statuses)}")
    lines += ["", "## Score与敏感性网格", "",
              "| 任务 | h/σ0 | 更新 | DSM均值 | 法向RMSE×σ0 | 符号一致率 |",
              "|---|---:|---:|---:|---:|---:|"]
    scores = []
    for d in plan["domains"]:
        for h in plan["score"]["bandwidths"]:
            for index, step in enumerate(plan["score"]["endpoints"]):
                records = []
                for seed in plan["seeds"]:
                    row = load(score_dir(root, seed, d, h) / "diagnostics.pt")[index]
                    records.append(dict(seed=seed, domain=d, bandwidth=h, step=step,
                                        validation=row["validation"], **row["all"]))
                scores.extend(records)
                items = [seed_summary([r[field] for r in records])["mean"] for field in
                         ("validation", "normal_rmse_sigma", "sign_agreement")]
                rendered = " | ".join("NA" if v is None else f"{v:.6f}" for v in items)
                lines.append(f"| 0→{d} | {h} | {step} | {rendered} |")
    write_csv(out / "score_endpoints.csv", scores)

    for d in plan["domains"]:
        subset = [r for r in final if r["domain"] == d and r["group"] == "M11"
                  and r["density"] == "learned" and r["bandwidth"] == 1]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, ref in zip(axes, ("source", "zero", "reverse")):
            by_grid = {(r["epsilon"], r["weight"]): r["nll_vs_" + ref]["mean"] for r in subset}
            values = np.array([[by_grid[e, w] for w in plan["sensitivity_lambda"]]
                               for e in plan["sensitivity_epsilon"]])
            limit = max(np.abs(values).max(), 1e-12)
            image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set(xticks=range(3), xticklabels=plan["sensitivity_lambda"],
                   yticks=range(3), yticklabels=plan["sensitivity_epsilon"],
                   xlabel="lambda", ylabel="margin budget", title=f"NLL vs {ref}")
            for (i, j), value in np.ndenumerate(values):
                ax.text(j, i, f"{value:+.4f}", ha="center", va="center")
            fig.colorbar(image, ax=ax)
        fig.suptitle(f"Mutagenicity 0 -> {d}, M11, learned 7200, h = sigma0")
        path = out / f"sensitivity_d{d}.png"
        if not path.exists():
            temp = path.with_suffix(".partial.png")
            fig.savefig(temp, dpi=160)
            temp.rename(path)
        plt.close(fig)
        lines += ["", f"![0→{d}九格NLL](sensitivity_d{d}.png)"]
    document = out / "report.md"
    text = "\n".join(lines) + "\n"
    if document.exists():
        if document.read_text() != text:
            raise ValueError("Previously written report differs")
    else:
        temp = document.with_suffix(".partial.md")
        temp.write_text(text)
        temp.rename(document)
