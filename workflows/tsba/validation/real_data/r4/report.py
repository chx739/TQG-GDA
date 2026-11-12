"""Publication tables and figures for the fixed R4 endpoint."""
from collections import Counter
import statistics

import numpy as np

from ..r2.evaluate import write_csv
from ..r3.protocol import read_json
from .protocol import role


def render_heatmap(out, summary, plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    matrices = []
    for contrast in ("A2−B0", "A1−B2"):
        values = np.full((4, 4), np.nan)
        for s, t in plan["directions"]:
            values[s, t] = 100 * summary[contrast, f"M{s}→M{t}"]["accuracy"]["mean"]
        matrices.append(values)
    limit = max(*(np.nanmax(abs(matrix)) for matrix in matrices), .01)
    for ax, matrix, contrast in zip(axes, matrices, ("A2−B0", "A1−B2")):
        image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
        for s, t in plan["directions"]:
            value = matrix[s, t]
            ax.text(t, s, f"{value:+.3f}", ha="center", va="center",
                    color="white" if abs(value) > .6 * limit else "black")
        ax.set(xticks=range(4), yticks=range(4), xticklabels=[f"M{i}" for i in range(4)],
               yticklabels=[f"M{i}" for i in range(4)], xlabel="Target", ylabel="Source", title=contrast)
    fig.colorbar(image, ax=axes, label="Accuracy difference (pp)")
    save_figure(fig, out / "directions.png")
    plt.close(fig)


def save_figure(fig, path):
    if not path.exists():
        temp = path.with_suffix(".partial.png")
        fig.savefig(temp, dpi=160)
        temp.rename(path)


def render_curves(out, plan, rounds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(plan["model"]["adapt_rounds"] + 1))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for method in plan["methods"]:
        if method == "B0":
            continue
        rows = [[r for r in rounds if r["method"] == method and r["attempt"] == step] for step in steps]
        for ax, field, scale, label in zip(
            axes, ("accuracy", "net_correct", "accepted"), (100, 1, 100),
            ("Target Accuracy (%)", "Net corrected graphs vs initial head", "Acceptance (%)")
        ):
            values = [scale * statistics.mean(r[field] for r in batch) for batch in rows]
            start = 1 if field == "accepted" else 0
            ax.plot(steps[start:], values[start:], label=method)
            ax.set(xlabel="Attempt", ylabel=label, xticks=plan["endpoints"])
    axes[0].legend()
    save_figure(fig, out / "round_curves.png")
    plt.close(fig)


def report(root, plan, out, summaries, rounds):
    summary = {(r["method_or_contrast"], r["scope"]): r for r in summaries}
    scopes = [f"M{s}→M{t}" for s, t in plan["directions"]]
    methods = plan["methods"]
    overall = summary["A2−B0", "Avg"]["accuracy"]
    replay = read_json(root / "replay.json")
    positive = sum(summary["A2−B0", scope]["accuracy"]["mean"] > 0 for scope in scopes)
    equal = sum(summary["A2−B0", scope]["accuracy"]["mean"] == 0 for scope in scopes)
    lines = [
        "# Mutagenicity R4 固定配置全方向评估", "",
        f"A2 的 12 方向等权平均 Accuracy 相对原始源头变化 **{100*overall['mean']:+.4f} 个百分点**。"
        f"方向均值有 {positive} 个改善、{equal} 个持平、{len(scopes)-positive-equal} 个退化。"
        "主终点为第 30 次尝试，M0→M1 和 M0→M3 是历史开发方向，其余十个方向为新增覆盖。", "",
        "数据使用仓库 SEGA 的有序域索引，结构和特征逐图核对至 R3。"
        "与 CoCo 原论文的划分对应关系尚未核实。"
        "实验使用完整目标域做开发计分，目标标签在所有轨迹和预测封存后按域读取。", "",
        "[逐种子端点与辅助指标](endpoints.csv) · [逐种子配对](paired.csv) · "
        "[逐轮预测指标与纠错计数](rounds.csv) · [旁路](side.csv) · "
        "[Score诊断](score_endpoints.csv) · [源训练曲线数据](source_curves.csv) · [完整汇总](summary.csv)", "",
        "## Accuracy 主表", "",
        "百分数为五个种子的均值±样本标准差。Avg 先在每个种子内等权平均 12 方向，再计算五个种子的波动。", "",
        "| 方法 | " + " | ".join(scopes + ["Avg"]) + " |",
        "|---|" + "---:|" * (len(scopes) + 1),
    ]
    for method in methods:
        values = [summary[method, scope]["accuracy"] for scope in scopes + ["Avg"]]
        lines.append(f"| {method} | " + " | ".join(f"{100*v['mean']:.3f} ± {100*v['sd']:.3f}" for v in values) + " |")
    lines += ["", "B0 为原始源头。B1、B2 分别为 M00、M01 零请求。"
              "A1 为 M01 学习 score，A2 为预指定主方案 M11 学习 score，A3 为 M11 经验 KDE。", "",
              "### 源域与目标域各三方向平均", "",
              "| 方法 | " + " | ".join(f"{side} M{d}" for side in ("source", "target") for d in plan["domains"]) + " |",
              "|---|" + "---:|" * (2 * len(plan["domains"]))]
    for method in methods:
        values = [summary[method, f"{side} M{d}"]["accuracy"] for side in ("source", "target")
                  for d in plan["domains"]]
        lines.append(f"| {method} | " + " | ".join(f"{100*v['mean']:.3f} ± {100*v['sd']:.3f}" for v in values) + " |")
    lines += ["",
              "## Accuracy 配对差值", "",
              "单位为百分点。括号为改善／持平／退化的种子数，持平按正确数差值精确为零计。"
              "Avg、新增方向和域汇总的比较在种子内先平均相应方向。", ""]
    pairs = [f"{left}−{right}" for left, right in plan["pairs"]]
    lines += ["| 范围 | " + " | ".join(pairs) + " |", "|---|" + "---:|" * len(pairs)]
    for scope in scopes + [f"{side} M{d}" for side in ("source", "target") for d in plan["domains"]] + ["New coverage", "Avg"]:
        values = [summary[pair, scope]["accuracy"] for pair in pairs]
        lines.append(f"| {scope} | " + " | ".join(
            f"{100*v['mean']:+.3f} ± {100*v['sd']:.3f} ({v['wins']}/{v['ties']}/{v['losses']})" for v in values
        ) + " |")
    lines += ["", "![两个登记对比的方向热力图](directions.png)", "",
              "## 辅助指标与轮次", "",
              "下表为 12 方向等权均值。每方向和每种子的混淆矩阵、召回及预测类别比例在逐种子端点表。", "",
              "| 方法 | BA (%) | Macro-F1 (%) | NLL | Recall 0 (%) | Recall 1 (%) |",
              "|---|---:|---:|---:|---:|---:|"]
    for m in methods:
        r = summary[m, "Avg"]
        lines.append(f"| {m} | " + " | ".join(
            f"{r[key]['mean']*scale:.5f} ± {r[key]['sd']*scale:.5f}" for key, scale in
            (("balanced_accuracy",100),("macro_f1",100),("nll",1),("recall0",100),("recall1",100))) + " |")
    lines += ["", "| 方法 | " + " | ".join(f"第 {step} 次 Accuracy (%)" for step in plan["endpoints"]) + " |",
              "|---|" + "---:|" * len(plan["endpoints"])]
    for m in methods:
        if m != "B0":
            lines.append(f"| {m} | " + " | ".join(
                f"{100*statistics.mean(r['accuracy'] for r in rounds if r['method']==m and r['attempt']==step):.4f}"
                for step in plan["endpoints"]) + " |")
    lines += ["", "![逐轮准确率、净纠错和接受率](round_curves.png)", "",
              "净纠错以初始头为参照。逐轮表另含相对前一状态的错→对、对→错、净纠错及按类分解。", "",
              "## 诊断与完整性", "",
              "| 方法 | 尝试状态计数 | 平均边界覆盖 | KDE ESS 中位数 | 生成图权重中位数 | 最大图权重中位数 | 真实置信度拒绝试验 | 真实风险拒绝试验 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    diagnostics = []
    for m in methods:
        if m == "B0":
            continue
        rows = [r for r in rounds if r["method"] == m and r["attempt"]]
        def median(key):
            values = [r[key] for r in rows if r.get(key) is not None]
            return statistics.median(values) if values else None
        stats = dict(method=m, statuses=dict(Counter(r["status"] for r in rows)),
                     coverage=statistics.mean(r.get("anchors", 0) / plan["model"]["target_batch"] for r in rows),
                     ess=median("mean_ess"), origin=median("mean_origin_weight"), maximum=median("mean_maximum_weight"),
                     confidence=sum(r["confidence_rejections"] for r in rows), risk=sum(r["risk_rejections"] for r in rows))
        diagnostics.append(stats)
        shown = ["NA" if stats[key] is None else f"{stats[key]:.4f}" for key in ("coverage", "ess", "origin", "maximum")]
        lines.append(f"| {m} | `{stats['statuses']}` | " + " | ".join(shown) +
                     f" | {stats['confidence']} | {stats['risk']} |")
    write_csv(out / "diagnostics.csv", diagnostics)
    lines += ["", "KDE 数值列是各轮锚点均值的中位数。旁路使用第 1、30 次尝试前的同一状态和可行步长。",
              "", "| 源域 | 每种子训练／验证数 |", "|---|---|"]
    for d in plan["domains"]:
        counts = [f"{len(r['fit'])}/{len(r['validation'])}" for r in (role(root, d, seed) for seed in plan["seeds"])]
        lines.append(f"| M{d} | {', '.join(counts)} |")
    lines += ["", f"重放确认 {replay['sources']} 份源模型、{replay['scores']} 份 score、"
              f"{replay['trajectories']} 条轨迹及 {replay['attempts']} 次尝试。"
              f"逐项核对 {replay['predictions_replayed']} 份轨迹状态预测和 "
              f"{replay.get('side_predictions_replayed',0)} 份可用旁路预测。"
              f"旁路状态为 `{replay['side_statuses']}`。", "",
              "所有模型和轨迹采用新训练产物。R3 仅提供经过哈希认证的数据来源与数值实现。"
              "经验 KDE 是同一有限目标样本池上的参考。"]
    render_heatmap(out, summary, plan)
    render_curves(out, plan, rounds)
    path = out / "report.md"
    text = "\n".join(lines) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Existing report differs from sealed evaluation")
    else:
        temp = path.with_suffix(".partial.md")
        temp.write_text(text)
        temp.rename(path)
