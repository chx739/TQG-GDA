"""Setting-specific tables and scientific plots."""
import statistics
import numpy as np
from .shared import output


def report(out, plan, rows, rounds, sides):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = {(r["method_or_contrast"], r["scope"]): r for r in rows}
    lines = [f"# R5 {plan['setting']['id']}", "",
             "第 30 次尝试的完整目标域开发评价。主比较为 A0−B1。", "",
             "| 配对 | Accuracy 差（百分点） | SD | 胜／平／负 | NLL 差 |",
             "|---|---:|---:|---:|---:|"]
    for left, right in plan["pairs"]:
        r = summary[f"{left}−{right}", "Avg"]
        a = r["accuracy"]
        lines.append(f"| {left}−{right} | {100*a['mean']:+.5f} | {100*a['sd']:.5f} | "
                     f"{a['wins']}/{a['ties']}/{a['losses']} | {r['nll']['mean']:+.6f} |")
    available = sum(r["status"] == "available" and r["pool"] == "target" and r["direction"] == 1
                    for r in sides if "pool" in r)
    lines += ["", "| 方向 | B0 Accuracy | B1 Accuracy | A0 Accuracy |", "|---|---:|---:|---:|"]
    for scope in [f"M{s}→M{t}" for s, t in plan["directions"]] + ["Avg"]:
        cells = [f"{100*summary[method, scope]['accuracy']['mean']:.4f} ± {100*summary[method, scope]['accuracy']['sd']:.4f}"
                 for method in ("B0", "B1", "A0")]
        lines.append("| " + " | ".join([scope, *cells]) + " |")
    lines += ["", f"计划旁路 120 块，可用 {available} 块。", "",
              "历史复用设置的新增几何字段属于历史状态新增分析；历史预测保持原值。" if plan["setting"]["action"] == "reuse"
              else "本设置全部源模型、score 和适配轨迹为新运行。",
              "",
              "置信区间描述固定数据设置下五个种子的训练随机性。", "",
              "CSV 中 accuracy/balanced_accuracy/macro_f1 为 0–1 比例，显式 *_pp 列为百分点；NLL 为原量纲。",
              "旁路汇总保留全部计划 seed，缺少任一方向时完整均值记 NA；available_mean 仅表示注明覆盖的可用子集。",
              "",
              "[逐方向配对](paired.csv) · [五种子原始均值](paired_seeds.csv) · [旁路配对](side_paired.csv) · "
              "[旁路种子汇总](side_seeds.csv) · [旁路覆盖及统计](side_summary.csv) · [完整端点](endpoints.csv)",
              "", "![配对与轮次](overview.png)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    matrix = np.full((4, 4), np.nan)
    for s, t in plan["directions"]:
        matrix[s, t] = summary["A0−B1", f"M{s}→M{t}"]["accuracy"]["mean"] * 100
    lim = max(float(np.nanmax(abs(matrix))), .01)
    image = axes[0].imshow(matrix, cmap="RdBu", vmin=-lim, vmax=lim)
    axes[0].set(title="A0 - B1 accuracy (pp)", xlabel="Target", ylabel="Source")
    fig.colorbar(image, ax=axes[0])
    for method in ("B1", "A0"):
        values = [statistics.mean(r["accuracy"] for r in rounds if r["method"] == method and r["attempt"] == step)*100
                  for step in range(31)]
        axes[1].plot(range(31), values, label=method)
    axes[1].set(xlabel="Attempt", ylabel="Accuracy (%)")
    axes[1].legend()
    for direction in (-1, 0, 1):
        values = []
        for step in plan["side_attempts"]:
            subset = [r["nll"] for r in sides if r.get("pool") == "target" and r["attempt"] == step
                      and r.get("direction") == direction]
            values.append(statistics.mean(subset) if subset else np.nan)
        axes[2].plot(plan["side_attempts"], values, marker="o", label=str(direction))
    axes[2].set(xlabel="Before attempt", ylabel="Same-state target NLL")
    axes[2].legend()
    if not (out / "overview.png").exists():
        fig.savefig(out / "overview.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for index, (left, right) in enumerate(plan["pairs"]):
        values = [summary[f"{left}−{right}", f"M{s}→M{t}"]["accuracy"]["mean"] * 100 for s, t in plan["directions"]]
        ax.scatter(np.arange(len(values)) + .2*(index-1), values, label=f"{left}-{right}")
    ax.axhline(0, color="grey", linewidth=.8)
    ax.set(xticks=np.arange(len(plan["directions"])), xticklabels=[f"{s}→{t}" for s,t in plan["directions"]],
           ylabel="Accuracy difference (pp)", xlabel="Source → target")
    ax.legend()
    if not (out / "endpoint_pairs.png").exists():
        fig.savefig(out / "endpoint_pairs.png", dpi=160)
    plt.close(fig)
    lines += ["", "![端点配对](endpoint_pairs.png)"]
    text = "\n".join(lines) + "\n"
    path = out / "report.md"
    if path.exists() and path.read_text() != text:
        raise ValueError("Report differs on recovery")
    if not path.exists():
        path.write_text(text)
