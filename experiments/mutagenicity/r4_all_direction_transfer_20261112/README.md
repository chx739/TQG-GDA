# 全方向迁移对照

历史轮次：R4。状态：已完成。

## 问题与结论

**研究问题：** 固定配置在所有迁移方向上是否有稳定收益？

**结论摘要：** 包含 A0 的完整比较中，score 尚未表现出平均额外收益。 数值、失败项与适用范围见[完整报告](report.md)。

## 目录

```text
README.md     阅读与运行入口
src/          本轮实验实现
configs/      计划配置与实际配置
docs/         实验方案及补充说明（如有）
results/      汇总表、图表、日志和核验材料
report.md     结论与分析
```

## 配置与证据

- [r4-plan.json](configs/r4-plan.json)
- [r4-protocol.md](docs/r4-protocol.md)
- [实际执行配置](configs/executed-plan.json)
- [配置快照](configs/executed-configs.json)
- [执行来源与计划差异](results/evidence.json)
- [原始产物位置与校验](results/archive-index.json)
- [结果汇总：summary.csv](results/summary.csv)

R4 主运行和 A0 补跑统一记录在这里；实际配置中的 `main` 与 `A0_extension` 保留执行来源，整合记录不是单一运行器配置。

## 如何运行

从仓库根目录安装 `python -m pip install -e '.[experiments]'`。启动脚本放在仓库根目录的 `scripts/`，本轮实现放在 `src/`。先查看参数，再按方案提供输入和新输出目录：

- [run.py](../../../scripts/mutagenicity/r4_all_direction_transfer_20261112/run.py)：`python -m scripts.mutagenicity.r4_all_direction_transfer_20261112.run --help`
- [supplement.py](../../../scripts/mutagenicity/r4_all_direction_transfer_20261112/supplement.py)：`python -m scripts.mutagenicity.r4_all_direction_transfer_20261112.supplement --help`

历史计划中的机器路径与已封存运行身份保留原貌；重现原记录需使用对应的代码快照与档案，当前入口用于按照协议开展新的执行。不会仅因查看帮助而访问真实目标标签。

## 代码依赖

核心算法来自 `tsba/`；通用工具来自 [experiments/common](../../common/README.md)。

本轮显式沿用以下历史实验的实现，不将其伪装成通用工具，也不复制多份：

- [r1_representation_boundary_support_20261111](../r1_representation_boundary_support_20261111/README.md)
- [r1b_density_and_single_step_diagnostics_20261111](../r1b_density_and_single_step_diagnostics_20261111/README.md)
- [r2_score_budget_and_adaptation_20261111](../r2_score_budget_and_adaptation_20261111/README.md)
- [r3_risk_and_confidence_constraints_20261112](../r3_risk_and_confidence_constraints_20261112/README.md)

## 验证

在仓库根目录运行 `python -m unittest discover -s tests/tsba -v`。工程检查与研究效果分开报告；原始结果不因目录整理重新生成。
