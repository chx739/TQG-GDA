# Score 训练预算与多轮适配

历史轮次：R2。状态：已完成。

## 问题与结论

**研究问题：** score 训练预算、λ 与多轮适配如何影响结果？

**结论摘要：** 本轮沿用旧数据口径，不能直接与后续完整目标域实验比分。 数值、失败项与适用范围见[完整报告](report.md)。

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

- [execution-plan.json](configs/execution-plan.json)
- [r2-plan.json](configs/r2-plan.json)
- [r2-protocol.md](docs/r2-protocol.md)
- [实际执行配置](configs/executed-plan.json)
- [配置快照](configs/executed-configs.json)
- [执行来源与计划差异](results/evidence.json)
- [原始产物位置与校验](results/archive-index.json)
- [原始产物存档](results/archive.json)
- [结果汇总：summary.csv](results/summary.csv)
- [结果汇总：metrics.csv](results/metrics.csv)

## 如何运行

从仓库根目录安装 `python -m pip install -e '.[experiments]'`。启动脚本放在仓库根目录的 `scripts/`，本轮实现放在 `src/`。先查看参数，再按方案提供输入和新输出目录：

- [interpret.py](../../../scripts/mutagenicity/r2_score_budget_and_adaptation_20261111/interpret.py)：`python -m scripts.mutagenicity.r2_score_budget_and_adaptation_20261111.interpret --help`
- [publish.py](../../../scripts/mutagenicity/r2_score_budget_and_adaptation_20261111/publish.py)：`python -m scripts.mutagenicity.r2_score_budget_and_adaptation_20261111.publish --help`
- [run.py](../../../scripts/mutagenicity/r2_score_budget_and_adaptation_20261111/run.py)：`python -m scripts.mutagenicity.r2_score_budget_and_adaptation_20261111.run --help`
- [run_resources.py](../../../scripts/mutagenicity/r2_score_budget_and_adaptation_20261111/run_resources.py)：`python -m scripts.mutagenicity.r2_score_budget_and_adaptation_20261111.run_resources --help`

历史计划中的机器路径与已封存运行身份保留原貌；重现原记录需使用对应的代码快照与档案，当前入口用于按照协议开展新的执行。不会仅因查看帮助而访问真实目标标签。

## 代码依赖

核心算法来自 `tsba/`；通用工具来自 [experiments/common](../../common/README.md)。

本轮显式沿用以下历史实验的实现，不将其伪装成通用工具，也不复制多份：

- [r1_representation_boundary_support_20261111](../r1_representation_boundary_support_20261111/README.md)
- [r1b_density_and_single_step_diagnostics_20261111](../r1b_density_and_single_step_diagnostics_20261111/README.md)

## 验证

在仓库根目录运行 `python -m unittest discover -s tests/tsba -v`。工程检查与研究效果分开报告；原始结果不因目录整理重新生成。
