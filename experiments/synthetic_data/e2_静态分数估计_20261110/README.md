# E2｜固定训练预算下，样本量与带宽如何影响 score？

状态：已完成。

## 问题与结论

**研究问题：** 固定训练预算下，样本量与带宽如何影响 score？

**结论摘要：** 400 步结果不能解释为网络不可学；训练预算判断由 P0 后续更新。 数值、失败项与适用范围见[完整报告](report.md)。

## 目录

```text
README.md     阅读与运行入口
src/          本轮实验实现
scripts/      本轮启动与分析入口
configs/      计划配置与实际配置
docs/         实验方案及补充说明（如有）
results/      汇总表、图表、日志和核验材料
report.md     结论与分析
```

## 配置与证据

- [frozen-plan.json](configs/frozen-plan.json)
- [static-score-plan.json](configs/static-score-plan.json)
- [实际执行配置](configs/executed-plan.json)
- [配置快照](configs/executed-configs.json)
- [执行来源与计划差异](results/evidence.json)
- [原始产物位置与校验](results/archive-index.json)
- [结果汇总：summary.json](results/summary.json)
- [结果汇总：metrics.csv](results/metrics.csv)
- [结果汇总：fits.csv](results/fits.csv)

## 如何运行

从仓库根目录安装 `python -m pip install -e '.[experiments]'`。本轮入口见 `scripts/`，实现见 `src/`。先查看参数，再按方案提供输入和新输出目录：

- [run_static_score.py](scripts/run_static_score.py)：`python -m experiments.synthetic_data.e2_静态分数估计_20261110.scripts.run_static_score --help`
- [summarize_static_score.py](scripts/summarize_static_score.py)：`python -m experiments.synthetic_data.e2_静态分数估计_20261110.scripts.summarize_static_score --help`

历史计划中的机器路径与已封存运行身份保留原貌；重现原记录需使用对应的代码快照与档案，当前入口用于按照协议开展新的执行。不会仅因查看帮助而访问真实目标标签。

## 代码依赖

核心算法来自 `tsba/`；通用工具来自 [experiments/common](../../common/README.md)。

本轮显式沿用以下历史实验的实现，不将其伪装成通用工具，也不复制多份：

- [e1_e2_符号几何与适配_20261109](../e1_e2_符号几何与适配_20261109/README.md)
- [e2_目标请求对照_20261109](../e2_目标请求对照_20261109/README.md)
- [e2_全源批次对照_20261110](../e2_全源批次对照_20261110/README.md)
- [e2_适配轨迹回放_20261110](../e2_适配轨迹回放_20261110/README.md)

## 验证

在仓库根目录运行 `python -m unittest discover -s tests/tsba -v`。工程检查与研究效果分开报告；原始结果不因目录整理重新生成。
