# 图表示与边界支持

历史轮次：R1。状态：已完成。

## 问题与结论

**研究问题：** 冻结的真实图表示是否具备类别信息和可诊断的边界支持？

**结论摘要：** 局部支持不足，不能据此确认低密度分隔。 数值、失败项与适用范围见[完整报告](report.md)。

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

- [stage1-plan.json](configs/stage1-plan.json)
- [mutagenicity-stage1-protocol.md](docs/mutagenicity-stage1-protocol.md)
- [实际执行配置](configs/executed-plan.json)
- [配置快照](configs/executed-configs.json)
- [执行来源与计划差异](results/evidence.json)
- [原始产物位置与校验](results/archive-index.json)

## 如何运行

从仓库根目录安装 `python -m pip install -e '.[experiments]'`。启动脚本放在仓库根目录的 `scripts/`，本轮实现放在 `src/`。先查看参数，再按方案提供输入和新输出目录：

- [repair_evaluation.py](../../../scripts/mutagenicity/r1_representation_boundary_support_20261111/repair_evaluation.py)：`python -m scripts.mutagenicity.r1_representation_boundary_support_20261111.repair_evaluation --help`
- [run.py](../../../scripts/mutagenicity/r1_representation_boundary_support_20261111/run.py)：`python -m scripts.mutagenicity.r1_representation_boundary_support_20261111.run --help`
- [support_summary.py](../../../scripts/mutagenicity/r1_representation_boundary_support_20261111/support_summary.py)：`python -m scripts.mutagenicity.r1_representation_boundary_support_20261111.support_summary --help`

历史计划中的机器路径与已封存运行身份保留原貌；重现原记录需使用对应的代码快照与档案，当前入口用于按照协议开展新的执行。不会仅因查看帮助而访问真实目标标签。

## 代码依赖

核心算法来自 `tsba/`；通用工具来自 [experiments/common](../../common/README.md)。
本轮没有跨轮实现依赖。

## 验证

在仓库根目录运行 `python -m unittest discover -s tests/tsba -v`。工程检查与研究效果分开报告；原始结果不因目录整理重新生成。
