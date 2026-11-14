# r1b_20261111

状态：已完成。

## 问题与结论

**研究问题：** 参考头、密度尺度、静态 score 和单步更新反映哪些问题？

**结论摘要：** 这些诊断不能单独证明闭环适配有效，具体结论见完整报告。 数值、失败项与适用范围见[完整报告](report.md)。

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

- [next-round-plan.json](configs/next-round-plan.json)
- [r1b-protocol.md](docs/r1b-protocol.md)
- [实际执行配置](configs/executed-plan.json)
- [配置快照](configs/executed-configs.json)
- [执行来源与计划差异](results/evidence.json)
- [原始产物位置与校验](results/archive-index.json)

## 如何运行

从仓库根目录安装 `python -m pip install -e '.[experiments]'`。启动脚本放在仓库根目录的 `scripts/`，本轮实现放在 `src/`。先查看参数，再按方案提供输入和新输出目录：

- [locality_figures.py](../../../scripts/mutagenicity/r1b_20261111/locality_figures.py)：`python -m scripts.mutagenicity.r1b_20261111.locality_figures --help`
- [mechanism_summary.py](../../../scripts/mutagenicity/r1b_20261111/mechanism_summary.py)：`python -m scripts.mutagenicity.r1b_20261111.mechanism_summary --help`
- [observation_summary.py](../../../scripts/mutagenicity/r1b_20261111/observation_summary.py)：`python -m scripts.mutagenicity.r1b_20261111.observation_summary --help`
- [publish.py](../../../scripts/mutagenicity/r1b_20261111/publish.py)：`python -m scripts.mutagenicity.r1b_20261111.publish --help`
- [run.py](../../../scripts/mutagenicity/r1b_20261111/run.py)：`python -m scripts.mutagenicity.r1b_20261111.run --help`
- [training_curves.py](../../../scripts/mutagenicity/r1b_20261111/training_curves.py)：`python -m scripts.mutagenicity.r1b_20261111.training_curves --help`

历史计划中的机器路径与已封存运行身份保留原貌；重现原记录需使用对应的代码快照与档案，当前入口用于按照协议开展新的执行。不会仅因查看帮助而访问真实目标标签。

## 代码依赖

核心算法来自 `tsba/`；通用工具来自 [experiments/common](../../common/README.md)。

本轮显式沿用以下历史实验的实现，不将其伪装成通用工具，也不复制多份：

- [r1_20261111](../r1_20261111/README.md)

## 验证

在仓库根目录运行 `python -m unittest discover -s tests/tsba -v`。工程检查与研究效果分开报告；原始结果不因目录整理重新生成。
