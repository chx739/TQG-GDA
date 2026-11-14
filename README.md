# TSBA: Target-Score Boundary Adaptation

本分支实现用户提供的 TSBA 设计：固定整图编码与坐标，以无标签目标分布的 score 提出边界法向位移，再用源 CE 和阻尼 Gauss–Newton 更新分类模块。代码遵循 SEGA 的模块划分和“候选求解 → 非线性重评估 → 接受／回溯”范式。

这是候选研究方法的工程实现。最新 Mutagenicity R4 覆盖 12 个迁移方向，并已补齐原始 TSBA（M00＋score）对照：平均 Accuracy 从源模型的 69.658% 提升至 70.001%，但零请求对照为 70.020%，尚未展示 score 的平均额外收益。M11 为 69.701%。详见 [R4 完整报告](experiments/mutagenicity/r4-20261112/report.md)。


| 分支 | 用途 |
| --- | --- |
| `SEGA-ref` | SEGA 参考基线；历史清理后的提交为 `ab08590` |
| `TSBA` | TSBA 实现、设计映射与合成检查 |
| `TQG` | 旧 TQG 历史归档，已移除旧实验数组 |

## 仓库导航

- [docs：方法与使用文档](docs/README.md)
- [experiments：方案、实际配置、报告与汇总](experiments/README.md)
- [数据域索引](idx/README.md)：附件索引副本与 Mutagenicity 一致性核验。
- [仓库维护规范](docs/guides/repository.md)

## 设计与实现

- [用户设计原文](docs/method/design-zh.md)：原样保存；[English translation](docs/method/design-en.md)。
- [实验总览与当前结论](experiments/README.md)：区分模拟结果、真实图计划与工程证据。
- [模拟数据验证](experiments/synthetic/README.md)：已完成至P0，含训练预算与源约束结论。
- [真实数据验证](experiments/mutagenicity/README.md)：Mutagenicity R1–R4 及补跑结果、协议与档案。
- [历史完整实验设计](experiments/validation-plan.md)：E0–E5及各轮执行协议。
- [人工代码审查导航](docs/guides/code-review-guide.md)：阅读顺序、公式与测试入口。
- [实现映射与数值选择](docs/method/implementation.md)。
- [数据角色、运行和推断](docs/guides/usage.md)。
- [SEGA 参考及来源](docs/method/sega-reference.md)：参考快照保持原字节。

```text
tsba/
  config.py, data.py, encoder.py, pretrain.py
  classifier.py, score.py, boundary.py
  problem.py, subproblem.py, adapt.py, predict.py, pipeline.py
workflows/tsba/run_task.py
tests/tsba/
```

## 安装与检查

Python >= 3.10，PyTorch >= 2.1，PyG >= 2.4。TSBA 不需要 SEGA 的 CVXPY 求解器依赖。

```bash
python -m pip install -e .
python -m unittest discover -s tests/tsba -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m workflows.tsba.run_task --synthetic --output /tmp/tsba-smoke
```

输出目录必须不存在。`--synthetic` 仅生成小图检查工程链路，不读取真实数据或目标标签。`model.pt` 可独立推断；`result.json` 记录源风险、投影拒绝、求解残差和回溯接受结果。训练入口不计算目标准确率。

SEGA 参照分类器没有 TSBA 所需的线性分类头，不能直接导入其 checkpoint 并声称预测等价。TSBA 默认从明确的源训练池重新训练同类 GIN/GINE 架构及线性头；也可复用本实现生成且源数据、角色、配置身份匹配的 checkpoint。
