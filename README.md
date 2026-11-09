# TSBA: Target-Score Boundary Adaptation

本分支实现用户提供的 TSBA 设计：固定整图编码与坐标，以无标签目标分布的 score 提出边界法向位移，再用源 CE 和阻尼 Gauss–Newton 更新分类模块。代码遵循 SEGA 的模块划分和“候选求解 → 非线性重评估 → 接受／回溯”范式。

这是候选研究方法的工程实现，尚未进行真实数据有效性实验。测试通过不代表低密度分隔假设成立或目标准确率提高。

| 分支 | 用途 |
| --- | --- |
| `SEGA-ref` | 原 `main`，保留参考基线 `aff6582` |
| `TSBA` | TSBA 实现、设计映射与合成检查 |
| `history/tqg-v2-20261106` | 旧 TQG 实现和实验历史，固定 `eb39fd7` |

## 设计与实现

- [用户设计原文](docs/tsba/TSBA_Graph_Level_GDA_Design.md)：原样保存；[English translation](docs/tsba/TSBA_Graph_Level_GDA_Design_EN.md)。
- [人工代码审查导航](docs/tsba/code-review-guide.md)：阅读顺序、公式与测试入口。
- [实现映射与数值选择](docs/tsba/implementation.md)。
- [数据角色、运行和推断](docs/tsba/usage.md)。
- [独立 review 与修复记录](docs/tsba/review.md)。
- [SEGA 参考及来源](docs/design-baseline.md)：参考快照保持原字节。

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
