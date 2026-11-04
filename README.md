# TQG-GDA

TQG（Target-Queried Gradient Learning）的图级域适应研究代码、实验设计和可核验实验记录。

**当前结论（2026-11-04）：真实数据上尚未获得跨任务稳定收益，也未证明逐图源标签提供稳定迁移增量。** 友好合成数据说明机制可以成立；真实任务中的迁移条件仍待诊断。工程验收与研究有效性分别记录。

最新：[局部源经验／核读取优化完整结论](docs/v2/results/optimization_20261104/README.md)。A4、A2-b、A6均完成并审计；LOCAL/KERNEL均0/3通过，不触发连续适配。

## 从这里开始

| 你要做什么 | 入口 |
|---|---|
| 看当前方法、计划和结果 | [v2研究主页](docs/v2/README.md) |
| 回顾odd/even原方法及早期实验 | [v1历史研究主页](docs/v1/README.md) |
| 看下一步怎么验证四项迁移条件 | [§3.8验证方案与已有证据](docs/v2/plans/assumptions_20261103/README.md)；第一批A0／A1／A2-a已完成 |
| 查哪些实验已做、结论是什么 | [v2执行状态](docs/v2/results/status.md) · [全部实验归档](experiments/README.md) |
| 精读代码、找运行入口或配置 | [代码导航](docs/code/README.md) · [配置导航](configs/README.md) |
| 安装、运行或核验归档 | [使用指南](docs/usage.md) |

## v1与v2的区别

| 版本 | 读取方式 | 研究材料 | 完整训练结果 |
|---|---|---|---|
| v1 | 有限参数扰动、odd/even响应，用源残差拟合组合权重 | [方法](docs/v1/method/design.md) · [实验设计](docs/v1/plans/README.md) | [PROTEINS及Mutagenicity汇总](docs/v1/results/overview.md) |
| v2（当前） | 目标Jacobian空间、签名核、矩阵读出与参数梯度反馈 | [方法](docs/v2/method/design.md) · [实验设计](docs/v2/plans/README.md) | [Mutagenicity及batch对照](docs/v2/results/overview.md) |

v2诊断链：P0 → D3/D4 → H-only → D8／残差分解 → 标签／预测归因与D7-H。各轮均有[结果、报告和原始记录](docs/v2/results/README.md)，不把诊断单步收益当作完整训练准确率。

Mutagenicity：[§3.8第一批条件诊断](docs/v2/results/tqg_assumptions_20261103/结果解读.md)完成。同签名LR／MLP可在各域学习，但存在源到目标读出缺口；H支持分层三个任务均有诊断价值。下一步优先局部bank与核读取对照，尚不代表梯度反馈有效。

最新：[PROTEINS同方案复验，2026-11-04](docs/v2/results/tqg_assumptions_proteins_20261104/结果解读.md)完成。同域稳定可学前提未过，支持分组图数不足；未复现Mutagenicity的完整证据链，不把原因简单归为数据量。

## 目录

```text
docs/
  v1/                 原方法、改进计划、实现说明、结果
  v2/                 当前方法、诊断计划、实现说明、结果
  code/               模块与脚本导航
  research_plans/     保持原字节的2026-10-31冻结协议及兼容入口
  history/            原始阶段文档与来源清单
  validation/         历史工程验收记录
experiments/
  v1/README.md        v1原始实验归档索引
  v2/README.md        v2原始实验归档索引
  <实验名>/           原始日志、checkpoint、代码快照、SHA与评价索引
configs/              按版本索引配置；现有运行路径保持兼容
scripts/              基础运行、评价、归档与校验工具
tqg_gda/              共用骨干、v1/v2算法及训练模块
tests/                数学、训练、诊断和权限测试
```

主文档已按版本归组。旧文档路径保留兼容符号链接；日常浏览以本页的新路径为准。原始实验归档、冻结协议、历史快照保留原路径与字节，不改写其来源记录。详见[整理规则与路径映射](docs/REPOSITORY_LAYOUT.md)。

## 安装与检查

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

真实图缓存需自行准备；数据哈希和角色应匹配具体实验协议。PROTEINS／Mutagenicity的0→1/2/3沿用SEGA提供的图规模域索引，不是数据集官方自然域。目标final标签不用于日常诊断、调参或选模。

完成修改或实验后，更新对应版本文档、核验产物并同步GitHub，约定见[AGENTS.md](AGENTS.md)。
