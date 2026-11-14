# 文档导航

`docs/` 解释 TSBA 的方法、实现和使用方式。每轮实验的方案、配置、报告和结果统一放在 [experiments](../experiments/README.md)。

| 分类 | 文档 | 用途 |
|---|---|---|
| 方法 | [中文设计原文](method/design-zh.md) · [英文译文](method/design-en.md) | 理解方法定义与公式 |
| 方法 | [实现映射与数值选择](method/implementation.md) | 对照设计理解当前实现 |
| 方法 | [SEGA 参考来源](method/sega-reference.md) | 了解参考快照及适用边界 |
| 使用 | [数据协议、运行与推断](guides/usage.md) | 使用训练和推断入口 |
| 使用 | [实验记录规范](guides/experiment-records.md) | 区分计划、实际配置与结果证据 |
| 使用 | [仓库维护约定](guides/repository.md) | 目录分工与变更范围 |
| 使用 | [代码审查导航](guides/code-review-guide.md) | 按公式、代码和测试检查实现 |

## 阅读顺序

首次了解方法：中文设计 → 实现映射 → 使用说明。检查代码：实现映射 → 代码审查导航 → 相关测试。

查看研究进展直接进入[实验总览](../experiments/README.md)：[合成实验](../experiments/synthetic/README.md)从几何与控制实验读到 P0；[Mutagenicity](../experiments/mutagenicity/README.md)从表示诊断读到 R4 全方向比较。每轮先看证据入口，再看报告。

## 历史状态说明

设计原文保持原始字节，“尚未实现或验证”描述的是提供时状态。审查导航中按日期记录的测试数量也不是当前状态。当前代码已实现并完成多轮实验；是否有效应依据实验报告，不能用工程测试通过代替效果证据。具体数据划分和标签权限以对应轮次协议为准。

目录只保留 `method/`、`guides/` 两类；不在这里复制实验报告。旧 TQG 见 `TQG` 分支，参考快照见仓库的 `references/`。
