# TQG v1：原方法与历史实验

v1使用有限参数扰动，读取源／目标图的odd、even响应，再用源标签残差拟合组合权重，构造目标输出纠错。后续的Jacobian核查询方案属于[v2](../v2/README.md)。

| 阅读顺序 | 入口 |
|---|---|
| 1．理解方法 | [原始设计稿](method/design.md) · [odd/even与公式对应](method/response_formulas.md) |
| 2．看代码边界 | [实现与SEGA复用说明](implementation/notes.md) · [代码导航](../code/README.md) |
| 3．看实验为什么这样设计 | [实验计划、E1/E4与失败分析](plans/README.md) |
| 4．看实际结果 | [PROTEINS／Mutagenicity详细汇总](results/overview.md) · [结果目录](results/README.md) |
| 5．查日志与模型 | [v1实验归档](../../experiments/v1/README.md) |

当前历史结论：两数据集的完整对照未呈现跨任务稳定收益，even没有稳定增量。不能把v1的Adam结果与v2的SGD结果当作完全匹配的版本比较。

其他历史材料：[阶段README与来源清单](../history/README.md)、[E1](../validation/e1/)／[E2](../validation/e2/)／[E3](../validation/e3/)／[E4](../validation/e4/)工程与运行记录、[v1配置](../../configs/v1/README.md)。阶段文档中的旧绝对路径按来源记录理解。
