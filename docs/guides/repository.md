# 仓库维护

## 分工

- `docs/`：方法、实现、使用、维护与历史说明。
- `experiments/`：TSBA 的实验方案、计划、实际执行记录、报告与必要汇总。
- `tsba/`：核心算法；`workflows/tsba/run_task.py`：通用任务入口；`tests/tsba/`：工程测试。
- `experiments/common/`：跨轮复用工具；各轮实验目录保存专用运行与分析代码。
- 旧 TQG 保留在 `TQG` 分支；参考快照 `references/` 不改字节。

移动目录的提交只改变组织和引用，不混入算法调整。已完成实验结果不重算；修改结论另给依据。

## 可追溯性

已完成实验以 `executed-plan.json`、`executed-configs.json`、`evidence.json` 和原档案为执行依据。历史路径保持原样；各轮 `evidence.json` 记录原始执行提交号及历史清理后的对应提交号。

重新运行工程测试可以验证路径和实现；不据此授权真实数据训练、访问目标标签或重新生成研究结果。
