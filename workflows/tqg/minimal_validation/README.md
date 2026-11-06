# SEGA式 TQG 最小验证执行器

执行[已登记的 M0/M1/M2 方案](../../../docs/research_plans/tqg_sega_minimal_20261106/README.md)。算法模块保持不变；此包只提供实验控制、临时更新、开发标签隔离与预设阶段门槛。

```bash
python -m workflows.tqg.minimal_validation.run \
  --cache-path /path/to/Mutagenicity/processed/data.pt \
  --output /path/to/new-run
```

- 先固定输入、配置、代码和全部抽样/置乱；仅在已提交且清洁的代码上启动真实训练。
- `diagnostics.py`：v1 共用响应/Gram、v2 共用 V/W；PERM 仅换 Y；MEAN-Y 保留逐图预测；RAND 使用各版本实际可执行空间。主 FP32 步与实际参数转 FP64、理想 FP64 步分别评价，不重新计算查询。
- `run.py`：每种子一次 fit-only 预训练；每块先封存两版反馈，再授权 A/B 开发标签访问；保存原始几何、梯度和更新方向，源模型不变。
- `decisions.py`：三个种子等权，保留零反馈和失败，按原门槛判断各版本是否进入 M2。
- `training.py`：条件执行 50 步。跨目标共用源基线；固定 PERM-1；保存 0/10/50；全部臂预测封存后才访问开发标签。

`--resume` 只复用已完成且哈希匹配的源缓存/M1块。未完成的源模型、M1块或 M2 轨迹会明确拒绝复用；不会自动删除、补写或把不完整记录算作成功。没有任何 final 标签读取入口。

验收见 [test_minimal_validation.py](../../../tests/test_minimal_validation.py)。CPU/单线程；三份 100epoch 新源模型，48个版本诊断块。正式训练与合成验收分开记录。查询的确定性重放属于额外工程检查，不另算独立实验块。
