# 新 SEGA式 TQG v1/v2 最小验证原始归档

日期：2026-11-07。执行提交 [`140a73b`](https://github.com/chx739/TQG-GDA/tree/140a73becb112d181a36b66dadc0b8cce38e55fb)；[可读结果与表格](../../docs/results/tqg_sega_minimal_20261107/README.md)；[原冻结方案](../../docs/research_plans/tqg_sega_minimal_20261106/README.md)。

**已完成3份源模型、48个版本诊断块；两版均未过M1，M2未触发。** 没有读取final标签。合成工程验收见[执行器检查](../../docs/validation/tqg_minimal_driver_20261107/README.md)。

- `raw/`：原样JSON/JSONL/CSV，包括源训练history、结果、标签访问、全部批次与置乱、配置/代码SHA及完成记录。
- `archives/source_checkpoints.tar.gz`：三个新源模型；对应关系见[checkpoint_index.json](checkpoint_index.json)。
- `archives/M1_seed*_0to*.tar.gz`：全部48块的两版查询几何、响应、系数/读出、反馈梯度、Oracle梯度和实际临时步方向。每包一个种子/任务的四个配对块。
- `archives/execution_code.tar.gz`：固定提交的执行代码与协议。原始数据缓存未打包；输入身份与既有域索引链接保留在协议中。
- [artifact_manifest.json](artifact_manifest.json)：全部247个运行文件及各压缩包的大小/SHA；[publication_audit.json](publication_audit.json)确认所有打包字节与本机原件一致。
- [archive_replay_audit.json](archive_replay_audit.json)：在临时目录解包后，不读取数据集，独立复算结果完全一致；70个执行代码/协议文件SHA匹配。
- [补充梯度诊断CSV](derived_gradient_diagnostics.csv)：从已封存向量用FP64复算，补齐QUERY-CE、ORACLE-A、FIT-CE及各范数/余弦字段；没有新增训练或标签访问，也未更改主结果与门槛。
- [数值审计汇总](numeric_audit.json)与[运行时统计](runtime_profile.json)：包括FP32写回、主查询中位数/p95和估算成本；估算不是已执行M2。

压缩包合计 **193.14 MiB**，单包最大 **33.04 MiB**。完整张量分包保存，源码和可读结果仍可直接浏览。

在本目录恢复张量文件后，可不读取数据集而复算矩阵关系、门槛和SHA：

```bash
for archive in archives/source_checkpoints.tar.gz archives/M1_*.tar.gz; do
  tar -xzf "$archive" -C raw
 done
python audit/recompute.py raw --output audit_replay.json
```

`raw/`保留原记录中的本机路径用于追溯；复算器按本目录中的相对结构查找源checkpoint。运行目录原件仍保留，历史实验、旧协议和本地汇报草稿均未改写。审计不读取目标原始标签；它验证已经封存的开发诊断记录，不能证明研究假设成立。
