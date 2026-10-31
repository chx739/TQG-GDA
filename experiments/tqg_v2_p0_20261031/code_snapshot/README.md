# P0诊断实现与复现

参考[冻结方案](research_plans/tqg_v2_20261031/README.md)。本实现只执行D0/D1/D2；不启动D3—D9，不改变生产训练算法，不产生可部署的Oracle模型。

- `p0_experiment.py`：来源预检、工程验收、固定pilot、5模型种子的诊断执行。
- `tqg_gda/p0_diagnostics.py`：无标签几何、显式development Oracle、梯度及functional单步对照。
- `p0_summarize.py`：先在模型种子内平均块／快照／置乱，再汇总5种子；判据沿用冻结方案。
- `configs/p0_diagnostics.json`：执行配置，CPU/FP64、5个单线程worker、评价微批32、8小时包上限。评价微批只控制内存，B池仍完整271张且按图加权。

运行目录为 `/home/hxchen/runs/2026-10-31-tqg-v2-diagnostic-plan/stage3_execution/`，代码在其 `code/` 子目录单独保存Git历史，计划从同run的 `stage2_method/` 固定读取。初始化时复制本仓库的诊断入口、`tqg_gda/`、配置和依赖锁定文件；记录源仓库提交与全部执行文件哈希。每个结果行的git_commit指这个冻结执行副本，不随主仓库后续的结果归档提交变化。

在冻结的 `code/` 目录中依次执行：

```bash
python p0_experiment.py prepare --repo /home/hxchen/TQG-GDA --stage ../
python p0_experiment.py d0 --repo /home/hxchen/TQG-GDA --stage ../
python p0_experiment.py pilot --repo /home/hxchen/TQG-GDA --stage ../
python p0_experiment.py run --repo /home/hxchen/TQG-GDA --stage ../
python p0_summarize.py --stage ../
```

prepare和D0失败时不继续；已有输出目录不覆盖，修复留新attempt并说明原因。pilot固定为0→1、3405、初始快照、第0块，包含完整D1和36个D2临时步，用于测时且与正式统计分开。预算不足不自动减少种子、批次或臂。

无标签几何没有标签参数，拒绝携带y的输入。Oracle只在显式启用、查询封存且ID属于development时返回标签；final请求在取值前拒绝。B梯度与标签可在同快照内缓存，仍属于隔离的诊断数据。每一单步臂使用独立参数张量，原模型和buffer哈希必须不变。求解器非有限异常直接报错；有限但未收敛的解完整保留并标记。

日志为逐块JSONL及本地TensorBoard；未定义指标保留null，不以0代替。保存局部无标签几何缓存以支持后续复核，但不把全部高维向量无上限提交GitHub。公开归档包括原始指标、角色ID、来源／哈希、执行代码和结果解释，输入模型仍引用已有35个独立checkpoint文件（45个分析引用），不重复上传。
