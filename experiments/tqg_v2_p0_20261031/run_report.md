# Run Report

D0通过；D1未通过，D2未通过。当前真实源标签没有在至少两个任务上同时达到对打乱／随机反馈的预定方向优势。不能据此把旧实验的小幅分类变化归因于稳定的监督迁移，也不能从这次有限批次诊断直接推论所有TQG变体无效。

## Executed as planned

- 冻结原计划与哈希；D0→D1→D2，不运行后续阶段。
- 3任务×5模型种子×3快照×8块＝360个D1块；D2前4块共180，6480临时步、180个冻结参考。
- 无标签几何先封存，之后显式development Oracle；最终标签不读取；临时模型不保存或作为训练起点。
- FP64、CPU、5个单线程worker；共同输入、读取器、随机数、源fit梯度；原模型和buffer未变。
- 本轮新增6项单元测试及既有16项v2数学回归测试通过；D0另有50项验收及9个权限拒绝记录。
- 所有逐块指标、5种子汇总、TensorBoard、源模型SHA、代码历史与计划保持可追溯。pilot独立，不计正式样本。

## Deviations from plan

- 没有改变数据、种子、批次、臂、学习率、阈值和执行精度。GPU驱动不可用，按方案CPU默认配置执行。
- 现有TensorBoard缺失，补齐2.20.0；实际依赖版本固定在执行副本requirements.txt，不冒充原环境已具备。
- pilot单worker外推约135.9秒，正式5worker墙钟245.7秒；外推未充分覆盖并发竞争／加载。未超8小时包上限；后续预算应用并发pilot改进计时，不改变本轮样本数。
- 使用functional独立参数张量执行临时步，D0用真实torch.optim.SGD核对执行等式；不保存数千个临时Oracle checkpoint。

## Failed runs

- 无正式失败作业、数值回退或种子删除。一次汇总入口的相对路径调用失败，保留aggregation_relative_path_failure.log；改用绝对stage路径成功汇总，原始实验与判据未变。主仓库已修复后续入口。有限但未收敛的校准块为0/360，完整记录，未混称精确最优解。
- 研究判据未通过与运行错误分开；失败／不充分结果保留，不修改阈值。

## Compute consumed

- 规划硬上限8小时；GPU实际0小时。
- D1/D2正式墙钟245.7秒（4.09分钟）。5个单线程worker累计墙钟1093.0秒，约0.3036 CPU-worker小时（含I/O，非纯CPU user-time）。
- 固定pilot另计2.097秒；D0／预检／文档归档不计入上述正式D1/D2墙钟。
- 每worker峰值RSS约804.4MiB；高维几何缓存在本地保留，公开指标与SHA，避免无限上传向量。

## Headline number

D1达到门槛任务0/3；D2达到主学习率门槛任务0/3。是否允许当前版本进入D9：False。

| 任务 | TRUE余弦（均值±SD） | 打乱标签余弦 | 随机反馈余弦 | TRUE−打乱 | TRUE−随机 | D1 |
|---|---:|---:|---:|---:|---:|---|
| 0→1 | 0.0221 ± 0.1058 | -0.0019 | 0.0199 | 0.0240 | 0.0022 | 未通过 |
| 0→2 | -0.0509 ± 0.1316 | 0.1108 | -0.0058 | -0.1616 | -0.0450 | 未通过 |
| 0→3 | 0.0307 ± 0.1986 | 0.1179 | 0.0095 | -0.0872 | 0.0212 | 未通过 |

| 任务 | 真实反馈额外降CE（%） | 五种子正增益数 | 等幅度Oracle额外降CE（%） | 有害更新率（%） | D2 |
|---|---:|---:|---:|---:|---|
| 0→1 | -0.007446 | 1/5 | 0.005949 | 51.67 | 未通过 |
| 0→2 | -0.001677 | 1/5 | 0.007218 | 41.67 | 未通过 |
| 0→3 | 0.005664 | 3/5 | 0.021887 | 46.67 | 未通过 |


## Reproducibility manifest

- 冻结执行Git提交：99df2b4326ae30c783ae7294416514c816f624bd；公开代码副本与execution_history.bundle共同保存该历史。
- 源仓库实现提交：76bfb8a95f7dde04c8cd3063443483d4ce3d3ef3。
- 计划SHA256：cd67ab90d1bcad45bd0dfdfdca4ae07833ecbc905b90e858b19f1859cd709c35；protocol中全部文档保持原始哈希。
- 硬件：cpu:Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz;threads=1;FP64；种子3405—3409；配置configs/p0_diagnostics.json。
- 输入：snapshot_manifest.json的45引用／35个独立文件，已有归档checkpoint按SHA核对，不重复发布源模型。
- 命令：在code目录依次运行p0_experiment.py prepare/d0/pilot/run --repo /home/hxchen/TQG-GDA --stage ../，再运行p0_summarize.py --stage /home/hxchen/runs/2026-10-31-tqg-v2-diagnostic-plan/stage3_execution；运行前OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1。
- 原始：gradient_blocks.csv、one_step_blocks.csv、logs/*/*.jsonl及TensorBoard；汇总：results.csv、results_summary.json；简明解释：结果解读.md。
- 限制：非盲测、非长期适配结论、非条件期望梯度证明、非统计显著性结论。
