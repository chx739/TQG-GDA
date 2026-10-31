# Run Report

D3未通过、D4未通过。主配置的复杂校准与响应签名均未获得预定增量；主比较全部收敛。详见结果解读.md。

## Executed as planned

- 原计划／数据／快照／P0缓存哈希固定；3任务、5模型种子、3快照，360真实目标块和120源内块。
- D3为5400逻辑校准配置；D4为1800真实目标、600源内梯度记录。原幅度／等范数实际功能参数更新共8400次，均独立从原θ开始。
- 每块完整封存全部W后才读取Oracle标签；development与源query-audit分别记日志；最终标签未取值。
- 源bank98和audit98原ID排序；A16、B82；源fit16、源侧快照canonical0→1；不重复计三个任务。
- D3/D4新增及核心回归28项测试通过，3个真实任务旧W/预测/响应精确回放。独立CONST解析观测器另有1项最优解一阶条件／收敛解测试通过。
- D3原JOINT0.01回放P0、ISO与MEAN每α等价逐块核对；未换目标方向、丢种子、减批次或改变门槛。

## Deviations from plan

- 原方案未细列D3单步子矩阵，本次在新标签读取前固定为D2前4块、主η、两幅度、全部15臂；D4五核均joint，不启动可选MEAN矩阵。执行细节见experiment_manifest.json，未改研究门槛。
- 正式solve_seconds含内部source分解，未逐次单列分解计时。补充在相同360块上无标签单worker测source eig，中央値0.002013秒；这不是正式并发时长拆分或效率优势证明。
- 发现68个CONST控制未收敛后，另做二分类解析解核对；只作数值审计，不重跑或替换正式指标，不改变任何求解器上限。
- D4 HA真实目标与D3主JOINT的360个相同求解复用并明确标记；逻辑配置数与物理求解耗时分开。源几何构造器另外有120次内部HA计算，不计成新的独立条件。

## Failed runs

- 无正式异常退出、非有限回退或种子缺失。
- D3非收敛0/5400；D4非收敛68/2400，全部为CONST真实目标组。有限近似权重、gap和termination原样保存，不称精确最优。
- 研究判据未通过与运行失败分开。不能按alpha／snapshot事后选较好组改判。

## Compute consumed

- 规划8小时硬上限；正式D3/D4墙钟210.6秒（3.51分钟）。
- 五个单线程worker累计墙钟1015.0秒，约0.2819 CPU-worker小时（含I/O，不是纯CPU user-time）；GPU0小时。
- 固定pilot另计3.010秒，补充无标签数值计时另计49.748秒；工程验收与汇总／归档不计入正式墙钟。
- 最大worker RSS约733.5MiB。

## Headline number

D3过门槛任务0/3，D4过门槛任务0/3；不据此进入D9。Ω已强烈专门化但收益不稳定，HA等范数单步在三个真实任务上均未超过H。

| 任务 | MEAN／ISO余弦 | JOINT余弦 | JOINT−ISO | 同幅单步额外降CE（%） | D3 |
|---|---:|---:|---:|---:|---|
| 0→1 | 0.0238 | 0.0221 | -0.0017 | -0.000421 | 未通过 |
| 0→2 | -0.0635 | -0.0509 | 0.0126 | -0.000734 | 未通过 |
| 0→3 | 0.0175 | 0.0307 | 0.0132 | 0.001299 | 未通过 |

| 相似度核 | 0→1余弦 | 0→2余弦 | 0→3余弦 | 源内留出余弦 |
|---|---:|---:|---:|---:|
| H | 0.0192 | -0.0168 | 0.1885 | 0.0453 |
| A | 0.1051 | -0.0151 | 0.0558 | 0.0938 |
| HA | 0.0221 | -0.0509 | 0.0307 | 0.0896 |
| CONST | 0.1274 | 0.0824 | 0.2494 | -0.0237 |
| HA_PERM | -0.0264 | -0.0192 | 0.0132 | -0.0466 |

| 任务 | H原幅度 | HA原幅度 | H等范数 | HA等范数 | D4 |
|---|---:|---:|---:|---:|---|
| 0→1 | -0.003482 | -0.007446 | -0.003482 | -0.006026 | 未通过 |
| 0→2 | 0.015808 | -0.001677 | 0.015808 | -0.004047 | 未通过 |
| 0→3 | 0.044686 | 0.005664 | 0.044686 | 0.005761 | 未通过 |


## Reproducibility manifest

- 冻结执行提交：51fe524d640c4fc95e9ac32521f276040701d500；code_snapshot与execution_history.bundle保存完整历史。
- 源仓库实现提交：2c73c2a767c61052b76367e75b01e73e715fdffc。
- 原计划SHA256：cd67ab90d1bcad45bd0dfdfdca4ae07833ecbc905b90e858b19f1859cd709c35；protocol保持逐字节一致。
- 配置configs/d34_diagnostics.json；cpu:Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz;threads=1;FP64；种子3405—3409；依赖锁定见code_snapshot/requirements.txt。
- source模型45引用／35独立文件见snapshot_manifest；360份P0缓存及索引SHA见p0_cache_manifest，不重复发布模型／几何向量。
- code目录依次运行d34_experiment.py prepare/d0/pilot/run --repo /home/hxchen/TQG-GDA --stage ../，再运行d34_summarize.py --stage ../；线程环境OMP/MKL/OPENBLAS_NUM_THREADS均为1。
- 补充分析源代码与来源见analysis_code；正式结果代码版本不被后续报告／归档提交替换。
- 原始表：calibration_grid_blocks.csv、signature_target_blocks.csv、source_audit_blocks.csv、step_blocks.csv；配对与种子汇总同目录。results.csv每行按模型种子汇总，有代码／硬件／配置／事件标记。
- 开发标签已经观察；源内与目标bank数量不同；经验梯度含噪声；增量实验不替代候选D1/D2与长期分类验证。
