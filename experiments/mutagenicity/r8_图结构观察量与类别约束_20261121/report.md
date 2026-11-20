# R8 图结构观察量与类别约束试点结果

2026-11-21 完成登记的 Mutagenicity 边数划分 M0→M1、M0→M3，五种子共 10/10 块。**Accuracy 晋级条件未通过。** 按登记带宽，十块的 SourceCal 已在矩约束可行集内。主投影和同信息先验对照原样返回，A/H logits 和决策逐值相同，两个方向相对 SourceCal 和 SamePrior 的 Accuracy 差均为 0。主臂两方向等权准确率低于仅源融合 0.654 pp。此门槛只决定是否投入更大开发验证。

## 留出图普通 Accuracy

下表为 H 的五种子均值（%）。Avg 在种子内等权平均两个方向，再等权平均种子。主臂及新增对照使用普通 argmax，历史 J 保留其原登记的均衡动作。零带宽两臂属于登记的敏感性分析。

| 方法 | M0→M1 | M0→M3 | Avg |
| --- | ---: | ---: | ---: |
| B0 | 72.70 | 65.44 | 69.07 |
| SourceCal | 73.68 | 65.98 | 69.83 |
| MLLS | 72.96 | 59.41 | 66.19 |
| v 单独 | 74.92 | 66.15 | 70.54 |
| SourceFusion | 74.40 | 66.57 | 70.48 |
| Joint 主臂 | 73.68 | 65.98 | 69.83 |
| SamePrior 主宽度 | 73.68 | 65.98 | 69.83 |
| JointExact 零带宽 | 68.73 | 66.57 | 67.65 |
| SamePriorExact 零带宽 | 68.14 | 61.36 | 64.75 |
| J_argmax（历史） | 74.79 | 62.84 | 68.81 |
| J（历史均衡） | 72.12 | 65.80 | 68.96 |

SourceFusion 相对 SourceCal 在两方向分别提高 0.717 和 0.592 pp，等权平均 +0.654 pp，95% 配对区间 [−0.050, +1.350] pp，种子 SD 0.889 pp。区间按 R7 登记的目标域 × 类别分层图重采样，2000 次，同一域各方法和种子共享抽样，条件于冻结模型和开发图。

零带宽的 10 个 JointExact 与 10 个 SamePriorExact 都得到有限更新。JointExact 的 Accuracy 相对 SourceCal 在 M0→M1 为 −4.951 pp [−8.406, −1.368]，在 M0→M3 为 +0.592 pp [−2.664, +3.905]。M0→M3 同先验下保留交叉矩相对 SamePriorExact 提高 5.207 pp [3.018, 7.456]，五个种子均提高。相对 SourceCal，M0→M3 的 AUROC 平均提高 3.193 pp，五个种子均为正，NLL 从 0.6659 增至 0.7227，平均 Accuracy 与 SourceFusion 持平。这个方向的零带宽结果保留了排序线索，正式主臂看不到同类更新。M0→M1 的零带宽平均 Accuracy 明显下降，不能为两个方向统一改用零带宽提供证据。

## 观察量及条件均值

来源验证集的类均值差 μ₁−μ₀ 在五个源模型上为 0.295–0.325。真标签解封后的 A 上，M0→M1 两类均值均下移，类 0 的变化为 −0.099 至 −0.033，类 1 为 −0.107 至 −0.028。M0→M3 的类 0 均值上移 0.077–0.131，A 上两类分离仅 0.166–0.223。各类源、A/H 均值及 Welch 区间全量保留。

A 和 H 各有 20 个真标签类条件矩，两池都没有超过对应样本带宽的条目。A 上主带宽最小 SourceCal 约束余量仍有 0.03581，说明当前这些约束同时容许冻结基准和观测到的真标签矩。零带宽强制各目标类均值等于对应源类均值后，A 上平均先验绝对误差相对 SourceCal 从 0.092 增至 0.232（M0→M1），从 0.055 增至 0.134（M0→M3）。目前的带宽容许非小的类均值变化，精确矩所隐含的先验也与真实标签比例有系统偏差。这些是开发图上的机制读数。晋级仍只按登记的 Accuracy 比较，NCI1 保持关闭。

## 复现与证据

执行代码为 `dd12d512434afab239fb90ad8f0df0928e294935`。运行前 R8/R7 相关工程测试 38 项通过。目录检查中的 R8 条目通过，历史 R5 条目仍因缺少 src/__init__.py 失败。完整核验覆盖 R8_OBS/FINAL 成员哈希、A/H 的主臂逐值零更新，以及 50 条复用 R7 H 端点的四项指标。独立按逐图概率重算，零带宽 JointExact 在 A 的最大原始矩绝对残差为 $1.71\times10^{-9}$。H 的零带宽 logits 可由冻结的 A 投影参数逐值重放。

补充的 `cb1b9b0` 与 `685d247` 处理零带宽下 A 均值恰好位于类均值端点的边界情形。正式执行在两个补丁前已冻结代码并启动，保留 attempt-001 原始版本与产物。用修订后的求解器只读重放冻结 A 分数与 H 推断，十块的四个投影臂状态和 A/H 分类动作全部相同，logit 最大绝对差 $5.64\times 10^{-12}$。此补充核验仅使用冻结分数。正式运行的全部目标标签读取发生在 R8_OBS 建立后。

[逐块状态与参数](results/attempt-001/block-status.json) · [源统计和模型](results/attempt-001/source-statistics.json) · [A/H 各臂指标](results/attempt-001/endpoints.csv) · [净纠错及配对差](results/attempt-001/paired-seeds.csv) · [配对区间](results/attempt-001/paired-intervals.json) · [类条件均值](results/attempt-001/conditional-means.csv) · [矩残差和比例误差](results/attempt-001/moment-residuals.csv) · [门槛判定](results/attempt-001/gate.json)

[实际命令和代码版本](results/attempt-001/executed-config.json) · [OBS](results/attempt-001/OBS.json) · [FINAL](results/attempt-001/FINAL.json) · [封存与重放核验](results/attempt-001/verification.json) · [运行日志](results/attempt-001/run.log) · [工程测试](results/attempt-001/engineering-tests.log)

[补丁后冻结分数重放](results/attempt-001/post-run-solver-check.json)

独立审查从封存的逐图预测和相应图 ID 重算 220 行端点、170 行配对、40 行类均值及 480 行矩残差，净纠错数逐项一致，四类浮点结果的最大绝对差不超过 3.34×10⁻¹⁶。按同一批图重建两个域各 2,000 次 bootstrap，153 行区间端点最大差不超过 4.27×10⁻¹⁴。OBS 的 24 个成员和 FINAL 的 7 个成员摘要全部吻合。正式运行在拟合前会读取历史封存文件的字节校验哈希，上述目标标签读取时间指标签值的解码和评分。

针对两个方向的 Accuracy 分化，保留一次事后诊断，固定各臂在 A 的平均预测概率，比较源与目标斜率在这一条件下的净纠错。其结果只用于归因。[固定预测正类比例的诊断](docs/fixed-mass-diagnostic.md)
