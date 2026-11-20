# R10｜冻结 GRADE 的目标池交叉推断

**用 A3 适配的模型在两个留出域的 Accuracy 点估计都较高。本域匹配的两方向等权增量为 +0.166 个百分点，95% 配对区间为 [−0.285, +0.604]。** H3 偏好本域模型，H1 的点估计偏好异域模型。本轮证据与 A3 checkpoint 在这两个留出域整体较强的解释相容，区间仍容许这些差异来自有限图样本。[登记协议](reference-crossover.md) · [原 R10 实验](../report.md)

## 交叉格与同架构纯源

表内为五种子均值 ± 样本标准差，单位为 %。A1、A3 分别表示用对应目标池训练的原生图级 GRADE。所有方法在每个 H 使用完全相同的图与顺序，类别取原生 argmax。

| 冻结模型 | H1 Accuracy | H1 Macro-F1 | H3 Accuracy | H3 Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| GRADE A1 | 71.987 ± 1.219 | 69.051 ± 2.234 | 57.811 ± 1.947 | 55.898 ± 2.839 |
| GRADE A3 | 72.248 ± 1.183 | 69.230 ± 2.187 | 58.402 ± 1.213 | 56.654 ± 2.185 |
| SourceOnly | 72.313 ± 1.004 | 69.774 ± 1.506 | 57.929 ± 0.675 | 55.997 ± 1.767 |

[交叉格](../results/attempt-004-crossover/matrix.csv) · [逐种子指标](../results/attempt-004-crossover/endpoints.csv)

## 逐图配对

| 评价域 | 比较 | Accuracy 差（pp） | 种子 SD（pp） | 95% 配对区间（pp） | 纠正／误改 |
| --- | --- | ---: | ---: | ---: | ---: |
| H1 | A1 − A3 | −0.261 | 0.146 | [−0.912, +0.327] | 12／16 |
| H3 | A3 − A1 | +0.592 | 0.754 | [0.000, +1.183] | 18／8 |
| H1、H3 等权 | 本域 − 异域 | +0.166 | 0.351 | [−0.285, +0.604] | — |
| H1 | A1 − SourceOnly | −0.326 | 0.515 | [−1.238, +0.651] | 29／34 |
| H1 | A3 − SourceOnly | −0.065 | 0.425 | [−1.042, +1.042] | 29／30 |
| H3 | A1 − SourceOnly | −0.118 | 1.428 | [−0.828, +0.592] | 25／27 |
| H3 | A3 − SourceOnly | +0.473 | 0.799 | [−0.355, +1.302] | 29／21 |

| 种子 | H1 本域−异域（pp） | H3 本域−异域（pp） | 等权交互（pp） |
| --- | ---: | ---: | ---: |
| 3405 | −0.326 | +0.888 | +0.281 |
| 3406 | −0.326 | +0.296 | −0.015 |
| 3407 | 0.000 | 0.000 | 0.000 |
| 3408 | −0.326 | +1.775 | +0.725 |
| 3409 | −0.326 | 0.000 | −0.163 |

H1 的 307 张图上，A1/A3 五种子共分歧 28 次，平均分歧率 1.824%。H3 的 338 张图上共分歧 26 次，平均 1.538%。分歧率是在这些冻结动作之间选择时可额外分对的比例上界。纠正和误改按同图、同种子统计，跨种子求和。交互先逐种子计算两域半和，再对五种子等权。区间沿用 R7 的 2,000 次按目标域和类别分层图重抽样，同一域内五种子和三模型共用抽样权重。每次重抽样内先计算两域交互，再取分位数。区间条件于冻结模型，种子 SD 单列。

[配对汇总](../results/attempt-004-crossover/paired-summary.csv) · [逐种子纠正和误改](../results/attempt-004-crossover/paired-seeds.csv) · [重抽样审计](../results/attempt-004-crossover/cohort-audit.json)

## 同源顺序、模型与归档

每个种子使用 868 张源训练图，同一种子的三个模型具有相同的初始参数哈希和源角色。A1 有 777 张图，A3 有 747 张图。按已认证的每步源图序列 SHA-256 比较，**两个 GRADE 训练臂的 200/200 步源图顺序在全部五个种子相同**。两个 GRADE 臂分别与 SourceOnly 的源图顺序均为 0/200 步相同。目标池大小、目标组成和其前向消耗的随机数可以共同影响训练。实验估计的是整套原生适配过程的参考域与评估域交互，范围为这两个开发方向。[逐种子顺序比较](../results/attempt-004-crossover/source-sequences.csv)

运行于 2026-11-21，执行提交为 `bb43c0c`。原始 15 个 checkpoint 仍在 `/home/hxchen/research/GDA/TSBA/experiments/r10/attempt-001` 内，补充运行的独立档案为 `attempt-004-crossover`。全部 30 个预测单元先进入 OBS，重载后才从原权限读取 H。20 个对角及纯源预测和原 R10 的 logits 逐项相等，类别差异均为零。工程测试 23 项通过。另对 30 个端点、35 条逐种子配对、7 个配对区间及封存哈希重算核对。

[运行配置](../results/attempt-004-crossover/executed-config.json) · [角色](../results/attempt-004-crossover/roles.json) · [模型与文件校验](../results/attempt-004-crossover/lineage.json) · [逐图原结果回放核验](../results/attempt-004-crossover/replay.csv) · [OBS](../results/attempt-004-crossover/OBS.json) · [FINAL](../results/attempt-004-crossover/FINAL.json) · [复算](../results/attempt-004-crossover/verification.json) · [运行日志](../results/attempt-004-crossover/run.log) · [工程测试](../results/attempt-004-crossover/engineering-tests.log)
