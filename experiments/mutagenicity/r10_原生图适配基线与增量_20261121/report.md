# R10 同图、同权限的 GRADE 比较

2026-11-21 完成五个 source-only 和十个官方 graph-mode GRADE，共 15/15 模型、3,000 次 Adam 更新。十块预测统一封存并计分。**GRADE 的两方向等权 Accuracy 为 65.19%，同架构 source-only 为 65.12%，配对差 +0.074 pp [−0.504, +0.710]。** 这两个开发方向未显示稳定适配增量。

## 主比较

| 留出方向 | H / 种子 | GRADE Accuracy | Source-only Accuracy | 配对差 [95% 区间] | 五种子差值 SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0→M1 | 307 | 71.99% | 72.31% | −0.326 [−1.238, +0.651] pp | 0.515 pp |
| M0→M3 | 338 | 58.40% | 57.93% | +0.473 [−0.355, +1.302] pp | 0.799 pp |
| 两方向等权 | 307 / 338 | 65.19% | 65.12% | +0.074 [−0.504, +0.710] pp | 0.638 pp |

M0→M1 五个种子的净纠错为 −2、0、+1、−3、−1，合计 29 次纠正、34 次损害。M0→M3 为 −2、+3、+5、0、+2，合计 29 次纠正、21 次损害。五次训练复用各方向的 H 图，计数可能重复同一张图。先在种子内等权平均方向，再等权平均种子。2,000 次置信区间沿用 R7 按目标域与类的配对图重采样，同一域共享抽样，条件于冻结模型。逐种子净纠错、配对区间、每类召回和样本数全部归档。

GRADE 的 BA 比 source-only 等权低 0.175 pp，NLL 低 0.00247 nats/图。M0→M1 类 0 召回提高 0.89 pp，类 1 下降 2.69 pp。M0→M3 类 0 提高 2.24 pp，类 1 下降 1.13 pp。两方向分类和损失变化可分开核对。

## 同一批 H 上的历史开发对照

数值为 Accuracy（%）。各方法保留原冻结决策。TargetSlope 来自事后固定均值诊断，Joint 来自 R8 主臂并逐图等于 SourceCal。

| 方法 | M0→M1 | M0→M3 | 两方向等权 | 等权 BA | 等权 NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRADE | 71.99 | 58.40 | 65.19 | 63.59 | 0.67704 |
| 同架构 source-only | 72.31 | 57.93 | 65.12 | 63.76 | 0.67951 |
| B0 | 72.70 | 65.44 | 69.07 | 69.54 | 0.62189 |
| SourceCal | 73.68 | 65.98 | 69.83 | 69.79 | 0.60264 |
| SourceFusion | 74.40 | 66.57 | 70.48 | 70.31 | 0.59347 |
| R7 J_argmax | 74.79 | 62.84 | 68.81 | 68.41 | 0.63397 |
| R8 Joint | 73.68 | 65.98 | 69.83 | 69.79 | 0.60264 |
| R8 TargetSlope | 75.11 | 67.99 | 71.55 | 71.36 | 0.59385 |

GRADE 相对历史 B0 的等权 Accuracy 差为 −3.88 pp [−6.89, −0.80]，其中 M0→M3 低 7.04 pp。配对文件也分别列齐两个新臂对每个历史臂的结果。同架构比较用于估计 GRADE 适配增量。与历史方法的比较还包含源表示、训练与决策规则的差异。两个方向使用本仓库的边数划分，且已参与研发。作者原始划分的绝对成绩和跨数据集优势仍需各自的验证。

## 执行与复核

官方固定版本的 graph-mode GRADEBase 和 GradReverse 原样调用。JS 损失、梯度反转调度、shuffle、Adam 与 200 轮 full-batch 保持。初始参数和源 fit 按种子匹配。run loader 在 GRADE 中迭代 S/A，在 source-only 中只迭代 S，原生随机数消费和源样本顺序随臂变化。该对照比较各臂完整训练过程。源码、Git blob、MIT 许可、逐步等价及运行恢复记录见[实现核验](docs/implementation-check.md)。

CUDA 不可用，CPU 每模型两个 Torch 线程、一个 interop 线程。首种子的 source-only、适配 M1、适配 M3 分别耗时 139.5、201.3、169.5 秒，进程峰值为 2,305、2,215、2,383 MiB。全部模型训练耗时为 127.4–212.5 秒，进程峰值 2,038–2,405 MiB。每模型完整采样日志、启动和退出的 PID、命令与回执均已归档。独立验收覆盖 15 份完成清单及 3,000 行更新。20 份未计分 H logits 从 checkpoint 重载后逐值一致，全部十块封存后一次性读取 H 标签计分。

训练提交为 `ca91fcd`，评分提交为 `4fd8f49`。原始 checkpoint、二进制预测、日志和源码 tar 在 `/home/hxchen/research/GDA/TSBA/experiments/r10/attempt-001`。Git 档案保留索引、所有哈希、逐轮轨迹、配对结果和可定位的进程记录。

策略侧在 FINAL 发布后独立按 R7 的图 ID 对齐 H 标签与八臂预测。80 个端点、130 个逐块对照与 117 行配对区间均可复现，最大数值差小于 7×10⁻¹⁴。FINAL 文件的 SHA-256 为 `e30d36d00fb06ccadd7f7da8010ed30aeb8108149964e913aee3eb0dcf7ccc98`。R7 共享更新的普通 Accuracy 高于 GRADE 3.62 个百分点，却低于只用源验证标签的 SourceCal 1.01 个百分点。两项比较共同解释了为什么超越一个已发表模型仍不足以把改善归因于共享更新。[R7 同决策对照](../r7_共同解释与共享更新验证_20261119/docs/accuracy-and-prior-control-results.md)

[端点](results/attempt-001/evaluation/endpoints.csv) · [逐种子对比](results/attempt-001/evaluation/paired-seeds.csv) · [方向及等权区间](results/attempt-001/evaluation/paired-intervals.json) · [全量训练验收](results/attempt-001/observation/training-verification.json) · [历史预测身份与重载](results/attempt-001/observation/lineage.json) · [实际配置](results/attempt-001/executed-config.json) · [过程档案索引](results/evidence.json) · [OBS](results/attempt-001/OBS.json) · [FINAL](results/attempt-001/FINAL.json)
