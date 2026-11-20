# R7 源校准与类别先验对照

2026-11-21 完成。按[登记协议](accuracy-and-prior-control.md)使用 attempt-020/P 主宽度 τ=0.005 的两个 Mutagenicity 边数开发方向，各五个种子，共 10/10 个块。M0→M1 每种子有 777 张 A 图和 307 张 H 图，M0→M3 有 747 张 A 图和 338 张 H 图。两方向共享 5 个冻结源模型，每方向五个种子共享目标图。

## 普通 Accuracy

表中为 H 上五种子平均准确率（%）。Avg 对两个方向等权。B0、SourceCal、MLLS 与 J_argmax 都用固定 argmax，J 保留登记的均衡决策。

| 方法 | M0→M1 | M0→M3 | Avg |
| --- | ---: | ---: | ---: |
| B0 | 72.70 | 65.44 | 69.07 |
| SourceCal | 73.68 | 65.98 | 69.83 |
| MLLS | 72.96 | 59.41 | 66.19 |
| J_argmax | 74.79 | 62.84 | 68.81 |
| J | 72.12 | 65.80 | 68.96 |
| B0_balanced | 72.31 | 65.68 | 69.00 |
| SourceCal_balanced | 73.81 | 66.04 | 69.92 |
| MLLS_balanced | 73.81 | 66.04 | 69.92 |

下表为指明配对的 Accuracy 差（pp）及 95% 区间。沿用 P 的 2,000 次按目标域、类别分层的图重采样，同一目标域内所有方法及种子共用抽样，不重采样训练种子。完整表另列每 seed 差值、训练种子 SD、各类召回和净纠错。

| 配对 | M0→M1 | M0→M3 | Avg | Avg 种子 SD |
| --- | ---: | ---: | ---: | ---: |
| SourceCal − B0 | +0.98 [−0.13, +2.08] | +0.53 [−0.83, +2.01] | +0.75 [−0.12, +1.64] | 0.88 |
| MLLS − SourceCal | −0.72 [−3.26, +2.02] | −6.57 [−9.23, −3.91] | −3.64 [−5.50, −1.76] | 2.21 |
| J_argmax − MLLS | +1.82 [−1.04, +4.56] | +3.43 [+1.48, +5.50] | +2.63 [+0.91, +4.29] | 4.03 |
| J − MLLS | −0.85 [−4.10, +2.35] | +6.39 [+3.91, +8.82] | +2.77 [+0.79, +4.79] | 1.89 |
| J_argmax − SourceCal | +1.11 [−0.20, +2.35] | −3.14 [−4.79, −1.42] | −1.01 [−2.03, +0.004] | 3.92 |
| J − SourceCal | −1.56 [−3.32, +0.20] | −0.18 [−1.36, +1.07] | −0.87 [−1.92, +0.20] | 1.42 |
| J − MLLS_balanced | −1.69 [−3.32, −0.07] | −0.24 [−1.42, +0.95] | −0.97 [−1.97, +0.10] | 1.26 |

两种 J 决策都超过普通 MLLS 的两方向平均。普通 MLLS 在 M0→M3 的 5/5 种子均低于 SourceCal。J 两种固定决策在两方向平均都低于 SourceCal。M0→M1 的 J_argmax 比 B0 高 2.08 pp，且五种子均改善。相对 SourceCal 的优势为 1.11 pp，3/5 种子改善，图重采样区间跨零。M0→M3 的 J_argmax 低于 SourceCal 3.14 pp，是两方向中差值较差的一侧。两方向的固定决策表现不同，按 H 为每个方向挑选决策会改变比较口径。

## 排序、BA 与 NLL

SourceCal 与 MLLS 在全部十块严格保持 B0 的 H 样本排序，未新增浮点并列。两种 J 决策使用相同概率与排序。相对 B0 或 MLLS，J 的 AUROC 在 M0→M1 平均增加 0.417 pp，五种子中四个增加，范围为 −0.133 至 +0.739 pp。M0→M3 平均增加 0.0007 pp，两个种子增加，范围为 −0.428 至 +0.456 pp。M0→M1 的排序变化是方向相关线索，当前两个开发方向缺少可复现的 Accuracy 优势。M0→M3 的单种子 AUROC 图样本区间见[五种子诊断](iteration-14-p-risk-replay.md#分类动作与排序)。

下表为 H 的两方向等权均值。均衡动作只改变分类决定，NLL 由原概率计分。

| 方法 | BA（%） | AUROC（%） | NLL（nats/图） |
| --- | ---: | ---: | ---: |
| B0 | 69.54 | 76.215 | 0.62189 |
| SourceCal | 69.79 | 76.215 | 0.60264 |
| MLLS | 62.50 | 76.215 | 0.65100 |
| J_argmax | 68.41 | 76.424 | 0.63397 |
| J | 69.99 | 76.424 | 0.63397 |
| SourceCal_balanced | 70.04 | 76.215 | 0.60264 |
| MLLS_balanced | 70.04 | 76.215 | 0.65100 |

## 模型、标签与复现

设 B0 的正类 log-odds 为 $m$，校准后为 $c=m/T+b$。在每个源模型的 216 张验证图上以二元 NLL 拟合 $T,b$。这些图也用于原 R5 源模型的最低验证 CE 选轮次。校准温度约束在 $[0.05,100]$，以 $(T,b)=(1,0)$ 初始化。该二分类形式与 Alexandari 等的带偏置温度校准对应。本实现固定复用 R5 验证集，使用单个二元截距、显式温度范围与先验下限，直接在 log-odds 空间计算。原文讨论了跨任务校准与最大似然标签偏移估计，实测仍依赖类条件分布在两域间的稳定性。[Alexandari 等，ICML 2020，第 3 节](https://proceedings.mlr.press/v119/alexandari20a/alexandari20a.pdf)

源先验 $\pi_S$ 为源验证图校准概率的经验均值。以 $\pi_S$ 初始化，用无标签 A 的校准概率做二元 EM，估计 $\pi_A$。先验限制在 $[10^{-8},1-10^{-8}]$，EM 按先验步长 ≤ $10^{-10}$ 停止，上限 100,000 次。逐图概率用稳定 log-odds 算法，先验下限只用于 EM 的先验变量。十个估计均在 38–280 步收敛，$\pi_A$ 在 0.171–0.722 之间，最终均未碰下限。

MLLS 的 log-odds 为 $c+\operatorname{logit}(\pi_A)-\operatorname{logit}(\pi_S)$。对其减去对应的均衡先验 log-odds 后得到 $c-\operatorname{logit}(\pi_S)$，与 SourceCal_balanced 相同。十块均衡决策逐图一致，数值抵消最大误差 $1.78\times10^{-15}$。排序和抵消由 OBS 前检查固定，H 标签未参与拟合、先验估计或阈值选择。NCI1 仍依照原协议等待进入条件。

从仓库根目录执行，输出选择新目录。入口复用 R7 的封存、R5 标签认证、决策与配对重采样。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.scripts.audit_prior_control \
  --config experiments/mutagenicity/r7_共同解释与共享更新验证_20261119/configs/r7-p-accuracy-prior-control.json \
  --output /tmp/r7-accuracy-prior-control
```

完整参数及十块 H 预测在新 OBS 中封存后统一打开 H 标签。24 项 R7 工程测试通过。独立输出目录重放的 15 个参数、预测与结果文件逐字节相同，原有四臂共 40 个 H 端点复核通过。区间条件于固定模型和开发图，适用范围仍是这两个方向。

[执行配置](../results/iteration-14/accuracy-prior-control/executed-config.json) · [端点 CSV](../results/iteration-14/accuracy-prior-control/endpoints.csv) · [逐种子差值](../results/iteration-14/accuracy-prior-control/paired-seeds.csv) · [配对区间与种子 SD](../results/iteration-14/accuracy-prior-control/paired-intervals.json) · [拟合参数](../results/iteration-14/accuracy-prior-control/parameters.json) · [OBS](../results/iteration-14/accuracy-prior-control/OBS.json) · [FINAL](../results/iteration-14/accuracy-prior-control/FINAL.json) · [复核记录](../results/iteration-14/accuracy-prior-control/verification.json)
