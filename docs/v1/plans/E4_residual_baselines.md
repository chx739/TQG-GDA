# E4：平均残差与带常数项的响应回归

这两个设置是验证模块贡献的对照，不替代原设计稿。默认配置仍用原 TQG：目标选方向、共享 RMS、无常数项岭回归。

## 平均残差

先在当前 source query 批次计算 `r = onehot(y) - softmax(z)`，再取图均值 `mu = mean_graph(r)`。
每张目标图使用同一个 `mu` 作为纠错，参考输出为 `sg(z_target + mu)`。

例如源图某一类的残差是 +0.2、−0.1、+0.2，平均为 +0.1，就给每张目标图该坐标加 +0.1。
实际会同时处理所有类别坐标。

它也就是“仅常数项、不惩罚常数项”的最小二乘解，不能把二者当作两种独立基线。
每个训练 step 重新计算均值，不训练额外网络，不做参数探测。

- 配置：`mean_residual_q16.json`、`mean_residual_q32.json`。
- 接口：`feedback.build_mean_feedback`。
- `adaptation.feedback_predictor = "mean_residual"`。
- 仍保留 source fit/query 划分，用相同 source fit CE 和目标反馈损失更新一次 Adam。

## 响应回归加常数项

原 TQG 预测 `Phi a`。这个对照预测 `Phi a + b`，让 b 负责共同偏差，让响应权重 a 解释图之间的差异。

目标是：

```text
mean_graph sum_class (Phi a + b - r)^2 + ridge * ||a||^2
```

a 使用岭回归惩罚，b 不加惩罚。实现步骤：

1. 按原方法生成方向、响应与两域共享 RMS；这一步不改。
2. 仅沿源拟合图维度计算 `Phi_mean` 和 `r_mean`。
3. 对 `Phi - Phi_mean` 与 `r - r_mean` 解原岭回归，得到 a。
4. `b = r_mean - Phi_mean @ a`。
5. 目标与独立源图使用 `Phi_new @ a + b`；不能重新用评价图计算均值。

- 配置：`intercept_q16.json`、`intercept_q32.json`。
- 接口：`feedback.fit_residual_with_intercept`；`build_feedback(..., predictor="responses_with_intercept")`。
- `adaptation.feedback_predictor = "responses_with_intercept"`。
- `Feedback.intercept` 保存停止梯度的 b；使用 `heldout_response_features` 后仍需加上这个 b。
- 无有效响应方向时，带常数项的变体退化为平均残差；默认 TQG 仍退化为源 CE 更新。

所有方向、响应、残差、a、b 和参考输出都停止梯度。最终只有普通 GNN 的 source CE 和目标平方反馈损失反向传播。

带常数项的变体增加了 C−1 个有效自由度；平均残差又便宜得多。可以比较同样训练步数，不能把它们称为同时间或同模型容量的比较。

## 实验位置

[E4 冻结方案](/home/hxchen/TQG_runs/20261024_residual_baselines_e4/experiment_plan.json) 与 [实验报告](/home/hxchen/TQG_runs/20261024_residual_baselines_e4/run_report.md)。
源诊断先保存并封存，再进行完整训练和目标开发集评价；最终保留集不参与本轮。
