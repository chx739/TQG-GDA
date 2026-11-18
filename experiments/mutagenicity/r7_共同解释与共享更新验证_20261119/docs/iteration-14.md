# R7 第十四轮迭代：固定总量的类别比例伪计数

第十三轮固定扩展在 `well_specified` 只有 11/15 个联合通过，未达到 12/15，真实阶段停止。随后在同一批已封存合成输入上，只将类别比例锚点的总伪计数从适配图数 $n_A$ 改为 1，正确设定世界达到 15/15，无偏移保护仍通过。这是形成下一轮假设的配对诊断；独立验证使用新种子 80–94，保留全部四个世界和原门槛。

## 单一改变及其理由

令 $b$ 为冻结 B0 在无标签 A 上的平均类别概率，$m_\theta$ 为候选解释在同一 A 上的平均后验；二者沿用原 $10^{-8}$ 平滑。原观测目标为

$$
\ell_{\mathrm{old}}(\theta)=\ell_{\mathrm{obs}}(\theta)
+\frac{n_A}{n_S+n_A}\operatorname{KL}(b\Vert m_\theta).
$$

每个 A 图贡献一个相同的软类别伪观测，会让 B0 类别比例的偏差持续约束估计。新目标使用固定总量 $\kappa=1$：

$$
\ell_{\mathrm{new}}(\theta)=\ell_{\mathrm{obs}}(\theta)
+\frac{1}{n_S+n_A}\operatorname{KL}(b\Vert m_\theta).
$$

诊断通过已有接口 `posterior_mass_options.mass_weight=1/n_A` 实现。这里 $n_S=n_A=512$，KL 系数由 0.5 降至 1/1024。源类别 GMM 模板、共同 affine/prior 参数空间、原始观测似然、相对最优拟合的阈值宽度 τ、共享更新目标、α/λ、候选后验 power 和分类决策规则保持原定义。新参考最优值及兼容集合随新目标重新拟合；它们属于此次单因素改变的结果。没有以合成真值选择参数。

二维 S 保持既有规则：angular density 预处理旁路，候选后验实际 β=1；真实高维模型沿用 angular density 和 β=0.375。后续实施必须对所有数据使用同一固定总伪计数规则，不能仅改变 S 或按目标方向分支。该诊断支持缩小软锚点的有效样本量，不保证真实图的模型错设也会改善。

## 固定输入的配对诊断

使用 attempt-019/S 已有 FINAL 的 `well_specified`、`no_shift`，各为种子 65–79，主宽度 τ=0.005。旧臂直接复用封存拟合；新臂使用相同 source/A 和 source-only 标准化，源 GMM 模板逐项完全相等。最多四个 worker。先完成并封存全部 30 个新 A 拟合、预测及决策，再读取已承诺的合成真值和 H。新臂全部 30 个主更新有效。旧臂 J/E 的 60 条对应指标与原记录在 $10^{-12}$ 内一致，重建的 H logits 与原封存数组逐值相同。

下表 ΔNLL 是 H 上相对冻结 B0 的真实后验期望 NLL 变化，单位 nats。联合通过要求期望 ΔNLL < 0、非零共享更新以及 A 的真实风险缺口不超过 0.005，与原定义一致。

| 世界及指标 | 原 $n_A$ 伪计数 | 固定 1 伪计数 |
| --- | ---: | ---: |
| well_specified 联合通过 | 11/15 | 15/15 |
| well_specified 平均 ΔNLL | −0.01095253 | −0.05369335 |
| well_specified 最差 ΔNLL | +0.00192882 | −0.00077730 |
| well_specified 最大 A 真实风险缺口 | +0.00317201 | +0.00015037 |
| well_specified 参考解释平均正类概率 | 0.565479 | 0.650830 |
| no_shift 平均 ΔNLL | +0.00007894 | +0.00007840 |
| no_shift 最大 ΔNLL | +0.00034366 | +0.00034347 |

原失败种子 65、68、73、75 的 ΔNLL 分别从 +0.00027670、+0.00192882、+0.00098558、+0.00121255，变为 −0.04347551、−0.00485716、−0.00077730、−0.01698778。无偏移两臂都满足均值 ≤0.005、每种子 ≤0.02 的保护门槛。

在正确设定世界，真实 A 后验正类均值为 0.656459，B0 均值为 0.502589。参考解释的逐种子类别比例绝对误差均值从 0.090980 降到 0.032709；同时 J 的真实期望风险改善，支持原软锚点过强的解释。源模板完全相同，因此此次改善不来自重新选择 GMM。保留集合仍由局部搜索近似，这些结果不证明完整解释集合覆盖真实后验，也没有比较所有可能的锚点强度。

无偏移世界的参考比例绝对误差反而从 0.010951 增至 0.023348；单一最优解释 E 的平均 ΔNLL 从 +0.00066081 增至 +0.00247834，而共同解释更新 J 仍约为 +0.000078。弱锚点提升正确设定下的适应能力，同时增加单一估计在无偏移时的方差；共同约束在这批输入上保留了保护作用。不能据此把参考估计精度提升概括为普遍结论。

[诊断摘要](../results/iteration-14/mass-diagnostic/summary.json) · [逐种子指标](../results/iteration-14/mass-diagnostic/endpoints.csv) · [诊断配置](../results/iteration-14/mass-diagnostic/config.json) · [身份与核验索引](../results/iteration-14/mass-diagnostic/archive-index.json)

## 新种子的正式验证与停止规则

正式 S 固定种子 **80–94**，不复用 65–79 计分。四个世界均保留：`well_specified`、`angle_shift`、`no_shift`、`semantic_reflection`。两个正对照分别要求至少 **12/15** 个联合通过，解析 oracle 仍须满足原条件；真风险松弛 0.005、非零更新阈值 $10^{-6}$、无偏移均值/每种子上限 0.005/0.02 保持不变。语义反射继续作为适用范围对照。完整性、工程检查、数值检查及原有宽度诊断照常执行。

此配对诊断只重拟合 J/E 的两个世界和主宽度，未执行完整正式 S，也未运行真实 P/C/D。正式 S 若失败则停止；通过后仍须依次满足原真实阶段的效应大小、稳定性、风险、来源和独立确认门槛。真实模型、数据划分和门槛均不因这次诊断放宽。

## 复现与封存身份

原始外部归档为 `/home/hxchen/research/GDA/TSBA/experiments/r7/attempt-019/diagnostics/fixed-mass-pseudocount`。执行脚本及依赖按 commit `b10c51f673501ab8f003d3f3e662310881a5e14e` 固定；[脚本副本](../results/iteration-14/mass-diagnostic/diagnostic.py)、[代码 SHA-256 清单](../results/iteration-14/mass-diagnostic/code-hashes.json)和三个封存清单保留执行时原始字节。源 attempt-019/S FINAL digest 为 `1aceaf5a3dc7fed8d17d83952907bac1226716f11b8f63ff6097786357396b96`；诊断 FINAL digest 为 `05017ab499bee4d293cdd5928295fabf327cbb324949f5b75971c83f8aadaec9`。归档前已核验父 FINAL 链、三个诊断 digest、其全部文件哈希和代码快照哈希。

脚本按其目录定位输入：放置于 `attempt-019/diagnostics/<新的空目录>/diagnostic.py`，以同级 `code_snapshot/` 保存上述 commit 的 `tsba`、`experiments/__init__.py`、`experiments/common`、`experiments/mutagenicity/__init__.py` 与本轮 `src/`。使用配置记录的 Python/NumPy/SciPy/Torch 环境执行 `python -B -u diagnostic.py`。脚本只以排他创建方式写文件；重复运行须另建空目录。公共目录不包含原始数组和每块求解细节，其原始相对路径和哈希由封存清单指向外部只追加归档。
