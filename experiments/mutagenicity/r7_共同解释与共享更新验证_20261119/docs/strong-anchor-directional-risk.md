# 完整更新使 Accuracy 下降，它在起点的 NLL 方向是否也错了？

在固定种子 3405 的 36 个开发方向，R7 强源起点试验已给出单位步端点。本项事后用已封存的逐图 margin 和对应方向的 NLL 汇总，分解方向和步幅。两种 Mutagenicity 设置均以源选模 WL 为起点，PROTEINS 以 Size 为起点。[强源起点结果](strong-anchor-refit-results.md) · [已评分端点](../results/iteration-12/strong-anchor-refit-001/endpoints.csv) · [冻结预测](../results/iteration-12/strong-anchor-refit-001/OBS.json)

设 cᵢ 为原强源 margin，tᵢ 为拟合头给同一张图的 margin 变化，ℓ(α) 是沿 cᵢ＋αtᵢ 行走时所有 H 图的平均二元 NLL。取 pᵢ=σ(cᵢ)、rᵢ=σ(cᵢ＋tᵢ)，有精确等式

$$
\ell(1)-\ell(0)=\ell'(0)+\frac1n\sum_i D_{\rm KL}\!\left(\operatorname{Bern}(p_i)\,\|\,\operatorname{Bern}(r_i)\right).
$$

左侧取同一方向、同一 H 的已公布 NLL 差。KL 项只需两组冻结预测，因此可以反推起点导数，无需再打开分类标签。每个设置对十二个方向等权。独立计算分别采用 Bernoulli KL 和 softplus 的一阶展开；Full、CalTeacher、B0Teacher 的 108 个块与方法组合最大绝对差为 4.45×10⁻¹⁶。

| 设置 | Full 起点导数 | 单位步 KL 余项 | 单位步 NLL 变化 | 负导数方向 | 其中单位步 NLL 反而上升 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mutagenicity 边数 | −0.02119 | 0.03411 | +0.01292 | 8/12 | 4 |
| Mutagenicity 节点数 | −0.02525 | 0.02318 | −0.00208 | 10/12 | 5 |
| PROTEINS 边数 | −0.27991 | 0.40821 | +0.12830 | 11/12 | 4 |

在这批已评价的图上，每个设置中都存在足够小的共同正步长，可令设置平均 NLL 下降。分子边数和 PROTEINS 的原单位步跨过了这项局部收益。Accuracy 只在硬判决翻转时变化，较小步长可能保留原有的全部类别。此分析没有为部署选择步长。

归因还需减去简单教师的改善。在同一起点，导数对 t 线性，从 Full 导数减去 CalTeacher 导数就得到两教师权重差方向的起点导数。三个设置依次为 −0.00516、+0.00294、−0.00638，分别在 6、4、8 个方向为负。分子节点数设置的平均导数为正，沿该固定差向量的设置平均 NLL 是凸函数，因此任何共同正步长都无法使它低于零步起点。在 PROTEINS 上，未经校准的 B0Teacher 的 Full-space 更新起点导数已达 −0.45037，说明局部 NLL 改善也可由普通源模型蒸馏产生。上述各导数均按各自实际向量的单位计。[两教师差的分类检验](centered-target-update.md)

从方法形式看，将两个模型的 log-odds 差加回基座，与 DExperts 在生成模型中采用的 logit 加减操作有先例。这里需要验证的是由一个无标签目标池得到的两种教师之差是否包含可迁移的分类纠错信息。[DExperts 原文，式 1](https://aclanthology.org/2021.acl-long.522.pdf)
