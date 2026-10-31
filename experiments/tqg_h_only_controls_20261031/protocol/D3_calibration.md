# D3：联合校准是否真的优于简单核回归

[返回实验列表](README.md#experiment-list) · [统一协议](method.md) · 后续：[D4](D4_signature.md)、[D7](D7_query_statistics.md)

**优先级P1；待实现、待执行；新增训练0。**

白话：先排除“复杂方法只是用了更弱的正则”这个解释，再判断最大方向误差项有没有价值。

## 1. 关键等价关系

当前mean-only解：
W=H_STᵀ(K_SS⊗I_q+lambda_W I)⁻¹。

固定Ω=I/r时D=2I，条件最优解等价于mean-only使用lambda_W/2。二者同设0.01不代表正则匹配；一般joint对不同方向使用lambda_W/δ_a，单一KRR正则不能完全匹配其各向异性。

## 2. 冻结设计与对照

使用D1的360块、同θ/V/签名/核/source residual，不重新抽图。

定义标尺α∈{0.001,0.005,0.01,0.05,0.1}，每个标尺三臂：

| 臂 | 设置 | 用途 |
|---|---|---|
| MEAN | mean-only，lambda_W=α | 简单读取 |
| ISO | 固定Ω=I/r，lambda_W=2α | 与MEAN代数匹配的工程／机制参照 |
| JOINT | 可学习Ω，lambda_W=2α | 相对ISO只改变方向权重的分配 |

主比较α=0.005：JOINT lambda0.01对ISO lambda0.01／MEAN lambda0.005，即当前主版本的公平参照。历史KRR lambda0.01作为α=0.01的MEAN行保留。

共360×15=5400次小矩阵求解；同块source核分解、V、源JVP可缓存。15是配置求解数，不是15次独立训练。不得给joint挑最优α，而把KRR固定在较差α。

## 3. 记录什么

- 主要：各臂c_A、相对误差；主α下JOINT−ISO的配对差。
- 次要：D2原始反馈及等范数反馈的一步B池增益。等范数参照选同块MEAN α0.005；零范数记缺失。
- 代理目标：trace(E)/r、lambda_max(E)、正则项、gap、迭代／回溯次数及有限性。
- Ω结构：特征值、||Ω−I/r||_F、D的δ最小／最大、有效正则lambda_W/δ。
- 作用差别：||W_joint−W_iso||_F/max(||W_iso||_F,1e-12)、两者反馈余弦和范数比。
- 时间：核／JVP、分解、各求解器独立计时，不能只报一次较快调用。

## 4. 假设与判据

预期方向专门化应产生超过单纯幅度变化的收益。主α下，至少2/3任务JOINT对ISO平均余弦优势≥0.03、各至少3/5种子差为正，并且等范数一步B池CE改善平均为正，才支持保留最大方向项作为改进贡献。覆盖率按统一协议≥80%。

代理Q下降本来就是求解目标，不能单独当研究成功；ISO≈MEAN是代数验收，也不是性能发现。

- 若Ω接近均匀、反馈与ISO近乎共线，且收益不达门槛：采用简单MEAN作为下一候选默认，联合版本保留为历史对照。
- 若只有原始幅度有差、等范数差消失：优先解释为正则／幅度作用，不主张方向校准增益。
- 若joint只在一个事后挑出的α胜出：报告敏感性，不改主比较。
- 达到内部上限的有限近似解保留并标注；非有限回退逐块列出，不混成正常最优解。

## 5. 输出与实现

calibration_grid_blocks.csv、calibration_seed_summary.csv、omega_spectrum.jsonl、regularization_curves.png/pdf、decision.md。

现代码支持joint与mean_only；ISO条件解需新增显式接口及D0等价验收。不得伪造“冻结Ω”配置名后实际仍运行自适应joint。
