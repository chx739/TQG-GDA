# TSBA 实现与设计映射

依据同目录 `design-zh.md` 用户原文，SHA256 `57ef49dae959d2331a91bdb26f536c54adde07165078df5186e0ea1ce36b8855`。原文“尚未实现或验证”是提供时的状态；本分支实现算法，但没有真实任务有效性结论。

完整英文译文见 [英文设计](design-en.md)，保留原文历史状态、公式和证据边界。

## SEGA 范式与 TSBA 定义

沿用 SEGA 的配置、数据、源预训练、固定坐标、目标辅助信息、problem、subproblem、adapt、predict、pipeline 职责拆分。每轮固定参考，求解候选，真实前向重评估，通过验收后才提交。候选使用函数式参数调用，拒绝或异常不会污染正在使用的模型。

GNN 结构改写自参考 `encoder.py`：GIN/GINE、MLP/外层 BatchNorm、逐层 sum pooling 拼接、两层投影。TSBA 版本让输入跟随参数 dtype，支持 GINE float64；参考快照保持原字节。

TSBA 定义优先于 SEGA 的具体算法：不继承参照库分类器、可信目标伪标签任务、共享度量或逐类硬风险预算。源 CE 是软监督项，不能宣称具有 SEGA 的硬预算保证。

## 逐节映射

| 原文 | 实现 | 合同 |
| --- | --- | --- |
| 3.1 | `data.py` | 明确源训练/源验证/无标签目标池；ID 互斥；目标禁止类别字段 |
| 3.2 | `encoder.py`, `pretrain.py`, `classifier.Coordinates` | 源 CE 线性头；冻结参数、BN、dropout；均值和全局 RMS 只来自源训练表示 |
| 3.3 | `classifier.ResidualClassifier` | 中心化 logits；SiLU；U2=0、W=sW0、b=W0μ+b0，初始化概率等价 |
| 3.4–3.5 | `score.py` | 正距离近邻中位数带宽；单噪声 DSM；两层 SiLU MLP；训练后冻结 |
| 3.6–3.8 | `boundary.py` | 固定类对；球内有限次投影/回溯；有效类面；法向 score 和裁剪位移 |
| 3.9–3.10 | `problem.py` | 旧梯度范数归一化；真实 margin 差；源 CE + λB movement loss |
| 3.11 | `subproblem.py` | 源 GGN + λB DᵀD/M + μI；直接解或 JVP/VJP CG；不再应用 Adam |
| 3.12–3.14 | `adapt.py` | 固定参考 Armijo 与变化预算；每次更新后刷新；空/零反馈跳过；非有限停止 |
| 3.15–3.17 | `config.py`, `pipeline.py`, `predict.py` | 先验固定配置；最后接受状态；独立推断模型 |

不反传穿过 score、投影、法向、响应构造或线性求解器训练编码器。输入梯度、参数 Jacobian 和分布 score 职责独立。

## 文档留给实现的数值选择

以下是工程默认值，不是通过目标标签选择的最优值。

- GNN 默认 3 层、hidden=64、embedding=16，与 SEGA 配置相同。源预训练 float32；坐标、分类适配和 score 默认 float64。
- 适配器默认宽度 min(16,d)。score 两层 hidden=64 SiLU MLP。源和 score 默认各训练 100 epochs。
- 源 checkpoint 按独立源验证 CE 最低选取；没有源验证池时使用最后一个预定 epoch，不另取或增加标签。普通源 CE，不隐式类平衡。
- 为满足继承的 BatchNorm 结构，节点数不足 2 的微批次按原顺序与相邻批次合并；不丢图或重复采样，适用于单节点图及 batch size=1。
- sigma=kappa_sigma*median(positive_nearest_distances)/sqrt(d)，默认 kappa=1。偶数样本中位数取中间两数平均；重复坐标保留权重但不作为正距离邻居。不足两个不同目标坐标时跳过密度适配。
- 距离、梯度范数、投影分母与归一化残差的溢出属于数值错误，不能伪装成重复点退化或正常零反馈。不同坐标的距离若下溢为零，也必须报 NumericalError，不能当作重复点掩盖。
- score 验证噪声与训练使用不同随机流，在同一无标签池上验证去噪；固定独立验证噪声以比较快照。这不是独立图样本的总体泛化证据，更不是分类模型选择。
- R_proj=sigma*sqrt(d)，Delta_max=Delta_exec=0.1*sigma，epsilon_B=1e-5*sigma；tau=0.1、lambda_B=1。
- 投影最多 20 步、每步最多 10 次回溯，归一化残差严格下降。二分类固定 (0,1)；多分类稳定排序，并列按类序号。投影后检查两类仍为最高两类，允许文档规定的数值容差。
- mu_t=0.01 固定；参数数 <=2048 时直接解，否则 CG。CG 最多 200 步。cg_tolerance 未指定或为 null 时按 adaptation_dtype 解析：float64 为 1e-7，float32 为 1e-5；显式正容差不覆盖。直接求解与 CG 都以真实相对残差 <=cg_tolerance 验收，配置记录解析后的值。该精度策略预先固定，不按候选或目标标签动态放宽；有限但未收敛或非下降方向拒绝本轮。
- Armijo 系数 1e-4，alpha 从 1 起减半，最多 15 次。固定本轮源/目标批次和边界参考。失败后保持参数，进入下一轮。
- 总轮数固定 30，包含拒绝和空反馈，不按接受次数无限重试。非有限 score、损失或求解结果记录 numerical_error 并停止本次执行，保留最后接受模型；CLI 返回非零状态。
- 分类适配不使用源验证集选择最终模型。源逐类 CE 仅记录，不新增硬约束。目标标签不参与任何阶段。

## 初始化与数据边界

分类器中的两个残差层使用 `U1`、`U2` 命名：`U1.weight` 对应公式矩阵 U1，`U1.bias` 对应 a，`U2.weight` 对应无偏置矩阵 U2。新保存的模型使用这些键；`load_predictor` 仍兼容旧 `first/last` 键名，若同时出现新旧同义键则拒绝加载，避免歧义。

SEGA 距离参照分类器不能直接提供 W0,b0。本实现采用文档允许的普通源监督初始化，训练 SEGA 同类编码器与线性头。`source.pt` 可跨目标池复用，但必须匹配源数据、角色、架构、预训练配置。

不自动把旧 TQG query 加入源训练，也不自动继承 SEGA ref/fit/cal 划分。真实任务需事先提供合法角色。程序不读取整个目标域标签或自动计算目标准确率。

运行身份包含完整配置、图 ID/内容/源标签/角色及协议，复用 checkpoint 时记录文件 SHA256。输出目录必须新建；score、边界点和求解状态不进入最终 predictor。

## 保留的研究局限

设计原文保持用户提供时的字节内容。其 §1.2“实际位移”和 §2.2“执行误差”需结合 §3.12 理解：实现以归一化 margin 变化 v 为局部位移代理，没有重新定位候选模型的边界，也没有直接测量真实几何位移误差。

验收控制代理目标及旧边界点上的归一化 margin 变化，不保证真实边界位移相等、目标风险下降、类别不坍塌或跨轮统一目标下降。没有添加核密度替代 score、重新定位边界的验收或不确定性筛选；此前分析中的这些建议不是用户文档的主算法。

验证只涉及合成图和数值机制，不读取真实 final 标签，不报告真实图分类提升。

## 代码内的中文阅读说明

核心模块已在计算位置添加中文注释，按设计原文 §3.1–3.17 标明变量、公式与梯度/状态边界。公式旁的符号沿用原文；数值默认值仍以 Config 和本文件为准。2026-11-08 的注释提交仅新增注释，经可执行 token 与 AST 等价检查及当时 23 项测试确认。早期数值修复与当时测试证据见 [review.md](../review/initial-review.md) 和 [validation.json](../review/initial-validation.json)。人工阅读入口见 [code-review-guide.md](../review/code-review-guide.md)。
