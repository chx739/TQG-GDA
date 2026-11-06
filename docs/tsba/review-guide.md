# TSBA 人工 review 导航

这份导航供逐项阅读代码使用，不是新的审查结论。代码基准为 `536ba10`；下面按“文档要求 → 核心入口 → 要核对的问题 → 已有测试”组织。算法代码位于 `tqg_gda/tsba/`，不要与历史 TQG 实现或 `references/sega-gda/` 混读。

## 先打开这三份材料

1. [设计原文](TSBA_Graph_Level_GDA_Design.md)：以 §3 的数学定义和行为要求为准；§1、§2 用于理解研究动机。
2. [实现映射](implementation.md)：区分原文规定与原文留给实现的数值选择，例如固定阻尼、带宽倍数、训练预算。
3. [独立 review 记录](review.md)：已有审查发现、修复及证据边界；它不能替代本次人工判断。

## 核心代码目录

```text
tqg_gda/tsba/
├── pipeline.py       四阶段总流程：先从这里建立全局认识
├── data.py           源训练、源验证、无标签目标的数据契约
├── encoder.py        SEGA 同类 GIN/GINE 与冻结逻辑
├── pretrain.py       源线性分类器预训练、源验证选模、表示缓存
├── classifier.py     固定坐标、残差分类头、函数式参数候选
├── score.py          目标带宽、score 网络、单尺度去噪训练
├── boundary.py       边界定位、有效性检查、法向位移请求 ★
├── problem.py        源 CE 与归一化 margin 变化目标 ★
├── subproblem.py     GGN、规范化响应、直接求解 / CG ★
├── adapt.py          每轮快照、回溯验收、提交或跳过 ★
├── predict.py        最终模型保存、加载、独立推断
├── config.py         所有预先固定的预算与数值参数
└── utils.py          有限值检查、随机种子、内容身份

workflows/tsba/run_task.py   命令行入口、合成小图输入
tests/tsba/
├── test_math.py            公式、几何与求解一致性
├── test_adapt.py           接受、拒绝、零反馈、异常回滚
└── test_pipeline.py        数据边界、冻结、持久化与端到端
```

★ 四个文件构成边界适配的核心。建议先通读 pipeline，再沿下面顺序审查；第一遍不需要钻进所有自动微分细节。

## 建议阅读顺序与检查问题

| 顺序 | 文档位置 | 代码入口 | 这一步要回答什么 |
| --- | --- | --- | --- |
| 1. 总流程 | §3.16–3.17 | [pipeline.run_task](../../tqg_gda/tsba/pipeline.py#L35) | 源初始化、固定坐标、score 训练、分类适配是否按阶段分开？最后输出什么？ |
| 2. 数据权限 | §3.1–3.2 | [TaskData.validate](../../tqg_gda/tsba/data.py#L33)、[from_payload](../../tqg_gda/tsba/data.py#L86) | 三个池是否明确、ID 是否互斥？目标标签会不会进入训练、验收或选模？源验证是否被并入训练？ |
| 3. 编码器与源模型 | §3.2 | [GraphEncoder](../../tqg_gda/tsba/encoder.py#L11)、[pretrain_encoder](../../tqg_gda/tsba/pretrain.py#L59)、[encode](../../tqg_gda/tsba/pretrain.py#L100) | 是否训练普通源 CE 线性头？适配时参数、BN 统计、dropout 是否全部固定？源选模依赖什么？ |
| 4. 坐标与分类头 | §3.2–3.3 | [Coordinates.fit](../../tqg_gda/tsba/classifier.py#L15)、[ResidualClassifier.from_source](../../tqg_gda/tsba/classifier.py#L36) | 是否只用源训练表示计算均值与全局 RMS？初始概率是否等价？究竟更新哪些参数？ |
| 5. 分布信息 | §3.4–3.5 | [bandwidth](../../tqg_gda/tsba/score.py#L8)、[denoising_loss](../../tqg_gda/tsba/score.py#L42)、[train_score](../../tqg_gda/tsba/score.py#L46) | sigma 如何确定？DSM 的符号与尺度是否正确？score 选模是否只看去噪损失？分类阶段是否冻结？ |
| 6. 边界请求 ★ | §3.6–3.9 | [project](../../tqg_gda/tsba/boundary.py#L33)、[build_boundaries](../../tqg_gda/tsba/boundary.py#L88) | 固定的是哪两个类别？投影到的根是否有效？score 如何变成有方向、有限幅度的位移？ |
| 7. 本轮目标 ★ | §3.9–3.10 | [TsbaProblem](../../tqg_gda/tsba/problem.py#L8) | 是否使用新旧 margin 的差？分母和请求是否保持旧快照值？源 CE 与位移损失的系数是否正确？ |
| 8. 参数候选 ★ | §3.11 | [GGNSystem](../../tqg_gda/tsba/subproblem.py#L7)、[solve](../../tqg_gda/tsba/subproblem.py#L43) | 左端矩阵、右端符号、平均系数是否与公式相同？直接解和 CG 是否求同一系统？ |
| 9. 验收与提交 ★ | §3.12–3.14 | [backtrack](../../tqg_gda/tsba/adapt.py#L9)、[adapt_classifier](../../tqg_gda/tsba/adapt.py#L25)、[ParameterMap](../../tqg_gda/tsba/classifier.py#L53) | 是否真实前向复算？拒绝会不会污染参数？空反馈与数值异常是否区分？接受后是否刷新参考？ |
| 10. 推断与复现 | §3.15–3.17 | [Config](../../tqg_gda/tsba/config.py#L7)、[Predictor](../../tqg_gda/tsba/predict.py#L11)、[load_predictor](../../tqg_gda/tsba/predict.py#L43) | 默认值是否事先确定？保存文件能否独立推断？是否意外依赖源库或 score？ |

## 最值得逐式对照的五处

| 文档表达 | 代码位置 / 表达 | 核对重点 |
| --- | --- | --- |
| x=(h−μH)/sH；W=sH W0，b=W0 μH+b0，U2=0 | classifier.py：Coordinates、from_source、last.weight | 全局标量尺度；同一变换用于两域；残差初始化不改变概率 |
| E[‖sigma·s(x+sigma·epsilon)+epsilon‖²/d] | score.py：denoising_loss 的 square().mean() | mean 同时平均样本与坐标维度；只更新 score 参数 |
| nu=−clip(tau·sigma²·nᵀs)，zeta=−nu | boundary.py：gamma、zeta | 代码直接保存 zeta，不显式保存 nu；不要因此漏掉或重复加负号 |
| v=(m_new−m_old)/kappa；D=∂m/∂psi / kappa | problem.py：movement、normalized_margin | normalized_margin 是 m/kappa；它的参数 Jacobian 就是 D。旧 margin 是固定项 |
| (FS+lambda_B DᵀD/M+mu I) delta=−gS+lambda_B Dᵀzeta/M | subproblem.py：coefficient、rhs、dense、matvec | 源批次平均与边界点平均分别处理；CG 路径无需显式存储 D |

检查位移方向时可用一维例子：m(x)=a(x−b)，a>0。旧边界点 margin 增大意味着边界 b 向左移动，因此 D delta≈−nu。不要把 margin 增量方向直接等同于边界移动方向。

## 梯度和状态检查清单

- [ ] 编码器及归一化运行统计固定；表示与坐标统计不被适配梯度更新。
- [ ] score 只在独立训练阶段更新；边界请求构造不训练 score 网络。
- [ ] 本轮类对、投影点、法向、梯度范数、旧 margin、zeta 固定，不随回溯候选重算。
- [ ] 计算候选分类器在固定边界点上的输出时保留参数响应；没有把所有输出都误 detach。
- [ ] GGN/响应只作为本轮数值算子；不通过求解器取元梯度，不再用 Adam 施加 delta。
- [ ] Armijo 使用本轮固定目标的真实前向结果，且同时检查归一化变化预算。
- [ ] 只有接受后才调用 commit；拒绝、零反馈和异常不会产生隐式源训练更新。
- [ ] 每次接受后进入下一轮重新定位边界；有限回溯失败是跳过本轮，非有限异常是停止执行。

关键张量可对照：source 为 [BS,d]，target 为 [BT,d]，有效边界 points 为 [M,d]，pairs 为 [M,2]，normal_norm / old_margin / zeta 为 [M]，参数向量为 [P]，D 为 [M,P]，FS 为 [P,P]。

## SEGA 需要对照哪些文件

| TSBA | SEGA 参考 | 主要看什么 |
| --- | --- | --- |
| [encoder.py](../../tqg_gda/tsba/encoder.py) | [encoder.py](../../references/sega-gda/sega/encoder.py) | 消息传递、BN、逐层 sum pooling 拼接和投影结构；TSBA 的 dtype 处理 |
| [pipeline.py](../../tqg_gda/tsba/pipeline.py) | [pipeline.py](../../references/sega-gda/sega/pipeline.py) | 分阶段编排及冻结表示的组织方式 |
| [problem.py](../../tqg_gda/tsba/problem.py)、[subproblem.py](../../tqg_gda/tsba/subproblem.py)、[adapt.py](../../tqg_gda/tsba/adapt.py) | [problem.py](../../references/sega-gda/sega/problem.py)、[subproblem.py](../../references/sega-gda/sega/subproblem.py)、[adapt.py](../../references/sega-gda/sega/adapt.py) | 目标定义、候选求解、非线性重评估、接受或回溯的职责分工 |

这里继承的是结构与执行范式。TSBA 文档明确改用线性头＋残差分类器、目标 score 与边界位移，因此不能要求它保留 SEGA 的参照库分类、可信目标伪标签或逐类硬风险预算。SEGA 的真实数据入口会读目标标签，本次人工 review 只需阅读它，不必运行。

## 每块代码对应的现有测试

| Review 对象 | 测试入口 | 现有证据 |
| --- | --- | --- |
| 概率等价初始化 | [test_math.py](../../tests/tsba/test_math.py#L41) | 坐标变换与中心化 logits 不改变初始概率 |
| 投影、方向、带宽 | [test_math.py](../../tests/tsba/test_math.py#L51) | 根、半径、无效类面、重复点、溢出与法向请求 |
| DSM、GGN、直接解/CG | [test_math.py](../../tests/tsba/test_math.py#L96) | 解析 score 例子、有限差分、矩阵/算子一致性、非零残差多分类 |
| 接受、拒绝与异常 | [test_adapt.py](../../tests/tsba/test_adapt.py) | 方向、目标下降、有限变化、零反馈跳过、异常和拒绝不改参数 |
| 数据边界、冻结、保存 | [test_pipeline.py](../../tests/tsba/test_pipeline.py) | 目标标签拒绝、角色互斥、BN 冻结、单节点批次、身份与独立加载 |

已有 23 项测试和独立 review 的详细记录见 [验证摘要](validation.json)。这些检验实现与数值行为，尚不证明低密度方向改善真实目标分类，也未验证 GPU 路径。

## 记录疑问的方式

每个疑问记四项即可：`文件 / 函数 → 对应文档小节 → 你期望的行为 → 代码中不理解或不一致的地方`。例如：`boundary.build_boundaries → §3.8 → 密度升高时边界应反向移动 → 为什么 zeta 是正号？`

先区分“实现是否符合文档”和“文档假设是否合理”：前者定位代码，后者讨论方法。这样后续可以直接围绕具体函数和公式继续，不必重述整套流程。
