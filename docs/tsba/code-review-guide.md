# TSBA Code Review Guide（人工审查导航）

研究进展见[实验总览](experiments/README.md)：[模拟结果](experiments/synthetic/README.md)与[真实图验证计划](experiments/real-data/README.md)分开记录；工程测试不代表真实任务有效。


这份导航供逐项阅读代码使用，不是新的审查结论。初始实现来自 `536ba10`；本导航已按 2026-11-09 数值修复与研究观察器更新，残差层使用 `U1/U2` 命名。下面按“文档要求 → 核心入口 → 要核对的问题 → 已有测试”组织。算法代码位于 `tsba/`，不要与历史 TQG 实现或 `references/sega-gda/` 混读。

## 先打开这三份材料

1. [设计原文](https://github.com/chx739/TQG-GDA/blob/TSBA/docs/tsba/TSBA_Graph_Level_GDA_Design.md)（[English translation](https://github.com/chx739/TQG-GDA/blob/TSBA/docs/tsba/TSBA_Graph_Level_GDA_Design_EN.md)）：以 §3 的数学定义和行为要求为准；§1、§2 用于理解研究动机。
2. [实现映射](https://github.com/chx739/TQG-GDA/blob/TSBA/docs/tsba/implementation.md)：区分原文规定与原文留给实现的数值选择，例如固定阻尼、带宽倍数、训练预算。
3. [独立 review 记录](https://github.com/chx739/TQG-GDA/blob/TSBA/docs/tsba/review.md)：已有审查发现、修复及证据边界；它不能替代本次人工判断。

## 核心代码目录

```text
tsba/
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
├── test_pipeline.py        数据边界、冻结、持久化与真实训练 score 的端到端更新
├── test_solver_precision.py 精度相关容差与求解回归
└── test_numerical_edges.py  极端数值与异常处理回归
```

★ 四个文件构成边界适配的核心。建议先通读 pipeline，再沿下面顺序审查；第一遍不需要钻进所有自动微分细节。

## 建议阅读顺序与检查问题

| 顺序 | 文档位置 | 代码入口 | 这一步要回答什么 |
| --- | --- | --- | --- |
| 1. 总流程 | §3.16–3.17 | [pipeline.run_task](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/pipeline.py#L38) | 源初始化、固定坐标、score 训练、分类适配是否按阶段分开？最后输出什么？ |
| 2. 数据权限 | §3.1–3.2 | [TaskData.validate](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/data.py#L37)、[from_payload](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/data.py#L92) | 三个池是否明确、ID 是否互斥？目标标签会不会进入训练、验收或选模？源验证是否被并入训练？ |
| 3. 编码器与源模型 | §3.2 | [GraphEncoder](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/encoder.py#L13)、[pretrain_encoder](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/pretrain.py#L63)、[encode](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/pretrain.py#L107) | 是否训练普通源 CE 线性头？适配时参数、BN 统计、dropout 是否全部固定？源选模依赖什么？ |
| 4. 坐标与分类头 | §3.2–3.3 | [Coordinates.fit](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/classifier.py#L17)、[ResidualClassifier.from_source](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/classifier.py#L44) | 是否只用源训练表示计算均值与全局 RMS？初始概率是否等价？究竟更新哪些参数？ |
| 5. 分布信息 | §3.4–3.5 | [bandwidth](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/score.py#L10)、[denoising_loss](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/score.py#L55)、[train_score](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/score.py#L60) | sigma 如何确定？距离下溢会不会被误判成重复点？DSM 的符号与尺度是否正确？score 选模是否只看去噪损失？分类阶段是否冻结？ |
| 6. 边界请求 ★ | §3.6–3.9 | [project](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/boundary.py#L42)、[build_boundaries](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/boundary.py#L107) | 固定的是哪两个类别？投影到的根是否有效？score 如何变成有方向、有限幅度的位移？ |
| 7. 本轮目标 ★ | §3.9–3.10 | [TsbaProblem](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/problem.py#L10) | 是否使用新旧 margin 的差？分母和请求是否保持旧快照值？源 CE 与位移损失的系数是否正确？ |
| 8. 参数候选 ★ | §3.11 | [GGNSystem](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/subproblem.py#L10)、[solve](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/subproblem.py#L58) | 左端矩阵、右端符号、平均系数是否与公式相同？直接解和 CG 是否求同一系统？默认容差是否匹配计算精度，显式容差是否保留？ |
| 9. 验收与提交 ★ | §3.12–3.14 | [backtrack](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/adapt.py#L14)、[adapt_classifier](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/adapt.py#L35)、[ParameterMap](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/classifier.py#L66) | 是否真实前向复算？限制的是归一化 margin 变化还是精确边界距离？拒绝会不会污染参数？空反馈与数值异常是否区分？接受后是否刷新参考？ |
| 10. 推断与复现 | §3.15–3.17 | [Config](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/config.py#L9)、[Predictor](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/predict.py#L12)、[load_predictor](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/predict.py#L48) | 默认值是否事先确定？保存文件能否独立推断？是否意外依赖源库或 score？ |

## 最值得逐式对照的五处

| 文档表达 | 代码位置 / 表达 | 核对重点 |
| --- | --- | --- |
| x=(h−μH)/sH；W=sH W0，b=W0 μH+b0，U2=0 | classifier.py：Coordinates、from_source、U2.weight | 全局标量尺度；同一变换用于两域；残差初始化不改变概率 |
| E[‖sigma·s(x+sigma·epsilon)+epsilon‖²/d] | score.py：denoising_loss 的 square().mean() | mean 同时平均样本与坐标维度；只更新 score 参数 |
| nu=−clip(tau·sigma²·nᵀs)，zeta=−nu | boundary.py：gamma、zeta | 代码直接保存 zeta，不显式保存 nu；不要因此漏掉或重复加负号 |
| v=(m_new−m_old)/kappa；D=∂m/∂psi / kappa | problem.py：movement、normalized_margin | normalized_margin 是 m/kappa；它的参数 Jacobian 就是 D。旧 margin 是固定项 |
| (FS+lambda_B DᵀD/M+mu I) delta=−gS+lambda_B Dᵀzeta/M | subproblem.py：coefficient、rhs、dense、matvec | 源批次平均与边界点平均分别处理；CG 路径无需显式存储 D |

检查位移方向时可用一维例子：m(x)=a(x−b)，a>0。旧边界点 margin 增大意味着边界 b 向左移动，因此 D delta≈−nu。不要把 margin 增量方向直接等同于边界移动方向。

## 容易混淆的保证与证据边界

- **找边界与移动边界是两步。** 投影时固定模型，移动辅助坐标 ξ 来寻找 m≈0；适配时才改变分类器参数。ξ 不是生成的真实图。
- **请求不是伪标签。** top-two 只定义本轮类别面，目标密度提供法向移动方向，不把获胜类别作为目标真标签。
- **movement 不是精确测得的边界位移。** 代码计算 v=(m_new(ξ)−m_old(ξ))/κ_old。局部情况下边界法向位移约为 −v；验收没有重新定位新边界，也没有测量精确位移误差。设计原文 §1.2 的“实际位移”描述研究意图，实现保证以 §3.12 的代理检查为准。
- **源 CE 是软目标。** 总目标下降不要求源 CE 单独下降；逐类 CE 只记录，没有 SEGA 式逐类硬风险预算。
- **score 验证不等于边界方向正确。** 同一无标签池上的独立噪声验证只比较去噪表现，尚不能证明边界处 score 的法向符号可靠。
- **接受更新不等于目标准确率提高。** 验收只检查本轮固定参考的代理目标和变化预算；不同轮次的目标会变，也没有跨轮全局单调保证。

## 数值实现需要额外核对的地方

`cg_tolerance` 同时用于直接求解与 CG 的真实相对残差验收。未指定或设为 null 时，float64 默认 1e-7、float32 默认 1e-5；显式给出的正容差保持不变，不在失败时偷偷放宽。配置与运行记录保存解析后的值。这是计算精度下的工程选择，不是目标标签选出的超参数，也不保证任意病态系统都能通过。

投影不仅检查梯度范数，还要检查平方与分母是否有限。计算距离为零也不必然说明两个坐标相同：不同点的距离下溢必须报数值错误。出现这些错误时不得伪装成正常的无边界或重复点退化。

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
| [encoder.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/encoder.py) | [encoder.py](https://github.com/chx739/TQG-GDA/blob/TSBA/references/sega-gda/sega/encoder.py) | 消息传递、BN、逐层 sum pooling 拼接和投影结构；TSBA 的 dtype 处理 |
| [pipeline.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/pipeline.py) | [pipeline.py](https://github.com/chx739/TQG-GDA/blob/TSBA/references/sega-gda/sega/pipeline.py) | 分阶段编排及冻结表示的组织方式 |
| [problem.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/problem.py)、[subproblem.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/subproblem.py)、[adapt.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tsba/adapt.py) | [problem.py](https://github.com/chx739/TQG-GDA/blob/TSBA/references/sega-gda/sega/problem.py)、[subproblem.py](https://github.com/chx739/TQG-GDA/blob/TSBA/references/sega-gda/sega/subproblem.py)、[adapt.py](https://github.com/chx739/TQG-GDA/blob/TSBA/references/sega-gda/sega/adapt.py) | 目标定义、候选求解、非线性重评估、接受或回溯的职责分工 |

这里继承的是结构与执行范式。TSBA 文档明确改用线性头＋残差分类器、目标 score 与边界位移，因此不能要求它保留 SEGA 的参照库分类、可信目标伪标签或逐类硬风险预算。SEGA 的真实数据入口会读目标标签，本次人工 review 只需阅读它，不必运行。

## 每块代码对应的现有测试

| Review 对象 | 测试入口 | 现有证据 |
| --- | --- | --- |
| 概率等价初始化 | [test_math.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tests/tsba/test_math.py#L41) | 坐标变换与中心化 logits 不改变初始概率 |
| 投影、方向、带宽 | [test_math.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tests/tsba/test_math.py#L51) | 根、半径、无效类面、重复点、溢出与法向请求 |
| DSM、GGN、直接解/CG | [test_math.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tests/tsba/test_math.py#L96) | 解析 score 例子、有限差分、矩阵/算子一致性、非零残差多分类 |
| 接受、拒绝与异常 | [test_adapt.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tests/tsba/test_adapt.py) | 方向、目标下降、有限变化、零反馈跳过、异常和拒绝不改参数 |
| 数据边界、冻结、保存与实际更新 | [test_pipeline.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tests/tsba/test_pipeline.py) | 目标标签拒绝、角色互斥、BN 冻结、单节点批次；真实训练 score 后产生有效请求、接受更新并保存加载，覆盖 direct/CG |
| 精度与数值异常 | [test_solver_precision.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tests/tsba/test_solver_precision.py)、[test_numerical_edges.py](https://github.com/chx739/TQG-GDA/blob/TSBA/tests/tsba/test_numerical_edges.py) | 默认精度策略、显式容差、投影平方溢出、不同点距离下溢与快照保持 |

实现与数值修复阶段的历史测试、运行环境和审查记录见 [验证摘要](https://github.com/chx739/TQG-GDA/blob/TSBA/docs/tsba/validation.json)。原持久化测试允许无边界而不更新；新增的完整链路测试必须有真实参数更新，二者职责不同。这些检验实现与数值行为，尚不证明低密度方向改善真实目标分类，也未验证 GPU 路径。

## 记录疑问的方式

每个疑问记四项即可：`文件 / 函数 → 对应文档小节 → 你期望的行为 → 代码中不理解或不一致的地方`。例如：`boundary.build_boundaries → §3.8 → 密度升高时边界应反向移动 → 为什么 zeta 是正号？`

先区分“实现是否符合文档”和“文档假设是否合理”：前者定位代码，后者讨论方法。这样后续可以直接围绕具体函数和公式继续，不必重述整套流程。

## 从代码 review 到实验验证

下一步见 [TSBA 验证实验设计](https://github.com/chx739/TQG-GDA/blob/TSBA/docs/tsba/experiments/validation-plan.md)。它区分公式执行、score 的边界可靠性、实际局部位移与真实任务收益，并列出待实现的研究观察器与基线；该设计稿不是已完成实验报告。

## 合成研究入口

[E1/E2方案](experiments/validation-plan.md)与[运行器](../../workflows/tsba/validation/run.py)单独审查：oracle是否与坐标/带宽一致、目标评价是否发生在封存之后、四臂批次是否配对、失败和零更新是否保留。`test_validation.py`检查解析score、独立几何观察器、零信息符号指标和被动观察器。新增锚点来源索引和回调不改变生产求解、验收或零反馈规则。

补充控制入口：[controls.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/controls.py)、[run_controls.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/run_controls.py)。核对纯源GGN是否移除DᵀD/位移预算，零请求几何控制是否保留它们，是否确实绕过研究分支的零请求跳过而没有修改生产规则，以及是否复用原始逐轮ID。对应 `test_controls.py`。

全源诊断入口：[run_full_source.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/run_full_source.py)。检查配置是否只改变 `source_adapt_batch`、每轮是否完整覆盖512个源ID、目标ID是否与旧档案一致，以及候选和每次回溯是否都使用全源CE。对应 `test_full_source.py`；不改变生产TSBA目标与接受规则。

全源实验入口加入后，完整工程测试为44项通过，见[测试日志](experiments/e2-full-source-20261110/engineering-checks.txt)。[全源诊断报告](experiments/e2-full-source-20261110/report.md)另列30个checkpoint、900轮状态链和1380次候选CE的重载审计；研究结果与工程通过分开报告。

checkpoint回放入口：[run_replay.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/run_replay.py)、[replay.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/replay.py)。检查解析后验与初始头是否一致、NLL四组贡献是否加和、边界探针是否固定并保留失败，以及观察前后参数是否逐位相同。对应 `test_replay.py`。

回放入口加入后，完整工程测试为48项通过，见[测试日志](experiments/e2-replay-20261110/engineering-checks.txt)；[回放报告](experiments/e2-replay-20261110/report.md)保存930次状态核验和三类研究诊断，未训练新模型。

解析源后验诊断：[source_posterior.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/source_posterior.py)。检查源期望CE是否同时进入梯度与真实验收、GGN公式是否保留、三个分支的跳过/回溯规则是否与旧入口一致，以及是否没有强制控制组不动。对应 `test_source_posterior.py`；当前51项测试通过。

[解析源后验对照报告](experiments/e2-source-posterior-20261110/report.md)记录候选和验收的独立复算、20条数值平稳控制轨迹及总体score的实际收益；[51项测试日志](experiments/e2-source-posterior-20261110/engineering-checks.txt)可直接核对。


静态score诊断：[static_score.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/static_score.py)、[运行器](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/run_static_score.py)、[重载分析](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/summarize_static_score.py)。检查嵌套样本池、原512基准带宽、固定查询与400次更新是否保持；主最终状态与辅助选中快照是否分开；是否同时报告法向/请求误差与正确边界方向，避免只看全向量DSM或同带宽oracle符号一致率。对应`test_static_score.py`，当前完整56项工程测试通过。[研究报告](experiments/e2-static-score-20261110/report.md)记录50组拟合、400次checkpoint查询集重载和3100个oracle探针；没有启动闭环适配。

P0预算诊断：[run_p0.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/run_p0.py)、[p0_oracle.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/p0_oracle.py)、[summarize_p0.py](https://github.com/chx739/TQG-GDA/blob/TSBA/workflows/tsba/validation/summarize_p0.py)。检查400步Adam恢复与随机前缀是否精确，主固定末尾模型是否区别于DSM选中模型；oracle阈值是否采用原坐标缩放，有限/总体源后验是否区分，回溯停止是否如实报告；不要把标量收益当作完整TSBA上界。对应`test_p0.py`，完整59项测试通过。[P0报告](experiments/p0-20261110/report.md)记录6400步静态检查5/5通过，但源约束下oracle收益仍小，未启动学习score闭环。

真实图R1详细设计：[Mutagenicity流程](experiments/real-data/mutagenicity-stage1-protocol.md)、[机器清单](experiments/real-data/stage1-plan.json)。诊断包含类别相关密度结构与源边界方向；480个小头拟合的执行驱动仍待实现/验收，当前未运行真实图训练。
