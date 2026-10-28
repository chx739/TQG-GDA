# TQG v2 文档与代码审查，2026-10-28

审查基准为主仓库提交 `fe8698eebf558a6f4a571382e63d2bf73e628593`，对照用户提供的 [v2设计稿](../../TQG_Graph_Level_GDA_Design_v2.md)。附件与仓库设计稿逐字节相同，SHA256为 `907382b0cc56103e38da8ae0182ec5a0fb93c6490ef86f73d570cb7f433fc915`。

**结论：本次审查未发现当前主版本核心计算链违反设计稿的实现错误。** 方向、响应、核、联合校准、残差读取、停止梯度和单步SGD均有对应实现与检查。这个结论限于已审查代码及以下验证，不能证明所有输入和设备都没有错误，也不能证明反馈具有迁移效果。本次没有修改训练算法，没有重新训练真实数据模型，没有读取目标标签。

## 文档逐项对应

| 文档 | 代码 | 审查结果 |
|---|---|---|
| §3.1–3.2 完整图分类、类别坐标 | `models.py`、`feedback_v2.contrast_basis` | 整图readout、两个二分类logits、类别中心化和固定Helmert基；q=C−1；全部可训练参数包含分类头。 |
| §3.3 源监督角色 | `data.source_partition`、`training.run_warmup` | fit/query固定且不重叠，源预训练和主CE使用fit；另有源验证。query是参与训练的监督，不能当独立验证集。 |
| §3.4 同一快照和确定状态 | `ParameterSnapshot`、`TQGTrainer.step` | 参数与buffer复制，适配使用eval固定dropout/BN运行状态；可学习参数继续求梯度，每步重新构建反馈。 |
| §3.5 全目标梯度空间 | `ParameterSnapshot.jacobian`、`target_space` | 堆叠目标对比输出的全参数导数；薄SVD保留全部数值非零方向，没有固定K或主动秩截断。 |
| §3.5 响应 | `ParameterSnapshot.response`、`build_gradient_feedback` | 源响应用JVP，目标响应复用J乘V；使用同一V，不使用odd/even有限差分。 |
| §3.6 梯度查询 | `query_matrices` | B=A/bT，H_ST含一次1/bT，H_TT含两次；图优先、类内坐标排列与显式块核相符。 |
| §3.7 签名核 | `signature_kernels` | h和A的平方距离进入同一个RBF；尺度是源query两两平方距离的中位数，下限epsilon²；SS/ST/TT共用尺度。 |
| §3.8 迁移条件 | 方法假设，不能仅靠代码验证 | 同签名条件标签关系稳定、任务信息充分、核函数族近似及监督支持均未由工程测试证明。 |
| §3.9 源读取 | `build_gradient_feedback` | W先求解，源标签值随后进入r_S=(Y−p)Q；d=W r_S，没有额外除以源query图数。 |
| §3.10 校准目标 | `calibration_v2.solve_calibration` | E=H_TT−WH_ST−H_STᵀWᵀ+W K_SS Wᵀ；目标是trace(E)/r+lambda_max(E)+lambda_W||W||²/r。类别块通过tensor实现，未错误省略类别坐标。 |
| §3.11 求解 | `conditional`、`project_density`及回溯循环 | 固定Omega解Sylvester方程；D=I+rOmega；gap使用配对Omega与条件最优W；新Omega的回溯评价重新求其W。 |
| §3.12 基准校正 | `build_gradient_feedback`、`GradientFeedback.gradient` | 直接计算−VW r_S，保留Y−p；没有源目标核支持时W为零。没有用目标伪标签替代源监督。 |
| §3.13 损失与执行 | `parameter_feedback_loss`、`TQGTrainer.step` | 0.5||Vᵀ(theta−snapshot)−d||²，不除以rank；在快照处梯度−Vd，合并源CE后执行一次无动量、无weight decay的SGD。源预训练使用Adam不违反适配执行规则。 |
| §3.14 停止梯度 | 快照、查询模块及内部求解器 | Jacobian/V/A/核/W/r_S/d均停止梯度；构建查询时的自动微分不会对查询求解建立训练用高阶梯度。 |
| §3.15–3.16 循环和推断 | `run_adaptation`、`GraphClassifier` | 更新后重新查询；推断只保留GNN。目标训练视图无y，训练接口拒绝带目标标签的batch；源验证用于选模。 |

## 本次额外验证

1. **完整回归测试116项通过**，见 [unit_tests.log](unit_tests.log)。覆盖完整空间重构真实CE梯度、源JVP对显式Jacobian、多分类块矩阵、坐标旋转不变性、标签隔离、冻结dropout/BN、停止梯度、反馈梯度、SGD更新和恢复等。测试中的标签来自合成样本，不是读取真实目标数据标签。
2. **独立求解器检查5例通过**，见 [脚本](independent_calibration.py)和[结果](independent_calibration.json)。使用合成PSD核，显式展开多分类块矩阵，直接以W和最大误差上界t为变量，用SciPy SLSQP求原始约束问题：最小化trace(E)/r+t+lambda_W||W||²/r，约束tI−E半正定。这个参照不复用生产求解器的Omega循环或Sylvester实现。q取1/2/3，r取2/3；目标函数最大绝对差约2.09e−9，全部在生产求解gap与检查容差范围内。SLSQP也是数值参照，不构成任意实例的形式化证明。
3. **实际运行快照与当前代码一致**：归档中的14个`tqg_gda/*.py`文件与本次审查工作树逐字节相同，见 [archived_execution_check.json](archived_execution_check.json)。排除了“只检查当前代码，而实验用了另一份核心代码”的情况。
4. **复核归档运行日志**：105次适配共有21,000步；其中12,000步实际构建查询，全部为`gradient_feedback`，无数值/零秩回退。11,939步rank16，61步rank15。反馈梯度等式最大误差0，SGD实际步长等式最大误差约1.01e−15。它们验证执行关系，不能验证估计梯度接近真实目标梯度。此项复核读取既有日志，没有重新评价目标标签。

独立检查可在仓库根目录执行（需要NumPy、SciPy及PyTorch；本次使用既有bfn4jet环境）：

```bash
OMP_NUM_THREADS=1 python docs/validation/v2_review_20261028/independent_calibration.py
```

脚本会更新相邻的合成检查JSON；运行差异在求解器容差内是允许的。

## 需要明确的局限及改进

**对照解释：KRR与联合校准的有效正则尚未匹配。** 当前两者均配置lambda_W=0.01，但联合问题在固定Omega=I/r时有D=2I，相当于KRR用lambda_W/2=0.005。一般联合求解还会按方向改变有效正则。因此现有差异不能单独归因于最大方向误差项。实现符合文档，这属于实验归因局限，原实验报告也已披露。后续应增加固定Omega对照，以及KRR正则敏感性/0.005对照；一般情形下单一KRR正则不能完全匹配所有方向。

**内部达到上限的有限近似解符合文档。** 实际12,000次查询有3次未达到gap容差，全部来自仅表示核对照；主版本3,000次均达到容差。文档允许保留达到迭代上限的有限近似读出，不能将其误报为主版本未执行或求解失败。

**目标batch16是可选预算，不是公式要求。** 当前源fit每步16、query全部196、目标每步16。按池独立随机抽样，同一步内不重复、跨步骤可重复，符合批次定义。对当前batch的完整空间不等于对整个目标域的完整空间；batch16/32/64及更均匀覆盖的效果尚待对照，不应预先认定增大batch必然改善。

**核心迁移假设尚无直接验证。** 代码能正确执行一个不准确的监督估计。当前正确query标签未稳定优于打乱标签，工程检查不能解释或消除这一问题。优先考虑开发集上的离线反馈方向诊断，再决定是否调整签名或局部支持；final标签继续保留，不作为调参依据。

审查期间仅新增本目录的报告与验证证据，未改写既有实验结果或checkpoint。
