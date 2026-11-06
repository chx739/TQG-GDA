# 共同协议：数据、配置、抽样和判断口径

[返回列表](README.md) · [执行清单](execution.md)

## 1. 固定数据与起点

沿用 [源划分](../../../experiments/mutagenicity_v2_20261026/protocol/data_roles/source_protocol.json) 与 [目标角色](../../../experiments/mutagenicity_v2_20261026/protocol/data_roles/)，只读取元数据/索引核验，不读取 final 标签。

| 项目 | 固定值 |
|---|---|
| 数据集 | Mutagenicity，4,337 张图，二分类；SEGA 图规模域索引，非官方自然域 |
| 任务 | 0→1、0→3；0→2 不进入本轮选择 |
| 源域 | 1,084 = 975 train + 109 validation |
| 源角色 | 975 = 779 fit + 196 query；按类别划分，seed=431，fit_fraction=0.8 |
| 目标输入 | 域1：1,084；域3：1,085；全域输入可用于无标签 transductive 适配 |
| 目标开发标签 | 每域542张；沿用既有ID，不重新划分出所谓“全新测试集” |
| final | 域1：542；域3：543；只允许输入，禁止读取标签 |
| 模型种子 | 3405、3406、3407；所有种子保留，不挑最好起点 |
| 缓存SHA256 | `c4a03120edd2494ab82bcc9792605ea7e71d52024985624313ae11d9f37b4fab` |

本轮只读核验确认：本机缓存14,977,860字节；节点标签输入14维、边特征3维；fit/query/validation数量与上述一致。本轮 GIN 默认不使用边属性，GINE 不在最小矩阵中。

M1 的开发 A/B 沿用 `development_pools` 的按图 ID 哈希规则：542分为271/271。A每块抽16张，用于**无标签查询**；B整池271张用于独立单步评价。A/B互不相交，均与final不相交。这里的“独立”指图ID不同，**不是从未被历史研究观察过的测试集**。

## 2. 冻结配置

| 项目 | 设置 |
|---|---|
| 新骨干 | SEGA GIN，3层，hidden=64，embedding=16，逐层sum pooling后拼接；线性二分类头；35,957个参数 |
| 预训练 | fit-only 普通CE；100 epoch，batch128，Adam lr=0.001，weight decay=0 |
| 预训练预算 | 每epoch `ceil(779/128)=7` 步，最多700步/种子；不丢最后一批 |
| 起点选取 | 独立source validation的最低CE epoch；不同方法和任务使用同一份已选checkpoint |
| 适配 | SGD，无动量、无weight decay；lr=0.0003，β=0.1 |
| 每步数据 | fit16、全部query196、目标16；fit/query/target各自独立RNG |
| 网络状态 | 预训练正常train/eval；查询与适配全程eval，固定BN运行统计和dropout，参数仍可训练 |
| 精度/设备 | 主配置FP32，核与v2校准FP64；首轮CPU、1线程、确定性模式 |
| v1 | target方向，K=4，ρ=0.05，ridge=0.01，odd+even |
| v2 | H+A RBF，完整数值有效空间，λW=0.01，联合均值+最大方向校准；其余沿用当前默认容差 |
| M1 | 仅θ0；每个任务/种子4块；不加入源训练50/200步快照 |
| M2 | 50步；保存0/10/50，主终点固定50；不挑最佳目标checkpoint |

CPU选择来自当前环境只检测到CPU。若后续指定GPU，先登记单独运行时并完成合成/真实输入数值检查；设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。不要混用CPU与GPU种子结果假装同一执行条件。

## 3. 配对、随机数和标签对照

- 先冻结全部图ID和RNG状态，再构造反馈；最后才允许诊断读取目标开发标签。
- M1对 `(seed, domain, block=0..3, role)` 使用确定性SHA256派生整数种子，前缀固定为 `tqg-sega-minimal-20261106`；禁止 Python 随机化 `hash()`。实现应保存完整抽样清单。
- fit从779中每块无放回抽16；query使用排序后的全部196；A从固定A池每块无放回抽16，跨块允许重复。两版共用fit/query/A/B图ID；v1另有独立probe RNG。
- M2沿用当前Adapter的角色RNG（seed+201/202/203/301）；各臂每步fit与目标ID一致。source-only和query-CE轨迹不依赖目标，可以训练一次、分别评价两个域。
- PERM的三个置乱用源query**全局图ID→置乱标签**映射固定，前缀同上、角色 `perm/1..3`，每个模型种子一组三映射，两版/两个目标任务共用。不置乱整个 `Y-p`，预测项保持原图对应。
- M1使用3个置乱；M2预先固定用 `perm/1`，不能按M1结果挑“最好或最坏”的置乱。每步即使query顺序改变，也按图ID取固定错配标签。
- MEAN-Y：用真实query类别均值 `barY=mean_i Y_i`，残差为 `barY-p_i`。TRUE与MEAN-Y之差只来自逐图中心化标签 `Y_i-barY`。

M1种子派生固定为以下规则；fit/A角色跨版本共用，probe/rand的role包含版本名。置乱使用 `domain=0, block=0, role=perm, rep=1/2/3`，因此跨目标任务和版本共用映射：

```python
message = f"tqg-sega-minimal-20261106|seed={seed}|domain={domain}|block={block}|role={role}|rep={rep}"
derived_seed = int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "big") % (2**63 - 1)
```

## 4. 标签边界与结果封存

训练接口只接受目标输入；M1的Oracle由独立开发评价接口提供，记录每次label access、用途和图ID。先封存快照、查询几何、TRUE/置乱/均值/随机反馈，再读A/B标签。Oracle随后只在临时模型副本中用于定位，不进入训练轨迹。

M2的checkpoint和所有臂预测必须先封存，才读取完整542张开发标签进行比较。阶段结果可以决定是否扩大后续研究，但这属于**开发集研究选择**，不能声称完全无选择偏差。final标签本轮始终禁用。

## 5. 指标与聚合

- M1：主要是目标A经验CE梯度的余弦、对null的配对优势，以及B上的额外CE改善。目标标签梯度有抽样噪声，不等于条件期望梯度真值。
- M2：balanced accuracy（BA，%）和普通NLL/CE同时报告，另报AUROC、类别召回、混淆矩阵与源验证CE。
- BA差使用**百分点**；CE改善使用相对比例。三个种子先分别聚合其4块/3个null重复，再跨种子等权聚合。
- `±` 为三个模型种子的样本标准差（ddof=1）；同时列出每个种子及胜/平/负。不把48块或临时单步当独立模型数，不用三种子宣布显著。
- 零反馈、零秩、非有限fallback、未收敛、缺失比较全部保留；余弦遇零范数为NA，不填0，不只汇总“有效块”。
- 下文门槛是**预设研究筛查规则**，不是理论保证或通用最优阈值；调整门槛要登记新方案，不能回改使当前结果过关。
