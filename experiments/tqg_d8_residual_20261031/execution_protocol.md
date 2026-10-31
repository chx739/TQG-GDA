# D8与真实残差分解：执行前固定协议

授权：用户要求开始已知规律验证及平均残差／逐图关系分解。旧方案所有文件不修改。本补充协议用于冻结旧方案未列明的实现细节。

## D8

完整执行原D8。线性组seed27300+(model_seed−3405)，源fit512/query196、目标1024，θ三个原固定值，每θ8块目标16，不预训练。生成角色偏移按原D8。线性全目标梯度含A，不称独立B泛化。图组seed7300+(model_seed−3405)，四情形、源角色512/196/128/256、目标构造与独立评价各1024、GINE2 hidden32 mean、warmup固定200个Adam更新batch32 lr0.003、快照0/50/200。fit每步从512中随机不重复抽32，各步独立，采样generator model_seed+70000。

线性block_seed=81000000+(seed−3405)*10000+theta_index*100+block。图block_seed=82000000+(seed−3405)*10000+snapshot_step*10+block。图四情形共享A样本索引与目标隐变量；S3输入和S1完全相同，复用几何与读出，η翻转仅Oracle可见。源fit16用block_seed+1000。三次标签PERM和同空间等范数RAND沿用D1偏移。所有方法只用抽样源Y，不读η。源码不根据η选择V／核／λ或模型。源条件η仅作额外读取Oracle。

480图块、120线性块，三个方法原定HA-JOINT0.01、HA-MEAN0.005、H-MEAN0.005。S1每方法分别按阈值判：种子内快照等权、块平均后，整体TRUE条件余弦>0.2、两类null优势各≥0.05、至少3/5配对种子自身同时满足三阈值、覆盖≥80%。其他情形仅描述机制边界，不事后修改生成器。近零梯度、精确数值零、非收敛均保留。

首4图块额外η0.0003、β0.1独立条件CE单步：Q、S+TRUE、S+源条件读取、S+等范数条件Oracle，baseline为原θ与匹配源SGD。完整独立1024池评价，五种子共2880更新。该辅助结果不改变S1门槛。经验BA/AUROC在原快照公开，经验标签梯度与条件梯度分开。真实条件梯度、源噪声项、合并失配项、交叉项同时保存。

## 真实残差分解

H-JOINT与HA-JOINT，λ0.01，使用旧H/P0高维缓存，按SHA验证。三个任务、五种子、三原快照、8块，源query196、fit16、A16、B完整271。FULL=GLOBAL+CENTERED，r_i=(Y_i−p_i)Q。GLOBAL使用r的逐图均值，CENTERED为r_i−mean(r)。源标签置乱只改Y，不改p；其GLOBAL完全不变，PERM-CENTERED取置乱残差减去同GLOBAL。三次RAND-CENTERED在同V空间匹配CENTERED范数。不能将向量范数比例称为贡献百分比。

每快照前4块、η0.0003、β0.1；每候选13臂：S、S+FULL、S+GLOBAL、S+CENTERED、S+GLOBAL-M、S+CENTERED-M、S+ORACLE-M，以及S+GLOBAL+PC-1/2/3、S+GLOBAL+RC-1/2/3。M将相应反馈匹配FULL范数；PC/RC匹配CENTERED范数并保留同GLOBAL，控制逐图增量方向。共4680独立临时更新，不连续适配。

执行前约定逐图方向门槛：CENTERED c_A>0，相对PC和RC平均优势各≥0.05，各至少3/5种子差为正，覆盖≥80%。单步增量门槛：S+FULL相对S+GLOBAL的B池相对CE改善≥1e-4，至少3/5种子为正，S+FULL对两类同幅逐图null均值的B池改善均>0，覆盖≥80%。一个候选至少2/3任务同时满足两项才称稳定逐图监督增量；仅为研究筛选，不作显著性或真实完整适配声明。分种子／快照完整报告，旧P0结果不重写。

## 完整性、预算

全FP64 CPU、5 workers、每进程1线程，硬上限8小时，GPU时0。pilot独立目录、仅0步合成快照，不计正式五个模型。全部W在Oracle真相读取前封存；目标η/标签只用于诊断。真实final不可读。不新增真实预训练／完整适配。先19单测与50项D0，再pilot，再正式执行。失败日志、收敛状态与零比较不得删除。输出原始JSONL/TensorBoard、seed CSV、结果、代码SHA、15份合成checkpoint、报告并按用户授权同步GitHub。
