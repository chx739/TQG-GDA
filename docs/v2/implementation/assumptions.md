# §3.8条件诊断：A0／A1／A2-a实现

按[2026-11-03方案](../plans/assumptions_20261103/README.md)执行第一批。A2-b及A3—A6未包含在此执行器。

- `assumptions_experiment.py`：旧几何回放、完整源／目标签名提取、四折／三折嵌套探针、标签权限、预测封存和运行预算。
- `tqg_gda/assumption_diagnostics.py`：KRR、LR、小MLP、训练折标准化、支持距离／ESS、显式A训练／B评价标签权限。
- `configs/assumptions_diagnostics.json`：冻结快照、模型种子、探针网格、训练步数及资源预算。
- `tests/test_assumption_diagnostics.py`：残差概率恢复、独立／批量MLP一致性、LR梯度、训练折标准化、折划分及权限边界。

复用初始fit模型的旧A16第0块V，同一V提取query196及目标A/B各271的签名。骨干不训练；只有诊断探针拟合。线性和MLP用当次训练折标准化，核带宽从全196源query输入固定。内部三折选择各探针超参数，完整预测封存后才读B标签。目标A探针不是UDA候选模型。

每个训练域／签名／外折有3个探针族、真实标签＋三次乱标签，各自选参数。2880个选择单元、34560次逻辑／物理拟合。MLP向量化训练多个独立模型，数值等价于逐模型拟合；不会把不同null混成一个模型。

正式执行代码保存在run的stage3_execution/code独立Git仓库，具体命令、版本与SHA见该轮execution_protocol.md和experiment_manifest.json。先prepare→A0→无B标签的5% pilot→完整矩阵；不从当前主仓库临时变化的代码接着训练。

## PROTEINS适配（2026-11-04）

新增`proteins_assumptions_experiment.py`作为数据／初始模型／新几何backend；共同worker按query实际大小计算四折训练数，38/39源与目标监督量逐折匹配。汇总与独立审计由实际ID、配置计数推导，不再假设147／196。13单测、六组旧Mutagenicity封存预测回归通过；旧归档保持原字节。

新A0检查现算几何的显式／生产路径一致性，不称旧PROTEINS v2缓存回放。目标B经验梯度投影仅在封存后计算。单类别评价池的BA记null，NLL主指标不变。具体[方案](../plans/assumptions_proteins_20261104/README.md)、[配置](../../../configs/assumptions_proteins.json)、[结果](../results/tqg_assumptions_proteins_20261104/结果解读.md)。
