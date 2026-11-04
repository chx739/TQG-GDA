# PROTEINS：TQG v2 §3.8条件诊断复验

源fit199／query51，源与目标每折各38/39张训练、源留出13/12张，目标A69／B70。三方向、五模型种子，H/HA×KRR/LR/MLP×真实/3次乱标签，共2880选择单元、34560次拟合、219960条逐图预测。所有GNN固定，无final标签。

**没有复现Mutagenicity的完整证据链：所有候选的源内和目标内主可学门槛均0/3；支持分组较小侧最低只有2～7张，不足30，支持判断证据不足。** 0→2 H-MLP有平均正信号但仅3/5种子同向。不能将这些结果直接解读为条件失稳或签名无信息，也不能把跨数据集差异单独归因于数据量。

- [白话结果与完整表格](结果解读.md) · [执行报告](run_report.md)
- [逐种子指标](results.csv) · [汇总JSON](results_summary.json)
- [三角比较](triangle_summary.csv) · [支持分层](support_summary.csv) · [跨数据集比较](cross_dataset_comparison.json)
- [独立审计](publication_audit.json) · [预先冻结的执行协议](execution_protocol.md)

D0/含13单测、六组旧Mutagenicity预测回归和15个新几何交叉核验；这里没有三个方向的历史PROTEINS v2缓存，因此不冒充历史回放。新几何的生产反馈、显式Jacobian、梯度、标签盲性与参数损失均通过。内层LR未收敛0条，MLP固定步结束不声称收敛。

logs/保存内层选参、逐步训练、TensorBoard、2880个选中探针状态、逐图预测、封存和权限日志。features/保存固定签名和坐标，protocol/保存原计划与旧开发/final输入ID清单。code_snapshot/与execution_history.bundle固定执行代码；analysis_code/和publication_code/保存只读汇总及独立发布校验。source GNN引用旧PROTEINS归档的5个模型，不重复上传；无真实图缓存。

目标B仅在模型、支持分组及预测封存后评价，并用已授权B标签计算A16空间对B经验梯度的投影外能量；不参与选参。A2-b、A3—A6及完整适配未执行。代码／计划／文件SHA见provenance.json、freeze_formal.json及manifest.json。

[全部日志、探针模型、预测与冻结代码](../../../../experiments/tqg_assumptions_proteins_20261104/README.md)。
