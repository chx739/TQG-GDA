# 标签归因与D7-H执行说明

当前入口：`python -m workflows.v2.label_query.label_query_experiment`；汇总：`python -m workflows.v2.label_query.label_query_summarize`。见[脚本导航](../../../workflows/README.md)；旧归档使用当时的冻结命令。

执行前协议放新run的execution_protocol.md，旧方案不修改。本次仅H-only：360块标签／预测拆分，3420独立单步；D7-H仅初始模型120块，6000逻辑配置／4920物理解、3000逻辑单步／2460实际更新。复制恒等式8例为D7-B工程验收，D7-C未执行。

新增tqg_gda/query_label_diagnostics.py、label_query_experiment.py、label_query_summarize.py及configs/label_query_diagnostics.json。复用旧H缓存，固定公共带宽、目标V／W及模型θ；标签控制保留BASE，仅改变并匹配标签项。完整等幅控制单列，避免将标签／预测项抵消误读为新信号。query子样本严格在原196池，按SHA排序嵌套，λ固定／n缩放两约定并报。

沿用所有数据角色和Oracle先封存W后读dev边界，不读final、不新预训练或长期适配。全部随机种子、角色ID、权重SHA、紧凑系数、收敛状态和原始指标可复核。工程检查与研究门槛分开，门槛写入执行前协议。

入口prepare→d0→pilot→run，汇总label_query_summarize.py。使用PyG原诊断模板与函数式参数更新，CPU五进程、FP64；试跑包含完整50配置及19臂，按总360块保守计时估算，8小时硬上限。旧H的FULL和n196各逐块回放。
