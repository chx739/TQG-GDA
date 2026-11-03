# TQG v2 §3.8：A0／A1／A2-a条件诊断

三迁移任务、五模型种子，源／目标探针训练各147张，四外折与三内折。2880个选择单元／34560次拟合；851040条逐图预测。不是完整UDA训练，不更新GNN，目标A标签只训练诊断探针，B在所有预测与选择封存后评价，final不读。

- [白话结论与全部三角／支持表](结果解读.md) · [执行报告](run_report.md)
- [汇总JSON](results_summary.json) · [模型种子指标](results.csv)
- [三角汇总](triangle_summary.csv) · [支持汇总](support_summary.csv) · [原查询几何](geometry_summary.csv)
- [独立审计](publication_audit.json) · [执行协议](execution_protocol.md) · [正式执行前工程补充](execution_addendum.md)

H／HA的LR与MLP在源内、目标内均能读到标签信息，但三任务均有源训练向目标迁移的NLL缺口。H支持分层三个任务都通过代理筛选；原HA残差KRR的同域读取较弱。结果支持优先登记局部源bank与核复杂度诊断，不唯一证明跨域条件失稳，也未证明梯度反馈已经有效。

logs/含内层选择、逐步训练、TensorBoard、选中模型、逐图预测、样本ID及预测／模型封存。features/含固定签名与坐标，不含真实图原始缓存。源GNN模型引用已有归档，不重复上传。代码快照与execution_history.bundle保留正式及initial/pilot执行提交；analysis_code/与publication_code/保留只读后处理及独立发布审计。

A0与pilot使用初始执行提交b6fc5d6；正式执行7222b65。差异仅为空分组字段及增强审计封存，补充协议在首次B评价前固定。1条内层LR乱标签拟合有限未收敛，完整保留；所有选中重拟合正常，MLP按固定步结束不宣称收敛。A2-b、A3—A6未执行。

校验全部归档文件请读manifest.json；完整训练模型的旧evaluation_index格式不适用于探针本归档。复现需匹配真实数据缓存SHA、已有模型和协议布局，不能直接在已完成目录重跑覆盖。

[全部日志、选中探针、逐图预测、冻结代码及SHA](../../../../experiments/tqg_assumptions_20261103/README.md)。
