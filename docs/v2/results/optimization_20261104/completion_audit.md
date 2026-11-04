# 目标逐项验收

对应用户目标文件SHA：`cf0ea09b5356d37baf15a5a13f896883a619b599e521061cec3173cfb73bd98d`。完成类型为“有限候选未通过，并形成可复核的负面结果与路线判断”；没有将目标缩减为跑通代码。

本表记录本地验收。远端同步以最终`git ls-remote origin refs/heads/main`核验为准，在同步成功前不标记目标完成。

| 目标要求 | 权威证据 | 判定 |
|---|---|---|
| 读取既有材料，区分执行/未执行，复用历史产物 | [执行协议](../../plans/optimization_execution_20261104/README.md)、各归档protocol/freeze_manifest.json及代码来源；历史报告链接见总报告 | 已完成，旧结果未覆写 |
| 预登记假设、候选数、权限、预算、源选择及停止规则 | 三阶段归档protocol/method.md，SHA均`51a5161eab4bfc95bcd3d6ba9c7b583268aa5dbf5938c1e8d6227f74e5b908a8`；config SHA均`304f301bbb3d1f2c1ed78219302aa53f6d189a4ba1ef634224729d35d6236796` | 已核验冻结字节 |
| MEAN/KRR及ISO正则等价、新增路径检查 | [D0恒等式](../../../../experiments/tqg_optimization_feedback_20261104/D0/mean_identity.json)，q=1/2/3和三真实任务H/HA；最大差4.47e-12；feedback_checks.log13项通过 | 已完成，不作性能证据 |
| A2-b全部同数量对照、两正则约定、全196 | [A2-b归档](../../../../experiments/tqg_optimization_local_20261104/README.md)，120块、6960配置，所有读出/控制/独立步齐全；audit.json重算一致 | 已完成 |
| A4-a完整四外折三内折和学习曲线 | [A4归档](../../../../experiments/tqg_optimization_a4_20261104/README.md)，960基础单元、368640逻辑拟合、230400曲线指标、92160网格外层重拟合 | 已完成，源/跨域/目标内分开 |
| 记录NLL/Brier/残差MSE/越界/核谱/复杂度/W | A4 curves.csv、models参数/特征谱、选中读取metadata及独立audit | 已完成；JOINT只报告读取，不伪装逐点概率模型 |
| 每路线至多一个源端选定候选 | A6 selection_choices.csv、每块selection_cv.json.gz；LOCAL四折8项、KERNEL三折48项；selection.json在评价前封存 | 已完成；未按目标B选路线 |
| 乱标签重做完整选参与拟合 | A6每块4套选择及variant_labels；TEACHER_LOCAL另外保存whole_pipeline_null；audit检查选择规则、标签排列、输出 | 已完成；条件W与全流程置乱分开 |
| 简单教师匹配源bank/输入/V，源内选择 | A6 TEACHER_H/HA/LOCAL源模型与选参；8640直接预测指标逐条重算；全/投影梯度一致 | 已完成；投影相等不当两种独立算法 |
| A6全部梯度、BASE、条件/全幅null及Oracle | A6 85320梯度记录，24读出控制臂/23教师控制臂；Oracle_M仅封存后用于诊断单步 | 已完成 |
| 同起点独立一步、源50/200快照、eta/beta固定 | A6 44640步骤记录；每配置原模型hash不变；3600次FULL/BASE普通模型SGD重放 | 已完成；不是连续适配 |
| 明确CE分母，逐种子配对和覆盖，后期守卫 | [A6聚合审计](../tqg_optimization_feedback_20261104/summary_audit.json)，3540数值/门槛检查最大差4.08e-16；全部结果/覆盖/snapshot CSV | 已完成 |
| A2-b误差降低门槛与敏感性 | [A2-b聚合审计](../tqg_optimization_local_20261104/summary_audit.json)，480检查，公开两种聚合口径 | 已完成；只有一个单任务判定变化，任何路线均未达到2/3 |
| 真实标签是否有增量、TQG是否胜教师/MEAN | [A6报告](../tqg_optimization_feedback_20261104/结果解读.md)，条件标签对照、全流程null、同bank/签名教师配对、固定MEAN参照 | 已完成；稳定机制优势未获支持，方法间一步幅度差明确披露 |
| 至少2/3通过才扩训练/组合 | results_summary.json LOCAL/KERNEL均0/3；独立聚合审计确认 | 条件未触发，不需执行组合/200步训练及四基线/新PROTEINS/门控 |
| final不读取、开发标签封存后评价 | seal.json/hash/label_access；原图角色重读审计检查A/B与final互斥；A4模型/CV/预测封存后B评价 | 已完成，final读取0；开发集已历史观察 |
| 数据、模型、样本与版本来源 | 各归档manifest、protocol冻结来源、sample_ids、每块ids中的checkpoint/cache SHA、code_provenance、代码快照及Git bundle | 已完成；旧GNN按SHA引用，不重复原图数据 |
| 保存日志、模型/读出、结果、图表、报告 | 三归档共10017文件，1,912,501,773字节；逐文件SHA重新核对通过，见completion_evidence.json | 已完成；小矩阵/教师参数归档，未上传原始图数据 |
| 公开所有失败/未收敛，不扩大候选 | A4有限未收敛观察器1；A6主读出1/教师内折1；导出修复日志及run_report说明 | 已完成，无额外候选或筛种子 |
| 计算预算与pilot | 各phase compute_estimate、pilot产物、done与publication_compute_ledger | 已完成；已计量主要计算约0.93 CPU核时、GPU0；未完全计量的导入/编排/归档开销单独披露，不称精确全项目CPU总量 |
| 结论、失败边界、下一步和论文主张 | [研究总报告](README.md) | 已完成；未将负面结果归因到单一因素 |
| GitHub同步与远端核验 | 归档提交及最终远端HEAD | 最终同步后核验；此前不完成goal |

## 已知审计边界

数值审计覆盖全部封存系数、读取、控制、指标及所选CV规则；普通模型只独立前向FULL/BASE参数步，没有再对每个null重复参数前向。教师优化器没有以另一套独立实现重新训练全部LR；审计校验训练数据角色、标准化、选参、保存权重生成的预测及停止梯度执行，保留优化器收敛标志。未把这些有限检查宣称为所有代码无错。

原始运行产物字节不变。归档前修正了阶段marker的开始时间为各worker RUN_INFO中的最早正式起点；publication_code保留该修正。publication_provenance的仓库commit表示归档工具基础版本，文件最终字节以artifact_manifest为准；这项时间元数据修正不影响实验代码、标签、预测、参数或研究结果。

本轮没有新增泛化/不可识别性理论主张，因此沿用已读的原始理论背景，不进行无边界文献搜索。负面结果不能单独换算成ICML接收概率，完整论文写作不是本轮交付要求。
