# 执行清单与准备程度

[返回列表](README.md) · [机器可读方案](plan.json) · [输入身份](input_manifest.json)

本轮设计资产核验：[design_checks.json](design_checks.json)。这里只记录配置/索引/文档核验，不包含训练或目标标签评价。

**本轮是设计工作，未启动真实预训练或适配。** 已做：源/目标索引核验、源角色数量与新模型参数量核验、配置解析。下面清楚区分现有能力与执行前仍需补的部分。

## 1. 可复用与必须适配的代码

| 工作 | 现有入口 | 状态/执行要求 |
|---|---|---|
| source fit初始化/缓存 | [pretrain.py](../../../tqg_gda/pretrain.py) | 已有；driver调用，不为了只预训练而额外跑一次适配 |
| 新模型加载/推断 | [predict.py](../../../tqg_gda/predict.py) | 已有；旧training.load_model不兼容新checkpoint |
| v1 TRUE | [v1/query.py](../../../tqg_gda/v1/query.py) | 已有；诊断需复用同一响应/Gram生成标签对照 |
| v2 TRUE/校准 | [v2/query.py](../../../tqg_gda/v2/query.py)、[calibration.py](../../../tqg_gda/v2/calibration.py) | 已有；诊断换标签不能重算W |
| 最终50步TRUE/FIT-CE | [pipeline.py](../../../tqg_gda/pipeline.py) | 已有；实验driver需增加step0/10快照与矩阵封存 |
| 目标A/B与开发标签隔离 | [p0_diagnostics.py](../../../tqg_gda/p0_diagnostics.py) | `development_pools`、`DevelopmentOracle`可复用/审计；禁止直接运行旧P0 driver加载旧模型 |
| FP32诊断几何 | 同上旧helper | **待适配**：旧`unlabeled_geometry`含FP64/CPU假设，新driver必须使用当前模型dtype及canonical查询，不能无声改变算法 |
| 固定标签置乱/MEAN-Y/同空间RAND | 新实验driver | **待实现/合成验收**；不往canonical Config偷偷放非主版本标志 |
| FIT+QRY-CE | 新实验driver | **待实现**；独立控制臂，保持query不参与预训练 |
| M1临时步与数值审计 | 新实验driver | **待实现/验收**；每臂复位，原模型与buffer不变 |
| 新checkpoint开发集评价与汇总 | 新实验driver/评价器 | **待实现/验收**；旧`scripts/evaluate.py`使用旧加载器，不能直接套用，更不能传整个target索引读取final |

新增实验代码建议放 `workflows/tqg/minimal_validation/`，当前尚未创建该runner。生产TQG公式保持不变；控制臂明确命名，不把消融冒充原版。

## 2. 配置资产

[配置目录](../../../configs/sega_style/minimal_20261106/README.md)中登记15个**现有Config可以解析的核心任务**：v1 TRUE 6个、v2 TRUE 6个、FIT-CE 3个。FIT-CE只运行一次，再由开发评价器评价两个目标域。它们不是完整的30轨迹矩阵；另15个控制任务需要上述driver（QUERY-CE 3个、PERM 12个）。

核心JSON都使用50步、统一源缓存目录及其余预设设置；缓存路径是可移植占位符。不要在M1前批量启动这些M2任务。`plan.json`的`stage`、`gate`、`control`等是实验元数据，**不是**可直接传给生产Config的字段。

## 3. 正式启动前的顺序

1. 补齐driver并完成以下合成验收：TRUE回放等于生产路径；PERM只改Y；MEAN-Y公式正确；v1/v2 RAND空间和范数正确；QUERY-CE梯度正确；临时步复位；source-only跨任务复用；final访问拒绝；FP32/FP64审计不重新求查询。
2. 固定完整代码commit、配置SHA、input_manifest和抽样/置乱清单。当前bc5bd81是算法起点，未来driver提交才是正式执行版本；不能在运行中悄悄换代码。
3. 检查缓存SHA与预登记一致；准备空输出目录。协议与数据身份保存到每个运行目录。
4. 执行M0→M1，生成逐版本/逐任务decision；只有满足门槛才运行M2。
5. 封存预测后做开发评价，报告全部种子与失败。更新独立结果页、原始CSV/JSON、SHA及模型对应关系，再推送TQG分支。

当前 `run_all` 只负责执行配置列表，不会自动实现上述门槛和标签边界，也不会把PERM或QUERY-CE补出来。

## 4. 预算上限

| 项目 | 上限 |
|---|---:|
| 新源预训练 | 3个模型 × 100epoch = 2,100优化步 |
| M1主查询几何 | 48个版本块，M0前3块可复用 |
| M1临时单步 | 624逻辑单步，最多576实际参数步；没有连续Oracle模型 |
| M2，两版都通过 | 30条50步轨迹；1,500连续优化步，其中1,200步构造TQG反馈 |
| M2，仅一版通过 | 18条轨迹；900连续优化步，其中600步构造TQG反馈 |
| M2，两版都不通过 | 0 |

墙钟时间以M0实测为准：`总耗时≈3×预训练 + 24×(v1查询+v2查询) + M1评价 + 入选版本各600×单步反馈训练 + 基线/封存评价`。不能把一个廉价源CE步与一个v2完整Jacobian步视为同等成本。

本轮没有超参数网格，也没有自动扩大到五种子/三个任务/200步的授权动作；先把这份最小证据链跑清楚。
