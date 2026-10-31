# H-only D1/D2对照实现

用户授权验证D4中H候选，不能将D4相对HA优势当作已证明源监督有效。

`h_only_experiment.py`继承P0的完整方向／单步driver。P0新增geometry与worker_entrypoint扩展点，默认行为仍走原函数。候选仅替换输入无标签缓存中的核和W为D4 H读出，并逐块检查W哈希；模型、V、响应、样本ID、概率、query、随机种子不变。

读取器仍为JOINT λ=0.01。完整保留D1/D2三次null、Oracle、三步长及主步长0.0003、β0.1、360／180块与原门槛，旧方案不修改。正式执行前在新run根目录封存candidate_protocol.md，作为显式候选修订。源query196；源fit每块16，不从query拿标签训练骨干。

读取旧P0／D34输入，验证所有输入SHA及checkpoint，先固定W再读开发标签。已观察的开发集仅作探索性诊断；无final标签，无连续训练。

执行顺序：`h_only_experiment.py prepare`、`d0`、`pilot`、`run`；每次传`--repo`、`--stage`、`--config configs/h_only_diagnostics.json`，最后`h_only_report.py --stage ...`。冻结代码从stage3_execution/code运行；原P0汇总器复用不改判据，H报告新增候选身份、D4逐块回放审计和解释。归档复用`scripts/archive_p0.py`角色／数值／哈希审计后，标明H-only身份。

验收：`python -m unittest tests.test_h_only_diagnostics tests.test_p0_diagnostics tests.test_d34_diagnostics -q`，14项通过。D0另执行旧生产路径代数／权限回归，以及三任务H权重回放。研究效果与工程验收分开记录。
