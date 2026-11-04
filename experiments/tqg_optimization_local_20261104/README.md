# A2b：已完成实验原始归档

[白话结果](结果解读.md) · [逐种子指标](results.csv) · [数值审计](audit.json) · [聚合审计](summary_audit.json) · [冻结协议](protocol/method.md)

包含全部读出、教师参数、CV选择、系数、封存前logits、开发评价标签、单步/梯度记录、日志、TensorBoard、样本ID、代码及Git bundle。没有原始图数据；不重复历史GNN。封存文件按原字节保存，seal.json与artifact_manifest.json分别验证块和整个归档。

原checkpoint/几何路径和SHA保留在blocks/*/ids.json；其来源为历史P0/H-only归档，可按SHA映射，而不是声称绝对路径可在任意机器直接运行。当前独立单步不是连续适配轨迹。

没有候选达到多任务继续门槛。A2-b误差聚合口径敏感性完整公开；A6一个主读出和一个教师内折有限未收敛保留。publication_compute_ledger.json区分已计量CPU与未完全计量的编排开销。
