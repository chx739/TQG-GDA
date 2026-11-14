# Mutagenicity R4 固定配置全方向评估

R4 登记 12 个有向域任务和五个种子。主方案 A2 为 M11 学习 score，辅助方案 A1 为 M01 学习 score。第 30 次尝试的目标 Accuracy 是主终点。M0→M1 和 M0→M3 标为历史开发任务，其余十个方向是新增覆盖。四域使用仓库 SEGA 的有序索引。原始文件及逐图结构和特征哈希核对至 R3 的输入封存。比较范围是本实验的完整目标域开发评价。[机器计划](../configs/r4-plan.json)

每个源域按 SEGA 的 0.4/0.4 规则合并 ref 与 fit 用于训练，用 cal 的 CE 选源模型，平局取最早 epoch。实际角色数量按域和种子登记。源模型在各自三个任务间共用。每条任务使用自己的冻结目标坐标、带宽和批次序列，五条适配轨迹分别从共同初始头开始。

源训练 CPU float32，score 与适配 float64。冻结的完整 Config、Adam 设置、求解器容差和边界规则均写入计划。使用同带宽的经验 KDE 对照，学习 score 固定使用第 7,200 次更新。900、1,800、3,600 和 7,200 步留存诊断和优化器状态。源模型每 epoch、score 每 50 epoch、适配每次尝试原子发布 checkpoint，恢复已发布状态与随机流。四域全部重新训练，运行目录包含 R3 数据来源哈希和零训练产物复用的清单。

B0 为初始头。B1 和 B2 使用 R3 的 M00/M01 零请求求解。A3 为 M11 经验 KDE。每条任务的 M11 零请求验证工程驻点。第 1、30 次尝试使用 R3 的同状态正常、零和反向候选检查。完整状态预测逐次写盘，封存前逐份重放。状态、无锚点、驻点和拒绝照常计入 30 次尝试。工程异常在阶段失败日志中保留。

源标签按当前源域、种子和角色单独保存。源训练接收本源域训练和验证标签，score 只使用坐标，适配只读本源域训练标签文件。R4_OBS 校验任务注册、checkpoint、完整预测及重放计数后，评价入口才读取四个完整目标域的类别。评价输出保留每种子原始指标、端点及逐轮正确性转换，按种子配对后进行等权方向和域汇总。Accuracy 持平按正确图数的精确分数计算。跨方向汇总的 SD 是五个种子的平均方向成绩波动。

执行版本以 R3 报告提交 `7a3ac191b46ee6867a7840fc3b9542536d0efc78` 为基础，实际提交、依赖版本和代码哈希由运行器登记。输出目录通过 `--root` 指定，原始 R3 输入封存通过 `--r3` 指定。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.mutagenicity.r4_20261112.run \
  --root /home/hxchen/experiments/tsba/mutagenicity-r4/attempt-001 \
  --r3 /home/hxchen/experiments/tsba/mutagenicity-r3/attempt-001
```

## 事后补充对照 A0

A0 使用 R3 的 M00 正常请求，保留持续源 CE、学习 score 和原 R4 数值验收。M00 的源每类风险预算和置信度约束均不启用。每个已登记的方向与种子新增一条 30 次尝试的轨迹，共 60 条、1,800 次尝试。原计划保留 A2 为预登记主方案。

适配复用 R4 的源模型、冻结坐标、初始头、7,200 步 score、批次序列和源训练标签。补跑目录的 `protocol/references.json` 记录每个任务引用文件及哈希。统一重载预测并核对两条历史方向对应的 R3 M00 轨迹后，封存 `A0_OBS`，随后读取目标标签。第 30 次尝试为主终点。A0−B0、A0−B1、A1−A0、A2−A0 在同任务同种子内配对，汇总时在每个种子内等权平均方向。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m scripts.mutagenicity.r4_20261112.supplement \
  --root /home/hxchen/experiments/tsba/mutagenicity-r4/supplement-m00-001 \
  --r4 /home/hxchen/experiments/tsba/mutagenicity-r4/attempt-001 \
  --r3 /home/hxchen/experiments/tsba/mutagenicity-r3/attempt-001
```
