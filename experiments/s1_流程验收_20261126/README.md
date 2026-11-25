# S1 流程验收

本轮检查从图输入到冻结预测能否完整复现。使用固定 seed 3405，每个任务源训练 8 图、源验证 4 图、A 与 H 各 4 图，源训练 2 epoch。合成任务使用有类型的路径／环及少量附加边，在预测落盘后独立评分。真实任务使用 TSBA R5 源角色和 R7 固定目标角色，方向均为边划分 M0→M1。真实源端仅从 n≤64、m≤120 的图中选取，每类训练 4 张、验证 2 张；Mutagenicity 纳入一张包含稀有键型的源训练图。每个真实 H 仅输出预测。

```bash
python -m experiments.s1_流程验收_20261126.scripts.run --task synthetic --output smoke-runs/synthetic
python -m experiments.s1_流程验收_20261126.scripts.run --task Mutagenicity --raw-dir /path/to/Mutagenicity/raw --output smoke-runs/Mutagenicity
python -m experiments.s1_流程验收_20261126.scripts.run --task PROTEINS --raw-dir /path/to/PROTEINS/raw --output smoke-runs/PROTEINS
```

也可单独运行各阶段（以下在仓库根目录执行；原始数据路径由配置或环境指定）。

```bash
cfg=experiments/s1_流程验收_20261126/configs/Mutagenicity.json
python -m scripts.run_task train   --config "$cfg" --output run/source.pt
python -m scripts.run_task adapt   --config "$cfg" --source run/source.pt --output run/adaptation.json
python -m scripts.run_task predict --config "$cfg" --source run/source.pt --adaptation run/adaptation.json --output run/predictions.json
```

对获准评分的任务，独立执行 `python -m scripts.run_task score --predictions run/predictions.json --labels H-labels.json --output run/scores.json`。S1 只对合成 H 执行该步骤。验收脚本检查五种模式的自然概率、冻结源权重和跨进程重载预测的一致性。合成标签只用于测试接口；4 图规模的 bootstrap、预测变化及评分仅用于检验计算流程。[验收结果](report.md)
