# 同图、同权限的图级适配比较

2026-11-21，登记于 R9 的来源内筛查后。R8 固定均值开发诊断相对 SourceCal 的平均 Accuracy 有改善，交换斜率对照未通过。R9 两跳结构投票未胜过源监督结构模型与同权重对照。当前问题是，一种独立发表的图级适配实现，在同一批真实图上能否取得源监督以外的 Accuracy，以及我们的历史结果处于什么位置。本轮只训练公开参照及其源监督对照。

## 锁定的数据和模型

- 复用 R8 attempt-001 的 Mutagenicity 边数划分 M0→M1、M0→M3，种子 3405–3409，认证源 fit、A/H 的 ID。OBS 身份为 `3e691eb79b6027a8077cb78eb94ff3333d988dc8e5dc4123618579dfc4328969`。每种子仅使用源训练角色标签，类别空间来自源。适配可用 S 与无标签 A。H 单独构建推断 loader，固定顺序、eval 模式、逐图核对 ID。这两个方向已参与研发。
- 采用 PyGDA 为 GDABench 提供的 graph-mode GRADE，提交 `4ae3d8e7ce0ccef9f815a544ae763363e5210db0`。作者在 Mutagenicity 两方向使用相同参数：JS 域判别器，5 层 GCN，hidden 128、dropout 0.5、Adam lr 0.001、weight decay 0.001、域损失权重 0.005。按默认 batch_size=0 为 S 和 A 各一个 full-batch，每轮一次更新，训练 200 轮，用第 200 轮预测。梯度反转系数按作者 epoch 序号调度。使用作者模型原生 node features、edge_index 与图池化。本仓库按边数划域，作者原表按密度二分。[附录 E](https://papers.neurips.cc/paper_files/paper/2024/file/a2c548b2ed3141400d86a235eaa6fad0-Supplemental-Datasets_and_Benchmarks_Track.pdf) · [作者参数](https://github.com/pygda-team/pygda/blob/4ae3d8e7ce0ccef9f815a544ae763363e5210db0/benchmark/graph/run_all_M.sh) · [训练实现](https://github.com/pygda-team/pygda/blob/4ae3d8e7ce0ccef9f815a544ae763363e5210db0/pygda/models/grade.py)
- 固定 PyTorch、NumPy 及 loader 随机状态。每个种子的 source-only 与两条适配方向共享初始模型参数。source-only 只前向 S，用同一源 CE、优化器与 200 次更新，训练一次，在两方向 H 上推断。GRADE 分别在两个 A 训练。记录设备、每轮样本覆盖和步数、首末损失、初始化与源角色摘要。可按模型和种子分批执行；改变 full-batch 训练轨迹或缩短轮数需要另行说明。
- 优先直接复用固定版本的 `GRADEBase`、`GradReverse` 及 JS 损失。使用本地薄适配器时记录官方文件及 blob，保留 MIT 许可；用生成的源/目标图批次从相同初始化独立核对官方 graph forward、总损失、分类和域判别梯度及一次 Adam 更新。作者 `predict(data)` 复用训练 loader；这里用独立的 H loader 推断。工程回归还覆盖 A 结构改变适配梯度、source-only 参数跨方向复用、H 标签与训练分离。[原生基线源码核对](../../r9_目标近邻与源标签传递_20261121/docs/native-comparators.md)

## 计分与用途

工程等价检查通过后运行五个源基线和十个适配模型。先用种子 3405 核验实际内存、耗时、步数及无标签输出，再用同一配置继续。十块预测全部冻结后开放 H 标签计分。运行中断时保留完整模型与哈希，同配置重做中断的模型，所有登记种子都计入分母。

第一比较是 GRADE 减同架构 source-only 的普通 Accuracy。逐方向、逐种子报告分母、纠正、损害、pp；种子内方向等权。辅报 BA、每类召回、NLL，复用 R8 的 2,000 次目标域×类配对图重采样，另报训练种子波动。再列与同批 H 的已冻结 B0、SourceCal、SourceFusion、R7 J_argmax、R8 主臂 Joint 和事后固定均值 TargetSlope 的配对差。逐块认证预测与 H ID，标注事后开发结果，保留各自固定决策。LP-TGNN 尚无同协议实测。

若 GRADE 在两方向都胜同架构源模型，可定位这个独立适配机制纠正了哪些错误；若只在一个方向受益，需把另一方向的损害单列；若未胜 source-only，则其绝对成绩本身不能证明域适配带来了增量。这两个开发方向不足以支持跨数据集的稳定优势。
