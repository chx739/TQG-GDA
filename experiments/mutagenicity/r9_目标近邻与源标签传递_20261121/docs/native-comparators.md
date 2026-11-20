# 真实开发阶段的图级基线准备

2026-11-21，来源内试验期间核对。此处记录如何公平接入两份公开实现，尚无这两个方法在 TSBA 的相同图 ID 上的 Accuracy。

## LP-TGNN

作者模型同时使用图卷积和拓扑图像，在主表比较图级无监督迁移方法。已核对作者提交 `c8625a329a9f061994a27586888c407452c57858`。默认训练 50 轮、每轮 50 个 batch，源标签训练分类头，目标图提供硬类别。训练入口每轮在目标图上读取真标签并输出最高准确率。此处若运行，将沿用同一批 S/A/H，类别表从源定义，锁定轮数，统一在 H 上评估。配套 source-only 用相同拓扑图像和模型，在相同输入特征编码下从已训练的 source_branch 推断 H。直接走默认的独立 target_branch 会把未受源监督的随机输入层引入对照。[论文](https://proceedings.mlr.press/v280/wen25a.html) · [训练和选模](https://github.com/TaoWen0309/Label-Propagation-GNN/blob/c8625a329a9f061994a27586888c407452c57858/main.py) · [双输入分支](https://github.com/TaoWen0309/Label-Propagation-GNN/blob/c8625a329a9f061994a27586888c407452c57858/models/tensorgcn.py)

独立源码审查和无图的 PyTorch 2.5.1 小样例表明，作者的 0.8 门槛比较最大原始 logit。目标输出通过离散类别和阈值决定各类损失权重，专属目标输入线性层没有来自该损失的反传梯度；共享模型参数及 BatchNorm 统计量仍可受目标输入影响。官方 CPU 代码在有样本过门时对叶张量做原地累加会抛错，若源小批缺少某个已接纳的类别，空均值可能产生 NaN。接入时保留原阈值单位，单列纯数值等价的修复及对应回归；凡需改变缺类损失定义的情况记录次数并另行说明。真实数据上的影响须实测。图像缓存需从每张图的无标签拓扑独立计算。当前本地研究环境有 torch 2.5.1、PyG 2.6.1，缺少 GUDHI 与 tensorly-torch。[标签传播和图像计算](https://github.com/TaoWen0309/Label-Propagation-GNN/blob/c8625a329a9f061994a27586888c407452c57858/util.py) · [分类头](https://github.com/TaoWen0309/Label-Propagation-GNN/blob/c8625a329a9f061994a27586888c407452c57858/models/mlp.py)

## GDABench 的图级 GRADE

GDABench 附录 E 在 Mutagenicity、PROTEINS 等的两域密度划分上评测图级版本。作者发布的 Mutagenicity 脚本使用 5 层、隐藏维 128、学习率及 weight decay 均 0.001、200 轮、dropout 0.5、适配损失权重 0.005。已核对提交 `4ae3d8e7ce0ccef9f815a544ae763363e5210db0`。其 `predict(data)` 迭代 fit 时创建的目标 loader，直接传 H 仍会预测 A。需要用模型 eval 和单独的无标签 H loader 取出 logits，逐图验证 H ID，预测封存后再评分。相同图网络只用源监督训练作为配套对照。作者结果使用 Micro-F1；单标签全类别时等于 Accuracy。两域原表不能作为本仓库四域任务的实测对照。[附录 E、表 15](https://papers.neurips.cc/paper_files/paper/2024/file/a2c548b2ed3141400d86a235eaa6fad0-Supplemental-Datasets_and_Benchmarks_Track.pdf) · [作者参数](https://github.com/pygda-team/pygda/blob/4ae3d8e7ce0ccef9f815a544ae763363e5210db0/benchmark/graph/run_all_M.sh) · [fit 与 predict](https://github.com/pygda-team/pygda/blob/4ae3d8e7ce0ccef9f815a544ae763363e5210db0/pygda/models/grade.py)
