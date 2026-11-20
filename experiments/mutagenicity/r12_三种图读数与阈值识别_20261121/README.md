# R12｜同一张图的三种读数能否识别不同的分类工作点？

R11 显示，用 A 的标签选择阈值能改善分类，直接从分类分数估计类别比例在三个设置均未达到门槛。R12 先用带源标签的已有图构造两个可精确计分的目标分布。两者的分档分数和类别比例完全相同，最优阈值不同。只给估计器同一图上的分类分数、节点类型组成和无属性拓扑三个读数。首轮只评估它在真实源图所具有的依赖关系下能否恢复分类方向。[已登记方案](docs/source-screen.md) · [R11 先验对照](../r11_分类阈值的跨域检验_20261121/docs/prior-control-results.md)

已完成八源域，两个设置真实配对均为 0/4 域同时选对阈值。[筛查报告](report.md)

另已登记 [按图大小判决分组的三票先导](docs/conditional-votes.md)，检验自然目标域中能否利用联合投票纠正 GNN 和 WL 的共同错误。[相同观测下的正反例](docs/vote-identification.md)给出该方法可识别和失效的具体条件。

## 运行源内筛查

从仓库根目录使用同一配置。鉴权预检核对八组 S/V、冻结分数和图记录，无需解码标签。正式运行使用新的输出路径。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.scripts.run --config experiments/mutagenicity/r12_三种图读数与阈值识别_20261121/configs/source-screen.json --authenticate-only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.scripts.run --config experiments/mutagenicity/r12_三种图读数与阈值识别_20261121/configs/source-screen.json --output /tmp/r12-source-screen
```

[配置](configs/source-screen.json)绑定 R7 C 封存和 R5 历史输入。U 使用冻结的 V 原始 B0 margin。L 和 T 只在各自源域 S 拟合，随后封存 V 的预测和切点。经验 V 标签参与构造及评分。估计入口只接收匿名概率表。阈值打分在 float64 归一概率求和的舍入尺度内按同分处理。
