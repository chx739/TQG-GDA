# R10 作者源码对照与执行记录

官方文件来自 PyGDA 固定提交 `4ae3d8e7ce0ccef9f815a544ae763363e5210db0`。登记的三份 Git blob 已逐项吻合。源码、MIT 许可、作者执行参数和文件身份见 [固定快照](../src/upstream/SOURCE.json)。

生产 adapter 直接调用作者 GRADEBase 与 GradReverse。图级 JS 支路沿用 source 再 target 的前向顺序、源 NLL、两域交叉熵、权重 0.005、零起点 epoch 的 alpha 与 Adam。作者的 benchmark 把预生成 Data 中的 x 和 edge_index 原样交给模型。这里用同图协议认证出的 14 列节点 one-hot 与有向 edge_index，图读出仍使用作者 global mean pooling。A 与 H 均无 y，H 另用有序推断 loader。

合成工程检查执行冻结版官方 `forward_model` 与 `fit` 的方法体。JS 不使用的 MMD 和评价分支由只用于测试的桩隔离。相同随机状态下核对源与目标 logits、总损失、所有参数梯度、Adam 更新和整一步 shuffle。图结构变化必须改变适配梯度，A 带 y 必须报错，source-only 的冻结参数跨两方向推断保持不变。

CPU 实际设置为每模型两个 Torch 线程、一个 interop 线程。原生 DataLoader 和 dropout 沿用每进程固定 Torch 全局 RNG，所有臂共享参数初始化。source-only 只迭代 S，适配臂迭代 S 和 A，因此随机数消费随臂区别。训练严格为每轮每域一个 full-batch 和一个 Adam 更新。

独立复核用生成图把适配臂与固定版作者循环逐轮对比。CPU float32 的 200 次损失和最终参数逐项一致。另用 3 张源图、2 张目标图构造 float64 稠密 GCN 参照，logits、所有梯度和一次 Adam 更新的最大绝对差依次小于 3×10⁻¹⁷、9×10⁻¹⁷、4×10⁻¹⁵。alpha=0 时域判别器收到梯度，域损失传向编码器的梯度为零。只删除 A 的边，在末轮 alpha 下可改变编码器梯度。这些是生成数据上的实现核验。

实际首种子 3405 的三臂初始化哈希相同，S 有 868 张图，两个 A 分别有 777 与 747 张。第一轮 source-only 的源 CE 为 0.69433862，两个 GRADE 臂均为 0.69421184，同一初始参数下三臂并未使用完全配对的源顺序和 dropout。因此 GRADE 减 source-only 估计两套原生训练流程的端到端差值；单种子差值无法单独识别域损失的因果贡献。

持续记录目录为 `/home/hxchen/research/GDA/TSBA/experiments/r10/attempt-001`。模型进程列表见 `supervision/launch-*.jsonl`，每个模型的 PID、每轮覆盖与耗时见 `runs/<job>/attempt-*/process.json` 与 `epochs.jsonl`。完整模型写入独立 DONE 清单，继续运行会验证哈希并跳过已有完整模型，保留中断痕迹。首次启动只先安排首种子的三个模型。其 checkpoint 验证通过后才安排其余种子。

第一种子完成后，独立验收从原档案核验三份模型身份、命令、PID、成功退出、文件集合和哈希、200 行采样覆盖、同初始化及未计分 H 预测。[首种子独立验收](../results/startup/seed-3405-verification.json)

恢复入口先比对原准备提交的训练源码，再校验已完成任务的精确成员集合、监管进程及子进程事件、角色和步数。训练进行时只追加读取核验，不改动原训练代码。评分前，全量验收覆盖 15 个模型，并从 checkpoint 重放两臂的 20 份 H logits。完整评价先写独立暂存目录；若只缺 FINAL，可用 COMPLETE 和 OBS 哈希恢复封存。

本次原始 launch 使用准备时已提交的训练代码。常规 `train` 和内部 `worker` 不是单独的版本认证入口，原训练源码始终与冻结提交逐文件核验。后续重启必须走 `validate --resume`，其校验范围为全部 15 个登记任务。若完整 DONE 缺少父进程退出记录，当前验收会明确拒绝该记录，保留独立回放或恢复所需的原始证据。
