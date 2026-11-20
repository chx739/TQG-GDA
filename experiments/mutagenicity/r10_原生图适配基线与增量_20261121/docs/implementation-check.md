# R10 作者源码对照与执行记录

官方文件来自 PyGDA 固定提交 `4ae3d8e7ce0ccef9f815a544ae763363e5210db0`。登记的三份 Git blob 已逐项吻合。源码、MIT 许可、作者执行参数和文件身份见 [固定快照](../src/upstream/SOURCE.json)。

生产 adapter 直接调用作者 GRADEBase 与 GradReverse。图级 JS 支路沿用 source 再 target 的前向顺序、源 NLL、两域交叉熵、权重 0.005、零起点 epoch 的 alpha 与 Adam。作者的 benchmark 把预生成 Data 中的 x 和 edge_index 原样交给模型。这里用同图协议认证出的 14 列节点 one-hot 与有向 edge_index，图读出仍使用作者 global mean pooling。A 与 H 均无 y，H 另用有序推断 loader。

合成工程检查执行冻结版官方 `forward_model` 与 `fit` 的方法体。JS 不使用的 MMD 和评价分支由只用于测试的桩隔离。相同随机状态下核对源与目标 logits、总损失、所有参数梯度、Adam 更新和整一步 shuffle。图结构变化必须改变适配梯度，A 带 y 必须报错，source-only 的冻结参数跨两方向推断保持不变。

CPU 实际设置为每模型两个 Torch 线程、一个 interop 线程。原生 DataLoader 和 dropout 沿用每进程固定 Torch 全局 RNG，所有臂共享参数初始化。source-only 只迭代 S，适配臂迭代 S 和 A，因此随机数消费随臂区别。训练严格为每轮每域一个 full-batch 和一个 Adam 更新。

持续记录目录为 `/home/hxchen/research/GDA/TSBA/experiments/r10/attempt-001`。模型进程列表见 `supervision/launch-*.jsonl`，每个模型的 PID、每轮覆盖与耗时见 `runs/<job>/attempt-*/process.json` 与 `epochs.jsonl`。完整模型写入独立 DONE 清单，继续运行会验证哈希并跳过已有完整模型，保留中断痕迹。首次启动只先安排首种子的三个模型。其 checkpoint 验证通过后才安排其余种子。
