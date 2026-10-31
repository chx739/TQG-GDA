# D3校准与D4相似度诊断实现

依据[冻结方案](research_plans/tqg_v2_20261031/README.md)，只运行D3/D4，不启动D5—D9。P0原始记录、输入快照、数据角色和方案哈希均保持不变。

入口为`d34_experiment.py`及`d34_summarize.py`，执行配置为`configs/d34_diagnostics.json`。无标签核、ISO读取器和源内Oracle位于`tqg_gda/d34_diagnostics.py`。生产校准只新增可选`return_details`观测参数；默认返回内容、计算和训练调用不变。新增测试验证详细观测不会改变W或原统计。

执行前固定以下补充细节；不改变原假设或门槛：

- D3复用P0全部360个无标签几何块；5个α×3个读取器，共5400逻辑配置。主α0.005；MEAN λ=α，ISO/JOINT λ=2α。每个α都验ISO与MEAN等价；原JOINT0.01逐块回放P0 W。
- D3单步采用D2前4块和主η0.0003、β0.1；15臂各做原幅度／等范数，共5400个临时步。等范数参考为同块MEAN α0.005；增量比较JOINT与匹配ISO。
- D4五核始终使用joint λ0.01，不根据D3结果临时切换某个核的读取器。本轮不运行可选统一MEAN矩阵。
- D4真实目标组360块、1800条梯度记录；前4块做1800个临时步。等范数参考H；主增量为HA相对H。HA的相同求解可复用D3主JOINT，标记复用，不把复用耗时记成独立求解。
- D4源内组5种子×3快照×2折×4块=120；bank98、A16、B82，产生600条梯度和1200个临时步。源侧快照统一取公共fit best及0→1 B_fit 50/200步；源fit16抽样seed=源块seed+1000。
- HA-PERM使用输入无关generator(block_seed+9501)，同时置乱K_SS两轴及K_ST源行；残差和目标响应不置乱。bank行顺序固定为原图ID排序。
- 全部单步共8400个逻辑更新；每次从原θ开始。D3/D4的正增量门槛不替代候选的D1/D2门槛。

源内bank98和真实bank196不是相同样本量的域对照；只作机制定位，不能据此严格归因域偏移。真实和源内经验梯度均有标签噪声，均不等于已知条件期望梯度。

运行副本保存在`/home/hxchen/runs/2026-10-31-tqg-d3-d4/stage3_execution/code/`，阶段2是原计划的逐字节复制。复制本仓库上述入口、P0公共接口、tqg_gda、测试及配置并固定Git提交；记录完整来源。先在该目录依次执行：

```bash
python d34_experiment.py prepare --repo /home/hxchen/TQG-GDA --stage ../
python d34_experiment.py d0 --repo /home/hxchen/TQG-GDA --stage ../
python d34_experiment.py pilot --repo /home/hxchen/TQG-GDA --stage ../
python d34_experiment.py run --repo /home/hxchen/TQG-GDA --stage ../
python d34_summarize.py --stage ../
```

固定pilot只运行3405、初始快照、第0个目标块及第0折源块，不计正式统计。CPU／FP64、5个单线程worker；每个标签请求必须在该块全部读取矩阵封存之后，开发Oracle与源query留出Oracle分别记录。最终目标标签全程禁止。大几何缓存仅留本地，公开原始指标、有限日志、代码和哈希。
