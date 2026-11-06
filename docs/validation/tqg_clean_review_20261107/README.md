# TQG 当前实现独立代码审查

审查日期：2026-11-07。被审仓库：[chx739/TQG-GDA](https://github.com/chx739/TQG-GDA)；分支 `TQG`；被审提交 `8ff9abe7e38f5f64578f22994bcdd6d9117eaae5`；算法重构提交 `bc5bd81d34fff10cd953e34a331ceaf98d875992`。

## 结论与 findings

**本轮未发现能够确认、需要修复的新增 P1/P2/P3 缺陷。** 在下面界定的审查与合成验证范围内，没有证据表明 canonical v1/v2 偏离方法文档的核心公式、梯度符号、停止梯度边界或一次查询一次更新规则。这是工程审查结论，不是对迁移假设、真实数据收益或所有运行环境的保证。

| 严重程度 | 已确认 findings |
| --- | --- |
| P1 | 0 |
| P2 | 0 |
| P3 | 0 |

研究假设尚未得到验证和执行清单中已记录的待办项，单独列在末尾的限制中。

## 审查方式与范围

本 reviewer 以干净子会话启动，未继承主会话的实施推理。先读 `AGENTS.md`、两版 `docs/v*/method/design.md` 与 canonical 源码，再阅读现有测试和实现说明。参考 `references/sega-gda/sega/encoder.py` 的结构，只读参考快照。主端随后提供的测试结果与本 reviewer 新执行的检查在下文分开登记。

逐项阅读：`tqg_gda/{config,data,encoder,classifier,pretrain,adapt,predict,pipeline,provenance}.py`、`tqg_gda/v1/query.py`、`tqg_gda/v2/{query,calibration}.py`、`workflows/tqg/`；补查了相关 `utils.py`、`metrics.py`、测试和新实验配置/执行清单。

| 检查对象 | 核对位置（当前提交行号） | 审查结果 |
| --- | --- | --- |
| v1 目标方向与探测作用域 | `tqg_gda/v1/query.py:51`、`:149` | 编码器参数 VJP，类别中心化随机投影，除 `sqrt(bT)`；方向用目标 JVP 的每图类别平方和 RMS 规范化；头参数不受探测扰动。 |
| v1 双向响应、共同尺度 | `tqg_gda/v1/query.py:80`、`:173` | 同方向/同半径的正负探测；odd 除 `2*rho`、even 除 `rho²`；源/目标各半权重、按图均值、沿类别求和；两域共用尺度，无跨图减均值。 |
| v1 拟合、符号与执行 | `tqg_gda/v1/query.py:89`、`:143`、`:195`、`:255` | 源残差 `Y-p`，Gram/相关量均除 `bS`，线性求解带 ridge；目标参考为基准加校正；损失是 `1/(2*bT)` 的图/类总平方，提供负目标 Jacobian 转置乘校正的梯度。 |
| v2 全参数空间与图响应 | `tqg_gda/v2/query.py:42`、`:80`、`:102`、`:203` | 分类头与编码器共同纳入 Jacobian；固定 Helmert 对比基；按记录的机器精度阈值取数值行空间；源 JVP 与目标 `J@V` 坐标一致。 |
| v2 签名核、块顺序 | `tqg_gda/v2/query.py:117`、`:146` | 源侧成对距离中位数和显式下限；H+A 非线性核；三类核共享尺度；graph-major 类别块，`B` 已含 `1/bT`，Kronecker 张量计算与文档一致。 |
| v2 联合校准 | `tqg_gda/v2/calibration.py:76`、`:108`、`:124`、`:150` | `D=I+r*Omega`；条件 Sylvester 解包含相应方向特征值与 ridge；误差矩阵、均值/最大方向/权重正则一致；回溯候选先求自身 W；gap 始终对应同一 Omega/W。 |
| v2 监督读出与反馈梯度 | `tqg_gda/v2/query.py:220`、`:239` | 求 W 前不读取源监督值；随后 `d=W*rS`，没有多除 `bS`；参数坐标损失没有多除 rank，当前点梯度是 `-V*d`。 |
| 状态与停止梯度 | 两版 snapshot；`tqg_gda/adapt.py:31` | 参数/buffer 克隆、功能前向，查询张量停止梯度；适配所有前向使用 eval，BN 仿射参数仍可训练；新查询后唯一一次 optimizer step。 |
| 源角色与目标权限 | `tqg_gda/data.py:51`、`:101`、`:121`、`:162`；`pretrain.py:54` | manifest SHA、角色互斥与边界检查；fit-only 预训练和源主损失；query 标签进入反馈；validation 用于源预训练选模；目标分离前输入视图已无 y，两版查询拒绝带 y 的目标 batch。 |
| 缓存与 checkpoint 推断 | `pretrain.py:22`、`:82`；`provenance.py:29`；`predict.py:11`；`pipeline.py:19` | 源缓存跨目标/版本共享，身份含数据/代码/配置/运行时，复用校验产物 SHA；恢复模型结构、dtype、状态及输入预处理；checkpoint 推断不依赖查询库。 |
| SEGA 骨干与 CLI | `encoder.py:13`；`classifier.py:9`；`workflows/tqg/run_task.py:9`、`run_all.py:11` | MLP 内 BN、卷积后 BN、逐层 sum pooling 拼接及投影一致，保留 TQG 线性头；dtype 适配明确；主 CLI 可导入和解析。 |

## 本 reviewer 新执行的合成检查

环境：`/home/hxchen/anaconda3/envs/bfn4jet/bin/python`，CPU，单线程。复现脚本：[synthetic_checks.py](synthetic_checks.py)；独立 reviewer 原始输出：[synthetic.original.log](synthetic.original.log)。归档脚本仅将仓库根目录改为按文件位置解析；主端复跑输出见 [synthetic.replay.log](synthetic.replay.log)。在仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python \
  docs/validation/tqg_clean_review_20261107/synthetic_checks.py
```

生成小型完整合成图，使用默认宽度 hidden=64、3 层、out_dim=16，并改为三分类，以覆盖两个类别对比坐标。仅构造查询、求梯度和推断，没有真实数据训练，也没有向查询传入目标标签。

| 骨干 | 精度 | 参数数 | v1 梯度相对误差 | v2 梯度绝对误差 | v2 rank | 校准 gap / 收敛 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GIN | FP32 | 35,974 | 9.76477e-8 | 0 | 6 | 1.00084e-11 / 是 |
| GINE | FP32 | 36,542 | 0 | 0 | 6 | 3.16547e-12 / 是 |
| GIN | FP64 | 35,974 | 0 | 0 | 6 | 2.27374e-13 / 是 |
| GINE | FP64 | 36,542 | 2.39760e-16 | 0 | 6 | 2.01617e-12 / 是 |

- v1 对照独立计算的 `-(model(target)*correction).sum()/bT` 梯度；每组有效方向为 4，同随机种子重建的 correction 逐元素一致。
- v2 参数损失自动微分梯度与 `-Vd` 逐元素一致。两版返回的关键查询张量无梯度，全部模型参数/buffer 在查询前后逐元素一致。
- 新 checkpoint 保存/加载后，从含两列连续属性的原始合成图执行 `use_node_attributes=False` 推断，与手动去属性后的模型结果逐元素一致，最大概率误差为 0；原图附带的合成 y 不被修改或用于预测。
- `core_tasks.json` 中 15 个配置全部通过 `Config.from_dict`，与 15 个单文件配置完全一致；v1/v2/source_only 分别为 6/6/3；源清单及目标索引引用文件存在。只检查路径和配置，没有读取真实 cache 或目标标签。
- `python -m workflows.tqg.run_task --help` 和 `python -m workflows.tqg.run_all --help` 均以状态 0 退出。

## 现有测试证据（由主端执行）

主端在同一被审提交重新执行并提供：`test_sega_pipeline.py` 13/13、`test_feedback.py` 13/13、`test_v2_math.py` 16/16，共 **42 项通过**。原始输出分别为：

- [pipeline.tests.log](pipeline.tests.log)
- [v1.tests.log](v1.tests.log)
- [v2.tests.log](v2.tests.log)

本 reviewer 阅读对应测试及公式断言，未重复全量运行。现有检查涵盖两版端到端合成流水线、缓存复用及破坏检测、BN/dropout、源角色、目标 y 拒绝、矩阵特例、坐标旋转、无支持/零秩/数值 fallback。主端另报告 SEGA 参考快照 30 个文件尺寸和 SHA256 全部匹配；本 reviewer 没有改写任何参考文件。

## 限制与明确不登记为 bug 的事项

1. 未执行真实预训练/适配、真实开发集评价或全目标评价；没有读取真实目标标签，合成检查不能确认条件任务关系可迁移或任何准确率收益。
2. GPU/CUDA、完整 query=196/target=16 的资源与真实数据数值表现未验证。当前检查是 CPU 合成证据。
3. FP32 数值秩阈值可能丢弃 FP64 保留的弱方向；当前规则与实现文档明确一致，不把此精度选择本身登记为实现错误。正式 M0 仍需记录 rank/阈值和反馈健康情况。
4. 新流水线不支持中途训练 resume；未完成目录会明确拒绝复用。实现说明已记录该限制。这里确认的是完成的源缓存和模型推断恢复，不声称具备 optimizer/RNG 中断恢复。
5. 新最小实验的 PERM、QUERY-CE、开发集评价器、0/10/50快照及阶段门槛 driver 尚待实现，执行清单已明确。`core_tasks.json` 是可解析的核心任务配置，不是完整 gated runner；占位 cache 路径也已显式标注。这些不属于本轮新增 bug。
6. 直接在仓库内验证入口，未验证构建 wheel 后脱离仓库的安装部署；没有把未测试平台或打包场景描述为已支持。

独立 reviewer 审查全过程未修改算法/测试/配置，没有切换分支、提交或推送；结束时仓库工作树为空。随后由主端归档本报告、检查脚本和日志并更新导航；算法实现与实验配置保持被审提交的内容。所有归档文件的 SHA256 见 [evidence.json](evidence.json)。
