# R12｜三种图读数的源内阈值筛查

2026-11-21 完成 Mutagenicity 边数和 PROTEINS 边数划分各四个源域，固定种子 3405。**两种设置均为 0/4 域在真实配对表上同时选对两个阈值，未达到各 3/4 的登记门槛。** 三个域因验证集中存在空 `(U,Y)` 组而无法构造。其余五个域中，保留真实同图配对的十张表全部选了中间阈值。使用同样类条件边际构造的乘积对照选对 9/10 张表。[登记协议](docs/source-screen.md)

## 各域和两个模拟分布

同一域内两种分布的类别比例均为 50%，四档各占 25%。阈值 `j` 用 `U>j` 预测正类。两个世界的已知最优分别为 `j=1` 和 `j=3`，Accuracy 均为 74%。中间阈值在两者均为 67.5%，交换两世界的最优阈值均为 60%。

表内阈值按已知最优为 1、3 的两个世界依次列出。支持不足的域计入原分母。

| 设置 | 源域 | 支持 | 同图配对 | 给定类别的乘积 | 档内条件分布构造核对 |
| --- | --- | --- | --- | --- | --- |
| Mutagenicity 边数 | M0 | `U=4, Y=0` 为空 | — | — | — |
| Mutagenicity 边数 | M1 | 有 | 2 / 2 | 1 / 3 | 1 / 3 |
| Mutagenicity 边数 | M2 | 有 | 2 / 2 | 1 / 3 | 1 / 3 |
| Mutagenicity 边数 | M3 | 有 | 2 / 2 | 1 / 3 | 1 / 3 |
| PROTEINS 边数 | M0 | `U=1, Y=1` 为空 | — | — | — |
| PROTEINS 边数 | M1 | `U=2, Y=1` 为空 | — | — | — |
| PROTEINS 边数 | M2 | 有 | 2 / 2 | 1 / 3 | 1 / 3 |
| PROTEINS 边数 | M3 | 有 | 2 / 2 | 1 / 0 | 1 / 3 |

真实配对表的 Accuracy 全部为 67.5%，离各自最优差 6.5 个百分点。两世界交换实际选择时差值也都为零，因为两边都选了 2。乘积对照在 Mutagenicity 的三个可构造域均同时选对，在 PROTEINS 的两个可构造域中有一个同时选对。

## 为什么没有识别出正确方向

- **真实图上三个读数的依赖足以破坏这次识别。** 在两个世界及五个可构造域的 10 张真实配对表上，有 9 张触发登记的可行性回退。Mutagenicity M1 第一个世界的矩解可行，仍选择了中间阈值。真实配对表沿 U 展开的矩阵秩均为 4，乘积对照均为 2。三项读数的两类条件边际在所有可构造表中都有区分。Mutagenicity M1、M2 的 L 和 T 类内相关在 0.63–0.79 之间。
- **PROTEINS M3 暴露了 L 的方向假设。** 在第二世界，L 在正类中的均值比负类低 3.94 个百分点。即便乘积对照严格满足条件独立，固定 L 正向的估计器仍解释成相反类别，选了 `j=0`，得到 50% Accuracy，较中间阈值低 17.5 个百分点。该表的观测重构最大误差低于 10⁻¹⁶。
- **档内条件分布构造核对恢复 40/40 个可评估档位及 10/10 个阈值。** 同一 V 提供每档 `P(L,T | U,Y)`，随后只重加权 `(U,Y)`，因此满足这项对照的假设。它核对了两个模拟分布的构造。三个空组域对应的 24 项档位记录未产生估计。

结论限于这三个已登记读数和这批源内重加权分布。两种设置均未达到进入有限目标样本实验的条件。Mutagenicity 有三个可评估域，乘积对照和真实配对的差异清楚。PROTEINS 同时受到支持不足、真实配对依赖和 L 方向不稳的影响。

## 执行与归档

执行版本在[实际配置](results/attempt-001/executed-config.json)中。源验证 B0 分数来自 R7 C，使用 R5 源头与已保存的 V 表示逐项复算。L、T 两个辅助探针各拟合八次，只使用各自 S 标签，V 的读数和切点写出后才读取 V 标签构造并评分。源读取清单只包含两个登记设置的版本、源角色、图、源标签及源 checkpoint。八组数据通过无标签身份预检。

共归档 64 个支持组、32 条表级状态，其中 20 张完成计分、12 张因支持不足保留状态。归档核验从封存读数和已保存的类别重算各经验组、乘积边际及精确阈值效用。

[支持组计数](results/attempt-001/support.csv) · [逐表阈值与效用](results/attempt-001/outcomes.csv) · [匿名表、估计参数与符号](results/attempt-001/observations-and-estimates.json) · [类内相关](results/attempt-001/within-class.csv) · [逐档构造核对](results/attempt-001/stratum-oracle.csv) · [门槛判定](results/attempt-001/gate.json)

[数据读取日志](results/attempt-001/input-access.json) · [源角色与 checkpoint](results/attempt-001/source-roles.json) · [预检日志](results/attempt-001/preflight.log) · [运行日志](results/attempt-001/run.log) · [工程回归](results/attempt-001/engineering-tests.log) · [封存](results/attempt-001/FINAL.json) · [归档核验](results/attempt-001/verification.json)
