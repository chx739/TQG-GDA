# 仓库布局与维护规则

整理日期：2026-11-03。

## 文档按版本、用途分类

- `docs/v1/`：有限扰动odd/even原方法；含method、plans、implementation、results。
- `docs/v2/`：Jacobian／核查询当前方法；使用相同四类目录。`results/status.md`是已完成进度，`plans/`是设计，不互相替代。
- `docs/code/`：代码和脚本导航；`docs/usage.md`提供安装与运行命令。
- `docs/history/`：当时的原始阶段说明及来源清单；`docs/validation/`保存历史验收。这些不写成当前效果结论。

原首页的v1与v2完整训练表格已分别迁入各自的`results/overview.md`。历史`docs/experiments/latest/`明确命名为`docs/v1/results/proteins_full_query_20261026/`，避免误解为当前最新结果。

## 原始证据保持稳定

`experiments/<实验名>/`的原路径、文件内容和校验记录保持不变。`experiments/v1/README.md`与`experiments/v2/README.md`负责按版本归类，不重复大文件。

2026-10-31冻结方案留在`docs/research_plans/tqg_v2_20261031/`，原字节与SHA不变。v2计划目录链接到它；已完成状态单独追踪。历史代码快照、数据清单、训练日志和模型不批量替换路径或重新生成。

旧文档位置使用相对符号链接兼容已有脚本和本机引用，实际内容只保存一份。GitHub直接打开符号链接可能显示目标路径；请从新版README进入正式文档。旧路径与正式路径的映射见[navigation_migration.json](navigation_migration.json)。不能对兼容链接单独复制一份内容再修改。

根目录Python编排入口、`scripts/`和配置执行路径本次不迁移：它们有互相导入、代码根定位及历史绝对路径依赖。先用[代码导航](code/README.md)分类；未来如改包结构，作为独立代码重构验证，避免与研究数值变更混在一起。

## 新实验怎么加

1. 方案放`docs/v2/plans/<名称_日期>/`；执行前冻结方案、配置和代码SHA。
2. 可读结果放`docs/v2/results/<名称_日期>/`；至少有结果解读、执行报告、指标和来源入口。
3. 已完成原始产物放`experiments/<名称_日期>/`；包含运行／复用区分、标签角色、日志、SHA及模型关联，不加入真实数据或未完成运行。
4. 更新`docs/v2/results/README.md`、`docs/v2/results/status.md`及`experiments/v2/README.md`。首页只保留导航和当前重要结论，不继续累计所有实验大表格。
5. 新README链接使用正式版本路径；历史来源文本保留当时路径。修改后检查链接、归档完整性、代码测试是否必要，再提交并同步GitHub。

本次实际核查记录：[链接、迁移内容与全部原始归档校验](validation/repository_20261103/README.md)。
