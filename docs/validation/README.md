# 历史验证与工程审计

这些是相应日期的检查快照，不能把历史“工程通过”当作当前研究有效性。

当前新实现：[2026-11-06 SEGA式 TQG 工程验收](sega_style_tqg_20261106.md)；[2026-11-07 干净上下文独立代码审查](tqg_clean_review_20261107/README.md)。后者未发现新增 P1/P2/P3 缺陷，包含 42 项现有测试与 GIN/GINE、FP32/FP64 补充检查；未运行真实数据实验。

| 分类 | 记录 |
|---|---|
| v1早期机制与数据泛化 | [E1](e1/) · [E2](e2/) · [E3](e3/) · [E4](e4/) |
| v1 Mutagenicity实验核查 | [mutagenicity_20261026](mutagenicity_20261026/) |
| v2实现验收 | [v2_20261026](v2_20261026/README.md) |
| v2对照设计稿审查 | [v2_review_20261028](v2_review_20261028/README.md) |
| 仓库整理与归档上传历史 | [repository_20261026](repository_20261026/) · [archive_upload_20261026](archive_upload_20261026/) |

P0以后工程与权限记录随[各轮原始实验](../../experiments/v2/README.md)归档。当前进度看[v2执行状态](../v2/results/status.md)。

文档整理核查：[2026-11-03路径、正文与归档完整性报告](repository_20261103/README.md)。

实验脚本整理核查：[2026-11-05模块迁移、入口与归档检查](repository_20261105/README.md)。
