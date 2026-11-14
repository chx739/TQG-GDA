# 新实验目录模板

新建 `experiments/<dataset>/<topic>_<YYYYMMDD>/`，按下列结构组织，不在其中另建 Git 仓库：

`topic` 用简短的英文词组写出研究问题，使用小写字母和下划线，例如 `score_training_budget_20261110`。在实验索引里用研究问题作显示名，历史轮次编号可在 README 中注明。已有合成实验目录沿用原名。

```text
README.md       研究问题、结论摘要、运行方法、代码依赖
src/            实验实现
configs/        计划与实际配置
docs/           实验方案
results/        汇总表、图表、核验材料
report.md       完整结论
```

将[方案模板](protocol.template.md)放入本轮 `docs/`，将[报告模板](report.template.md)用于 `report.md`。实际运行后再生成执行配置，不能用计划配置冒充执行记录。

启动脚本放在根目录 `scripts/<dataset>/<topic>_<YYYYMMDD>/`，目录名与实验目录一致，调用本轮 `src/`。通用工具放 `experiments/common/`；沿用其他轮次的特定实验实现时，在 README 明确列出依赖，不整块移入 common。
