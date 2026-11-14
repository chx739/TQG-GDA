# 新实验目录模板

新建 `experiments/<dataset>/<round_date>/`，按下列结构组织，不在其中另建 Git 仓库：

```text
README.md       研究问题、结论摘要、运行方法、代码依赖
src/            实验实现
configs/        计划与实际配置
docs/           实验方案
results/        汇总表、图表、核验材料
report.md       完整结论
```

将[方案模板](protocol.template.md)放入本轮 `docs/`，将[报告模板](report.template.md)用于 `report.md`。实际运行后再生成执行配置，不能用计划配置冒充执行记录。

启动脚本放在根目录 `scripts/<dataset>/<round_date>/`，调用本轮 `src/`。通用工具放 `experiments/common/`；沿用其他轮次的特定实验实现时，在 README 明确列出依赖，不整块移入 common。
