# 新实验目录模板

新建 `experiments/<dataset>/<编号>_<中文主题>_<YYYYMMDD>/`，按下列结构组织，不在其中另建 Git 仓库：

轮次编号使用小写，主题使用汉字和下划线，日期使用八位数字。例如 `mutagenicity/r2_分数训练预算与多轮适配_20261111`、`synthetic_data/p0_分数训练预算_20261110`。实验索引采用「编号｜研究问题」，如「P0｜训练不足能否解释 score 方向误差？」。

```text
README.md       研究问题、结论摘要、运行方法、代码依赖
src/            实验实现
scripts/        启动与分析入口
configs/        计划与实际配置
docs/           实验方案
results/        汇总表、图表、核验材料
report.md       完整结论
```

将[方案模板](protocol.template.md)放入本轮 `docs/`，将[报告模板](report.template.md)用于 `report.md`。实际运行后再生成执行配置，不能用计划配置冒充执行记录。

启动脚本放在本轮的 `scripts/`，调用本轮 `src/`。在各轮 README 记录从仓库根目录运行的 `python -m experiments.<数据集>.<实验目录>.scripts.<入口>` 命令。通用工具放 `experiments/common/`；沿用其他轮次的特定实验实现时，在 README 明确列出依赖，不整块移入 common。
