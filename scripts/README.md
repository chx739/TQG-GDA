# 启动脚本

通用 TSBA 任务用 `python -m scripts.run_task` 启动。安装项目后也可以用 `tsba` 命令。各轮入口位于 `experiments/<数据集>/<实验目录>/scripts/`。从仓库根目录按对应 README 所列命令启动。

```sh
python -m experiments.mutagenicity.r4_全方向迁移对照_20261112.scripts.run --help
python -m experiments.synthetic_data.p0_分数训练预算_20261110.scripts.run_p0 --help
python -m scripts.tools.package --help
```

每轮的可用脚本、配置、输入约束和结论都在对应实验 README 中。不要直接执行文件路径；使用模块方式确保仓库包可正确导入。启动脚本随仓库版本管理，机器专用临时调度脚本另行保管。
