# 启动脚本

从仓库根目录用 `python -m scripts.<数据集>.<轮次>.<入口>` 调用。这里负责启动与参数转交，实验实现位于 `experiments/<数据集>/<轮次>/src/`。

```sh
python -m scripts.mutagenicity.r4_20261112.run --help
python -m scripts.synthetic.p0_20261110.run_p0 --help
python -m scripts.tools.package --help
```

每轮的可用脚本、配置、输入约束和结论都在对应实验 README 中。不要直接执行文件路径；使用模块方式确保仓库包可正确导入。启动脚本随仓库版本管理，机器专用临时调度脚本另行保管。
