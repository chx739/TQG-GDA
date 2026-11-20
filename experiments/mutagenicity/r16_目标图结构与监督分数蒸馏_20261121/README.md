# R16｜独立目标图的结构能否帮助修正已有分类分数？

R10 发现源选模 WL 和源图网络在三个设置有 17%–21% 的留出图分歧。R15 的来源内新词规则只改变一张图的分类。R16 从源监督融合模型出发，尝试用独立目标图的局部结构预测 GNN 分类分数的一部分，再检验该预测能否帮助整图分类。[先导协议](docs/pilot.md) · [R10 分歧归因](../r10_原生图适配基线与增量_20261121/docs/classifier-choice.md) · [R15 结果](../r15_新局部结构与图类别_20261121/report.md)

## 运行

从仓库根目录使用[固定配置](configs/pilot.json)。预检验证 R10 模型来源和 R14 的等预算池身份。正式流程先冻结源融合及 V/A 的三套核岭系数，随后逐图产生 H 预测，全部预测封存后计分。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r16_目标图结构与监督分数蒸馏_20261121.scripts.run --config experiments/mutagenicity/r16_目标图结构与监督分数蒸馏_20261121/configs/pilot.json --preflight
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m experiments.mutagenicity.r16_目标图结构与监督分数蒸馏_20261121.scripts.run --config experiments/mutagenicity/r16_目标图结构与监督分数蒸馏_20261121/configs/pilot.json --output /tmp/r16-pilot
```

达到预登记投入门槛时，使用 `--extend-from <先导目录>` 和独立输出目录完成备用种子。结果依赖原 R10 attempt-003-structural 档案和 R7 C 封存。
