# 数据域索引

原样复制自用户提供的 `SEGA_GDA_fixed_2.zip` 中的 `SEGA_GDA/idx/`，共 44 个 NumPy 文件。来源哈希、逐文件哈希及比较结果见 [provenance.json](provenance.json)。未执行压缩包中的代码。

## Mutagenicity 核验

| 域 | 图数量 | 与现有参考索引 | 与 R4 实际域 ID 及顺序 |
|---|---:|---|---|
| 0 | 1084 | 字节完全一致 | 完全一致 |
| 1 | 1084 | 字节完全一致 | 完全一致 |
| 2 | 1084 | 字节完全一致 | 完全一致 |
| 3 | 1085 | 字节完全一致 | 完全一致 |

比较的是 `idx_Mutagenicity_0.npy` 至 `idx_Mutagenicity_3.npy`；四域互斥，覆盖全部 4337 张图（0-based 图 ID）。这里是域划分，不是类别标签，也不是源域内部 80%/20% 训练/验证划分。

`node_Mutagenicity_*.npy` 是另一套索引，每个域的图集合均不同于对应的 `idx_*` 文件，不能替代现有划分。其余数据集及 `node_*` 文件按附件原样保留，没有宣称它们与现有 TSBA 实验一致。

本次仅新增索引副本；历史实验仍使用原参考文件与封存配置，未更改数据加载入口或重新运行实验。现有参考索引位于 [references/sega-gda/splits](../references/sega-gda/splits)。
