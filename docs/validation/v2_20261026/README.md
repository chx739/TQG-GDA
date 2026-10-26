# v2 工程验证记录

日期：2026-10-26。环境：CPU，Python 3.12.4、PyTorch 2.5.1、PyG 2.6.1。未验证 CUDA 执行。

| 检查 | 结果 | 证据 |
|---|---|---|
| 全仓单元与集成测试 | 116 项通过，包含 23 项新增 v2 测试 | [unit_tests.log](unit_tests.log) |
| 合成图命令行全流程 | 主版本、KRR、表示核、源 fit/full SGD、伪标签与标签打乱均接通；断点和目标标签污染检查通过 | [synthetic_smoke.json](synthetic_smoke.json) |
| PROTEINS 0→1 | 既有 fit 起点，全 51 张 query，16 张目标图，v2/源 SGD 各两步 | [real_data_smoke.json](real_data_smoke.json) |
| Mutagenicity 0→1 | 既有 fit 起点，全 196 张 query，16 张目标图，v2/源 SGD 各两步 | [real_data_smoke.json](real_data_smoke.json) |

真实数据检查中，四次 v2 校准均在 2–4 次内部迭代达到默认绝对/相对 gap 容差；有效秩均为 16。参数反馈梯度与构造梯度的差为 0，实际 SGD 步长等式误差最大约 6.85e-16。末步 v2 与配对源 SGD 的参数差范数分别约 1.36e-4 和 2.62e-4。

真实数据检查复用种子 3405 的源预训练模型，没有重新预训练，也没有读取或评价目标标签。差异范数只用于确认反馈确实影响了参数，不能说明目标性能提高。全部检查均非论文实验结果。

完整检查输出在本机 `/home/hxchen/runs/2026-10-26-tqg-v2-implementation/`；训练日志、模型及各次执行环境文件保留在该目录，没有混入已有实验归档。这里保存精简证据。文件来源与校验值见 [validation.json](validation.json)。

实现映射及运行方法见 [TQG_V2_IMPLEMENTATION.md](../../TQG_V2_IMPLEMENTATION.md)。
