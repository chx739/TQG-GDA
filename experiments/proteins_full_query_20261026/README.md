# PROTEINS 全量源查询实验：日志与模型

本目录保存已完成实验的原始运行文件，不涉及新增训练。结果与设置见[实验报告](../../docs/experiments/latest/run_report.md)、[结果汇总](../../docs/experiments/latest/results_summary.json)和[复现范围](../../docs/experiments/latest/REPRODUCIBILITY.md)。

## 范围与大小

- 三个迁移任务：0→1、0→2、0→3；五个模型种子：3405–3409。
- 105次适配：85次本轮新训练＋20次复用源监督基线。
- 10次共享源预训练：5个fit模型＋5个全源模型。15个冻结基线引用这5个fit模型，不重复保存。
- 115份逐步训练日志，共66.3 MiB；755份模型checkpoint，共125.7 MiB；加上配置与阶段记录，总大小约195 MiB。
- 所有原始副本与运行目录逐文件校验 SHA256，模型未重新保存、日志未裁剪。未上传图数据缓存及其他轮次未参与本表的诊断模型。

## 目录怎么找

```text
logs/
  source_pretraining/{fit,all_source}/{seed}/history.jsonl
  adaptation/{0_to_1,0_to_2,0_to_3}/{seed}/{arm}/history.jsonl
checkpoints/
  source_pretraining/{fit,all_source}/{seed}/{best,last}.pt
  adaptation/{task}/{seed}/{arm}/
    step_0.pt  step_10.pt  step_50.pt  step_100.pt  step_200.pt
    last.pt  selected.pt
metadata/
  jobs/       每次训练的配置、数据角色、环境、结果与来源
  stages/     原始全查询阶段的训练清单、执行锁、完成核查与计算记录
```

| 方法代号 | 含义 |
|---|---|
| `Q51_O` / `Q51_OE` | 全部51张query，odd / odd＋even |
| `Q250_O` / `Q250_OE` | 全部250张源训练图，odd / odd＋even |
| `B_fit` | 继续199张fit源监督训练 |
| `B_full` | 用250张源图预训练并继续源监督训练 |
| `B_pseudo` | 源fit监督＋置信度≥0.9的目标硬伪标签 |
| 冻结基线 | `source_pretraining/fit/{seed}/best.pt`，无后续适配 |

odd记录带方向的有限变化；even记录正反扰动相对原始输出的弯曲变化，详见[公式对照](../../docs/响应类型与原文公式对照.md)。

## 从表格找到对应文件

- [job_index.csv](job_index.csv)：115次实际训练的日志、源起点、末步和选中模型路径，附原代码提交和复用标记。
- [evaluation_index.csv](evaluation_index.csv)：225条评价记录，包含105个末步、105个源验证选点和15个冻结参考。`archived_checkpoint`、`archived_history` 可在本目录直接定位文件；原 `checkpoint` 列保留运行时绝对路径。
- 示例：[0→1、seed3405、n51 odd＋even日志](logs/adaptation/0_to_1/3405/Q51_OE/history.jsonl)、[相应末步模型](checkpoints/adaptation/0_to_1/3405/Q51_OE/last.pt)、[训练配置](metadata/jobs/adaptation/0_to_1/3405/Q51_OE/config.json)。
- 源起点示例：[fit源模型](checkpoints/source_pretraining/fit/3405/best.pt)、[源预训练日志](logs/source_pretraining/fit/3405/history.jsonl)。

`last.pt` 是完成预算后的模型；`selected.pt` 按源验证CE选取，可能对应step0。源预训练的 `best.pt` 同样按源验证CE选取，并非必然是700步终点。五个源模型可共享给不同目标域。

## 校验与使用

从仓库根目录运行：

```bash
python scripts/verify_experiment_archive.py experiments/proteins_full_query_20261026
```

该命令只需Python标准库，会检查1,638份原始文件的大小、SHA256，以及所有评价记录与checkpoint、日志的对应关系。也可在本目录执行 `sha256sum -c SHA256SUMS`。清单覆盖原始副本，生成的索引和说明由Git跟踪。

文件保持原运行格式，内部绝对路径和历史代码哈希未改写。归档索引提供本仓库路径，但没有把所有历史编排依赖改造成独立运行器；重新训练和独立评价仍需匹配的数据缓存及运行环境。见[复现说明](../../docs/experiments/latest/REPRODUCIBILITY.md)。
