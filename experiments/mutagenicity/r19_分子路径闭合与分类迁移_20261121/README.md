# R19｜目标图的路径闭合率能否纠正分子分类？

Mutagenicity 边数划分，种子 3405，四个原始源域内各做小图到大图和大图到小图。无标签预检已完成，本域的路径闭合率在八个方向中有七个比等预算源参照更能预测另一批图的闭合情况。分类筛查比较固定源模型分别使用源和目标背景闭合率时的纠正与误改。[结构预检实测](docs/structure-results.md) · [分类协议](docs/source-screen.md)

分类筛查已完成。本域背景相对同模型的源背景提高 0.28 个百分点，相对静态 WL 低 2.65 个百分点，未达到晋级门槛。[分类报告](report.md) · [阅读入口](docs/README.md) · [封存索引](results/evidence.json)

另做一次固定补充试验。将目标背景变化直接加到原 WL，补充池 Accuracy 下降 0.53 个百分点，按协议结束当前环修正。[补充协议](docs/target-change-only.md) · [补充结果](docs/target-change-only-results.md)

## 重放

从仓库根目录执行。图和标签偏移表依赖 R9 指向的本机 R5/R10/R7 认证档案。新运行使用空目录，留存原档。冻结阶段使用 D/C 标签拟合，计分阶段只读取 E 标签。历史 WL 权重和环计数直接从本仓库封存读取。

```bash
r19=experiments/mutagenicity/r19_分子路径闭合与分类迁移_20261121
run=/tmp/r19-screen-replay
entry=experiments.mutagenicity.r19_分子路径闭合与分类迁移_20261121.scripts.source_screen
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m "$entry" \
  --config "$r19/configs/source-screen.json" --output "$run" --phase freeze
obs_sha=$(sha256sum "$run/OBS.json" | cut -d ' ' -f 1)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m "$entry" \
  --config "$r19/configs/source-screen.json" --output "$run" --phase score --obs-sha256 "$obs_sha"
```

复核原 OBS 与独立预测可单独运行，无需加载标签。独立目录同时保留了未计分预测的生成代码和摘要。

```bash
arch="$r19/results/attempt-001-source-screen"
python -m experiments.mutagenicity.r19_分子路径闭合与分类迁移_20261121.scripts.verify_independent \
  "$arch" 93438a531d71b6b0c1cd1e26bc7fee48dd24614c06de05776db99cad5095a2cb \
  "$arch/independent" 8a3d44caa222d9078a5ca834dcd683d3c896d475951898f58c7e8d16e1a26df2 \
  /tmp/r19-independent-verification.json
```

补充试验复用主试验 OBS 和 R9 静态分数。冻结后由计分入口先运行独立预测比对，确认一致才打开补充角色标签。

```bash
r19=experiments/mutagenicity/r19_分子路径闭合与分类迁移_20261121
run=/tmp/r19-target-change-replay
entry=experiments.mutagenicity.r19_分子路径闭合与分类迁移_20261121.scripts.target_change_only
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m "$entry" \
  --config "$r19/configs/target-change-only.json" --output "$run" --phase freeze
obs_sha=$(sha256sum "$run/OBS.json" | cut -d ' ' -f 1)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m "$entry" \
  --config "$r19/configs/target-change-only.json" --output "$run" --phase score --obs-sha256 "$obs_sha" \
  --blind-dir "$r19/results/attempt-002-target-change-only/independent"
```
