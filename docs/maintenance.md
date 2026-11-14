# 仓库维护

## 分工

- `docs/`：方法、实现、使用、维护与历史说明。
- `experiments/`：TSBA 的实验方案、计划、实际执行记录、报告与必要汇总。
- `tsba/`、`workflows/tsba/`、`tests/tsba/`：算法、流程与工程测试。
- 旧 TQG 保留在 `TQG` 分支；参考快照 `references/` 不改字节。

移动目录的提交只改变组织和引用，不混入算法调整。已完成实验结果不重算；修改结论另给依据。

## 大文件防线

单个 Git 文件上限为 **5 MiB**。这是仓库维护阈值，不是实验超参数。`.gitignore` 排除常见训练产物，但不全局禁止小型 `.npz` 测试数据。

新克隆启用提交钩子（先检查是否已有自定义 hooksPath；已有时整合检查命令，不覆盖）：

```sh
git config --get core.hooksPath
git config core.hooksPath .githooks
```

检查读取暂存区的实际 Git 对象，避免工作区与暂存区不一致漏检：

```sh
python3 tools/check_large_files.py --staged
python3 tools/check_large_files.py --all
python3 tools/check_doc_links.py
python3 -m unittest discover -s tests/maintenance -v
```

GitHub Actions 对推送和 PR 执行同样检查。要在合并时强制要求通过，还需仓库管理员配置分支保护；本次不修改分支保护。

三份既有历史 CSV 超过阈值，在 `.github/large-file-exceptions.json` 按路径与 blob 哈希固定例外。内容变大或发生任何变化，例外不再匹配。新增例外须说明保留价值；通常应将大文件移至外部档案，只提交摘要与哈希。

## 可追溯性

已完成实验以 `executed-plan.json`、`executed-configs.json`、`evidence.json` 和原档案为执行依据。历史路径保持原样，提交号按[映射](archive/history-commit-map.txt)查询。Markdown 检查只核验本地文件目标，不检查外网可访问性或标题锚点。

重新运行工程测试可以验证路径和实现；不据此授权真实数据训练、访问目标标签或重新生成研究结果。
