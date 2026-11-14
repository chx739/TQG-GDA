# 仓库维护

## 分工

- `docs/`：方法、实现、使用、维护与历史说明。
- `experiments/`：TSBA 的实验方案、计划、实际执行记录、报告与必要汇总。
- `tsba/`、`workflows/tsba/`、`tests/tsba/`：算法、流程与工程测试。
- 旧 TQG 保留在 `TQG` 分支；参考快照 `references/` 不改字节。

移动目录的提交只改变组织和引用，不混入算法调整。已完成实验结果不重算；修改结论另给依据。

## 文档链接检查

```sh
python3 tools/check_doc_links.py
```

GitHub Actions 对推送和 PR 执行本地 Markdown 文件链接检查。

2026-11-14 按用户要求撤回本次整理新增的大文件防线：不再设置 5 MiB 阈值、提交钩子、例外清单或体积 CI 检查；本次新增的产物忽略规则也已撤回。整理前已有的 `.gitignore` 规则保持原样。

## 可追溯性

已完成实验以 `executed-plan.json`、`executed-configs.json`、`evidence.json` 和原档案为执行依据。历史路径保持原样，提交号按[映射](archive/history-commit-map.txt)查询。Markdown 检查只核验本地文件目标，不检查外网可访问性或标题锚点。

重新运行工程测试可以验证路径和实现；不据此授权真实数据训练、访问目标标签或重新生成研究结果。
