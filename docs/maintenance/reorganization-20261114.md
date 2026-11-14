# 2026-11-14 仓库整理核验

基于历史清理后的 TSBA 提交 `d07dff02d0c538d62711eb383bbe7bd8a7a99320`。

- 将 304 份实验文件从 `docs/tsba/experiments/` 迁移到 `experiments/`，每轮方案与报告归拢。
- 原有 279 份非 Markdown 实验文件 SHA256 全部保持一致；Markdown 只调整导航、路径及组织说明。
- `tsba/` 与 `references/` 无内容变化；工作流与原有测试只调整实验路径引用。
- 12 轮历史配置从现存运行档案提取，核对对应成员哈希；未重新训练或读取真实目标标签。
- R4 主运行与 A0 统一为同一证据包，保留内部执行来源与事后登记信息。
- 138 项 TSBA 工程测试通过；2 项大文件检查测试通过。
- 本地 Markdown 文件链接检查通过；不覆盖外部 URL 或标题锚点。
- 所有工作流和测试中的字面量实验路径存在；全部暂存文件通过 5 MiB 限制及既有例外校验。

工程测试命令：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests/tsba -v
python3 tools/check_doc_links.py
```

后续变更：2026-11-14 按用户要求撤回大文件防线。上面的体积检查和 2 项维护测试仅记录整理时的验证结果；相关脚本、测试、提交钩子、例外清单和 CI 步骤已删除，本地 hooksPath 设置及此次新增忽略规则已撤回。链接检查继续保留，见[维护说明](../maintenance.md)。
