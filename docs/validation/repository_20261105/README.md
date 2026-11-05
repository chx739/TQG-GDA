# 2026-11-05 实验脚本整理验收

将原根目录的52个Python文件迁入`workflows/v1/`及`workflows/v2/<阶段>/`。完整[路径映射](../../workflow_migration.json)记录整理前提交。同步修改包导入、仓库定位、子进程模块入口、源码检查／归档路径和导航文档。

算法包、训练配置、数据索引、冻结协议和历史实验归档保持原样。本次没有启动真实数据训练，也没有使用目标标签做调参或验证。

## 检查范围

- 52个迁移模块的导入与仓库根定位。
- 40个带`argparse`的阶段CLI通过`python -m ... --help`检查；没有参数解析器的早期脚本只检查导入，避免触发真实实验。
- 数学、训练、标签边界、报告和路径检查，共170项通过（168项unittest与2个独立报告函数测试）。
- 训练配置测试原来误把诊断矩阵也传给训练配置校验器；现在显式排除8份不同schema的诊断配置，并纳入v2训练配置。
- Python wheel构建及迁移模块收录检查。
- 当前导航链接、全部归档manifest文件SHA、完整训练的评价→checkpoint关联。
- 本地汇报`assumptions_brief_20261105.md`仍未跟踪且被忽略。

具体数量与逐归档核验见[layout_checks.json](layout_checks.json)，CLI列表见[cli_checks.json](cli_checks.json)。

## 复核

```bash
# stdlib检查；不启动任何数据实验
python docs/validation/repository_20261105/verify_layout.py --output /tmp/tqg-layout-checks.json
# 已安装项目依赖的环境
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
PYTHONPATH=tests python - <<'PY'
import unittest
from test_optimization_report import (
    test_incomplete_null_does_not_silently_use_two_repetitions,
    test_snapshot_weight_and_missing_coverage_are_preserved,
)
suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in [
    test_incomplete_null_does_not_silently_use_two_repetitions,
    test_snapshot_weight_and_missing_coverage_are_preserved,
])
result = unittest.TextTestRunner().run(suite)
raise SystemExit(not result.wasSuccessful())
PY
```

独立函数测试也可用pytest运行；本次环境未安装pytest，实际使用`unittest.FunctionTestCase`包装两个函数执行。历史精确执行应使用归档代码；新主分支的模块入口不能代替旧快照的原始命令。
