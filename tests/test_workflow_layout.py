"""Guard executable paths and imports after moving the experiment drivers."""
import ast
import importlib
import unittest

from workflows._paths import REPO_ROOT


class WorkflowLayoutTests(unittest.TestCase):
    def test_workflows_import_without_running_an_experiment(self):
        for path in sorted((REPO_ROOT/'workflows').rglob('*.py')):
            module = '.'.join(path.relative_to(REPO_ROOT).with_suffix('').parts)
            with self.subTest(module=module):
                importlib.import_module(module)
        from workflows.v1.revision_common import CODE, DESIGN
        self.assertEqual(CODE, REPO_ROOT)
        self.assertTrue((DESIGN/'experiment_plan.yaml').is_file())

    def test_repo_relative_code_screen_paths_exist(self):
        # The AST screens run before real data experiments; moving a driver must
        # not silently skip it or point a screen at a dotted module filename.
        checked = 0
        for path in (REPO_ROOT/'workflows').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                value = node.value
                if value.startswith(('workflows/', 'tqg_gda/', 'tests/', 'scripts/')) and value.endswith('.py') and '\n' not in value:
                    with self.subTest(driver=path.name, reference=value):
                        self.assertTrue((REPO_ROOT/value).is_file(), value)
                    checked += 1
        self.assertGreater(checked, 20)


if __name__ == '__main__':
    unittest.main()
