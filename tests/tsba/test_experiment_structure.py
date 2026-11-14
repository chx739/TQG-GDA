"""Keep the experiment layout's architectural boundaries explicit."""
import ast
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiments.common.io import digest, provenance

REPO = Path(__file__).resolve().parents[2]


class ExperimentStructureTests(unittest.TestCase):
    def test_provenance_hashes_tracked_paths_with_chinese_and_whitespace(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(['git', 'init', '--quiet', str(repo)], check=True)
            paths = ('实验/配置.json', '实验/报告\n续篇.md')
            for name in paths:
                path = repo / name
                path.parent.mkdir(exist_ok=True)
                path.write_text(name, encoding='utf-8')
            subprocess.run(['git', 'add', '--', *paths], cwd=repo, check=True)
            subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            '-c', 'core.hooksPath=/dev/null', '-c', 'commit.gpgsign=false',
                            'commit', '--quiet', '-m', 'Fixture'], cwd=repo, check=True)
            self.assertEqual(provenance(repo)['file_hashes'],
                             {name: digest(repo / name) for name in paths})

    def test_common_is_round_independent_and_has_no_cli(self):
        common = REPO / 'experiments/common'
        for path in common.rglob('*.py'):
            with self.subTest(path=path.relative_to(REPO)):
                self.assertEqual(path.parent, common, 'Do not introduce round namespaces in common')
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        self.assertFalse(module.startswith(('experiments.synthetic_data.', 'experiments.mutagenicity.', 'scripts.')))
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertFalse(alias.name.startswith(('experiments.synthetic_data.', 'experiments.mutagenicity.', 'scripts.')))
                self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name == 'main' for n in tree.body))

    def test_rounds_have_code_config_results_and_reading_entry(self):
        rounds = sorted((REPO / 'experiments').glob('*/*/report.md'))
        self.assertTrue(rounds)
        for report in rounds:
            root = report.parent
            with self.subTest(round=root.name):
                for name in ('README.md', 'src/__init__.py', 'configs/executed-plan.json', 'docs/README.md', 'results/evidence.json'):
                    self.assertTrue((root / name).is_file(), name)
                self.assertFalse(any(p.name != '__init__.py' for p in root.glob('*.py')))
                scripts = root / 'scripts'
                self.assertTrue((scripts / '__init__.py').is_file())
                self.assertTrue(any(p.name != '__init__.py' for p in scripts.glob('*.py')))

    def test_literal_repository_input_paths_exist(self):
        for root in ('experiments', 'scripts'):
            for path in (REPO / root).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        value = node.value
                        if value.startswith('experiments/') and '\n' not in value and '{' not in value:
                            self.assertTrue((REPO / value).exists(), f'{path}: {value}')
