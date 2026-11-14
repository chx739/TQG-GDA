import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/check_large_files.py'


class LargeFileGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git('init', '-q')

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root)

    def check(self, *args):
        return subprocess.run(['python3', str(SCRIPT), *args], cwd=self.root, capture_output=True).returncode

    def test_reads_staged_blob_even_when_worktree_shrinks(self):
        p = self.root / 'array.npz'
        p.write_bytes(b'x' * (5 * 1024 * 1024 + 1))
        self.git('add', 'array.npz')
        p.write_bytes(b'small')
        self.assertEqual(self.check('--staged'), 1)
        self.git('add', 'array.npz')
        self.assertEqual(self.check('--staged'), 0)

    def test_exception_pins_content_and_policy_in_index(self):
        p = self.root / 'evidence.csv'
        p.write_bytes(b'x' * (5 * 1024 * 1024 + 1))
        self.git('add', 'evidence.csv')
        blob = self.git('rev-parse', ':evidence.csv').decode().strip()
        policy = self.root / '.github/large-file-exceptions.json'
        policy.parent.mkdir()
        policy.write_text(json.dumps({'exceptions': [{'path': 'evidence.csv', 'blob': blob, 'reason': 'legacy evidence'}]}))
        self.assertEqual(self.check('--all'), 1)  # unstaged exception has no effect
        self.git('add', '.github')
        self.assertEqual(self.check('--all'), 0)
        p.write_bytes(b'y' * (5 * 1024 * 1024 + 1))
        self.git('add', 'evidence.csv')
        self.assertEqual(self.check('--all'), 1)
