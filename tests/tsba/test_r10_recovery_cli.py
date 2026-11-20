"""Resumption must validate all completed jobs, not only a selected seed."""
from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import patch

from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.scripts import validate


class RecoveryCLITests(unittest.TestCase):
    def test_seed_filter_is_rejected_for_resume(self):
        with patch('sys.argv',['validate','--root','/tmp/unused','--resume','--seed','3405']), \
             patch.object(validate.validation,'validate_jobs') as jobs, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                validate.main()
            self.assertEqual(result.exception.code,2)
            jobs.assert_not_called()
