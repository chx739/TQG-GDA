"""Post-seal publication guards and independent metric audits."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from experiments.mutagenicity.r5_原始方法与请求价值_20261114.src import archive

EXPERIMENT = Path(__file__).resolve().parents[2] / 'experiments/mutagenicity/r5_原始方法与请求价值_20261114'

class ArchiveTest(unittest.TestCase):
    def test_unfinished_settings_cannot_publish_or_read_metrics(self):
        with TemporaryDirectory() as tmp, patch.object(archive, 'audit_evaluation') as audit:
            with self.assertRaises(PermissionError):
                archive.archive(Path(tmp),Path(tmp)/'output')
            audit.assert_not_called()
            self.assertFalse((Path(tmp)/'output').exists())

    def test_summary_mismatch_is_detected(self):
        plan=json.loads((EXPERIMENT/'results/execution/mutagenicity_edge/plan.json').read_text())
        source=EXPERIMENT/'results/mutagenicity_edge'
        values,seeds,audit=archive.audit_evaluation(source,plan)
        self.assertEqual((len(values),len(seeds),audit['final_endpoints']), (24,30,180))
        with TemporaryDirectory() as tmp:
            dest=Path(tmp)
            for name in ('endpoints.csv','side_paired.csv'):
                (dest/name).write_bytes((source/name).read_bytes())
            summary=json.loads((source/'summary.json').read_text())
            next(r for r in summary if r['method_or_contrast']=='A0−B1' and r['scope']=='Avg')['accuracy']['mean']+=.01
            (dest/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                archive.audit_evaluation(dest,plan)
