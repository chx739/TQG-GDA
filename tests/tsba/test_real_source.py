import tempfile
from pathlib import Path
import unittest
import torch
from tsba.config import Config
from tsba.pretrain import pretrain_encoder
from tsba.utils import set_seed
from scripts.run_task import synthetic_task
from experiments.mutagenicity.r1_图表示与边界支持_20261111.src.source import train_source
from experiments.common.runtime import seal


class RealSourceTests(unittest.TestCase):
    def test_source_training_matches_production_and_retains_optimizer(self):
        torch.set_num_threads(1)
        cfg = Config(seed=103, hidden=4, num_layers=2, emb_dim=4, source_epochs=3, source_batch=16)
        data = synthetic_task()
        set_seed(cfg.seed)
        expected, expected_info = pretrain_encoder(cfg, data)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'source'
            actual, info = train_source(cfg, data, out, lambda _: None)
            self.assertEqual(info, expected_info)
            for key, v in expected.state_dict().items():
                torch.testing.assert_close(v, actual.state_dict()[key], rtol=0, atol=0)
            files = list((out / 'epochs').glob('*.pt'))
            self.assertEqual(len(files), 3)
            self.assertTrue(torch.load(files[0], weights_only=True)['optimizer']['state'])
            first = seal(Path(tmp), 'S1', ['source'])
            self.assertEqual(len(first['files']), 5)
            with self.assertRaises(FileExistsError):
                seal(Path(tmp), 'S1', ['source'])


if __name__ == '__main__':
    unittest.main()
