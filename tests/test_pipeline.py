import importlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cela import pipeline
from cela.artifacts import digest, output_file, read_json, write_json
from scripts.run_task import config_from


class PipelineTests(unittest.TestCase):
    def test_order_risks_are_distinct(self):
        from types import SimpleNamespace
        traces = [SimpleNamespace(prior_logits=torch.zeros(2)) for _ in range(2)]
        # Different probabilities of observing the two orders under the model.
        masses = [torch.tensor([[.09, .01]], dtype=torch.float64),
                  torch.tensor([[.20, .80]], dtype=torch.float64)]
        with patch.object(pipeline, 'edge_log_probs', side_effect=[m.log() for m in masses]):
            log_p, log_gen, spread = pipeline.ensemble(traces, None, None, 0.)
        torch.testing.assert_close(log_p.exp(), torch.tensor([.55, .45], dtype=torch.float64))
        torch.testing.assert_close(log_gen.exp(), sum(masses).squeeze() / 1.10)
        torch.testing.assert_close(spread, torch.full((2,), .35, dtype=torch.float64))

    def test_atomic_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'stage.json'
            with self.assertRaises(ValueError):
                write_json(path, {'invalid': math.nan})
            self.assertEqual(list(Path(tmp).iterdir()), [])
            with self.assertRaises(FileExistsError):
                with output_file(path) as prepared:
                    prepared.write_text('loser')
                    write_json(path, {'value': 3})
            self.assertEqual(read_json(path), {'value': 3})

    def test_synthetic_separate_capabilities_and_replay(self):
        fixture = importlib.import_module('experiments.s1_流程验收_20261126.scripts.synthetic')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = fixture.make_fixture(root / 'fixture')
            cfg = config_from(config_path)
            cfg['settings'].update(epochs=1, hidden=8, source_orders=2, adapt_orders=2,
                                   inference_orders=2, bootstrap_samples=0)
            source, adaptation, pred = [root / name for name in ('source.pt', 'A.json', 'H.json')]
            label_file = config_path.parent / 'H-labels.json'
            labels = read_json(label_file)
            label_file.unlink()
            pipeline.train(cfg, source)
            original = digest(source)
            for name in ('train-labels.json', 'val-labels.json'):
                (config_path.parent / name).unlink()
            pipeline.adapt(cfg, source, adaptation)
            pipeline.predict(cfg, source, adaptation, pred)
            replay = root / 'replay.json'
            pipeline.predict(cfg, source, adaptation, replay)
            self.assertEqual(digest(pred), digest(replay))
            self.assertEqual(digest(source), original)
            altered = read_json(adaptation)
            altered['source_sha256'] = 'different-checkpoint'
            write_json(root / 'wrong.json', altered)
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                pipeline.predict(cfg, source, root / 'wrong.json', root / 'unused.json')
            manifest = read_json(cfg['data']['manifest'])
            manifest['roles']['A'] = manifest['roles']['A'][:1]
            write_json(root / 'mismatched-pool.json', manifest)
            other = dict(cfg, data=dict(cfg['data'], manifest=root / 'mismatched-pool.json'))
            with self.assertRaisesRegex(ValueError, 'adaptation pool'):
                pipeline.predict(other, source, adaptation, root / 'unused.json')
            write_json(root / 'incomplete.json', {})
            with self.assertRaisesRegex(ValueError, 'IDs'):
                pipeline.score(pred, root / 'incomplete.json', root / 'unused.json')
            bad = dict(labels)
            bad[next(iter(bad))] = True
            write_json(root / 'invalid-class.json', bad)
            with self.assertRaisesRegex(ValueError, 'integer'):
                pipeline.score(pred, root / 'invalid-class.json', root / 'unused.json')
            write_json(label_file, labels)
            pipeline.score(pred, label_file, root / 'scores.json')
            rows = read_json(pred)['records']
            report = read_json(root / 'scores.json')
            for name in report['modes']:
                expected = -sum(row['modes'][name]['log_probability'][labels[row['id']]]
                                for row in rows) / len(rows)
                self.assertAlmostEqual(expected, report['modes'][name]['nll'])
            self.assertEqual(set(report['modes']), set(read_json(adaptation)['modes']))
            self.assertIn('structural_only', report['equal_positive_vs_prior_only'])


if __name__ == '__main__':
    unittest.main()
