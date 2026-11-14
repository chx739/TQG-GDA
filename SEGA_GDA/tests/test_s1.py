"""Only S1's data, encoder, conversion and checkpoint contracts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from sega_gda.data import SourceGraphs, sha256
from sega_gda.models import SourceClassifier
from sega_gda.initialization import GeometricHead, IdentityGeometry, fixed_scale_pairs
from sega_gda.train_source import run, load_source_checkpoint, load_initialized_checkpoint


def make_fixture(root):
    graphs = []
    for i in range(12):
        label = i % 2
        continuous = torch.tensor([float(label + 1), float(label + 3)])[:, None]
        categorical = torch.tensor([[1., 0.], [0., 1.]])
        graphs.append(Data(x=torch.cat((continuous, categorical), -1),
                           edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([label])))
    # A forbidden graph need not even contain a valid class label.
    graphs.append(Data(x=torch.full((2, 3), 10000.), edge_index=torch.tensor([[0, 1], [1, 0]]),
                       y=torch.tensor([999])))
    data, slices = InMemoryDataset.collate(graphs)
    cache = root / 'cache.pt'
    torch.save((data.to_dict(), slices, dict(num_node_attributes=1, num_node_labels=2,
                                           num_edge_attributes=0, num_edge_labels=0), Data), cache)
    split = root / 'split.json'
    split.write_text(json.dumps(dict(dataset='fixture', source_train_indices=list(range(8)),
                                    source_validation_indices=list(range(8, 12)),
                                    dataset_cache_sha256=sha256(cache))))
    return cache, split


class S1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_source_data_and_train_only_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, split = make_fixture(Path(tmp))
            dataset = SourceGraphs(cache, split, True)
            pp = dataset.fit_preprocessor()
            self.assertEqual(dataset.classes, [0, 1])
            self.assertEqual(pp['fit_ids'], list(range(8)))
            self.assertAlmostEqual(float(pp['mean'][0]), 2.5)
            self.assertEqual(pp['mean'][1:].tolist(), [0., 0.])
            with self.assertRaises(PermissionError):
                dataset.graph(12)
            # Changing validation/forbidden values must not fit different statistics.
            raw, slices, sizes, cls = torch.load(cache, weights_only=False)
            raw['x'][int(slices['x'][8]):] = 1.e8
            raw['y'][12] = -999
            torch.save((raw, slices, sizes, cls), cache)
            manifest = json.loads(split.read_text())
            manifest['dataset_cache_sha256'] = sha256(cache)
            split.write_text(json.dumps(manifest))
            other = SourceGraphs(cache, split, True).fit_preprocessor()
            torch.testing.assert_close(other['mean'], pp['mean'], atol=0, rtol=0)
            torch.testing.assert_close(other['scale'], pp['scale'], atol=0, rtol=0)

    def test_tampered_cache_and_split_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache, split = make_fixture(Path(tmp))
            manifest = json.loads(split.read_text())
            manifest['source_validation_indices'][0] = 0
            split.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                SourceGraphs(cache, split)
            with cache.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                SourceGraphs(cache, split)

    def test_gin_and_gine_graph_coordinates_and_gradients(self):
        for edge_dim in (0, 2):
            with self.subTest(edge_dim=edge_dim):
                graph = Data(x=torch.randn(3, 3, dtype=torch.float64),
                             edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), y=torch.tensor([1]))
                if edge_dim:
                    graph.edge_attr = torch.randn(4, edge_dim, dtype=torch.float64)
                batch = next(iter(DataLoader([graph, graph.clone()], batch_size=2)))
                model = SourceClassifier(dict(input_dim=3, hidden_dim=8, num_layers=2,
                                              edge_dim=edge_dim, readout='mean'), 2).double()
                coordinates = model.encoder(batch)
                self.assertEqual(tuple(coordinates.shape), (2, 8))
                torch.nn.functional.cross_entropy(model(batch), batch.y).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                if edge_dim:
                    self.assertGreater(float(model.encoder.layers[0].lin.weight.grad.norm()), 0)

    def test_exact_head_conversion_probabilities_margins_and_classes(self):
        linear = torch.nn.Linear(6, 3).double()
        u = torch.randn(30, 6, dtype=torch.float64)
        for scale in (0.1, 2., 20.):
            for temperature in (0.3, 1.7):
                head = GeometricHead.from_linear(linear, u.new_tensor(scale), temperature)
                a, b = linear(u), head(u)
                torch.testing.assert_close(a.softmax(-1), b.softmax(-1), atol=1e-10, rtol=1e-8)
                torch.testing.assert_close(a-a[:, :1], b-b[:, :1], atol=1e-10, rtol=1e-8)
                self.assertTrue(torch.equal(a.argmax(-1), b.argmax(-1)))
                # New classifier parameters receive no old linear optimizer state.
                self.assertNotEqual(head.q.data_ptr(), linear.weight.data_ptr())

    def test_scale_identity_and_no_nonidentity_fallback(self):
        u = torch.randn(8, 4, dtype=torch.float64)
        pairs, ids, scale = fixed_scale_pairs(u, list(range(8)), 12, 5)
        self.assertEqual(len(ids), 12)
        self.assertTrue(all(a != b for a, b in ids))
        torch.testing.assert_close(scale, (pairs[:, 0]-pairs[:, 1]).square().sum(-1).mean())
        with self.assertRaises(ValueError):
            fixed_scale_pairs(torch.zeros_like(u), list(range(8)), 12, 5)
        geometry = IdentityGeometry(4)
        torch.testing.assert_close(geometry(u), torch.eye(4, dtype=u.dtype).expand(8, 4, 4))
        geometry.B[0, 0, 0] = 0.1
        with self.assertRaises(ValueError):
            geometry.squared_distance(u, u)

    def test_training_checkpoint_resume_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, split = make_fixture(root)
            cfg = dict(experiment_kind='development_source_only', data_cache=str(cache),
                       split_manifest=str(split), use_node_attributes=True,
                       hidden_dim=8, num_layers=2, readout='mean', dtype='float64',
                       epochs=4, batch_size=4, lr=.01, seed=4, num_threads=1,
                       device='cpu', selection='source_validation_ce_min_earliest',
                       scale_pairs=8, scale_seed=5, temperature=1., identity_regions=1)
            with contextlib.redirect_stdout(io.StringIO()):
                run(cfg, root/'continuous')
                run({**cfg, 'epochs': 2}, root/'partial')
                run(cfg, root/'resumed', root/'partial/last.pt')
                run(cfg, root/'repeat')
            states = [torch.load(root/p/'last.pt', weights_only=False)
                      for p in ['continuous', 'resumed', 'repeat']]
            for other in states[1:]:
                for key, value in states[0]['model'].items():
                    torch.testing.assert_close(value, other['model'][key], atol=0, rtol=0)
                self.assertEqual(states[0]['best_epoch'], other['best_epoch'])
                self.assertEqual(states[0]['best_ce'], other['best_ce'])
                torch.testing.assert_close(states[0]['loader_rng'], other['loader_rng'], atol=0, rtol=0)
                torch.testing.assert_close(states[0]['rng']['torch'], other['rng']['torch'], atol=0, rtol=0)
            source, state = load_source_checkpoint(root/'continuous/best.pt')
            initialized, init_state = load_initialized_checkpoint(root/'continuous/initialization.pt')
            self.assertEqual(init_state['source_epoch'], state['epoch'])
            self.assertFalse(state['identity']['target_labels_accessed'])
            self.assertTrue(set(sum(init_state['scale_pair_graph_ids'], [])) <= set(range(8)))
            dataset = SourceGraphs(cache, split, True)
            graphs = dataset.materialize(dataset.val_ids, state['preprocessing'])
            single = next(iter(DataLoader([graphs[0]], batch_size=1)))
            mixed = next(iter(DataLoader([graphs[1], graphs[0]], batch_size=2)))
            initialized.eval()
            torch.testing.assert_close(initialized(single)[0], initialized(mixed)[1], atol=1e-10, rtol=1e-8)
            self.assertEqual(init_state['verification']['prediction_mismatches'], 0)


if __name__ == '__main__':
    unittest.main()
