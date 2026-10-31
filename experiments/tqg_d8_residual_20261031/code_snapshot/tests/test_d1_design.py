"""D1 design contracts on synthetic inputs; no dataset diagnostics or training."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from diagnostics_sample_size import (build_block, configurations,
                                     evaluate_configuration, nested_stratified_indices,
                                     prefix_columns, summarize, valid_existing)
from tqg_gda.utils import fingerprint, sha256


class ToyTask:
    def __init__(self):
        self.query_ids = tuple(range(51))
        self.fit_ids = tuple(range(60, 80))
        self.labels = {i: int(i % 5 == 0) for i in self.query_ids + self.fit_ids}
        self.calls = []

    def batch(self, ids, role):
        self.calls.append((tuple(ids), role))
        if role == 'query' and not set(ids) <= set(self.query_ids):
            raise PermissionError('Synthetic source role violation')
        if role == 'target' and not all(100 <= i < 164 for i in ids):
            raise PermissionError('Synthetic target role violation')
        x = torch.tensor([[.2 + i / 120, .6 + (i % 7) / 10, .1 + (i % 11) / 20]
                          for i in ids], dtype=torch.float64)
        labels = torch.tensor([self.labels[i] for i in ids]) if role == 'query' else None
        return SimpleNamespace(x=x, y=labels)


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 4), nn.Tanh())
        self.head = nn.Linear(4, 2)

    def forward(self, batch):
        z = self.head(self.encoder(batch.x))
        return z - z.mean(-1, keepdim=True)


class D1DesignTests(unittest.TestCase):
    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(142)
        self.task = ToyTask()
        self.model = ToyClassifier().eval()
        self.ids = dict(bank=tuple(range(35)), audit=tuple(range(35, 51)),
                        query=tuple(range(100, 116)), oracle_extra=tuple(range(116, 132)),
                        held=tuple(range(132, 164)))

    def tearDown(self):
        torch.set_default_dtype(self.previous_dtype)

    def block(self, task=None):
        return build_block(self.model, task or self.task, self.ids, seed=3405, domain=2,
                           block=0, checkpoint='/synthetic/checkpoint.pt', checkpoint_sha='synthetic')

    def test_complete_block_reuses_scales_has_all_nested_comparisons_and_does_not_mutate_model(self):
        before = copy.deepcopy(self.model.state_dict())
        block = self.block()
        self.assertEqual(block['configuration_counts'], dict(fixed=54, source_CV=18, stratified=2))
        self.assertFalse(block['target_labels_read'])
        self.assertFalse(block['model_updated'])
        for key, tensor in before.items():
            self.assertTrue(torch.equal(tensor, self.model.state_dict()[key]))
        for direction, dictionary in block['dictionaries'].items():
            self.assertEqual(dictionary['source_scale_ids'], self.ids['bank'][:32])
            self.assertEqual(dictionary['phi_bank'].shape, (35, 2, 8))
            self.assertEqual(dictionary['phi_held'].shape, (32, 2, 8))
            # Every normalized direction has unit shared RMS in exactly the
            # common source32+target16 scale inputs, independent of n.
            variance = (.5 * dictionary['phi_bank'][:32].square().sum(1).mean(0)
                        + .5 * dictionary['phi_query'].square().sum(1).mean(0))
            torch.testing.assert_close(variance, torch.ones(8), atol=1e-12, rtol=1e-12)
            for n in (8, 16, 32):
                for columns in (2, 4, 8):
                    rows = [row for row in block['predictions']
                            if row['direction'] == direction and row['family'] == 'fixed'
                            and row['n'] == n and row['requested_columns'] == columns]
                    self.assertEqual(len(rows), 3)
                    for row in rows:
                        self.assertEqual(row['source_indices'], tuple(range(n)))
                        self.assertEqual(row['column_indices'], tuple(range(columns)))
                        torch.testing.assert_close(row['d_query'],
                                                   dictionary['phi_query'][:, :, :columns] @ row['coefficients'])
                        self.assertFalse(row['coefficients'].requires_grad)
                        self.assertFalse(row['d_query'].requires_grad)
        self.assertFalse(any(role == 'target' and set(ids) & set(self.ids['oracle_extra'])
                             for ids, role in self.task.calls))

    def test_outer_audit_label_changes_cannot_change_coefficients_CV_or_target_predictions(self):
        original = self.block()
        changed_task = ToyTask()
        for graph_id in self.ids['audit']:
            changed_task.labels[graph_id] = 1 - changed_task.labels[graph_id]
        changed = self.block(changed_task)
        for left, right in zip(original['predictions'], changed['predictions']):
            self.assertEqual(left['key'], right['key'])
            self.assertEqual(left['source_indices'], right['source_indices'])
            self.assertEqual(left['ridge'], right['ridge'])
            self.assertEqual(left['cv_scores'], right['cv_scores'])
            torch.testing.assert_close(left['coefficients'], right['coefficients'], atol=0, rtol=0)
            torch.testing.assert_close(left['d_query'], right['d_query'], atol=0, rtol=0)
        self.assertTrue(any(left['metrics']['source_audit']['residual_mse']
                            != right['metrics']['source_audit']['residual_mse']
                            for left, right in zip(original['predictions'], changed['predictions'])))

    def test_stratified_subsets_are_nested_unique_and_report_insufficient_minority(self):
        labels = torch.tensor([0] * 20 + [1] * 6 + [0] * 9)
        subsets = nested_stratified_indices(labels, minority_class=1)
        self.assertEqual(subsets[16]['actual_minority'], 4)
        self.assertEqual(subsets[32]['actual_minority'], 6)
        self.assertFalse(subsets[16]['insufficient_minority'])
        self.assertTrue(subsets[32]['insufficient_minority'])
        self.assertTrue(set(subsets[16]['indices']) <= set(subsets[32]['indices']))
        self.assertEqual(len(set(subsets[32]['indices'])), 32)
        absent = nested_stratified_indices(torch.zeros(35, dtype=torch.long), minority_class=1)
        self.assertEqual(absent[32]['actual_minority'], 0)
        self.assertTrue(absent[16]['insufficient_minority'])
        # The smaller bank is never independently resampled from the larger.
        plenty = nested_stratified_indices(torch.tensor([0] * 25 + [1] * 10), minority_class=1)
        self.assertEqual(plenty[32]['actual_minority'], 8)
        self.assertTrue(set(plenty[16]['indices']) <= set(plenty[32]['indices']))

    def test_discarded_direction_does_not_shift_candidate_prefix_budget(self):
        feedback = SimpleNamespace(statistics=dict(candidate_directions=8,
                                                   discarded=[dict(probe=1), dict(probe=5)]),
                                   source_features=torch.empty(32, 2, 6))
        self.assertEqual(prefix_columns(feedback, 2), (0,))
        self.assertEqual(prefix_columns(feedback, 4), (0, 1, 2))
        self.assertEqual(prefix_columns(feedback, 8), (0, 1, 2, 3, 4, 5))

    def test_summary_averages_blocks_before_independent_model_seed_SD(self):
        rows = []
        for seed, values in ((1, (0., 0., 0., 0.)), (2, (10.,))):
            for gain in values:
                scalar = dict(residual_mse=1., zero_prediction_mse=1.+gain,
                              mse_improvement=gain, prediction_rms=1.)
                metrics = dict(source_fitted=scalar, source_audit=scalar, coefficient_norm=1.,
                               minority_source_count=2, stable_rank=1., effective_degrees_of_freedom=1.,
                               ridge_condition_number=2., target_query_prediction_rms=1., target_heldout_prediction_rms=1.)
                rows.append(dict(identity=dict(task='0_to_2', seed=seed),
                                 predictions=[dict(key='fixed', actual_columns=4, ridge=.01, metrics=metrics)]))
        summary = summarize(rows)[0]
        self.assertAlmostEqual(summary['metrics']['source_audit_mse_improvement']['mean'], 5.)
        self.assertAlmostEqual(summary['metrics']['source_audit_mse_improvement']['seed_std'], 50**.5)
        self.assertEqual(summary['metrics']['source_audit_mse_improvement']['model_seeds'], 2)

    def test_resume_rejects_changed_SHA_identity_and_orphaned_artifact(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'block_00.pt'
            identity = dict(contract='same', ids=[1, 2])
            self.assertIsNone(valid_existing(path, identity))
            path.write_bytes(b'sealed evidence')
            with self.assertRaises(FileExistsError):
                valid_existing(path, identity)
            metadata = dict(identity=identity, identity_sha256=fingerprint(identity),
                            artifact=dict(path=str(path), sha256=sha256(path)))
            path.with_suffix('.json').write_text(json.dumps(metadata))
            self.assertEqual(valid_existing(path, identity), metadata)
            with self.assertRaises(ValueError):
                valid_existing(path, dict(contract='different', ids=[1, 2]))
            path.write_bytes(b'changed evidence')
            with self.assertRaises(ValueError):
                valid_existing(path, identity)


if __name__ == '__main__':
    unittest.main()
