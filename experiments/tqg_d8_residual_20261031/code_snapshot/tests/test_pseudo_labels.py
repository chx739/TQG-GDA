"""Protocol tests for the confidence-filtered, hard pseudo-label control."""
import contextlib
import copy
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from helpers import fixture_files, small_config, task_from
from tqg_gda.models import GraphClassifier
from tqg_gda.training import (TQGTrainer, model_spec, pseudo_label_loss,
                              run_warmup, run_adaptation)
from tqg_gda.utils import merge, validate_config


class PseudoLabelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def config(self, threshold=.9):
        return merge(small_config(), {'adaptation': {
            'feedback_predictor': 'pseudo_labels', 'confidence_threshold': threshold}})

    def test_retained_mean_hard_targets_and_stopped_selection(self):
        logits = torch.tensor([[4., 0.], [0., 3.], [.1, 0.]], dtype=torch.float64, requires_grad=True)
        loss, stats = pseudo_label_loss(logits, .9)
        expected = F.cross_entropy(logits[:2], torch.tensor([0, 1]))
        torch.testing.assert_close(loss, expected)
        actual_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(stats['retained_count'], 2)
        self.assertEqual(stats['retained_fraction'], 2/3)
        self.assertTrue(torch.equal(actual_gradient[2], torch.zeros_like(actual_gradient[2])))

    def test_no_retained_graphs_gives_exact_zero_value_and_gradient(self):
        logits = torch.zeros(3, 2, dtype=torch.float64, requires_grad=True)
        loss, stats = pseudo_label_loss(logits, .9)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(stats['retained_count'], 0)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits)))
        # The threshold is inclusive, including an exact confidence boundary.
        boundary, stats = pseudo_label_loss(logits, .5)
        self.assertEqual(stats['retained_count'], 3)
        torch.testing.assert_close(boundary, logits.new_tensor(2.).log())

    def test_target_poison_ignored_source_fit_batches_paired_and_resume_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(0.)  # Ensure this test exercises nonzero pseudo supervision.
            files = fixture_files(tmp)
            task = task_from(files, config)
            model = GraphClassifier(model_spec(task, config), 2).double()
            trainer = TQGTrainer(model, config)
            baseline = TQGTrainer(copy.deepcopy(model), small_config())
            a, b = trainer.step(task), baseline.step(task)
            self.assertEqual(a['batches']['fit'], b['batches']['fit'])
            self.assertEqual(a['batches']['target'], b['batches']['target'])
            self.assertEqual(a['batches']['query'], [])
            self.assertGreater(a['feedback_loss'], 0.)
            saved = copy.deepcopy(trainer.state())
            expected = trainer.step(task)
            task._raw.y[list(task.target_ids)] = -987654
            restored = TQGTrainer(GraphClassifier(model_spec(task, config), 2).double(), config)
            restored.restore(saved)
            actual = restored.step(task)
            expected.pop('seconds'); actual.pop('seconds')
            self.assertEqual(expected, actual)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[name]))

    def test_core_adaptation_never_calls_target_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config = self.config()
            files = fixture_files(root/'inputs')
            source = task_from(files, config, target=False)
            target = task_from(files, config)
            run_warmup(source, config, root/'warmup')
            # Standalone evaluation must remain separate from adaptation.
            validations = [{'ce': .1}] + [{'ce': .2}] * config['adaptation']['steps']
            with patch('scripts.evaluate.evaluate_target', side_effect=AssertionError('target labels accessed')), \
                 patch('tqg_gda.training.evaluate_source', side_effect=validations):
                result = run_adaptation(target, config, root/'warmup/best.pt', root/'adaptation')
            self.assertEqual(result['selected_step'], 0)
            initial = torch.load(root/'warmup/best.pt', weights_only=False)['model']
            selected = torch.load(root/'adaptation/selected.pt', weights_only=False)['model']
            for name, value in initial.items():
                self.assertTrue(torch.equal(value, selected[name]))
            self.assertFalse(result['target_evaluated'])
            self.assertFalse(result['target_labels_used'])

    def test_threshold_validation(self):
        validate_config(self.config())
        for threshold in (-.1, 1.1, float('nan'), True, '0.9'):
            with self.assertRaises(ValueError):
                validate_config(self.config(threshold))
        with self.assertRaises(ValueError):
            validate_config(merge(self.config(), {'method': 'source_only'}))
        with self.assertRaises(ValueError):
            validate_config(merge(self.config(), {'adaptation': {
                'query_label_mode': 'permute_within_batch', 'query_permutation_seed': 1}}))


if __name__ == '__main__':
    unittest.main()
