import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from helpers import fixture_files, small_config, task_from, ROOT
from tqg_gda.feedback_v2 import build_gradient_feedback
from tqg_gda.models import GraphClassifier
from tqg_gda.training import TQGTrainer, load_model, model_spec, run_adaptation, run_warmup
from tqg_gda.utils import load_config, merge, validate_config


def v2_config(**changes):
    config = small_config(method='tqg_v2', adaptation=dict(optimizer='sgd'), v2={})
    config.pop('probes')
    return merge(config, changes)


class V2TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(73)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = v2_config()
        self.files = fixture_files(self.root/'inputs')
        self.task = task_from(self.files, self.config)

    def model(self):
        return GraphClassifier(model_spec(self.task, self.config), 2).double()

    def test_all_v2_configs_resolve_and_invalid_execution_rules_are_rejected(self):
        for path in (ROOT/'configs/v2').glob('*.json'):
            validate_config(load_config(path))
        for change in (dict(adaptation=dict(optimizer='adam')),
                       dict(adaptation=dict(query_batch_size=1)),
                       dict(adaptation=dict(source_pool='all_source')),
                       dict(v2=dict(calibration=dict(lambda_w=0))),
                       dict(v2=dict(calibration=dict(max_iterations=0)))):
            with self.assertRaises(ValueError):
                validate_config(merge(self.config, change))

    def test_one_joint_sgd_update_executes_the_constructed_gradient(self):
        trainer = TQGTrainer(self.model(), self.config)
        before = copy.deepcopy(trainer.model.state_dict())
        row = trainer.step(self.task)
        self.assertIsInstance(trainer.optimizer, torch.optim.SGD)
        self.assertEqual(trainer.optimizer.defaults['momentum'], 0)
        self.assertEqual(row['query']['status'], 'gradient_feedback')
        self.assertEqual(row['query']['rank_policy'], 'all_numerically_nonzero')
        self.assertLess(row['execution']['feedback_gradient_identity_error'], 1e-12)
        self.assertLess(row['execution']['sgd_step_identity_error'], 1e-12)
        self.assertGreater(row['execution']['feedback_gradient_norm'], 0)
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in trainer.model.state_dict().items()))
        json.dumps(row, allow_nan=False)

    def test_zero_feedback_matches_source_sgd_with_the_same_fit_batches(self):
        cfg = merge(self.config, dict(adaptation=dict(feedback_weight=0.)))
        first_model = self.model()
        first = TQGTrainer(first_model, cfg)
        second = TQGTrainer(copy.deepcopy(first_model), merge(cfg, dict(method='source_only')))
        for _ in range(3):
            a, b = first.step(self.task), second.step(self.task)
            self.assertEqual(a['batches']['fit'], b['batches']['fit'])
            for name, value in first.model.state_dict().items():
                torch.testing.assert_close(value, second.model.state_dict()[name], atol=0, rtol=0)

    def test_restoring_state_reproduces_the_next_update_and_label_control(self):
        cfg = merge(self.config, dict(adaptation=dict(query_label_mode='permute_within_batch', query_permutation_seed=7)))
        first = TQGTrainer(self.model(), cfg)
        first.step(self.task)
        saved = copy.deepcopy(first.state())
        expected = first.step(self.task)
        second = TQGTrainer(self.model(), cfg)
        second.restore(saved)
        actual = second.step(self.task)
        expected.pop('seconds'); actual.pop('seconds')
        self.assertEqual(expected, actual)
        for name, value in first.model.state_dict().items():
            self.assertTrue(torch.equal(value, second.model.state_dict()[name]), name)

    def test_target_labels_are_isolated_and_label_bearing_inputs_are_rejected(self):
        model = self.model()
        query = self.task.batch(self.task.query_ids, 'query')
        target = self.task.batch(self.task.target_ids[:3], 'target')
        first = build_gradient_feedback(model, query, query.y, target)
        self.task._raw.y[list(self.task.target_ids)] = -123456
        second = build_gradient_feedback(model, query, query.y, self.task.batch(self.task.target_ids[:3], 'target'))
        torch.testing.assert_close(first.gradient, second.gradient, atol=0, rtol=0)
        target.y = torch.tensor([0, 1, 0])
        with self.assertRaises(PermissionError):
            build_gradient_feedback(model, query, query.y, target)

    def test_gine_and_float32_support_the_new_feedback(self):
        model = GraphClassifier(dict(input_dim=2, hidden_dim=5, num_layers=2, edge_dim=1), 2).float()
        query = self.task.batch(self.task.query_ids, 'query', dtype=torch.float32)
        target = self.task.batch(self.task.target_ids[:3], 'target', dtype=torch.float32)
        query.edge_attr = torch.ones(query.num_edges, 1, dtype=torch.float32)
        target.edge_attr = torch.ones(target.num_edges, 1, dtype=torch.float32)
        feedback = build_gradient_feedback(model, query, query.y, target)
        self.assertEqual(feedback.statistics['status'], 'gradient_feedback')
        self.assertEqual(feedback.readout.dtype, torch.float64)
        self.assertEqual(feedback.gradient.dtype, torch.float32)
        self.assertGreater(float(feedback.gradient.norm()), 0)

    def test_warmup_adapt_and_resume_preserve_models_and_complete_history(self):
        source_task = task_from(self.files, self.config, target=False)
        with contextlib.redirect_stdout(io.StringIO()):
            run_warmup(source_task, self.config, self.root/'source')
            checkpoint = self.root/'source/best.pt'
            run_adaptation(self.task, self.config, checkpoint, self.root/'full')
            run_adaptation(self.task, self.config, checkpoint, self.root/'resumed', resume=self.root/'full/step_1.pt')
        a = torch.load(self.root/'full/last.pt', weights_only=False)
        b = torch.load(self.root/'resumed/last.pt', weights_only=False)
        for name, value in a['model'].items():
            self.assertTrue(torch.equal(value, b['model'][name]), name)
        self.assertEqual(a['selection']['step'], b['selection']['step'])
        for name in ('full', 'resumed'):
            rows = [json.loads(line) for line in (self.root/name/'history.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in rows], [1, 2, 3])
        _, selected = load_model(self.root/'full/selected.pt')
        self.assertEqual(selected['config']['method'], 'tqg_v2')
        with self.assertRaisesRegex(ValueError, 'source warmup checkpoint'):
            run_adaptation(self.task, self.config, self.root/'full/selected.pt', self.root/'bad_source')


if __name__ == '__main__':
    unittest.main()
