import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from helpers import fixture_files, small_config, task_from, ROOT
from tqg_gda.data import GraphDomainTask
from tqg_gda.feedback import ProbeConfig, build_feedback
from tqg_gda.metrics import classification_metrics, binary_auc
from tqg_gda.models import GraphClassifier
from tqg_gda.training import TQGTrainer, run_warmup, run_adaptation, load_model, model_spec
from tqg_gda.utils import load_config, merge, seed_everything


class DataTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(17)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.files = fixture_files(self.root / 'inputs')
        self.config = small_config()
        self.task = task_from(self.files, self.config)

    def model(self):
        return GraphClassifier(model_spec(self.task, self.config), 2).double()

    def test_roles_are_disjoint_and_preprocessing_is_fit_only(self):
        roles = [set(self.task.ids(r)) for r in ('fit', 'query', 'validation', 'target')]
        for i, a in enumerate(roles):
            for b in roles[i+1:]:
                self.assertFalse(a & b)
        self.assertEqual(roles[0] | roles[1], set(self.task.train_ids))
        self.assertEqual(set(self.task.preprocessing['fit_ids']), roles[0])
        for role in ('fit', 'query'):
            self.assertEqual({self.task.labels[i] for i in self.task.ids(role)}, {0, 1})
        with self.assertRaises(PermissionError):
            self.task.batch(self.task.validation_ids, 'query')
        with self.assertRaises(PermissionError):
            self.task._source_label(self.task.target_ids[0])

    def test_target_labels_absent_and_poison_does_not_change_feedback(self):
        target = self.task.batch(self.task.target_ids[:3], 'target')
        self.assertIsNone(target.y)
        self.assertNotIn('y', self.task._inputs)
        model = self.model()
        query = self.task.batch(self.task.query_ids, 'query')
        first = build_feedback(model, query, query.y, target, ProbeConfig(count=1), torch.Generator().manual_seed(4))
        self.task._raw.y[list(self.task.target_ids)] = -123456
        target = self.task.batch(self.task.target_ids[:3], 'target')
        second = build_feedback(model, query, query.y, target, ProbeConfig(count=1), torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(first.correction, second.correction))

    def test_actual_gin_queries_and_training_update(self):
        model = self.model(); before = copy.deepcopy(model.state_dict())
        trainer = TQGTrainer(model, self.config)
        row = trainer.step(self.task)
        self.assertGreater(row['query']['valid_directions'], 0)
        self.assertLessEqual(row['query']['response_columns'], 2*self.config['probes']['count'])
        self.assertTrue(any(not torch.equal(before[n], v) for n, v in model.state_dict().items()))
        json.dumps(row, allow_nan=False)

    def test_gine_edge_attributes_support_queries(self):
        model = GraphClassifier(dict(input_dim=2, hidden_dim=5, num_layers=2, edge_dim=1), 2).double()
        query = self.task.batch(self.task.query_ids, 'query')
        target = self.task.batch(self.task.target_ids[:3], 'target')
        query.edge_attr = torch.ones(query.num_edges, 1, dtype=torch.float64)
        target.edge_attr = torch.ones(target.num_edges, 1, dtype=torch.float64)
        feedback = build_feedback(model, query, query.y, target, ProbeConfig(count=1), torch.Generator().manual_seed(2))
        self.assertTrue(torch.isfinite(feedback.reference).all())

    def test_zero_feedback_matches_source_only_including_fit_batch_sequence(self):
        tqg = merge(self.config, {'adaptation': {'feedback_weight': 0.}})
        baseline = merge(tqg, {'method': 'source_only'})
        a = self.model(); b = copy.deepcopy(a)
        first, second = TQGTrainer(a, tqg), TQGTrainer(b, baseline)
        for _ in range(3):
            ra, rb = first.step(self.task), second.step(self.task)
            self.assertEqual(ra['batches']['fit'], rb['batches']['fit'])
            for name, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)

    def test_metrics_missing_class_is_json_safe_and_auc_handles_ties(self):
        metrics = classification_metrics(torch.tensor([[2., 0.], [1., 0.]]), torch.tensor([0, 0]))
        self.assertEqual(metrics['class_recall'], [1., None])
        self.assertIsNone(metrics['auroc']); json.dumps(metrics, allow_nan=False)
        self.assertEqual(binary_auc([0, 1], [.5, .5]), .5)
        self.assertEqual(binary_auc([0, 1], [.1, .9]), 1.)

    def test_trainer_state_restores_identical_next_update(self):
        a = self.model(); trainer = TQGTrainer(a, self.config)
        trainer.step(self.task)
        state = copy.deepcopy(trainer.state())
        expected = trainer.step(self.task)
        restored = TQGTrainer(self.model(), self.config); restored.restore(state)
        actual = restored.step(self.task)
        expected.pop('seconds'); actual.pop('seconds')
        self.assertEqual(actual, expected)
        for name, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, restored.model.state_dict()[name]), name)

    def test_end_to_end_warmup_adaptation_and_checkpoint_resume(self):
        source_task = task_from(self.files, self.config, target=False)
        with contextlib.redirect_stdout(io.StringIO()):
            run_warmup(source_task, self.config, self.root/'source')
            cp = self.root/'source/best.pt'
            run_adaptation(self.task, self.config, cp, self.root/'fresh')
            run_adaptation(self.task, self.config, cp, self.root/'resumed', resume=self.root/'fresh/step_1.pt')
        a = torch.load(self.root/'fresh/last.pt', weights_only=False)
        b = torch.load(self.root/'resumed/last.pt', weights_only=False)
        for name, value in a['model'].items():
            self.assertTrue(torch.equal(value, b['model'][name]), name)
        self.assertEqual(a['selection']['step'], b['selection']['step'])
        self.assertEqual(a['selection']['ce'], b['selection']['ce'])
        model, state = load_model(self.root/'fresh/selected.pt')
        self.assertEqual(state['kind'], 'tqg_predictor')
        self.assertEqual(state['identity']['source_fit_ids'], list(self.task.fit_ids))

    def test_warmup_resume_matches_uninterrupted_epochs(self):
        task = task_from(self.files, self.config, target=False)
        short = merge(self.config, {'warmup': {'epochs': 1}})
        with contextlib.redirect_stdout(io.StringIO()):
            run_warmup(task, self.config, self.root/'full')
            run_warmup(task, short, self.root/'short')
            run_warmup(task, self.config, self.root/'extended', resume=self.root/'short/last.pt')
        a = torch.load(self.root/'full/last.pt', weights_only=False)
        b = torch.load(self.root/'extended/last.pt', weights_only=False)
        for name, value in a['model'].items():
            self.assertTrue(torch.equal(value, b['model'][name]), name)
        self.assertEqual(a['best_epoch'], b['best_epoch'])

    def test_all_configs_resolve_and_full_source_baseline_uses_all_labels(self):
        from tqg_gda.utils import validate_config
        for path in (ROOT/'configs').glob('*.json'):
            validate_config(load_config(path))
        config = merge(self.config, {'method': 'source_only', 'warmup': {'source_pool': 'all_source'},
                                     'adaptation': {'source_pool': 'all_source'}})
        task = task_from(self.files, config)
        self.assertEqual(set(task.preprocessing['fit_ids']), set(task.train_ids))
        self.assertEqual(set(task.ids('all_source')), set(task.fit_ids) | set(task.query_ids))


if __name__ == '__main__':
    unittest.main()
