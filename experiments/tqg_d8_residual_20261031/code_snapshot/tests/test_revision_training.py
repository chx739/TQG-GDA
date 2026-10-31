"""Synthetic graph D0 contracts for paired revision training and recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from helpers import fixture_files, small_config, task_from
from experiment_revision import job_keys, phase_arms, require_phase_gate, run_phase
from revision_common import DESIGN
from training_revision import RevisionTrainer, paired_ids, run_job
from tqg_gda.models import GraphClassifier
from tqg_gda.training import model_spec, load_model
from tqg_gda.utils import cpu_state


class NullWriter:
    def __init__(self, *args):
        self.rows = []
    def add_scalar(self, *args):
        self.rows.append(args)
    def flush(self):
        pass
    def close(self):
        pass


class FixtureContext:
    def __init__(self, root):
        self.root = Path(root)
        self.plan = yaml.safe_load((DESIGN / 'experiment_plan.yaml').read_text())
        self.plan['adaptation'].update(steps=4, source_validation_interval=2,
                                      complete_state_checkpoints=[0, 2, 4])
        self.arms = json.loads((DESIGN / 'resolved_arms_preview.json').read_text())['arms']
        for arm in self.arms.values():
            arm.update(n=4, K=2)
        self.config = small_config(split=dict(fit_fraction=.5))
        self.graph_task = task_from(fixture_files(self.root / 'fixture'), self.config)
        torch.manual_seed(144)
        self.model = GraphClassifier(model_spec(self.graph_task, self.config), len(self.graph_task.classes)).double().eval()
        self.source_state = dict(schema_version=1, kind='tqg_predictor', model=cpu_state(self.model),
            encoder_spec=self.model.encoder.spec, class_labels=self.graph_task.classes,
            config=self.config, identity=self.graph_task.identity)
        self.source_path = self.root / 'source.pt'
        torch.save(self.source_state, self.source_path)
        self.commit = 'synthetic-test-commit'

    def task(self, domain, all_source=False):
        return self.graph_task

    def source(self, domain, seed, all_source=False):
        return copy.deepcopy(self.model), copy.deepcopy(self.source_state), self.source_path

    def arm(self, name):
        return dict(copy.deepcopy(self.arms[name]), arm=name)


class RevisionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.context = FixtureContext(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def trainer(self, arm):
        model = copy.deepcopy(self.context.model)
        return RevisionTrainer(model, copy.deepcopy(model), self.context.graph_task,
                               self.context.arm(arm), self.context.plan, 3405, 2)

    def test_fit_query_target_pairing_and_query_prefix_are_arm_independent(self):
        short, long = self.trainer('T0'), self.trainer('T1')
        long.arm['n'] = 6
        row_short, row_long = short.step(), long.step()
        self.assertEqual(row_short['paired_schedule'], row_long['paired_schedule'])
        self.assertEqual(row_short['batches']['fit'], row_long['batches']['fit'])
        self.assertEqual(row_short['batches']['target'], row_long['batches']['target'])
        self.assertEqual(row_short['batches']['query'], row_long['batches']['query'][:4])
        self.assertEqual(row_short['rng_state_ids']['data'], row_long['rng_state_ids']['data'])

    def test_source_only_never_builds_feedback_and_matches_independent_adam(self):
        trainer = self.trainer('B_fit')
        expected_model = copy.deepcopy(trainer.model)
        expected_optimizer = torch.optim.Adam(expected_model.parameters(), lr=.0003,
            betas=(.9, .999), eps=1e-8, weight_decay=0, amsgrad=False)
        ids, _ = paired_ids(self.context.graph_task, 3405, 2, 1)
        batch = self.context.graph_task.batch(ids['fit'], 'fit')
        expected_loss = torch.nn.functional.cross_entropy(expected_model(batch), batch.y)
        expected_loss.backward()
        expected_optimizer.step()
        with patch('training_revision.ResponseDictionary.build', side_effect=AssertionError('baseline built feedback')):
            with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
                row = trainer.step()
                self.assertEqual(step.call_count, 1)
        for parameter, expected in zip(trainer.model.parameters(), expected_model.parameters()):
            torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        self.assertEqual(row['feedback_loss'], 0.)
        self.assertEqual(row['feedback_gradient_norm']['full'], 0.)
        self.assertEqual(row['batches']['query'], [])
        self.assertEqual(row['batches']['target'], [])

    def test_all_predeclared_training_handlers_take_one_finite_step(self):
        for arm_name, config in self.context.arms.items():
            if not config.get('training'):
                continue
            with self.subTest(arm=arm_name):
                trainer = self.trainer(arm_name)
                self.assertEqual(len(trainer.optimizer.state), 0)
                with patch.object(trainer.optimizer, 'step', wraps=trainer.optimizer.step) as step:
                    row = trainer.step()
                    self.assertEqual(step.call_count, 1)
                self.assertEqual(row['step'], 1)
                self.assertGreater(row['actual_step_norm']['full'], 0.)
                json.dumps(row, allow_nan=False)
                if arm_name == 'T13':
                    self.assertIsNone(row['query']['ridge'])
                if arm_name in ('T12', 'T16'):
                    self.assertIsNone(row['query']['cv_scores'])

    def test_profile_resume_equals_uninterrupted_training_including_adam_state(self):
        full_path, resumed_path = self.root / 'full', self.root / 'resumed'
        with patch('training_revision.hardware', return_value={'synthetic': True}):
            full = run_job(self.context, 2, 3405, 'T5', output=full_path, writer_factory=NullWriter)
            partial = run_job(self.context, 2, 3405, 'T5', output=resumed_path,
                              stop_after=2, writer_factory=NullWriter)
            self.assertEqual(partial['status'], 'profile_paused')
            state = torch.load(resumed_path / 'last.pt', weights_only=False)
            self.assertEqual(state['step'], 2)
            self.assertTrue(state['optimizer']['state'])
            resumed = run_job(self.context, 2, 3405, 'T5', output=resumed_path,
                              resume=True, writer_factory=NullWriter)
        self.assertEqual(full['status'], resumed['status'])
        left = torch.load(full_path / 'last.pt', weights_only=False)
        right = torch.load(resumed_path / 'last.pt', weights_only=False)
        for name in left['model']:
            torch.testing.assert_close(left['model'][name], right['model'][name], atol=0, rtol=0)
        for index, values in left['optimizer']['state'].items():
            for key, value in values.items():
                torch.testing.assert_close(value, right['optimizer']['state'][index][key], atol=0, rtol=0)
        rows_left = [json.loads(line) for line in (full_path / 'history.jsonl').read_text().splitlines()]
        rows_right = [json.loads(line) for line in (resumed_path / 'history.jsonl').read_text().splitlines()]
        for left_row, right_row in zip(rows_left, rows_right):
            left_row.pop('seconds')
            right_row.pop('seconds')
            self.assertEqual(left_row, right_row)
        self.assertEqual([row['step'] for row in rows_right], [1, 2, 3, 4])
        for step in (0, 2, 4):
            self.assertTrue((resumed_path / f'step_{step}.pt').is_file())
        loaded, saved = load_model(resumed_path / 'last.pt')
        self.assertEqual(saved['step'], 4)
        self.assertIn('revision_config', saved)
        self.assertFalse(saved['config']['target_labels_used'])
        load_model(resumed_path / 'selected.pt')

    def test_step_zero_is_in_source_checkpoint_selection(self):
        with patch('training_revision.hardware', return_value={'synthetic': True}), patch(
                'training_revision.evaluate_source', side_effect=[dict(ce=.1), dict(ce=.2), dict(ce=.3)]):
            result = run_job(self.context, 2, 3405, 'B_fit', output=self.root / 'selection', writer_factory=NullWriter)
        self.assertEqual(result['selected_step'], 0)
        selected = torch.load(self.root / 'selection' / 'selected.pt', weights_only=False)
        self.assertTrue(selected['evaluation_only'])
        self.assertFalse(selected['optimizer_state_matches_model'])
        for name, value in self.context.model.state_dict().items():
            self.assertTrue(torch.equal(selected['model'][name], value))

    def test_target_labels_are_rejected_even_if_task_accidentally_exposes_them(self):
        trainer = self.trainer('T5')
        original = trainer.task.batch
        def unsafe_batch(ids, role, *args):
            value = original(ids, role, *args)
            if role == 'target':
                value.y = torch.zeros(value.num_graphs, dtype=torch.long)
            return value
        with patch.object(trainer.task, 'batch', side_effect=unsafe_batch):
            with self.assertRaises(PermissionError):
                trainer.step()

    def test_complete_phase_lists_and_b_gate(self):
        self.assertEqual(len(phase_arms(self.context, 'A')), 8)
        self.assertEqual(len(phase_arms(self.context, 'B')), 12)
        self.assertEqual(len(set(job_keys(self.context, 'A'))), 80)
        self.assertEqual(len(set(job_keys(self.context, 'B'))), 120)
        self.assertEqual(job_keys(self.context, 'A')[0], (2, 3405, 'T5'))
        self.assertEqual(job_keys(self.context, 'B')[0], (2, 3405, 'T6'))
        require_phase_gate('A', self.root)
        with self.assertRaises(PermissionError):
            require_phase_gate('B', self.root)
        gate_path = self.root / 'diagnostics' / 'D2' / 'decision.json'
        gate_path.parent.mkdir(parents=True)
        gate_path.write_text(json.dumps(dict(allow_B=False)))
        with self.assertRaises(PermissionError):
            require_phase_gate('B', self.root)
        gate_path.write_text(json.dumps(dict(allow_B=True)))
        require_phase_gate('B', self.root)

    def test_completed_job_hash_verification_and_recovery_identity(self):
        output = self.root / 'verified'
        with patch('training_revision.hardware', return_value={'synthetic': True}):
            run_job(self.context, 2, 3405, 'B_fit', output=output, writer_factory=NullWriter)
            with self.assertRaises(FileExistsError):
                run_job(self.context, 2, 3405, 'B_fit', output=output, writer_factory=NullWriter)
            with patch('training_revision.RevisionTrainer.step', side_effect=AssertionError('completed job reran')):
                result = run_job(self.context, 2, 3405, 'B_fit', output=output, resume=True, writer_factory=NullWriter)
            self.assertEqual(result['status'], 'completed')
            (output / 'config.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                run_job(self.context, 2, 3405, 'B_fit', output=output, resume=True, writer_factory=NullWriter)

    def test_phase_seal_is_absent_for_profile_and_covers_all_completed_jobs(self):
        # A lightweight fake job isolates orchestration from network training.
        # The scientific training behavior is checked by the other tests.
        from tqg_gda.utils import sha256
        stage = self.root / 'phase'
        stage.mkdir()
        observed = []
        def fake_job(context, domain, seed, arm, *, output, stop_after, resume, budget):
            observed.append((domain, seed, arm, stop_after, resume))
            output.mkdir(parents=True, exist_ok=True)
            names = ('last.pt', 'selected.pt', 'history.jsonl', 'config.json', 'RUN_INFO.json')
            for name in names:
                (output / name).write_text('synthetic-fixture')
            result = dict(status='profile_paused' if stop_after else 'completed',
                completed_updates=stop_after or 200, config_sha256='synthetic-config',
                source_checkpoint_sha256='synthetic-source',
                last_checkpoint=str(output / 'last.pt'), selected_checkpoint=str(output / 'selected.pt'),
                artifact_hashes={name: sha256(output / name) for name in names})
            (output / 'result.json').write_text(json.dumps(result))
            return result
        with patch('experiment_revision.run_job', side_effect=fake_job):
            profile = run_phase(self.context, 'A', profile=True, stage=stage)
            self.assertEqual(profile['status'], 'profile_paused')
            self.assertEqual(observed, [(2, 3405, 'T5', 10, False)])
            self.assertFalse((stage / 'phase_A_training_seal.json').exists())
            completed = run_phase(self.context, 'A', resume=True, stage=stage)
        self.assertEqual(completed['status'], 'completed')
        self.assertEqual(len(completed['records']), 80)
        seal = json.loads((stage / 'phase_A_training_seal.json').read_text())
        self.assertTrue(seal['complete'])
        self.assertEqual(seal['commit'], self.context.commit)
        self.assertEqual(len(seal['jobs']), 80)
        for job in seal['jobs']:
            self.assertEqual(Path(job['path']).parts[-3:], (job['task'], str(job['seed']), job['arm']))
            self.assertIn('last.pt', job['outputs'])
            self.assertIn('result.json', job['outputs'])
            for artifact in job['outputs'].values():
                self.assertEqual(sha256(artifact['path']), artifact['sha256'])


if __name__ == '__main__':
    unittest.main()
