"""Numerical contracts on the new SEGA encoder and complete TQG pipeline."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from helpers import fixture_files, ROOT
from tqg_gda.adapt import Adapter
from tqg_gda.classifier import GraphClassifier
from tqg_gda.config import AdaptConfig, Config, EncoderConfig, PretrainConfig
from tqg_gda.data import load_task
from tqg_gda.encoder import GraphEncoder
from tqg_gda.pipeline import run_task
from tqg_gda.predict import load_predictor, predict_graphs, predict_task
from tqg_gda.pretrain import pretrain_model
from tqg_gda.provenance import code_identity, runtime_identity, source_key, task_key
from tqg_gda.v1.query import ProbeConfig, Snapshot, build_feedback, feedback_loss
from tqg_gda.v2.query import (ParameterSnapshot, V2Config, build_gradient_feedback,
                            contrast_basis, parameter_feedback_loss, target_space)


class SegaPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        files = fixture_files(self.root/'inputs')  # target labels = 999 (invalid class)
        self.config = Config(cache_path=str(files[0]), source_split=str(files[1]),
            target_index=str(files[2]), dataset='fixture', seed=31, use_node_attributes=True,
            fit_fraction=.75, dtype='float64', encoder=EncoderConfig(hidden=5, num_layers=2, out_dim=3, dropout=.3),
            pretrain=PretrainConfig(epochs=2, batch_size=4),
            adapt=AdaptConfig(steps=2, fit_batch_size=4, query_batch_size=4, target_batch_size=3),
            v1=ProbeConfig(count=2, radius=.1),
            v2=V2Config(calibration=dict(max_iterations=50)), out_dir=str(self.root/'run'))
        self.task = load_task(self.config)

    def model(self, dtype=torch.float64, classes=2):
        return GraphClassifier(dict(in_dim=2, edge_dim=0, **self.config.to_dict()['encoder']), classes).to(dtype).eval()

    def batches(self):
        return (self.task.batch(self.task.query_ids, 'query'),
                self.task.batch(self.task.target_ids[:3], 'target'))

    def test_encoder_matches_reference_gin_and_gine_in_eval_and_training(self):
        spec = importlib.util.spec_from_file_location('sega_encoder_reference', ROOT/'references/sega-gda/sega/encoder.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        batch = self.task.batch(self.task.fit_ids[:4], 'fit', dtype=torch.float32)
        batch.edge_attr = torch.ones(batch.num_edges, 1)
        for conv in ('gin', 'gine'):
            reference = module.GraphEncoder(2, 1, hidden=5, num_layers=2, out_dim=3, conv=conv).float()
            actual = GraphEncoder(2, 1, hidden=5, num_layers=2, out_dim=3, conv=conv).float()
            actual.load_state_dict(reference.state_dict(), strict=True)
            for train in (True, False):
                reference.train(train); actual.train(train)
                torch.testing.assert_close(actual(batch), reference(batch), atol=0, rtol=0)
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)

    def test_v1_finite_responses_and_joint_gradient_on_sega_encoder(self):
        model = self.model()
        query, target = self.batches()
        original = copy.deepcopy(model.state_dict())
        feedback = build_feedback(model, query, query.y, target, self.config.v1, torch.Generator().manual_seed(7))
        self.assertGreater(feedback.statistics['valid_directions'], 0)
        snap = Snapshot(model)
        direction = tuple(feedback.directions[0][name] for name in snap.names)
        sensitivity = snap.jvp(target, direction).square().sum(-1).mean().sqrt()
        self.assertAlmostEqual(float(sensitivity), 1., places=5)
        self.assertTrue(all(n.startswith('encoder.') for d in feedback.directions for n in d))
        for graph, measured in ((query, feedback.source_features), (target, feedback.target_features)):
            rho = self.config.v1.radius
            plus = tuple(p+rho*v for p, v in zip(snap.values, direction))
            minus = tuple(p-rho*v for p, v in zip(snap.values, direction))
            zp, zm, z0 = snap.logits(graph, plus), snap.logits(graph, minus), snap.logits(graph)
            expected = torch.stack(((zp-zm)/(2*rho), (zp+zm-2*z0)/rho**2), -1)
            torch.testing.assert_close(measured[..., :2]*feedback.scales[:2], expected)
        parameters = tuple(model.parameters())
        computed = torch.autograd.grad(feedback_loss(model, target, feedback), parameters)
        expected = torch.autograd.grad(-(model(target)*feedback.correction).sum()/3, parameters)
        for actual, correct in zip(computed, expected):
            torch.testing.assert_close(actual, correct, atol=1e-12, rtol=1e-10)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], atol=0, rtol=0)
        self.assertFalse(feedback.reference.requires_grad)

    def test_v2_complete_space_contains_multiclass_oracle_gradient(self):
        model = self.model(classes=3)
        _, target = self.batches()
        snapshot = ParameterSnapshot(model)
        contrasts = contrast_basis(3, dtype=torch.float64, device='cpu')
        jacobian = snapshot.jacobian(target, contrasts)
        basis, _ = target_space(jacobian)
        loss = F.cross_entropy(model(target), torch.tensor([2, 0, 1]))  # synthetic oracle only
        gradient = torch.cat([p.flatten() for p in torch.autograd.grad(loss, model.parameters())])
        torch.testing.assert_close(basis@(basis.T@gradient), gradient, atol=1e-11, rtol=1e-10)
        self.assertTrue(any(n.startswith('head.') for n in snapshot.names))
        self.assertEqual(len(snapshot.flat), sum(p.numel() for p in model.parameters()))

    def test_v2_readout_is_label_independent_and_loss_executes_same_gradient(self):
        model = self.model()
        query, target = self.batches()
        first = build_gradient_feedback(model, query, query.y, target, self.config.v2)
        second = build_gradient_feedback(model, query, 1-query.y, target, self.config.v2)
        self.assertEqual(first.statistics['status'], 'gradient_feedback')
        torch.testing.assert_close(first.readout, second.readout, atol=0, rtol=0)
        self.assertFalse(torch.equal(first.coefficients, second.coefficients))
        calculated = torch.cat([v.reshape(-1) for v in torch.autograd.grad(parameter_feedback_loss(model, first), model.parameters())])
        torch.testing.assert_close(calculated, first.gradient, atol=1e-12, rtol=1e-10)
        for name in ('snapshot', 'basis', 'coefficients', 'source_residual', 'readout'):
            self.assertFalse(getattr(first, name).requires_grad)

    def test_adaptation_freezes_bn_and_dropout_and_executes_single_joint_step(self):
        for version in ('v1', 'v2'):
            model = self.model()
            initial = copy.deepcopy(model.state_dict())
            adapter = Adapter(model, replace(self.config, version=version))
            row = adapter.step(self.task)
            self.assertEqual(row['execution']['updates_per_query'], 1)
            self.assertLess(row['execution']['sgd_step_identity_error'], 1e-12)
            self.assertGreater(row['execution']['feedback_gradient_norm'], 0)
            if version == 'v2':
                self.assertLess(row['execution']['feedback_gradient_identity_error'], 1e-12)
            self.assertFalse(any(module.training for module in model.modules()))
            for name, value in model.named_buffers():
                torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
            for prefix in ('encoder.', 'head.'):
                self.assertTrue(any(not torch.equal(initial[n], p) for n,p in model.named_parameters() if n.startswith(prefix)))

    def test_zero_feedback_matches_fit_only_sgd_with_identical_fit_batches(self):
        config = replace(self.config, adapt=replace(self.config.adapt, feedback_weight=0.))
        for version in ('v1', 'v2'):
            model = self.model()
            first = Adapter(model, replace(config, version=version))
            control = Adapter(copy.deepcopy(model), replace(config, version='source_only'))
            for _ in range(2):
                a, b = first.step(self.task), control.step(self.task)
                self.assertEqual(a['batches']['fit'], b['batches']['fit'])
                for n, value in first.model.state_dict().items():
                    torch.testing.assert_close(value, control.model.state_dict()[n], atol=0, rtol=0)

    def test_target_label_boundary_both_versions(self):
        model = self.model()
        query, target = self.batches()
        target.y = torch.tensor([0, 1, 0])
        with self.assertRaises(PermissionError):
            build_feedback(model, query, query.y, target, self.config.v1, torch.Generator())
        with self.assertRaises(PermissionError):
            build_gradient_feedback(model, query, query.y, target, self.config.v2)

    def test_end_to_end_cache_and_inference_for_both_versions(self):
        first = run_task(replace(self.config, version='v1'))
        second = run_task(self.config)
        self.assertFalse(first['pretrain']['cache_hit'])
        self.assertTrue(second['pretrain']['cache_hit'])
        self.assertEqual(first['pretrain']['source_key'], second['pretrain']['source_key'])
        self.assertFalse(second['target_labels_used'])
        self.assertFalse(second['target_evaluated'])
        self.assertEqual(second['completed_steps'], 2)
        for result in (first, second):
            model, saved = load_predictor(Path(result['output'])/'model.pt')
            actual = predict_task(model, self.task)
            graphs = [self.task._input_graph(i) for i in self.task.target_ids]
            standalone, labels = predict_graphs(model, saved, graphs)
            torch.testing.assert_close(standalone, actual, atol=1e-12, rtol=1e-10)
            self.assertEqual(len(labels), len(graphs))
            self.assertNotIn('readout', saved)
        reused = run_task(self.config)
        self.assertTrue(reused['reused'])
        (Path(second['output'])/'predictions.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            run_task(self.config)

    def test_pretrain_updates_bn_but_never_uses_query_labels(self):
        code, runtime = code_identity(), runtime_identity(self.config)
        model, info, folder = pretrain_model(self.config, self.task, code, runtime)
        self.assertEqual(info['query_graphs_used'], 0)
        self.assertTrue(any(int(v) > 0 for n, v in model.named_buffers() if n.endswith('num_batches_tracked')))
        task = load_task(self.config)
        for i in task.query_ids:
            task.labels[i] = 999999  # CE would fail if warmup read these labels
        other = replace(self.config, out_dir=str(self.root/'other'))
        second, _, _ = pretrain_model(other, task, code, runtime)
        for n, v in model.state_dict().items():
            torch.testing.assert_close(v, second.state_dict()[n], atol=0, rtol=0)
        (folder/'best.pt').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            pretrain_model(self.config, self.task, code, runtime)

    def test_cache_identity_includes_data_code_configuration_and_runtime(self):
        code, runtime = code_identity(), runtime_identity(self.config)
        baseline = source_key(self.config, self.task, code, runtime)
        self.assertEqual(baseline, source_key(replace(self.config, version='v1'), self.task, code, runtime))
        self.assertNotEqual(baseline, source_key(replace(self.config, encoder=replace(self.config.encoder, hidden=9)), self.task, code, runtime))
        self.assertNotEqual(baseline, source_key(self.config, self.task, dict(code, sha256='changed'), runtime))
        task = copy.copy(self.task)
        task.identity = dict(task.identity, cache_sha256='changed')
        self.assertNotEqual(baseline, source_key(self.config, task, code, runtime))
        key = task_key(self.config, self.task, code, runtime)
        self.assertNotEqual(key, task_key(replace(self.config, adapt=replace(self.config.adapt, target_batch_size=4)), self.task, code, runtime))

    def test_numerical_failure_falls_back_to_one_source_step(self):
        model = self.model()
        adapter = Adapter(model, self.config)
        control = Adapter(copy.deepcopy(model), replace(self.config, version='source_only'))
        with patch('tqg_gda.v2.query.solve_calibration', side_effect=FloatingPointError('synthetic failure')):
            row = adapter.step(self.task)
        control.step(self.task)
        self.assertEqual(row['query']['status'], 'numerical_fallback_source_only')
        for n, value in model.state_dict().items():
            torch.testing.assert_close(value, control.model.state_dict()[n], atol=0, rtol=0)

    def test_config_rejects_silent_method_changes(self):
        for change in (dict(version='v2', adapt=replace(self.config.adapt, optimizer='adam')),
                       dict(v1=ProbeConfig(responses='odd')),
                       dict(v2=V2Config(signature='representation_only')),
                       dict(dtype='float16')):
            with self.assertRaises(ValueError):
                replace(self.config, **change)
        self.assertEqual(Config.from_dict(self.config.to_dict()), self.config)

    def test_initialization_and_cache_identity_ignore_ambient_default_dtype(self):
        previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous)
        # Includes GINE's internally constructed edge projection, not only Linear.
        for conv in ('gin', 'gine'):
            states = []
            for ambient in (torch.float32, torch.float64):
                torch.set_default_dtype(ambient)
                torch.manual_seed(19)
                model = GraphClassifier(dict(in_dim=2, edge_dim=1, hidden=4, num_layers=2,
                    out_dim=3, conv=conv), 2, dtype=torch.float32)
                self.assertEqual(torch.get_default_dtype(), ambient)
                states.append(copy.deepcopy(model.state_dict()))
            for name in states[0]:
                torch.testing.assert_close(states[0][name], states[1][name], atol=0, rtol=0)
        results = []
        for index, ambient in enumerate((torch.float32, torch.float64)):
            torch.set_default_dtype(ambient)
            config = replace(self.config, dtype='float32', version='source_only',
                out_dir=str(self.root/f'dtype_{index}'))
            result = run_task(config)
            model, _ = load_predictor(Path(result['output'])/'model.pt')
            self.assertEqual(torch.get_default_dtype(), ambient)
            results.append((result, model.state_dict()))
        self.assertEqual(results[0][0]['task_key'], results[1][0]['task_key'])
        self.assertEqual(results[0][0]['pretrain']['source_key'], results[1][0]['pretrain']['source_key'])
        for name in results[0][1]:
            torch.testing.assert_close(results[0][1][name], results[1][1][name], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
