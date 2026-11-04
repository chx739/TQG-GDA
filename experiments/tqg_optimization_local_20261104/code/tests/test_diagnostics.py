import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from tqg_gda.diagnostics import reserve_target_ids, measure_feedbacks, feedback_digest, summarize_blocks
from tqg_gda.feedback import ProbeConfig, build_feedback
from test_feedback import Tiny
from helpers import fixture_files, small_config, task_from


ROOT = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(31)
        self.model = Tiny().double().eval()
        self.source = torch.randn(5, 2, dtype=torch.float64)
        self.query = torch.randn(4, 2, dtype=torch.float64)
        self.evaluation = torch.randn(3, 2, dtype=torch.float64)
        self.source_labels = torch.tensor([0, 1, 0, 1, 0])
        self.eval_labels = torch.tensor([1, 0, 1])

    def feedbacks(self):
        return {m: build_feedback(self.model, self.source, self.source_labels, self.query,
                                 ProbeConfig(count=2, direction=m), torch.Generator().manual_seed(7))
                for m in ('target', 'random', 'source')}

    def test_target_partition_is_label_blind_fixed_disjoint_and_exhaustive(self):
        ids = tuple(range(20))
        diag, final = reserve_target_ids(ids, .5, 271828)
        self.assertEqual((diag, final), reserve_target_ids(ids, .5, 271828))
        self.assertFalse(set(diag) & set(final))
        self.assertEqual(set(diag) | set(final), set(ids))
        self.assertEqual(len(diag), 10)

    def test_stateless_diagnosis_and_target_labels_cannot_change_feedback(self):
        original = copy.deepcopy(self.model.state_dict())
        feedbacks = self.feedbacks()
        digests = {m: feedback_digest(f) for m, f in feedbacks.items()}
        a = measure_feedbacks(self.model, self.query, self.evaluation, self.eval_labels, feedbacks)
        b = measure_feedbacks(self.model, self.query, self.evaluation, 1-self.eval_labels, feedbacks)
        self.assertNotEqual(a['baseline_eval_ce'], b['baseline_eval_ce'])
        for m in feedbacks:
            self.assertEqual(a['methods'][m]['feedback_sha256'], digests[m])
            self.assertEqual(b['methods'][m]['feedback_sha256'], digests[m])
        for n, t in self.model.state_dict().items():
            self.assertTrue(torch.equal(t, original[n]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_small_step_decrease_matches_gradient_cosine_and_equal_length(self):
        measured = measure_feedbacks(self.model, self.query, self.evaluation, self.eval_labels,
                                     self.feedbacks(), step_fractions=(1e-6,))
        lengths = []
        for mode in ('target', 'random', 'source'):
            row = measured['methods'][mode]
            self.assertFalse(row['zero_feedback'])
            step = row['steps'][0]
            predicted = measured['oracle_gradient_norm'] * row['cosine'] * step['parameter_step_length']
            self.assertAlmostEqual(step['ce_decrease'], predicted, delta=1e-9)
            lengths.append(step['parameter_step_length'])
        self.assertEqual(len(set(lengths)), 1)
        self.assertEqual(measured['methods']['no_feedback']['steps'][0]['ce_decrease'], 0.)

    def test_zero_feedback_stays_in_average_and_is_not_discarded(self):
        feedbacks = self.feedbacks()
        feedbacks['target'].correction.zero_()
        with torch.no_grad():
            feedbacks['target'].reference = self.model(self.query).detach()
        measured = measure_feedbacks(self.model, self.query, self.evaluation, self.eval_labels, feedbacks)
        self.assertEqual(measured['methods']['target']['cosine'], 0.)
        self.assertTrue(measured['methods']['target']['zero_feedback'])
        summary = summarize_blocks([measured, measured])
        self.assertEqual(summary['blocks'], 2)
        self.assertEqual(summary['methods']['target']['zero_feedback_fraction'], 1.)
        self.assertEqual(summary['methods']['target']['mean_cosine'], 0.)

    def test_zero_oracle_gradient_excludes_all_methods_together(self):
        with torch.no_grad():
            self.model.head.weight.zero_(); self.model.head.bias.zero_()
        evaluation = self.evaluation[:1].repeat(4, 1)
        measured = measure_feedbacks(self.model, self.query, evaluation, torch.tensor([0, 1, 0, 1]),
                                     self.feedbacks())
        self.assertFalse(measured['informative'])
        self.assertTrue(all(r['cosine'] is None for r in measured['methods'].values()))
        summary = summarize_blocks([measured])
        self.assertEqual(summary['uninformative_fraction'], 1.)
        self.assertIsNone(summary['target_minus_random']['mean_cosine'])

    def test_label_reader_rejects_final_holdout_and_never_reads_poisoned_labels(self):
        reader = script_module('run_diagnostic').read_diagnostic_labels
        with tempfile.TemporaryDirectory() as root:
            files = fixture_files(root, target_label=None)
            task = task_from(files, small_config())
            diag, final = reserve_target_ids(task.target_ids, .5, 271828)
            # Poison FINAL labels in memory. Diagnostic reading must still work.
            for i in final:
                start = int(task.slices['y'][i]); end = int(task.slices['y'][i+1])
                task._raw.y[start:end] = 999
            self.assertEqual(len(reader(task, diag, diag)), len(diag))
            with self.assertRaises(PermissionError):
                reader(task, final, diag)

    def test_aggregation_requires_five_seeds_and_rejects_duplicates(self):
        import json
        from tqg_gda.utils import sha256
        aggregate = script_module('summarize_diagnostic').aggregate
        path = ROOT / 'configs/diagnostics/e1_target_vs_random.json'
        plan = json.loads(path.read_text()); plan_hash = sha256(path)
        measured = measure_feedbacks(self.model, self.query, self.evaluation, self.eval_labels, self.feedbacks())
        summaries = []
        for seed in plan['source_seeds']:
            summary = summarize_blocks([measured] * plan['blocks'])
            summary.update(experiment=plan['experiment'], seed=seed, task_fingerprint='test',
                           comparison_fingerprint='same_config_and_code', plan_sha256=plan_hash)
            summaries.append(summary)
        result = aggregate(summaries[:1], plan, plan_hash)
        self.assertEqual(result['decision'], 'incomplete_registered_seeds')
        result = aggregate(summaries, plan, plan_hash)
        self.assertTrue(result['complete_registered_seeds'])
        with self.assertRaises(ValueError):
            aggregate([summaries[0], summaries[0]], plan, plan_hash)

    def test_support_requires_positive_transfer_not_only_beating_random(self):
        import json
        from tqg_gda.utils import sha256
        aggregate = script_module('summarize_diagnostic').aggregate
        path = ROOT / 'configs/diagnostics/e1_target_vs_random.json'
        plan = json.loads(path.read_text()); plan_hash = sha256(path)
        measured = measure_feedbacks(self.model, self.query, self.evaluation, self.eval_labels, self.feedbacks())
        summaries = []
        for seed in plan['source_seeds']:
            s = summarize_blocks([measured] * plan['blocks'])
            s.update(experiment=plan['experiment'], seed=seed, task_fingerprint='test',
                     comparison_fingerprint='test', plan_sha256=plan_hash)
            s['target_minus_random']['mean_cosine'] = .1
            s['methods']['target']['mean_cosine'] = -.1
            for step in s['target_minus_random']['steps']:
                step['mean_ce_decrease_gain'] = .005
            for step in s['methods']['target']['steps']:
                step['mean_ce_decrease'] = .01
            summaries.append(s)
        result = aggregate(summaries, plan, plan_hash)
        self.assertNotEqual(result['decision'], 'initial_mechanism_support')
        for s in summaries:
            s['methods']['target']['mean_cosine'] = .2
        result = aggregate(summaries, plan, plan_hash)
        self.assertEqual(result['decision'], 'initial_mechanism_support')


if __name__ == '__main__':
    unittest.main()
