"""Generated topology, analytic KL projection, and transfer counterexample."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.special import expit, logit
import torch

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import composition, projection

ROOT = Path(__file__).resolve().parents[2] / 'experiments/mutagenicity/r8_图结构观察量与类别约束_20261121'
OPTIONS = json.loads((ROOT/'configs/pilot-plan.json').read_text())['projection']


class PilotTests(unittest.TestCase):
    def test_unordered_pairs_unknown_and_edgeless(self):
        def graph(nodes, edge_index, kinds):
            return dict(x=torch.nn.functional.one_hot(torch.tensor(nodes), 3),
                        edge_index=torch.tensor(edge_index, dtype=torch.int64).reshape(2, -1),
                        edge_attr=torch.nn.functional.one_hot(torch.tensor(kinds, dtype=torch.long), 2))
        original = graph([1, 0], [[0, 1], [1, 0]], [1, 1])
        fit = [composition.graph_counts(original, 3, 2)]
        vocabulary = composition.Vocabulary.fit(fit)
        query = [composition.graph_counts(graph([2, 2], [[0, 1], [1, 0]], [0, 0]), 3, 2),
                 composition.graph_counts(graph([0], [[], []], []), 3, 2)]
        np.testing.assert_array_equal(vocabulary.encode([*fit, *query]),
                                      [[.5, .5, 0, 1, 0], [0, 0, 1, 0, 1], [1, 0, 0, 0, 0]])
        with self.assertRaises(ValueError):
            composition.graph_counts(original | {'y': torch.tensor(0)}, 3, 2)
        with self.assertRaises(ValueError):
            composition.graph_counts(graph([1, 0], [[0, 1], [1, 0]], [1, 0]), 3, 2)

    def test_base_feasibility_returns_original_logits_exactly(self):
        v = np.array([0., 1.]*40)
        z = logit(np.tile([.22, .78], 40))
        bands = projection.Bands.build(v, np.array([.2, .8]), np.array([.1, .1]), .05)
        interval = projection.prior_interval(bands, OPTIONS)
        result = projection.joint_projection(z, bands, interval, OPTIONS)
        self.assertEqual(result['status'], 'base_feasible')
        np.testing.assert_array_equal(projection.projected_logits(z, v, result), z)
        self.assertEqual(projection.prior_projection(z, interval, OPTIONS)['status'], 'base_feasible')

    def test_two_score_exact_solution_and_prior_control(self):
        v = np.tile([0., 1.], 40)
        z = logit(np.tile([.62, .38], 40))
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), 0.)
        interval = projection.prior_interval(bands, OPTIONS)
        np.testing.assert_allclose([interval['lower'], interval['upper']], .5, atol=1e-10, rtol=0)
        result = projection.joint_projection(z, bands, interval, OPTIONS, exact=True)
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(expit(projection.projected_logits(z, v, result)),
                                   np.tile([.2, .8], 40), atol=1e-8, rtol=0)
        self.assertLess(result['dual_kkt_inf'], 1e-8)
        self.assertEqual(projection.prior_projection(z, interval, OPTIONS)['status'], 'base_feasible')

    def test_banded_solution_kkt_and_pointwise_analytic_limits(self):
        v = np.array([0., 1.]*40)
        z = logit(np.full(len(v), .5))
        # Symmetry sets E[q]=.5. Both moments allow q(1) >= .8-2*eps_A.
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), .025)
        result = projection.joint_projection(z, bands, projection.prior_interval(bands, OPTIONS), OPTIONS)
        self.assertTrue(result['valid'], result)
        self.assertLess(result['dual_kkt_inf'], 1e-8)
        np.testing.assert_allclose(expit(projection.projected_logits(z, v, result)),
                                   np.tile([.25, .75], 40), rtol=0, atol=1e-8)

    def test_exact_empty_and_boundary_have_distinct_status(self):
        v = np.repeat(.1, 6)
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), 0.)
        self.assertEqual(projection.prior_interval(bands, OPTIONS)['status'], 'infeasible')
        v = np.tile([0., 1.], 5)
        bands = projection.Bands.build(v, np.array([0., 1.]), np.zeros(2), 0.)
        intervals = projection.prior_interval(bands, OPTIONS)
        self.assertEqual(projection.joint_projection(np.zeros(len(v)), bands, intervals, OPTIONS, exact=True)['status'],
                         'boundary_only_or_near_boundary')

    def test_reflection_matches_prior_and_marginal_but_projection_harms(self):
        v = np.tile([0., 1.], 50)
        stable, reflected = .2+.6*v, .8-.6*v
        self.assertAlmostEqual(stable.mean(), reflected.mean(), places=14)
        logits = logit(reflected)
        bands = projection.Bands.build(v, np.array([.2, .8]), np.zeros(2), 0.)
        result = projection.joint_projection(logits, bands, projection.prior_interval(bands, OPTIONS), OPTIONS, exact=True)
        after = projection.projected_logits(logits, v, result)
        expected_nll = lambda z: (np.logaddexp(0., z)-reflected*z).mean()
        self.assertGreater(expected_nll(after), expected_nll(logits)+.8)
        self.assertLess(bands.slack(reflected).min(), -.2)

    def test_obs_tamper_aborts_before_target_label_access(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import pilot

        plan = json.loads((ROOT/'configs/pilot-plan.json').read_text())
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'OBS.json').write_text('{}')
            with patch.object(pilot.r5_inputs, 'evaluation_labels') as broker:
                with self.assertRaises(PermissionError):
                    pilot.evaluate_observation(output, 'bad-digest', {}, plan, ROOT)
                broker.assert_not_called()

    def test_legal_obs_scores_both_target_pools_and_retains_gate(self):
        import csv
        from tempfile import TemporaryDirectory
        from unittest.mock import call, patch

        from experiments.common.io import digest, write_json
        from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import inputs, pilot

        plan = json.loads((ROOT/'configs/pilot-plan.json').read_text())
        plan.update(setting='generated', directions=[[0, 1]], seeds=[7])
        original = dict(real_model_seeds=[7], phases={'P': dict(setting='generated', directions=[[0, 1]])},
                        evaluation=dict(cohort_bootstrap_replicates=20, cohort_bootstrap_seed=123,
                                        paired_interval=dict(strata=['target_domain', 'class'],
                                                             reuse_draws_across_methods_seeds_and_source_directions=True,
                                                             conditional_on_fitted_models=True,
                                                             minimum_graphs_per_class_for_interval=2,
                                                             percentiles=[2.5, 97.5], level=.95)))
        key, task = next(inputs.task_grid(plan))
        manifest = f'blocks/{key}/manifest.json'
        source = {'validation_v_by_class': [dict(n=10, mean=.3, sd=.1), dict(n=9, mean=.65, sd=.13)]}
        result = dict(status='base_feasible', valid=True, intercept=0., slope=0.)
        status = dict(task=task, source_key='src', manifest=manifest, mu=[.3, .65],
                      solutions={name: result for name in pilot.PROJECTED})
        v = np.linspace(.1, .9, 8)
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([-2., -.9, -.8, .2, -.1, .6, .9, 1.8])
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'predictions').mkdir()
            write_json(output/'source-statistics.json', {'src': source})
            write_json(output/'block-status.json', {key: status})
            write_json(output/'executed-config.json', {'plan': plan})
            write_json(output/'input-checks.json', {})
            for suffix, start in (('A', 0), ('H', 8)):
                data = dict(ids=np.arange(start, start+len(v)), v=v)
                data.update({f'z_{name}': scores for name in pilot.METHODS})
                data.update({f'decision_{name}': (scores > 0).astype(int) for name in pilot.METHODS})
                pilot.fit.write_npz(output/f'predictions/{key}-{suffix}.npz', **data)
            hashes = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
            write_json(output/'OBS.json', dict(stage='R8_OBS', parent=plan['r7_final_digest'], files=hashes))
            obs = digest(output/'OBS.json')
            with patch.object(pilot.r5_inputs, 'evaluation_labels') as broker:
                load = broker.return_value.__enter__.return_value.load
                load.return_value = labels
                gate = pilot.evaluate_observation(output, obs, original, plan, ROOT)
                self.assertEqual(load.call_args_list, [call(manifest, 'adapt', authenticated_snapshot=True),
                                                       call(manifest, 'holdout', authenticated_snapshot=True)])
            self.assertFalse(gate['passed'])
            self.assertEqual(gate['directions']['M0->M1']['SamePrior']['accuracy_pp'], 0.)
            with (output/'endpoints.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2*len(pilot.METHODS))
            self.assertEqual({row['pool'] for row in rows}, {'adapt', 'holdout'})
            self.assertEqual(json.loads((output/'FINAL.json').read_text())['observation_sha256'], obs)

    def test_gate_uses_exact_direction_and_pooled_accuracy_ties(self):
        from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import pilot

        seeds = [3405, 3406, 3407, 3408, 3409]
        original = dict(real_model_seeds=seeds, phases={'P': dict(directions=[[0, 1], [0, 3]])})
        rows = []
        for target, n, net in [(1, 307, [-3, -2, -1, 3, 3]), (3, 338, [-3, -3, 2, 1, 3])]:
            for seed, delta in zip(seeds, net):
                rows.append(dict(seed=seed, source=0, target=target, n=n, available=True,
                                 net=delta, accuracy_pp=100*delta/n, left='Joint', right='SourceCal'))
                rows.append(rows[-1] | {'right': 'SourceFusion'})
                rows.append(rows[-1] | {'right': 'SamePrior', 'net': 1, 'accuracy_pp': 100/n})
        gate = pilot.decide({}, rows, original)
        self.assertFalse(gate['passed'])
        for direction in gate['directions'].values():
            self.assertEqual(direction['SourceCal']['accuracy_pp'], 0.)
            self.assertFalse(direction['SourceCal']['strictly_positive'])
            self.assertTrue(direction['SamePrior']['strictly_positive'])
        self.assertEqual(gate['pooled_vs_source_fusion']['exact_accuracy_fraction'], '0')
