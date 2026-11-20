"""R11 phase barriers, prior solvers, and paired accuracy controls."""
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.special import expit, logit

from experiments.common.io import write_json
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import (
    inputs, prior_estimation, prior_run, prior_score, run, thresholds,
)

OPTIONS = dict(prior_floor=1e-8, initial_prior='source_prior', max_iterations=100000, tolerance=1e-10)


class PriorTests(unittest.TestCase):
    def test_bounded_root_matches_finite_mixture(self):
        # With pS=1/2 these probabilities give an interior optimum q=5/6.
        scores = logit(np.repeat([.8, .2], [7, 3]))
        fits = prior_estimation.estimate(scores, .5, OPTIONS)
        self.assertAlmostEqual(fits['root']['target_prior'], 5/6, places=13)
        self.assertAlmostEqual(fits['em']['target_prior'], 5/6, places=8)
        self.assertAlmostEqual(fits['root']['derivative'], 0., places=13)
        self.assertAlmostEqual(fits['root']['final_log_likelihood'], fits['em']['final_log_likelihood'], places=14)

    def test_exact_constant_scores_keep_prior_even_after_logit_rounding(self):
        v = np.full(29, .1)
        source_prior = float(expit(v).mean())
        self.assertNotEqual(float(logit(source_prior)), v[0])
        fits = prior_estimation.estimate(v, source_prior, OPTIONS, constant_validation_score=v[0])
        self.assertTrue(fits['root']['flat'])
        self.assertEqual(fits['root']['target_prior'], source_prior)
        self.assertAlmostEqual(fits['em']['target_prior'], source_prior, places=14)

    def test_tiny_nonzero_drift_is_distinct_from_flat(self):
        fits = prior_estimation.estimate(np.full(29, 1e-12), .5, OPTIONS, constant_validation_score=0.)
        self.assertFalse(fits['root']['flat'])
        self.assertEqual(fits['root']['target_prior'], 1 - OPTIONS['prior_floor'])
        self.assertTrue(fits['root']['prior_at_bound'])
        self.assertAlmostEqual(fits['em']['target_prior'], .5, places=10)

    def test_em_failure_is_not_replaced_with_root(self):
        fits = prior_estimation.estimate(np.full(29, .1), .5, OPTIONS | {'max_iterations': 1})
        self.assertIn('error', fits['em'])
        self.assertEqual(fits['root']['target_prior'], 1 - OPTIONS['prior_floor'])

    def test_source_only_fit_once_and_exact_a_scores(self):
        v = Mock()
        v.load.return_value = np.array([0, 1])
        group = {}
        selected = {}
        for t in (1, 2, 3):
            group[t] = {'ids': {'adapt': [t*10, t*10+1]}, 'margins': {'adapt': np.array([-t, t])}}
            selected[str(t)] = [t*10+1]
        with patch.object(prior_run.prior_control, 'fit_calibrator',
                          return_value=dict(inverse_temperature=1., intercept=0.)) as fit, patch.object(
                              prior_estimation, 'estimate', return_value={}) as estimate:
            _, learned = prior_run.fit_scores(dict(margin=np.array([-1., 1.]), v_port=v), group, selected,
                                                dict(calibration={}, mlls=OPTIONS))
        v.load.assert_called_once_with()
        fit.assert_called_once()
        self.assertEqual(len(learned['targets']), 3)
        self.assertEqual([args[0][0] for args, _ in estimate.call_args_list], [1, 2, 3])

    def test_same_source_swap_controls_use_unchanged_foreign_priors(self):
        item = dict(ids=dict(holdout=[100]), task=dict(target=2), margins=dict(holdout=np.array([0.])))
        result = prior_run.predict(item, 0., dict(
            calibration=dict(inverse_temperature=1., intercept=0.), source_prior=.5, targets={
                '1': dict(em=dict(target_prior=.1), root=dict(target_prior=.9)),
                '2': dict(em=dict(target_prior=.6), root=dict(target_prior=.6)),
                '3': dict(em=dict(error='no convergence'), root=dict(target_prior=.9))}), [1, 3], 1e-12)
        self.assertTrue(result['MLLS'][0])
        self.assertFalse(result['Swap1'][0])
        self.assertNotIn('Swap2', result)
        self.assertTrue(result['SwapRoot1'][0])
        self.assertTrue(result['SwapRoot2'][0])
        self.assertFalse(result['B0'][0])
        self.assertFalse(prior_estimation.majority(2, 4))

    def test_incomplete_or_mutated_seal_stops_before_a_or_h_session(self):
        with TemporaryDirectory() as tmp, patch.object(inputs, 'GraphLabelSession') as labels:
            root = Path(tmp)
            plan = dict(settings=[], domains=[], seeds=[])
            with self.assertRaises(FileNotFoundError):
                prior_run.supervise(root, root, {}, {}, plan, {}, 'x')
            with self.assertRaises(FileNotFoundError):
                prior_score.score(root, root, {}, plan, {}, 'x', 'y')
            write_json(root / 'roles.json', {})
            checksum = prior_run.seal(root, 'UNLABELED', ['roles.json'], {})
            with self.assertRaises(PermissionError):
                prior_run.supervise(root, root, {}, {}, plan, {}, checksum)
            (root / 'roles.json').write_text('tampered')
            with self.assertRaises(PermissionError):
                prior_run.verified_members(root, 'UNLABELED', checksum, ['roles.json'], {})
            labels.assert_not_called()

    def test_main_group_rejects_shifted_a_or_h_ids(self):
        setting = 'generated'
        ids = [10, 20, 30]
        item = dict(key='block', task=dict(setting=setting, source=0, target=1, seed=2),
                    ids=dict(adapt=ids, holdout=[40]), manifest_sha256='role')
        role = dict(**item['task'], manifest_sha256='role', holdout_ids=[40], other_targets=[])
        validation = dict(ids=[5], lineage={'epoch':1})
        receipt = dict(validation_ids=[5], source_checkpoint={'epoch':1},
                       selected_adaptation_ids={'1': thresholds.subset_ids(setting, ids, 3)})
        self.assertEqual(prior_run.verify_group(setting, validation, {1:item}, receipt, {'block':role}),
                         receipt['selected_adaptation_ids'])
        for changed in (role | {'holdout_ids':[41]}, role | {'manifest_sha256':'altered'}):
            with self.assertRaises(PermissionError):
                prior_run.verify_group(setting, validation, {1:item}, receipt, {'block':changed})
        with self.assertRaises(PermissionError):
            prior_run.verify_group(setting, validation, {1:item},
                                   receipt | {'selected_adaptation_ids':{'1':[10,20,99]}}, {'block':role})

    def test_swap_mean_counts_and_intervals_share_the_same_draw(self):
        correct = dict(MLLS=np.array([True, True, False]),
                       Swap1=np.array([False, True, True]), Swap2=np.array([False, False, False]))
        weights = np.array([[1,0,0], [0,.3,.7], [.1,.7,.2]])
        rows = {p: (gain, harm, net, draw) for p, gain, harm, net, draw in prior_score.compare(correct, weights)}
        gain, harm, net, draw = rows['MLLS', 'SwapMean']
        self.assertEqual((gain, harm, net), (Fraction(3,2), Fraction(1,2), Fraction(1)))
        np.testing.assert_allclose(draw, (rows['MLLS','Swap1'][-1] + rows['MLLS','Swap2'][-1])/2)

    def test_gate_requires_each_control_and_both_exact_directions(self):
        plan = dict(settings=['g'], domains=[0, 1], seeds=[1])
        registered = dict(minimum_mean_accuracy_fraction=[1,100], positive_directions_per_setting=2, settings_required=1)
        pairs = {('g', s, t, 1, 'MLLS', arm): Fraction(1,100)
                 for s,t in [(0,1),(1,0)] for arm in prior_score.GATE_CONTROLS}
        summarize = lambda: run.summarize_pairs(plan, pairs, {}, prior_score.PAIRS)
        self.assertTrue(prior_score.gate(plan, registered, summarize(), pairs)['passed'])
        key = ('g', 0, 1, 1, 'MLLS', 'SourceCal')
        pairs[key] = Fraction(0)
        self.assertFalse(prior_score.gate(plan, registered, summarize(), pairs)['passed'])
        del pairs[key]
        result = prior_score.gate(plan, registered, summarize(), pairs)
        self.assertFalse(result['settings']['g']['complete'])
