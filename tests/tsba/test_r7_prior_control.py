"""Synthetic checks for the frozen calibration and MLLS controls."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import prior_control as control
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import prior_control_run as run


R7 = Path(__file__).resolve().parents[2] / 'experiments/mutagenicity/r7_共同解释与共享更新验证_20261119'
CONFIG = json.loads((R7/'configs/r7-p-accuracy-prior-control.json').read_text())


class PriorControlTests(unittest.TestCase):
    def test_positive_temperature_fit_recovers_constructed_calibration(self):
        probabilities = np.array([.2, .35, .6, .85])
        expected_slope, expected_bias = .8, -.35
        source_scores = (logit(probabilities) - expected_bias) / expected_slope
        scores = np.repeat(source_scores, 100)
        labels = np.concatenate([np.r_[np.ones(n, dtype=int), np.zeros(100-n, dtype=int)]
                                 for n in (20, 35, 60, 85)])
        fitted = control.fit_calibrator(scores, labels, CONFIG['calibration'])
        np.testing.assert_allclose([fitted['inverse_temperature'], fitted['intercept']],
                                   [expected_slope, expected_bias], rtol=0, atol=1e-7)
        np.testing.assert_allclose(expit(control.calibrate(scores, fitted)).mean(), labels.mean(), atol=1e-8)
        self.assertLess(fitted['source_validation_nll'], fitted['source_validation_nll_before'])

    def test_no_shift_is_identity_with_distinct_scores(self):
        scores = np.array([-2., -1.1, -.2, .2, 1.1, 2.])
        fitted = dict(inverse_temperature=1., intercept=0.)
        prior = control.fit_prior(scores, .5, CONFIG['mlls'])
        calibrated, shifted, balanced, target_balanced = control.adjusted_scores(scores, fitted, prior)
        self.assertEqual(prior['target_prior'], .5)
        for values in (calibrated, shifted, balanced, target_balanced):
            np.testing.assert_array_equal(values, scores)
        self.assertEqual(control.verify_rank_preservation(scores, shifted, 1e-12), 0)

    def test_em_recovers_label_shift_and_balanced_prior_cancels(self):
        # P(x=1|y=1)=.8 and P(x=1|y=0)=.2, source prior .5.
        # Target prior .7 yields a target mixture with .62 values of x=1.
        adaptation = logit(np.repeat([.2, .8], [38, 62]))
        prior = control.fit_prior(adaptation, .5, CONFIG['mlls'])
        self.assertAlmostEqual(prior['target_prior'], .7, places=8)
        self.assertGreater(prior['final_log_likelihood'], prior['initial_log_likelihood'])
        unique = np.array([-2.2, -.9, -.2, .1, .6, 1.8])
        calibrated, shifted, balanced, target_balanced = control.adjusted_scores(
            unique, dict(inverse_temperature=.8, intercept=-.2), prior)
        np.testing.assert_array_equal(rankdata(unique), rankdata(calibrated))
        np.testing.assert_array_equal(rankdata(unique), rankdata(shifted))
        np.testing.assert_allclose(balanced, target_balanced, atol=1e-14, rtol=0)
        self.assertTrue(np.any((calibrated > 0) != (shifted > 0)))
        self.assertEqual(control.verify_rank_preservation(unique, shifted, 1e-12), 0)

    def test_scoring_rejects_prediction_mutation_before_label_access(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = root/'frozen'
            member.write_bytes(b'predictions')
            write_json(root/'OBS.json', {'files': {'frozen': digest(member)}})
            seal = digest(root/'OBS.json')
            member.write_bytes(b'changed predictions')
            with patch.object(run, 'evaluation_labels') as labels:
                with self.assertRaises(PermissionError):
                    run.score(root, CONFIG, root, root, seal, {})
                labels.assert_not_called()
