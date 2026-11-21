"""Exact ATC confidence ties and fixed-action aggregate scoring."""
from fractions import Fraction
import unittest

import numpy as np

from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import (
    target_confidence as atc, target_confidence_scoring as score)


class ATCTests(unittest.TestCase):
    def test_fractional_tie_credit_and_endpoints(self):
        validation = [-4, -2, 2, 2, 1]
        fit = atc.fit_threshold(validation, 2)
        self.assertEqual(fit['threshold'], 2)
        self.assertEqual(Fraction(fit['tie_credit']), Fraction(1, 3))
        self.assertEqual(atc.estimate(validation, fit), Fraction(2, 5))
        self.assertEqual(atc.estimate(np.array([9, -2, np.nextafter(2., 0)]), fit), Fraction(4, 9))
        for k in (0, len(validation)):
            self.assertEqual(atc.estimate([9, 0], atc.fit_threshold(validation, k)), Fraction(k, len(validation)))

    def test_selection_ties_use_source_accuracy_and_then_registered_order(self):
        estimates = dict.fromkeys(atc.MODELS, Fraction(8, 9))
        fit = {m: dict(k=7,n=9) for m in atc.MODELS}
        self.assertEqual(atc.select(estimates, fit), 'WL')
        fit['Size']['k'] = 8
        self.assertEqual(atc.select(estimates, fit), 'Size')
        estimates['B0'] = Fraction(9, 10)
        self.assertEqual(atc.select(estimates, fit), 'B0')

    def test_class_counts_recover_hard_accuracy_and_foreign_mean_of_metrics(self):
        cells = [(0, 0)]*8
        cells[0] = (4, 2)
        cells[1] = (1, 3)
        cells[6] = (0, 2)
        rows = np.repeat(np.arange(8), np.asarray(cells).sum(axis=1))
        unit = 2*np.array([[atc.BITS[m][r] for m in atc.MODELS] for r in rows])-1
        fit = atc.v_fit(unit, np.array(cells).T.reshape(2, 2, 4).tolist())
        self.assertEqual({m: k['k'] for m,k in fit.items()}, {'B0':7, 'WL':7, 'Size':7})
        select = {'SourceV':'WL','OwnATC':'WL','OtherATC1':'WL','OtherATC2':'B0'}
        policies, results = score.score_policies(cells, select)
        self.assertEqual(results['OtherATCMean'][1], (results['WL'][1]+results['B0'][1])/2)
        correct, harm = score.paired_counts(cells, policies['OwnATC'], policies['OtherATCMean'])
        self.assertEqual((correct,harm), (Fraction(3,2), Fraction(3,2)))
