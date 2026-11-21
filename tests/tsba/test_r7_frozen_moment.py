"""Exact mass allocation and role sequencing for frozen R7 risk replay."""
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    frozen_moment as m, frozen_moment_observation as o, frozen_moment_scoring as s)


class FrozenMomentTests(unittest.TestCase):
    def test_fractional_mass_bounds_match_linear_program(self):
        norm = np.array([.2,.2,.2])
        delta = np.array([-2.,1.,4.])
        mass = Fraction(3,2)
        self.assertEqual(m.fixed_mass_bounds(norm,delta,mass),(-1.3,.7000000000000001))
        rng = np.random.default_rng(107)
        for n in (2,7,14):
            d = rng.normal(size=n)
            x = rng.normal(size=n)
            for target in (Fraction(0),Fraction(1,3),Fraction(n,2),Fraction(n)):
                expected = []
                for sign in (1,-1):
                    solved = linprog(-sign*d,A_eq=np.ones((1,n)),b_eq=[float(target)],bounds=(0,1))
                    self.assertTrue(solved.success)
                    expected.append(float((x.mean()+sign*solved.fun/n)))
                np.testing.assert_allclose(m.fixed_mass_bounds(x,d,target),expected,atol=1e-12,rtol=0.)

    def test_constant_direction_and_mass_endpoint(self):
        normalizer = np.array([1.,2.,3.])
        low,high = m.fixed_mass_bounds(normalizer,np.full(3,-1.),Fraction(7,4))
        self.assertEqual(low,high)
        self.assertAlmostEqual(low,31/12)
        self.assertEqual(m.sum_mass(np.array([0.,.5,1.])), Fraction(3,2))

    def test_moment_gap_is_mass_plus_biased_covariance(self):
        norm = np.array([.1,.2,.4,.6])
        delta = np.array([-3.,1.,2.,4.])
        q = np.array([.1,.6,.9,1.])
        y = np.array([0,0,1,0])
        r = m.decomposition(norm,delta,q,y)
        self.assertAlmostEqual(r['error_gap'],float((q-y)@delta/len(q)))
        self.assertAlmostEqual(r['mass_term'],(q.mean()-y.mean())*delta.mean())
        self.assertAlmostEqual(r['assignment_term'],np.cov(q-y,delta,bias=True)[0,1])

    def test_freeze_uses_first_a_candidate_within_tolerance(self):
        choice,ties = m.most_adverse(np.array([-2e-12,0.,1e-16,-1e-15]),1e-12)
        self.assertEqual((choice,ties),(1,[1,2,3]))

    def test_mismatched_observation_cannot_open_old_labels(self):
        with (patch.object(o,'observed',side_effect=PermissionError),
              patch.object(s,'evaluation_labels') as reader):
            with self.assertRaises(PermissionError):
                s.evaluate_frozen(None,None,None,{},None,'invalid',{})
            reader.assert_not_called()
