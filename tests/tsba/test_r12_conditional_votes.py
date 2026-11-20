from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
from scipy.optimize import minimize_scalar

from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import (
    conditional_votes, vote_mixture as mix, vote_scoring)


class ConditionalVotesTests(TestCase):
    def test_source_smoothing_and_equal_posteriors(self):
        v = np.array([[5,0,0,4],[2,3,0,4]],dtype=np.int64)
        g0,g1,p = mix.template(v)
        self.assertEqual(p,F(1,2))
        self.assertEqual(g0,(F(1,2),F(1,22),F(1,22),F(9,22)))
        self.assertEqual(g1,(F(5,22),F(7,22),F(1,22),F(9,22)))
        fit = mix.fit_row(v,np.zeros(4,dtype=np.int64))
        self.assertEqual(fit['status'],'empty_A')
        self.assertEqual(fit['actions'],[0,1,0,0])
        self.assertIsNone(fit['residual_tv'])
        self.assertEqual(fit['estimated_p'],float(p))

    def test_fallbacks_apply_original_counts(self):
        cases = [
            (np.array([[10,5,0,2],[0,0,0,0]]), 'missing_V_class', [0,0,0,0]),
            (np.array([[0,0,0,0],[10,5,0,2]]), 'missing_V_class', [1,1,0,1]),
            (np.array([[1,1,1,1],[4,4,4,4]]), 'equal_templates', [1,1,1,1]),
            (np.zeros((2,4),dtype=np.int64), 'missing_V_class', [0,0,0,0]),
        ]
        for v, status, expected in cases:
            fit = mix.fit_row(v,np.array([100,3,7,9]))
            self.assertEqual((fit['status'],fit['actions']),(status,expected))
            self.assertIsNone(fit['derivative_signs'])

    def test_identical_a_vote_at_critical_prior_ties_to_zero(self):
        v = np.array([[3,0,0,0],[0,3,0,0]])
        fit = mix.fit_row(v,np.array([20,20,3,3]))
        self.assertEqual(fit['estimated_p'],.5)
        self.assertEqual(fit['derivative_signs'],[-1,1,0,0])
        self.assertEqual(fit['actions'],[0,1,0,0])
        above = mix.fit_row(v,np.array([200,201,30,30]))
        below = mix.fit_row(v,np.array([201,200,30,30]))
        self.assertEqual(above['actions'],[0,1,1,1])
        self.assertEqual(below['actions'],[0,1,0,0])

    def test_fit_preserves_joint_votes_and_matches_independent_likelihood(self):
        v = np.array([[36,0,0,4],[4,0,0,36]])
        # Marginally identical B0/WL votes by symmetry, but agreement cells
        # remain much more frequent than the product of the separate votes.
        g0,g1,p0 = mix.template(v)
        self.assertNotEqual(g0[0],(g0[0]+g0[1])*(g0[0]+g0[2]))
        a = np.array([3,1,0,27])
        fitted = mix.fit_row(v,a)
        d = np.array(g1,dtype=float)-np.array(g0,dtype=float)

        def negative(q):
            density = np.asarray(g0,dtype=float)+q*d
            return -(a@np.log(density)+float(p0)*np.log(q)+float(1-p0)*np.log1p(-q))

        result = minimize_scalar(negative,bounds=(1e-8,1-1e-8),method='bounded')
        self.assertTrue(result.success)
        self.assertAlmostEqual(fitted['estimated_p'],result.x,places=5)

    def test_stratified_and_global_priors_swap_whole_rules_and_keep_lock(self):
        v = np.array([[[20,7,0,1],[20,7,0,1]], [[1,0,3,20],[1,0,3,20]]])
        a = np.array([[100,0,0,0],[0,0,0,100]])
        reverse = a[::-1].copy()
        local = mix.rules(v,a)
        opposite = mix.rules(v,reverse)
        empty_layer = mix.rules(v,np.array([[100,0,0,0],[0,0,0,0]]))
        self.assertEqual(empty_layer['conditional'][1]['status'],'empty_A')
        self.assertNotEqual(local['conditional'][0]['actions'],local['conditional'][1]['actions'])
        bins = np.arange(8)
        pred = mix.predict(local,[opposite,empty_layer],bins)
        np.testing.assert_array_equal(pred['Other1'],mix.predict(opposite,[local,empty_layer],bins)['Own'])
        np.testing.assert_array_equal(pred['OwnLock'][[0,3,4,7]],[0,1,0,1])
        np.testing.assert_array_equal(pred['OwnLock'][[1,2,5,6]],pred['Own'][[1,2,5,6]])
        np.testing.assert_array_equal(pred['Global'],local['global_prior']['actions'])
        np.testing.assert_array_equal(
            mix.encode((bins//2)%2,bins%2,bins//4),bins)
        self.assertEqual(mix.source_counts(bins,np.arange(8)%2).sum(),8)

    def test_identical_observation_can_mask_a_different_optimal_action(self):
        g0=(F(6,10),F(2,10),F(1,10),F(1,10))
        g1=(F(2,10),F(6,10),F(1,10),F(1,10))
        shift=tuple((b-a)/5 for a,b in zip(g0,g1))
        alternate0=tuple(a+b for a,b in zip(g0,shift))
        alternate1=tuple(a+b for a,b in zip(g1,shift))
        observed=lambda low,high,p: tuple((1-p)*a+p*b for a,b in zip(low,high))
        self.assertEqual(observed(g0,g1,F(3,5)),observed(alternate0,alternate1,F(2,5)))
        self.assertGreater(F(3,5)*g1[2],F(2,5)*g0[2])
        self.assertLess(F(2,5)*alternate1[2],F(3,5)*alternate0[2])

    def test_ownlock_strata_and_paired_foreign_mean(self):
        acts = dict(B0=np.array([0,0,1,1]),WL=np.array([0,0,1,0]),Size=np.array([1,0,0,1]),
                    Own=np.array([1,1,0,1]),OwnLock=np.array([0,0,1,1]))
        label = np.array([1,0,0,1])
        flags = {k:v==label for k,v in acts.items()}
        strata = {r['stratum']:r for r in vote_scoring.lock_rows(acts,flags)}
        self.assertEqual((strata['all_agree']['corrected'],strata['all_agree']['harmed']),(2,1))
        self.assertEqual(strata['size_opposes']['net'],2)
        self.assertEqual(strata['size_agrees']['net'],-1)
        keep = flags['OwnLock']
        both = {m:keep for m in conditional_votes.METHODS} | flags
        both['Other2']=flags['Own']
        weights = np.eye(4)
        summary = {arm:(gain,harm,draw) for arm,gain,harm,draw in vote_scoring.compare(both,weights)}
        gain,harm,draw = summary['OtherMean']
        self.assertEqual((gain,harm),(F(1),F(1,2)))
        np.testing.assert_array_equal(draw,[.5,-.5,.5,0])

    def test_no_observation_no_h_reader(self):
        plan = dict(settings=['proteins_edge'],domains=[0,1],seeds=[3405])
        keys=['r10_fit_sha256','r10_obs_sha256','r11_unlabeled_sha256','r16_fit_sha256','r16_obs_sha256']
        history = mock.Mock(plan=plan)
        with TemporaryDirectory() as directory, mock.patch.object(vote_scoring.inputs,'GraphLabelSession') as reader:
            with self.assertRaises(FileNotFoundError):
                vote_scoring.score(Path(directory),None,dict.fromkeys(keys,'hash'),history,'missing')
            reader.assert_not_called()

    def test_gate_uses_exact_minima_foreign_average_and_static_wins(self):
        settings=['mutagenicity_edge','mutagenicity_node','proteins_edge']
        plan=dict(settings=settings,domains=[0,1,2,3],seeds=[3405])
        cfg=dict(minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        arms=('Static','Global',*conditional_votes.FIXED,'OtherMean')
        measured={(s,a,b,3405,'Own',arm):F(1,200) for s in settings
                  for a,b in permutations(plan['domains'],2) for arm in arms}
        self.assertTrue(vote_scoring.gate(plan,cfg,measured)['extend_to_reserve_seeds'])
        measured['proteins_edge',0,1,3405,'Own','PoolMajority']=F(0)
        result=vote_scoring.gate(plan,cfg,measured)
        self.assertEqual(result['settings']['proteins_edge']['highest_fixed'],'PoolMajority')
        self.assertFalse(result['extend_to_reserve_seeds'])
