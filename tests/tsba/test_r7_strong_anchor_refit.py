from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
from scipy.special import expit

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import (
    anchor_solver as solver, strong_anchor_refit as refit, strong_anchor_scoring as scoring)
from tsba.shared_update.solver import relative_risks, solve_master


class StrongAnchorTests(TestCase):
    def setUp(self):
        self.u = np.array([[-2.,1.,1.],[-.4,.2,1.],[.1,-1.,1.],[1.5,2.,1.],[3.,0.,1.]])
        self.c = np.array([-.4,1.2,0.,-1.,2.])
        self.s = np.array([.1,.5,.12,.8,.72])@self.u/len(self.c)
        self.lam = .1
        self.opt = dict(maxiter=100,tolerance=1e-7)

    def test_zero_solve_teacher_and_anchor_independent_initial_gradient_difference(self):
        u,c,lam = self.u,self.c,self.lam
        teacher = expit(c)@u/len(c)
        d,row = solver.solve_binary(c,u,teacher,lam,**self.opt)
        np.testing.assert_array_equal(d,np.zeros(3))
        self.assertLess(row['kkt_inf'],1e-14)
        for start in (c,-4*c+2):
            actual = solver.terms(start,u,self.s,lam,d)[1]-solver.terms(start,u,teacher,lam,d)[1]
            np.testing.assert_allclose(actual,teacher-self.s,rtol=0,atol=1e-14)

    def test_same_binary_objective_and_e_solution_as_original_master(self):
        u,c,s,lam = self.u,self.c,self.s,self.lam
        d,row = solver.solve_binary(c,u,s,lam,**self.opt)
        z = np.column_stack((np.zeros(len(c)),c))
        moments = np.array([[u.mean(0)-s,s]])
        w = np.array([-d/2,d/2])
        expected = relative_risks(z,u,w,moments)[0]+lam*(w*w).sum()/2
        self.assertAlmostEqual(expected,row['objective'],places=13)
        w_full,meta = solve_master(z,u,moments,regularization=lam,feasibility_tol=1e-8,
                                   optimality_tol=1e-6,master_maxiter=200)
        self.assertTrue(meta['valid'])
        np.testing.assert_allclose(w_full[1]-w_full[0],d,atol=2e-5)
        np.testing.assert_array_equal((c+u@d)>0, (c+u@(w_full[1]-w_full[0]))>0)

    def test_score_only_basis_preserves_penalty_and_projected_stationarity(self):
        q = solver.score_basis(np.array([[0.,0.,0.],[2.,1.,5.]]))
        np.testing.assert_allclose(q.T@q,np.eye(2),atol=1e-14)
        d,row = solver.solve_binary(self.c,self.u,self.s,self.lam,q=q,**self.opt)
        t,small = solver.solve_binary(self.c,self.u@q,self.s@q,self.lam,**self.opt)
        np.testing.assert_allclose(d,q@t,rtol=0,atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(d),np.linalg.norm(t),places=14)
        self.assertAlmostEqual(row['objective'],small['objective'],places=14)
        only_intercept = solver.score_basis(np.array([[0.,0.,-1.],[0.,0.,1.]]))
        np.testing.assert_array_equal(only_intercept,[[0.],[0.],[1.]])

    def test_calibrated_teacher_matches_two_moments_and_recovers_affine_teacher(self):
        for sign in (-1.,1.):
            v0 = np.array([[0.,0.,0.],[1.,3.,1.]])
            g = self.u@(v0[1]-v0[0])
            prob = expit(-.7+sign*1.3*g)
            s = prob@self.u/len(g)
            cal,note = solver.calibrated_teacher(g,self.u,s,v0)
            np.testing.assert_allclose(cal,s,rtol=0,atol=1e-12)
            self.assertAlmostEqual(note['slope']/note['sd'],sign*1.3,places=11)
            self.assertLess(abs(note['mass_error']),1e-13)
            self.assertLess(abs(note['raw_score_error']),1e-13)
        v0 = np.array([[0.,0.,0.],[0.,0.,1.]])
        cal,note = solver.calibrated_teacher(np.ones(len(self.u)),self.u,self.s,v0)
        np.testing.assert_allclose(cal,self.s[-1]*self.u.mean(0),atol=1e-13)
        self.assertEqual(note['slope'],0.)

    def test_calibrated_teacher_rejects_infinite_solution_at_score_boundary(self):
        u = np.array([[-1.,1.],[1.,1.]])
        v0 = np.array([[0.,0.],[1.,0.]])
        with self.assertRaises(ArithmeticError):
            solver.calibrated_teacher(u[:,0],u,u[1]/2,v0)

    def test_reference_moment_rejects_duplicate_theta(self):
        stats = np.array([[self.u.mean(0)-self.s,self.s],[self.u.mean(0)-self.s,self.s]])
        theta = np.array([[1.,2.],[2.,1.]])
        row,k = solver.reference_moment(theta,stats,theta[1],self.u)
        np.testing.assert_array_equal(row,self.s)
        self.assertEqual(k,1)
        with self.assertRaises(ValueError):
            solver.reference_moment(np.tile(theta[0],(2,1)),stats,theta[0],self.u)

    def test_full_foreign_weight_is_transferred_without_changes_to_held_features(self):
        h = np.array([[-.5,1.],[.25,1.]])
        item = dict(ids={'holdout':[5,2]},u=h,archived={'holdout_logits_B0':np.array([[0.,1.],[0.,-1.]]),
                                                       'holdout_logits_E':np.array([[0.,2.],[0.,-3.]])})
        fixes = {t:{refit.arm('WL',m):np.array([t,1.]) for m in refit.UPDATES} for t in (1,2,3)}
        x = refit.predict_h(item,fixes,{'WL':np.array([.3,-1.])},target=1)
        np.testing.assert_array_equal(x['margin_WL_Full_A3'],[.3-.5, .75])
        np.testing.assert_allclose(x['margin_WL_OriginalE'],[1.3,-3.])
        self.assertEqual(x['WL_EqualCount'].sum(),x['WL_Full_A1'].sum())

    def test_gate_uses_exact_required_comparisons_without_cal_teacher_as_condition(self):
        settings = dict(mutagenicity_edge='WL',mutagenicity_node='WL',proteins_edge='Size')
        cfg = dict(primary_anchors=settings,minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        rows = [dict(setting=s,anchor=a,right=m,accuracy_exact='1/200' if m!='CalTeacher' else '-1/2',
                     positive=7,tied=2,negative=3)
                for s,a in settings.items() for m in ('Anchor','WL','Size','B0Teacher','CalTeacher','ScoreOnly','OtherMean')]
        self.assertTrue(scoring.gate(rows,cfg)['continue_remaining_seeds'])
        p = next(r for r in rows if r['setting']=='proteins_edge' and r['right']=='Anchor')
        p['accuracy_exact']=str(Fraction(1,200)-Fraction(1,10**30))
        self.assertFalse(scoring.gate(rows,cfg)['continue_remaining_seeds'])

    def test_missing_frozen_predictions_do_not_open_labels_or_reserve_score(self):
        cfg = dict(phase1_seed=3405,settings=['proteins_edge'],domains=[0,1,2,3],
                   r7_final_digest='r7',structural_fit_sha256='fit',structural_obs_sha256='obs',r14_fit_sha256='r14')
        with TemporaryDirectory() as tmp, mock.patch.object(scoring.scoring,'read_h_cohorts') as reader:
            out = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                scoring.score(out,None,cfg,None,'bad',execution={})
            reader.assert_not_called()
            self.assertFalse((out/'score-execution.json').exists())
