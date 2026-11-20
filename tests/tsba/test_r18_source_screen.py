from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r18_跨图同类关系与分类纠错_20261121.src import anchors, relations, screen, scoring

OPT = dict(max_iterations=1000,gradient_tolerance=1e-10,function_tolerance=1e-15,stationarity_tolerance=1e-7)
SETTINGS = ('mutagenicity_edge','mutagenicity_node','proteins_edge')


def features():
    c = 1/np.sqrt(3)
    return {
        1:({(0,b'a'):c, (1,b'new'):c, (2,b'new2'):c},np.array([1.,2.])),
        2:({(0,b'a'):c, (1,b'new'):c, (2,b'other'):c},np.array([3.,6.])),
        3:({(0,b'b'):c, (1,b'old'):c, (2,b'old'):c},np.array([.5,.5])),
        4:({(0,b'b'):c, (1,b'old'):c, (2,b'old'):c},np.array([.6,.4])),
        5:({(0,b'b'):c, (1,b'old'):c, (2,b'old'):c},np.array([.6,.3])),
        6:({(0,b'b'):c, (1,b'old'):c, (2,b'old'):c},np.array([.1,.2])),
    }


class R18SourceScreenTests(TestCase):
    def test_product_voting_is_exact_identity_for_any_reference_distribution(self):
        mi = relations.signed_margin(np.array([-.2,8.,0.,1e9,-1e9]))
        mj = relations.signed_margin(np.array([-.7,.1,3.]))
        p = relations.product(mi[:,None],mj)
        np.testing.assert_array_equal(mi,relations.apply_vote(mi,mj,p,p))
        self.assertTrue(np.all((p>0)&(p<1)))
        q = p.copy()
        q[0,0]=.9
        vote = relations.apply_vote(mi,mj,p,q)
        hand = (mi[0]+np.mean((2*q[0]-1)*mj))/(1+np.mean(mj**2))
        self.assertAlmostEqual(vote[0],hand,places=15)
        perm,pure=relations.other_votes(mi,mj,np.full_like(p,.5))
        np.testing.assert_array_equal(pure,mi)
        np.testing.assert_allclose(perm,mi/(1+np.mean(mj**2)))

    def test_symmetric_geometry_retains_words_not_in_fitting_D(self):
        geom=relations.Geometry.build(features())
        x=relations.columns(geom,[1],[2],np.array([.3]),np.array([-.7]))
        reverse=relations.columns(geom,[2],[1],np.array([-.7]),np.array([.3]))
        np.testing.assert_array_equal(x,reverse)
        np.testing.assert_allclose(x[0,3:],[1.,1.,0.,2.,4.],atol=1e-14)

    def test_fold_weights_and_skipping_singletons(self):
        g=relations.Geometry.build(features())
        ids=[[1,2],[3,4,5],[6],[]]
        y={1:1,2:0,3:1,4:1,5:0,6:1}
        a={i:(i-3)/2 for i in y}
        x,p,reply,w=relations.training_pairs(ids,a,y,g)
        np.testing.assert_array_equal(reply,[0,1,0,0])
        np.testing.assert_allclose(w,[.4,.2,.2,.2],atol=1e-15)
        self.assertEqual((x.shape,p.shape),((4,8),(4,)))
        self.assertEqual(relations.folds([1,2,3,4,5,6],'toy',2),
                         relations.folds([6,5,4,3,2,1],'toy',2))
        self.assertEqual(set(sum(relations.folds([1,2,3,4,5,6],'toy',2),[])),set(a))
        empty=relations.relation_head(np.empty((0,3)),np.empty(0),np.empty(0),np.empty(0),1,OPT)
        np.testing.assert_array_equal(relations.relation_probability(p,x[:,:3],empty),p)

    def test_calibration_respects_zero_slope_and_head_fits_intercept_with_penalty(self):
        a=np.array([-12.,-8.,8.,12.]); y=np.array([1,1,0,0])
        cal=relations.calibrator(a,y,OPT)
        self.assertEqual(cal['coefficients'][1],0.)
        x=np.ones((4,3))
        p=np.full(4,.5)
        out=relations.relation_head(x,p,np.ones(4,dtype=int),np.full(4,.25),4,OPT)
        self.assertTrue(np.all(np.asarray(out['scale'])==1.))
        self.assertLess(out['gradient_inf'],1e-7)
        self.assertGreater(out['coefficients'][0],0.)
        self.assertEqual(out['coefficients'][1:],[0.,0.,0.])

    def test_anchor_selection_checks_correct_nll_C_then_registered_method_order(self):
        def branch(C,nll=.5):
            points=[dict(C=c,correct=(3 if c==C else 2),nll=nll) for c in (.1,1.,10.,100.)]
            return dict(selected_C=C,validation=points)
        candidates={n:branch(10.) for n in anchors.MODELS}
        f=dict(status='ok',candidates=candidates)
        self.assertEqual(anchors.select(f),('WL',10.))
        candidates['Base']=branch(1.)
        self.assertEqual(anchors.select(f),('Base',1.))
        candidates['Size']=branch(100.,nll=.4)
        self.assertEqual(anchors.select(f),('Size',100.))

    def test_cross_fit_never_supplies_either_held_endpoint_label(self):
        ids=dict(D=[1,2,3,4],C=[5,6],U=[7],E=[8])
        calls=[]
        def fit(_,train,cal,y_train,y_cal,options):
            calls.append((train,cal,y_train.tolist(),y_cal.tolist()))
            return dict(status='single_class_D',constant=0.)
        with mock.patch.object(anchors,'fit',side_effect=fit):
            z,receipt=anchors.cross_fit({},ids,[[1,2],[3,4],[],[]],
                                         np.array([0,0,1,1]),np.array([1,0]),OPT)
        self.assertEqual(calls,[([3,4],[5,6],[1,1],[1,0]),([1,2],[5,6],[0,0],[1,0])])
        self.assertEqual(receipt[0]['held_ids'],[1,2])
        self.assertEqual(z,{i:0. for i in range(1,5)})

    def test_brier_uses_each_unordered_probability_and_joint_gate_is_strict(self):
        self.assertEqual(scoring.brier_exact(np.array([.25,.75]),np.array([1,0])),Fraction(9,16))
        rows=[dict(setting=s,source=None,right=m,accuracy_exact='1/200') for s in SETTINGS for m in screen.CONTROLS]
        rows.extend(dict(setting=s,source=i,right=screen.BASE,accuracy_exact='1/200' if i!=3 else '0')
                    for s in SETTINGS for i in range(4))
        pairs=[dict(setting=s,source=None,method=m,brier_exact='1/5' if m=='Full' else '1/4')
               for s in SETTINGS for m in relations.PAIR_ARMS]
        self.assertTrue(scoring.gate(rows,pairs)['continue_real_development'])
        next(r for r in pairs if r['setting']=='proteins_edge' and r['method']=='CalProduct')['brier_exact']='1/5'
        self.assertFalse(scoring.gate(rows,pairs)['continue_real_development'])

    def test_missing_OBS_cannot_reserve_score_or_open_E(self):
        cfg=dict(source_obs_sha256='s',geometry_obs_sha256='g')
        prep=SimpleNamespace(original=SimpleNamespace(reference=SimpleNamespace(roles={})))
        with TemporaryDirectory() as tmp,mock.patch.object(scoring.data,'ScopedLabels') as read:
            out=Path(tmp)
            with self.assertRaises(FileNotFoundError):
                scoring.score(out,cfg,prep,'bad',execution={})
            read.assert_not_called()
            self.assertFalse((out/'score-execution.json').exists())
