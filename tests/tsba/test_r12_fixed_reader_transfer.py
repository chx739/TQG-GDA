from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import (
    fixed_reader_inputs as readers, fixed_reader_scoring as scoring, fixed_reader_transfer as screen,
    reader_ranges as rules)


class FixedReaderTests(TestCase):
    def test_global_size_cut_does_not_split_ties_and_uses_smaller_cut(self):
        self.assertIsNone(rules.size_cut(np.array([2,2,2])))
        self.assertEqual(rules.size_cut(np.array([1,2,3])),1)
        self.assertEqual(rules.size_cut(np.array([1,1,4,4,4,7])),1)
        self.assertEqual(rules.size_cut(np.array([1,1,4,4,7,7,7])),4)

    def test_graph_cache_reads_only_missing_graphs(self):
        graph = SimpleNamespace(x=None,edge_index=np.array([[0,1,1],[1,0,1]]),edge_attr=None)
        store = mock.Mock(graph_sha='graph',take=mock.Mock(return_value=[graph]))
        archive = readers.Archive.__new__(readers.Archive)
        archive.stores = {'mutagenicity_edge':('r5',store)}
        archive.cached = {'mutagenicity_edge':{10:('cached',{},3)}}
        ctx = {'root':'r5'}
        with mock.patch.object(readers.readouts,'node_frequency',return_value='fresh'), \
                mock.patch.object(readers.readouts,'topology_counts',return_value={}), \
                mock.patch.object(readers.wl,'normalize',return_value={}):
            nodes,_,edges = archive.graph_features('mutagenicity_edge',ctx,[10],[10,11],'graph')
            store.take.assert_not_called()
            self.assertEqual(nodes,{10:'cached'})
            np.testing.assert_array_equal(edges,[3])
            nodes,_,edges = archive.graph_features('mutagenicity_edge',ctx,[11,10],[10,11],'graph')
            store.take.assert_called_once_with([11])
            self.assertEqual(nodes,{11:'fresh',10:'cached'})
            np.testing.assert_array_equal(edges,[1,3])

    def test_gate_compares_other_targets_by_mean(self):
        roles = {k:{'setting':'toy'} for k in range(12)}
        points = {k:{m:(F(1,2),F(1,2)) for m in (*screen.ACTIONS,'OtherMean')} for k in roles}
        per_control = {m:(F(1,100),F(0),F(1),F(0),F(1)) for m in scoring.CONTROLS}
        per_control['Other1'] = (F(-1,100),F(0),F(0),F(1),F(1))
        per_control['Other2'] = (F(3,100),F(0),F(3),F(0),F(3))
        cfg = dict(settings=['toy'],minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        _,pairs,decision = scoring.summarize(roles,points,{k:per_control for k in roles},cfg)
        self.assertTrue(decision['advance'])
        self.assertEqual({r['right']:r['accuracy_pp'] for r in pairs if r['right'].startswith('Other')},
                         {'Other1':-1.,'Other2':3.,'OtherMean':1.})

    def test_size_hash_group_counts_match_inside_each_class_bin(self):
        ids=np.arange(8)
        y,u=np.array([0,0,1,1]*2),np.array([1]*4+[2]*4)
        x=np.array([0,1,2,3]*2)
        sizes=np.array([1,9,2,7,3,6,4,5])
        table,a,b=rules.source_tables(ids,y,u,x,sizes,'toy',0)
        self.assertEqual(table['cut'],4)
        np.testing.assert_array_equal(np.asarray(table['groups']['SizeRange']).sum(axis=-1),
                                      np.asarray(table['groups']['HashRange']).sum(axis=-1))
        for mask in (a,b):
            self.assertEqual(int(mask.sum()),4)
        one,*_=rules.source_tables(ids,y,u,x,np.ones(8,dtype=int),'toy',0)
        self.assertEqual(len(one['groups']['SizeRange']),1)
        self.assertFalse(any(rules.templates(one)[0]['missing']['SizeRange']))

    def test_affine_bounds_exactly_include_boundaries_and_detect_empty(self):
        self.assertEqual(rules.interval([[F(0),F(0)],[F(1),F(1)]],F(1,3)),(F(1,3),F(1,3)))
        self.assertEqual(rules.interval([[F(1),F(1)],[F(0),F(0)]],F(1,3)),(F(2,3),F(2,3)))
        self.assertEqual(rules.interval([[F(0),F(1)],[F(0),F(1)]],F(1,2)),(F(0),F(1)))
        self.assertIsNone(rules.interval([[F(0),F(1,4)],[F(0),F(1,4)]],F(1,2)))
        self.assertIsNone(rules.interval([[F(0),F(0)],[F(1),F(1)]],F(-1,10**30)))

    def test_point_clip_and_fallbacks_keep_exact_source_prior(self):
        ids=np.arange(8)
        y=np.array([0,1,0,1,0,0,0,1])
        u=np.array([1,1,2,2,3,3,4,4])
        x=np.array([0,3,1,1,0,1,0,3])
        tbl,*_=rules.source_tables(ids,y,u,x,np.ones(8,dtype=np.int64),'toy',2)
        a=[[0,2,0,0],[0,1,0,0],[1,0,0,0],[0,0,0,0]]
        result=rules.fit_rules(rules.templates(tbl),a)
        self.assertEqual(result[0]['rules']['SizeRange']['feasible'],['1/2','1/2'])
        self.assertEqual(result[0]['rules']['SizeRange']['shift'],0.)
        self.assertEqual([result[b]['rules']['Point']['reason'] for b in range(1,4)],
                         ['zero_contrast','missing_V_class','empty_A'])
        self.assertEqual(rules.shift(F(0),F(1,2),1),float(np.log(1/3)))
        self.assertEqual(rules.shift(F(1,5),F(1,5),100),0.)

    def test_cut_ties_go_left_and_probe_threshold_ties_are_negative(self):
        ids=[44,50]
        margin=np.array([0.,2.])
        cuts=np.array([0.,1.,2.])
        p=dict(L=dict(weights=[1.,0.],intercept=0.,threshold=1.),
               T=dict(weights=[1.],intercept=0.,threshold=.5,vocabulary=[[0,'61']]))
        a=readers.encode(ids,margin,cuts,p,{44:np.array([1.,0.]),50:np.array([0.,1.])},
                         {44:{(0,b'a'):.5},50:{(0,b'a'):.6}})
        np.testing.assert_array_equal(a['u'],[1,3])
        np.testing.assert_array_equal(a['x'],[0,1])

    def test_swaps_copy_donor_shift_and_equal_count_uses_base_order(self):
        h=dict(ids=np.array([9,3,5],dtype=np.int64),u=np.array([1,1,3],dtype=np.int64))
        role=dict(target=1,other_targets=[2,3])
        by_target={j:[dict(rules={name:dict(shift=j if i==1 else 0.) for name in rules.RULES}) for i in range(1,5)]
                   for j in (1,2,3)}
        a=screen.actions(role,h,{},np.array([-1.,-1.5,1.]),by_target)
        np.testing.assert_array_equal(a['SizeRange'],[0,0,1])
        np.testing.assert_array_equal(a['Other1'],[1,1,1])
        np.testing.assert_array_equal(a['margin_Other2'],[2,1.5,1])
        np.testing.assert_array_equal(a['EqualCount'],[0,0,1])

    def test_H_missing_class_has_zero_weight_but_contributes_to_prior_error(self):
        ids=np.arange(4); y=np.array([0,0,1,1]); u=np.ones(4,dtype=np.int64)
        table,*_=rules.source_tables(ids,y,u,np.array([0,0,3,3]),np.ones(4,dtype=int),'toy',1)
        src=dict(bins=rules.templates(table))
        fitted=rules.fit_rules(src['bins'],[[0,0,0,1],[0]*4,[0]*4,[0]*4])
        role=dict(setting='toy',source=1,target=2,seed=3405)
        h={'u':np.array([1,1]),'x':np.array([3,2])}
        cls,bins,err,drift,_=scoring.diagnose(role,h,np.ones(2,dtype=np.int64),src,fitted,[fitted,fitted])
        self.assertEqual(drift[0]['class0_weighted'],'0')
        self.assertEqual(drift[0]['class1_weighted'],'-1/2')
        self.assertEqual(drift[0]['H_moment_drift'],'-1/2')
        self.assertEqual(drift[0]['A_minus_H'],'1/2')
        self.assertEqual(cls[0]['status'],'missing_H_class')
        self.assertTrue(bins[0]['contains_p_H'])
        self.assertEqual(next(r['error'] for r in err if r['method']=='Anchor'),.5)

    def test_invalid_OBS_keeps_H_label_port_closed(self):
        with TemporaryDirectory() as tmp,mock.patch.object(screen,'observed',side_effect=PermissionError), \
                mock.patch.object(scoring,'h_labels') as read:
            out=Path(tmp)
            with self.assertRaises(PermissionError):
                scoring.score(out,out,{}, {},None,'bad',execution={})
            read.assert_not_called()
            self.assertFalse((out/'score-execution.json').exists())
