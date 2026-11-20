from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from scipy.sparse import csr_matrix
import torch

from experiments.mutagenicity.r15_新局部结构与图类别_20261121.src import structure
from experiments.mutagenicity.r17_目标图外围与分类一致性_20261121.src import readout, screen
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl

SETTINGS = ('mutagenicity_edge','mutagenicity_node','proteins_edge')
OPTIONS = dict(max_iterations=1000,gradient_tolerance=1e-10,function_tolerance=1e-15,stationarity_tolerance=1e-7)


def path_graph(types):
    edges = [(j,j+1) for j in range(len(types)-1)]
    edges += [(dst,src) for src,dst in edges]
    index = torch.tensor(edges,dtype=torch.int64).T
    return dict(x=torch.eye(3)[types],edge_index=index,edge_attr=None)


class PairedReadoutTests(TestCase):
    def test_nonperipheral_views_use_original_colors_not_induced_graph(self):
        graph = path_graph([0,1,2,0])
        f,l,n = readout.views(graph,'proteins_edge')
        self.assertEqual(n,2)
        typed = dict(graph,edge_attr=torch.ones(6,1))
        encoded = structure.encode(typed,dict(node_types=3,edge_types=1))
        expected = wl.normalize([{layer[1]:1,layer[2]:1} for layer in encoded.layers])
        self.assertEqual(f,wl.normalize(encoded.counts))
        self.assertEqual(l,expected)
        self.assertNotEqual(l,readout.views(path_graph([1,2]),'proteins_edge')[0])
        _,_,n2 = readout.views(path_graph([0,1]),'proteins_edge')
        f2,l2,_ = readout.views(path_graph([0,1]),'proteins_edge')
        self.assertEqual(f2,l2)
        self.assertEqual(n2,0)

    def test_relabel_node_or_store_edges_in_reverse_order_keeps_views(self):
        graph = path_graph([2,1,0,2,1])
        order = torch.tensor([3,0,4,2,1])
        idx = torch.argsort(order)
        permuted = dict(x=graph['x'][order],edge_index=idx[graph['edge_index'].flip(1)],edge_attr=None)
        self.assertEqual(readout.views(graph,'proteins_edge'),readout.views(permuted,'proteins_edge'))

    def test_forward_shift_counts_each_local_reference_once_after_projection(self):
        full = {1:{'x':1.},2:{'x':2.},3:{'x':3.},4:{'x':4.}}
        local = {1:{'x':1.},2:{'x':20.,'outside':4.},3:{'x':30.},4:{'x':40.}}
        v = readout.differences(full,local,[2,3,4],[2,3,4],{'x':0})
        np.testing.assert_array_equal(v['TargetPair'].toarray().ravel(),[-18,-27,-36])
        np.testing.assert_array_equal(v['Shuffled'].toarray().ravel(),[-28,-37,-16])
        self.assertEqual(readout.geometry(v['Shuffled'])['T'],28**2+37**2+16**2)

    def test_gradient_and_trace_normalization_match_dense_formula(self):
        x = csr_matrix(np.array([[1.,2.],[1.,-.5],[-1.,0.]]))
        y = np.array([0,1,1])
        v = csr_matrix(np.array([[1.,-1.],[2.,-3.]]))
        t = readout.geometry(v)['T']
        p = np.array([.2,-.1,.7])
        value,grad = readout.objective(x,y,v,t,p)
        for i in range(len(p)):
            delta = np.eye(len(p))[i]*1e-6
            lo = readout.objective(x,y,v,t,p-delta)[0]
            hi = readout.objective(x,y,v,t,p+delta)[0]
            self.assertAlmostEqual(grad[i],(hi-lo)/2e-6,places=8)
        self.assertEqual(readout.geometry(v)['normalized_trace'],1.)
        shifted = readout.objective(x,y,v*3,t*9,p)
        self.assertAlmostEqual(value,shifted[0],places=14)
        np.testing.assert_allclose(grad,shifted[1],atol=1e-14)

    def test_zero_trace_exact_fallback_and_nonzero_arm_certification(self):
        x = csr_matrix(np.array([[1.,0.],[2.,0.],[-1.,0.],[-2.,0.]]))
        y = np.array([1,1,0,0])
        base = dict(weights=[.6,0.],intercept=0.,gradient_inf=1e-8,training_objective=.4)
        z = csr_matrix((2,2))
        model,note = readout.fit(x,y,z,readout.geometry(z),base,OPTIONS)
        self.assertIs(model,base)
        self.assertEqual(note['fallback'],'zero_trace')
        v = csr_matrix([[1.,0.]])
        model,note = readout.fit(x,y,v,readout.geometry(v),base,OPTIONS)
        self.assertLess(note['gradient_inf'],1e-7)
        self.assertEqual(model['weights'][1],0.)
        self.assertEqual(note['fallback'],None)

    def test_grant_cannot_open_C_U_or_E_during_fit(self):
        role = dict(setting='toy',status='ok',roles=dict(D=[1,2],C=[3],U=[4],E=[5]))
        g = screen.fit_roles({'task':role})
        with screen.data.ScopedLabels({},g,'fit') as port, port.direction('task'):
            for k,i in (('C',3),('U',4),('E',5)):
                with self.assertRaises(PermissionError):
                    port.load('task',k,[i])
        with self.assertRaises(PermissionError):
            screen.validate_fit_reads({'task':role},{'task':{'D':[1,2],'C':[3]}})

    def test_disagreement_bounds_and_exact_joint_gate(self):
        roles,preds = {},{}
        for s in SETTINGS:
            for source in range(4):
                for d in range(2):
                    key = s,source,d
                    roles[key] = dict(setting=s,source=source)
                    preds[key] = {m:np.zeros(200,dtype=int) for m in screen.METHODS}
                    preds[key]['e_ids'] = np.arange(200)
                    preds[key][screen.MAIN][0] = int(source<3)
        bound=screen.action_bounds(roles,preds)
        wl_bound=bound['settings']['proteins_edge']['comparisons'][screen.BASE]
        self.assertEqual(wl_bound['max_positive_directions'],6)
        self.assertEqual(wl_bound['max_positive_source_groups'],3)
        self.assertEqual(Fraction(wl_bound['upper_exact']),Fraction(3,800))
        self.assertTrue(bound['cannot_advance_by_pairwise_bounds'])
        rows = [dict(setting=s,source=None,right=m,accuracy_exact='1/200')
                for s in SETTINGS for m in screen.CONTROLS]
        rows += [dict(setting=s,source=i,right=screen.BASE,accuracy_exact='1/200' if i!=3 else '0')
                 for s in SETTINGS for i in range(4)]
        self.assertTrue(screen.gate(rows)['continue_real_development'])
        p=next(r for r in rows if r['setting']=='proteins_edge' and r['right']=='Shuffled')
        p['accuracy_exact']='0'
        self.assertFalse(screen.gate(rows)['continue_real_development'])

    def test_missing_OBS_does_not_reserve_score_or_open_E(self):
        cfg=dict(source_obs_sha256='s',geometry_obs_sha256='g')
        fake=SimpleNamespace(reference=SimpleNamespace(roles={}))
        with TemporaryDirectory() as tmp, mock.patch.object(screen.data,'ScopedLabels') as port:
            out=Path(tmp)
            with self.assertRaises(FileNotFoundError):
                screen.score(out,cfg,fake,'broken',execution={})
            port.assert_not_called()
            self.assertFalse((out/'score-execution.json').exists())
