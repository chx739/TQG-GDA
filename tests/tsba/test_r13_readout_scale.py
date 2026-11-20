"""Readout scaling, original role permissions and direction-weighted decision bounds."""
from hashlib import sha256
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import readout_scale as a
from tsba.encoder import GraphEncoder
from tsba.pretrain import SourceModel


class ReadoutScaleTests(unittest.TestCase):
    def test_native_interventions_start_from_same_sum_representation(self):
        torch.manual_seed(4)
        model=SourceModel(GraphEncoder(2,1,hidden=5,num_layers=2,out_dim=3,conv='gine'),2)
        model.eval().requires_grad_(False)
        model.encoder.freeze()
        nodes=torch.ones((3,2))
        edges=torch.tensor([[0,1,1,2],[1,0,2,1]])
        graph=Data(x=nodes,edge_index=edges,edge_attr=torch.ones((4,1)))
        before={k:v.clone() for k,v in model.state_dict().items()}
        record=a.interventions(model,[graph,graph],1,
          dict(Own=.5,Individual=np.array([.5,1.]),Other1=2.,Other2=1.))
        p=torch.from_numpy(record['P'])
        np.testing.assert_allclose(record['Own'],a.margin(a.native(model,p,.5)[0]),atol=5e-8,rtol=0.)
        np.testing.assert_array_equal(record['Individual'],[record['Own'][0],record['B0'][1]])
        np.testing.assert_array_equal(record['Other2'],record['B0'])
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def test_frozen_mask_uses_right_branch_at_zero(self):
        z=np.array([2.])
        u=np.array([[0.,0.,2.,-1.]])
        wp=np.array([[1.,-1.,-3.,4.]])
        w=np.array([1.,2.,3.,4.])
        np.testing.assert_array_equal(a.frozen_margin(z,u,wp,w,2.),[-6.])
        np.testing.assert_array_equal(a.frozen_margin(z,u,wp,w,0.),[13.])

    def test_fractional_medians_and_even_query_membership(self):
        setting='toy'
        lists=[[10+d*10+x for x in range(6)] for d in range(1,4)]
        tasks=[dict(target=i,ids=dict(source=[1,2,3],validation=[4,5],adapt=ids,holdout=[80+i]))
               for i,ids in enumerate(lists,1)]
        ranking=lambda xs: sorted(xs,key=lambda x:(sha256(f'r9-local-v1|{setting}|{x}'.encode()).hexdigest(),x))
        ids={str(t['target']):ranking(t['ids']['adapt']) for t in tasks}
        role=dict(S=[1,2,3],V=[4,5],**{str(i):v for i,v in enumerate(lists,1)})
        expected_read=dict(S=role['S'],V=ranking(role['V']),**{d:v[::2]+v[1::2][:2] for d,v in ids.items()})
        rec=dict(setting=setting,source=0,k=2,sha256_granted_role_ids={k:a.identity(v) for k,v in role.items()},
                 sha256_graph_ids_read={k:a.identity(v) for k,v in expected_read.items()})
        q,r,k=a.matched_roles(setting,0,tasks,rec)
        self.assertEqual(k,2)
        for d,v in ids.items():
            self.assertEqual(q[d],v[::2])
            self.assertEqual(r[d],v[1::2][:2])
        store=Mock()
        scope=a.GraphScope(store,q,r)
        for forbidden in ('H1','S','A1'):
            with self.assertRaises(PermissionError):
                scope.take(forbidden)
        store.take.assert_not_called()
        scope.take('Q1')
        store.take.assert_called_once_with(q['1'])
        self.assertEqual(a.median((1,2)),1.5)

    def test_spearman_is_midrank_and_constant_missing(self):
        self.assertIsNone(a.spearman(np.zeros(4),np.arange(4)))
        self.assertAlmostEqual(a.spearman([1,1,9,7],[0,0,2,1]),1.)

    def test_equal_positive_diagnostic_numeric_tie_and_positive_scaling(self):
        z=np.array([0.,1.,1.,-2.])
        arrays={p:z.copy() for p in a.ARMS if p!='EqualPositive'}
        arrays.update(Own=np.array([1.,0.,0.,-2.]),Proportional=2*z,ids=np.array([82,21,10,42]))
        a.summarize_direction(arrays)
        np.testing.assert_array_equal(arrays['action_EqualPositive'],[False,False,True,False])

    def test_equal_positive_is_action_only(self):
        z=np.array([-1.,0.,1.])
        arrays={p:z.copy() for p in a.ARMS if p!='EqualPositive'}
        arrays.update(Own=np.array([1.,1.,1.]),ids=np.arange(len(z)))
        row=next(r for r in a.summarize_direction(arrays) if r['arm']=='EqualPositive')
        self.assertEqual(row['positive_rate'],1.)
        self.assertEqual(row['decisions_different_from_own'],0)
        self.assertEqual(row['decisions_different_from_b0'],2)
        for field in ('median_abs_margin','p95_abs_margin','spearman_vs_b0','mean_abs_probability_change'):
            self.assertIsNone(row[field])

    def test_gate_is_equal_direction_and_foreign_means_remain_separate(self):
        plan=dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge'],domains=[0,1])
        cfg=dict(minimum_disagreement_fraction=[1,200])
        rows=[dict(setting=s,source=x,target=1-x,arm=m,n=200,decisions_different_from_own=1)
              for s in plan['settings'] for x in (0,1) for m in (*a.GATE[:-1],'Other1','Other2')]
        self.assertTrue(a.gate(rows,plan,cfg)['consider_classification'])
        for r in rows:
            if r['setting']=='proteins_edge' and r['arm']=='Other2':
                r['decisions_different_from_own']=0
        self.assertFalse(a.gate(rows,plan,cfg)['consider_classification'])
