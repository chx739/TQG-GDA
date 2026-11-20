from fractions import Fraction as F
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import torch

from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import (
    reverse_vote_scoring as score, reverse_votes as rules)


class ReverseVoteTests(TestCase):
    def test_unordered_edges_no_loop_and_edgeless_coordinate(self):
        x = torch.tensor([[1,0],[0,1],[1,0]])
        graph = SimpleNamespace(x=x,edge_index=torch.tensor([[0,1,1,0,2,0],[1,0,0,2,2,0]]))
        np.testing.assert_array_equal(rules.edge_histogram(graph,2),[1,1,0,0])
        graph.edge_index = torch.tensor([[0,1],[0,1]])
        np.testing.assert_array_equal(rules.edge_histogram(graph,2),[0,0,0,1])

    def test_exact_same_distance_with_different_denominators(self):
        v = np.array([[1,0]])
        a = np.array([[1,0],[1,2],[7,14],[3,4]])
        z = np.array([0,1,2,3])
        result = rules.reverse_weights(v,a,z,3)
        self.assertEqual(result,[(F(1,3),F(1,6),F(1,6),F(1,3))])
        self.assertEqual(result,rules.reverse_weights(v,a[::-1],z[::-1],3))
        self.assertEqual(rules.reverse_weights(v,a,z,8),[(F(1,4),)*4])

    def test_integer_distance_keeps_distinct_nearly_equal_uint64_rows(self):
        n = 10**9
        for dtype in (np.int64,np.uint64):
            v = np.array([[n,n+1]],dtype=dtype)
            a = np.array([[n,n+1],[n+1,n+2]],dtype=dtype)
            self.assertEqual(rules.reverse_weights(v,a,np.array([0,1]),1),[(F(1),F(0),F(0),F(0))])

    def test_selection_exact_brier_at_zero_mask_and_unobserved_cells(self):
        weights = [(F(1),F(0),F(0),F(0))]*3
        losses = rules.brier(weights,np.array([0,1,0],dtype=np.uint8))
        self.assertTrue(all(type(loss) is F for loss in losses))
        self.assertEqual(losses[0],F(1,3))
        self.assertEqual(losses[0],losses[12])
        self.assertEqual(rules.choose(losses,12)[0],12)
        self.assertEqual(rules.record(losses,12)['gap_exact'],'0')
        self.assertEqual(rules.choose([F(0) if m in (4,8) else F(1) for m in range(16)],12)[0],4)

    def test_reverse_selection_and_row_perm_use_different_losses(self):
        v = np.array([0,1])
        weights = [(F(1,4),F(3,4),F(0),F(0)),(F(3,4),F(1,4),F(0),F(0))]
        fit = rules.fit(v,v,v,weights,12)['fits']
        self.assertEqual(fit['Direct']['table'],14)
        self.assertEqual(fit['Reverse']['table'],13)
        self.assertEqual(F(fit['Reverse']['brier'][13]),F(1,16))
        self.assertEqual(fit['RowPerm']['table'],13)
        self.assertEqual(F(fit['RowPerm']['brier'][13]),F(5,16))

    def test_row_permutation_matches_enumeration(self):
        labels = [0,1,1]
        weights = [(F(0),F(1),F(0),F(0)),(F(1,3),F(1,3),F(0),F(1,3)),
                   (F(1,2),F(1,2),F(0),F(0))]
        expected = list(zip(*(rules.brier(weights,p) for p in permutations(labels)),strict=True))
        found = rules.brier(weights,labels,row_permutation=True)
        self.assertEqual(found,[sum(values)/len(values) for values in expected])

    def test_count_matched_shares_randomization_across_equal_fraction_cells(self):
        counts = [[1,0],[4,1],[0,1],[0,0]]
        h = [3,5,2,1]
        q = rules.count_matched(counts,h,5)
        self.assertEqual(q,[F(1,4),F(1,4),F(1),F(1)])
        self.assertEqual(sum(n*p for n,p in zip(h,q,strict=True)),5)
        cells = [[2,1],[4,1],[0,2],[0,1]]
        ref = [0,0,1,1]
        a,b = score.pairs(cells,ref,q)
        self.assertEqual(a,F(3,2))
        self.assertEqual(b,F(1,2))
        self.assertEqual(score.accuracy(score.confusion(cells,ref))-
                         score.accuracy(score.confusion(cells,q)),F(1,11))

    def test_swaps_apply_foreign_reverse_table_and_match_positives(self):
        def make(table):
            return dict(fits={m:{'table':table} for m in rules.LOCAL_ARMS},v_counts=[[1,1]]*4)
        fitted = rules.apply(make(7),[make(2),make(5)],[1,2,3,4],12)
        self.assertEqual(fitted['tables'],dict(Anchor=12,Reverse=7,Direct=7,PermMean=7,RowPerm=7,Other1=2,Other2=5))
        self.assertEqual(fitted['positive_counts']['CountMatched'],'6')
        self.assertEqual(fitted['policies']['CountMatched'],['3/5']*4)

    def test_invalid_obs_cannot_read_historical_final_or_create_score_receipt(self):
        with TemporaryDirectory() as tmp,mock.patch.object(score,'verify',side_effect=PermissionError), \
                mock.patch.object(score,'published_rows') as reader:
            path = Path(tmp)
            with self.assertRaises(PermissionError):
                score.score(path,path,{}, {},'invalid',execution={})
            reader.assert_not_called()
            self.assertFalse((path/'score-execution.json').exists())

    def test_gate_is_exact_and_excludes_row_permutation_and_individual_donors(self):
        cfg = dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge'],
                   minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        roles = {f'{s}-{i}':dict(setting=s) for s in cfg['settings'] for i in range(12)}
        endpoints = {k:{m:(F(101,200) if m in ('Reverse','Other1','RowPerm') else F(1,2),F(0))
                        for m in score.METHODS} for k in roles}
        self.assertTrue(score.gate(cfg,roles,endpoints)['advance'])
        for key in roles:
            endpoints[key]['PermMean'] = endpoints[key]['Reverse']
        self.assertFalse(score.gate(cfg,roles,endpoints)['advance'])

    def test_shared_oracle_maximizes_direction_equal_weight(self):
        roles,fits,cells,fixed = {},{},{},{}
        for target,n in ((1,2),(2,3),(3,100)):
            key = str(target)
            roles[key] = dict(setting='mutagenicity_edge',source=0,target=target,seed=3405,group='g')
            cells[key] = [[0,n] if target==3 else [n,0],[0,0],[0,0],[0,0]]
            fit = dict(fits={m:{'table':12} for m in rules.LOCAL_ARMS},v_counts=[[1,1]]*4)
            fits[key] = dict(deployment=rules.apply(fit,[fit,fit],[n,0,0,0],12))
            fixed[key] = {m:score.confusion(cells[key],rules.TABLES[12]) for m in score.previous.FIXED}
        _,_,ceilings,_ = score.score_blocks(roles,fits,cells,fixed)
        self.assertEqual([r['oracle_shared_table'] for r in ceilings],[12]*3)
        self.assertEqual(ceilings[-1]['oracle_direction_table'],13)
