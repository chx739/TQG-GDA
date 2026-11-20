from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import (
    conditional_rank as rank, conditional_rank_observation as obs, conditional_rank_scoring as scoring)


class ConditionalRankTests(TestCase):
    def test_max_correct_never_splits_ties_and_prefers_anchor(self):
        g = np.array([2.,2.,1.,0.])
        y = np.array([0,1,1,0],dtype=np.uint8)
        result = rank.candidates(g,y,0)
        self.assertEqual(result,dict(n=4,k=3,candidates=[[0,2],[2,2],[3,3],[4,2]]))
        same = np.array([2.,2.])
        self.assertEqual(rank.candidates(same,np.array([1,0]),0)['k'],0)
        self.assertEqual(rank.candidates(same,np.array([1,0]),1)['k'],2)

    def test_midrank_equality_and_absent_source_or_reference(self):
        v = (np.array([0.,2.,1.]),np.array([0,0,1]))
        y = np.array([0,1,1])
        rules = rank.fit(*v,y,(0,0,1,1))
        self.assertEqual([r['k'] for r in rules],[1,1,None,None])
        empty = (np.array([],dtype=float),np.array([],dtype=np.int64))
        h = (np.array([1.,0.,2.,-99.,-4.]),np.array([0,0,0,1,2]))
        got = rank.predict(rules,v,empty,h,(0,0,1,1))
        np.testing.assert_array_equal(got,[0,0,1,1,1])
        np.testing.assert_array_equal(rank.predict(rules,v,empty,h,(0,0,1,1),sign_mix=True),got)
        self.assertEqual(rank.compare(1,2,rules[0],1),1)

    def test_sign_mix_preserves_signs_while_changing_continuous_reference(self):
        v = (np.array([1.,2.]),np.array([0,0]))
        q = rank.fit(*v,np.array([0,1]),(0,0,1,1))
        a = (np.array([-2.,1.,10.,0.]),np.array([0,0,1,2]))
        h = (np.array([1.,1.]),np.array([0,2]))
        # In cell 0, own CDF is (2+1)/4. SignMix gives
        # (1/2)*1 + (1/2)*(1/4) = 5/8. At threshold 5/8 the anchor wins.
        q[0] = dict(n=8,k=3,candidates=[])
        np.testing.assert_array_equal(rank.predict(q,v,a,h,(0,0,1,1)),[1,1])
        np.testing.assert_array_equal(rank.predict(q,v,a,h,(0,0,1,1),sign_mix=True),[0,1])

    def test_deployment_swaps_whole_pool_and_randomizes_tie_at_B2_cut(self):
        v = (np.array([0.,2.]),np.array([0,0]))
        q = rank.fit(*v,np.array([0,1]),(0,0,1,1))
        base = dict(g=np.array([1.,3.,4.,5.]),z=np.zeros(4,dtype=int),b2_margin=np.array([2.,1.,1.,1.]),
                    ids=np.array([9,2,1,5]),WL=np.zeros(4,dtype=int))
        a = v
        foreign = [(np.array([2.,4.]),v[1]),(np.array([1.,2.]),v[1])]
        role = dict(setting='mutagenicity_edge',target=1)
        pred = obs.deployment(role,base,q,v,a,foreign)
        np.testing.assert_array_equal(pred['OwnRank'],[0,1,1,1])
        np.testing.assert_array_equal(pred['Other1'],[0,0,1,1])
        np.testing.assert_array_equal(pred['Other2'],[0,1,1,1])
        p = scoring.expectation(pred['CountMatched_num'],pred['CountMatched_den'])
        self.assertEqual(p,[F(1),F(2,3),F(2,3),F(2,3)])
        c = scoring.expected_confusion(p,np.array([1,0,1,0]))
        self.assertEqual(c,(F(2,3),F(4,3),F(1,3),F(5,3)))
        self.assertEqual(scoring.metrics(c,4)[0],F(7,12))

    def test_half_split_permutation_invariant_and_empty_supervised_cell(self):
        ids = np.array([8,1,2,91,22])
        split = rank.half_split('toy',3,ids)
        np.testing.assert_array_equal(split,rank.half_split('toy',3,ids[::-1])[::-1])
        self.assertEqual(int(split.sum()),2)
        g,z,y,halves = np.array([1.,2.,3.]),np.array([0,2,2]),np.array([0,1,1]),np.array([0,1,1])
        got = rank.supervised(g,z,y,halves,(0,0,1,1))
        for a in got.values():
            np.testing.assert_array_equal(a,[0,1,1])

    def test_pair_concordance_ties_and_class_coverage(self):
        g = np.array([0.,2.,2.,1.,3.])
        z,y = np.array([0,0,0,2,3]),np.array([0,0,1,1,0])
        self.assertEqual(rank.concordance(g,z,y,4),dict(pairs=2,credit_twice=3,covered_0=2,covered_1=1))
        empty = rank.concordance(g,np.arange(5),y,5)
        self.assertEqual(empty['pairs'],0)

    def test_gate_uses_exact_boundary_and_averaged_donors(self):
        cfg = dict(settings=['mutagenicity_edge','mutagenicity_node','proteins_edge'],
                   minimum_accuracy_fraction=[1,200],minimum_positive_directions=7)
        rows = [dict(setting=s,left='OwnRank',right=m,accuracy_exact='0' if m=='Other1' else '1/200')
                for s in cfg['settings'] for m in scoring.CONTROLS for _ in range(12)]
        self.assertTrue(scoring.gate(cfg,rows)['advance'])
        for r in rows:
            if r['right']=='SourceRank':
                r['accuracy_exact'] = '0'
        self.assertFalse(scoring.gate(cfg,rows)['advance'])

    def test_bad_observation_never_opens_h_labels(self):
        with TemporaryDirectory() as tmp,mock.patch.object(obs,'observed',side_effect=PermissionError), \
                mock.patch.object(scoring,'h_labels') as reader:
            p = Path(tmp)
            with self.assertRaises(PermissionError):
                scoring.score(p,None,{},'bad',execution={})
            reader.assert_not_called()
            self.assertEqual(list(p.iterdir()),[])

    def test_score_can_retry_after_partial_writes(self):
        def interrupted(stage,*args):
            (stage/'half-diagnostic').mkdir()
            (stage/'partial.csv').write_text('incomplete')
            raise RuntimeError('interrupted')

        def completed(stage,*args):
            (stage/'FINAL.json').write_text('complete')
            return 'digest'

        with TemporaryDirectory() as tmp,mock.patch.object(obs,'observed',return_value=({},{})), \
                mock.patch.object(scoring,'evaluate',side_effect=interrupted):
            p = Path(tmp)/'attempt'
            p.mkdir()
            blob = b'frozen'
            (p/'OBS.json').write_bytes(blob)
            sha = sha256(blob).hexdigest()
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                scoring.score(p,None,{},sha,execution={})
            self.assertFalse((p/'score').exists())
            self.assertEqual(list(p.parent.iterdir()),[p])
            with mock.patch.object(scoring,'evaluate',side_effect=completed):
                self.assertEqual(scoring.score(p,None,{},sha,execution={}), 'digest')
            self.assertEqual((p/'score'/'OBS.json').read_bytes(),blob)
            self.assertEqual((p/'score'/'FINAL.json').read_text(),'complete')
