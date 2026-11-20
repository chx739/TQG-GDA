"""Synthetic accuracy threshold, matched scope, and paired bootstrap regressions."""
from fractions import Fraction
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, run, thresholds


class ThresholdTests(unittest.TestCase):
    def test_equal_margins_move_together_and_reference_resolves_ties(self):
        x=np.array([.25,.25])
        y=np.array([0,1])
        result=thresholds.choose(x,y,.3)
        self.assertEqual(result['threshold'],.3)
        self.assertEqual(result['correct'],1)
        np.testing.assert_array_equal(thresholds.decisions(x,.25),[False,False])

    def test_all_one_and_all_zero_strict_thresholds(self):
        x=np.array([-2.,1.])
        p=thresholds.choose(x,np.array([1,1]),0.)
        n=thresholds.choose(x,np.array([0,0]),0.)
        self.assertEqual(p['threshold'],np.nextafter(-2.,-np.inf))
        self.assertEqual(n['threshold'],1.)
        self.assertEqual((p['correct'],n['correct']),(2,2))

    def test_equal_distance_secondary_tie_picks_smaller_threshold(self):
        # Reference lies in a lower-scoring gap with equally close candidates.
        step=np.nextafter(0.,1.)
        x=np.array([-step,2*step])
        y=np.array([1,0])
        result=thresholds.choose(x,y,0.)
        self.assertEqual(result['threshold'],-2*step)
        self.assertEqual(result['correct'],1)

    def test_h_upper_matches_independent_fraction_enumeration(self):
        pools=[(np.array([-1.,0.]),np.array([1,1])),
               (np.array([-1.,1.,3.]),np.array([0,0,1])),
               (np.array([0.,0.,1.,2.]),np.array([1,1,0,0]))]
        counts,separate,shared=thresholds.h_upper_bounds(pools)
        grid=np.array([-2.,-1.,0.,1.,2.,3.])
        directly=[[int(((x>t)==y).sum()) for t in grid] for x,y in pools]
        self.assertEqual(counts,[max(z) for z in directly])
        use=lambda c:sum(Fraction(count,len(x)) for count,(x,_) in zip(c,pools))/len(pools)
        self.assertEqual(separate,use(counts))
        self.assertEqual(shared,max(use(c) for c in zip(*directly)))
        self.assertGreater(separate,shared)

    def test_same_source_crossed_a_never_reads_h_or_crosses_selected_ids(self):
        groups={}
        for target,m in ((1,4),(2,3),(3,5)):
            groups[target]=dict(task=dict(source=0,target=target,seed=3405),
              ids={'adapt':list(range(10*target,10*target+m)),'holdout':[100+target]},
              adapt_port=target,margins={'adapt':np.linspace(-1,1,m)})
        v=Mock()
        v.load.return_value=np.array([0,1])
        a=Mock()
        a.load.side_effect=lambda port,ids: np.asarray(ids)%2
        _,models,access=run.fit_group('generated',{'margin':np.array([-1.,1.]),'v_port':v},groups,a)
        v.load.assert_called_once_with()
        self.assertEqual(set(models),{1,2,3})
        for call in a.load.call_args_list:
            target,ids=call.args
            self.assertEqual(ids,thresholds.subset_ids('generated',groups[target]['ids']['adapt'],3))
            self.assertEqual(access[str(target)],ids)
        a_port=inputs.RawLabelPort('adapt',Path('/unread'),'',(10,),{10:(0,2)},{'0':0,'1':1})
        h_port=inputs.RawLabelPort('holdout',Path('/unread'),'',(20,),{20:(2,2)},{'0':0,'1':1})
        with inputs.GraphLabelSession('adapt') as session:
            with self.assertRaises(PermissionError):
                session.load(a_port,[20])
            with self.assertRaises(PermissionError):
                session.load(h_port,[20])

    def test_paired_other_mean_is_formed_inside_each_draw(self):
        states=dict(B0=np.array([0,1]),SourceV=np.array([1,0]),OwnA=np.array([1,1]),
                    OtherA1=np.array([0,1]),OtherA2=np.array([1,0]))
        weights=np.array([[1.,0.],[.3,.7],[0.,1.]])
        curves=run.paired_changes(states,weights)
        indices=[run.PAIRS.index(('OwnA',name)) for name in ('OtherA1','OtherA2','OtherMean')]
        np.testing.assert_allclose(curves[:,indices[2]],curves[:,indices[:2]].mean(1))

    def test_no_new_obs_no_h_decoder(self):
        with TemporaryDirectory() as tmp, patch.object(inputs,'GraphLabelSession') as label_reader:
            with self.assertRaises(FileNotFoundError):
                run.score(Path(tmp),Path(tmp),'invalid',{}, {})
            label_reader.assert_not_called()

    def test_unsigned_labels_never_underflow_unit_counts(self):
        x=np.array([-1.,1.])
        y=np.array([0,1],dtype=np.uint8)
        result=thresholds.choose(x,y,0.)
        self.assertEqual(result['correct'],2)
        self.assertEqual(result['threshold'],0.)
        self.assertEqual(thresholds.h_upper_bounds([(x,y)]*3),([2,2,2],Fraction(1),Fraction(1)))

    def test_float64_distance_resolves_subnormal_gap_exactly(self):
        x=np.array([0.,2.])
        result=thresholds.choose(x,np.array([1,0]),1.)
        self.assertEqual(result['threshold'],2.)

    def test_label_session_seeks_only_granted_valid_records(self):
        with TemporaryDirectory() as tmp:
            raw=Path(tmp)/'labels.txt'
            payload=b'1\nFORBIDDEN\n0\n'
            raw.write_bytes(payload)
            port=inputs.RawLabelPort('adapt',raw,inputs.sha(payload),(10,12),{10:(0,2),12:(12,2)},{'0':0,'1':1})
            with inputs.GraphLabelSession('adapt') as session:
                np.testing.assert_array_equal(session.load(port,[12]),[0])
                np.testing.assert_array_equal(session.load(port,[10]),[1])
