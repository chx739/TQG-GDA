from fractions import Fraction
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    residual_inputs as data, residual_models as models, true_rate_replay as replay)


def fixture():
    x = dict(e_ids=np.arange(10,14),u_ids=np.arange(4),wl_u=np.array([-1.,-1.,1.,2.]),
             wl_e=np.array([-3.,-2.,-1.,4.]),size_u=np.full(4,-1.),size_e=np.full(4,-1.),
             WL=np.array([0,0,0,1]))
    counts = {('key','U',0):(4,1),('key','E',0):(4,3)}
    role = dict(setting='fixture',status='ok',roles=dict(D=[4],C=[9],U=x['u_ids'].tolist(),E=x['e_ids'].tolist()))
    return replay.Reference({},dict(key=role),{},counts,dict(key=x))


class TrueRateReplayTests(TestCase):
    def test_three_axes_follow_their_registered_pools_and_respect_ties(self):
        ref = fixture()
        actions, strata = replay.action_rules('key',ref)
        rules = strata[0]['rules']
        self.assertEqual(rules['A']['rate_exact'],'1/4')
        self.assertEqual(rules['B']['rate_exact'],'3/4')
        self.assertEqual(rules['B']['threshold_pool_positives'],2)
        self.assertEqual(rules['B']['threshold'],-1.)
        self.assertEqual(rules['C']['threshold_pool'],'E')
        self.assertEqual(rules['C']['threshold'],-3.)
        np.testing.assert_array_equal(actions['A'],[0,0,0,1])
        np.testing.assert_array_equal(actions['B'],[0,0,0,1])
        np.testing.assert_array_equal(actions['C'],[0,1,1,1])

    def test_empty_u_uses_original_wl_even_for_e_rank_arm(self):
        ref = fixture()
        ref.predictions['key']['size_e'] = np.ones(4)
        ref.counts = {('key','U',0):(4,1),('key','E',1):(4,3)}
        actions,strata = replay.action_rules('key',ref)
        for m in replay.RULES:
            np.testing.assert_array_equal(actions[m],actions['WL'])
            self.assertEqual(strata[1]['rules'][m]['status'],'fallback_WL_empty_U')
            self.assertIsNone(strata[1]['rules'][m]['threshold'])

    def test_counts_must_cover_every_original_method_and_agree(self):
        ref = fixture()
        names = ['setting','source','direction','seed','pool','size_action','method','n','positive']
        def payload(rows):
            stream=io.StringIO()
            writer=csv.writer(stream)
            writer.writerow(names)
            writer.writerows(rows)
            return stream.getvalue().encode()
        identity = ['fixture',0,'small_to_large',3405,'U',0]
        records = [identity+[m,4,1] for m in models.ESTIMATORS]
        key = data.key('fixture',0,'small_to_large',3405)
        roles = {key:ref.roles['key']}
        self.assertEqual(replay.published_counts(payload(records),roles),{(key,'U',0):(4,1)})
        with self.assertRaises(PermissionError):
            replay.published_counts(payload(records[:-1]),roles)
        records[-1][-1]=3
        with self.assertRaises(PermissionError):
            replay.published_counts(payload(records),roles)

    def test_freeze_and_missing_prediction_block_precede_scoring(self):
        ref = fixture()
        cfg = dict(source_obs_sha256='o',source_final_sha256='f')
        with TemporaryDirectory() as directory,mock.patch.object(replay.data,'Archive') as archive:
            out=Path(directory)
            write_json(out/'executed-config.json',dict(fixture=True))
            obs = replay.freeze(out,cfg,ref)
            replay.verify_actions(out,cfg,ref,obs)
            archive.assert_not_called()
            (out/'predictions/key.npz').write_bytes(b'tampered')
            with self.assertRaises(PermissionError):
                replay.score(out,None,cfg,ref,obs)
            archive.assert_not_called()

    def test_diagnostic_reader_has_e_permission_only(self):
        ref = fixture()
        port=data.ScopedLabels({},ref.roles,'replay_E')
        with mock.patch.object(port,'label_file') as opener,port.direction('key'):
            for role in ('D','C','U'):
                with self.assertRaises(PermissionError):
                    port.load('key',role,ref.roles['key']['roles'][role])
            opener.assert_not_called()

    def test_accuracy_and_macro_f1_use_direction_equal_weight(self):
        ref=fixture()
        ref.plan={'settings':['fixture']}
        ref.roles={str(i):ref.roles['key'] for i in range(8)}
        values={(str(i),arm):(Fraction(3,5) if arm=='A' else Fraction(1,2),Fraction(i,10))
                for i in range(8) for arm in replay.METHODS}
        pairs={(str(i),l,r):dict(corrected=1,harmed=1,net=0,changed=2)
               for i in range(8) for l,r in replay.PAIRS}
        summary,contrasts,checks=replay.summaries(values,pairs,ref,
            dict(published_a_accuracy_ranges_pp={'fixture':['58.00','62.21']}))
        self.assertEqual(next(r for r in summary if r['method']=='A')['macro_f1'],.35)
        self.assertEqual(next(r for r in contrasts if (r['left'],r['right'])==('A','WL'))['accuracy_pp'],10.)
        self.assertTrue(checks['fixture']['within'])
        with self.assertRaises(ArithmeticError):
            replay.summaries(values,pairs,ref,dict(published_a_accuracy_ranges_pp={'fixture':['62','65']}))
