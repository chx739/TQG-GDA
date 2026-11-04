import unittest
import torch
from tqg_gda.mechanism_diagnostics import LinearBinary,linear_pool
from tqg_gda.p0_diagnostics import unlabeled_geometry
from tqg_gda.query_label_diagnostics import label_prediction_vectors,label_prediction_steps,query_order,query_lambda,duplicate_check
from tqg_gda.d34_diagnostics import kernel_family

class QueryLabelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(507)
    def geometry(self):
        source,_,labels=linear_pool(20,546);target,_,_=linear_pool(8,548,target=True)
        return unlabeled_geometry(LinearBinary([.3,-.6]),source,target),labels
    def test_base_ignores_query_pairings_and_full_reconstructs(self):
        geo,labels=self.geometry()
        v,_,stats=label_prediction_vectors(geo,labels,551,[9101,9102,9103],[9201,9202,9203],1e-12)
        moved,_,_=label_prediction_vectors(geo,labels.roll(1),551,[9101,9102,9103],[9201,9202,9203],1e-12)
        torch.testing.assert_close(v['BASE'],moved['BASE'],atol=1e-13,rtol=1e-12)
        true,_=geo.gradient(labels)
        torch.testing.assert_close(v['FULL'],true)
        torch.testing.assert_close(v['FULL'],v['GLOBAL']+v['LABEL']+v['PRED'])
        self.assertGreaterEqual(stats['cancellation_ratio'],0.)
    def test_only_label_increment_is_scaled_in_primary_nulls(self):
        geo,labels=self.geometry();v,_,_=label_prediction_vectors(geo,labels,551,[9101,9102,9103],[9201,9202,9203],1e-12)
        gs=torch.ones_like(v['FULL']);ga=gs.flip(0)
        steps,valid=label_prediction_steps(v,gs,ga,.1,1e-12)
        self.assertEqual(len(steps),19)
        for kind in ['PL','RL']:
            for rep in [1,2,3]:
                increment=steps[f'{kind}-{rep}']-steps['A']
                self.assertAlmostEqual(float(increment.norm()),.1*float(v['LABEL'].norm()),places=12)
                matched=steps[f'{kind}-{rep}-M']-gs
                self.assertAlmostEqual(float(matched.norm()),.1*float(v['FULL'].norm()),places=12)
        # A is exactly source mean-label minus each graph's unchanged prediction.
        probs=geo.source_probabilities;y=torch.nn.functional.one_hot(labels,2).double()
        expected=-geo.basis@geo.readout@((y.mean(0)[None]-probs)@geo.contrasts).reshape(-1)
        torch.testing.assert_close(v['BASE'],expected,atol=1e-12,rtol=1e-10)
    def test_D7_nested_input_only_order_and_regularization(self):
        ids=list(range(196));order=query_order(ids,3405,0)
        self.assertEqual(order,query_order(list(reversed(ids)),3405,0))
        self.assertEqual(set(order),set(ids))
        self.assertTrue(set(order[:16])<set(order[:32])<set(order[:64]))
        self.assertEqual(query_lambda(196,196,'FIXED',.01),query_lambda(196,196,'NORMALIZED',.01))
        self.assertAlmostEqual(query_lambda(16,196,'NORMALIZED',.01),.01*16/196)
    def test_duplicate_invariance_in_multiclass_PSD_case(self):
        feat=torch.randn(9,7,dtype=torch.float64);gram=feat@feat.mT+.4*torch.eye(9,dtype=torch.float64)
        kernels=(gram[:5,:5],gram[:5,5:],gram[5:,5:]);response=torch.randn(4,2,3,dtype=torch.float64)
        cfg={'gap_atol':1e-14,'gap_rtol':1e-12,'max_iterations':2000}
        for mode in ['MEAN','JOINT']:
            r=duplicate_check(kernels,response,mode,.1,cfg)
            self.assertLess(r['scaled_collapse_error'],1e-8)
            self.assertLess(r['fixed_collapse_vs_half_lambda_error'],1e-8)

if __name__=='__main__':unittest.main()
