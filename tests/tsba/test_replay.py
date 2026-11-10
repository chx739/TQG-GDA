"""Independent checks for replay measurements, including failed root probes."""
import unittest
import torch
from torch import nn
from tsba.classifier import Coordinates
from workflows.tsba.validation.distributions import sample_gmm
from workflows.tsba.validation.replay import posterior_log_prob,source_metrics,target_metrics,boundary_probe


class RawMargin(nn.Module):
    def __init__(self,coordinates,kind):
        super().__init__();self.coordinates=coordinates;self.kind=kind
        self.weight=nn.Parameter(torch.tensor(1.,dtype=torch.float64),requires_grad=False)
    def forward(self,x):
        raw=x*self.coordinates.scale+self.coordinates.mean;t=raw[:,0]
        if self.kind=='tilted':m=self.weight*(t-.5-.25*raw[:,1])
        elif self.kind=='curved':m=t-.5-.25*raw[:,1].square()
        elif self.kind=='multiple':m=t.square()-1
        else:m=torch.ones_like(t)
        return torch.stack((-m/2,m/2),dim=1)


class ReplayTests(unittest.TestCase):
    def test_source_bayes_excess_is_conditional_kl(self):
        x,y=sample_gmm(2048,2,920001,.5);truth=posterior_log_prob(x,.5)
        baseline=source_metrics(truth,y,truth,truth)
        self.assertAlmostEqual(baseline['posterior_kl']['mean'],0.,places=14)
        current=posterior_log_prob(x,0.)
        row=source_metrics(current,y,truth,truth)
        self.assertGreater(row['posterior_kl']['mean'],0.)
        self.assertAlmostEqual(row['posterior_kl']['mean'],row['conditional_ce_delta']['mean'],places=14)

    def test_target_cohort_nll_identity_and_wrong_confidence(self):
        y=torch.tensor([0,0,0,0]);initial=torch.tensor([[2.,0.],[2.,0.],[0.,2.],[0.,2.]],dtype=torch.float64)
        final=torch.tensor([[3.,0.],[0.,3.],[3.,0.],[0.,3.]],dtype=torch.float64)
        row=target_metrics(final,y,initial)
        self.assertEqual([x['count'] for x in row['cohorts'].values()],[1,1,1,1])
        self.assertAlmostEqual(sum(x['nll_delta_contribution'] for x in row['cohorts'].values()),row['nll_delta'],places=14)
        self.assertEqual(row['high_confidence_error_fraction_90'],.5)
        self.assertEqual(row['high_confidence_error_fraction_99'],0.)
        self.assertAlmostEqual(row['wrong_nll_contribution']+row['correct_nll_contribution'],row['nll'],places=14)

    def test_tilted_and_curved_roots_in_raw_coordinates_without_mutation(self):
        coordinates=Coordinates(torch.tensor([2.,-1.],dtype=torch.float64),torch.tensor(2.,dtype=torch.float64))
        nuisance=torch.linspace(-1,1,21,dtype=torch.float64)[:,None]
        tilted=RawMargin(coordinates,'tilted');before={k:v.clone() for k,v in tilted.state_dict().items()}
        r,a=boundary_probe(tilted,coordinates,nuisance,grid_size=129)
        torch.testing.assert_close(a['roots'],.5+.25*nuisance[:,0],atol=1e-10,rtol=0)
        self.assertEqual(r['unique_detected'],21);self.assertAlmostEqual(r['plane_slope_norm'],.25,places=9)
        self.assertLess(r['nonplanarity_rms'],1e-10)
        self.assertAlmostEqual(r['normal_angle_mean_deg'],float(torch.atan(torch.tensor(.25,dtype=torch.float64))*180/torch.pi),places=9)
        for k,v in tilted.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        self.assertIsNone(tilted.weight.grad)
        curved=RawMargin(coordinates,'curved');r,a=boundary_probe(curved,coordinates,nuisance,grid_size=129)
        torch.testing.assert_close(a['roots'],.5+.25*nuisance[:,0].square(),atol=1e-10,rtol=0)
        self.assertGreater(r['nonplanarity_rms'],.05)

    def test_multiple_and_missing_roots_are_not_cherry_picked(self):
        coordinates=Coordinates(torch.zeros(2,dtype=torch.float64),torch.tensor(1.,dtype=torch.float64))
        nuisance=torch.zeros(3,1,dtype=torch.float64)
        for kind,status in [('multiple','multiple_detected_roots'),('missing','no_detected_root')]:
            r,a=boundary_probe(RawMargin(coordinates,kind),coordinates,nuisance,grid_size=129)
            self.assertEqual(r['unique_detected'],0);self.assertEqual(r['status_counts'],{status:3})
            self.assertTrue(torch.isnan(a['roots']).all())
