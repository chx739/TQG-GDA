"""D0 integration checks using synthetic PyG graphs, no real target labels."""
import copy
import unittest

import torch
from torch.nn import functional as F
from torch_geometric.data import Data, Batch

from tqg_gda.models import GraphClassifier
from tqg_gda.feedback import ProbeConfig, build_feedback
from tqg_gda.revision import ResponseDictionary, ResidualPredictor, auxiliary_loss
from revision_execution_diagnostics import dense_centered_jacobian, flatten, gradient, ExecutionDiagnostics


class RevisionIntegration(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        torch.manual_seed(734)
        self.model = GraphClassifier(dict(input_dim=3, hidden_dim=4, num_layers=2,
                                          readout='mean', edge_dim=0), 3).double().eval()
        def batch(count, labels=False):
            graphs=[]
            for i in range(count):
                n=3+i%3
                g=Data(x=torch.randn(n,3), edge_index=torch.tensor(
                    [[j for j in range(n-1)]+[j+1 for j in range(n-1)],
                     [j+1 for j in range(n-1)]+[j for j in range(n-1)]], dtype=torch.long))
                if labels:
                    g.y=torch.tensor([i%3])
                graphs.append(g)
            return Batch.from_data_list(graphs)
        self.source, self.target, self.held = batch(8,True),batch(5),batch(4)
        self.dictionary=ResponseDictionary.build(self.model,self.source,self.target,
            torch.Generator().manual_seed(49),K=2,response='odd')
        self.cfg=dict(method='tqg',predictor='weighted_ridge',matching='uniform',
                      ridge=.01,projection=False,gate=False)

    def predictor(self, **kwargs):
        cfg=dict(self.cfg,**kwargs)
        return ResidualPredictor(self.dictionary,self.source,cfg,torch.arange(8),self.model)

    def test_original_odd_reference_replayed(self):
        original=build_feedback(self.model,self.source,self.source.y,self.target,
                               ProbeConfig(count=2,responses='odd'),torch.Generator().manual_seed(49))
        pred=self.predictor().predict(self.target,phi=self.dictionary.feedback.target_features)
        torch.testing.assert_close(pred.reference,original.reference,atol=1e-10,rtol=1e-10)

    def test_same_dictionary_extends_without_rescaling(self):
        torch.testing.assert_close(self.dictionary.features(self.source),
                                   self.dictionary.feedback.source_features,atol=1e-10,rtol=1e-10)

    def test_projected_feedback_soft_CE_gradient_equivalence(self):
        pred=self.predictor(matching='response_local',projection=True).predict(self.target)
        parameters=tuple(self.model.parameters())
        mse=gradient(auxiliary_loss(self.model,self.source,self.target,pred),parameters)
        q=(pred.correction+self.model(self.target).detach().softmax(-1)).detach()
        ce=gradient(-(q*self.model(self.target).log_softmax(-1)).sum(-1).mean(),parameters)
        torch.testing.assert_close(mse,ce,atol=1e-10,rtol=1e-10)

    def test_gate_zero_means_no_auxiliary_gradient(self):
        pred=self.predictor().predict(self.target)
        pred.gate.zero_()
        values=gradient(auxiliary_loss(self.model,self.source,self.target,pred),tuple(self.model.parameters()))
        self.assertEqual(float(values.norm()),0.)

    def test_all_auxiliary_objects_stopped_and_no_mutation(self):
        saved=copy.deepcopy(self.model.state_dict())
        pred=self.predictor(matching='response_local',projection=True,ridge='source_CV').predict(self.held)
        for value in (pred.reference,pred.correction,pred.raw_correction,pred.weights,pred.coefficients):
            self.assertFalse(value.requires_grad)
            self.assertIsNone(value.grad_fn)
        for name,value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value,saved[name]))
        self.assertIsNone(self.target.y)

    def test_dense_gradient_matches_autograd(self):
        pred=self.predictor(matching='response_local',projection=True).predict(self.target)
        _,jac=dense_centered_jacobian(self.model,self.target)
        expected=-torch.einsum('ncp,nc->p',jac,pred.correction)/len(pred.correction)
        actual=gradient(auxiliary_loss(self.model,self.source,self.target,pred),tuple(self.model.parameters()))
        torch.testing.assert_close(expected,actual,atol=1e-10,rtol=1e-10)

    def test_frozen_and_current_embedding_agree_at_warmup(self):
        frozen=self.predictor(matching='frozen_embedding').predict(self.target)
        current=self.predictor(matching='current_embedding').predict(self.target)
        torch.testing.assert_close(frozen.weights,current.weights,atol=1e-12,rtol=1e-12)

    def test_stateless_actual_Adam_trials_and_zero_effect(self):
        original=copy.deepcopy(self.model.state_dict())
        execution=ExecutionDiagnostics(self.model,self.source,self.source,self.target,self.held,
                                       torch.tensor([0,1,2,0,1]),torch.tensor([0,1,2,0]))
        pred=self.predictor(predictor='zero').predict(self.target)
        pred_h=self.predictor(predictor='zero').predict(self.held)
        row=execution.score(dict(query=pred,held=pred_h,source_labels=self.source.y))
        self.assertEqual(row['actual_Adam_CE_gain_vs_source_only'],0.)
        self.assertIsNone(row['gradient_cosine'])
        self.assertEqual(row['normalized_SGD_CE_gain'],0.)
        for name,value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value,original[name]))


if __name__=='__main__':
    unittest.main()
