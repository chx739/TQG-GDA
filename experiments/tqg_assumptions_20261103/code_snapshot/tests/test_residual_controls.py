import copy
import tempfile
import unittest

import torch
from torch.nn import functional as F

from helpers import fixture_files,small_config,task_from
from test_feedback import Tiny
from tqg_gda.feedback import (ProbeConfig,build_feedback,build_mean_feedback,feedback_loss,
                              fit_residual_with_intercept)
from tqg_gda.models import GraphClassifier
from tqg_gda.training import TQGTrainer,model_spec
from tqg_gda.utils import merge,validate_config


class ResidualControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(71)
        self.model=Tiny().double().eval()
        self.source=torch.randn(5,2,dtype=torch.float64);self.target=torch.randn(3,2,dtype=torch.float64)
        self.labels=torch.tensor([0,0,1,0,1])

    def test_mean_is_fitting_residual_average_and_stopped(self):
        before=copy.deepcopy(self.model.state_dict())
        feedback=build_mean_feedback(self.model,self.source,self.labels,self.target)
        residual=F.one_hot(self.labels,2)-self.model(self.source).softmax(-1)
        expected=residual.mean(0).expand(3,2)
        torch.testing.assert_close(feedback.correction,expected)
        torch.testing.assert_close(feedback.reference,self.model(self.target)+expected)
        self.assertEqual(feedback.statistics['explicit_forward_evaluations'],2)
        self.assertEqual(feedback.statistics['candidate_directions'],0)
        for field in ('source_residual','correction','reference','intercept'):
            self.assertFalse(getattr(feedback,field).requires_grad)
        for name,p in self.model.state_dict().items():self.assertTrue(torch.equal(p,before[name]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_mean_feedback_gradient_is_stopped_jacobian_correction(self):
        feedback=build_mean_feedback(self.model,self.source,self.labels,self.target)
        parameters=tuple(self.model.parameters())
        a=torch.autograd.grad(feedback_loss(self.model,self.target,feedback),parameters)
        b=torch.autograd.grad(-(self.model(self.target)*feedback.correction).sum()/len(self.target),parameters)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-10)

    def test_intercept_matches_independent_augmented_least_squares(self):
        n,c,m=7,3,4;ridge=.13
        phi=torch.randn(n,c,m,dtype=torch.float64);r=torch.randn(n,c,dtype=torch.float64)
        a,b=fit_residual_with_intercept(phi,r,ridge)
        design=torch.cat((phi.reshape(n*c,m),torch.eye(c,dtype=torch.float64).repeat(n,1)),1)
        penalty=torch.zeros(m,m+c,dtype=torch.float64);penalty[:,:m]=(n*ridge)**.5*torch.eye(m,dtype=torch.float64)
        solution=torch.linalg.lstsq(torch.cat((design,penalty)),torch.cat((r.flatten(),torch.zeros(m,dtype=torch.float64)))).solution
        torch.testing.assert_close(a,solution[:m],atol=1e-12,rtol=1e-11)
        torch.testing.assert_close(b,solution[m:],atol=1e-12,rtol=1e-11)

    def test_constant_and_absent_features_reduce_to_mean(self):
        residual=torch.randn(6,2,dtype=torch.float64)
        for phi in (torch.randn(1,2,3,dtype=torch.float64).repeat(6,1,1),torch.empty(6,2,0,dtype=torch.float64)):
            a,b=fit_residual_with_intercept(phi,residual,.01)
            torch.testing.assert_close(a,torch.zeros_like(a),atol=1e-25,rtol=0)
            torch.testing.assert_close(b,residual.mean(0))

    def test_intercept_does_not_change_directions_responses_or_scaling(self):
        probe=ProbeConfig(count=2)
        a=build_feedback(self.model,self.source,self.labels,self.target,probe,torch.Generator().manual_seed(4))
        b=build_feedback(self.model,self.source,self.labels,self.target,probe,torch.Generator().manual_seed(4),predictor='responses_with_intercept')
        for key in ('source_features','target_features','scales'):self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)))
        for x,y in zip(a.directions,b.directions):
            for key in x:self.assertTrue(torch.equal(x[key],y[key]))
        torch.testing.assert_close(b.correction,b.target_features@b.coefficients+b.intercept)
        self.assertFalse(b.intercept.requires_grad)

    def test_intercept_remains_available_without_valid_directions(self):
        with torch.no_grad():self.model.head.weight.zero_()
        y=torch.tensor([0,0,0,0,1]);probe=ProbeConfig(count=2)
        feedback=build_feedback(self.model,self.source,y,self.target,probe,torch.Generator().manual_seed(4),predictor='responses_with_intercept')
        self.assertEqual(feedback.target_features.shape[-1],0)
        self.assertEqual(feedback.statistics['status'],'intercept_only_feedback')
        expected=(F.one_hot(y,2)-self.model(self.source).softmax(-1)).mean(0)
        torch.testing.assert_close(feedback.correction,expected.expand_as(feedback.correction))
        self.assertGreater(float(feedback_loss(self.model,self.target,feedback)),0.)

    def test_new_predictors_resume_and_ignore_poisoned_target_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            files=fixture_files(tmp)
            for predictor in ('mean_residual','responses_with_intercept'):
                config=merge(small_config(),{'adaptation':{'feedback_predictor':predictor}})
                task=task_from(files,config)
                model=GraphClassifier(model_spec(task,config),2).double();trainer=TQGTrainer(model,config)
                trainer.step(task);saved=copy.deepcopy(trainer.state());expected=trainer.step(task)
                expected.pop('seconds')
                task._raw.y[list(task.target_ids)]=-123456
                restored=TQGTrainer(GraphClassifier(model_spec(task,config),2).double(),config);restored.restore(saved)
                actual=restored.step(task);actual.pop('seconds')
                self.assertEqual(expected,actual)
                for name,p in model.state_dict().items():self.assertTrue(torch.equal(p,restored.model.state_dict()[name]))

    def test_config_preserves_reserved_query_roles(self):
        for predictor in ('mean_residual','responses_with_intercept'):
            config=merge(small_config(),{'adaptation':{'feedback_predictor':predictor}})
            validate_config(config)
            with self.assertRaises(ValueError):validate_config(merge(config,{'method':'source_only'}))
            with self.assertRaises(ValueError):validate_config(merge(config,{'warmup':{'source_pool':'all_source'}}))
        with self.assertRaises(ValueError):validate_config(merge(small_config(),{'adaptation':{'feedback_predictor':'bad'}}))


if __name__=='__main__':unittest.main()
