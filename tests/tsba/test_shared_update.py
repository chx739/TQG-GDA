import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.stats import norm

from scripts.run_task import synthetic_task
from tsba.config import Config
from tsba.pipeline import run_task
from tsba.shared_update import SharedUpdateConfig
from tsba.shared_update.explanations import Explanation, ExplanationSet
from tsba.shared_update.pipeline import run_shared_task
from tsba.shared_update.predict import SharedUpdatePredictor, load_shared_predictor
from tsba.shared_update.solver import relative_risks, solve_shared_update


class SharedUpdateTests(unittest.TestCase):
    def test_finite_risk_matches_direct_conditional_nll(self):
        z0 = np.array([[1.2, -0.3], [-1.4, 2.], [.9, .2]])
        u = np.array([[-2., 1.], [.5, 1.], [1.4, 1.]])
        w = np.array([[.7, .12], [-.5, -.32]])
        q = np.array([[.1, .9], [.6, .4], [.55, .45]])
        updated = torch.tensor(z0 + u @ w.T).log_softmax(-1)
        base = torch.tensor(z0).log_softmax(-1)
        expected = (torch.tensor(q) * (base - updated)).sum(-1).mean().item()
        statistic = q.T @ u / len(u)
        actual = relative_risks(z0, u, w, [statistic])[0]
        self.assertAlmostEqual(actual, expected, places=13)

    def test_fitted_explanation_uses_change_of_variables_and_fixed_threshold(self):
        source = np.array([[-1.5], [-.8], [-.5], [.6], [1.3], [2.]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        target = np.array([[-.6], [.15], [.9], [1.3], [1.8]])
        config = SharedUpdateConfig(inner_starts=2, tau=.05)
        family = ExplanationSet(source, labels, target, config)
        reference = family.fit()
        threshold = family.threshold
        correction = np.column_stack((target[:, 0], -target[:, 0]))
        adverse = family.search(correction, [reference])
        self.assertIsNotNone(adverse)
        self.assertIs(family.fit(), reference)
        self.assertEqual(family.threshold, threshold)
        self.assertLessEqual(adverse.observation_loss, threshold + config.feasibility_tol)
        means, log_variances, log_scale, shift, _, priors = family._unpack(adverse.parameters)
        scale = np.exp(log_scale[0])
        deviations = np.exp(log_variances[:, 0] / 2)
        source_log_density = norm.logpdf(source[:, 0], means[labels, 0], deviations[labels])
        target_log_joint = (norm.logpdf(target * scale + shift, means[:, 0], deviations)
                            + np.log(priors) + np.log(scale))
        target_log_density = logsumexp(target_log_joint, axis=1)
        expected_loss = -(source_log_density.sum() + target_log_density.sum()) / (len(source) + len(target))
        expected_q = np.exp(target_log_joint - target_log_density[:, None])
        self.assertAlmostEqual(adverse.observation_loss, expected_loss, places=12)
        np.testing.assert_allclose(adverse.statistic, expected_q.T @ family.features / len(target))

    def test_last_search_candidate_constrains_final_update(self):
        reference = Explanation(np.array([0.]), 0., np.array([[.9], [.1]]))
        adverse = Explanation(np.array([1.]), .01, np.array([[.1], [.9]]))

        class Family:
            reference_loss = 0.
            threshold = .02

            def fit(self):
                return reference

            def search(self, logits, starts):
                if logits[0, 0] <= logits[0, 1] or len(starts) != 1:
                    raise AssertionError("the first master must produce a nonzero correction for the reference")
                return adverse

        w, result = solve_shared_update(np.zeros((3, 2)), np.ones((3, 1)), Family(),
                                        SharedUpdateConfig(search_rounds=1))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stop_reason"], "search_budget_exhausted")
        self.assertEqual(result["candidate_count"], 2)
        self.assertGreater(result["iterations"][0]["separation_violation"], .1)
        self.assertFalse(result["full_set_certified"])
        np.testing.assert_allclose(w, 0., atol=1e-7)

    def test_reuses_tsba_source_and_loads_standalone_predictor(self):
        cfg = Config(hidden=8, num_layers=2, emb_dim=4, source_epochs=3, source_batch=16,
                     score_epochs=1, adapt_rounds=0)
        data = synthetic_task()
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "original", Path(tmp) / "shared"
            run_task(cfg, data, old)
            core_source = torch.load(old / "source.pt", weights_only=True)
            model, result = run_shared_task(cfg, SharedUpdateConfig(), data, new, old / "source.pt")
            self.assertEqual(result["adaptation"]["status"], "completed")
            self.assertFalse(result["target_labels_used"])
            self.assertIsNone(result["target_accuracy"])
            self.assertGreater(float(model.correction.norm()), 1e-4)
            for name, parameter in model.source.state_dict().items():
                torch.testing.assert_close(parameter, core_source["state"][name], atol=0, rtol=0)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            predictions = torch.load(new / "predictions.pt", weights_only=True)
            loaded = load_shared_predictor(new / "model.pt")
            torch.testing.assert_close(loaded.predict(data.target.graphs), predictions["probabilities"], atol=0, rtol=0)
            zero = SharedUpdatePredictor(model.source, model.mean, model.scale, torch.zeros_like(model.correction))
            torch.testing.assert_close(zero.predict(data.target.graphs), predictions["initial_probabilities"], atol=0, rtol=0)
            self.assertEqual(json.loads((new / "result.json").read_text())["run_identity"], result["run_identity"])


if __name__ == "__main__":
    unittest.main()
