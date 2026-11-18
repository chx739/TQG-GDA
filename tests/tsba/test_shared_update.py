import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, logit, logsumexp, softmax
from scipy.stats import norm

from scripts.run_task import synthetic_task
from tsba.config import Config
from tsba.pipeline import run_task
from tsba.shared_update import SharedUpdateConfig
from tsba.shared_update.explanations import Explanation, ExplanationSet
from tsba.shared_update.pipeline import run_shared_task
from tsba.shared_update.predict import SharedUpdatePredictor, corrected_logits, load_shared_predictor
from tsba.shared_update.solver import _partition, _risk_terms, relative_risks, solve_master, solve_shared_update


class SharedUpdateTests(unittest.TestCase):
    def test_invalid_warm_master_retries_once_from_zero(self):
        z0 = np.zeros((3, 2))
        u = np.ones((3, 1))
        statistics = np.array([[[.8], [.2]]])
        initial = np.array([[.1], [-.1]])
        launches = []

        def premature_warm_exit(objective, x0, **kwargs):
            launches.append((x0.copy(), kwargs["options"]))
            if len(launches) == 1:
                return OptimizeResult(x=x0, status=0, success=True, message="premature exit", nit=1)
            return minimize(objective, x0, **kwargs)

        with patch("scipy.optimize.minimize", side_effect=premature_warm_exit):
            w, info = solve_master(z0, u, statistics, regularization=.1, master_maxiter=200, initial=initial)
        self.assertTrue(info["valid"], info)
        self.assertTrue(info["zero_start_retry"])
        self.assertEqual(info["warm_start_failure"]["reason"], "stationary")
        self.assertFalse(info["warm_start_failure"]["valid"])
        self.assertEqual(len(launches), 2)
        np.testing.assert_array_equal(launches[0][0][:-1], initial.ravel())
        np.testing.assert_array_equal(launches[1][0], 0.)
        self.assertEqual(launches[0][1], launches[1][1])
        cold_w, _ = solve_master(z0, u, statistics, regularization=.1, master_maxiter=200)
        np.testing.assert_array_equal(w, cold_w)

        launches.clear()
        with patch("scipy.optimize.minimize", side_effect=premature_warm_exit):
            _, invalid = solve_master(z0, u, statistics, regularization=.1, master_maxiter=200)
        self.assertFalse(invalid["valid"])
        self.assertNotIn("zero_start_retry", invalid)
        self.assertEqual(len(launches), 1)

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
        with self.assertRaisesRegex(ValueError, "mean target features"):
            relative_risks(z0, u, w, [statistic / 2])

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

    def test_four_graph_example_from_design(self):
        # Design §3.15 uses zero regularization solely for this example. A
        # small positive value keeps the implemented master strictly convex.
        z0 = np.column_stack((logit([.42, .53, .47, .48]), np.zeros(4)))
        u = np.array([[-1.], [-1.], [1.], [1.]])
        q_class_1 = np.array([[.10, .60, .35, .85], [.50, .30, .75, .35]])
        q = np.stack((q_class_1, 1 - q_class_1), axis=-1)
        statistics = np.einsum("knc,nd->kcd", q, u) / len(u)
        w, master = solve_master(z0, u, statistics, regularization=1e-9)
        self.assertTrue(master["valid"], master)
        np.testing.assert_allclose(softmax(z0 + u @ w.T, axis=1)[:, 0],
                                   [.34805146, .45395642, .54604358, .55596431], atol=1e-6, rtol=0.)
        np.testing.assert_allclose(relative_risks(z0, u, w, statistics),
                                   [-.02663122, -.01138913], atol=1e-6, rtol=0.)

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
        first = result["iterations"][0]
        self.assertGreater(first["separation_violation"], .1)
        self.assertEqual(first["separation_violation"], first["search_risk"] - first["master"]["epigraph"])
        self.assertFalse(result["full_set_certified"])
        np.testing.assert_allclose(w, 0., atol=1e-7)

    def test_distant_class_independent_coordinate_preserves_posterior(self):
        source = np.array([[-1.5, 0.], [-.5, 0.], [.5, 0.], [1.5, 0.]])
        target = np.array([[.99, 0.], [.99, 1e6]])
        family = ExplanationSet(source, np.array([0, 0, 1, 1]), target, SharedUpdateConfig())
        posterior = family._target_terms(family._initial)[-1]
        self.assertTrue(np.isfinite(family._observation(family._initial)[0]))
        np.testing.assert_allclose(posterior.sum(axis=1), 1., atol=1e-15, rtol=0.)
        np.testing.assert_allclose(posterior[0], posterior[1], atol=1e-15, rtol=0.)
        self.assertLess(posterior[1, 0], 1e-3)

    def test_posterior_preserves_small_odds_from_large_squared_distances(self):
        source = np.repeat([[-.5, .5], [.5, -.5]], 2, axis=0)
        target = np.array([[1e6, 1e6 + .0005], [1e7, 1e7 + .0005]])
        cfg = SharedUpdateConfig()
        family = ExplanationSet(source, np.array([0, 0, 1, 1]), target, cfg)
        q = family._target_terms(family._initial)[-1][:, 0]
        # Equal variances and swapped means reduce the log-odds to the
        # coordinate difference divided by the variance.
        expected = expit((target[:, 1] - target[:, 0]) / cfg.variance_floor)
        np.testing.assert_allclose(q, expected, atol=3e-7, rtol=0.)

    def test_posterior_with_unequal_variances_near_a_narrow_class_mean(self):
        source = np.array([[0.], [0.], [.6], [1.4]])
        for floor in (1e-12, 1e-20):
            with self.subTest(variance_floor=floor):
                point = np.sqrt(floor * (np.log(.16 / floor) + 6.25 + 3))
                family = ExplanationSet(source, np.array([0, 0, 1, 1]), [[point]],
                                        SharedUpdateConfig(variance_floor=floor))
                *_, normalizer, posterior = family._target_terms(family._initial)
                log_components = norm.logpdf(point, loc=[0., 1.], scale=[np.sqrt(floor), .4]) - np.log(2.)
                np.testing.assert_allclose(posterior[0], softmax(log_components), atol=1e-13, rtol=0.)
                self.assertAlmostEqual(normalizer[0], logsumexp(log_components), places=12)

    def test_master_gradient_and_prediction_preserve_logit_differences_at_large_scale(self):
        cases = [([1., 0., 0.], [1e18, 1e18, -2e18], [1., 0., -np.inf]),
                 ([1e18, 0., 1.], [-1e18, 0., 0.], [0., 0., 1.]),
                 ([0., -1e18, -1e18], [1., 1e18, 1e18], [1., 0., 0.]),
                 ([0., 1e18, 1e18], [-1., -1e18, -1e18], [-1., 0., 0.])]
        for source, delta, shifted in cases:
            with self.subTest(source=source, delta=delta):
                source_logits = np.array([source])
                correction = np.array(delta)[:, None]
                _, gradient = _partition(source_logits, np.ones((1, 1)), correction)
                expected = softmax(shifted)
                np.testing.assert_allclose(gradient[:, 0], expected, atol=1e-15, rtol=0.)
                statistic = np.array([[[1.], [0.], [0.]]])
                _, risk_gradient = _risk_terms(source_logits, np.ones((1, 1)), correction, statistic)
                np.testing.assert_allclose(risk_gradient[:, 0], expected, atol=1e-15, rtol=0.)
                logits = corrected_logits(torch.from_numpy(source_logits), torch.ones((1, 1), dtype=torch.float64),
                                          torch.from_numpy(correction))
                np.testing.assert_allclose(logits.softmax(-1).numpy()[0], expected, atol=1e-15, rtol=0.)

    def test_relative_risk_preserves_small_change_with_large_common_correction(self):
        z0 = np.array([[1., 0., 0.]])
        correction = np.array([[1e18], [1e18], [1e18 - 128]])
        statistic = np.array([[[1.], [0.], [0.]]])
        expected = logsumexp([1., 0., -128.]) - logsumexp([1., 0., 0.])
        actual = relative_risks(z0, np.ones((1, 1)), correction, statistic)
        np.testing.assert_allclose(actual, [expected], atol=1e-12, rtol=0.)
        for base, delta, sign in ((0., 1e18, -1), (1e18, -1e18, 1)):
            risk = relative_risks([[base, 0.]], [[1.]], [[delta], [0.]], [[[1.], [0.]]])
            np.testing.assert_allclose(risk, [sign * np.log(2.)], atol=1e-14, rtol=0.)

    def test_master_certifies_optimum_with_close_cuts(self):
        statistics = np.array([[[.50102], [.49898]], [[.501], [.499]]])
        w, master = solve_master([[0., 0.]], [[1.]], statistics, regularization=.1)
        self.assertTrue(master["valid"], master)
        self.assertGreater(w[0, 0], 1e-3)
        self.assertLess(master["objective"], -1e-6)

    def test_multiplier_lp_verifies_nearly_dependent_active_cuts(self):
        u = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        z0 = np.zeros((len(u), 2))
        first_class = np.array([[.1, 1e-5], [-.1, 0.], [.1, 0.]])
        stats = np.stack((first_class, -first_class), axis=1)

        def stalled_multiplier_fit(objective, x0, **kwargs):
            if kwargs["options"]["maxiter"] == 1000:
                return OptimizeResult(x=np.array([.5, .5, 0.]), success=True)
            return minimize(objective, x0, **kwargs)

        with patch("scipy.optimize.minimize", side_effect=stalled_multiplier_fit):
            w, master = solve_master(z0, u, stats, regularization=.1, master_maxiter=200)
        self.assertTrue(master["valid"], master)
        kkt = master["kkt"]
        self.assertTrue(kkt["multiplier_lp"]["success"])
        self.assertFalse(kkt["initial_multiplier_failure"]["valid"])
        self.assertGreater(kkt["initial_multiplier_failure"]["stationarity_residual"],
                           kkt["stationarity_tolerance"])
        np.testing.assert_allclose(kkt["active_weights"], [0., .5, .5], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(w, 0., atol=1e-12, rtol=0.)

    def test_master_handles_large_features_without_changing_original_penalty(self):
        u = np.array([[1e6, 1.]])
        z0 = np.zeros((1, 2))
        q = np.array([[.6, .4]])
        statistics = q.T @ u
        w, master = solve_master(z0, u, [statistics], regularization=.1, master_maxiter=200)
        self.assertTrue(master["valid"], master)
        direct = (q * np.log(softmax(u @ w.T, axis=1) / .5)).sum()
        original_penalty = .1 * np.square(w).sum() / 2
        self.assertAlmostEqual(master["objective"], -direct + original_penalty, places=11)
        self.assertLess(master["objective"], -.02)

    def test_multiclass_search_retains_valid_update_with_near_active_cuts(self):
        rng = np.random.default_rng(5)
        y = np.tile(np.arange(3), 10)
        means = rng.normal(size=(3, 2))
        sources = means[y] + .6 * rng.normal(size=(len(y), 2))
        targets = (means[y] + .6 * rng.normal(size=(len(y), 2)) - .15) / .8
        z0 = rng.normal(size=(len(y), 3))
        mean, scale = sources.mean(axis=0), sources.std(axis=0)
        cfg = SharedUpdateConfig(inner_starts=3, search_rounds=4)
        family = ExplanationSet((sources - mean) / scale, y, (targets - mean) / scale, cfg)
        _, result = solve_shared_update(z0, family.features, family, cfg)
        self.assertEqual(result["status"], "completed", result["stop_reason"])
        self.assertGreater(result["candidate_count"], 1)
        self.assertLess(result["final_regularized_objective"], -.4)

    def test_master_and_prediction_remove_common_feature_weights_before_matmul(self):
        z0, u = np.zeros((1, 3)), np.ones((1, 2))
        w = np.array([[2., 1e18], [0., 1e18], [1., 1e18]])
        statistic = np.array([[[1., 1.], [0., 0.], [0., 0.]]])
        _, gradient = _risk_terms(z0, u, w, statistic)
        perturbation = np.array([[1e-5, 0.], [0., 0.], [0., 0.]])
        numerical = (relative_risks(z0, u, w + perturbation, statistic)
                     - relative_risks(z0, u, w - perturbation, statistic)) / 2e-5
        np.testing.assert_allclose(numerical, gradient[0, 0] - 1., atol=1e-9, rtol=0.)
        logits = corrected_logits(*(torch.from_numpy(value) for value in (z0, u, w)))
        np.testing.assert_allclose(logits.softmax(-1)[0].numpy(), softmax([2., 0., 1.]), atol=1e-15, rtol=0.)

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
