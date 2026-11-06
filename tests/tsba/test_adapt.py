import dataclasses
import unittest
import torch
from tqg_gda.tsba.adapt import adapt_classifier, backtrack
from tqg_gda.tsba.classifier import ParameterMap
from tqg_gda.tsba.problem import TsbaProblem
from tqg_gda.tsba.subproblem import solve
from test_math import fixture, ConstantScore


class AdaptationTests(unittest.TestCase):
    def test_accepted_step_changes_boundary_in_requested_direction(self):
        model, source, labels, target, cfg, _ = fixture(adapt_rounds=1)
        before = ParameterMap(model).vector()
        result = adapt_classifier(model, source, labels, target, ConstantScore(), 1., cfg)
        self.assertEqual(result["accepted"], 1)
        self.assertFalse(torch.equal(before, ParameterMap(model).vector()))
        logits = model(torch.zeros((1, 1), dtype=torch.float64))[0]
        self.assertGreater(float(logits[0] - logits[1]), 0)  # boundary moves toward -normal
        row = result["history"][0]
        accepted = row["trials"][-1]
        self.assertLess(accepted["objective"], row["before"]["objective"])
        self.assertLessEqual(accepted["max_movement"], cfg.execution_multiplier)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_empty_or_zero_feedback_never_falls_back_to_source_training(self):
        for score, radius in ((ConstantScore(0), 1.), (ConstantScore(), 1e-8)):
            model, source, labels, target, cfg, _ = fixture(adapt_rounds=2, radius_multiplier=radius)
            before = ParameterMap(model).vector()
            result = adapt_classifier(model, source, labels, target, score, 1., cfg)
            self.assertEqual(result["accepted"], 0)
            torch.testing.assert_close(before, ParameterMap(model).vector(), atol=0, rtol=0)

    def test_nonfinite_feedback_stops_and_preserves_snapshot(self):
        model, source, labels, target, cfg, _ = fixture(adapt_rounds=3)
        before = ParameterMap(model).vector()
        result = adapt_classifier(model, source, labels, target, ConstantScore(float("nan")), 1., cfg)
        self.assertEqual(result["stop_reason"], "numerical_error")
        self.assertEqual(len(result["history"]), 1)
        torch.testing.assert_close(before, ParameterMap(model).vector(), atol=0, rtol=0)

    def test_normal_overflow_is_an_execution_failure_with_rollback(self):
        model = torch.nn.Linear(2, 2, dtype=torch.float32)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1e20, 1e20], [0., 0.]]))
            model.bias.zero_()
        _, _, _, _, cfg, _ = fixture(adapt_rounds=3)
        x = torch.tensor([[0., 0.], [1., 1.]])
        before = ParameterMap(model).vector()
        result = adapt_classifier(model, x, torch.tensor([0, 1]), x, ConstantScore(), 1., cfg)
        self.assertEqual(result["stop_reason"], "numerical_error")
        self.assertEqual(result["accepted"], 0)
        torch.testing.assert_close(before, ParameterMap(model).vector(), atol=0, rtol=0)

    def test_exhausted_backtracking_does_not_mutate_live_model(self):
        model, source, labels, _, cfg, b = fixture(backtracks=1, execution_multiplier=1e-20)
        problem = TsbaProblem(model, source, labels, b, cfg)
        before = ParameterMap(model).vector()
        delta, info = solve(problem)
        candidate, history = backtrack(problem, delta, info["directional_derivative"], 1.)
        self.assertIsNone(candidate)
        self.assertFalse(history[0]["bounded"])
        torch.testing.assert_close(before, ParameterMap(model).vector(), atol=0, rtol=0)

    def test_missing_distinct_targets_returns_initial_classifier(self):
        model, source, labels, target, cfg, _ = fixture()
        before = ParameterMap(model).vector()
        result = adapt_classifier(model, source, labels, target, None, None, cfg)
        self.assertEqual(result["accepted"], 0)
        torch.testing.assert_close(before, ParameterMap(model).vector(), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
