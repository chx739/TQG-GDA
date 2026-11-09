"""Precision policy and accepted updates on paired synthetic inputs; no target labels."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from tsba.adapt import adapt_classifier
from tsba.classifier import ResidualClassifier
from tsba.config import Config
from tsba.subproblem import solve
from tsba.utils import NumericalError

torch.set_num_threads(1)


class ConstantScore(torch.nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


def paired_inputs(seed):
    torch.manual_seed(seed)
    model = ResidualClassifier(8, 3, 8, dtype=torch.float32)
    source = torch.randn(32, 8)
    # Synthetic source semantics; target inputs do not receive labels.
    labels = model(source).argmax(-1)
    return model, source, labels, torch.randn(16, 8)


class SolverPrecisionTests(unittest.TestCase):
    def test_precision_defaults_and_explicit_tolerance_round_trip(self):
        for dtype, expected in (("float32", 1e-5), ("float64", 1e-7)):
            cfg = Config(adaptation_dtype=dtype)
            self.assertEqual(cfg.cg_tolerance, expected)
            self.assertEqual(cfg.to_dict()["cg_tolerance"], expected)
            self.assertEqual(Config(**cfg.to_dict()), cfg)
            explicit = Config(adaptation_dtype=dtype, cg_tolerance=1e-9)
            self.assertEqual(explicit.cg_tolerance, 1e-9)
            self.assertEqual(Config(**explicit.to_dict()), explicit)
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Config(cg_tolerance=invalid)

    def test_float32_and_float64_direct_and_cg_accept_paired_updates(self):
        for seed in range(5):
            initial, source, labels, target = paired_inputs(seed)
            for dtype in (torch.float32, torch.float64):
                for solver in ("direct", "cg"):
                    with self.subTest(seed=seed, dtype=dtype, solver=solver):
                        cfg = Config(seed=seed, adapt_rounds=3, solver=solver,
                                     adaptation_dtype=str(dtype).split(".")[-1])
                        result = adapt_classifier(copy.deepcopy(initial).to(dtype), source.to(dtype), labels,
                                                  target.to(dtype), ConstantScore(), 1., cfg)
                        self.assertEqual(result["accepted"], 3, result["history"])
                        for row in result["history"]:
                            self.assertLessEqual(row["solver"]["relative_residual"], cfg.cg_tolerance)
                            self.assertLess(row["solver"]["directional_derivative"], 0.)
                            self.assertTrue(row["trials"][-1]["armijo"])
                            self.assertTrue(row["trials"][-1]["bounded"])

    def test_explicit_strict_float32_tolerance_is_not_relaxed(self):
        initial, source, labels, target = paired_inputs(0)
        before = {k: v.clone() for k, v in initial.state_dict().items()}
        cfg = Config(adapt_rounds=1, adaptation_dtype="float32", solver="direct", cg_tolerance=1e-12)
        result = adapt_classifier(initial, source, labels, target, ConstantScore(), 1., cfg)
        self.assertEqual(result["accepted"], 0)
        row = result["history"][0]
        self.assertEqual(row["reason"], "inaccurate_or_non_descent")
        self.assertGreater(row["solver"]["relative_residual"], cfg.cg_tolerance)
        for name, value in initial.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_finite_rhs_with_overflowing_norm_is_numerical_error(self):
        # Isolate solver entry: each component is finite, but the reduction is not.
        rhs = torch.tensor([1e20, 1e20], dtype=torch.float32)
        self.assertTrue(bool(torch.isfinite(rhs).all()))
        with patch("tsba.subproblem.GGNSystem", return_value=SimpleNamespace(rhs=rhs)):
            with self.assertRaises(NumericalError):
                solve(SimpleNamespace(cfg=Config()))


if __name__ == "__main__":
    unittest.main()
