"""Extreme finite inputs must fail explicitly instead of producing false feedback."""
import unittest

import torch

from tsba.adapt import adapt_classifier
from tsba.boundary import project
from tsba.classifier import ParameterMap
from tsba.config import Config
from tsba.score import bandwidth
from tsba.utils import NumericalError


class ConstantScore(torch.nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


class NumericalEdgeTests(unittest.TestCase):
    def test_projection_squared_norm_overflow_is_an_error_and_preserves_model(self):
        for dtype, slope in ((torch.float32, 1e20), (torch.float64, 1e155)):
            with self.subTest(dtype=dtype):
                model = torch.nn.Linear(1, 2, dtype=dtype)
                with torch.no_grad():
                    model.weight.copy_(torch.tensor([[slope], [0.]], dtype=dtype))
                    model.bias.copy_(torch.tensor([slope * .001, 0.], dtype=dtype))
                # Nearby root is -0.001; the numerator is still finite, but ||grad||² is not.
                x = torch.tensor([[0.], [.1]], dtype=dtype)
                cfg = Config(adapt_rounds=3, adaptation_dtype=str(dtype).split(".")[-1])
                before = ParameterMap(model).vector()
                with self.assertRaises(NumericalError):
                    project(model, x[0], torch.tensor([0, 1]), 1., 1e-5, cfg)
                result = adapt_classifier(model, x, torch.tensor([0, 1]), x,
                                          ConstantScore(), 1., cfg)
                self.assertEqual(result["stop_reason"], "numerical_error")
                self.assertEqual(result["accepted"], 0)
                self.assertEqual(len(result["history"]), 1)
                torch.testing.assert_close(ParameterMap(model).vector(), before, rtol=0, atol=0)

    def test_distance_underflow_cannot_discard_a_distinct_neighbor(self):
        for dtype, separation in ((torch.float32, 1e-30), (torch.float64, 1e-200)):
            with self.subTest(dtype=dtype):
                # Median positive nearest distance is separation, not the distant point at 1.
                # Include a genuine duplicate, and split blocks to exercise group alignment.
                x = torch.tensor([[0.], [separation], [0.], [1.]], dtype=dtype)
                self.assertNotEqual(float(x[0, 0]), float(x[1, 0]))
                with self.assertRaisesRegex(NumericalError, "distance underflow"):
                    bandwidth(x, 1., chunk=2)

    def test_exact_duplicates_still_preserve_positive_nearest_distances(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[0., 0.], [3., 4.], [0., 0.], [3., 4.]], dtype=dtype)
                self.assertAlmostEqual(bandwidth(x, 1., chunk=2), 5 / 2 ** .5, places=6)
                self.assertIsNone(bandwidth(x[[0, 2]], 1., chunk=1))


if __name__ == "__main__":
    unittest.main()
