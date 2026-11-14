"""Checks for the R3 ablation, exact constraints and label isolation."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch.func import jacrev

from tsba.adapt import backtrack
from tsba.boundary import BoundaryBatch, build_boundaries
from tsba.classifier import ParameterMap, ResidualClassifier
from tsba.problem import TsbaProblem
from tsba.subproblem import solve
from tsba.utils import NumericalError
from experiments.mutagenicity.r2_score_budget_and_adaptation_20261111.src.common import atomic_tensor, batches, load, seal, write_json
from experiments.mutagenicity.r3_risk_and_confidence_constraints_20261112.src.constraints import (
    AffineBudgets, Budgets, accept, margin_jacobian, propose,
)
from experiments.mutagenicity.r3_risk_and_confidence_constraints_20261112.src.diagnostics import relocate
from experiments.mutagenicity.r3_risk_and_confidence_constraints_20261112.src.protocol import Job, grid, read_json, target_values
from experiments.mutagenicity.r3_risk_and_confidence_constraints_20261112.src.trajectory import one_attempt, trajectory

PLAN = read_json(Path(__file__).resolve().parents[2] / "experiments/mutagenicity/r3_risk_and_confidence_constraints_20261112/configs/r3-plan.json")


class R3Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(72)
        self.model = ResidualClassifier(2, 2, 3, dtype=torch.float64).eval().requires_grad_(False)
        self.source = torch.randn(20, 2, dtype=torch.float64)
        self.labels = torch.arange(20) % 2
        self.target = torch.randn(12, 2, dtype=torch.float64)
        self.job = Job(3405, 1, "M00", "learned", 1, 1, 1., None)
        self.cfg = self.job.config(PLAN)
        self.sigma = 1.
        self.bounds = build_boundaries(self.model, self.target, torch.ones_like, self.sigma, self.cfg)
        self.assertGreater(len(self.bounds), 0)
        self.problem = TsbaProblem(self.model, self.source[:8], self.labels[:8], self.bounds, self.cfg)

    def budgets(self, job):
        return Budgets(self.model, self.source, self.labels, self.target, job, PLAN)

    def test_full_grid(self):
        jobs = grid(PLAN)
        self.assertEqual(len(jobs), 520)
        self.assertEqual(sum(j.density == "zero" for j in jobs), 40)
        self.assertEqual(sum(j.group == "M11" for j in jobs), 250)
        self.assertEqual(sum(j.seed == 3405 and j.domain == 1 for j in jobs), 52)

    def test_production_proposal_and_acceptance(self):
        b = self.budgets(self.job)
        delta, info = propose(self.problem, b, PLAN["solver"])
        expected, expected_info = solve(self.problem)
        torch.testing.assert_close(delta, expected, rtol=0, atol=0)
        self.assertEqual(info, expected_info)
        actual, trials = accept(self.problem, delta, info["directional_derivative"], self.sigma, b)
        expected, expected_trials = backtrack(self.problem, delta, info["directional_derivative"], self.sigma)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(trials, expected_trials)

    def test_margin_jacobian_after_residual_update(self):
        vector = self.problem.base + .2 * torch.randn_like(self.problem.base)
        p = self.problem.params
        actual = margin_jacobian(p, vector, self.source)
        def margin(v):
            z = p.logits(v, self.source)
            return z[:, 0] - z[:, 1]
        torch.testing.assert_close(actual, jacrev(margin)(vector), rtol=1e-12, atol=1e-12)

    def test_protection_zero_and_real_constraint_solve(self):
        for group, epsilon in (("M10", None), ("M11", .003)):
            b = self.budgets(replace(self.job, group=group, epsilon=epsilon))
            zero = TsbaProblem(self.model, self.source[:8], self.labels[:8],
                               replace(self.bounds, zeta=torch.zeros_like(self.bounds.zeta)), self.cfg)
            delta, info = propose(zero, b, PLAN["solver"])
            self.assertEqual(info["reason"], "stationary")
            self.assertFalse(bool(delta.any()))
            self.assertTrue(b.check(zero.params, zero.base)["risk_ok"])
            self.assertTrue(b.check(zero.params, zero.base)["confidence_ok"])
            delta, info = propose(self.problem, b, PLAN["solver"])
            self.assertTrue(info["valid"], info)
            self.assertTrue(AffineBudgets(self.problem, b).feasible(delta.numpy()))
            if epsilon is not None:
                self.assertEqual(info["solver"], "SLSQP")
            candidate, trials = accept(self.problem, delta, info["directional_derivative"], self.sigma, b)
            self.assertIsNotNone(candidate, trials)
            actual = b.check(self.problem.params, candidate)
            self.assertTrue(actual["risk_ok"] and actual["confidence_ok"])

    def test_nonlinear_confidence_check_catches_affine_miss(self):
        with torch.no_grad():
            self.model.U1.weight.zero_()
            self.model.U1.bias.zero_()
        job = replace(self.job, group="M01", epsilon=.1)
        budgets = self.budgets(job)
        p = TsbaProblem(self.model, self.source, self.labels, self.bounds, self.cfg)
        delta = torch.zeros_like(p.base)
        blocks = p.params.mapping(delta)
        blocks["U1.bias"].fill_(3)
        blocks["U2.weight"][0].fill_(.5)
        self.assertTrue(AffineBudgets(p, budgets).feasible(delta.numpy()))
        self.assertFalse(budgets.check(p.params, p.base + delta)["confidence_ok"])
        before_rng = torch.random.get_rng_state()
        candidate, trials = accept(p, delta, -1e12, self.sigma, budgets)
        self.assertIsNone(candidate)
        self.assertFalse(trials[0]["confidence_ok"])
        torch.testing.assert_close(p.params.vector(), p.base, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), before_rng, rtol=0, atol=0)

    def test_relocation_on_known_affine_root(self):
        p = self.problem
        delta = torch.zeros_like(p.base)
        p.params.mapping(delta)["linear.bias"][0] = .03
        audit = relocate(p.params, p.base + delta, self.bounds, self.sigma, self.cfg, PLAN)
        for i, row in enumerate(audit["intersections"]):
            self.assertEqual(row["status"], "single_root")
            expected = -float((self.bounds.old_margin[i] + .03) / self.bounds.normal_norm[i])
            self.assertAlmostEqual(row["displacement"], expected, places=7)
            self.assertAlmostEqual(row["slope_ratio"], 1.)
            self.assertAlmostEqual(row["normal_cosine"], 1.)

    def test_target_gate_requires_complete_observations(self):
        import hashlib
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                target_values(root, [1, 3])
            label_path = root / "labels.txt"
            label_path.write_bytes(b"0\n1\n")
            write_json(root / "inputs/domains.json", {"1": [0], "3": [1]})
            write_json(root / "inputs/label_records.json", dict(
                path=str(label_path), offsets=[[0, 2], [2, 2]],
                sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(), mapping={0: 0, 1: 1}))
            write_json(root / "replay.json", {"fixture": True})
            seal(root, "R3_OBS", ["inputs", "replay.json"])
            with self.assertRaises(PermissionError):
                target_values(root, [1, 3])
            (root / "inputs/domains.json").write_text('{"1":[1],"3":[0]}')
            with self.assertRaises(PermissionError):
                target_values(root, [1, 3])

    def test_side_candidates_and_resume_reproduce_uninterrupted_trajectory(self):
        plan = dict(PLAN, endpoints=[0, 1, 3], side_attempts=[1, 3],
                    adapt=dict(PLAN["adapt"], adapt_rounds=3))
        job = replace(self.job, density="kde", group="M11", epsilon=.25)
        x = torch.cat((self.source, self.target, self.target[:4]))
        fit = list(range(20))
        target = list(range(20, 32))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("fresh", "resumed"):
                path = root / name
                write_json(path / "inputs/domains.json", {"1": target})
                write_json(path / "roles/seed3405.json", dict(fit=fit, validation=list(range(32, 36))))
                atomic_tensor(path / "source_labels/seed3405_fit.pt", self.labels)
                atomic_tensor(path / "source/seed3405/frozen.pt", dict(x=x, initial_head=self.model.state_dict()))
                atomic_tensor(path / "source/seed3405/geometry_d1.pt", dict(sigma=1.))
                atomic_tensor(path / "batches/seed3405_d1.pt", batches(fit, target, 3405, rounds=3, size=8))
            trajectory(root / "fresh", job, plan)
            trajectory(root / "resumed", job, plan, stop_after=1)
            published = root / "resumed" / "trajectories" / job.key / "round01.pt"
            inode = published.stat().st_ino
            trajectory(root / "resumed", job, plan)
            self.assertEqual(published.stat().st_ino, inode)
            for attempt in (1, 2, 3):
                relative = Path("trajectories") / job.key / f"round{attempt:02d}.pt"
                first, second = [load(root / name / relative) for name in ("fresh", "resumed")]
                self.assertEqual(first["status"], second["status"])
                torch.testing.assert_close(first["after"], second["after"], rtol=0, atol=0)
                if attempt in (1, 3):
                    self.assertEqual(first["side"]["status"], "available")
                    self.assertEqual(first["side"]["alpha"], second["side"]["alpha"])
                    self.assertEqual(set(first["side"]["candidates"]), {-1, 0, 1})

    def test_passive_failure_does_not_cancel_an_accepted_update(self):
        p = self.problem.params
        initial = p.vector()
        budgets = self.budgets(self.job)
        batch = dict(source_positions=list(range(8)), target_positions=list(range(12)))
        args = (self.model, batch, self.source, self.labels, self.target, torch.ones_like,
                self.sigma, budgets, self.job, self.cfg, PLAN, {"target": self.target})
        first = one_attempt(*args, side=True)
        self.assertTrue(first["accepted"])
        p.commit(initial)
        with patch("experiments.mutagenicity.r3_risk_and_confidence_constraints_20261112.src.trajectory.common_side",
                   side_effect=NumericalError("injected side error")):
            failed = one_attempt(*args, side=True)
        self.assertEqual(failed["side"]["status"], "numerical_failure")
        self.assertTrue(failed["accepted"])
        torch.testing.assert_close(p.vector(), first["after"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
