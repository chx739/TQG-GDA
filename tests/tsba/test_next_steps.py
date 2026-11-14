"""Synthetic engineering checks for the research-only R1b one-step runner."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.func import jacrev

from tsba.boundary import BoundaryBatch
from tsba.classifier import ResidualClassifier
from tsba.config import Config
from tsba.problem import TsbaProblem
from tsba.score import ScoreNetwork, denoising_loss, train_score
from tsba.subproblem import GGNSystem
from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.steps import (
    EmpiricalScore, _candidate_view, displacement_request, fit_score, old_boundaries,
    propose_arms, run_steps, score_comparison, shared_small_alpha,
)


torch.set_num_threads(1)


def fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = ResidualClassifier(2, 2, 3, dtype=torch.float64)
        with torch.no_grad():
            model.linear.weight.copy_(torch.tensor([[.5, 0.], [-.5, 0.]], dtype=torch.float64))
            model.linear.bias.zero_()
        source = torch.randn(12, 2, dtype=torch.float64)
    labels = (source[:, 0] < 0).long()
    labels[:3] = 1 - labels[:3]
    points = torch.tensor([[0., -.7], [0., .2], [0., .8]], dtype=torch.float64)
    normals = torch.tensor([[1., 0.]] * len(points), dtype=torch.float64)
    boundaries = BoundaryBatch(points, torch.tensor([[0, 1]] * len(points)),
                               torch.ones(len(points), dtype=torch.float64),
                               torch.zeros(len(points), dtype=torch.float64),
                               torch.zeros(len(points), dtype=torch.float64), normals, {})
    return model, source, labels, boundaries


class NextStepTests(unittest.TestCase):
    def test_exact_kde_score_matches_log_density_gradient_and_far_tail(self):
        centers = torch.tensor([[-1., .3], [0., .5], [2., -.2]], dtype=torch.float64)
        query = torch.tensor([[.2, -.1], [80., -90.]], dtype=torch.float64, requires_grad=True)
        kde = EmpiricalScore(centers, .7, chunk=1)
        gradient, = torch.autograd.grad(kde.log_density(query).sum(), query)
        torch.testing.assert_close(kde(query), gradient, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.isfinite(kde(query)).all())
        # At the distant query almost all floating-point kernel masses underflow;
        # stable weights retain the dominant center and a finite score.
        torch.testing.assert_close(kde(query)[1], (centers[2] - query[1]) / .7**2,
                                   rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            EmpiricalScore(centers, 0.)

    def test_score_training_is_bitwise_production_parity_and_saves_epoch_state(self):
        x = torch.arange(38, dtype=torch.float64).reshape(19, 2) / 20 - .8
        cfg = Config(seed=47, score_epochs=4, score_batch=8, score_hidden=64)
        before = torch.random.get_rng_state().clone()
        expected, expected_info = train_score(x, .3, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            observed, info = fit_score(Path(tmp) / "score", x, .3, cfg)
            self.assertEqual(info["status"], "success", info)
            self.assertEqual(info["updates_completed"], 12)
            self.assertEqual(info["selected_epoch"], expected_info["selected_epoch"])
            self.assertEqual(info["noise_validation_loss"], expected_info["noise_validation_loss"])
            history = json.loads((Path(tmp) / "score/history.json").read_text())
            self.assertEqual(history, expected_info["history"])
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(observed.state_dict()[key], value, rtol=0, atol=0)
            files = sorted((Path(tmp) / "score/epochs").glob("*.pt"))
            self.assertEqual(len(files), 4)
            last = torch.load(files[-1], weights_only=False)
            self.assertEqual(last["updates_completed"], 12)
            self.assertEqual(len(last["history"]), 4)
            self.assertEqual(tuple(last["validation_noise"].shape), tuple(x.shape))
            self.assertTrue(last["optimizer_state"]["state"])
            # Resume the preceding epoch's Adam and generator state and reproduce
            # the final epoch exactly, including its minibatch/noise sequence.
            previous = torch.load(files[-2], weights_only=False)
            with torch.random.fork_rng(devices=[]):
                resumed = ScoreNetwork(2, 64).double()
            resumed.load_state_dict(previous["model_state"])
            optimizer = torch.optim.Adam(resumed.parameters(), lr=cfg.score_lr)
            optimizer.load_state_dict(previous["optimizer_state"])
            rng = torch.Generator().set_state(previous["training_rng_state"])
            order = torch.randperm(len(x), generator=rng)
            for ids in order.split(cfg.score_batch):
                noise = torch.randn(x[ids].shape, generator=rng, dtype=x.dtype)
                loss = denoising_loss(resumed, x[ids], .3, noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for key, tensor in resumed.state_dict().items():
                torch.testing.assert_close(tensor, last["model_state"][key], rtol=0, atol=0)
            self.assertTrue(torch.equal(rng.get_state(), last["training_rng_state"]))
            self.assertTrue((Path(tmp) / "score/selected.pt").exists())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_matched_source_keeps_same_matrix_and_paired_decomposition(self):
        model, source, labels, boundary = fixture()
        cfg = Config(solver="direct")
        zeta = torch.tensor([.015, -.02, .01], dtype=torch.float64)
        before = copy.deepcopy(model.state_dict())
        proposals, error = propose_arms(model, source, labels, boundary, zeta, cfg)
        self.assertLess(error, 1e-12)
        matrices = [GGNSystem(v["problem"]).dense() for v in proposals.values()]
        for matrix in matrices[1:]:
            torch.testing.assert_close(matrix, matrices[0], rtol=0, atol=0)
        source_system = GGNSystem(proposals["matched_source"]["problem"])
        expected = torch.linalg.solve(matrices[0], -source_system.gs)
        torch.testing.assert_close(proposals["matched_source"]["delta"], expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(float(expected.norm()), 0)
        self.assertEqual(proposals["zero"]["solver"]["reason"], "zero_requests")
        torch.testing.assert_close(proposals["zero"]["delta"], torch.zeros_like(expected), rtol=0, atol=0)
        for key, tensor in before.items():
            torch.testing.assert_close(tensor, model.state_dict()[key], rtol=0, atol=0)

    def test_zero_views_exact_noop_and_accepted_views_archive_true_movement(self):
        model, source, labels, boundary = fixture()
        cfg = Config(solver="direct")
        proposals, _ = propose_arms(model, source, labels, boundary,
                                   torch.tensor([.01, -.02, .01], dtype=torch.float64), cfg)
        zero = proposals["zero"]
        d = jacrev(zero["problem"].normalized_margin)(zero["problem"].base)
        with torch.no_grad():
            logits = model(source)
        archived = dict(probabilities=logits.softmax(-1), log_probabilities=logits.log_softmax(-1))
        pools = {"source_fit": list(range(len(source))), "B1": [2, 4, 6]}
        for view in ("small_step", "production_backtrack"):
            payload, metadata = _candidate_view(zero, view, 1 / 16, 1., d, source, pools, archived)
            self.assertTrue(metadata["no_op"])
            self.assertFalse(metadata["accepted"])
            torch.testing.assert_close(payload["vector"], zero["problem"].base, rtol=0, atol=0)
            torch.testing.assert_close(payload["probabilities"]["B1"]["probabilities"],
                                       archived["probabilities"][[2, 4, 6]], rtol=0, atol=0)
        normal = proposals["normal"]
        payload, metadata = _candidate_view(normal, "small_step", 1 / 16, 1., d, source, pools, archived)
        self.assertTrue(metadata["accepted"])
        torch.testing.assert_close(payload["actual_margin_movement"], normal["problem"].movement(payload["vector"]))
        torch.testing.assert_close(payload["predicted_margin_movement"], d @ normal["delta"] / 16)
        restored = copy.deepcopy(model)
        restored.load_state_dict(payload["model_state"])
        torch.testing.assert_close(restored(source).softmax(-1),
                                   payload["probabilities"]["source_fit"]["probabilities"])
        _, unavailable = _candidate_view(normal, "small_step", None, 1., d, source, pools, archived)
        self.assertEqual(unavailable["status"], "failed")
        _, independent = _candidate_view(normal, "production_backtrack", None, 1., d, source, pools, archived)
        self.assertEqual(independent["status"], "success")
        _, zero_without_alpha = _candidate_view(zero, "small_step", None, 1., d, source, pools, archived)
        self.assertEqual(zero_without_alpha["status"], "success")
        invalid = dict(normal, solver={"valid": False, "reason": "inaccurate_or_non_descent"})
        rejected, rejection = _candidate_view(invalid, "production_backtrack", None, 1., d, source, pools, archived)
        self.assertEqual(rejection["status"], "success")
        self.assertEqual(rejection["reason"], "inaccurate_or_non_descent")
        torch.testing.assert_close(rejected["vector"], normal["problem"].base, rtol=0, atol=0)

    def test_request_sign_and_units_fixed_independent_of_bandwidth(self):
        normals = torch.tensor([[1., 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
        score = torch.tensor([[.3, 5.], [-.2, 5.], [10., 5.]], dtype=torch.float64)
        gamma, request = displacement_request(score, normals, 2.)
        torch.testing.assert_close(gamma, torch.tensor([.3, -.2, 10.], dtype=torch.float64))
        torch.testing.assert_close(request, torch.tensor([.12, -.08, .2], dtype=torch.float64))
        self.assertTrue(torch.equal(request.sign(), gamma.sign()))
        summary = score_comparison(score, score, normals, 2.)
        self.assertEqual(summary["normal_score_rmse_sigma"], 0.)
        self.assertEqual(summary["request_rmse_sigma"], 0.)
        self.assertEqual(summary["normal_sign_agreement"], 1.)
        tiny = score_comparison(torch.zeros_like(score), score, normals, 2.)
        self.assertEqual(tiny["normal_sign_threshold_denominator"], 0)
        self.assertIsNone(tiny["normal_sign_agreement_above_threshold"])

    def test_single_alpha_uses_all_four_combinations_and_three_solved_arms(self):
        groups = [{arm: dict(delta=torch.tensor([float(k + 1), 0.], dtype=torch.float64),
                             solver={"valid": True})
                   for arm in ("normal", "reverse", "matched_source")} for k in range(4)]
        groups[-1]["matched_source"]["delta"][0] = 5.
        alpha, info = shared_small_alpha(groups, torch.eye(2, dtype=torch.float64), 1.)
        self.assertEqual(alpha, 1 / 64)
        self.assertEqual(info["halvings"], 2)
        self.assertEqual(info["reason"], "motion_guard_halving")
        self.assertLessEqual(info["max_predicted_movement_sigma"], .1)
        groups[-1]["matched_source"]["delta"][0] = 1e9
        alpha, info = shared_small_alpha(groups, torch.eye(2, dtype=torch.float64), 1.)
        self.assertIsNone(alpha)
        self.assertEqual(info["reason"], "small_step_floor_exhausted")

    def test_old_bank_ignores_density_and_support_filters(self):
        model, _, _, boundaries = fixture()
        rows = [dict(origin_id=i, success=True, point=p, normal=n, normal_norm=1.,
                     panels=[], density_failure="sparse", ess_center=.001, support={"n": 0})
                for i, (p, n) in enumerate(zip(boundaries.points, boundaries.normals))]
        rows.append(dict(origin_id=4, success=False, failure="old_projection_failed"))
        result, rejected = old_boundaries({"anchors": rows}, model, 1.)
        self.assertEqual(len(result), 3)
        self.assertEqual(rejected, [dict(origin_id=4, position=3, reason="old_projection_failed")])

    def test_OBS_closes_entry_before_read_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seals").mkdir()
            (root / "seals/OBS.json").write_text("{}")
            with self.assertRaises(PermissionError):
                run_steps(root, {}, {})
            self.assertFalse((root / "steps").exists())

    def test_extra_source_labels_refused_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "deps/roles").mkdir(parents=True)
            labels = {}
            for seed in range(3405, 3410):
                (root / "deps/roles" / f"seed{seed}.json").write_text(json.dumps({"fit_ids": list(range(779))}))
                labels[seed] = {i: i % 2 for i in range(779)}
            labels[3408][900] = 0
            with self.assertRaises(PermissionError):
                run_steps(root, {"seeds": list(range(3405, 3410)), "domains": [1, 3]}, labels)
            self.assertFalse((root / "steps").exists())

    def test_full_archive_grid_with_synthetic_dependencies(self):
        # Exercise the archive orchestration without real inputs or 20 new fits.
        def inputs(root, seed, domain, labels_by_id, cfg):
            model, source, labels, boundaries = fixture()
            boundaries.input_indices = torch.arange(3)
            x = source.repeat(90, 1)
            with torch.no_grad():
                logits = model(x)
            archived = dict(probabilities=logits.softmax(-1), log_probabilities=logits.log_softmax(-1))
            source_path = root / "deps/source" / f"seed{seed}"
            if not source_path.exists():
                source_path.mkdir(parents=True)
                torch.save(dict(x=x, initial_head=model.state_dict()), source_path / "frozen.pt")
                torch.save(dict(archived, ids=torch.arange(len(x))), source_path / "predictions.pt")
            problem = TsbaProblem(model, source, labels, boundaries, cfg)
            system = GGNSystem(problem)
            return dict(seed=seed, domain=domain, x=x, model=model, source=source, labels=labels,
                        fit_ids=list(range(12)), target_ids=list(range(1025)),
                        bank={"origin_ids": list(range(64))}, boundaries=boundaries, rejected=[],
                        sigma0=1., pools={f"B{domain}": [14, 15], "source_fit": list(range(12)),
                                         "source_validation": [12, 13], "source_diagnostic": [16, 17],
                                         "source_all": list(range(14)) + [16, 17]},
                        archived=archived, cfg=cfg, matrix=system.dense(),
                        d=jacrev(problem.normalized_margin)(problem.base).detach(),
                        source_gradient=system.gs, source_rhs=-system.gs,
                        dependency_hashes={}, combinations={})

        class SyntheticScore(torch.nn.Module):
            def forward(self, x):
                return torch.ones_like(x) * .12

        def synthetic_fit(directory, x, sigma, cfg, resources, log):
            Path(directory).mkdir(parents=True)
            return SyntheticScore(), dict(status="success", seed=cfg.seed, updates_completed=900,
                                          epochs_completed=100, selected_epoch=32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "deps/roles").mkdir(parents=True)
            labels = {}
            for seed in range(3405, 3410):
                (root / "deps/roles" / f"seed{seed}.json").write_text(json.dumps({"fit_ids": list(range(779))}))
                labels[seed] = {i: i % 2 for i in range(779)}
            prefix = "experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.steps."
            with patch(prefix + "_cell_inputs", side_effect=inputs), patch(prefix + "fit_score", side_effect=synthetic_fit):
                result = run_steps(root, {"seeds": list(range(3405, 3410)), "domains": [1, 3],
                                          "resources": {}}, labels)
            self.assertEqual(len(result["steps"]), 160)
            self.assertEqual(len(result["scores"]), 20)
            self.assertTrue(all(r["status"] == "success" for r in result["steps"]))
            self.assertEqual(set(result["steps"][0]["views"]), {"small_step", "production_backtrack"})
            for seed in range(3405, 3410):
                for domain in (1, 3):
                    records = [r for r in result["steps"] if r["seed"] == seed and r["domain"] == domain]
                    self.assertEqual(len(set(r["common_alpha"] for r in records)), 1)
            record = result["steps"][2]
            self.assertEqual(record["arm"], "zero")
            payload = torch.load(root / record["path"], weights_only=False)
            for view in payload["views"].values():
                self.assertTrue(view["no_op"])
                self.assertIn("B1", view["probabilities"])
                for key, tensor in payload["base_state"].items():
                    torch.testing.assert_close(view["model_state"][key], tensor, rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(payload["source_proposal_delta"])), 0)
            self.assertEqual(int(torch.count_nonzero(payload["score_proposal_delta"])), 0)
            system = torch.load(root / "steps/seed3405_domain1/system.pt", weights_only=False)
            torch.testing.assert_close(system["source_ggn"] + system["d"].T @ system["d"] / len(system["d"])
                                       + .01 * torch.eye(len(system["matrix"]), dtype=torch.float64), system["matrix"])
            # The actual reporting/replay consumers must accept this exact schema.
            from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.evaluate import evaluate_steps, replay
            plan = dict(seeds=list(range(3405, 3410)), domains=[1, 3])
            replayed = replay(root, plan)
            self.assertEqual(replayed["step_view_pools"], 1600)
            (root / "evaluation").mkdir()
            b_labels = {domain: {14: 0, 15: 1} for domain in (1, 3)}
            evaluated = evaluate_steps(root, plan, b_labels, labels)
            self.assertEqual(len(evaluated["rows"]), 1600)
            for row in evaluated["rows"]:
                if row["arm"] == "zero":
                    self.assertEqual(row["nll_delta_source"], 0.)
                    self.assertEqual(row["prediction_flips"], 0)
            # Failure of one learned fit must not hide empirical production or
            # source-only controls behind the unavailable shared small step.
            failed_root = root / "failed_run"
            (failed_root / "deps/roles").mkdir(parents=True)
            for seed in range(3405, 3410):
                (failed_root / "deps/roles" / f"seed{seed}.json").write_text(json.dumps({"fit_ids": list(range(779))}))
            def one_failed_fit(directory, x, sigma, cfg, resources, log):
                if cfg.seed == 3405 and "domain1" in str(directory) and sigma == 1.:
                    Path(directory).mkdir(parents=True)
                    return None, dict(status="failed", seed=cfg.seed, failure="synthetic_time_limit")
                return synthetic_fit(directory, x, sigma, cfg, resources, log)
            with patch(prefix + "_cell_inputs", side_effect=inputs), patch(prefix + "fit_score", side_effect=one_failed_fit):
                partial = run_steps(failed_root, {"seeds": list(range(3405, 3410)), "domains": [1, 3],
                                                  "resources": {}}, labels)
            records = [r for r in partial["steps"] if r["seed"] == 3405 and r["domain"] == 1]
            empirical = next(r for r in records if r["estimator"] == "empirical" and r["arm"] == "normal")
            self.assertEqual(empirical["status"], "success")
            self.assertEqual(empirical["views"]["small_step"]["status"], "failed")
            self.assertEqual(empirical["views"]["production_backtrack"]["status"], "success")
            zero = next(r for r in records if r["estimator"] == "learned" and r["arm"] == "zero" and r["bandwidth_multiplier"] == 1.)
            self.assertFalse(zero["score_estimator_available"])
            self.assertEqual(zero["views"]["small_step"]["status"], "success")
            self.assertEqual(zero["views"]["production_backtrack"]["status"], "success")
            replayed = replay(failed_root, plan)
            self.assertLess(replayed["step_view_pools"], 1600)
            (failed_root / "evaluation").mkdir()
            evaluated = evaluate_steps(failed_root, plan, b_labels, labels)
            self.assertEqual(len(evaluated["rows"]), 1600)
            empirical_production = [r for r in evaluated["rows"] if r["seed"] == 3405 and r["domain"] == 1
                                    and r["estimator"] == "empirical" and r["view"] == "production_backtrack"]
            self.assertTrue(all(r["status"] == "success" for r in empirical_production))


if __name__ == "__main__":
    unittest.main()
