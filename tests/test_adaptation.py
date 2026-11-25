"""Constructive checks for the four identification/transfer claims and prior moments.

Run with python -m unittest discover -s tests -p test_adaptation.py.
Candidate fixtures have one edge per occupied (group, token) cell unless stated.
"""

import json
import math
import unittest

import numpy as np
import torch
from scipy.special import softmax

from cela.adaptation import (
    _helmert, _moment_terms, _order_response, _response_diagnostics, _structural_risk,
    fit_modes, fit_prior, fit_structure, prefix_moments,
)
from cela.model import Trace, edge_log_probs


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float64)


def trace(mass, scores, choices, prior=None):
    weights, a = tensor(mass), tensor(scores)
    n, (k, j) = len(choices), zip(*choices) if choices else ([], [])
    lw = torch.full((n, len(weights), 10), -torch.inf, dtype=torch.float64)
    lw[:, :, :weights.shape[1]] = weights.log()
    k, j = torch.tensor(k, dtype=torch.long), torch.tensor(j, dtype=torch.long)
    return Trace(tensor(np.zeros(a.shape[-1]) if prior is None else prior), lw,
                 a.expand(n, -1, -1).clone(), k, j, lw[torch.arange(n), k, j].clone())


ZERO = torch.zeros(10, dtype=torch.float64)
GLOBAL = torch.zeros(10, dtype=torch.long)
IDS = torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])


class StructuralTests(unittest.TestCase):
    def test_graph_order_weights_and_unselected_groups(self):
        mass = [[1., 1., 0.], [0., 1., 1.]]
        a = trace(mass, [[0., 0.], [0., 0.]], [(0, 1)])
        b = trace(mass, [[9., -4.], [1., 3.]], [(0, 0)] * 3)
        eta, ids, d = fit_structure([[a] * 3, [b]], ridge=0, tol=1e-11,
                                    bootstrap_samples=4, seed=37)
        self.assertAlmostEqual(float(eta[1] - eta[0]), -math.log(3), places=7)
        self.assertAlmostEqual(d["mean_edges"], 2.)
        self.assertAlmostEqual(d["conditional_nll_zero"], 2 * math.log(2))
        self.assertEqual(d["effective_dimension"], 1)
        self.assertEqual(d["pair_graph_support"][0][1], 2)
        self.assertEqual(d["pair_step_support"][0][1], 6)
        self.assertEqual(d["pair_step_support"][1][2], 0)
        self.assertEqual(ids.tolist(), [0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(eta[2:].abs().sum().item(), 0.)
        self.assertTrue(d["finite_unregularized_optimum"])
        self.assertEqual(d["graph_bootstrap"]["eta_std"][2], 0.)
        self.assertAlmostEqual(d["positive_eigenvalues"][0], 2 * 2 * .25 * .75)
        json.dumps(d, allow_nan=False)

    def test_theorem1_common_true_odds_under_different_group_frequencies(self):
        # Within either group, a three-fold odds tilt gives rational true probabilities.
        # Replicating either group changes the design distribution and leaves this optimum fixed.
        mass = [[1., 1.], [3., 1.]]
        a = trace(mass, [[-4., 7.], [8., -2.]], [(0, 1)] * 3 + [(0, 0)])
        b = trace(mass, [[10., 0.], [-7., 3.]], [(1, 1), (1, 0)])
        for na, nb in [(1, 5), (5, 1)]:
            pool = [[a]] * na + [[b]] * nb
            eta, _, _ = fit_structure(pool, ridge=0, tol=1e-12)
            self.assertAlmostEqual(float(eta[1] - eta[0]), math.log(3), places=7)
        # Structural estimation uses none of the source class fields.
        class StructuralTrace:
            true_j, true_k, log_w = a.true_j, a.true_k, a.log_w
        result, _, _ = fit_structure([[StructuralTrace()]], ridge=0, tol=1e-10)
        self.assertAlmostEqual(float(result[1] - result[0]), math.log(3), places=7)

    def test_theorem2_gradient_hessian_and_nullspace(self):
        tr = trace([[1, 2, 0, 0], [0, 0, 3, 1]], [[0, 0], [0, 0]],
                   [(0, 0), (1, 2), (1, 3), (0, 1), (0, 1)])
        eta, ids, d = fit_structure([[tr]], ridge=.01, tol=1e-11)
        np.testing.assert_array_equal(ids, IDS)
        self.assertEqual(d["numerical_rank"], 2)
        weight = np.ones(5)
        mass = tr.log_w[torch.arange(5), tr.true_k].numpy()
        x, obs = eta.numpy(), tr.true_j.numpy()
        value, grad, hess = _structural_risk(mass, obs, weight, x, True)
        for i in range(10):
            e = np.eye(10)[i] * 1e-5
            plus, gp = _structural_risk(mass, obs, weight, x + e)
            minus, gm = _structural_risk(mass, obs, weight, x - e)
            self.assertAlmostEqual((plus - minus) / 2e-5, grad[i], places=8)
            np.testing.assert_allclose((gp - gm) / 2e-5, hess[:, i], atol=2e-10)
        np.testing.assert_allclose(hess, d["information_matrix"], atol=1e-14)
        delta = np.array([5, 5, -2, -2, 3, 2, 1, 0, -4, 7])
        self.assertAlmostEqual(_structural_risk(mass, obs, weight, x + delta)[0], value)
        for l in ids.unique():
            np.testing.assert_allclose(hess @ (ids == l).double().numpy(), 0., atol=1e-14)

    def test_topological_weak_comparisons_and_solver_status(self):
        tr = trace([[1., 1.]], [[0., 0.]], [(0, 0)])
        tr.log_w[0, 0, 1] = -1000.  # finite log-mass, even though exp(-1000) underflows
        eta, ids, d = fit_structure([[tr]])
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(d["effective_dimension"], 1)
        self.assertEqual(d["numerical_rank"], 0)
        self.assertFalse(d["finite_unregularized_optimum"])
        _, _, unregularized = fit_structure([[tr]], ridge=0)
        self.assertEqual(unregularized["status"], "no_finite_unregularized_optimum")
        _, _, capped = fit_structure([[trace([[1, 1]], [[0, 0]], [(0, 0)] * 4)]],
                                     max_iter=1, tol=1e-12)
        self.assertFalse(capped["converged"])

    def test_isolated_tokens_and_empty_graph_budget(self):
        tr = trace([[1, 0], [0, 1]], [[1., -1.], [-1., 1.]], [(0, 0)])
        empty = trace([[1]], [[0., 0.]], [])
        for pool in [[[tr]], [[empty]]]:
            eta, ids, diag = fit_structure(pool)
            torch.testing.assert_close(eta, ZERO, atol=0, rtol=0)
            self.assertEqual(ids.tolist(), list(range(10)))
            self.assertEqual(diag["status"], "no_comparisons")
        modes, d = fit_modes([[empty]])
        self.assertEqual(d["prior_only"]["status"], "no_calibration_information")
        json.dumps(d, allow_nan=False)
        self.assertEqual(modes["prior_calibrated"][2].tolist(), [0., 0.])


class PredictionTests(unittest.TestCase):
    def test_dominant_token_response_retains_rare_probability_difference(self):
        tr = trace([[1., 0.], [0., 1e-20]], [[0., 0.], [0., math.log(2.)]], [(0, 0)])
        information = np.zeros((10, 10))
        information[:2, :2] = [[.25, -.25], [-.25, .25]]
        pair = _order_response(tr, ZERO.numpy(), IDS.numpy())[2][0]
        np.testing.assert_allclose(pair[:2], [-1e-20, 1e-20], rtol=1e-13, atol=0)
        result = _response_diagnostics([[tr]], ZERO.numpy(), IDS.numpy(),
                                       {"information_matrix": information.tolist()}, 2)["0/1"]
        self.assertAlmostEqual(result["mean_order_local_information_sensitivity"]["median"] / 4e-40,
                               1., places=12)

    def test_weak_component_sensitivity_uses_its_own_information_scale(self):
        # With one choice per component per graph, the two contrast eigenvalues
        # at eta=0 are .5 and 1e-20. The query response lives in the weaker one.
        mass = [[1., 1., 0., 0.], [0., 0., 1., 5e-21]]
        design = [[trace(mass, [[0., 0.]]*2, [(0, observed), (1, 2)])] for observed in (0, 1)]
        eta, ids, diag = fit_structure(design)
        self.assertEqual(diag["numerical_rank"], 2)
        np.testing.assert_allclose(diag["positive_eigenvalues"], [1e-20, .5], rtol=1e-12, atol=0)
        tr = trace([[0., 0., 1., 0.], [0., 0., 0., 1.]], [[1., 0.], [0., 1.]], [(0, 2)])
        result = _response_diagnostics(iter([[tr]]), eta.numpy(), ids.numpy(), diag, 2)["0/1"]
        actual = result["mean_order_local_information_sensitivity"]["median"]
        expected = 2 * math.tanh(.5)**2 / 1e-20
        self.assertAlmostEqual(actual / expected, 1., places=12)
        self.assertEqual(result["sensitivity_unresolved_orders"], 0)
        self.assertEqual(result["sensitivity_resolved_graphs"], 1)

    def test_unresolved_component_direction_with_response_is_counted(self):
        # Here the strong and very weak conductances share a single component.
        # The second contrast is below the relative precision of that block.
        info = np.zeros((10, 10))
        for j, k, weight in [(0, 1, .25), (1, 2, 1e-24)]:
            e = np.eye(10)[j] - np.eye(10)[k]
            info += weight * np.outer(e, e)
        ids = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
        mass = [[1., 1., 0.], [0., 0., 1.]]
        active = trace(mass, [[1., 0.], [0., 1.]], [(0, 0)])
        passive = trace(mass, [[0., 0.], [0., 0.]], [(0, 0)])
        diag = {"information_matrix": info.tolist()}
        response = _response_diagnostics([[active, passive], [passive]], ZERO.numpy(), ids, diag, 2)["0/1"]
        self.assertIsNone(response["mean_order_local_information_sensitivity"])
        self.assertEqual(response["sensitivity_unresolved_orders"], 1)
        self.assertEqual(response["sensitivity_unresolved_graphs"], 1)
        self.assertEqual(response["sensitivity_resolved_graphs"], 1)
        json.dumps(response, allow_nan=False)

    def test_theorem3_cross_component_groups_and_class_elimination(self):
        # This new state crosses A's components inside each group. Within a (group,
        # component) the edge probabilities conditioned on that state eliminate class.
        mass = [[2., 1., 4., .3], [.4, 2., 1., 5.]]
        scores = [[.7, -.5, 1.1], [1.3, 2.1, -.3]]
        eta = tensor([-.3, .3, .7, -.7] + [0.]*6)
        delta = tensor([5., 5., -2., -2.] + [3.]*6)
        probs, tilted, shifted, v2, v2_shifted = [], [], [], [], []
        for k in range(2):
            for j in range(4):
                tr = trace(mass, scores, [(k, j)])
                src = edge_log_probs(tr)[0].exp()
                torch.testing.assert_close(edge_log_probs(tr, ZERO, IDS)[0].exp(), src)
                probs.append(src)
                tilted.append(edge_log_probs(tr, eta, IDS)[0].exp())
                shifted.append(edge_log_probs(tr, eta + delta, IDS)[0].exp())
                v2.append(edge_log_probs(tr, eta, GLOBAL)[0].exp())
                v2_shifted.append(edge_log_probs(tr, eta + delta, GLOBAL)[0].exp())
        source, adapted = (torch.stack(arr).reshape(2, 4, 3) for arr in (probs, tilted))
        torch.testing.assert_close(adapted.sum((0, 1)), torch.ones(3, dtype=torch.float64))
        torch.testing.assert_close(torch.stack(shifted), torch.stack(tilted), atol=2e-14, rtol=0)
        for indices in [[0, 1], [2, 3]]:
            torch.testing.assert_close(source[:, indices].sum((0, 1)), adapted[:, indices].sum((0, 1)))
            conditional = adapted[:, indices] / adapted[:, indices].sum(1, keepdim=True)
            expected = (tensor(mass)[:, indices].log() + eta[indices]).softmax(1)
            torch.testing.assert_close(conditional, expected[:, :, None].expand_as(conditional))
        self.assertGreater((torch.stack(v2_shifted) - torch.stack(v2)).abs().max().item(), .01)
        tr = trace(mass, scores, [(0, 1), (1, 3)])
        _, matrix, obs = prefix_moments(tr, eta, IDS)
        for k in [0, 1]:
            for l in [0, 1]:
                torch.testing.assert_close(matrix[0, k*8 + l], adapted[k, 2*l:2*l+2].sum(0))
        self.assertEqual(obs.tolist(), [0, 9])

    def test_design_313_rank_reversal(self):
        scores = [[0., math.log(3)], [math.log(3), 0.]]
        eta = ZERO.clone()
        eta[:2] = tensor([-math.log(3)/2, math.log(3)/2])
        predictions = []
        for mass in [[[.05, .45], [.9, .1]], [[.9, .1], [.1, .9]]]:
            tr = trace(mass, scores, [(0, 0)])
            predictions.append([float(edge_log_probs(tr, offset, IDS)[0].softmax(0)[1]) for offset in (ZERO, eta)])
        np.testing.assert_allclose(predictions, [[.807692, .735294], [.75, .818182]], atol=5e-7)

    def test_theorem4_path_equalities_and_transfer_bound(self):
        # Population risk: two equally probable observed groups, each with two
        # tokens. Set expected choice frequencies exactly at the true parameter.
        logs = np.full((4, 10), -np.inf)
        logs[:2, :2] = np.log([.6, 1.2])
        logs[2:, 2:4] = np.log([.9, 1.7])
        j = np.arange(4)
        star = np.array([-.1, .1, .3, -.3] + [0.]*6)
        shift = np.array([-.4, .4, -.2, .2] + [0.]*6)
        weight = .5 * softmax(logs + star, axis=1)[j, j]
        endpoint = star + shift
        risk = lambda x: _structural_risk(logs, j, weight, x, True)
        regret = risk(endpoint)[0] - risk(star)[0]
        np.testing.assert_allclose(risk(star)[1], 0, atol=1e-15)
        query = trace([[1., .5, 3, .2], [.2, 2., 1., .2]], [[1., -.2], [-.8, .7]], [(0, 0)])
        def response(x):
            _, logp, pair_responses = _order_response(query, x, IDS.numpy())
            return logp, pair_responses[0]
        points, weights = np.polynomial.legendre.leggauss(24)
        info, b = np.zeros((10, 10)), np.zeros(10)
        for s, w in zip((points+1)/2, weights/2):
            x = star + s * shift
            info += 2 * w * (1-s) * risk(x)[2]
            b += w * response(x)[1]
        initial, final = response(star)[0], response(endpoint)[0]
        change = (final[0]-final[1]) - (initial[0]-initial[1])
        self.assertAlmostEqual(regret, .5 * shift @ info @ shift, places=13)
        self.assertAlmostEqual(change, b @ shift, places=13)
        bound = 2 * regret * (b @ np.linalg.pinv(info) @ b)
        self.assertGreater(change**2, 1e-6)
        self.assertLessEqual(change**2, bound + 1e-14)
        kl = np.exp(initial) @ (initial - final)
        self.assertLessEqual(kl, bound / 4 + 1e-14)
        for group in np.unique(IDS):
            self.assertAlmostEqual(b[IDS.numpy() == group].sum(), 0.)


class PriorTests(unittest.TestCase):
    def test_three_classes_two_occupied_states_cannot_calibrate(self):
        scores = [[0., 0., 0.], [-.4999999983976608, -.49999999993564487, -.4999999983232485]]
        tr = trace([[1.], [1.]], scores, [(0, 0)])
        nu, diag = fit_prior([[tr]], ZERO, IDS)
        self.assertEqual(diag["status"], "no_calibration_information")
        self.assertEqual(diag["S_omega"], 0.)
        np.testing.assert_array_equal(nu, np.zeros(3))

    def test_multiclass_rank_accounts_for_probability_roundoff_and_rare_states(self):
        # Three occupied states but only one reliable column contrast. Differences
        # of order eps from rounding the nearly equal columns cannot add rank.
        base = np.array([.31, .25, .44])
        movement = 1e-9 * np.array([-.1, .55, -.45])
        rank_one = base[:, None] + movement[:, None] * np.array([[0., .345, 1.]])
        U = _helmert(3)
        omega, _ = _moment_terms(rank_one[None, :, :], np.array([0]), U)
        self.assertEqual(omega[0], 0.)
        # Independent contrasts between two rare states remain visible despite the
        # dominant state's three probabilities all rounding to 1.
        matrix = np.array([[1., 1., 1.], [1e-20, 2e-20, 1e-20], [1e-20, 1e-20, 3e-20]])
        omega, _ = _moment_terms(matrix[None, :, :], np.array([0]), U)
        expected = np.array([[0., -1., -2.], [0., 1., 0.], [0., 0., 2.]]) @ U
        expected_omega = np.linalg.svd(expected, compute_uv=False)[-1]**2 * 1e-40
        self.assertAlmostEqual(omega[0] / expected_omega, 1., places=13)

    def test_binary_machine_endpoint_has_zero_iterations(self):
        tr = trace([[1.], [1e-20]], [[0., 0.], [0., math.log(2.)]], [(1, 0)])
        nu, diag = fit_prior([[tr]], ZERO, GLOBAL)
        self.assertEqual(diag["status"], "root_converged")
        self.assertEqual(diag["iterations"], 0)
        self.assertEqual(diag["moment_residual"], 0.)
        self.assertAlmostEqual(diag["S_B"] / 2e-40, 1., places=12)
        self.assertGreater(float(nu[1]), 1e20)

    def test_identifiable_rare_multiclass_direction_changes_calibrated_class(self):
        # In exact arithmetic the second row is 1-row0-row2. Reconstructing
        # that identity in float64 destroys the rare direction in the raw SVD.
        s = 1e-20
        matrix = np.array([[.2, .6, .2], [.8, .4, .8], [s, 2*s, 3*s]])
        basis = _helmert(3)
        omega, y = _moment_terms(matrix[None], np.array([0]), basis)
        self.assertAlmostEqual(omega[0] / (3 * s**2), 1., places=13)
        # The conditional moment vector is signed. M t = e_0 gives the target
        # t=(1/2,2,-3/2), so its direction as well as omega can be checked.
        target = np.array([.5, 2., -1.5])
        np.testing.assert_allclose(basis @ (y[0] / omega[0]) + 1/3, target, atol=1e-13)
        prior = np.array([.1, 0., -.1])
        tr = trace([[1.]]*3, np.log(matrix), [(0, 0)], prior=prior)
        nu, d = fit_prior([[tr]], ZERO, GLOBAL, ridge=.01, tol=1e-10)
        self.assertTrue(d["converged"])
        self.assertAlmostEqual(d["S_omega"] / (3 * s**2), 1., places=12)
        np.testing.assert_allclose(nu.numpy(), [50., 100., -150.], atol=1e-8)
        logits = prior + nu.numpy()
        np.testing.assert_allclose(softmax(logits) - target + .01 * nu.numpy(), 0., atol=1e-10)
        self.assertEqual((prior.argmax(), logits.argmax()), (0, 1))

    def test_binary_rounded_bracket_reports_nonzero_residual(self):
        s = 9.952444421688098e-21
        tr = trace([[1.], [s]], [[0., 0.], [0., math.log(2.)]], [(1, 0)])
        nu, d = fit_prior([[tr]], ZERO, GLOBAL, ridge=.01)
        self.assertTrue(d["converged"])
        self.assertEqual(d["iterations"], 0)
        self.assertEqual(d["status"], "root_at_float_resolution")
        self.assertNotEqual(d["moment_residual"], 0.)
        self.assertLessEqual(d["relative_moment_residual"], 2 * np.finfo(float).eps)
        self.assertAlmostEqual((nu[1]-nu[0]).item() / (200 / s), 1., places=12)

    def test_fit_modes_rejects_failed_or_nonfinite_fits(self):
        separated = trace([[1, 1]], [[0, 0]], [(0, 0)]*4)
        with self.assertRaisesRegex(RuntimeError, "no_finite_unregularized_optimum"):
            fit_modes([[separated]], ridge=0, calibrate_prior=False)
        with self.assertRaisesRegex(RuntimeError, "solver_stopped"):
            fit_modes([[separated]], max_iter=1, tol=1e-12)
        scores = np.log([[.75, .25], [.25, .75]])
        choices = [trace([[1], [1]], scores, [(g, 0)]) for g in (0, 1)]
        with self.assertRaisesRegex(RuntimeError, "prior_only.*no_finite_unregularized_root"):
            fit_modes([[choices[0]]], ridge=0)
        with self.assertRaisesRegex(RuntimeError, "prior_only.*solver_stopped"):
            fit_modes([[choices[0]]]*3 + [[choices[1]]]*5, max_iter=1)

    def test_prefix_is_exclusive_and_agrees_with_edge_probabilities(self):
        mass = [[1., .5, .2], [.4, 1., .7]]
        scores = [[1., -.8], [.1, .7]]
        a = trace(mass, scores, [(0, 0), (1, 1), (0, 1)], prior=[.2, -.2])
        b = trace(mass, scores, [(0, 0), (0, 1), (1, 1)], prior=[.2, -.2])
        eta = tensor([-.2, .2] + [0.]*8)
        da, ma, _ = prefix_moments(a, eta, IDS)
        db, mb, _ = prefix_moments(b, eta, IDS)
        torch.testing.assert_close(ma, mb)
        torch.testing.assert_close(da[:2], db[:2])
        self.assertFalse(torch.allclose(da[2], db[2]))
        ll = edge_log_probs(a, eta, IDS)
        for t in range(3):
            torch.testing.assert_close(da[t], a.prior_logits + ll[:t].sum(0))
        torch.testing.assert_close(ma.sum(1), torch.ones_like(da))

    def test_binary_exact_moment_root_and_ridge(self):
        # M rows = (.75,.25), (.25,.75); Pr(state 1)=5/8 implies q(class 1)=3/4.
        scores = np.log([[.75, .25], [.25, .75]])
        templates = [trace([[1], [1]], scores, [(g, 0)]) for g in [0, 1]]
        pool = [[templates[0]]]*3 + [[templates[1]]]*5
        nu, d = fit_prior(pool, ZERO, GLOBAL, ridge=0, tol=1e-12)
        self.assertAlmostEqual(float(nu[1] - nu[0]), math.log(3), places=10)
        self.assertAlmostEqual(d["S_B"], 4)
        nu, d = fit_prior(pool, ZERO, GLOBAL, ridge=.05, tol=1e-12)
        gamma = float(nu[1] - nu[0])
        self.assertAlmostEqual(.75 - 1 / (1 + math.exp(-gamma)) - .025 * gamma, 0.)
        self.assertGreater(gamma, 0)
        self.assertLess(gamma, math.log(3))
        _, boundary = fit_prior([[templates[0]]], ZERO, GLOBAL, ridge=0)
        self.assertEqual(boundary["status"], "no_finite_unregularized_root")
        _, capped = fit_prior(pool, ZERO, GLOBAL, max_iter=1)
        self.assertFalse(capped["converged"])

    def test_multiclass_exact_population_calibration_and_rank_deficiency(self):
        matrix = .2 + .4 * np.eye(3)
        templates = [trace([[1]] * 3, np.log(matrix), [(g, 0)]) for g in range(3)]
        # q=(.5,.3,.2); Mq=(.4,.32,.28); these 25 graph choices realize it exactly.
        pool = [[tr] for tr, count in zip(templates, [10, 8, 7]) for _ in range(count)]
        nu, diagnostics = fit_prior(pool, ZERO, GLOBAL, ridge=0, tol=1e-12)
        expected = np.log([.5, .3, .2])
        np.testing.assert_allclose(nu, expected-expected.mean(), atol=3e-8)
        self.assertEqual(diagnostics["informative_steps"], 25)
        basis = _helmert(3)
        omega, y = _moment_terms(np.repeat(matrix[None, :, :], 3, axis=0), np.arange(3), basis)
        q = np.array([.5, .3, .2])
        np.testing.assert_allclose((matrix @ q) @ y, omega[0] * (basis.T @ q), atol=1e-15)
        tr = trace([[1], [1]], [[2., 1., -.3], [-.5, .3, .7]], [(0, 0)]*2)
        nu, diag = fit_prior([[tr]], ZERO, GLOBAL)
        np.testing.assert_array_equal(nu, np.zeros(3))
        self.assertEqual(diag["status"], "no_calibration_information")
        self.assertFalse(diag["estimable"])

    def test_no_information_no_bias_and_immutable_modes(self):
        tr = trace([[1., 1.], [.2, 3.]], [[1., 1.], [-3., -3.]], [(0, 1), (1, 1)])
        tr.a.requires_grad_()
        tr.log_w.requires_grad_()
        modes, diagnostics = fit_modes([[tr]])
        self.assertEqual(set(modes), {"zero", "prior_only", "structural_only", "v2_global", "prior_calibrated"})
        for mode in ("prior_only", "prior_calibrated"):
            np.testing.assert_array_equal(modes[mode][2], [0., 0.])
            self.assertEqual(diagnostics[mode]["S_B"], 0.)
        self.assertFalse(modes["structural_only"][0].requires_grad)
        off, _ = fit_modes([[tr]], calibrate_prior=False)
        self.assertEqual(set(off), {"zero", "structural_only", "v2_global"})
        for eta, ids, nu in off.values():
            torch.testing.assert_close(edge_log_probs(tr, eta, ids), edge_log_probs(tr, eta, GLOBAL))
        torch.testing.assert_close(modes["structural_only"][0], modes["v2_global"][0])
        self.assertIsNone(tr.a.grad)
        self.assertIsNone(tr.log_w.grad)
        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
