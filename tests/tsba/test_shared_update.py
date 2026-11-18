import json
import importlib
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
    def test_round_local_powered_posterior_full_dimension_gradients_and_replay(self):
        prefix = 'experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.'
        fit = importlib.import_module(prefix+'fit')
        mixture = importlib.import_module(prefix+'mixture_explanations')
        independent_search = importlib.import_module(prefix+'separation_audit').independent_search
        rng = np.random.default_rng(1119)
        labels = np.repeat([0, 1], 32)
        raw = rng.normal(size=(64, 16))
        raw[:, 0] += .8*(2*labels-1)
        mean, scale = raw.mean(0), raw.std(0)
        source, target = (raw-mean)/scale, rng.normal(size=(19, 16))
        normal, bias = rng.normal(size=16)*.25, -.1
        v0 = np.stack((-np.r_[normal, bias]/2, np.r_[normal, bias]/2))
        cfg = SharedUpdateConfig(seed=57, inner_starts=2, inner_maxiter=100, search_rounds=2, tau=.015)
        plan = dict(separation_variant='mixture_mass_v1', density_coordinates='angular_raw_origin_v1',
                    source_mixture_options=dict(component_counts=[1], mixture_n_init=1),
                    posterior_mass_options=dict(mass_weight=1., mass_floor=1e-8))
        def family(power=None, query=target, dimension=16):
            options = plan if power is None else plan | dict(target_posterior_power=power)
            head = np.column_stack((v0[:, :dimension], v0[:, -1]))
            return fit.explanation_family(options, source[:, :dimension], labels, query[:, :dimension], cfg,
                                          head, source_mean=mean[:dimension], source_scale=scale[:dimension])
        old, one, powered = family(), family(1.), family(.5)
        theta = old._initial+np.array([.06, -.04, .03, -.02, .17])
        correction = rng.normal(size=(len(target), 2))
        for parameters in (old._initial, theta):
            for name, args in (('_observation', ()), ('_separation', (correction,))):
                a, b = getattr(old, name)(parameters, *args), getattr(one, name)(parameters, *args)
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
            raw_terms = old._target_terms(parameters)
            q = powered._target_terms(parameters)[-1]
            np.testing.assert_allclose(q, softmax(.5*(raw_terms[5]+np.log(raw_terms[3])), axis=1), atol=1e-15, rtol=0)
            self.assertGreater(np.max(np.abs(q-raw_terms[-1])), 1e-3)
            np.testing.assert_array_equal(mixture.MixtureExplanationSet._observation(powered, parameters)[0],
                                          mixture.MixtureExplanationSet._observation(old, parameters)[0])
            np.testing.assert_array_equal(mixture.MixtureExplanationSet._observation(powered, parameters)[1],
                                          mixture.MixtureExplanationSet._observation(old, parameters)[1])
            np.testing.assert_array_equal(powered.marginal_cdf(parameters, powered.target),
                                          old.marginal_cdf(parameters, old.target))
            np.testing.assert_allclose(powered._candidate(parameters).statistic,
                                       q.T@powered.features/len(target), atol=1e-15, rtol=0)
            functions = [powered._observation, lambda p: powered._mass_penalty(p)[:2],
                         lambda p: powered._separation(p, correction),
                         lambda p: mixture.MixtureExplanationSet._observation(powered, p)]
            for function in functions:
                _, gradient = function(parameters)
                numerical = [(function(parameters+1e-6*axis)[0]-function(parameters-1e-6*axis)[0])/2e-6
                             for axis in np.eye(len(parameters))]
                np.testing.assert_allclose(gradient, numerical, atol=5e-7, rtol=2e-5)
        np.testing.assert_array_equal(powered.features, old.features)
        np.testing.assert_array_equal(powered.mass_reference_probabilities, old.mass_reference_probabilities)
        np.testing.assert_array_equal(powered.mass_reference_probabilities[1], expit(target@normal+bias).mean())
        # Entire fitting/separation/master path, without changing the raw head.
        z = powered.features@v0.T
        frozen_z = z.copy()
        w, result = solve_shared_update(z, powered.features, powered, cfg)
        self.assertEqual(result['status'], 'completed', result)
        np.testing.assert_array_equal(z, frozen_z)
        self.assertEqual(powered.threshold, powered.reference_loss+cfg.tau)
        for candidate in result['candidates']:
            q = powered._target_terms(candidate['parameters'])[-1]
            direct_risk = np.mean(np.sum(q*(logsumexp(z+powered.features@w.T, axis=1)[:, None]
                                      -logsumexp(z, axis=1)[:, None]-powered.features@w.T), axis=1))
            np.testing.assert_allclose(relative_risks(z, powered.features, w, [candidate['statistic']])[0],
                                       direct_risk, atol=2e-14, rtol=0)
        candidates, audit = independent_search(powered, correction, seed=7058, starts=1, maxiter=80,
                                               known=[powered.fit()])
        self.assertEqual(audit['fresh_optimizer_launches'], 1)
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertLessEqual(candidate.observation_loss, powered.threshold+cfg.feasibility_tol)
            np.testing.assert_allclose(candidate.statistic,
                powered._target_terms(candidate.parameters)[-1].T@powered.features/len(target), atol=1e-15, rtol=0)
        held = rng.normal(size=(11, 16))
        replay = family(.5, held)
        together = family(.5, np.vstack((target, held)))
        np.testing.assert_array_equal(replay.projection_basis, powered.projection_basis)
        np.testing.assert_allclose(replay._target_terms(theta)[-1], together._target_terms(theta)[-1][len(target):],
                                   atol=1e-15, rtol=0)
        for a, b in zip(replay.source_templates, powered.source_templates):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        report = powered.fit_report['target_posterior_power']
        self.assertEqual(report['effective'], .5)
        self.assertTrue(report['raw_marginal_gradient_uses_unpowered_posterior'])
        json.dumps(powered.fit_report, allow_nan=False)
        # The analytical S gate keeps precisely the old 2D numerical branch.
        small_old, small = family(dimension=2), family(.5, dimension=2)
        self.assertEqual(small.target_posterior_power, 1.)
        a, b = small_old.fit(), small.fit()
        np.testing.assert_array_equal(a.parameters, b.parameters)
        np.testing.assert_array_equal(a.statistic, b.statistic)
        self.assertEqual(a.observation_loss, b.observation_loss)
        np.testing.assert_array_equal(small._separation(theta, correction)[1], small_old._separation(theta, correction)[1])
        for invalid in (0., -1., 1.01, True, float('nan')):
            with self.assertRaises(ValueError):
                family(invalid)
        with self.assertRaisesRegex(ValueError, 'only for angular'):
            fit.explanation_family(plan | dict(density_coordinates='source_standardized_v1', target_posterior_power=.5),
                                   source, labels, target, cfg, v0)

    def test_round_local_source_refit_schedules_once_and_reuses_pilot_checkpoints(self):
        module = importlib.import_module(
            'experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.source_refit_stage')
        plan = dict(source_refit=dict(pooling='mean', recipe='r5_frozen_role_source_refit_v1'))
        tasks = [('edges', 0, 1, 9), ('edges', 0, 3, 9), ('edges', 1, 3, 9)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prior = {('edges', 0, 9): root / 'P' / 'source-refit'}
            with patch.object(module, 'ProcessPoolExecutor') as pool:
                worker = pool.return_value.__enter__.return_value
                worker.map.side_effect = lambda fn, keys, *_: iter(key[0] for key in keys)
                roots = module.prepare(plan, tasks, root, 2, root, reuse=prior)
                self.assertEqual(worker.map.call_args.args[1], [('edges', 1, 9)])
            self.assertEqual(roots, prior | {('edges', 1, 9): root / 'source-refit'})
            with self.assertRaisesRegex(ValueError, 'unregistered checkpoint'):
                module.prepare(plan, tasks, root, 2, root, reuse={('edges', 3, 9): root / 'bad'})
            self.assertEqual(module.prepare({}, tasks, root, 2, root), {})

    def test_round_local_angular_density_preserves_raw_moments_and_mass(self):
        module = importlib.import_module(
            'experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.angular_explanations')
        result = module.check_angular_explanations()
        self.assertTrue(result['passed'])
        self.assertTrue(result['exact_2D_bypass'])
        self.assertTrue(result['original_B0_mass_and_full_moments'])
        self.assertLess(result['maximum_gradient_error'], 5e-7)

    def test_round_local_angular_plan_and_pins_cross_stage_boundaries(self):
        prefix = 'experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.'
        fit = importlib.import_module(prefix+'fit')
        stage_s = importlib.import_module(prefix+'stage_s')
        stage_p = importlib.import_module(prefix+'stage_p')
        stage_c = importlib.import_module(prefix+'stage_c')
        nci1_inputs = importlib.import_module(prefix+'nci1_inputs')
        from experiments.common.io import digest

        plan = dict(separation_variant='mixture_mass_v1', density_coordinates='angular_raw_origin_v1', target_posterior_power=.5,
                    shared_update={}, controls={}, source_mixture_options={}, posterior_mass_options={},
                    real_model_seeds=[0], phases=dict(S=dict(width_grid=[.005]),
                    P=dict(width_grid=[.005], independent_audit={}), C={}, D=dict(width_grid=[.005])))
        for phase in ('S', 'P', 'D'):
            view = stage_s._worker_plan(plan, phase)
            self.assertEqual(view['density_coordinates'], plan['density_coordinates'])
            self.assertEqual(view['separation_variant'], plan['separation_variant'])
            self.assertEqual(view['target_posterior_power'], .5)
        view = stage_c._fit_view(plan, .005)
        self.assertEqual(view['density_coordinates'], plan['density_coordinates'])
        self.assertEqual(stage_s._worker_plan(view, 'P')['density_coordinates'], plan['density_coordinates'])
        self.assertEqual(view['target_posterior_power'], .5)
        self.assertEqual(stage_s._worker_plan(view, 'P')['target_posterior_power'], .5)
        for stage in (stage_p, stage_c, nci1_inputs):
            self.assertIs(stage.assert_fitter_sources, fit.assert_fitter_sources)
            self.assertIs(stage.assert_density_execution, fit.assert_density_execution)
            stage.assert_density_execution(plan, dict(density_coordinates=plan['density_coordinates'], target_posterior_power=.5))
            with self.assertRaisesRegex(ValueError, 'posterior power'):
                stage.assert_density_execution(plan, dict(density_coordinates=plan['density_coordinates']))
            with self.assertRaisesRegex(ValueError, 'density coordinates'):
                stage.assert_density_execution(plan, {})
            with self.assertRaisesRegex(ValueError, 'density coordinates'):
                stage.assert_density_execution(plan, dict(density_coordinates='source_standardized_v1'))
        fit.assert_density_execution({}, {})  # Original archives retain their default.
        files = fit.fitter_source_files(plan)
        self.assertTrue({'angular_explanations.py', 'projected_explanations.py', 'coverage.py',
                         'stage_s.py', 'stage_p.py', 'stage_c.py'} <= set(files))
        repo = Path(__file__).resolve().parents[2]
        paths = [Path(fit.__file__).with_name(name) for name in files]
        frozen = {str(path.relative_to(repo)): digest(path) for path in paths}
        for stage in (stage_p, stage_c):
            stage.assert_fitter_sources(plan, repo, frozen)
            bad = dict(frozen)
            angular_key = str(Path(fit.__file__).with_name('angular_explanations.py').relative_to(repo))
            bad[angular_key] = '0'*64
            with self.assertRaisesRegex(ValueError, 'angular_explanations.py'):
                stage.assert_fitter_sources(plan, repo, bad)

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
