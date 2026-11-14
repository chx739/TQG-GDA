"""Static experiment controls must be testable independently of research outcomes."""
from dataclasses import replace
import unittest
import torch
from tsba.config import Config
from tsba.classifier import Coordinates
from tsba.score import train_score
from experiments.common.synthetic.distributions import GaussianMixtureScore, sample_gmm
from experiments.common.synthetic.static_score import enlarged_pool, schedules, fit, evaluate, audit_scores, sign_metrics


class StaticScoreTests(unittest.TestCase):
    def test_paired_fit_exactly_reproduces_production_baseline(self):
        cfg = Config(seed=2, score_epochs=3, score_hidden=8)
        x, _ = sample_gmm(512, 2, 2002)
        original, meta = train_score(x, .3, cfg)
        schedule = schedules(2, cfg.seed, updates=12)
        saved = []
        _, new, best = fit(x, .3, cfg, schedule, checkpoint=saved.append)
        self.assertEqual(new['updates'], 12)
        self.assertEqual(new['selected_update'], 4*(meta['selected_epoch']+1))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, best[key], atol=0, rtol=0)
        for old, new_value in zip(meta['history'], new['validation_history']):
            self.assertEqual(old['noise_validation_loss'], new_value['noise_validation_loss'])
        self.assertEqual(int(next(iter(saved[0]['optimizer']['state'].values()))['step']), 12)

    def test_nested_pool_and_fixed_noise_update_budget(self):
        x, _ = sample_gmm(512, 2, 123)
        c = Coordinates.fit(x, 1e-8)
        target = c(x)
        pool = enlarged_pool(target, c, 3)
        torch.testing.assert_close(pool[:512], target, atol=0, rtol=0)
        self.assertEqual(tuple(pool.shape), (8192, 2))
        schedule = schedules(2, 3)
        for n, ids in schedule['ids'].items():
            self.assertEqual(tuple(ids.shape), (400, 128))
            self.assertLess(int(ids.max()), n)
            self.assertEqual(len(torch.unique(ids[:n//128])), n)
        self.assertEqual(tuple(schedule['noise'].shape), (400, 128, 2))

    def test_kde_query_chunk_does_not_change_score(self):
        pool, _ = sample_gmm(1024, 16, 10)
        queries, _ = sample_gmm(71, 16, 11)
        score = GaussianMixtureScore(pool, torch.zeros_like(pool), .7)
        torch.testing.assert_close(evaluate(score, queries, 7), score(queries), atol=0, rtol=0)

    def test_signed_request_error_exposes_opposite_equal_magnitudes(self):
        points = torch.tensor([[-.5], [.5]], dtype=torch.float64)
        normals = -torch.ones_like(points)
        population = points.sign()  # Density increases away from the valley.
        values = dict(population=population, empirical=population, learned=-population,
                      selected=-population, zero=torch.zeros_like(points))
        result, arrays = audit_scores(points, normals, values, .5, 2., Config(),
                                     torch.ones(2, dtype=torch.bool), points[:, 0])
        m = result['comparisons']['learned_vs_population']
        self.assertEqual(m['sign']['balanced'], 0.)
        self.assertEqual(m['request_magnitude_rmse_raw'], 0.)
        self.assertGreater(m['request_rmse_raw'], 0.)
        self.assertEqual(result['signals']['population']['toward_target_boundary'], 1.)
        self.assertEqual(result['signals']['learned']['toward_target_boundary'], 0.)
        one_sided = sign_metrics(torch.ones(3), torch.ones(3), torch.ones(3, dtype=torch.bool))
        self.assertIsNone(one_sided['balanced'])
        self.assertEqual(one_sided['negative_count'], 0)

    def test_broad_smoothing_reverses_population_direction(self):
        means = torch.tensor([[-1.], [1.]], dtype=torch.float64)
        query = torch.tensor([[.5]], dtype=torch.float64)
        small = GaussianMixtureScore(means, torch.full_like(means, .25), .4)
        large = GaussianMixtureScore(means, torch.full_like(means, .25), 1.5)
        self.assertGreater(float(small(query)), 0)
        self.assertLess(float(large(query)), 0)
