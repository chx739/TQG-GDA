"""Generated probability tables, source-only receipts, and graph-readout regressions."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import (
    identification as ident, readouts, screen, source_inputs, worlds,
)

OPTIONS = dict(covariance_tolerance=1e-12, feasibility_tolerance=1e-10)
PLAN = dict(estimator=OPTIONS, settings=['generated'], domains=[0, 1, 2, 3], minimum_correct_domains=3,
            worlds=[dict(name='j1', positive_percent=[2, 63, 65, 70], optimal_j=1),
                    dict(name='j3', positive_percent=[30, 35, 37, 98], optimal_j=3)])


def independent(p, u0, u1, l=(.18, .78), t=(.8, .2)):
    bins = np.array([u0, u1])
    l, t = (np.column_stack((1. - np.asarray(c), c)) for c in (l, t))
    return ident.product_joint([1. - p, p], bins, l, t)


class R12Tests(unittest.TestCase):
    def test_quantiles_are_linear_and_ties_go_left(self):
        cuts, bins = readouts.quantile_bins(np.arange(5, dtype=np.float64))
        np.testing.assert_array_equal(cuts, [1, 2, 3])
        np.testing.assert_array_equal(bins, [1, 1, 2, 3, 4])
        _, same = readouts.quantile_bins(np.ones(9))
        np.testing.assert_array_equal(same, np.ones(9))

    def test_signed_independent_recovery_in_two_registered_worlds(self):
        for row, accuracy, optimal in screen.specifications(PLAN):
            q = np.asarray(row['positive_percent']) / 100.
            joint = independent(.5, (1. - q)/2, q/2)
            fit = ident.recover(joint.sum(axis=0), **OPTIONS)
            self.assertEqual(fit['status'], 'valid')
            self.assertEqual(fit['j'], optimal)
            self.assertLess(fit['covariance']['D_T'], 0.)
            np.testing.assert_allclose(fit['estimated_accuracy'], np.asarray(accuracy, float), atol=1e-13)
            self.assertAlmostEqual(float(accuracy[optimal]), .74)
            self.assertAlmostEqual(float(accuracy[2]), .675)
            self.assertAlmostEqual(float(accuracy[4-optimal]), .60)

    def test_nonhalf_prior_and_explicit_label_switch(self):
        p = .28
        u0, u1 = [.55,.25,.13,.07], [.1,.1,.35,.45]
        joint = independent(p, u0, u1)
        observed = joint.sum(axis=0)
        fit = ident.recover(observed, **OPTIONS)
        self.assertAlmostEqual(fit['prior'], p, places=13)
        np.testing.assert_allclose(fit['positive_mass'], p*np.asarray(u1))
        swapped = ident.recover(observed[:, ::-1, :], **OPTIONS)
        self.assertAlmostEqual(swapped['prior'], 1-p, places=13)
        np.testing.assert_allclose(swapped['positive_mass'], (1-p)*np.asarray(u0))

    def test_flat_view_falls_back_and_records_rank_and_moments(self):
        joint = independent(.5, [.2,.3,.3,.2], [.1,.1,.35,.45], l=(.5,.5))
        fit = ident.recover(joint.sum(axis=0), **OPTIONS)
        self.assertEqual(fit['j'], 2)
        self.assertEqual(fit['status'], 'fallback')
        self.assertIn('small_covariance_D_L', fit['reasons'])
        self.assertIn('small_covariance_L_T', fit['reasons'])
        self.assertEqual(fit['unfolding_rank'][1], 1)

    def test_infeasible_positive_cell_falls_back_before_truncation(self):
        # D is a noisy observation of the latent class. Redistribute the
        # first two U cells with a stronger L association at fixed (D,L,T).
        p = .5
        joint = independent(p, [.4,.4,.1,.1], [.1,.1,.4,.4], l=(.25,.75), t=(.3,.7))
        x = joint.sum(axis=0)
        pair = x[0] + x[1]
        x[0, 0] = pair[0]
        x[0, 1] = 0.
        x[1, 0] = 0.
        x[1, 1] = pair[1]
        fit = ident.recover(x, **OPTIONS)
        self.assertEqual(fit['status'], 'fallback')
        self.assertEqual(fit['j'], 2)
        self.assertTrue(any(reason.startswith('infeasible_positive_mass_U') for reason in fit['reasons']))

    def test_identical_tables_can_encode_different_label_semantics(self):
        observed = np.zeros((4,2,2))
        observed[:2,0,0] = .25
        observed[2:,1,1] = .25
        # Each view precisely measures a skeleton; even a positive true L
        # association is insufficient if both skeletons mostly contain Y=1.
        y_skeleton = np.stack((observed.copy(), np.zeros_like(observed)))
        y_skeleton[1,2:] = y_skeleton[0,2:]
        y_skeleton[0,2:] = 0.
        y_mixed = observed[None] * np.array([[.4,.4,.15,.15],[.6,.6,.85,.85]])[:, :, None, None]
        np.testing.assert_allclose(y_skeleton.sum(axis=0), y_mixed.sum(axis=0))
        a = ident.recover(y_skeleton.sum(axis=0), **OPTIONS)
        b = ident.recover(y_mixed.sum(axis=0), **OPTIONS)
        self.assertEqual(a['j'], b['j'])
        self.assertEqual(a['j'], 2)
        self.assertEqual(b['reconstructed_table_max_error'], 0.)
        fit = ident.choose(ident.accuracies(observed.sum(axis=(1,2)), y_mixed[1].sum(axis=(1,2))))
        self.assertEqual(fit, 0)
        self.assertEqual(worlds.conditional_stats(y_mixed)['L_order'], 'positive')

    def test_product_preserves_single_view_conditionals_and_u_threshold_risk(self):
        counts = np.arange(1, 33).reshape(2,4,2,2)
        spec = PLAN['worlds'][0]
        pair, product = worlds.tables(counts, spec['positive_percent'])
        self.assertFalse(np.allclose(pair, product))
        np.testing.assert_allclose(pair.sum(axis=(2,3)), product.sum(axis=(2,3)))
        for x, y in zip(worlds.class_marginals(pair), worlds.class_marginals(product)):
            np.testing.assert_allclose(x, y, atol=1e-14, rtol=0)
        result, j = worlds.stratum_oracle(counts, pair.sum(axis=0), spec['positive_percent'], 1e-10)
        self.assertTrue(all(row['identifiable'] for row in result))
        self.assertEqual(j, 1)
        counts[:, 0] = [[3,1],[1,5]]
        pair, _ = worlds.tables(counts, spec['positive_percent'])
        result, j = worlds.stratum_oracle(counts, pair.sum(axis=0), spec['positive_percent'], 1e-10)
        self.assertFalse(result[0]['identifiable'])
        self.assertIsNone(j)

    def test_empty_stratum_retained_in_domain_denominator(self):
        counts = np.ones((2,4,2,2), int)
        counts[1,3] = 0
        state, outcomes, fits, diagnostics, oracle = screen.analyze(counts, PLAN)
        self.assertEqual(state, 'insufficient_support')
        self.assertEqual(counts[1,3].sum(), 0)
        self.assertEqual(len(outcomes), 4)
        self.assertEqual(len(oracle), 8)
        self.assertFalse(fits or diagnostics)
        rows = [dict(row, setting='generated', source=src) for src in PLAN['domains'] for row in outcomes]
        result = screen.verdict(PLAN, rows)
        self.assertFalse(result['passed'])
        self.assertEqual(result['settings']['generated']['denominator'], 4)

    def test_threshold_tie_break_prefers_middle_then_smaller(self):
        self.assertEqual(ident.choose(np.full(5, .5)), 2)
        self.assertEqual(ident.choose(np.array([.2,.7,.1,.7,.1])), 1)

    def test_wl_removes_attributes_and_excludes_v_only_vocabulary(self):
        edge = torch.tensor([[0,1,1,2],[1,0,2,1]])
        graph = dict(x=torch.tensor([[1.,0.],[0.,1.],[1.,0.]]), edge_index=edge, edge_attr=None)
        type_swap = graph | dict(x=graph['x'].flip(1), edge_attr=torch.zeros((4, 2)))
        self.assertEqual(readouts.topology_counts(graph), readouts.topology_counts(type_swap))
        np.testing.assert_allclose(readouts.node_frequency(graph,2), [2/3,1/3])
        v = graph | dict(edge_index=torch.tensor([[0,1],[1,0]]))
        captured = []
        def mock_fit(x_s, x_v, y_s, options):
            captured.append(x_s)
            return {}, np.zeros(x_v.shape[0]), np.zeros(x_v.shape[0],dtype=np.int8)
        with patch.object(readouts, 'fit_linear', side_effect=mock_fit):
            models, _ = readouts.fit({5:graph, 6:type_swap, 7:v}, [5,6], [7], [0,1], 2, {})
        normalized = readouts.wl.normalize(readouts.topology_counts(graph))
        self.assertEqual(captured[1].shape[1], len(normalized))
        self.assertEqual(models['T']['vocabulary'], [[depth,color.hex()] for depth,color in sorted(normalized)])

    def test_source_label_reader_rejects_wrong_scope_before_opening(self):
        ar = object.__new__(source_inputs.SourceArchive)
        ar.role = Mock(return_value={'fit':[1,2], 'validation':[3,4]})
        ar.label_reads = {}
        ar._load = Mock()
        for role, ids in [('holdout',[3,4]),('adapt',[1,2]),('fit',[2,1]),('validation',[1,2])]:
            with self.assertRaises(PermissionError):
                ar.labels(0, role, ids)
        ar._load.assert_not_called()

    def test_historical_binding_never_requires_unopened_target_members(self):
        from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import seals
        from hashlib import sha256
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'seals').mkdir()
            plan = dict(phases={'C':{'settings':['generated'], 'domains':[0,1]}}, real_model_seeds=[1])
            payload = json.dumps(plan).encode()
            (root/'executed-plan.json').write_bytes(payload)
            parent = None
            for stage in seals.STAGES:
                record = dict(stage=stage,parent=parent,files={'some/target/member.npz':'must-not-open'})
                if stage == 'R7_INPUT':
                    record['files']['executed-plan.json'] = sha256(payload).hexdigest()
                parent = record['digest'] = seals._signature(record)
                (root/'seals'/f'{stage}.json').write_text(json.dumps(record))
            ar = source_inputs.ValidationArchive(dict(r7_archive=tmp, r7_final_digest=parent,
                                                       settings=['generated'], seed=1, domains=[0,1]))
            self.assertEqual(ar.plan, plan)
            self.assertEqual(list(ar.source_reads), ['executed-plan.json'])
            with self.assertRaises(PermissionError):
                ar.validation('NCI1', 0, 1)
