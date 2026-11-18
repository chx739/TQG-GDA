"""Generated-only regression for frozen A actions and unchanged probabilities."""
from copy import deepcopy
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import softmax


PREFIX = 'experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.'
d = importlib.import_module(PREFIX+'decisions')
fit = importlib.import_module(PREFIX+'fit')
evaluate = importlib.import_module(PREFIX+'evaluate')
paired = importlib.import_module(PREFIX+'paired')


def fixture():
    plan = dict(decision_policy=dict(version=d.VERSION, floor=1e-8), shared_update=dict(primary_tau=.005),
                controls=dict(primary_groups=['B0', 'J', 'C05', 'C075', 'O']), real_model_seeds=[1, 2],
                phases=dict(P=dict(setting='toy', directions=[[0, 1]])),
                evaluation=dict(cohort_bootstrap_replicates=40, cohort_bootstrap_seed=13,
                    paired_interval=dict(strata=['target_domain', 'class'],
                        reuse_draws_across_methods_seeds_and_source_directions=True, conditional_on_fitted_models=True,
                        minimum_graphs_per_class_for_interval=2, percentiles=[2.5, 97.5], level=.95)))
    x = np.array([-1., 1., 2., 2.])
    h = np.array([.8, 1.2, -.3, 2.7])
    z, u = np.column_stack((np.zeros(4), x)), np.column_stack((x, np.ones(4)))
    hz, hu = np.column_stack((np.zeros(4), h)), np.column_stack((h, np.ones(4)))
    arrays, records = {}, {}
    for name in ('B0', 'J', 'C05', 'C075'):
        w = np.zeros((2, 2))
        if name == 'J':
            w[1, 1] = -.7
        arrays['w_'+name] = w
        arrays['adapt_logits_'+name] = fit.predict(name, z, u, w)
        arrays['adapt_probability_'+name] = softmax(arrays['adapt_logits_'+name], axis=1)
        records[name] = dict(valid=True, status='fixed', stop_reason='fixture', found_worst_risk=-.01)
    standard = dict(adapt_z0=z, adapt_u=u, source_u=u, mean=np.zeros(1), scale=np.ones(1))
    held = dict(holdout_z0=hz, holdout_u=hu)
    return plan, arrays, records, standard, held


class DecisionTests(unittest.TestCase):
    def test_freeze_is_A_only_and_probabilities_replay_without_change(self):
        plan, arrays, records, standard, held = fixture()
        before = {k: v.copy() for k, v in arrays.items()}
        d.freeze_decisions(plan, arrays, records)
        for key, value in before.items():
            np.testing.assert_array_equal(arrays[key], value)
        self.assertEqual(set(evaluate.methods(arrays)), evaluate.exact_methods(plan, .005))
        for alias, original in d.ALIASES.items():
            for prefix in ('w_', 'adapt_logits_', 'adapt_probability_'):
                np.testing.assert_array_equal(arrays[prefix+alias], arrays[prefix+original])
        for name in d.BALANCED:
            expected = np.maximum(arrays['adapt_probability_'+name].mean(0), 1e-8)
            expected /= expected.sum()
            np.testing.assert_array_equal(arrays['decision_prior_'+name], expected)
        self.assertFalse(np.array_equal(arrays['decision_prior_J'], arrays['decision_prior_B0_balanced']))
        replay = evaluate.replay_predictions(standard, held, arrays)
        np.testing.assert_array_equal(replay['holdout_logits_J'], replay['holdout_logits_J_argmax'])
        np.testing.assert_array_equal(replay['holdout_decision_J'], [0, 1, 0, 1])
        np.testing.assert_array_equal(replay['holdout_decision_J_argmax'], [1, 1, 0, 1])
        before_prior = arrays['decision_prior_J'].copy()
        evaluate.replay_predictions(standard, {key: value*3 for key, value in held.items()}, arrays)
        np.testing.assert_array_equal(arrays['decision_prior_J'], before_prior)
        y = np.array([0, 1, 0, 1])
        b0 = replay['holdout_logits_B0'].argmax(1)
        raw = evaluate.classification(replay['holdout_logits_J'], y, b0)
        action = evaluate.classification(replay['holdout_logits_J'], y, b0, decisions=replay['holdout_decision_J'])
        self.assertEqual(raw['sampled_nll'], action['sampled_nll'])
        self.assertEqual(action['net_corrections'], 1)
        self.assertEqual(action['balanced_accuracy']-raw['balanced_accuracy'], .25)
        # Serialization/replay uses only numerical arrays, never object pickle.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'observed.npz'
            fit.write_npz(path, **arrays)
            restored = fit.read_npz(path)
            replayed = evaluate.replay_predictions(standard, held, restored)
            for key in replay:
                np.testing.assert_array_equal(replayed[key], replay[key])

    def test_bootstrap_pairs_frozen_actions_but_uses_raw_logits_for_NLL(self):
        plan, arrays, records, standard, held = fixture()
        d.freeze_decisions(plan, arrays, records)
        arrays.update(evaluate.replay_predictions(standard, held, arrays))
        actions = d.cohort_decisions(arrays, records)
        logits = {name: arrays['holdout_logits_'+name] for name in records}
        y = np.array([0, 1, 0, 1])
        blocks = [dict(setting='toy', source=0, target=1, seed=seed,
                       ids=np.arange(4), y=y, logits=logits, decisions=actions) for seed in (1, 2)]
        # Different input order in one seed must retain matching graph actions.
        reverse = np.arange(3, -1, -1)
        blocks[1] = blocks[1] | dict(ids=np.arange(4)[reverse], y=y[reverse],
                        logits={k: v[reverse] for k, v in logits.items()},
                        decisions={k: v[reverse] for k, v in actions.items()})
        result = paired.paired_bootstrap(plan, blocks, phase='P', comparisons=[('J', 'J_argmax')])
        for row in result['interval_rows']:
            if row['metric'] == 'NLL':
                self.assertEqual((row['estimate'], row['lower'], row['upper']), (0., 0., 0.))
            elif row['metric'] in ('BA', 'Accuracy'):
                self.assertEqual(row['estimate'], 25.)
                self.assertEqual(row['complete_pairs'], 2)
        invalid = deepcopy(blocks)
        invalid[0]['decisions']['J'] = np.array([0, 1, 0, 2])
        with self.assertRaisesRegex(ValueError, 'binary decision'):
            paired.paired_bootstrap(plan, invalid, phase='P', comparisons=[('J', 'J_argmax')])
        missing = deepcopy(blocks)
        missing[0]['logits']['J'] = None
        missing[0]['decisions']['J'] = None
        result = paired.paired_bootstrap(plan, missing, phase='P', comparisons=[('J', 'J_argmax')])
        self.assertTrue(all(row['estimate'] is None for row in result['interval_rows']))

    def test_legacy_exact_path_and_tamper_rejection(self):
        plan, arrays, records, standard, held = fixture()
        old = deepcopy(arrays), deepcopy(records)
        d.freeze_decisions({}, arrays, records)
        self.assertEqual(arrays.keys(), old[0].keys())
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], old[0][key])
        self.assertEqual(records, old[1])
        self.assertEqual(d.cohort_decisions(arrays, records), {})
        self.assertEqual(d.policy_metadata({}), {})
        self.assertEqual(d.decision_contrasts({}, [('J', 'B0')]), [('J', 'B0')])
        before = evaluate.replay_predictions(standard, held, arrays)
        self.assertFalse(any('decision' in key for key in before))
        d.freeze_decisions(plan, arrays, records)
        for key, value in before.items():
            np.testing.assert_array_equal(evaluate.replay_predictions(standard, held, arrays)[key], value)
        for key in ('decision_prior_J', 'decision_prior_raw_J', 'adapt_decision_J'):
            changed = deepcopy(arrays)
            changed[key][0] += .01 if 'prior' in key else 1
            with self.assertRaises((ValueError, AssertionError)):
                evaluate.replay_predictions(standard, held, changed)
        altered = deepcopy(arrays)
        del altered['decision_rule_J']
        with self.assertRaisesRegex(ValueError, 'Missing sealed'):
            evaluate.replay_predictions(standard, held, altered)
        for policy in ({'version': d.VERSION, 'floor': .01}, {'version': 'wrong', 'floor': 1e-8}):
            with self.assertRaises(ValueError):
                d.decision_policy({'decision_policy': policy})

    def test_floor_invalidity_and_cross_stage_policy_lock(self):
        plan, arrays, records, standard, held = fixture()
        arrays['adapt_logits_J'][:] = [0., 1000.]
        arrays['adapt_probability_J'] = softmax(arrays['adapt_logits_J'], axis=1)
        records['J']['valid'] = False
        d.freeze_decisions(plan, arrays, records)
        np.testing.assert_allclose(arrays['decision_prior_J'], np.array([1e-8, 1.])/(1+1e-8), rtol=0, atol=0)
        self.assertFalse(records['J_argmax']['valid'])
        stage_s = importlib.import_module(PREFIX+'stage_s')
        stage_d = importlib.import_module(PREFIX+'stage_d')
        stage_p = importlib.import_module(PREFIX+'stage_p')
        nci1 = importlib.import_module(PREFIX+'nci1_inputs')
        worker_plan = plan | dict(phases={key: {'width_grid': [.005]} for key in ('S', 'P', 'D')})
        for phase in ('S', 'P', 'D'):
            self.assertEqual(stage_s._worker_plan(worker_plan, phase)['decision_policy'], plan['decision_policy'])
        self.assertIs(stage_d.cohort_decisions, d.cohort_decisions)
        for verifier in (stage_p.assert_density_execution, nci1.assert_density_execution):
            verifier(plan, d.policy_metadata(plan))
            with self.assertRaisesRegex(ValueError, 'decision policy'):
                verifier(plan, {})
            verifier({}, {})
        self.assertIn('decisions.py', fit.fitter_source_files(plan))


if __name__ == '__main__':
    unittest.main()
