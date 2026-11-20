from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import numpy as np
import torch

from experiments.common.io import write_json
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import crossover, training
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run


PLAN = dict(setting='mutagenicity_edge', source=0, targets=[1, 3], seeds=[3405, 3406])


def fixture():
    pools = {'S': [0, 1], 'A-M1': [2], 'A-M3': [3], 'H-M1': [10, 11], 'H-M3': [21, 22, 23, 24]}
    roles = dict(pools=pools, source_fit={'3405': [0], '3406': [1]}, tasks={})
    for (seed, h), key in crossover.tasks(PLAN).items():
        roles['tasks'][key] = dict(setting=PLAN['setting'], source=0, target=h, seed=seed,
                                   source_ids=roles['source_fit'][str(seed)], a_ids=pools[f'A-M{h}'],
                                   h_ids=pools[f'H-M{h}'])
    return roles


def prediction(ids, **actions):
    row = {'ids': np.asarray(ids, dtype=np.int64)}
    for method, action in actions.items():
        a = np.asarray(action, dtype=np.int64)
        row[f'actions_{method}'] = a
        row[f'logits_{method}'] = np.eye(2, dtype=np.float32)[a]
    return row


class TestCrossover(TestCase):
    def test_pool_overlap_and_cross_seed_order(self):
        roles = fixture()
        crossover.verify_roles(PLAN, roles)
        other = next(iter(roles['tasks']))
        roles['tasks'][other]['h_ids'] = [11, 10]
        with self.assertRaises(PermissionError):
            crossover.verify_roles(PLAN, roles)
        roles = fixture()
        roles['pools']['S'].append(10)
        with self.assertRaises(PermissionError):
            crossover.verify_roles(PLAN, roles)

    def test_source_order_counts_at_same_updates(self):
        a, b, c = [char * 64 for char in 'abc']
        orders = {'A1': [a, b, c], 'A3': [a, c, b], 'SourceOnly': [a, b, b]}
        row = crossover.sequence_row(3405, orders, 777, 747)
        self.assertEqual(row['A1_A3_same_steps'], 1)
        self.assertEqual(row['A1_A3_same_step_numbers'], '1')
        self.assertEqual(row['A3_SourceOnly_same_step_numbers'], '1 3')
        self.assertEqual(crossover.source_history([dict(event='fit_started'),
                          dict(event='epoch', source_order_sha256=a)], 1), [a])
        with self.assertRaises(ValueError):
            crossover.source_history([dict(event='epoch', source_order_sha256='wrong')], 1)

    def test_paired_interaction_with_unequal_domain_sizes(self):
        roles = fixture()
        labels = {1: np.asarray([0, 1]), 3: np.asarray([0, 1, 1, 0])}
        # A1 improves by 50 pp on both domains. This common gain cancels.
        predictions = {(seed, h): prediction(roles['tasks'][key]['h_ids'],
                         A1=labels[h], A3=[0, 0] if h == 1 else [0, 0, 0, 0],
                         SourceOnly=np.zeros(len(labels[h])))
                       for (seed, h), key in crossover.tasks(PLAN).items()}
        opts = dict(strata=['target_domain', 'class'], conditional_on_fitted_models=True,
                    reuse_draws_across_methods_seeds_and_source_directions=True,
                    minimum_graphs_per_class_for_interval=1, level=.95, percentiles=[2.5, 97.5])
        native = dict(evaluation=dict(paired_interval=opts, cohort_bootstrap_replicates=40,
                                      cohort_bootstrap_seed=1))
        endpoints, matrix, pairs, summary, audit = crossover.estimate(PLAN, native, predictions, labels)
        by_h = {(r['holdout'], r['left'], r['right']): r for r in summary}
        self.assertEqual(by_h[1, 'A1', 'A3']['accuracy_pp'], 50.)
        self.assertEqual(by_h[3, 'A3', 'A1']['accuracy_pp'], -50.)
        interaction = by_h['equal_mean', 'Own', 'Other']
        self.assertEqual([interaction[k] for k in ('accuracy_pp', 'ci95_low_pp', 'ci95_high_pp')], [0., 0., 0.])
        self.assertEqual(len(endpoints), 12)
        self.assertEqual(len(matrix), 6)
        self.assertEqual(len(pairs), 14)
        self.assertEqual(audit['replicates'], 40)
        point, _, counts = crossover.compare(np.array([0, 0, 0]), np.array([1, 1, 0]), np.array([0, 1, 0]))
        self.assertEqual((point, counts['corrected'], counts['harmed'], counts['disagreements']),
                         (Fraction(0), 1, 1, 2))

    def test_complete_reloaded_predictions_required_before_h_labels(self):
        roles = fixture()
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / 'predictions').mkdir()
            historic = {}
            for (_, h), key in crossover.tasks(PLAN).items():
                ids = roles['tasks'][key]['h_ids']
                a = np.zeros(len(ids), dtype=np.int64)
                row = prediction(ids, A1=a, A3=a, SourceOnly=1-a)
                historic[key] = {f'{field}_{old}': row[f'{field}_{new}']
                                 for field in ('logits', 'actions')
                                 for old, new in (('GRADE', f'A{h}'), ('SourceOnly', 'SourceOnly'))}
                prior_run.save_predictions(output, crossover.prediction_file(key), row)
            write_json(output / 'roles.json', roles)
            for name in ('executed-config.json', 'lineage.json', 'source-sequences.csv'):
                (output / name).write_text('fixture')
            parents = dict(FINAL='p')
            authenticated = (None, roles, None, historic, None, parents)
            paths = [crossover.prediction_file(key) for key in historic]
            checksum = prior_run.seal(output, 'OBS', [*crossover.METADATA, *paths], parents)
            frozen, replay = crossover.verify_observation(output, PLAN, authenticated, checksum)
            self.assertEqual((len(frozen), len(replay)), (4, 8))
            (output / paths[-1]).write_bytes(b'changed')
            with mock.patch.object(crossover, 'holdout_labels') as labels, self.assertRaises(PermissionError):
                crossover.score(output, None, PLAN, authenticated, checksum)
            labels.assert_not_called()

    def test_native_inference_is_eval_in_order_and_preserves_state(self):
        model = training.new_model(dict(hidden=4, layers=2, dropout=.5, disc='JS', mode='graph'), 2, 2, 'cpu')
        plan = dict(execution=dict(device='cpu', inference_batch_size=1))
        graphs = {i: dict(x=torch.full((2, 2), float(i)), edge_index=torch.tensor([[0, 1], [1, 0]]))
                  for i in [4, 1]}
        before = training.state_hash(model)
        first = training.infer(model, graphs, [4, 1], plan)
        second = training.infer(model, graphs, [1, 4], plan)
        self.assertEqual(first['ids'].tolist(), [4, 1])
        self.assertEqual(before, training.state_hash(model))
        self.assertFalse(model.training)
        torch.testing.assert_close(first['logits'], second['logits'].flip(0), rtol=0., atol=0.)
