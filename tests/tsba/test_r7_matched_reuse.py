"""Generated matched donors exercise seal, identity, replay and worker routing."""
from contextlib import contextmanager
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.io import digest, save_tensor, write_json
from tsba.classifier import Coordinates, ResidualClassifier
from tsba.config import Config


_PACKAGE = 'experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src.'
c = import_module(_PACKAGE + 'stage_c')
m = import_module(_PACKAGE + 'matched_reuse')
p = import_module(_PACKAGE + 'stage_p')
d = import_module(_PACKAGE + 'decisions')
_REPO = Path(__file__).resolve().parents[2]


class TestMatchedReuse(TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.root = Path(temp.name)
        self.donor_root = self.root / 'donor'
        self.new_root = self.root / 'candidate'
        self.task = ('toy', 0, 1, 17)
        self.relative = Path('blocks') / c._block_name(self.task)
        self.tasks = [(str(self.relative), self.task)]
        self.old = self.donor_root / self.relative
        self.new = self.new_root / self.relative
        self.old.mkdir(parents=True)
        self.new.mkdir(parents=True)
        self.names = c._matched_files(30)
        self.budget = dict(compared_groups=['B1', 'A0'], use_R5_M00_implementation=True,
                           seed_offsets=m.matched_tsba._OFFSETS.copy(), score_exact_updates=7200,
                           attempts_per_group=30, source_and_target_batch=128, score_batch=128,
                           score_lr=.001, lambda_boundary=1, boundary_tau=.1, damping=.01, kappa_sigma=1)
        self.plan = dict(execution={'threads_per_worker': 1}, real_settings=[{'id': 'toy'}], features={},
                         target_split={}, real_model_seeds=[17], phases={'C': {'settings': ['toy'],
                         'domains': [0, 1], 'matched_tsba': self.budget}})
        setting, source, target, seed = self.task
        self.identity = dict(setting=setting, source=source, target=target, seed=seed)
        self._inputs()
        self._donor()

    def _inputs(self):
        cfg = Config(seed=17, emb_dim=2, score_hidden=4, adapter_width=2, source_adapt_batch=128,
                     target_batch=128, score_batch=128, score_lr=.001, adapt_rounds=30,
                     lambda_boundary=1, tau=.1, damping=.01, kappa_sigma=1)
        source = torch.tensor([[-2., 0.], [-1., 1.], [1., -1.], [2., 0.]], dtype=torch.float64)
        adapt = torch.tensor([[-1., -1.], [0., 0.], [1., 1.]], dtype=torch.float64)
        y = torch.tensor([0, 0, 1, 1])
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(cfg.seed)
            head = torch.nn.Linear(2, 2, dtype=torch.float64)
            coordinates = Coordinates.fit(source, cfg.coordinate_eps)
            initial = ResidualClassifier.from_source(head, coordinates, cfg.adapter_width, cfg.seed)
            source_logits, adapt_logits = head(source), head(adapt)
        provenance = {'cache_sha256': 'fixed', 'target_labels_loaded': False}
        fit = dict(source_h=source, source_y=y, source_ids=[0, 1, 2, 3], source_z0=source_logits,
                   adapt_h=adapt, adapt_z0=adapt_logits, adapt_ids=[100, 103, 104],
                   source_head_weight=head.weight.detach(), source_head_bias=head.bias.detach())
        held = dict(holdout_h=adapt[:2], holdout_z0=adapt_logits[:2], holdout_ids=[101, 102])
        valid = dict(validation_h=source[:2], validation_z0=source_logits[:2], validation_ids=[4, 5])
        model_config = asdict(cfg)
        adam = {'betas': [.9, .999], 'eps': 1e-8}
        old_plan = dict(model=model_config, adam=adam, setting={'id': 'toy'}, seeds=[17], solver={}, weight=1,
                        methods={'B1': {'group': 'M00', 'density': 'zero'},
                                 'A0': {'group': 'M00', 'density': 'learned'}})
        self.blind = {key: value for key, value in fit.items() if key != 'source_z0'}
        self.blind.update(task=self.identity, source_x=coordinates(source), adapt_x=coordinates(adapt),
                          tsba_mean=coordinates.mean, tsba_scale=coordinates.scale, tsba_initial_head=initial.state_dict(),
                          r5_plan=old_plan, r5_model_config=model_config, r5_adam_config=adam, provenance=provenance)
        manifest = self.identity | dict(provenance=provenance, ids={'source': fit['source_ids'], 'adapt': fit['adapt_ids'],
                        'holdout': held['holdout_ids'], 'validation': valid['validation_ids']})
        for block, version in ((self.old, 'donor'), (self.new, 'candidate')):
            write_json(block / 'manifest.json', manifest | {'r7_plan': {'candidate': version}})
            # Distinct dictionary iteration order proves byte equality is unnecessary.
            save_tensor(block / 'matched_input.pt', self.blind if version == 'donor'
                        else dict(reversed(list(self.blind.items()))))
            for name, values in [('fit_input.npz', fit), ('holdout_input.npz', held), ('validation_input.npz', valid)]:
                np.savez_compressed(block / name, **values)

    def _donor(self):
        deps = {name: digest(_REPO / name) for name in m.matched_tsba._DEPENDENCIES}
        code = deps | {name: digest(_REPO / name) for name in m._EXTRA_CODE}
        module = Path(m.matched_tsba.__file__).resolve()
        implementation = digest(module)
        code[str(module.relative_to(_REPO))] = implementation
        cfg, _, head, coordinates, _, _, adapt_x, ids = m.matched_tsba._prepare(self.plan, self.blind, 17)
        meta = dict(task=self.identity, method='R7_matched_R5_M00', seed=17, r5_model_config=self.blind['r5_model_config'],
                    effective_adaptation_config=asdict(cfg), adam=self.blind['r5_adam_config'], score_exact_updates=7200,
                    attempts_per_group=30, seed_offsets=self.budget['seed_offsets'], source_count=len(ids[0]),
                    adaptation_count=len(adapt_x), target_labels_used=False, holdout_inputs_received=False,
                    score_uses_adaptation_pool_only=True, original_global_RMS_coordinates=True, torch_threads=1,
                    score_initialization_dtype='float32_then_float64', initial_head_sha256=m.matched_tsba._state_hash(head.state_dict()),
                    r5_provenance=self.blind['provenance'], implementation_sha256=implementation,
                    implementation_dependencies_sha256=deps)
        state = head.state_dict()
        arms = {name: dict(valid=True, attempts=30, status='completed', stop_reason='attempt_budget_completed',
                          state=state, adapt_logits=head(adapt_x).detach()) for name in ('B1', 'A0')}
        self.result = dict(metadata=meta, coordinates=dict(mean=coordinates.mean, scale=coordinates.scale), arms=arms)
        write_json(self.donor_root / 'executed-plan.json', self.plan | {'new_candidate': False})
        write_json(self.donor_root / 'executed-configs.json', dict(phase='C', threads_per_worker=1, matched_tsba=self.budget))
        write_json(self.donor_root / 'provenance.json', dict(file_hashes=code, torch=torch.__version__, git_commit='toy'))
        write_json(self.donor_root / 'blocks.json', [{'path': str(self.relative)} | self.identity])
        inputs = [str(self.relative / name) for name in (*c._INPUT_FILES, 'matched_input.pt')]
        inputs.extend(['executed-plan.json', 'executed-configs.json', 'provenance.json', 'blocks.json'])
        input_seal = c.seal(self.donor_root, 'R7_INPUT', inputs, expected=inputs)
        for name in self.names:
            path = self.old / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == 'matched/metadata.json':
                write_json(path, meta)
            elif name == 'matched/complete.json':
                write_json(path, dict(valid=True, updates=7200, arms={name: {'attempts': 30} for name in arms}))
            elif name == 'matched/predictor.pt':
                save_tensor(path, self.result)
            elif name == 'matched-predictions.npz':
                arrays, _ = c._matched_observations(self.plan, self.old)
                np.savez_compressed(path, **arrays)
            else:
                # Passive trajectories are present and hashed, never refit for this fixture.
                path.write_text(f'generated artifact {name}')
        observed = [str(self.relative / name) for name in self.names]
        obs_seal = c.seal(self.donor_root, 'R7_OBS', observed, expected=observed)
        (self.donor_root / 'seals' / 'R7_ORACLE.json').write_text('not available for donor reuse')
        self.plan['matched_tsba_reuse'] = dict(input_seal_sha256=input_seal['digest'], observation_seal_sha256=obs_seal['digest'])

    def _open(self):
        return m.open_donor(self.plan, self.donor_root, _REPO, self.tasks, self.names)

    def test_pinned_donor_matches_inputs_copies_replays_and_seals(self):
        donor = self._open()
        donor.verify_inputs(self.new, self.relative, self.task, self.plan)
        inputs = [str(self.relative / name) for name in (*c._INPUT_FILES, 'matched_input.pt')]
        c.seal(self.new_root, 'R7_INPUT', inputs, expected=inputs)
        tau = .005
        key = c.width_key(tau)

        def replay_heldout(*args, **kwargs):
            for name in c._shared_observations(tau):
                path = self.new / name
                if name == f'{key}-observed.json':
                    write_json(path, {'J': {'valid': True}})
                else:
                    np.savez_compressed(path, **{'w_J': np.zeros((2, 2))})

        with patch.object(c, 'replay_heldout', side_effect=replay_heldout), patch.object(c, 'exact_methods', return_value={'J'}):
            observed = c._freeze_observations(self.plan, self.plan, self.new_root, None,
                                              [(self.new, self.task)], tau, set(), set(), {}, donor=donor)
        c.seal(self.new_root, 'R7_OBS', observed, expected=observed)
        verified = c.verify_seal(self.new_root, 'R7_OBS')['files']
        copied = {str(self.relative / name) for name in self.names}
        self.assertEqual({name: verified[name] for name in copied}, donor.observation_members)
        with patch.object(c, 'fit_block') as shared, patch.object(c, 'fit_matched_tsba') as baseline:
            c._fit_blind(self.plan, self.new, 17, False, False, reuse_matched=True)
            shared.assert_called_once()
            baseline.assert_not_called()
            c._fit_blind(self.plan, self.new, 17, True, False)
            shared.assert_called_once()
            baseline.assert_called_once()

    def test_registration_changed_graph_order_and_corrupt_artifact_fail_closed(self):
        old = self.plan | {}
        old.pop('matched_tsba_reuse')
        with self.assertRaisesRegex(ValueError, 'requires matched_tsba_reuse'):
            m.open_donor(old, self.donor_root, _REPO, self.tasks, self.names)
        plan = self.plan | {'matched_tsba_reuse': self.plan['matched_tsba_reuse'] | {'observation_seal_sha256': '0' * 64}}
        with self.assertRaisesRegex(ValueError, 'seals differ'):
            m.open_donor(plan, self.donor_root, _REPO, self.tasks, self.names)
        donor = self._open()
        path = self.new / 'holdout_input.npz'
        held = c.read_npz(path)
        np.savez_compressed(path, **(held | {'holdout_ids': held['holdout_ids'][::-1]}))
        with self.assertRaisesRegex(ValueError, 'holdout_ids'):
            donor.verify_inputs(self.new, self.relative, self.task, self.plan)
        name = 'matched/A0/round11.pt'
        (self.old / name).write_text('changed after donor verification')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            donor.copy_observations(self.new, self.relative, [name])
        self.assertFalse((self.new / name).exists())

    def test_changed_code_or_predictions_fail_closed(self):
        def changed(path):
            return '0' * 64 if Path(path) == _REPO / 'tsba/config.py' else digest(path)
        with patch.object(m, 'digest', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                self._open()
        donor = self._open()
        donor.copy_observations(self.new, self.relative, self.names)
        predictions, state = c._matched_observations(self.plan, self.new)
        predictions['holdout_logits_A0'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'holdout_logits_A0'):
            donor.verify_replay(self.new, predictions, state)

    def test_c_oracle_calls_real_builder_through_adaptation_only_scope(self):
        tau = .005
        plan = {'phases': {'P': {'width_grid': [tau]}}}
        arrays = c.read_npz(self.new / 'fit_input.npz')
        held = c.read_npz(self.new / 'holdout_input.npz')
        adapt_u = np.column_stack((arrays['adapt_h'], np.ones(len(arrays['adapt_h']))))
        held_u = np.column_stack((held['holdout_h'], np.ones(len(held['holdout_h']))))
        np.savez_compressed(self.new / 'standardization.npz', adapt_z0=arrays['adapt_z0'], adapt_u=adapt_u)
        np.savez_compressed(self.new / 'holdout_standardization.npz', holdout_z0=held['holdout_z0'], holdout_u=held_u)
        labels = np.array([1, 0, 1])
        calls = []

        class Reader:
            def load(self, manifest, pool):
                calls.append((manifest, pool))
                return labels

        @contextmanager
        def scoped_reader(root, *, repository, stage):
            self.assertEqual((root, repository, stage), (self.new_root, _REPO, 'R7_OBS'))
            yield Reader()
            self.assertFalse((self.new_root / 'seals' / 'R7_ORACLE.json').exists())

        with patch.object(c, 'evaluation_labels', side_effect=scoped_reader), patch.object(p, 'config'), \
                patch.object(p, 'master', return_value=(np.zeros((2, 3)), {'valid': True})), \
                patch.object(p, 'load_labels', side_effect=AssertionError('unscoped label reader called')):
            files = c._freeze_oracles(plan, self.new_root, _REPO, [(self.new, self.task)], None, tau, set(), {})
        self.assertEqual(calls, [(str(self.relative / 'manifest.json'), 'adapt')])
        self.assertEqual(files, [str(self.relative / name) for name in c._oracle_files(tau)])
        np.testing.assert_array_equal(c.read_npz(self.new_root / files[0])['adapt_labels'], labels)

    def test_c_cohort_receives_frozen_actions_and_unchanged_probability_logits(self):
        tau = .005
        phases = self.plan['phases'] | {'P': {'independent_audit': {}}}
        legacy = self.plan | dict(phases=phases, shared_update={}, controls={})
        plan = legacy | {'decision_policy': {'version': d.VERSION, 'floor': 1e-8}}
        view = c._fit_view(plan, tau)
        self.assertNotIn('decision_policy', c._fit_view(legacy, tau))
        self.assertEqual(view['decision_policy'], plan['decision_policy'])
        self._open().copy_observations(self.new, self.relative, self.names)

        key = c.width_key(tau)
        std = dict(adapt_z0=np.array([[0., 2.], [0., 3.], [0., 3.]]), adapt_u=np.zeros((3, 3)))
        held = dict(holdout_z0=np.array([[0., 1.], [0., 4.]]), holdout_u=np.zeros((2, 3)),
                    holdout_ids=np.array([101, 102]))
        np.savez_compressed(self.new / 'standardization.npz', **std)
        np.savez_compressed(self.new / 'holdout_standardization.npz', **held)

        def artifacts(names):
            result = {}
            for name in names:
                w = np.zeros((2, 3))
                z = p.predict(name, std['adapt_z0'], std['adapt_u'], w)
                result.update({f'w_{name}': w, f'adapt_logits_{name}': z,
                               f'adapt_probability_{name}': c.softmax(z, axis=1)})
            return result

        observed = artifacts(('B0', 'J', 'C05'))
        record = {'valid': True, 'stop_reason': 'completed'}
        records = {name: record.copy() for name in ('B0', 'J', 'C05')}
        d.freeze_decisions(view, observed, records)
        labels = dict(adapt=np.array([1, 0, 1]), holdout=np.array([0, 1]), validation=np.array([0, 1]))
        oracle = artifacts(('O',)) | {'adapt_labels': labels['adapt']}
        oracle.update(c.replay_predictions(std, held, oracle))
        np.savez_compressed(self.new / f'{key}-oracle.npz', **oracle)
        np.savez_compressed(self.new / f'{key}-observed.npz', **observed)
        np.savez_compressed(self.new / f'{key}-heldout.npz', **c.replay_predictions(std, held, observed))
        write_json(self.new / f'{key}-observed.json', records)
        write_json(self.new / f'{key}-oracle.json', record)

        reader = SimpleNamespace(load=lambda _manifest, pool, **kwargs: labels[pool])
        with patch.object(c, '_endpoints', return_value=[]), patch.object(c, 'config'), \
                patch.object(c, '_bank_diagnostics', return_value=([], {}, None, None)):
            *_, cohorts = c._evaluate_frozen(plan, view, self.new_root, [(self.new, self.task)], tau, set(), reader)
        cohort, = cohorts
        np.testing.assert_array_equal(cohort['logits']['J'], cohort['logits']['J_argmax'])
        np.testing.assert_array_equal(cohort['decisions']['J'], [0, 1])
        np.testing.assert_array_equal(cohort['decisions']['J_argmax'], [1, 1])
        self.assertNotIn('A0', cohort['decisions'])
        self.assertIn(('J', 'C05_balanced'), d.decision_contrasts(plan, c._PRIMARY_CONTRASTS))
