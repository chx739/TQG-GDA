import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.io import digest
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import scoring


class ScoreReceiptTests(unittest.TestCase):
    def test_observation_binds_job_arrays_and_historical_actions_to_their_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = scoring.jobs.target_key(3405, 1)
            ids = [12, 15]
            roles = {'tasks': {key: {'h_ids': ids, 'target': 1}}}
            # These historical actions deliberately differ from argmax.
            margins = np.array([1., -2.])
            actions = np.array([0, 1])
            history, row = {}, {'ids': np.array(ids)}
            for name in scoring.HISTORICAL:
                history['z_' + name] = margins
                history['decision_' + name] = actions
                row['logits_' + name] = np.column_stack((np.zeros_like(margins), margins))
                row['actions_' + name] = actions.copy()
            jobs, files = [], {}
            for target, arm, logits in [(None, 'SourceOnly', [[1., 0.], [1., 0.]]),
                                        (1, 'GRADE', [[0., 1.], [1., 0.]])]:
                attempt = root / arm
                attempt.mkdir()
                path = attempt / 'prediction-M1.pt'
                values = torch.tensor(logits)
                torch.save(dict(ids=torch.tensor(ids), logits=values, decisions=values.argmax(1)), path)
                files[str(path.relative_to(root))] = digest(path)
                jobs.append((arm, 3405, target, attempt, {'prediction_targets': [1]}))
                row['logits_' + arm] = np.array(logits)
                row['actions_' + arm] = np.argmax(logits, axis=1)
            historical_lineage = {key: dict(h_ids=ids, r8_sha256='a'*64, fixed_mass_sha256='b'*64)}
            lineage = dict(models=2, blocks=1, historical=historical_lineage,
                           evaluation_worktree='/previous/checkout', checkpoint_replay_max_abs_logit_error=0.)
            (root / 'observation').mkdir()
            path = root / 'observation/lineage.json'
            path.write_text(json.dumps(lineage))
            sources = ({key: history}, historical_lineage)
            with patch.object(scoring, '_upstream_role_arrays', return_value=sources):
                verify = scoring._verify_prediction_sources
                verify(root, root, roles, {}, jobs, files, {key: row})
                switched = row['logits_GRADE'][:, ::-1]
                for changed in ({'logits_GRADE': switched, 'actions_GRADE': switched.argmax(axis=1)},
                                {'actions_Joint': 1 - actions}, {'logits_TargetSlope': -row['logits_TargetSlope']}):
                    with self.subTest(fields=list(changed)), self.assertRaisesRegex(PermissionError, 'source prediction'):
                        verify(root, root, roles, {}, jobs, files, {key: row | changed})
                path.write_text(json.dumps(lineage | {'historical': {}}))
                with self.assertRaisesRegex(PermissionError, 'lineage'):
                    verify(root, root, roles, {}, jobs, files, {key: row})

    def test_ten_prediction_paths_alone_cannot_open_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'INPUT.json').write_text('{}')
            tasks = {f'block-{i}': {} for i in range(10)}
            record = dict(stage='R10_OBS', blocks=10, models=15, input_sha256=digest(root / 'INPUT.json'),
                          files={f'observation/{key}.npz': 'unverified' for key in tasks})
            (root / 'OBS.json').write_text(json.dumps(record))
            with patch.object(scoring.r5_inputs, 'evaluation_labels') as labels:
                with self.assertRaises(PermissionError):
                    scoring.score(root, root, {'tasks': tasks}, {}, root)
                labels.assert_not_called()

    def test_existing_final_must_bind_the_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'evaluation'
            output.mkdir()
            (root / 'OBS.json').write_text('{}')
            obs_sha = digest(root / 'OBS.json')
            for name in ('endpoints.csv','paired-seeds.csv','paired-intervals.json'):
                (output / name).write_text(name)
            tasks = {f'{seed}-{target}':dict(target=target, h_ids=[target, target+10])
                     for seed in range(5) for target in (1, 3)}
            config = dict(plan={'execution': {'torch_threads_per_worker': 1, 'interop_threads': 1}},
                          code_commit='a'*40)
            access = dict(obs_sha256=obs_sha,target_ids={'1':[1,11],'3':[3,13]},evaluated_blocks=len(tasks))
            execution = dict(trained_with=config['code_commit'], code_commit='b'*40, scoring_sha256='c'*64,
                             started_utc='2026-11-21T00:00:00+00:00')
            (root / 'FINAL.json').write_text('{"files":{}}')
            with (patch.object(scoring.data, 'read_inputs', return_value=({'tasks':tasks}, config)),
                  patch.object(scoring, 'verify_observation', return_value=(obs_sha, {})),
                  patch.object(scoring, 'score') as rescore,
                  patch.object(scoring.torch, 'set_num_threads'),
                  patch.object(scoring.torch, 'set_num_interop_threads'),
                  patch.object(scoring.torch, 'use_deterministic_algorithms')):
                cases = [
                    (access, execution, 'Invalid R10 final record'),
                    (access | {'target_ids':{'1':[11,1],'3':[3,13]}}, execution, 'label access'),
                    (access | {'obs_sha256':'d'*64}, execution, 'label access'),
                    (access | {'evaluated_blocks':9}, execution, 'label access'),
                    (access, execution | {'trained_with':'d'*40}, 'evaluation execution'),
                    (access, execution | {'scoring_sha256':''}, 'evaluation execution'),
                ]
                for label_access, score_execution, failure in cases:
                    with self.subTest(label_access=label_access, score_execution=score_execution):
                        (output/'label-access.json').write_text(json.dumps(label_access))
                        (output/'execution.json').write_text(json.dumps(score_execution))
                        members = {name: digest(output / name) for name in scoring.EVALUATION_FILES}
                        receipt = dict(observation_sha256=obs_sha, files=members)
                        (output/'COMPLETE.json').write_text(json.dumps(receipt))
                        with self.assertRaisesRegex(PermissionError, failure):
                            scoring.run(root, root)
                rescore.assert_not_called()
