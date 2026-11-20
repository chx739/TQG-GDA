import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.common.io import digest
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import scoring


class ScoreReceiptTests(unittest.TestCase):
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
            for name in scoring.EVALUATION_FILES:
                (output / name).write_text(name)
            members = {name: digest(output / name) for name in scoring.EVALUATION_FILES}
            (output / 'COMPLETE.json').write_text(json.dumps(dict(observation_sha256=obs_sha, files=members)))
            (root / 'FINAL.json').write_text('{"files":{}}')
            config = {'plan': {'execution': {'torch_threads_per_worker': 1, 'interop_threads': 1}}}
            with (patch.object(scoring.data, 'read_inputs', return_value=({}, config)),
                  patch.object(scoring, 'verify_observation', return_value=(obs_sha, {})),
                  patch.object(scoring, 'score') as rescore,
                  patch.object(scoring.torch, 'set_num_threads'),
                  patch.object(scoring.torch, 'set_num_interop_threads'),
                  patch.object(scoring.torch, 'use_deterministic_algorithms')):
                with self.assertRaises(PermissionError):
                    scoring.run(root, root)
                rescore.assert_not_called()
