"""Authenticated reuse of label-blind A0/B1 observations from a sealed C run."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.io import digest

from . import matched_tsba
from .seals import _file, verify_seal


_EXTRA_CODE = (
    'tsba/config.py', 'tsba/utils.py', 'experiments/common/io.py',
    'experiments/mutagenicity/r2_分数训练预算与多轮适配_20261111/src/adapt.py',
    'experiments/mutagenicity/r3_源风险与置信度约束_20261112/src/protocol.py',
)
_ARRAY_INPUTS = ('fit_input.npz', 'holdout_input.npz', 'validation_input.npz')


def _verified_bytes(root, name, members):
    if name not in members:
        raise ValueError(f'Matched donor artifact is outside its required seal: {name}')
    value = _file(root, name).read_bytes()
    if hashlib.sha256(value).hexdigest() != members[name]:
        raise ValueError(f'Sealed matched donor artifact changed: {name}')
    return value


def _assert_same(old, new, label):
    if type(old) is not type(new):
        raise ValueError(f'Matched donor differs at {label}')
    if isinstance(old, dict):
        same = old.keys() == new.keys()
        if same:
            for key in old:
                _assert_same(old[key], new[key], f'{label}/{key}')
    elif isinstance(old, (list, tuple)):
        same = len(old) == len(new)
        if same:
            for index, (left, right) in enumerate(zip(old, new)):
                _assert_same(left, right, f'{label}/{index}')
    elif isinstance(old, torch.Tensor):
        same = old.dtype == new.dtype and old.shape == new.shape and torch.equal(old, new)
    elif isinstance(old, np.ndarray):
        same = old.dtype == new.dtype and old.shape == new.shape and np.array_equal(old, new)
    else:
        same = old == new
    if not same:
        raise ValueError(f'Matched donor differs at {label}')


def _provenance_values(provenance):
    value = deepcopy(provenance)
    if 'source_refit' in value:
        # A retrained checkpoint can occupy a new directory. Its registered
        # roles, preparation/artifact digests and all file hashes must match.
        value['source_refit'].pop('artifact_root', None)
    return value


def _manifest_values(manifest):
    return {**{key: value for key, value in manifest.items() if key != 'r7_plan'},
            'provenance': _provenance_values(manifest['provenance'])}


def _json_values(value):
    """Use the same primitive containers as archived JSON metadata."""
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True)
class MatchedDonor:
    root: Path
    input_members: dict
    observation_members: dict
    lock: dict
    model_dependencies: dict
    implementation_sha256: str

    def verify_inputs(self, block, relative, task, view):
        """Prove the new registered task would fit and score the same A0/B1."""
        relative = Path(relative)
        label = relative.name
        old_manifest = json.loads(_verified_bytes(self.root, str(relative / 'manifest.json'), self.input_members))
        new_manifest = json.loads((block / 'manifest.json').read_text())
        _assert_same(_manifest_values(old_manifest), _manifest_values(new_manifest), f'{label}/manifest')
        for name in _ARRAY_INPUTS:
            payload = _verified_bytes(self.root, str(relative / name), self.input_members)
            with np.load(io.BytesIO(payload), allow_pickle=False) as old, np.load(block / name, allow_pickle=False) as new:
                _assert_same(dict(old), dict(new), f'{label}/{name}')
        payload = _verified_bytes(self.root, str(relative / 'matched_input.pt'), self.input_members)
        old = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        new = torch.load(block / 'matched_input.pt', map_location='cpu', weights_only=True)
        if set(old) != matched_tsba._BLIND_KEYS or set(new) != matched_tsba._BLIND_KEYS:
            raise ValueError(f'{label}: matched input differs from the blind whitelist')
        _assert_same(old | {'provenance': _provenance_values(old['provenance'])},
                     new | {'provenance': _provenance_values(new['provenance'])}, f'{label}/matched_input.pt')
        setting, source, target, seed = task
        identity = dict(setting=setting, source=source, target=target, seed=seed)
        _assert_same(new['task'], identity, f'{label}/task')
        cfg, _, head, _, _, _, adapt, ids = matched_tsba._prepare(view, new, seed)
        metadata_name = str(relative / 'matched' / 'metadata.json')
        metadata = json.loads(_verified_bytes(self.root, metadata_name, self.observation_members))
        budget = view['phases']['C']['matched_tsba']
        expected = dict(task=identity, method='R7_matched_R5_M00', seed=seed,
                        r5_model_config=new['r5_model_config'], effective_adaptation_config=_json_values(asdict(cfg)),
                        adam=new['r5_adam_config'], score_exact_updates=budget['score_exact_updates'],
                        attempts_per_group=budget['attempts_per_group'], seed_offsets=budget['seed_offsets'],
                        source_count=len(ids[0]), adaptation_count=len(adapt), target_labels_used=False,
                        holdout_inputs_received=False, score_uses_adaptation_pool_only=True,
                        original_global_RMS_coordinates=True, torch_threads=1,
                        score_initialization_dtype='float32_then_float64',
                        initial_head_sha256=matched_tsba._state_hash(matched_tsba._state(head)),
                        r5_provenance=old['provenance'], implementation_sha256=self.implementation_sha256,
                        implementation_dependencies_sha256=self.model_dependencies)
        for key, value in _json_values(expected).items():
            _assert_same(metadata[key], value, f'{label}/metadata/{key}')

    def copy_observations(self, block, relative, names):
        for name in names:
            payload = _verified_bytes(self.root, str(Path(relative) / name), self.observation_members)
            destination = block / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(payload)

    def verify_replay(self, block, replay, state):
        with np.load(block / 'matched-predictions.npz', allow_pickle=False) as saved:
            _assert_same(dict(saved), replay, f'{block.name}/matched-predictions.npz')
        metadata = json.loads((block / 'matched' / 'metadata.json').read_text())
        _assert_same(metadata, _json_values(state['metadata']), f'{block.name}/predictor/metadata')


def open_donor(plan, location, repo, tasks, artifact_names):
    """Match the registered C donor using only its INPUT and OBS seals."""
    registration = plan.get('matched_tsba_reuse')
    if 'matched_tsba_reuse' not in plan:
        if location is not None:
            raise ValueError('--matched-from requires matched_tsba_reuse in the executed plan')
        return None
    fields = {'input_seal_sha256', 'observation_seal_sha256'}
    if (not isinstance(registration, dict) or registration.keys() != fields
            or any(not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef')
                   for value in registration.values())):
        raise ValueError('matched_tsba_reuse requires pinned input_seal_sha256 and observation_seal_sha256')
    if location is None:
        raise ValueError('Registered matched_tsba_reuse requires --matched-from')
    root = Path(location).resolve()
    input_record = verify_seal(root, 'R7_INPUT')
    observed = verify_seal(root, 'R7_OBS')
    if (input_record['digest'] != registration['input_seal_sha256']
            or observed['digest'] != registration['observation_seal_sha256']
            or observed['parent'] != input_record['digest']):
        raise ValueError('Matched donor seals differ from the executed plan')
    inputs, observations = input_record['files'], observed['files']
    old_plan, configs, provenance, blocks = [json.loads(_verified_bytes(root, name, inputs)) for name in (
        'executed-plan.json', 'executed-configs.json', 'provenance.json', 'blocks.json')]
    if configs['phase'] != 'C' or configs['threads_per_worker'] != 1 or plan['execution']['threads_per_worker'] != 1:
        raise ValueError('Matched donor must be a C run using the registered worker thread count')
    if bool(configs.get('source_refit_applied')) != bool(plan.get('source_refit')):
        raise ValueError('Matched donor applied a different source refit')
    for name in ('real_settings', 'features', 'real_model_seeds', 'target_split', 'source_refit'):
        _assert_same(old_plan.get(name), plan.get(name), f'plan/{name}')
    _assert_same(configs.get('source_refit'), plan.get('source_refit'), 'executed-configs/source_refit')
    for key in ('settings', 'domains', 'matched_tsba'):
        _assert_same(old_plan['phases']['C'][key], plan['phases']['C'][key], f'plan/C/{key}')
    _assert_same(configs['matched_tsba'], plan['phases']['C']['matched_tsba'], 'executed-configs/matched_tsba')
    actual = [(row['path'], (row['setting'], row['source'], row['target'], row['seed'])) for row in blocks]
    _assert_same(actual, list(tasks), 'registered C task grid')
    for block, _ in tasks:
        for name in (*_ARRAY_INPUTS, 'manifest.json', 'matched_input.pt'):
            if str(Path(block) / name) not in inputs:
                raise ValueError(f'Missing matched donor INPUT artifact: {block}/{name}')
        for name in artifact_names:
            if str(Path(block) / name) not in observations:
                raise ValueError(f'Missing matched donor OBS artifact: {block}/{name}')
    module = Path(matched_tsba.__file__).resolve()
    expected_code = {name: digest(repo / name) for name in (*matched_tsba._DEPENDENCIES, *_EXTRA_CODE)}
    implementation = digest(module)
    expected_code[str(module.relative_to(repo))] = implementation
    for name, checksum in expected_code.items():
        if provenance['file_hashes'].get(name) != checksum:
            raise ValueError(f'Matched donor baseline implementation changed: {name}')
    if provenance['torch'] != torch.__version__:
        raise ValueError('Matched donor PyTorch version differs from the current replay runtime')
    lock = registration | {'root': str(root), 'git_commit': provenance['git_commit'], 'blocks': len(tasks)}
    dependencies = {name: expected_code[name] for name in matched_tsba._DEPENDENCIES}
    return MatchedDonor(root, inputs, observations, lock, dependencies, implementation)
