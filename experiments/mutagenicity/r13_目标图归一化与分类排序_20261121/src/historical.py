"""Authentic, label-free R5/R7 inputs plus historical H comparator decisions."""
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.common.io import digest
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.inputs import npz_member, sha
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11_inputs
from tsba.encoder import GraphEncoder
from tsba.pretrain import SourceModel


def selected_ids(setting, ids, n):
    if type(n) is not int or n < 1 or n > len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Cannot select the required unique statistics IDs')
    return sorted(ids, key=lambda i: (sha256(f'r13-v1|{setting}|{i}'.encode('utf-8')).digest(), i))[:n]


def member(root, checksums, name):
    payload = r5_inputs._inside(root, name).read_bytes()
    if sha(payload) != checksums[name]:
        raise PermissionError(f'Historical read does not match its sealed checksum: {name}')
    return payload


class CArchive:
    def __init__(self, plan):
        self.root = Path(plan['r7_archive'])
        self.digests, self.members = {}, {}
        parent = None
        for stage in seals.STAGES:
            seal = json.loads((self.root / f'seals/{stage}.json').read_bytes())
            record = {key: seal[key] for key in ('stage', 'parent', 'files')}
            if seal['stage'] != stage or seal['parent'] != parent or seals._signature(record) != seal['digest']:
                raise PermissionError('Historical C seal chain changed')
            self.members[stage] = seal['files']
            parent = seal['digest']
        if parent != plan['r7_final_digest']:
            raise PermissionError('Historical C final identity differs')
        self.plan = json.loads(self.read('R7_INPUT', 'executed-plan.json'))
        phase = self.plan['phases']['C']
        if (phase['settings'] != plan['settings'] or phase['domains'] != plan['domains']
                or self.plan['real_model_seeds'] != [plan['phase1_seed'], *plan['reserve_seeds']]):
            raise ValueError('R13 grid differs from historical C')
        self.tau = phase['primary_tau']

    def read(self, stage, name):
        result = member(self.root, self.members[stage], name)
        self.digests[name] = self.members[stage][name]
        return result

    def roles(self, task):
        key = r11_inputs.key(**task)
        blob = self.read('R7_INPUT', f'blocks/{key}/manifest.json')
        value = json.loads(blob)
        if any(value[k] != val for k, val in task.items()):
            raise PermissionError('Historical manifest task identity differs')
        return value, sha(blob)

    def b0(self, task, ids):
        key = r11_inputs.key(**task)
        path = f'blocks/{key}'
        z = npz_member(self.read('R7_INPUT', f'{path}/holdout_input.npz'), ['holdout_ids', 'holdout_z0'])
        values = npz_member(self.read('R7_OBS', f'{path}/tau-{self.tau:.3f}-heldout.npz'), ['holdout_logits_B0'])
        np.testing.assert_array_equal(ids, z['holdout_ids'])
        raw = z['holdout_z0']
        centered = raw - raw.max(axis=1, keepdims=True)
        np.testing.assert_allclose(centered, values['holdout_logits_B0'], atol=1e-12, rtol=0.)
        np.testing.assert_array_equal(raw.argmax(axis=1), values['holdout_logits_B0'].argmax(axis=1))
        if z['holdout_z0'].dtype != np.float64 or z['holdout_z0'].shape != (len(ids), 2):
            raise ValueError('Expected binary historical float64 B0 logits')
        return z['holdout_z0']


class GraphStore:
    def __init__(self, context):
        root = context['root']
        payload = (root/'seals/R5_INPUT.json').read_bytes()
        if sha(payload) != context['provenance']['r5_files']['seals/R5_INPUT.json']:
            raise PermissionError('R5 graph container authority changed')
        files = json.loads(payload)['files']
        name = 'inputs/graphs.pt'
        path = r5_inputs._inside(root, name)
        self.graph_sha = files[name]
        if digest(path) != self.graph_sha:
            raise PermissionError('R5 graph container changed')
        self.graphs = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
        self.hashes = json.loads(context['read']('inputs/graph_hashes.json', 'INPUT'))
        if len(self.graphs) != len(self.hashes) or len(self.graphs) != context['total']:
            raise ValueError('Historical graph inventory differs from the registered setting')
        self.graph_cache = {}

    def take(self, ids):
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Expected unique graph IDs')
        for i in ids:
            if i in self.graph_cache:
                continue
            record = self.graphs[i]
            if set(record) != {'x', 'edge_index', 'edge_attr'}:
                raise PermissionError('An archived graph has fields beyond the graph-only schema')
            hashes = {k: None if v is None else sha(v.numpy().tobytes()) for k, v in record.items()}
            expected = self.hashes[i]
            if expected['id'] != i or hashes != expected['input_sha256']:
                raise PermissionError(f'R5 graph or graph order differs for {i}')
            self.graph_cache[i] = Data(**record)
        return [self.graph_cache[i] for i in ids]


def context(c, task, repo):
    manifest, checksum = c.roles(task)
    data = r5_inputs._context(manifest['r7_plan'], **task, repository=repo)
    prior = manifest['provenance']
    provenance = data['provenance']
    if data['ids'] != manifest['ids'] or any(prior[k] != provenance[k]
        for k in ('archive_root', 'execution_commit', 'r5_seal_digests', 'repository_files')):
        raise PermissionError('R7 task role or R5 provenance changed')
    return manifest, checksum, data


def checkpoint(manifest, data):
    names = [name for name in manifest['provenance']['r5_files'] if name.endswith('/source.pt')]
    if len(names) != 1:
        raise ValueError('Expected one registered R5 source checkpoint')
    name = names[0]
    payload = data['read'](name)
    checksum = manifest['provenance']['r5_files'][name]
    if sha(payload) != checksum:
        raise PermissionError('R5 model differs from historical R7 model identity')
    record = torch.load(io.BytesIO(payload), weights_only=True, map_location='cpu')
    history = record['history']
    model_plan = data['old_plan']['model']
    if (len(history) != model_plan['source_epochs'] or [row['epoch'] for row in history] != list(range(len(history)))
            or not all(np.isfinite(row['source_validation']['ce']) for row in history)
            or record['selected_epoch'] != min(history, key=lambda row: row['source_validation']['ce'])['epoch']
            or record['selected_epoch'] != manifest['provenance']['selected_source_epoch']):
        raise ValueError('Original R5 source-validation checkpoint selection differs')
    model = SourceModel(GraphEncoder(**record['encoder_spec']), 2)
    model.load_state_dict(record['state'], strict=True)
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise ValueError('Expected the historical float32 encoder and source head')
    model.requires_grad_(False).eval()
    model.encoder.freeze()
    return model, dict(path=name, sha256=checksum, selected_epoch=record['selected_epoch'],
                       batch_size=model_plan['source_batch'], encoder=record['encoder_spec'])


class ReferenceArms:
    def __init__(self, config, repo):
        controls = config['reference_controls']
        self.r11 = repo / controls['r11_attempt']
        self.r10 = Path(controls['r10_archive'])
        self.obs11 = self._observe(self.r11, 'UNLABELED.json', 'UNLABELED', controls['r11_unlabeled_sha256'])
        self.obs10 = self._observe(self.r10, 'OBS.json', 'R10_OBS', controls['r10_obs_sha256'])

    @staticmethod
    def _observe(path, name, stage, expected):
        blob = (path / name).read_bytes()
        record = json.loads(blob)
        if sha(blob) != expected or record['stage'] != stage:
            raise PermissionError(f'Sealed comparison does not match: {path}/{name}')
        return record['files']

    def get(self, task, ids, b0):
        key = r11_inputs.key(**task)
        path = f'unlabeled/{key}.npz'
        val = npz_member(member(self.r11, self.obs11, path), ['ids', 'margin', 'calibrated', 'SourceCal'])
        np.testing.assert_array_equal(val['ids'], ids)
        np.testing.assert_allclose(val['margin'], b0[:, 1] - b0[:, 0], atol=1e-12, rtol=0.)
        np.testing.assert_array_equal(val['SourceCal'], val['calibrated'] > 0.)
        arms = {'SourceCal': np.column_stack((np.zeros(len(ids)), val['calibrated']))}
        receipt = {'SourceCal': self.obs11[path]}
        path = f'observation/{key}.npz'
        if path in self.obs10:
            names = ('B0', 'SourceCal', 'GRADE', 'SourceOnly', 'SourceFusion')
            z = npz_member(member(self.r10, self.obs10, path),
                           ['ids', *(f'{stem}_{name}' for name in names for stem in ('logits', 'actions'))])
            np.testing.assert_array_equal(z['ids'], ids)
            for name, baseline in (('B0', b0), ('SourceCal', arms['SourceCal'])):
                historical = z[f'logits_{name}']
                np.testing.assert_allclose(historical[:,1] - historical[:,0],
                                           baseline[:,1] - baseline[:,0], atol=1e-12, rtol=0.)
                np.testing.assert_array_equal(z[f'actions_{name}'], baseline.argmax(1))
            for name in ('GRADE', 'SourceOnly', 'SourceFusion'):
                np.testing.assert_array_equal(z[f'actions_{name}'], z[f'logits_{name}'].argmax(1))
                arms[name] = z[f'logits_{name}']
                receipt[name] = self.obs10[path]
        return arms, receipt
