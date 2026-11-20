"""Authenticated source roles and validation B0, without opening R7 target members."""
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.io import digest
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.inputs import npz_member, sha


class ValidationArchive:
    def __init__(self, config):
        self.root = Path(config['r7_archive'])
        self.source_reads = {}
        parent = None
        for stage in seals.STAGES:
            record = json.loads((self.root / f'seals/{stage}.json').read_bytes())
            signed = {key: record[key] for key in ('stage', 'parent', 'files')}
            if stage != record['stage'] or record['parent'] != parent or seals._signature(signed) != record['digest']:
                raise PermissionError(f'R7 seal chain metadata differs at {stage}')
            parent = record['digest']
            if stage == 'R7_INPUT':
                self.files = record['files']
        if parent != config['r7_final_digest']:
            raise PermissionError('R7 archive does not match the registered C run')
        # All other C inputs and outputs remain unopened, including A/H payloads.
        self.plan = json.loads(self._read('executed-plan.json'))
        if not set(config['settings']) <= set(self.plan['phases']['C']['settings']):
            raise ValueError('Configuration requests settings outside historical C')
        if config['seed'] not in self.plan['real_model_seeds'] or config['domains'] != self.plan['phases']['C']['domains']:
            raise ValueError('Configuration differs from historical C roles')
        self.domains = tuple(config['domains'])
        self.seeds = (config['seed'],)
        self.settings = tuple(config['settings'])

    def _read(self, name):
        payload = r5_inputs._inside(self.root, name).read_bytes()
        if sha(payload) != self.files[name]:
            raise PermissionError(f'R7 source member differs: {name}')
        self.source_reads[name] = self.files[name]
        return payload

    def validation(self, setting, source, seed):
        if setting not in self.settings or source not in self.domains or seed not in self.seeds:
            raise PermissionError('R7 validation request is outside registered source roles')
        # The donor direction names only the authenticated copy of the shared V.
        donor = next(t for t in self.domains if t != source)
        name = f'blocks/{setting}-M{source}_to_M{donor}-seed-{seed}/validation_input.npz'
        return npz_member(self._read(name), ['validation_ids', 'validation_h', 'validation_z0'])


class SourceArchive:
    def __init__(self, repo, registration, plan, spec, domains, seed, *, load_graphs=True):
        self.repo = repo
        self.root = Path(registration['archive_root'])
        self.spec, self.domains, self.seed = spec, tuple(domains), seed
        self.receipt, self.label_reads = {}, {}
        self.seal_files = {}
        for role in ('INPUT', 'OBS'):
            name = f'seals/R5_{role}.json'
            payload = (self.root / name).read_bytes()
            seal = json.loads(payload)
            signature = {k: seal[k] for k in ('stage', 'files')}
            if (sha(payload) != registration['seal_file_sha256'][role] or seal['stage'] != f'R5_{role}'
                    or seal['digest'] != registration['seal_digests'][role]
                    or r5_inputs._identity(signature, compact=True) != seal['digest']):
                raise PermissionError(f'R5 seal metadata differs: {name}')
            self.seal_files[role] = seal['files']
            self.receipt[name] = sha(payload)
        self.plan = json.loads(self._read('protocol/plan.json'))
        if (self.plan != plan or self.plan['dataset'] != spec['tu_dataset']
                or self.plan['setting']['id'] != spec['id'] or seed not in self.plan['seeds']):
            raise ValueError('R5 source design differs from its registered setting')
        self.count = sum(spec['domain_counts'])
        if load_graphs:
            graph_path = self._member('inputs/graphs.pt', 'INPUT')
            if digest(graph_path) != self.receipt['inputs/graphs.pt']:
                raise PermissionError('R5 graph container differs')
            self.graphs = torch.load(graph_path, mmap=True, weights_only=True, map_location='cpu')
            self.graph_hashes = json.loads(self._read('inputs/graph_hashes.json'))
            if len(self.graphs) != self.count or len(self.graph_hashes) != self.count:
                raise ValueError('R5 graph inventory differs')

    def _member(self, name, stage):
        checksum = self.seal_files[stage][name]
        if self.seal_files['INPUT'].get(name, checksum) != checksum:
            raise PermissionError(f'R5 source member changed between seals: {name}')
        self.receipt[name] = checksum
        return r5_inputs._inside(self.root, name)

    def _read(self, name, stage='INPUT'):
        payload = self._member(name, stage).read_bytes()
        if sha(payload) != self.receipt[name]:
            raise PermissionError(f'R5 source member differs: {name}')
        return payload

    def _load(self, name, stage='INPUT'):
        return torch.load(io.BytesIO(self._read(name, stage)), weights_only=True, map_location='cpu')

    def role(self, source):
        if source not in self.domains:
            raise PermissionError('R5 role request is outside registered source domains')
        roles = json.loads(self._read(f'roles/M{source}/seed{self.seed}.json'))
        roles = {k: roles[k] for k in ('fit', 'validation', 'class_counts')}
        for role in ('fit', 'validation'):
            r5_inputs._ids(roles[role], self.count, role)
        domain = self.plan['setting']['domain_indices'][source]
        name = self.spec['domain_index_pattern'].format(domain=source)
        payload = r5_inputs._inside(self.repo, name).read_bytes()
        ids = np.load(io.BytesIO(payload), allow_pickle=False)
        n = self.spec['domain_counts'][source]
        if (domain['domain'] != source or domain['path'] != name or sha(payload) != domain['sha256']
                or ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer) or domain['count'] != n
                or sorted(roles['fit'] + roles['validation']) != sorted(r5_inputs._ids(ids.tolist(), self.count, name))):
            raise PermissionError('Source fit/validation do not partition the authenticated source domain')
        self.receipt[f'repository:{name}'] = sha(payload)
        return roles

    def source_graphs(self, source):
        roles = self.role(source)
        output = {}
        for graph_id in roles['fit'] + roles['validation']:
            graph = self.graphs[graph_id]
            record = self.graph_hashes[graph_id]
            if type(graph) is not dict or set(graph) != {'x', 'edge_index', 'edge_attr'}:
                raise ValueError('Only label-free R5 graph records are allowed')
            actual = {key: None if value is None else sha(value.numpy().tobytes()) for key, value in graph.items()}
            if record['id'] != graph_id or record['input_sha256'] != actual:
                raise PermissionError(f'R5 source graph bytes differ: {graph_id}')
            output[graph_id] = graph
        return output

    def labels(self, source, role, requested):
        if role not in ('fit', 'validation'):
            raise PermissionError('Only S and V labels are authorized')
        roles = self.role(source)
        if list(requested) != roles[role] or (source, role) in self.label_reads:
            raise PermissionError('Source label request changed cohort, order or repeated')
        name = f'source_labels/M{source}/seed{self.seed}_{role}.pt'
        value = self._load(name)
        y = r5_inputs._tensor(value, (len(requested),), f'source {role} labels', integer=True).numpy()
        if np.bincount(y, minlength=2).tolist() != roles['class_counts'][role] or not np.isin(y, (0, 1)).all():
            raise ValueError(f'Source labels disagree with sealed {role} counts')
        self.label_reads[source, role] = list(requested)
        return y

    def b0(self, source, roles, validation, features):
        config_hash = r5_inputs._identity({k: v for k, v in self.plan.items() if k != 'workers'})
        name = f"source/{self.plan['dataset']}/M{source}/seed{self.seed}/{config_hash}/source.pt"
        state = self._load(name, 'OBS')
        if state['selected_epoch'] != min(state['history'], key=lambda r: r['source_validation']['ce'])['epoch']:
            raise ValueError('R5 checkpoint differs from source-validation CE selection')
        w, b = (state[features['source_model_state_key']][name].double().numpy()
                for name in features['source_head_state_keys'])
        np.testing.assert_array_equal(roles['validation'], validation['validation_ids'])
        logits = validation['validation_z0']
        if logits.dtype != np.float64 or logits.shape != (len(roles['validation']), 2):
            raise ValueError('Expected historical binary float64 validation logits')
        np.testing.assert_allclose(validation['validation_h'] @ w.T + b, logits, rtol=0, atol=1e-12)
        if not np.isfinite(logits).all():
            raise ValueError('R7 validation logits are not finite')
        return logits[:, 1] - logits[:, 0], dict(path=name, selected_epoch=state['selected_epoch'], sha256=self.receipt[name])
