"""Authenticate the single R8 source role and grant only per-direction labels."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.io import digest


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def npz_member(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as member:
        return dict(member)


class RoleLabels:
    """Use a role-bound direction grant, separately for fit and score."""

    def __init__(self, evidence, role_record, phase):
        field = {'fit': 'source_ids', 'score': 'holdout_ids'}[phase]
        self._approved = {(d['source'], d['target']): tuple(d[field]) for d in role_record['directions']}
        self._evidence = evidence
        self.actual = {}

    def load(self, source, target, requested):
        task = source, target
        approved = self._approved[task]
        if tuple(requested) != approved or not approved or task in self.actual:
            raise PermissionError('Label request is out of order, outside the sealed role, or repeated')
        result = self._evidence.label_values(requested, approved)
        self.actual[task] = list(requested)
        return result


def original_sources(repo, plan):
    parent = (repo/'experiments/mutagenicity/r8_图结构观察量与类别约束_20261121'
              /'results/attempt-001')
    raw = (parent/'OBS.json').read_bytes()
    if sha(raw) != plan['r8_observation_sha256']:
        raise PermissionError('R8 source role seal differs')
    obs = json.loads(raw)
    raw = (parent/'source-statistics.json').read_bytes()
    if sha(raw) != obs['files']['source-statistics.json']:
        raise PermissionError('R8 source-role member differs')
    source = json.loads(raw)[plan['source_role_key']]
    ids = source['source_fit_ids']
    if len(ids) != plan['allowed_graph_count'] or len(set(ids)) != len(ids):
        raise ValueError('Unexpected R8 source registration size')
    return ids


class SourceEvidence:
    def __init__(self, repo, plan):
        self.root = Path(plan['r5_archive']).resolve()
        raw = (self.root/'seals/R5_INPUT.json').read_bytes()
        if sha(raw) != plan['r5_input_seal_sha256']:
            raise PermissionError('R5 INPUT identity differs')
        input_record = json.loads(raw)
        if input_record['stage'] != 'R5_INPUT':
            raise ValueError('Expected the R5 INPUT stage')
        self.files = input_record['files']
        self.ids = original_sources(repo, plan)
        registered = self.read_json(f"roles/M0/seed{plan['source_model_seed']}.json")
        if registered['fit'] != self.ids:
            raise ValueError('R5 and R8 source-fit graph IDs or order differ')
        self.provenance = dict(r5_input_file_sha256=plan['r5_input_seal_sha256'],
                               r5_input_digest=input_record['digest'],
                               r8_obs_sha256=plan['r8_observation_sha256'],
                               scope='R8 M0-seed-3405 source fit only', source_fit_ids=self.ids)
        self.ids_set = set(self.ids)

    def read_json(self, relative):
        blob = (self.root/relative).read_bytes()
        if sha(blob) != self.files[relative]:
            raise PermissionError(f'Sealed R5 member differs: {relative}')
        return json.loads(blob)

    def graphs(self):
        relative = 'inputs/graphs.pt'
        path = self.root/relative
        if digest(path) != self.files[relative]:
            raise PermissionError('Sealed R5 graph archive differs')
        # The full file is hashed. mmap leaves unselected tensor storage
        # untouched by the feature extractor.
        recorded = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
        rows = self.read_json('inputs/graph_hashes.json')
        outputs = {}
        for i in self.ids:
            graph = recorded[i]
            if set(graph) != {'x', 'edge_index', 'edge_attr'}:
                raise ValueError(f'R5 graph {i} carries unexpected fields')
            actual = {k: sha(v.numpy().tobytes()) for k, v in graph.items()}
            if rows[i]['id'] != i or actual != rows[i]['input_sha256']:
                raise ValueError(f'R5 individual graph identity differs: {i}')
            outputs[i] = graph
        self.provenance['graphs_sha256'] = self.files[relative]
        return outputs

    def label_values(self, ids, allowed_ids):
        """Parse only explicitly requested authorized graph offsets."""
        if len(set(ids)) != len(ids) or set(ids) != set(allowed_ids) or not set(ids) <= self.ids_set:
            raise PermissionError('Label request differs from the granted source or sealed H cohort')
        meta = self.read_json('inputs/label_records.json')
        raw = Path(meta['path']).read_bytes()
        if sha(raw) != meta['sha256'] or sorted(meta['mapping'].values()) != [0, 1]:
            raise PermissionError('R5 label bytes or class mapping changed')
        records = []
        for i in ids:
            offset, length = meta['offsets'][i]
            piece = raw[offset:offset+length]
            if len(piece) != length:
                raise ValueError(f'Label offset outside authenticated file: {i}')
            records.append(meta['mapping'][str(int(piece))])
        return np.array(records, dtype=np.int64)
