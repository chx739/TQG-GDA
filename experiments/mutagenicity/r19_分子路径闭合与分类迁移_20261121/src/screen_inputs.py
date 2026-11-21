"""Bind the R19 input roles, structure counts and frozen R9 static WL margins."""
import csv
import json
from zipfile import ZipFile

import numpy as np
import torch
from torch_geometric.data import Data

from experiments.common.io import digest
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import (
    feature_dropout_transfer as dropout, reference_geometry, residual_inputs as data, true_rate_replay, wl)
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run, roc_transfer


R9 = 'r9_目标近邻与源标签传递_20261121'
GEOMETRY_CONFIG = 'configs/reference-geometry.json'
STATIC_CONFIG = 'configs/feature-dropout-transfer-check.json'
PARENTS = ('structure_sha256', 'geometry_obs_sha256', 'static_obs_sha256',
           'static_original_zip_sha256', 'static_original_manifest_sha256')


def parents(cfg):
    return {name: cfg[name] for name in PARENTS}


def structure_counts(path, expected_zip, roles_sha):
    if digest(path) != expected_zip:
        raise PermissionError('R19 structural precheck archive differs')
    with ZipFile(path) as zip_file:
        metadata = json.loads(zip_file.read('inputs.json'))
        if metadata['R9_roles'] != roles_sha or inputs.sha(zip_file.read('probe.py')) != metadata['script']:
            raise PermissionError('Structural counts belong to a different role snapshot or program')
        text = zip_file.read('per_graph_counts.csv').decode('ascii')
        reader = csv.reader(text.splitlines())
        header = ['graph', 'nodes', 'edges', *(f'O{j}' for j in range(4)), *(f'C{j}' for j in range(4))]
        if next(reader) != header:
            raise ValueError('Unexpected R19 structural count layout')
        rows = {}
        for line in reader:
            gid, n, e, *counts = map(int, line)
            if (gid in rows or n <= 0 or e < 0 or len(counts) != 8
                    or any(c < 0 or c > o for o, c in zip(counts[:4], counts[4:], strict=True))):
                raise ValueError('Invalid path counts or graph ID')
            rows[gid] = (n, e, *counts)
    if set(rows) != set(range(4337)):
        raise ValueError('R19 structural precheck must cover all 4,337 Mutagenicity graphs')
    return rows


class Archive:
    def __init__(self, repo, here, cfg):
        self.repo, self.cfg = repo, cfg
        r9 = here.parent / R9
        geometry_cfg = json.loads((r9 / GEOMETRY_CONFIG).read_bytes())
        self.reference = reference_geometry.authenticate(r9, geometry_cfg)
        geom = r9 / 'results/reference-geometry-001'
        geom_obs = roc_transfer.stage(geom, 'OBS', cfg['geometry_obs_sha256'])
        if geom_obs['predecessors'] != {'source_obs_sha256': geometry_cfg['source_obs_sha256']}:
            raise PermissionError('R19 geometry predecessor differs')
        for name, value in (('roles', self.reference.roles), ('source-topology', self.reference.topology),
                            ('source-provenance', self.reference.provenance)):
            if json.loads(prior_run.read_member(geom, geom_obs['files'], f'{name}.json')) != value:
                raise PermissionError(f'Geometry source roles changed: {name}')
        setting = cfg['setting']
        self.roles = {key: row for key, row in self.reference.roles.items() if row['setting'] == setting}
        if (set(self.roles) != set(data.grid(dict(settings=[setting], domains=cfg['domains'], seed=cfg['seed'])))
                or sum(len(row['roles']['E']) for row in self.roles.values()) != cfg['evaluation_count']):
            raise PermissionError('R19 source-only directional scope differs')
        self.counts = structure_counts(here / 'results/structure-screen/structure-run.zip',
                                       cfg['structure_sha256'], geom_obs['files']['roles.json'])
        self.sources = data.Archive(repo, self.reference.plan, load_graphs=False)
        true_rate_replay.validate_e_roles(self.sources, self.reference)
        self.by_setting = {setting: self.sources.by_setting[setting]}

        self.static = r9 / 'results/feature-dropout-transfer-001'
        static_obs = roc_transfer.stage(self.static, 'OBS', cfg['static_obs_sha256'])
        static_cfg = json.loads((r9 / STATIC_CONFIG).read_bytes())
        pinned = dict(geometry_obs_sha256=cfg['geometry_obs_sha256'], zip_sha256=cfg['static_original_zip_sha256'],
                      freeze_sha256=cfg['static_original_manifest_sha256'])
        if static_obs['predecessors'] != pinned or dropout.parents(static_cfg) != pinned:
            raise PermissionError('Frozen static WL references differ')
        self.static_members = static_obs['files']
        self.static_roles = json.loads(self.static_member('roles.json'))
        self.static_fits = json.loads(self.static_member('fitted.json'))
        self.package = r9 / 'results/feature-dropout-screen/frozen-run.zip'
        manifest = self.package.with_name('freeze.json')
        if digest(self.package) != pinned['zip_sha256'] or digest(manifest) != pinned['freeze_sha256']:
            raise PermissionError('Original static WL prediction package changed')
        self.original = json.loads(manifest.read_bytes())['rows']

    def static_member(self, name):
        return prior_run.read_member(self.static, self.static_members, name)

    def graph_features(self):
        setting = self.cfg['setting']
        archive = self.by_setting[setting]
        path = archive._member('inputs/graphs.pt', 'INPUT')
        if digest(path) != archive.receipt['inputs/graphs.pt']:
            raise PermissionError('Mutagenicity graph container changed')
        archive.graphs = torch.load(path, mmap=True, weights_only=True, map_location='cpu')
        archive.graph_hashes = json.loads(archive._read('inputs/graph_hashes.json'))
        if len(archive.graphs) != archive.count or len(archive.graph_hashes) != archive.count:
            raise PermissionError('Mutagenicity graph inventory changed')
        output = {}
        for source in self.cfg['domains']:
            group = structural_observation.group_key(setting, source, self.cfg['seed'])
            old = self.sources.fit_access[group]
            s, v = old['source_ids'], old['validation_ids']
            if old['graph_container_sha256'] != archive.receipt['inputs/graphs.pt']:
                raise PermissionError('R10 FIT refers to a different graph container')
            graphs = data.source_graphs(self.repo, archive, source, s, v)
            expected = self.reference.topology[group]
            if [self.counts[i][1] for i in s+v] != expected['sizes']:
                raise PermissionError('Structural edge counts differ from the original R9 split')
            if any(self.counts[i][0] != len(graph['x']) for i, graph in graphs.items()):
                raise PermissionError('Structural node counts differ from the authenticated input')
            output[source] = {i: structural.features(Data(**graph), setting)[0] for i, graph in graphs.items()}
        return output

    def static_margins(self, key, features, previous):
        role = self.roles[key]['roles']
        lex = wl.vocabulary(features, role['D'])
        model = self.static_fits[key]
        registration = self.original[key]
        model_name = f'models/{key}.npz'
        if (self.static_roles[key]['roles'] != role or model['params']['Static'] != registration['params']['Static']
                or any(registration[k] != len(role[k]) for k in ('D', 'C', 'E'))
                or dropout.lex_hash(lex) != model['vocabulary_sha256']
                or model['initial_member_sha256'] != registration['prediction_sha']):
            raise PermissionError(f'Static WL provenance differs: {key}')
        model_blob = self.static_member(model_name)
        arrays = inputs.npz_member(model_blob)
        coeff = arrays['Static']
        intercepts = arrays['intercepts']
        if (coeff.shape != (len(lex),) or intercepts.shape != (len(dropout.ORIGINAL),)
                or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (coeff, intercepts))):
            raise ValueError(f'Malformed Static WL parameters: {key}')
        bias = intercepts[dropout.ORIGINAL.index('Static')]
        c, e = (wl.sparse_rows(features, role[k], lex) @ coeff + bias for k in ('C', 'E'))
        raw = previous.read(f'{key}.npz')
        if inputs.sha(raw) != registration['prediction_sha']:
            raise PermissionError(f'Original R9 E predictions changed: {key}')
        old = inputs.npz_member(raw)
        np.testing.assert_array_equal(old['e'], role['E'])
        np.testing.assert_allclose(old['Static'], e, atol=self.cfg['absolute_replay_tolerance'], rtol=0)
        np.testing.assert_array_equal(old['Static'] > 0, e > 0)
        return c, e, dict(static_C=model['params']['Static'], vocabulary_sha256=model['vocabulary_sha256'],
                          model_sha256=self.static_members[model_name], original_e_sha256=registration['prediction_sha'],
                          replay_max_abs=float(np.max(np.abs(old['Static']-e))), replay_actions_equal=True)
