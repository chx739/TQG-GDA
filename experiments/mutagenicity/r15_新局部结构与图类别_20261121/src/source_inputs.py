"""Authenticate R9 source-only graph IDs, roles, fitted weights and original B margins."""
import io
import json

import numpy as np
from scipy.special import expit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import inputs, roles, wl

from . import structure


R9_ATTEMPT = 'experiments/mutagenicity/r9_目标近邻与源标签传递_20261121/results/attempt-001'


class Original:
    def __init__(self, repo, plan):
        self.root = repo / R9_ATTEMPT
        raw = (self.root / 'OBS.json').read_bytes()
        self.obs_sha256 = inputs.sha(raw)
        if self.obs_sha256 != plan['r9_observation_sha256']:
            raise PermissionError('R9 OBS identity differs')
        obs = json.loads(raw)
        if obs['stage'] != 'R9_OBS':
            raise ValueError('Expected R9 OBS')
        self.files = obs['files']
        self.executed = self.read_json('executed-config.json')
        self.plan = self.executed['plan']
        if (obs['r8_parent_sha256'] != self.plan['r8_observation_sha256']
                or self.plan['source_role_key'] != plan['source_role']
                or self.plan['allowed_graph_count'] != plan['allowed_graph_count']
                or self.plan['directions'] != plan['directions']
                or self.plan['wl_depths'] != [0, 1, 2]
                or self.plan['source_model']['C'] != 1.0):
            raise PermissionError('R9 registered scope or B model differs')
        self.roles = self.read_json('roles.json')
        self.evidence = inputs.SourceEvidence(repo, self.plan)
        self.provenance = self.read_json('source-provenance.json')
        if self.provenance != {**self.evidence.provenance,
                               'graphs_sha256': self.evidence.files['inputs/graphs.pt']}:
            raise PermissionError('R9 certified source provenance differs')

    def read(self, name):
        blob = (self.root / name).read_bytes()
        if inputs.sha(blob) != self.files[name]:
            raise PermissionError(f'R9 sealed member differs: {name}')
        return blob

    def read_json(self, name):
        return json.loads(self.read(name))

    def records(self):
        registered = self.read_json('fingerprints.json')
        graphs = self.evidence.graphs()
        if set(registered) != {str(graph_id) for graph_id in graphs}:
            raise PermissionError('R9 fingerprint scope differs')
        records = {i: structure.encode(graph, self.plan) for i, graph in graphs.items()}
        counts = {i: rec.counts for i, rec in records.items()}
        edges = {i: rec.edges for i, rec in records.items()}
        keys = {i: wl.count_key(row) for i, row in counts.items()}
        for i in graphs:
            actual = dict(undirected_edges=edges[i], group_sha256=inputs.sha(keys[i]),
                          raw_layer_counts=[{color.hex(): n for color, n in layer.items()}
                                            for layer in counts[i]])
            if actual != registered[str(i)]:
                raise ValueError(f'R9 typed WL fingerprint or edge count differs for graph {i}')
        reconstructed = roles.build(edges, counts, self.plan)
        reconstructed['pseudo_domains'] = {str(k): v for k, v in reconstructed['pseudo_domains'].items()}
        if reconstructed != self.roles:
            raise ValueError('Rebuilt R9 source roles differ')
        return records, {i: wl.normalize(row) for i, row in counts.items()}

    def model(self, name, direction, rows):
        model = self.read_json(f'directions/{name}-model.json')
        checks = self.read_json(f'directions/{name}-checks.json')
        d_ids, e_ids = direction['source_ids'], direction['holdout_ids']
        if (checks['source_label_scope'] != 'source_fit_and_source_endpoint_voting_only'
                or checks['source_label_ids'] != d_ids
                or checks['requested_source_label_ids'] != d_ids
                or model['n_labels'] != len(d_ids)
                or not 0 <= model['gradient_inf'] <= self.plan['source_model']['stationarity_tolerance']):
            raise PermissionError(f'R9 fitted model label or fit lineage differs for {name}')
        vocab = wl.vocabulary(rows, d_ids)
        registered = [[depth, color.hex()] for depth, color in vocab]
        weights = np.asarray(model['weights'], dtype=np.float64)
        if registered != model['vocabulary'] or weights.shape != (len(vocab),):
            raise ValueError(f'R9 vocabulary differs for {name}')
        if not np.isfinite(weights).all() or not np.isfinite(model['intercept']):
            raise ValueError(f'R9 nonfinite parameters for {name}')
        margin = scores(wl.sparse_rows(rows, e_ids, vocab), model)
        comparison = self.verify_b(name, d_ids, e_ids, margin)
        coefficients = {key: float(weights[column]) for key, column in vocab.items()}
        return margin, coefficients, comparison

    def verify_b(self, name, d_ids, e_ids, margin):
        parity = {}
        for seed in self.plan['adaptation_seeds']:
            path = f'predictions/{name}-seed-{seed}.npz'
            with np.load(io.BytesIO(self.read(path)), allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['s_ids'], d_ids)
                np.testing.assert_array_equal(archive['h_ids'], e_ids)
                logp = archive['logp_B']
                old_margin = logp[:, 1] - logp[:, 0]
                np.testing.assert_allclose(margin, old_margin, rtol=0, atol=1e-12)
                np.testing.assert_allclose(expit(margin), archive['probability_B'], rtol=0, atol=1e-14)
                np.testing.assert_array_equal(margin > 0., archive['action_B'])
                parity[str(seed)] = dict(archive_sha256=self.files[path],
                                         max_abs_margin_error=float(np.abs(margin-old_margin).max()),
                                         class_one_margin_positive=True)
        return parity


def direction_name(direction):
    return f"ps{direction['source']}_to_{direction['target']}"


def direction_roles(direction):
    d_ids, e_ids = direction['source_ids'], direction['holdout_ids']
    groups = direction['groups']
    u_groups = [group for group in groups if group['pool'] == 'U']
    u_ids = sorted(i for group in u_groups for i in group['graph_ids'])
    representative_ids = sorted(min(group['graph_ids']) for group in u_groups)
    if (set(u_ids) & set(e_ids) or set(d_ids) & (set(u_ids) | set(e_ids))
            or len({group['group_sha256'] for group in groups}) != len(groups)
            or sorted(i for group in groups if group['pool'] == 'H' for i in group['graph_ids']) != e_ids):
        raise ValueError('D/U/E membership or normalized fingerprint separation differs')
    return dict(D=d_ids, U=u_ids, U_representatives=representative_ids, E=e_ids)
