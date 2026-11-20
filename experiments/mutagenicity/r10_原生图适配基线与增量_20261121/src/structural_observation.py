"""Bind historical cohorts, seal source fits, then predict on exact H IDs."""
import json

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.inputs import sha
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import observation as r13_observation
from . import structural


R10_ARMS = ('GRADE', 'SourceOnly', 'SourceFusion')
R13_ARMS = ('A_node', 'A_graph', 'S_node', 'S_graph')
REQUIRED_ARMS = ('B0', 'SourceCal', *structural.METHODS)
OPTIONAL = (*R10_ARMS, *(f'R13_{name}' for name in R13_ARMS))


def open_c(plan):
    return historical.CArchive(dict(plan, phase1_seed=plan['seeds'][0], reserve_seeds=plan['seeds'][1:]))


def group_key(setting, source, seed):
    return f'{setting}-M{source}-seed-{seed}'


def tasks(plan):
    return {inputs.key(setting, source, target, seed):
            dict(setting=setting, source=source, target=target, seed=seed)
            for setting, source, seed, targets in inputs.registered(plan) for target in targets}


def fit_members(plan):
    groups = [f'fits/{group_key(setting, source, seed)}.json'
              for setting, source, seed, _ in inputs.registered(plan)]
    return ['executed-config.json', 'historical-plan.json', 'fit-roles.json', 'fit-access.json', *groups]


def observation_members(plan):
    return ['FIT.json', 'roles.json', 'observation-access.json', *prior_run.inventory(tasks(plan), 'predictions')]


def preflight(repo, plan):
    """Authenticate all role manifests and registries without reading graph tensors or labels."""
    c = open_c(plan)
    arms = historical.ReferenceArms(plan, repo)
    r13 = PublishedR13(plan, repo)
    r10_blocks, r11_blocks, r13_blocks = 0, 0, 0
    for setting, source, seed, targets in inputs.registered(plan):
        pools = []
        for target in targets:
            task = dict(setting=setting, source=source, target=target, seed=seed)
            key = inputs.key(**task)
            manifest, _, _ = historical.context(c, task, repo)
            pools.append(manifest['ids'])
            r10_blocks += f'observation/{key}.npz' in arms.obs10
            r11_blocks += f'unlabeled/{key}.npz' in arms.obs11
            r13_blocks += key in r13.roles
        structural.check_roles(pools)
    if r11_blocks != len(tasks(plan)):
        raise PermissionError('SourceCal does not cover the entire registered grid')
    return dict(groups=len(fit_members(plan))-4, blocks=len(tasks(plan)),
                sourcecal=r11_blocks, historical_r10=r10_blocks, r13=r13_blocks)


def source_labels(context, manifest, source, seed, role):
    if role not in ('fit', 'validation'):
        raise PermissionError('Only source fit and validation labels may be used here')
    pool = 'source' if role == 'fit' else 'validation'
    relative = f'source_labels/M{source}/seed{seed}_{role}.pt'
    payload = context['read'](relative, 'INPUT')
    counts = context['roles']['class_counts'][role]
    if pool == 'source' and sha(payload) != manifest['provenance']['r5_files'][relative]:
        raise PermissionError('S labels differ from the original C input')
    return inputs.ValidationPort(payload, len(context['ids'][pool]), tuple(counts)).load(), sha(payload)


def fit(output, repo, plan):
    archive = open_c(plan)
    (output / 'fits').mkdir()
    stores, roles, access = {}, {}, {}
    for setting, source, seed, targets in inputs.registered(plan):
        manifests, contexts = [], []
        for target in targets:
            task = dict(setting=setting, source=source, target=target, seed=seed)
            manifest, checksum, context = historical.context(archive, task, repo)
            key = inputs.key(**task)
            manifests.append(manifest)
            contexts.append(context)
            roles[key] = dict(**task, manifest_sha256=checksum, holdout_ids=manifest['ids']['holdout'])
        manifest, context = manifests[0], contexts[0]
        structural.check_roles([m['ids'] for m in manifests])
        if any(data['root'] != context['root'] or data['label_meta']['mapping'] != context['label_meta']['mapping']
               for data in contexts):
            raise PermissionError('Source archive or binary label semantics differ across one group')
        if setting not in stores:
            stores[setting] = historical.GraphStore(context)
        store = stores[setting]
        ids = context['ids']
        y_s, s_sha = source_labels(context, manifest, source, seed, 'fit')
        y_v, v_sha = source_labels(context, manifest, source, seed, 'validation')
        s, v = ([structural.features(graph, setting) for graph in
                 structural.take_role(store, ids[pool], ids, pool, fitting=True)]
                for pool in ('source', 'validation'))
        fitted = structural.fit_models(s, v, y_s, y_v, plan)
        group = group_key(setting, source, seed)
        write_json(output / f'fits/{group}.json',
                   dict(setting=setting, source=source, seed=seed, **fitted))
        access[group] = dict(source_ids=ids['source'], validation_ids=ids['validation'],
                             graph_container_sha256=store.graph_sha,
                             s_label_sha256=s_sha, v_label_sha256=v_sha, class_mapping=context['label_meta']['mapping'])
        print(f'R10 source fit {group}, S={len(s)}, V={len(v)}, '
              f"WL C={fitted['wl']['selected_C']}, size C={fitted['size']['selected_C']}", flush=True)
    write_json(output / 'historical-plan.json', archive.plan)
    write_json(output / 'fit-roles.json', roles)
    write_json(output / 'fit-access.json', access)
    return prior_run.seal(output, 'FIT', fit_members(plan), {'r7_final_digest': plan['r7_final_digest']})


def verify_fit(output, plan, checksum):
    frozen = prior_run.verified_members(output, 'FIT', checksum, fit_members(plan),
                                        {'r7_final_digest': plan['r7_final_digest']})
    if json.loads(frozen['executed-config.json'])['plan'] != plan:
        raise PermissionError('Sealed fit used a different structural plan')
    roles = json.loads(frozen['fit-roles.json'])
    if set(roles) != set(tasks(plan)):
        raise PermissionError('Sealed source fits refer to another task inventory')
    access = json.loads(frozen['fit-access.json'])
    groups = {group_key(s, source, seed) for s, source, seed, _ in inputs.registered(plan)}
    if set(access) != groups:
        raise PermissionError('Sealed source fit access does not cover the 60 model groups')
    return frozen, roles, access


def bind(archive, repo, task, role, access):
    manifest, checksum, context = historical.context(archive, task, repo)
    if (any(role[k] != v for k, v in task.items()) or role['manifest_sha256'] != checksum
            or role['holdout_ids'] != manifest['ids']['holdout']
            or access['source_ids'] != manifest['ids']['source']
            or access['validation_ids'] != manifest['ids']['validation']
            or access['class_mapping'] != context['label_meta']['mapping']):
        raise PermissionError('Sealed H and the reused S/V model are bound to different historical roles')
    return manifest, context


class PublishedR13:
    def __init__(self, plan, repo):
        controls = plan['reference_controls']
        self.root = repo / controls['r13_attempt']
        payload = (self.root / 'OBS.json').read_bytes()
        seal = json.loads(payload)
        if (sha(payload) != controls['r13_obs_sha256'] or seal['stage'] != 'OBS'
                or seal['predecessors'] != {'r7_final_digest': plan['r7_final_digest']}):
            raise PermissionError('R13 observation is not the published matched pilot')
        self.members = seal['files']
        self.roles = json.loads(prior_run.read_member(self.root, self.members, 'roles.json'))
        expected = {key for key, task in tasks(plan).items() if task['seed'] == 3405}
        if set(self.roles) != expected:
            raise PermissionError('R13 pilot does not cover exactly the common seed')

    def get(self, key, role, baseline):
        if key not in self.roles:
            return {}, {}
        ref = self.roles[key]
        if any(ref[k] != role[k] for k in (*structural.IDENTITY, 'manifest_sha256', 'held_ids')):
            raise PermissionError('Published R13 predictions have a different H task or role')
        path = f'predictions/{key}.npz'
        payload = inputs.npz_member(prior_run.read_member(self.root, self.members, path))
        r13_observation.registered_predictions(payload, ref, True)
        np.testing.assert_array_equal(payload['logits_B0'], baseline)
        return ({f'R13_{arm}': payload[f'logits_{arm}'] for arm in R13_ARMS},
                {f'R13_{arm}': self.members[path] for arm in R13_ARMS})


def freeze(output, repo, plan, fit_sha):
    frozen, fitted_roles, access = verify_fit(output, plan, fit_sha)
    archive = open_c(plan)
    if json.loads(frozen['historical-plan.json']) != archive.plan:
        raise PermissionError('Historical C plan has changed after source fitting')
    controls = historical.ReferenceArms(plan, repo)
    r13 = PublishedR13(plan, repo)
    (output / 'predictions').mkdir()
    stores, features, roles = {}, {}, {}
    for setting, source, seed, targets in inputs.registered(plan):
        group = group_key(setting, source, seed)
        blob = frozen[f'fits/{group}.json']
        model = json.loads(blob)
        if tuple(model[k] for k in structural.GROUP) != (setting, source, seed):
            raise PermissionError('Sealed model group identity differs')
        for target in targets:
            task = dict(setting=setting, source=source, target=target, seed=seed)
            key = inputs.key(**task)
            role = fitted_roles[key]
            manifest, context = bind(archive, repo, task, role, access[group])
            if setting not in stores:
                stores[setting] = historical.GraphStore(context)
            store = stores[setting]
            if store.graph_sha != access[group]['graph_container_sha256']:
                raise PermissionError('Prediction graph container differs from the sealed fit')
            h_ids = role['holdout_ids']
            cohort = (setting, target)
            if cohort not in features:
                graphs = structural.take_role(store, h_ids, manifest['ids'], 'holdout', fitting=False)
                features[cohort] = (h_ids, [structural.features(graph, setting) for graph in graphs])
            expected, held = features[cohort]
            if h_ids != expected:
                raise PermissionError('Shared H features differ from the exact original ID order')
            logits = {name: np.column_stack((np.zeros(len(h_ids)), margins))
                      for name, margins in structural.predict(model, held).items()}
            b0 = archive.b0(task, h_ids)
            references, lineage = controls.get(task, h_ids, b0)
            extra, extra_lineage = r13.get(key, dict(role, held_ids=h_ids), b0)
            logits.update(B0=b0, **references, **extra)
            lineage.update(extra_lineage)
            arrays = dict(ids=np.asarray(h_ids))
            for name, z in logits.items():
                arrays[f'logits_{name}'] = z
                arrays[f'action_{name}'] = structural.decisions(z[:, 1] - z[:, 0])
            prior_run.save_predictions(output, f'predictions/{key}.npz', arrays)
            roles[key] = dict(role, group=group, fit_sha256=sha(blob), reference_sha256=lineage,
                              b0_sha256=archive.digests[f'blocks/{key}/holdout_input.npz'])
    write_json(output / 'roles.json', roles)
    write_json(output / 'observation-access.json', archive.digests)
    return prior_run.seal(output, 'OBS', observation_members(plan), {'fit_sha256': fit_sha})


def verify_predictions(arrays, role):
    n = len(role['holdout_ids'])
    np.testing.assert_array_equal(arrays['ids'], role['holdout_ids'])
    references = set(role['reference_sha256'])
    if 'SourceCal' not in references or not references <= {'SourceCal', *OPTIONAL}:
        raise PermissionError('Sealed historical arms differ from the registered controls')
    methods = (*REQUIRED_ARMS, *(method for method in OPTIONAL if method in references))
    if set(arrays) != {'ids', *(f'{stem}_{method}' for method in methods for stem in ('action', 'logits'))}:
        raise PermissionError('Missing or unregistered frozen predictions')
    for method in methods:
        z, y = arrays[f'logits_{method}'], arrays[f'action_{method}']
        if z.shape != (n, 2) or y.shape != (n,) or not np.isfinite(z).all() or y.dtype != np.int64:
            raise ValueError(f'Invalid frozen predictions for {method}')
        np.testing.assert_array_equal(y, structural.decisions(z[:, 1] - z[:, 0]))
    return methods
