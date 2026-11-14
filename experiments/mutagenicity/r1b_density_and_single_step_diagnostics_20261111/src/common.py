"""Immutable R1 dependencies and scoped label access for the R1b follow-up."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.data import canonical_sha


def _frozen_membership(root, stage, files):
    directories = ('deps','protocol') if stage == 'INPUT' else (
        ('deps','warm','scales','steps','scores','protocol','training-labels') if stage == 'OBS' else ())
    for name in directories:
        actual = {str(p.relative_to(root)) for p in (root / name).rglob('*') if p.is_file()}
        expected = {rel for rel in files if rel.startswith(name+'/')}
        if actual != expected:
            raise PermissionError(f'Frozen directory membership changed: {name}')


def verify_seal(root, stage, token=None):
    root = Path(root)
    record = json.loads((root / 'seals' / f'{stage}.json').read_text())
    content = {k: record[k] for k in ('stage', 'files')}
    if record['stage'] != stage or canonical_sha(content) != record['digest']:
        raise PermissionError('Invalid seal digest')
    if token is not None and token != record['digest']:
        raise PermissionError('Incorrect seal token')
    if not record['files']:
        raise PermissionError('Empty seal')
    _frozen_membership(root,stage,record['files'])
    for name, expected in record['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise PermissionError(f'Sealed evidence changed: {name}')
    return record


def validate_roles(root, plan):
    root = Path(root)
    roles = json.loads((root / 'deps/roles.json').read_text())
    domains = roles['domains']
    joined = [i for d in domains.values() for i in d]
    if sorted(joined) != list(range(4337)):
        raise ValueError('Domain IDs must partition all 4337 graphs')
    source = roles['source']['domain_ids']
    if sorted(source) != sorted(domains['0']) or len(source) != 1084:
        raise ValueError('Source domain mismatch')
    for domain in plan['domains']:
        t = roles['targets'][str(domain)]
        if len(t['A']) != 271 or len(t['B']) != 271 or len(t['final']) != 542 + (domain == 3):
            raise ValueError('Target role counts mismatch')
        if sorted(t['A'] + t['B'] + t['final']) != sorted(t['ids']) or sorted(t['ids']) != sorted(domains[str(domain)]):
            raise ValueError('Target overlap or missing IDs')
        if len(set(t['query_ids'])) != 64 or not set(t['query_ids']).issubset(t['A']):
            raise ValueError('Frozen query IDs mismatch')
    for seed in plan['seeds']:
        s = json.loads((root / 'deps/roles' / f'seed{seed}.json').read_text())
        groups = [s[k + '_ids'] for k in ('fit', 'validation', 'diagnostic')]
        if list(map(len, groups)) != [779, 109, 196] or sorted(sum(groups, [])) != sorted(source):
            raise ValueError('Source role overlap/count/coverage mismatch')
        for domain in plan['domains']:
            for fold in range(4):
                f = s['folds']['targets'][str(domain)][str(fold)]
                if len(f['train_ids']) != 118 or len(f['validation_ids']) != 29 or len(set(f['budget_ids'])) != 147:
                    raise ValueError('Probe inner split counts mismatch')
                if sorted(f['train_ids'] + f['validation_ids']) != sorted(f['budget_ids']) or not set(f['budget_ids']).issubset(roles['targets'][str(domain)]['A']):
                    raise ValueError('Probe split leakage/overlap')
    return roles


def prepare_dependencies(root, plan):
    """Copy selected sealed evidence, never old evaluation labels or full y."""
    root = Path(root)
    old = Path(plan['dependency']['root'])
    old_seal = json.loads((old / 'seals/S2.json').read_text())
    if old_seal['digest'] != plan['dependency']['s2_digest'] or canonical_sha({k: old_seal[k] for k in ('stage', 'files')}) != old_seal['digest']:
        raise ValueError('Historical S2 identity mismatch')
    archive = Path(plan['dependency']['archive'])
    if digest(archive) != plan['dependency']['archive_sha256']:
        raise ValueError('Historical full archive SHA256 mismatch')
    selected = {'roles.json', 'inputs/broker.json', 'inputs/class_mapping.json', 'inputs/size.pt', 'inputs/graphs.json'}
    for seed in plan['seeds']:
        selected.add(f'roles/seed{seed}.json')
        selected.update(f'source/seed{seed}/{name}' for name in ('frozen.pt', 'predictions.pt', 'source.pt', 'history.json', 'checks.json'))
        for domain in plan['domains']:
            geom = old / f'geometry/seed{seed}_domain{domain}'
            selected.update(str(p.relative_to(old)) for p in geom.rglob('*') if p.is_file())
            for fold in range(4):
                for head in ('linear', 'mlp'):
                    base = Path(f'probes/target{domain}/seed{seed}/fold{fold}/H/{head}/true')
                    meta = json.loads((old / base / 'metadata.json').read_text())
                    names = {'metadata.json', 'initial.pt', 'history.json', 'probabilities.pt', *meta['checkpoints'].values()}
                    selected.update(str(base / name) for name in names)
    copies = {}
    for rel in sorted(selected):
        if rel not in old_seal['files'] or digest(old / rel) != old_seal['files'][rel]:
            raise ValueError(f'Historical dependency not sealed or changed: {rel}')
        dst = root / 'deps' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise FileExistsError(dst)
        shutil.copyfile(old / rel, dst)
        if digest(dst) != old_seal['files'][rel]:
            raise ValueError('Dependency copy mismatch')
        copies[rel] = dict(bytes=dst.stat().st_size, sha256=old_seal['files'][rel])
    write_json(root / 'dependency-manifest.json', dict(original_root=str(old), original_s2=old_seal['digest'],
        original_archive=str(archive), original_archive_sha256=plan['dependency']['archive_sha256'], files=copies))
    roles = validate_roles(root, plan)
    write_json(root / 'role-verification.json', dict(source_counts=[779,109,196], target_A=271, target_B=271,
        target_final=[542,543], frozen_queries=64, partition_disjoint_exhaustive=True,
        roles_sha256=digest(root / 'deps/roles.json'), final_values_parsed=False))
    return roles


class NextLabelBroker:
    """An audited API boundary; values are parsed only at requested byte offsets."""
    def __init__(self, root):
        self.root = Path(root)
        self.roles = json.loads((self.root / 'deps/roles.json').read_text())
        self.info = json.loads((self.root / 'deps/inputs/broker.json').read_text())
        self.mapping = json.loads((self.root / 'deps/inputs/class_mapping.json').read_text())['raw_to_class']
        if digest(self.info['path']) != self.info['sha256']:
            raise ValueError('Raw label bytes changed')
        self._verified = {}

    def _check(self, stage):
        # Revalidate on any change of sealed file metadata, not just the seal token.
        record = json.loads((self.root / 'seals' / f'{stage}.json').read_text())
        if record.get('stage') != stage or canonical_sha({k:record[k] for k in ('stage','files')}) != record.get('digest'):
            raise PermissionError('Invalid seal identity')
        _frozen_membership(self.root,stage,record['files'])
        signature = [(rel, (self.root / rel).stat().st_size, (self.root / rel).stat().st_mtime_ns,
                      (self.root / rel).stat().st_ctime_ns) for rel in record['files']]
        if self._verified.get(stage) != (record['digest'], signature):
            verify_seal(self.root, stage)
            self._verified[stage] = (record['digest'], signature)
        return record['digest']

    def read(self, ids, role, purpose, *, seed=None, domain=None):
        ids = list(map(int, ids))
        event = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), role=role,
                     ids=ids, purpose=purpose, seed=seed, domain=domain, permitted=False)
        try:
            if not purpose or len(ids) != len(set(ids)):
                raise PermissionError('Distinct IDs and purpose required')
            final = set().union(*(set(t['final']) for t in self.roles['targets'].values()))
            if role == 'final' or set(ids) & final:
                raise PermissionError('Final class values remain closed')
            if role in ('target_A', 'target_B'):
                allowed = self.roles['targets'][str(domain)][role[-1]]
                stage = 'INPUT' if role == 'target_A' else 'OBS'
            elif role in ('source_fit', 'source_validation', 'source_diagnostic'):
                s = json.loads((self.root / 'deps/roles' / f'seed{seed}.json').read_text())
                allowed = s[role.removeprefix('source_') + '_ids']
                stage = 'INPUT' if role == 'source_fit' else 'OBS'
            else:
                raise PermissionError('Unknown role')
            if not set(ids).issubset(allowed):
                raise PermissionError('IDs outside declared role')
            event['token'] = self._check(stage)
            values = []
            with open(self.info['path'], 'rb', buffering=0) as stream:
                # Authenticate the same opened bytes on every access, including
                # after lengthy training. Hashing bytes does not parse classes.
                if hashlib.sha256(stream.read()).hexdigest() != self.info['sha256']:
                    raise ValueError('Raw label bytes changed since registration')
                for i in ids:
                    offset, length = self.info['offsets'][i]
                    stream.seek(offset)
                    values.append(self.mapping[str(int(stream.read(length).strip()))])
            event['permitted'] = True
            return dict(zip(ids, values))
        except Exception as exc:
            event['reason'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            with (self.root / 'logs/label-access.jsonl').open('a') as stream:
                stream.write(json.dumps(event, separators=(',', ':')) + '\n')
