"""Immutable contracts, source-only data access, seeds and serial compute ledger."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
import yaml

from tqg_gda.data import GraphDomainTask
from tqg_gda.training import load_model, environment
from tqg_gda.utils import sha256, write_json

CODE = Path(__file__).resolve().parent
STAGE = CODE.parent
RUN = STAGE.parent
DESIGN = RUN / 'stage2_revision_v1'
OLD = RUN / 'stage3_execution'


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def generator(base, seed, domain, block_or_step, role):
    value = int(np.random.SeedSequence([base, seed, domain, block_or_step, role])
                .generate_state(1, dtype=np.uint64)[0]) % (2**63 - 1)
    return torch.Generator().manual_seed(value)


def git_identity():
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True)
    if dirty:
        raise RuntimeError('Measured execution requires a clean committed tree: ' + dirty)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()


def hardware():
    cpu = platform.processor()
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):
            cpu = line.split(':', 1)[1].strip()
            break
    return dict(cpu=cpu, threads=1, device='cpu', platform=platform.platform(),
                logical_cpus=os.cpu_count(), **environment())


class Budget:
    """Sum measured serial job elapsed seconds; model-editing time is not compute."""
    def __init__(self, phase):
        self.path = STAGE / 'compute_ledger.json'
        self.phase = phase
        self.started = time.perf_counter()
        self.record = read(self.path)
        self.previous = self.record['consumed_seconds']
        self.warned = set()

    def check(self):
        consumed = self.previous + time.perf_counter() - self.started
        cap = self.record['cap_seconds']
        for fraction in (.5, .8):
            if consumed >= fraction * cap and fraction not in self.warned:
                print(json.dumps(dict(event='budget_fraction', fraction=fraction,
                                      consumed_seconds=consumed)), flush=True)
                self.warned.add(fraction)
        if consumed >= .8 * cap:
            raise TimeoutError('80% resource review threshold reached; preserve current artifacts')

    def finish(self, status='completed'):
        elapsed = time.perf_counter() - self.started
        current = read(self.path)
        if current['consumed_seconds'] != self.previous:
            raise RuntimeError('Concurrent measured jobs would double-count serial budget')
        current['consumed_seconds'] += elapsed
        current['records'].append(dict(phase=self.phase, seconds=elapsed, status=status,
                                      finished_unix=time.time()))
        atomic_json(self.path, current)
        return elapsed


class Context:
    def __init__(self):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        torch.use_deterministic_algorithms(True)
        self.plan = yaml.safe_load((DESIGN / 'experiment_plan.yaml').read_text())
        self.lock = read(STAGE / 'execution_lock.json')
        for key, item in self.lock['inputs'].items():
            if sha256(item['path']) != item['sha256']:
                raise ValueError('Locked input changed: ' + key)
        self.commit = git_identity()
        self.roles = read(RUN / 'stage2_method/data_roles/manifest.json')
        self.old_manifest = read(OLD / 'training_manifest.json')
        self.arms = read(DESIGN / 'resolved_arms_preview.json')['arms']
        self._tasks = {}

    def task(self, domain, all_source=False):
        key = domain, all_source
        if key not in self._tasks:
            role = self.roles['domains'][str(domain)]
            task = GraphDomainTask(self.plan['datasets'][0]['cache'],
                '/home/hxchen/TQG_data/PROTEINS_official_20261023/source_protocol_official.json',
                target_index=role['index_path'], target_policy='transductive-unlabeled',
                fit_fraction=.8, split_seed=431, use_node_attributes=False,
                preprocess_pool='all_source' if all_source else 'fit')
            if [len(task.fit_ids), len(task.query_ids), len(task.validation_ids)] != [199, 51, 28]:
                raise ValueError('Source roles differ from plan')
            self._tasks[key] = task
        return self._tasks[key]

    def development_ids(self, domain):
        role = self.roles['domains'][str(domain)]
        for name in ('development', 'final'):
            if sha256(role[name]['path']) != role[name]['sha256']:
                raise ValueError('Role index changed')
        dev = tuple(map(int, np.load(role['development']['path'], allow_pickle=False)))
        final = set(map(int, np.load(role['final']['path'], allow_pickle=False)))
        if set(dev) & final:
            raise PermissionError('Development/final overlap')
        return dev

    def source(self, domain, seed, all_source=False):
        arm = 'full_source_CE_700plus200' if all_source else 'original_TQG_target_probes'
        record = next(r for r in self.old_manifest['records']
                      if r['task'] == f'0_to_{domain}' and r['seed'] == seed and r['arm'] == arm)
        path = Path(record['warmup_checkpoint'])
        expected = record.get('warmup_checkpoint_sha256')
        if expected and sha256(path) != expected:
            raise ValueError('Source checkpoint changed')
        model, state = load_model(path)
        task = self.task(domain, all_source)
        old_identity, new_identity = copy.deepcopy(state['identity']), copy.deepcopy(task.identity)
        for value in (old_identity, new_identity):
            for key in ('target_ids', 'target_index_sha256', 'target_policy', 'fingerprint'):
                value.pop(key, None)
        if old_identity != new_identity:
            raise ValueError('Source checkpoint roles/preprocessing differ')
        return model, state, path

    def final_state(self, domain, seed):
        record = next(r for r in self.old_manifest['records']
                      if r['task'] == f'0_to_{domain}' and r['seed'] == seed
                      and r['arm'] == 'original_TQG_target_probes')
        path = Path(record['final_checkpoint'])
        if record.get('final_checkpoint_sha256') and sha256(path) != record['final_checkpoint_sha256']:
            raise ValueError('Historical checkpoint changed')
        model, state = load_model(path)
        if 'optimizer' not in state:
            raise ValueError('Historical optimizer unavailable; no fresh substitute permitted')
        return model, state, path

    def block_ids(self, domain, seed, block):
        task = self.task(domain)
        rng = generator(910000, seed, domain, block, 1)
        order = torch.randperm(51, generator=rng).tolist()
        source = tuple(task.query_ids[i] for i in order)
        dev = self.development_ids(domain)
        order = torch.randperm(len(dev), generator=rng).tolist()
        target = tuple(dev[i] for i in order)
        return dict(bank=source[:35], audit=source[35:], query=target[:16],
                    oracle_extra=target[16:32], held=target[32:64])

    def arm(self, name):
        value = copy.deepcopy(self.arms[name])
        value['arm'] = name
        return value
