"""Experiment configuration, identities and reproducible CPU/GPU state."""
import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError('Cyclic configuration inheritance')
    value = json.loads(path.read_text())
    parent = value.pop('extends', None)
    return merge(load_config(path.parent / parent, (*seen, path)), value) if parent else value


def validate_config(config):
    if config['method'] not in ('tqg', 'source_only'):
        raise ValueError('method must be tqg or source_only')
    if config['dtype'] not in ('float32', 'float64'):
        raise ValueError('Finite differences require float32 or float64')
    for key in ('warmup', 'adaptation'):
        c = config[key]
        if c['source_pool'] not in ('fit', 'all_source'):
            raise ValueError('source_pool must be fit or all_source')
        if not math.isfinite(c['lr']) or c['lr'] <= 0:
            raise ValueError('Learning rate must be finite and positive')
    if config['method'] == 'tqg' and any(config[k]['source_pool'] != 'fit' for k in ('warmup', 'adaptation')):
        raise ValueError('Standard TQG reserves query graphs from warmup and source CE')
    counts = (config['num_threads'], config['warmup']['epochs'], config['warmup']['batch_size'],
              config['adaptation']['steps'], config['adaptation']['fit_batch_size'],
              config['adaptation']['query_batch_size'], config['adaptation']['target_batch_size'],
              config['adaptation']['validation_interval'], config['adaptation']['checkpoint_interval'])
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError('Epoch, step, thread and batch budgets must be positive integers')
    if not 0 < config['split']['fit_fraction'] < 1:
        raise ValueError('fit_fraction must be between zero and one')
    if not math.isfinite(config['adaptation']['feedback_weight']) or config['adaptation']['feedback_weight'] < 0:
        raise ValueError('feedback_weight must be finite and nonnegative')
    mode = config['adaptation'].get('query_label_mode', 'true')
    if mode not in ('true', 'permute_within_batch'):
        raise ValueError('Unknown query label control')
    if mode != 'true':
        seed = config['adaptation'].get('query_permutation_seed')
        if config['method'] != 'tqg' or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('Permuted query labels require TQG and a nonnegative permutation seed')
    predictor = config['adaptation'].get('feedback_predictor', 'responses')
    if predictor not in ('responses', 'mean_residual', 'responses_with_intercept'):
        raise ValueError('Unknown feedback predictor')
    if predictor != 'responses' and config['method'] != 'tqg':
        raise ValueError('Residual-predictor controls require the TQG data roles')
    if 'update_budget' in config['warmup']:
        for key in ('update_budget', 'validation_interval_updates'):
            value = config['warmup'].get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError('Update-budget warmup requires positive update and validation budgets')
    return config


def seed_everything(seed, threads, deterministic=True):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
