"""Warmup and one-update-per-query adaptation; source-only checkpoint selection."""
import copy
from dataclasses import asdict
import json
import platform
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import torch_geometric

from .feedback import ProbeConfig, build_feedback, build_mean_feedback, feedback_loss
from .feedback_v2 import V2Config, build_gradient_feedback, parameter_feedback_loss
from .metrics import evaluate_source
from .models import GraphClassifier
from .utils import (atomic_save, cpu_state, restore_rng, rng_state, seed_everything,
                    validate_config, write_json, sha256)


def model_spec(task, config):
    return dict(input_dim=task.input_dim, edge_dim=task.edge_dim, **config['model'])


def load_model(checkpoint, device='cpu'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state.get('kind') not in ('tqg_source', 'tqg_adaptation', 'tqg_predictor') or state.get('schema_version') != 1:
        raise ValueError('Not a TQG checkpoint. Standard TQG needs a source model trained only on fit graphs')
    model = GraphClassifier(state['encoder_spec'], len(state['class_labels']))
    model.to(device=device, dtype=getattr(torch, state['config']['dtype']))
    model.load_state_dict(state['model'])
    model.eval()
    return model, state


def environment():
    root = Path(__file__).resolve().parents[1]
    return dict(python=platform.python_version(), torch=torch.__version__, pyg=torch_geometric.__version__,
                cuda_available=torch.cuda.is_available(),
                code_sha256={p.name: sha256(p) for p in sorted((root/'tqg_gda').glob('*.py'))})


def _common(model, task, config):
    return dict(schema_version=1, encoder_spec=model.encoder.spec, class_labels=task.classes,
                config=copy.deepcopy(config), identity=copy.deepcopy(task.identity),
                preprocessing=copy.deepcopy(task.preprocessing))


def run_warmup(task, config, output, device='cpu', resume=None):
    validate_config(config)
    if 'update_budget' in config['warmup']:
        if resume is not None:
            raise ValueError('Update-budget warmup does not use epoch-based resume; use a new job directory')
        return run_warmup_updates(task, config, output, device)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    seed_everything(config['seed'], config['num_threads'], config['deterministic'])
    dtype = getattr(torch, config['dtype'])
    model = GraphClassifier(model_spec(task, config), len(task.classes)).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['warmup']['lr'], weight_decay=0.)
    generator = torch.Generator().manual_seed(config['seed'] + 101)
    start, best_ce, best_epoch, best_model = 0, float('inf'), 0, None
    if resume:
        _, saved = load_model(resume, device)
        expected, actual = copy.deepcopy(config), copy.deepcopy(saved['config'])
        expected['warmup'].pop('epochs'); actual['warmup'].pop('epochs')
        if saved['kind'] != 'tqg_source' or expected != actual or saved['identity'] != task.identity:
            raise ValueError('Warmup recovery changed configuration or source data roles')
        if config['warmup']['epochs'] <= saved['epoch']:
            raise ValueError('Warmup resume needs an extended epoch budget')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start, best_ce, best_epoch, best_model = saved['epoch'], saved['best_ce'], saved['best_epoch'], saved['best_model']
        generator.set_state(saved['batch_rng']); restore_rng(saved['rng'])
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', environment())
    write_json(output / 'data_roles.json', task.identity)
    pool = task.ids(config['warmup']['source_pool'])
    started = time.perf_counter()
    with (output / 'history.jsonl').open('w') as log:
        for epoch in range(start + 1, config['warmup']['epochs'] + 1):
            model.train()
            order = [pool[i] for i in torch.randperm(len(pool), generator=generator).tolist()]
            losses = []
            for j in range(0, len(order), config['warmup']['batch_size']):
                batch = task.batch(order[j:j + config['warmup']['batch_size']], config['warmup']['source_pool'], device, dtype)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(batch), batch.y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite source loss')
                loss.backward()
                _finite_gradients(model)
                optimizer.step()
                _finite_parameters(model)
                losses.append((float(loss.detach()), batch.num_graphs))
            validation = evaluate_source(model, task)
            if validation['ce'] < best_ce:
                best_ce, best_epoch, best_model = validation['ce'], epoch, cpu_state(model)
            state = dict(**_common(model, task, config), kind='tqg_source', model=cpu_state(model),
                optimizer=optimizer.state_dict(), epoch=epoch, best_ce=best_ce, best_epoch=best_epoch,
                best_model=best_model, batch_rng=generator.get_state(), rng=rng_state())
            atomic_save(output / 'last.pt', state)
            best = {**state, 'model': best_model, 'epoch': best_epoch, 'kind': 'tqg_predictor'}
            atomic_save(output / 'best.pt', best)
            row = dict(epoch=epoch, source_ce=sum(v * n for v, n in losses) / sum(n for _, n in losses),
                       source_validation=validation, selected_epoch=best_epoch)
            log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
            print(json.dumps(dict(stage='source', epoch=epoch, validation_ce=validation['ce'])), flush=True)
    model.load_state_dict(best_model)
    result = dict(status='completed', selected_epoch=best_epoch, source_validation=evaluate_source(model, task),
                  seconds=time.perf_counter() - started, target_labels_used=False, formal_paper_result=False)
    write_json(output / 'result.json', result)
    return model, best


def run_warmup_updates(task, config, output, device='cpu'):
    """An explicitly separate control with exact optimizer/validation budgets.

    update_budget overrides epochs; graph permutations and partial minibatches
    follow the ordinary epoch trainer. Epoch-based resume is deliberately rejected.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    seed_everything(config['seed'], config['num_threads'], config['deterministic'])
    model = GraphClassifier(model_spec(task, config), len(task.classes)).to(device=device, dtype=getattr(torch, config['dtype']))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['warmup']['lr'], weight_decay=0.)
    generator = torch.Generator().manual_seed(config['seed'] + 101)
    pool = task.ids(config['warmup']['source_pool']); batch_size = config['warmup']['batch_size']
    budget = config['warmup']['update_budget']; interval = config['warmup']['validation_interval_updates']
    order, cursor, epoch, processed_graphs = [], 0, 0, 0
    best_ce, best_step, best_model = float('inf'), 0, None
    for name, value in (('config', config), ('environment', environment()), ('data_roles', task.identity)):
        write_json(output / (name + '.json'), value)
    started = time.perf_counter()
    with (output/'history.jsonl').open('x') as log:
        for step in range(1, budget + 1):
            if cursor >= len(order):
                order = [pool[i] for i in torch.randperm(len(pool), generator=generator).tolist()]
                cursor = 0; epoch += 1
            ids = order[cursor:cursor + batch_size]; cursor += len(ids); processed_graphs += len(ids)
            batch = task.batch(ids, config['warmup']['source_pool'], device, getattr(torch, config['dtype']))
            model.train(); optimizer.zero_grad(set_to_none=True)
            ce = F.cross_entropy(model(batch), batch.y)
            if not torch.isfinite(ce):
                raise FloatingPointError('Nonfinite update-budget source loss')
            ce.backward(); _finite_gradients(model); optimizer.step(); _finite_parameters(model)
            row = dict(step=step, epoch=epoch, source_ce=float(ce.detach()), source_ids=ids)
            if step % interval == 0 or step == budget:
                metrics = evaluate_source(model, task)
                row['source_validation'] = metrics
                if metrics['ce'] < best_ce:
                    best_ce, best_step, best_model = metrics['ce'], step, cpu_state(model)
                    best = dict(**_common(model, task, config), kind='tqg_predictor', model=best_model,
                        warmup_selection=dict(step=best_step, source_validation_ce=best_ce),
                        completed_updates=step, training_mode='exact_update_budget')
                    atomic_save(output/'best.pt', best)
            row['selected_step'] = best_step
            log.write(json.dumps(row, allow_nan=False)+'\n'); log.flush()
    last = dict(**_common(model, task, config), kind='tqg_predictor', model=cpu_state(model),
        completed_updates=budget, training_mode='exact_update_budget',
        warmup_selection=dict(step=best_step, source_validation_ce=best_ce))
    atomic_save(output/'last.pt', last)
    model.load_state_dict(best_model)
    result = dict(status='completed', training_mode='exact_update_budget', completed_updates=budget,
        selected_update=best_step, source_graphs_processed=processed_graphs,
        source_validation=evaluate_source(model, task), seconds=time.perf_counter()-started,
        target_labels_used=False, formal_paper_result=False)
    write_json(output/'result.json', result)
    return model, torch.load(output/'best.pt', map_location='cpu', weights_only=False)


def _finite_gradients(model):
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite network gradient; optimizer not stepped')


def _finite_parameters(model):
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('Nonfinite network parameters')


def pseudo_label_loss(logits, threshold):
    """Hard, stopped pseudo labels; normalize by retained graphs, never labels.

    Confidence and labels are both taken from the current pre-update snapshot.
    The empty-selection value stays attached to logits with exactly zero gradient.
    """
    with torch.no_grad():
        confidence, labels = logits.detach().softmax(-1).max(-1)
        retained = confidence >= threshold
    loss = (F.cross_entropy(logits[retained], labels[retained])
            if retained.any() else logits.sum() * 0.)
    count = int(retained.sum())
    return loss, dict(status='pseudo_labels', retained_count=count,
        target_graphs=len(logits), retained_fraction=count / len(logits), threshold=threshold)


class TQGTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = copy.deepcopy(config)
        self.is_v2 = config['method'] == 'tqg_v2'
        if self.is_v2:
            validate_config(config)
        optimizer = config['adaptation'].get('optimizer', 'sgd' if self.is_v2 else 'adam')
        if optimizer not in ('adam', 'sgd') or (self.is_v2 and optimizer != 'sgd'):
            raise ValueError('v2 requires SGD; supported optimizers are adam and sgd')
        factory = torch.optim.SGD if optimizer == 'sgd' else torch.optim.Adam
        self.optimizer = factory(model.parameters(), lr=config['adaptation']['lr'], weight_decay=0.)
        # Isolate roles so source-only and TQG see identical fit batch sequences.
        self.batch_rngs = {role: torch.Generator().manual_seed(config['seed'] + offset)
                           for role, offset in (('fit', 201), ('query', 202), ('target', 203))}
        self.probe_rng = torch.Generator().manual_seed(config['seed'] + 301)
        self.label_mode = config['adaptation'].get('query_label_mode', 'true')
        self.label_rng = (torch.Generator().manual_seed(config['seed'] + 10000 + config['adaptation']['query_permutation_seed'])
                          if self.label_mode == 'permute_within_batch' else None)
        self.probes = None if self.is_v2 else ProbeConfig(**config.get('probes', {}))
        self.v2 = V2Config(**config.get('v2', {})) if self.is_v2 else None
        self.step_count = 0

    def step(self, task):
        self.model.eval()
        p = next(self.model.parameters()); c = self.config['adaptation']
        batch, fit_ids = task.sample(c['source_pool'], c['fit_batch_size'], self.batch_rngs['fit'], p.device, p.dtype)
        self.optimizer.zero_grad(set_to_none=True)
        feedback, query_ids, target_ids, label_control = None, (), (), None
        pseudo_auxiliary, pseudo_statistics = None, None
        started = time.perf_counter()
        if self.config['method'] in ('tqg', 'tqg_v2'):
            predictor = c.get('feedback_predictor', 'responses')
            if predictor == 'pseudo_labels':
                target, target_ids = task.sample('target', c['target_batch_size'], self.batch_rngs['target'], p.device, p.dtype)
                pseudo_auxiliary, pseudo_statistics = pseudo_label_loss(
                    self.model(target), c.get('confidence_threshold', .9))
            else:
                query, query_ids = task.sample('query', c['query_batch_size'], self.batch_rngs['query'], p.device, p.dtype)
                target, target_ids = task.sample('target', c['target_batch_size'], self.batch_rngs['target'], p.device, p.dtype)
                labels = query.y
                if self.label_rng is not None:
                    permutation = torch.randperm(len(labels), generator=self.label_rng).to(labels.device)
                    labels = labels[permutation]
                label_control = dict(mode=self.label_mode, graphs=len(labels), changed=int((labels != query.y).sum()),
                                     permutation_seed=c.get('query_permutation_seed'))
                if self.is_v2:
                    feedback = build_gradient_feedback(self.model, query, labels, target, self.v2)
                elif predictor == 'mean_residual':
                    feedback = build_mean_feedback(self.model, query, labels, target)
                elif predictor == 'responses':
                    feedback = build_feedback(self.model, query, labels, target, self.probes, self.probe_rng)
                else:
                    feedback = build_feedback(self.model, query, labels, target, self.probes, self.probe_rng,
                                              predictor=predictor)
        ce = F.cross_entropy(self.model(batch), batch.y)
        if self.is_v2:
            auxiliary = parameter_feedback_loss(self.model, feedback)
        else:
            auxiliary = (feedback_loss(self.model, target, feedback)
                         if feedback is not None and (feedback.target_features.shape[-1] or feedback.intercept is not None)
                         else ce.new_zeros(()))
        if pseudo_auxiliary is not None:
            auxiliary = pseudo_auxiliary
        total = ce + c['feedback_weight'] * auxiliary
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite total loss')
        execution = {}
        if self.is_v2:
            active = tuple(p for p in self.model.parameters() if p.requires_grad)
            before = torch.cat([p.detach().reshape(-1) for p in active]).clone()
            feedback_parts = torch.autograd.grad(auxiliary, active, retain_graph=True, allow_unused=True)
            feedback_gradient = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                          for p, g in zip(active, feedback_parts)]).detach()
            execution.update(feedback_gradient_norm=float(feedback_gradient.norm()),
                feedback_gradient_identity_error=float((feedback_gradient-feedback.gradient).norm()),
                optimizer='sgd', momentum=0., weight_decay=0.)
        total.backward()
        _finite_gradients(self.model)
        gradient_norm = sum(float(p.grad.detach().square().sum()) for p in self.model.parameters() if p.grad is not None) ** .5
        if self.is_v2:
            full_gradient = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                       for p in active]).detach().clone()
            execution['source_gradient_norm'] = float((full_gradient-c['feedback_weight']*feedback_gradient).norm())
        self.optimizer.step()  # Exactly ONE update; all feedback is discarded afterwards.
        _finite_parameters(self.model)
        if self.is_v2:
            increment = torch.cat([p.detach().reshape(-1) for p in active])-before
            execution.update(actual_step_norm=float(increment.norm()),
                sgd_step_identity_error=float((increment+c['lr']*full_gradient).norm()))
        self.step_count += 1
        return dict(step=self.step_count, source_ce=float(ce.detach()), feedback_loss=float(auxiliary.detach()),
            total_loss=float(total.detach()), gradient_norm=gradient_norm, seconds=time.perf_counter() - started,
            query=(pseudo_statistics if pseudo_statistics is not None else
                   feedback.statistics if feedback is not None else {'status': 'source_only_baseline'}),
            batches=dict(fit=list(fit_ids), query=list(query_ids), target=list(target_ids)),
            query_label_control=label_control, **({'execution': execution} if self.is_v2 else {}))

    def state(self):
        return dict(model=cpu_state(self.model), optimizer=self.optimizer.state_dict(), step=self.step_count,
                    batch_rngs={role: gen.get_state() for role, gen in self.batch_rngs.items()},
                    probe_rng=self.probe_rng.get_state(), rng=rng_state(),
                    label_rng=self.label_rng.get_state() if self.label_rng is not None else None)

    def restore(self, state):
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.step_count = state['step']
        for role, gen in self.batch_rngs.items():
            gen.set_state(state['batch_rngs'][role])
        self.probe_rng.set_state(state['probe_rng'])
        if self.label_rng is not None:
            if state.get('label_rng') is None:
                raise ValueError('Missing permutation RNG state for negative-control resume')
            self.label_rng.set_state(state['label_rng'])
        restore_rng(state['rng'])


def run_adaptation(task, config, source_checkpoint, output, device='cpu', resume=None, step_callback=None):
    validate_config(config)
    if config['method'] in ('tqg', 'tqg_v2') and not task.target_ids:
        raise ValueError('TQG adaptation requires an explicitly declared target input pool')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    seed_everything(config['seed'], config['num_threads'], config['deterministic'])
    model, source = load_model(source_checkpoint, device)
    if config['method'] == 'tqg_v2' and (
            source['kind'] == 'tqg_adaptation' or 'source_checkpoint_sha256' in source):
        raise ValueError('v2 initialization requires a source warmup checkpoint, not an adapted model')
    source_identity, current_identity = dict(source['identity']), dict(task.identity)
    # Source warmup never loads a target pool. Only target-specific fields may differ.
    for identity in (source_identity, current_identity):
        for key in ('target_ids', 'target_index_sha256', 'target_policy', 'fingerprint'):
            identity.pop(key, None)
    if source_identity != current_identity:
        raise ValueError('Source checkpoint used different data, preprocessing or fit/query roles')
    for key in ('dtype', 'model', 'split', 'use_node_attributes', 'warmup'):
        if config[key] != source['config'][key]:
            raise ValueError('Source checkpoint configuration differs: ' + key)
    trainer = TQGTrainer(model, config)
    best_ce, best_step, best_model = float('inf'), 0, cpu_state(model)
    source_hash = sha256(source_checkpoint)
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        expected, actual = copy.deepcopy(config), copy.deepcopy(saved['config'])
        expected['adaptation'].pop('steps'); actual['adaptation'].pop('steps')
        if (saved['kind'] != 'tqg_adaptation' or saved['identity'] != task.identity or expected != actual
                or saved['source_checkpoint_sha256'] != source_hash or saved['device'] != str(torch.device(device))):
            raise ValueError('Resume changed data roles, configuration, initial source model or device')
        trainer.restore(saved)
        best_ce, best_step, best_model = saved['selection']['ce'], saved['selection']['step'], saved['selection']['model']
    else:
        best_ce = evaluate_source(model, task)['ce']
    if trainer.step_count >= config['adaptation']['steps']:
        raise ValueError('No adaptation steps remain')
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', environment())
    write_json(output / 'data_roles.json', task.identity)

    def save(path):
        state = dict(**_common(model, task, config), kind='tqg_adaptation', **trainer.state(),
            source_checkpoint_sha256=source_hash, device=str(torch.device(device)),
            selection=dict(ce=best_ce, step=best_step, model=best_model))
        atomic_save(path, state)

    started = time.perf_counter()
    with (output / 'history.jsonl').open('w') as log:
        if resume:
            # Keep the full per-step history when continuing into a new directory.
            previous = Path(resume).parent / 'history.jsonl'
            rows = [json.loads(line) for line in previous.read_text().splitlines() if line]
            rows = [row for row in rows if row['step'] <= trainer.step_count]
            if [row['step'] for row in rows] != list(range(1, trainer.step_count + 1)):
                raise ValueError('Resume history is incomplete before the checkpoint')
            for row in rows:
                log.write(json.dumps(row, allow_nan=False) + '\n')
        while trainer.step_count < config['adaptation']['steps']:
            row = trainer.step(task)
            step = trainer.step_count
            if step % config['adaptation']['validation_interval'] == 0 or step == config['adaptation']['steps']:
                row['source_validation'] = evaluate_source(model, task)
                if row['source_validation']['ce'] < best_ce:
                    best_ce, best_step, best_model = row['source_validation']['ce'], step, cpu_state(model)
            row['selected_step'] = best_step
            log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
            if step % config['adaptation']['checkpoint_interval'] == 0 or step == config['adaptation']['steps']:
                save(output / f'step_{step}.pt'); save(output / 'last.pt')
            if step_callback is not None:
                step_callback(row)
            print(json.dumps(dict(stage='adapt', step=step, source_ce=row['source_ce'],
                                  feedback_loss=row['feedback_loss'], query=row['query'])), flush=True)
    selected = dict(**_common(model, task, config), kind='tqg_predictor', model=best_model,
                    selection=dict(ce=best_ce, step=best_step), source_checkpoint_sha256=source_hash)
    atomic_save(output / 'selected.pt', selected)
    result = dict(status='completed', method=config['method'], completed_steps=trainer.step_count,
        selected_step=best_step, selected_source_validation_ce=best_ce, seconds=time.perf_counter() - started,
        target_labels_used=False, target_evaluated=False, formal_paper_result=False,
        resumed_from=str(resume) if resume else None)
    write_json(output / 'result.json', result)
    return result
