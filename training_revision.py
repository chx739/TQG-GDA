"""Paired, source-only-controlled revision-v1 training and exact recovery.

This module never imports the offline target-label evaluator. All target inputs
come through GraphDomainTask's label-free target view.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from revision_common import CODE, STAGE, OLD, generator, atomic_json, hardware
from tqg_gda.metrics import evaluate_source
from tqg_gda.revision import ResponseDictionary, ResidualPredictor, auxiliary_loss
from tqg_gda.training import pseudo_label_loss, _finite_gradients, _finite_parameters
from tqg_gda.utils import atomic_save, cpu_state, fingerprint, rng_state, restore_rng, seed_everything, sha256


def paired_ids(task, seed, domain, step):
    """Draw the complete shared schedule regardless of the requested arm.

    The single data-role generator always draws fit, all-source, query32 and
    target, in that order. Thus changing an arm's role usage cannot advance a
    different arm's data stream. Query16 is literally query32's prefix.
    """
    rng = generator(910000, seed, domain, step, 1)
    before = rng.get_state().clone()
    result = {}
    for name, role, count in (('fit', 'fit', 16), ('all_source', 'all_source', 16),
                              ('query32', 'query', 32), ('target', 'target', 16)):
        pool = tuple(task.ids(role))
        if not pool:
            raise ValueError('Empty data role: ' + role)
        order = torch.randperm(len(pool), generator=rng)[:count].tolist()
        result[name] = tuple(int(pool[i]) for i in order)
    return result, dict(before=before, after=rng.get_state().clone())


def _norms(model, values):
    squares = dict(full=0., encoder=0., head=0.)
    for (name, _), value in zip(model.named_parameters(), values):
        if value is None:
            continue
        square = float(value.detach().square().sum())
        squares['full'] += square
        if name.startswith('encoder.'):
            squares['encoder'] += square
        elif name.startswith('head.'):
            squares['head'] += square
    return {name: value ** .5 for name, value in squares.items()}


def _gradient(loss, parameters):
    if not loss.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _rng_digest(value):
    result = {}
    for role, states in value.items():
        result[role] = {point: hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                        for point, state in states.items()}
    return result


class RevisionTrainer:
    def __init__(self, model, frozen_model, task, arm, plan, seed, domain):
        self.model, self.task = model, task
        self.frozen_model = frozen_model
        self.arm, self.plan = copy.deepcopy(arm), copy.deepcopy(plan)
        self.seed, self.domain, self.step_count = int(seed), int(domain), 0
        settings = plan['adaptation']
        adam = settings['adam']
        self.optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
            betas=(adam['beta1'], adam['beta2']), eps=adam['epsilon'],
            weight_decay=adam['weight_decay'], amsgrad=adam['amsgrad'])
        self.role_rngs = {}
        self.last_ids = None
        self.model.eval()
        self.frozen_model.eval()

    def step(self):
        step = self.step_count + 1
        started = time.perf_counter()
        arm, task = self.arm, self.task
        self.model.eval()
        parameter = next(self.model.parameters())
        schedule, data_state = paired_ids(task, self.seed, self.domain, step)
        source_pool = 'all_source' if arm['method'] == 'all_source_CE' else 'fit'
        fit_ids = schedule[source_pool]
        fit = task.batch(fit_ids, source_pool, parameter.device, parameter.dtype)
        query_ids, target_ids = (), ()
        query, target, prediction = None, None, None
        rngs = dict(probes=generator(910000, self.seed, self.domain, step, 2),
                    CV=generator(920000, self.seed, self.domain, step, 3),
                    weight_permutation=generator(930000, self.seed, self.domain, step, 4))
        states = dict(data=data_state, **{name: dict(before=rng.get_state().clone())
                                        for name, rng in rngs.items()})
        method = arm['method']
        statistics = dict(status='source_only_baseline')
        self.optimizer.zero_grad(set_to_none=True)
        if method in ('tqg', 'weighted_source_query_CE', 'hard_pseudo_labels'):
            target_ids = schedule['target']
            target = task.batch(target_ids, 'target', parameter.device, parameter.dtype)
            if getattr(target, 'y', None) is not None:
                raise PermissionError('Target training batch contains labels')
        if method in ('tqg', 'weighted_source_query_CE'):
            query_ids = schedule['query32'][:arm['n']]
            if len(query_ids) != arm['n']:
                raise ValueError('Insufficient distinct source query graphs')
            query = task.batch(query_ids, 'query', parameter.device, parameter.dtype)
            dictionary = ResponseDictionary.build(self.model, query, target, rngs['probes'],
                K=arm['K'], response=arm['response'], direction=arm['direction'])
            cv_order = torch.randperm(len(query_ids), generator=rngs['CV'])
            predictor = ResidualPredictor(dictionary, query, arm, cv_order, self.frozen_model)
            prediction = predictor.predict(target, phi=dictionary.feedback.target_features,
                permutation_generator=rngs['weight_permutation'])
            statistics = dict(status='revision_feedback', **prediction.statistics,
                source_class_counts=torch.bincount(query.y, minlength=len(task.classes)).tolist(),
                probe_statistics=dictionary.feedback.statistics)
        elif method not in ('source_fit_CE', 'all_source_CE', 'hard_pseudo_labels'):
            raise ValueError('Unknown training method: ' + method)
        source_ce = F.cross_entropy(self.model(fit), fit.y)
        if prediction is not None:
            auxiliary = auxiliary_loss(self.model, query, target, prediction)
        elif method == 'hard_pseudo_labels':
            auxiliary, statistics = pseudo_label_loss(self.model(target), arm.get('threshold', .9))
        else:
            auxiliary = source_ce.new_zeros(())
        beta = self.plan['adaptation']['feedback_weight']
        if method == 'hard_pseudo_labels':
            beta = arm.get('target_weight', beta)
        total = source_ce + beta * auxiliary
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite revision training loss')
        parameters = tuple(self.model.parameters())
        source_grad = _gradient(source_ce, parameters)
        auxiliary_grad = _gradient(auxiliary, parameters)
        if any(grad is not None and not bool(torch.isfinite(grad).all())
               for grad in source_grad + auxiliary_grad):
            raise FloatingPointError('Nonfinite source or feedback gradient')
        weighted_grad = tuple(None if grad is None else beta * grad for grad in auxiliary_grad)
        source_norm = _norms(self.model, source_grad)
        feedback_norm = _norms(self.model, auxiliary_grad)
        weighted_norm = _norms(self.model, weighted_grad)
        before = tuple(p.detach().clone() for p in parameters)
        total.backward()  # one combined backward, followed by exactly one update
        _finite_gradients(self.model)
        total_norm = _norms(self.model, tuple(p.grad for p in parameters))
        self.optimizer.step()
        _finite_parameters(self.model)
        actual_step = tuple(p.detach() - value for p, value in zip(parameters, before))
        step_norm = _norms(self.model, actual_step)
        for name, rng in rngs.items():
            states[name]['after'] = rng.get_state().clone()
        self.step_count, self.role_rngs = step, states
        self.last_ids = dict(fit=list(fit_ids), query=list(query_ids), target=list(target_ids))
        return dict(step=step, arm=arm['arm'], task=f'0_to_{self.domain}', model_seed=self.seed,
            source_ce=float(source_ce.detach()), feedback_loss=float(auxiliary.detach()),
            feedback_weight=beta, total_loss=float(total.detach()), source_gradient_norm=source_norm,
            feedback_gradient_norm=feedback_norm, weighted_feedback_gradient_norm=weighted_norm,
            total_gradient_norm=total_norm, actual_step_norm=step_norm,
            batches=self.last_ids, paired_schedule={key: list(ids) for key, ids in schedule.items()},
            query=statistics, rng_state_ids=_rng_digest(states),
            seconds=time.perf_counter()-started)

    def state(self):
        return dict(model=cpu_state(self.model), optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    step=self.step_count, role_rngs=copy.deepcopy(self.role_rngs), rng=rng_state(),
                    last_batch_ids=copy.deepcopy(self.last_ids),
                    stateless_rng_contract=dict(components=['base', 'model_seed', 'domain', 'step', 'role'],
                        data_base=910000, probe_base=910000, CV_base=920000,
                        weight_permutation_base=930000,
                        data_draw_order=['fit16', 'all_source16', 'query32', 'target16']))

    def restore(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.step_count = state['step']
        self.role_rngs = copy.deepcopy(state['role_rngs'])
        self.last_ids = copy.deepcopy(state['last_batch_ids'])
        restore_rng(state['rng'])


def tensorboard_writer(output, resume_step=None):
    dependencies = OLD / 'runtime_deps'
    if str(dependencies) not in sys.path:
        sys.path.insert(0, str(dependencies))
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter(log_dir=str(Path(output) / 'tensorboard'),
                         purge_step=resume_step + 1 if resume_step is not None else None)


def _write_scalars(writer, row):
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(prefix + '/' + str(key) if prefix else str(key), child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            writer.add_scalar(prefix, value, row['step'])
    for name in ('source_ce', 'feedback_loss', 'total_loss', 'source_gradient_norm',
                 'feedback_gradient_norm', 'weighted_feedback_gradient_norm',
                 'total_gradient_norm', 'actual_step_norm', 'source_validation', 'query', 'seconds'):
        if name in row:
            visit(name, row[name])


def _trim_history(path, completed_step):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows = [row for row in rows if row['step'] <= completed_step]
    if [row['step'] for row in rows] != list(range(1, completed_step + 1)):
        raise ValueError('History is incomplete or duplicated before resume checkpoint')
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def run_job(context, domain, seed, arm_name, *, output=None, stop_after=None, resume=False,
            budget=None, writer_factory=tensorboard_writer):
    """Run one planned job or continue its profile checkpoint without reset."""
    arm = context.arm(arm_name)
    if not arm.get('training'):
        raise ValueError('Frozen reference has no training job')
    all_source = arm['initialization'] == 'all_source_warmup'
    task = context.task(domain, all_source)
    seed_everything(seed, 1, True)
    model, source_state, source_path = context.source(domain, seed, all_source)
    frozen_model = copy.deepcopy(model).eval()
    for parameter in frozen_model.parameters():
        parameter.requires_grad_(False)
    trainer = RevisionTrainer(model, frozen_model, task, arm, context.plan, seed, domain)
    total_steps = context.plan['adaptation']['steps']
    until = total_steps if stop_after is None else int(stop_after)
    if not 1 <= until <= total_steps:
        raise ValueError('Requested stop lies outside the planned update budget')
    output = Path(output or STAGE / 'jobs' / f'0_to_{domain}' / str(seed) / arm_name)
    source_hash = sha256(source_path)
    config = copy.deepcopy(source_state['config'])
    config.update(revision=arm, seed=int(seed), task=f'0_to_{domain}',
                  revision_id=context.plan['revision_id'],
                  adaptation=copy.deepcopy(context.plan['adaptation']),
                  source_checkpoint=str(source_path), source_checkpoint_sha256=source_hash,
                  git_commit=context.commit, target_labels_used=False)
    config_sha = fingerprint(config)
    initial_validation = evaluate_source(model, task)
    selection = dict(ce=initial_validation['ce'], step=0, model=cpu_state(model))
    previous_seconds = 0.
    exists = output.exists()
    if exists:
        if not resume:
            raise FileExistsError('Existing job requires explicit --resume: ' + str(output))
        saved = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        if (saved.get('kind') != 'tqg_adaptation' or saved.get('config_sha256') != config_sha
                or saved.get('source_checkpoint_sha256') != source_hash
                or saved.get('git_commit') != context.commit or saved.get('identity') != task.identity):
            raise ValueError('Resume changed configuration, source checkpoint, code or role IDs')
        trainer.restore(saved)
        selection = saved['selection']
        if (output / 'result.json').exists():
            previous = json.loads((output / 'result.json').read_text())
            previous_seconds = previous.get('seconds', 0.)
            if previous.get('status') == 'completed':
                for name, expected in previous['artifact_hashes'].items():
                    if sha256(output / name) != expected:
                        raise ValueError('Completed job artifact changed: ' + name)
                if trainer.step_count != total_steps:
                    raise ValueError('Completed result disagrees with saved update count')
                return previous
        _trim_history(output / 'history.jsonl', trainer.step_count)
    else:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'config.json', config)
        atomic_json(output / 'environment.json', hardware())
        atomic_json(output / 'data_roles.json', task.identity)
        atomic_json(output / 'initial_source_validation.json', initial_validation)
        atomic_json(output / 'RUN_INFO.json', dict(
            revision_id=context.plan['revision_id'], arm=arm_name, seed=seed, task=f'0_to_{domain}',
            git_commit=context.commit, config_sha256=config_sha,
            hardware=hardware(), gpu_hours=0,
            source_checkpoint=str(source_path), source_checkpoint_sha256=source_hash,
            final_target_labels_allowed=False, target_labels_used=False,
            initialization='fresh_adaptation_Adam', training_then_offline_evaluation=True))
        (output / 'history.jsonl').touch(exist_ok=False)
    common = dict(schema_version=1, kind='tqg_adaptation', encoder_spec=model.encoder.spec,
        class_labels=task.classes, identity=copy.deepcopy(task.identity),
        preprocessing=copy.deepcopy(task.preprocessing), config=config,
        config_sha256=config_sha, revision_config=copy.deepcopy(arm), git_commit=context.commit,
        source_checkpoint=str(source_path), source_checkpoint_sha256=source_hash, device='cpu')

    def save_checkpoint(snapshot=False):
        state = dict(**common, **trainer.state(), selection=copy.deepcopy(selection))
        atomic_save(output / 'last.pt', state)
        if snapshot:
            atomic_save(output / f'step_{trainer.step_count}.pt', state)
        selected = {**state, 'kind': 'tqg_predictor', 'model': selection['model'],
                    'selected_step': selection['step'], 'evaluation_only': True,
                    'optimizer_state_matches_model': selection['step'] == trainer.step_count}
        atomic_save(output / 'selected.pt', selected)

    if not exists:
        save_checkpoint(snapshot=True)
    if trainer.step_count > until:
        raise ValueError('Job has already passed the requested stopping step')
    started = time.perf_counter()
    writer = writer_factory(output, trainer.step_count if exists else None)
    status = 'profile_paused' if until < total_steps else 'completed'
    checkpoint_steps = set(context.plan['adaptation']['complete_state_checkpoints'])
    try:
        with (output / 'history.jsonl').open('a') as log:
            while trainer.step_count < until:
                if budget is not None:
                    budget.check()
                row = trainer.step()
                if row['step'] % context.plan['adaptation']['source_validation_interval'] == 0:
                    validation = evaluate_source(model, task)
                    row['source_validation'] = validation
                    if validation['ce'] < selection['ce']:
                        selection = dict(ce=validation['ce'], step=row['step'], model=cpu_state(model))
                row['selected_step'] = selection['step']
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                if row['step'] > 100 and row['step'] % 50 == 0:
                    previous_rows=[json.loads(line) for line in (output/'history.jsonl').read_text().splitlines()]
                    recent=previous_rows[-100:]
                    first=sum(r['total_loss'] for r in recent[:50])/50
                    last=sum(r['total_loss'] for r in recent[50:])/50
                    if last >= first:
                        with (output/'monitor_events.jsonl').open('a') as events:
                            events.write(json.dumps(dict(event='no_decrease_warning',step=row['step'],
                                first_mean=first,last_mean=last,action='continue_dynamic_objective'))+'\n')
                _write_scalars(writer, row)
                # Persist every completed update so process interruption and
                # profile recovery never silently replay an optimizer step.
                save_checkpoint(snapshot=row['step'] in checkpoint_steps)
    except BaseException:
        status = 'interrupted'
        raise
    finally:
        writer.flush()
        writer.close()
        elapsed = previous_seconds + time.perf_counter() - started
        result = dict(status=status, arm=arm_name, task=f'0_to_{domain}', seed=seed,
            completed_updates=trainer.step_count, total_updates=total_steps,
            selected_step=selection['step'], selected_source_validation_ce=selection['ce'],
            seconds=elapsed, target_labels_used=False, final_target_labels_used=False,
            config_sha256=config_sha, git_commit=context.commit,
            source_checkpoint_sha256=source_hash,
            last_checkpoint=str(output / 'last.pt'), selected_checkpoint=str(output / 'selected.pt'))
        files = ('config.json', 'environment.json', 'data_roles.json', 'RUN_INFO.json',
                 'initial_source_validation.json', 'history.jsonl', 'last.pt', 'selected.pt')
        result['artifact_hashes'] = {name: sha256(output / name) for name in files}
        atomic_json(output / 'result.json', result)
    return result
