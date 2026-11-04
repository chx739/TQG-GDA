"""D1 source-only sample-size experiment; no target labels or model updates.

The common maximum odd-response dictionary is built on source bank[:32] and
target query[:16]. Every n/M/ridge comparison reuses its directions and scales.
Blocks are sealed separately and can be resumed without re-running or replacing
completed evidence. A full source seal requires all 200 predeclared blocks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from revision_common import (Budget, Context, DESIGN, STAGE, atomic_json,
                             generator, read)
from tqg_gda.residual_solver import select_ridge_cv, uniform_ridge
from tqg_gda.revision import ResponseDictionary
from tqg_gda.utils import atomic_save, fingerprint, sha256


N_VALUES = (8, 16, 32)
M_VALUES = (2, 4, 8)
RIDGES = (.01, .1, 1.)
DIRECTIONS = ('target', 'random')
BLOCKS = 20
SCHEMA = 'tqg_D1_source_v1'


def prefix_columns(feedback, requested):
    """Honor candidate-direction prefixes even if a probe was discarded.

    A valid direction sampled after the first requested candidates must not
    silently replace a discarded candidate from inside that prefix.
    """
    candidates = feedback.statistics['candidate_directions']
    discarded = {row['probe'] for row in feedback.statistics['discarded']}
    valid = [index for index in range(candidates) if index not in discarded]
    if len(valid) != feedback.source_features.shape[-1]:
        raise ValueError('D1 requires one odd column per valid direction')
    return tuple(column for column, candidate in enumerate(valid) if candidate < requested)


def nested_stratified_indices(labels, *, sizes=(16, 32), quotas=(4, 8), minority_class=None):
    """Construct nested no-replacement subsets from a randomized source bank.

    Class counts identify the source minority. Equal counts deterministically
    choose the smaller class ID. Existing smaller subsets are retained when the
    quota grows; insufficient minority observations are explicitly reported.
    """
    labels = torch.as_tensor(labels).detach().cpu().long().reshape(-1)
    if len(sizes) != len(quotas) or sorted(sizes) != list(sizes):
        raise ValueError('sizes/quotas must be equally sized with increasing n')
    counts = {int(c): int((labels == c).sum()) for c in labels.unique()}
    if not counts:
        raise ValueError('Cannot stratify an empty source bank')
    minority = (min(counts, key=lambda c: (counts[c], c))
                if minority_class is None else int(minority_class))
    minority_pool = [i for i, label in enumerate(labels.tolist()) if label == minority]
    majority_pool = [i for i, label in enumerate(labels.tolist()) if label != minority]
    selected, results = [], {}
    for n, quota in zip(sizes, quotas):
        if not 0 <= quota <= n <= len(labels):
            raise ValueError('Invalid stratified size or quota')
        current_minority = sum(int(labels[i]) == minority for i in selected)
        desired = min(quota, len(minority_pool))
        take_minority = max(0, desired - current_minority)
        available_minority = [i for i in minority_pool if i not in selected]
        if len(selected) + take_minority > n:
            raise ValueError('Requested quotas cannot preserve nested subsets')
        selected.extend(available_minority[:take_minority])
        remaining = n - len(selected)
        available_majority = [i for i in majority_pool if i not in selected]
        selected.extend(available_majority[:remaining])
        # This fallback covers an unusual source bank whose non-minority pool
        # cannot fill n; preserve uniqueness and report the actual class count.
        if len(selected) < n:
            selected.extend(i for i in minority_pool if i not in selected)
            selected = selected[:n]
        if len(selected) != n or len(set(selected)) != n:
            raise ValueError('Stratified source subset is not unique/full-size')
        actual = sum(int(labels[i]) == minority for i in selected)
        results[n] = dict(indices=tuple(selected), minority_class=minority,
                          desired_minority=quota, actual_minority=actual,
                          available_minority=len(minority_pool),
                          insufficient_minority=len(minority_pool) < quota)
    return results


def mse_metrics(residual, prediction):
    error = float((prediction - residual).square().sum(-1).mean())
    zero = float(residual.square().sum(-1).mean())
    if not all(math.isfinite(value) for value in (error, zero)):
        raise FloatingPointError('Nonfinite source residual metric')
    return dict(residual_mse=error, zero_prediction_mse=zero,
                mse_improvement=zero-error,
                prediction_rms=float(prediction.square().sum(-1).mean().sqrt()))


def spectrum_metrics(phi, ridge):
    columns = phi.shape[-1]
    if columns:
        gram = torch.einsum('ncm,nck->mk', phi, phi) / len(phi)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
        trace, maximum = float(eigenvalues.sum()), float(eigenvalues.max())
        tolerance = max(phi.shape[0] * phi.shape[1], columns) * torch.finfo(phi.dtype).eps * maximum
        return dict(gram_eigenvalues=eigenvalues.tolist(),
                    algebraic_rank=int((eigenvalues > tolerance).sum()),
                    stable_rank=trace/maximum if maximum > 0 else 0.,
                    effective_degrees_of_freedom=float((eigenvalues/(eigenvalues+ridge)).sum()),
                    ridge_condition_number=float((eigenvalues[-1]+ridge)/(eigenvalues[0]+ridge)))
    return dict(gram_eigenvalues=[], algebraic_rank=0, stable_rank=0.,
                effective_degrees_of_freedom=0., ridge_condition_number=1.)


def class_counts(labels):
    return {str(int(label)): int((labels == label).sum()) for label in labels.unique()}


def evaluate_configuration(dictionary, source_indices, requested_columns, ridge,
                           *, direction, family, cv_permutation=None, coverage=None):
    """Fit on source bank only and cache stopped target predictions.

    No source outer-audit value is passed to the CV routine. The full maximum
    dictionary and its target labels-free features were frozen beforehand.
    """
    columns = dictionary['prefix_columns'][requested_columns]
    rows = torch.as_tensor(source_indices, dtype=torch.long)
    phi_fit = dictionary['phi_bank'][rows][:, :, columns]
    residual = dictionary['source_residual'][rows]
    cv_scores, cv_order = None, None
    if ridge == 'source_CV':
        if cv_permutation is None:
            raise ValueError('CV configuration requires an independent source permutation')
        cv_order = cv_permutation.tolist()
        ridge_value, cv_scores = select_ridge_cv(phi_fit, residual, cv_permutation)
    else:
        ridge_value = float(ridge)
    coefficients = uniform_ridge(phi_fit, residual, ridge_value)
    audit_prediction = dictionary['phi_audit'][:, :, columns] @ coefficients
    query_prediction = dictionary['phi_query'][:, :, columns] @ coefficients
    held_prediction = dictionary['phi_held'][:, :, columns] @ coefficients
    n = len(source_indices)
    key = f'{family}__{direction}__n{n}__M{requested_columns}__ridge{ridge}'
    metrics = dict(source_fitted=mse_metrics(residual, phi_fit @ coefficients),
                   source_audit=mse_metrics(dictionary['audit_residual'], audit_prediction),
                   coefficient_norm=float(coefficients.norm()),
                   source_class_counts=class_counts(dictionary['source_labels'][rows]),
                   minority_source_count=int((dictionary['source_labels'][rows] == dictionary['minority_class']).sum()),
                   target_query_prediction_rms=float(query_prediction.square().sum(-1).mean().sqrt()),
                   target_heldout_prediction_rms=float(held_prediction.square().sum(-1).mean().sqrt()),
                   **spectrum_metrics(phi_fit, ridge_value))
    return dict(key=key, family=family, direction=direction, n=n,
                requested_columns=requested_columns, actual_columns=len(columns),
                source_indices=tuple(source_indices), source_ids=tuple(dictionary['bank_ids'][i] for i in source_indices),
                column_indices=columns, ridge_rule=ridge, ridge=ridge_value,
                cv_permutation=cv_order, cv_scores=cv_scores, coverage=coverage,
                coefficients=coefficients.detach(),
                d_query=query_prediction.detach(), d_held=held_prediction.detach(),
                target_query_prediction=query_prediction.detach(),
                target_heldout_prediction=held_prediction.detach(),
                source_audit_prediction=audit_prediction.detach(), metrics=metrics)


def configurations(dictionary, direction, seed, domain, block):
    predictions = []
    cv_generator = generator(920000, seed, domain, block, 3)
    for n in N_VALUES:
        # Each n/M has its own independent, recorded source-only CV ordering.
        for columns in M_VALUES:
            source_indices = tuple(range(n))
            for ridge in RIDGES:
                predictions.append(evaluate_configuration(dictionary, source_indices, columns, ridge,
                                                          direction=direction, family='fixed'))
            cv_order = torch.randperm(n, generator=cv_generator)
            predictions.append(evaluate_configuration(dictionary, source_indices, columns, 'source_CV',
                                                      direction=direction, family='source_CV', cv_permutation=cv_order))
    if direction == 'target':
        controls = nested_stratified_indices(dictionary['source_labels'], minority_class=dictionary['minority_class'])
        for n, coverage in controls.items():
            predictions.append(evaluate_configuration(dictionary, coverage['indices'], 4, .01,
                                                      direction=direction, family='stratified', coverage=coverage))
    return predictions


def build_block(model, task, ids, *, seed, domain, block, checkpoint, checkpoint_sha, budget=None):
    """Construct one immutable diagnostic block without access to target y."""
    ids = {key: tuple(map(int, value)) for key, value in ids.items()}
    if [len(ids[key]) for key in ('bank', 'audit', 'query', 'oracle_extra', 'held')] != [35, 16, 16, 16, 32]:
        raise ValueError('D1 role sizes differ from the frozen design')
    if set(ids['bank']) & set(ids['audit']):
        raise ValueError('D1 source bank and outer audit overlap')
    target_roles = [set(ids[key]) for key in ('query', 'oracle_extra', 'held')]
    if any(a & b for i, a in enumerate(target_roles) for b in target_roles[i+1:]):
        raise ValueError('D1 target query/oracle/held roles overlap')
    source_scale = task.batch(ids['bank'][:32], 'query')
    source_tail = task.batch(ids['bank'][32:], 'query')
    source_bank = task.batch(ids['bank'], 'query')
    source_audit = task.batch(ids['audit'], 'query')
    target_query = task.batch(ids['query'], 'target')
    target_held = task.batch(ids['held'], 'target')
    if target_query.y is not None or target_held.y is not None:
        raise PermissionError('D1 target inputs must have no label tensor')
    saved_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    saved_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    dictionaries, predictions = {}, []
    for direction in DIRECTIONS:
        if budget is not None:
            budget.check()
        probe_generator = generator(910000, seed, domain, block, 2)
        probe_rng_state = probe_generator.get_state().clone()
        response = ResponseDictionary.build(model, source_scale, target_query, probe_generator,
                                            K=8, response='odd', direction=direction)
        snapshot = response.snapshot
        with torch.no_grad():
            bank_logits = snapshot.logits(source_bank).detach()
            audit_logits = snapshot.logits(source_audit).detach()
            query_logits = snapshot.logits(target_query).detach()
            held_logits = snapshot.logits(target_held).detach()
            source_residual = F.one_hot(source_bank.y, bank_logits.shape[1]).to(bank_logits) - bank_logits.softmax(-1)
            audit_residual = F.one_hot(source_audit.y, audit_logits.shape[1]).to(audit_logits) - audit_logits.softmax(-1)
        # Define minority from the pre-existing source FIT role. Source outer
        # audit labels must not even choose the stratification class.
        fit_class_counts = {label: sum(task.labels[index] == label for index in task.fit_ids)
                            for label in range(bank_logits.shape[-1])}
        minority_class = min(fit_class_counts, key=lambda label: (fit_class_counts[label], label))
        dictionary = dict(bank_ids=ids['bank'], minority_class=minority_class,
                          source_fit_class_counts=fit_class_counts,
                          source_scale_ids=ids['bank'][:32], target_scale_ids=ids['query'],
                          phi_bank=torch.cat((response.feedback.source_features, response.features(source_tail))),
                          phi_audit=response.features(source_audit),
                          phi_query=response.feedback.target_features,
                          phi_held=response.features(target_held),
                          source_residual=source_residual.detach(), audit_residual=audit_residual.detach(),
                          source_labels=source_bank.y.detach().clone(), audit_labels=source_audit.y.detach().clone(),
                          bank_logits=bank_logits, audit_logits=audit_logits,
                          target_query_logits=query_logits, target_heldout_logits=held_logits,
                          scales=response.feedback.scales, directions=response.feedback.directions,
                          dictionary_statistics=response.feedback.statistics,
                          prefix_columns={m: prefix_columns(response.feedback, m) for m in M_VALUES},
                          probe_rng_state=probe_rng_state, probe_rng_final_state=probe_generator.get_state())
        dictionaries[direction] = dictionary
        predictions.extend(configurations(dictionary, direction, seed, domain, block))
    for name, parameter in model.named_parameters():
        if not torch.equal(saved_parameters[name], parameter):
            raise RuntimeError('D1 mutated model parameter ' + name)
    for name, buffer in model.named_buffers():
        if not torch.equal(saved_buffers[name], buffer):
            raise RuntimeError('D1 mutated model buffer ' + name)
    families = {family: sum(row['family'] == family for row in predictions)
                for family in ('fixed', 'source_CV', 'stratified')}
    if families != dict(fixed=54, source_CV=18, stratified=2):
        raise ValueError('D1 configuration count differs from frozen design')
    return dict(schema=SCHEMA, task=f'0_to_{domain}', domain=domain, seed=seed, block=block,
                model_state='source_warmup', checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                ids=ids, snapshot=dict(parameters=saved_parameters, buffers=saved_buffers),
                source_bank_logits=dictionaries['target']['bank_logits'],
                source_audit_logits=dictionaries['target']['audit_logits'],
                target_query_logits=dictionaries['target']['target_query_logits'],
                target_heldout_logits=dictionaries['target']['target_heldout_logits'],
                dictionaries=dictionaries, predictions=predictions, configuration_counts=families,
                target_labels_read=False, final_target_labels_read=False,
                source_outer_audit_used_for_lambda_selection=False, model_updated=False)


def prediction_json(prediction):
    return {key: value for key, value in prediction.items()
            if key not in ('coefficients', 'target_query_prediction',
                           'target_heldout_prediction', 'd_query', 'd_held', 'source_audit_prediction')}


def block_identity(contract, domain, seed, block, checkpoint, checkpoint_sha, ids):
    return dict(contract=contract, task=f'0_to_{domain}', seed=seed, block=block,
                checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                ids={key: list(value) for key, value in ids.items()})


def valid_existing(path, identity):
    metadata_path = Path(path).with_suffix('.json')
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.exists() or not metadata_path.exists():
        raise FileExistsError('Incomplete block must be explicitly recovered, never silently overwritten: ' + str(path))
    metadata = read(metadata_path)
    if metadata['identity'] != identity or metadata['identity_sha256'] != fingerprint(identity):
        raise ValueError('Existing D1 block has a different design/checkpoint/IDs: ' + str(path))
    if sha256(path) != metadata['artifact']['sha256']:
        raise ValueError('Existing D1 block changed: ' + str(path))
    return metadata


def summarize(metadata_rows):
    """First average sampling blocks within each model seed, then report seed SD."""
    groups = {}
    for metadata in metadata_rows:
        identity = metadata['identity']
        for prediction in metadata['predictions']:
            key = (identity['task'], prediction['key'])
            metrics = prediction['metrics']
            scalar = {f'{role}_{name}': value for role in ('source_fitted', 'source_audit')
                      for name, value in metrics[role].items()}
            for name in ('coefficient_norm', 'minority_source_count', 'stable_rank', 'effective_degrees_of_freedom',
                         'ridge_condition_number', 'target_query_prediction_rms', 'target_heldout_prediction_rms'):
                scalar[name] = metrics[name]
            scalar.update(actual_columns=prediction['actual_columns'], selected_ridge=prediction['ridge'])
            groups.setdefault(key, {}).setdefault(identity['seed'], []).append(scalar)
    result = []
    for (task, key), seeds in sorted(groups.items()):
        per_seed = [dict(seed=seed, blocks=len(rows),
                         metrics={name: statistics.mean(row[name] for row in rows) for name in rows[0]})
                    for seed, rows in sorted(seeds.items())]
        metric_summary = {}
        for name in per_seed[0]['metrics']:
            values = [row['metrics'][name] for row in per_seed]
            metric_summary[name] = dict(mean=statistics.mean(values),
                                        seed_std=statistics.stdev(values) if len(values) > 1 else None,
                                        model_seeds=len(values))
        result.append(dict(task=task, configuration=key, per_seed=per_seed, metrics=metric_summary))
    return result


class D1Diagnostics:
    def __init__(self, context):
        self.context = context
        self.output = STAGE / 'diagnostics' / 'D1'
        self.output.mkdir(parents=True, exist_ok=True)
        self.contract = dict(schema=SCHEMA, git_commit=context.commit,
                             script_sha256=sha256(__file__),
                             method_sha256=sha256(DESIGN / 'method.md'),
                             plan_sha256=sha256(DESIGN / 'experiment_plan.yaml'),
                             execution_lock_sha256=sha256(STAGE / 'execution_lock.json'),
                             blocks_per_seed_task=BLOCKS, source_outer_split=[35, 16],
                             maximum_dictionary_columns=8, source_scale_count=32,
                             target_query_count=16, target_heldout_count=32)

    def source(self, *, blocks=BLOCKS, domains=(2, 3), seeds=None, budget=None):
        seeds = tuple(self.context.plan['seeds'] if seeds is None else seeds)
        for domain in domains:
            for seed in seeds:
                model, _state, checkpoint = self.context.source(domain, seed)
                model.eval()
                task = self.context.task(domain)
                checkpoint_sha = sha256(checkpoint)
                folder = self.output / f'0_to_{domain}' / str(seed)
                folder.mkdir(parents=True, exist_ok=True)
                for block in range(blocks):
                    if budget is not None:
                        budget.check()
                    ids = self.context.block_ids(domain, seed, block)
                    identity = block_identity(self.contract, domain, seed, block, checkpoint, checkpoint_sha, ids)
                    path = folder / f'block_{block:02d}.pt'
                    if valid_existing(path, identity) is not None:
                        print(json.dumps(dict(event='D1_resumed', task=f'0_to_{domain}', seed=seed, block=block)), flush=True)
                        continue
                    started = time.perf_counter()
                    payload = build_block(model, task, ids, seed=seed, domain=domain, block=block,
                                          checkpoint=checkpoint, checkpoint_sha=checkpoint_sha, budget=budget)
                    payload['contract'] = self.contract
                    payload['identity'] = identity
                    payload['identity_sha256'] = fingerprint(identity)
                    atomic_save(path, payload)
                    elapsed = time.perf_counter() - started
                    metadata = dict(identity=identity, identity_sha256=fingerprint(identity),
                                    artifact=dict(path=str(path), sha256=sha256(path)), elapsed_seconds=elapsed,
                                    configuration_counts=payload['configuration_counts'],
                                    predictions=[prediction_json(row) for row in payload['predictions']],
                                    target_labels_read=False, final_target_labels_read=False, model_updated=False)
                    atomic_json(path.with_suffix('.json'), metadata)
                    print(json.dumps(dict(event='D1_block_completed', task=f'0_to_{domain}', seed=seed,
                                          block=block, elapsed_seconds=elapsed, configurations=74)), flush=True)
        return self.refresh_summary()

    def refresh_summary(self):
        rows, artifacts, expected = [], [], 0
        for domain in (2, 3):
            for seed in self.context.plan['seeds']:
                checkpoint = next(row['warmup_checkpoint'] for row in self.context.old_manifest['records']
                                  if row['task'] == f'0_to_{domain}' and row['seed'] == seed
                                  and row['arm'] == 'original_TQG_target_probes')
                checkpoint_sha = sha256(checkpoint)
                for block in range(BLOCKS):
                    expected += 1
                    path = self.output / f'0_to_{domain}' / str(seed) / f'block_{block:02d}.pt'
                    identity = block_identity(self.contract, domain, seed, block, checkpoint, checkpoint_sha,
                                              self.context.block_ids(domain, seed, block))
                    metadata = valid_existing(path, identity)
                    if metadata is not None:
                        rows.append(metadata)
                        artifacts.extend((metadata['artifact'],
                                          dict(path=str(path.with_suffix('.json')), sha256=sha256(path.with_suffix('.json')))))
        complete = len(rows) == expected == 200
        summary = dict(schema=SCHEMA, contract=self.contract, complete=complete,
                       completed_blocks=len(rows), expected_blocks=expected,
                       aggregation='arithmetic block mean within model seed, then mean and sample SD over model seeds',
                       target_labels_read=False, final_target_labels_read=False,
                       source_mse_is_not_a_necessary_target_gain_gate=True,
                       primary_regressions=sum(row['configuration_counts']['fixed'] for row in rows),
                       CV_refits=sum(row['configuration_counts']['source_CV'] for row in rows),
                       CV_inner_fits=12*sum(row['configuration_counts']['source_CV'] for row in rows),
                       stratified_regressions=sum(row['configuration_counts']['stratified'] for row in rows),
                       total_ridge_solves=sum(row['configuration_counts']['fixed']
                                              +13*row['configuration_counts']['source_CV']
                                              +row['configuration_counts']['stratified'] for row in rows),
                       measured_block_seconds=sum(row['elapsed_seconds'] for row in rows),
                       groups=summarize(rows))
        summary_path = self.output / 'source_summary.json'
        atomic_json(summary_path, summary)
        if complete:
            artifacts.append(dict(path=str(summary_path), sha256=sha256(summary_path)))
            seal = dict(schema=SCHEMA, contract=self.contract, complete=True,
                        completed_blocks=200, fixed_regressions=10800,
                        source_CV_refits=3600, source_CV_inner_fits=43200,
                        stratified_regressions=400, total_ridge_solves=58000,
                        artifacts=artifacts, target_labels_read=False,
                        final_target_labels_read=False, model_updated=False)
            seal_path = self.output / 'source_seal.json'
            if seal_path.exists():
                if read(seal_path) != seal:
                    raise ValueError('D1 source seal would change')
            else:
                atomic_json(seal_path, seal)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=int, default=BLOCKS, help='Run the first N predeclared blocks, allowing profiling/resume')
    parser.add_argument('--domain', type=int, choices=(2, 3))
    parser.add_argument('--seed', type=int, choices=(3405, 3406, 3407, 3408, 3409))
    args = parser.parse_args()
    if not 1 <= args.blocks <= BLOCKS:
        parser.error('--blocks must be in 1..20')
    context = Context()
    diagnostics = D1Diagnostics(context)
    budget = Budget('D1_source')
    status = 'failed'
    try:
        summary = diagnostics.source(blocks=args.blocks,
                                     domains=(args.domain,) if args.domain is not None else (2, 3),
                                     seeds=(args.seed,) if args.seed is not None else None,
                                     budget=budget)
        status = 'completed' if summary['complete'] else 'partial_completed'
        print(json.dumps(dict(event='D1_source_finished', status=status,
                              completed_blocks=summary['completed_blocks'], expected_blocks=200)), flush=True)
    finally:
        budget.finish(status)


if __name__ == '__main__':
    main()
