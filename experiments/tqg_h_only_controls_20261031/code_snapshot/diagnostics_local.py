"""D2 sealed source construction and separate offline development evaluation."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from revision_common import Context, STAGE, Budget, generator, read, atomic_json
from revision_evaluation import DevelopmentLabels
from revision_execution_diagnostics import ExecutionDiagnostics
from tqg_gda.revision import ResponseDictionary, ResidualPredictor
from tqg_gda.residual_solver import uniform_ridge, select_ridge_cv
from tqg_gda.utils import sha256

OUT = STAGE / 'diagnostics/D2'
ARMS = ['T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14', 'T15', 'T16',
        'zero', 'global_mean']


def configuration(ctx, arm):
    if arm in ('zero', 'global_mean'):
        value = ctx.arm('T5')
        value.update(arm=arm, predictor=arm, projection=False)
        return value
    return ctx.arm(arm)


def source(ctx, blocks=20, only_domain=None, only_seed=None):
    budget = Budget('D2_source')
    status = 'failed'
    try:
        for domain in (2, 3):
            if only_domain is not None and domain != only_domain:
                continue
            task = ctx.task(domain)
            for seed in ctx.plan['seeds']:
                if only_seed is not None and seed != only_seed:
                    continue
                frozen, _, _ = ctx.source(domain, seed)
                for state_name, count in [('warmup', blocks), ('historical_final', min(blocks, 3))]:
                    model, state, checkpoint = (ctx.source(domain, seed) if state_name == 'warmup'
                                                 else ctx.final_state(domain, seed))
                    for block in range(count):
                        budget.check()
                        folder = OUT / f'0_to_{domain}' / str(seed) / state_name
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / f'block_{block:02d}.pt'
                        meta_path = path.with_suffix('.json')
                        if path.exists() and meta_path.exists():
                            meta = read(meta_path)
                            if meta['commit'] != ctx.commit or sha256(path) != meta['sha256']:
                                raise RuntimeError('Incompatible cached D2 block')
                            continue
                        if path.exists() or meta_path.exists():
                            raise RuntimeError('Partial D2 block must be preserved and audited')
                        ids = ctx.block_ids(domain, seed, block)
                        bank = task.batch(ids['bank'][:32], 'query')
                        query, held = task.batch(ids['query'], 'target'), task.batch(ids['held'], 'target')
                        extra = task.batch(ids['oracle_extra'], 'target')
                        fit_rng = generator(910000, seed, domain, block, 5)
                        fit_ids = tuple(task.fit_ids[i] for i in torch.randperm(len(task.fit_ids), generator=fit_rng)[:16].tolist())
                        ids['fit'] = fit_ids
                        dictionaries = {}
                        for name, direction, response in [('odd', 'target', 'odd'),
                                                           ('random', 'random', 'odd'),
                                                           ('linear', 'target', 'linear_JVP')]:
                            dictionaries[name] = ResponseDictionary.build(model, bank, query,
                                generator(910000, seed, domain, block, 2), K=4,
                                response=response, direction=direction)
                        folds = torch.randperm(32, generator=generator(920000, seed, domain, block, 3))
                        predictions = []
                        for perm in (None, 113, 227, 349):
                            labels = bank.y.clone()
                            if perm is not None:
                                order = torch.randperm(32, generator=generator(perm, seed, domain, block, 4))
                                labels = labels[order]
                            for arm in ARMS:
                                cfg = configuration(ctx, arm)
                                dictionary = dictionaries['random' if arm == 'T14' else 'linear' if arm == 'T15' else 'odd']
                                predictor = ResidualPredictor(dictionary, bank, cfg, folds,
                                                              frozen_model=frozen, source_labels=labels)
                                qpred = predictor.predict(query, phi=dictionary.feedback.target_features,
                                    permutation_generator=generator(930000, seed, domain, block, 4))
                                hpred = predictor.predict(held,
                                    permutation_generator=generator(930001, seed, domain, block, 4))
                                predictions.append(dict(arm=arm, label_condition='true' if perm is None else f'perm_{perm}',
                                                        query=qpred, held=hpred, source_labels=labels))
                        dictionary = dictionaries['odd']
                        payload = dict(commit=ctx.commit, domain=domain, seed=seed, block=block, state=state_name,
                            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), ids=ids,
                            z_query=dictionary.snapshot.logits(query).detach(), z_held=dictionary.snapshot.logits(held).detach(),
                            phi_oracle=torch.cat((dictionary.feedback.target_features, dictionary.features(extra))),
                            phi_held=dictionary.features(held),
                            z_oracle=torch.cat((dictionary.snapshot.logits(query).detach(), dictionary.snapshot.logits(extra).detach())),
                            predictions=predictions)
                        torch.save(payload, path)
                        atomic_json(meta_path, dict(commit=ctx.commit, path=str(path), sha256=sha256(path),
                                                   prediction_count=len(predictions), target_labels_used=False))
                        print(json.dumps(dict(event='D2_source_block', task=f'0_to_{domain}', seed=seed,
                                              state=state_name, block=block)), flush=True)
        metas = sorted(OUT.glob('0_to_*/*/*/block_*.json'))
        expected = {OUT/f'0_to_{d}'/str(s)/state/f'block_{b:02d}.json'
                    for d in (2,3) for s in ctx.plan['seeds']
                    for state,count in [('warmup',20),('historical_final',3)] for b in range(count)}
        if set(metas) == expected:
            if any(read(p)['prediction_count'] != len(ARMS)*4 for p in metas):
                raise ValueError('Incomplete D2 prediction family')
            atomic_json(OUT/'source_seal.json', dict(commit=ctx.commit, complete=True, blocks=230,
                artifacts=[read(p) for p in metas], target_labels_used=False, final_labels_used=False))
        status = 'completed'
    finally:
        budget.finish(status)


@torch.no_grad()
def residual_metrics(logits, labels, prediction):
    r = F.one_hot(labels, logits.shape[1]).to(logits)-logits.softmax(-1)
    if prediction.source_ce_weights is not None:
        return dict(target_residual_mse=None, zero_residual_mse=float(r.square().sum(-1).mean()),
                    target_residual_gain=None,target_gate_rate=None,predicted_correction_rms=None)
    d = prediction.correction*prediction.gate[:, None]
    mse = float((r-d).square().sum(-1).mean())
    zero = float(r.square().sum(-1).mean())
    return dict(target_residual_mse=mse, zero_residual_mse=zero, target_residual_gain=zero-mse,
                target_gate_rate=float(prediction.gate.double().mean()),
                predicted_correction_rms=float(d.square().sum(-1).mean().sqrt()))


def target(ctx):
    budget = Budget('D2_offline_target_evaluation')
    status = 'failed'
    try:
        seal = read(OUT/'source_seal.json')
        if seal['commit'] != ctx.commit or not seal['complete']:
            raise ValueError('Missing compatible D2 source seal')
        for artifact in seal['artifacts']:
            if sha256(artifact['path']) != artifact['sha256']:
                raise ValueError('Source-sealed D2 predictions changed')
        for domain in (2, 3):
            task = ctx.task(domain)
            labels = DevelopmentLabels(ctx, domain, task, OUT/'target_label_access.jsonl')
            for seed in ctx.plan['seeds']:
                for state_name, count in [('warmup', 20), ('historical_final', 3)]:
                    model, state, checkpoint = (ctx.source(domain, seed) if state_name == 'warmup'
                                                 else ctx.final_state(domain, seed))
                    for block in range(count):
                        budget.check()
                        folder = OUT / f'0_to_{domain}' / str(seed) / state_name
                        result_path = folder / f'target_{block:02d}.json'
                        if result_path.exists():
                            if read(result_path)['commit'] != ctx.commit:
                                raise ValueError('Incompatible target evaluation')
                            continue
                        cache = torch.load(folder/f'block_{block:02d}.pt', weights_only=False, map_location='cpu')
                        ids = cache['ids']
                        bank = task.batch(ids['bank'][:32], 'query')
                        query, held = task.batch(ids['query'], 'target'), task.batch(ids['held'], 'target')
                        fit = task.batch(ids['fit'], 'fit')
                        yq, yh = labels(ids['query']), labels(ids['held'])
                        execution = ExecutionDiagnostics(model, fit, bank, query, held, yq, yh,
                            historical_optimizer=state['optimizer'] if state_name == 'historical_final' else None)
                        rows = []
                        for item in cache['predictions']:
                            result = residual_metrics(cache['z_held'], yh, item['held'])
                            result.update(execution.score(item))
                            rows.append(dict(arm=item['arm'], label_condition=item['label_condition'],
                                             metrics=result, source_statistics=item['query'].statistics))
                        yo = labels(tuple(ids['query'])+tuple(ids['oracle_extra']))
                        ro = F.one_hot(yo, cache['z_oracle'].shape[1]).to(cache['z_oracle'])-cache['z_oracle'].softmax(-1)
                        folds = torch.randperm(32, generator=generator(920000, seed, domain, block, 3))
                        ridge, cv = select_ridge_cv(cache['phi_oracle'], ro, folds)
                        a = uniform_ridge(cache['phi_oracle'], ro, ridge)
                        rh = F.one_hot(yh, cache['z_held'].shape[1]).to(cache['z_held'])-cache['z_held'].softmax(-1)
                        oracle_d = cache['phi_held']@a
                        oracle_mse = float((rh-oracle_d).square().sum(-1).mean())
                        for row, item in zip(rows, cache['predictions']):
                            row['metrics']['prediction_distance_to_global_oracle'] = (
                                float((item['held'].correction*item['held'].gate[:,None]-oracle_d)
                                      .square().sum(-1).mean())
                                if item['held'].source_ce_weights is None else None)
                        atomic_json(result_path, dict(commit=ctx.commit, domain=domain, seed=seed,
                            state=state_name, block=block, rows=rows,
                            oracle=dict(ridge=ridge, CV=cv, residual_mse=oracle_mse,
                                        is_UDA_result=False, is_theoretical_upper_bound=False),
                            source_only_trial=execution.source_trial))
                        print(json.dumps(dict(event='D2_offline_block', task=f'0_to_{domain}', seed=seed,
                                              state=state_name, block=block)), flush=True)
        status = 'completed'
    finally:
        budget.finish(status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['source', 'target'], required=True)
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--domain', type=int, choices=[2, 3])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    context = Context()
    if args.phase == 'source':
        source(context, args.blocks, args.domain, args.seed)
    else:
        target(context)
