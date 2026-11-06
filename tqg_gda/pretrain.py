"""Stage A: fit-only CE initialization, source validation and reusable cache."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .classifier import GraphClassifier
from .metrics import evaluate_source
from .predict import load_predictor, model_checkpoint
from .provenance import source_identity, source_key, verify_artifacts
from .utils import atomic_save, cpu_state, seed_everything, sha256, write_json


def check_finite(model, loss):
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite training loss')
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite gradient; refusing optimizer step')


def pretrain_model(config, task, code, runtime):
    key = source_key(config, task, code, runtime)
    folder = Path(config.out_dir)/'encoders'/key
    if folder.exists():
        if not (folder/'result.json').is_file():
            raise ValueError('Incomplete source cache; retain it for inspection and choose a new out_dir')
        info = json.loads((folder/'result.json').read_text())
        if info.get('source_key') != key or info.get('status') != 'completed':
            raise ValueError('Source cache identity mismatch')
        verify_artifacts(folder, info['artifacts'])
        model, saved = load_predictor(folder/'best.pt', runtime['device'])
        if saved.get('source_key') != key or saved.get('data') != source_identity(task):
            raise ValueError('Source model identity mismatch')
        return model, dict(info, cache_hit=True), folder

    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'config.json', dict(encoder=config.to_dict()['encoder'],
        pretrain=config.to_dict()['pretrain'], seed=config.seed, runtime=runtime))
    write_json(folder/'data_roles.json', source_identity(task))
    write_json(folder/'provenance.json', code)
    seed_everything(config.seed, config.num_threads, config.deterministic)
    model = GraphClassifier(dict(in_dim=task.input_dim, edge_dim=task.edge_dim,
        **config.to_dict()['encoder']), len(task.classes), dtype=getattr(torch, config.dtype)).to(
        device=runtime['device'], dtype=getattr(torch, config.dtype))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretrain.lr,
                                 weight_decay=config.pretrain.weight_decay)
    generator = torch.Generator().manual_seed(config.seed + 101)
    best_ce, best_epoch, best_state = float('inf'), None, None
    updates = 0
    with (folder/'history.jsonl').open('x') as log:
        for epoch in range(1, config.pretrain.epochs + 1):
            model.train()
            order = [task.fit_ids[i] for i in torch.randperm(len(task.fit_ids), generator=generator).tolist()]
            total, graphs = 0., 0
            for start in range(0, len(order), config.pretrain.batch_size):
                ids = order[start:start+config.pretrain.batch_size]
                batch = task.batch(ids, 'fit', runtime['device'], getattr(torch, config.dtype))
                # BatchNorm on a single node has no training variance. Fail
                # explicitly instead of silently changing the reference encoder.
                if batch.num_nodes < 2:
                    raise ValueError('SEGA BatchNorm pretraining requires at least two nodes per batch')
                labels = batch.y
                batch.y = None
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(batch), labels)
                check_finite(model, loss)
                loss.backward()
                check_finite(model, loss)
                optimizer.step()
                if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
                    raise FloatingPointError('Nonfinite source model state')
                total += float(loss.detach())*len(ids)
                graphs += len(ids)
                updates += 1
            validation = evaluate_source(model, task)
            if validation['ce'] < best_ce:
                best_ce, best_epoch, best_state = validation['ce'], epoch, cpu_state(model)
            log.write(json.dumps(dict(epoch=epoch, updates=updates, source_ce=total/graphs,
                source_validation=validation, selected_epoch=best_epoch), allow_nan=False)+'\n')
            log.flush()
    model.load_state_dict(best_state)
    model.eval()
    atomic_save(folder/'best.pt', model_checkpoint(model, task, config,
        stage='source', source_key=key, data=source_identity(task),
        selection=dict(epoch=best_epoch, source_validation_ce=best_ce)))
    info = dict(status='completed', source_key=key, selected_epoch=best_epoch,
        source_validation=evaluate_source(model, task), completed_updates=updates,
        fit_graphs=len(task.fit_ids), query_graphs_used=0, target_labels_used=False,
        artifacts={name: sha256(folder/name) for name in
                   ('best.pt', 'history.jsonl', 'config.json', 'data_roles.json', 'provenance.json')})
    write_json(folder/'result.json', info)
    return model, dict(info, cache_hit=False), folder
