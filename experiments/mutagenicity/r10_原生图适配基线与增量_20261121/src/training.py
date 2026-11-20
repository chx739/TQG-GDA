"""Thin JS graph-mode adapter around the exact upstream GRADE classes."""
import hashlib
import random
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .upstream.grade_base import GRADEBase
from .upstream.reverse_layer import GradReverse


def seed_process(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_model(model, in_dim, classes, device):
    return GRADEBase(in_dim, model['hidden'], classes, num_layers=model['layers'],
                     dropout=model['dropout'], disc=model['disc'], mode=model['mode']).to(device)


def state_hash(model):
    check = hashlib.sha256()
    for name, value in model.state_dict().items():
        x = value.detach().cpu().contiguous()
        check.update(f'{name}|{x.dtype}|{tuple(x.shape)}\n'.encode())
        check.update(x.numpy().tobytes())
    return check.hexdigest()


def optimizer_for(model, configuration):
    return torch.optim.Adam(model.parameters(), lr=configuration['learning_rate'],
                            weight_decay=configuration['weight_decay'])


def alpha_at(epoch, total):
    return float(2 / (1 + np.exp(-10 * epoch / total)) - 1)


def js_forward(model, source, target, alpha, weight):
    source_logits, source_features = model(source)
    if target is None:
        ce = F.nll_loss(F.log_softmax(source_logits, dim=1), source.y)
        return dict(loss=ce, source_ce=ce, domain_ce=None, source_logits=source_logits)
    if 'y' in target:
        raise PermissionError('GRADE target adaptation batch contains a label field')
    target_logits, target_features = model(target)
    ce = F.nll_loss(F.log_softmax(source_logits, dim=1), source.y)
    domain_logits = model.discriminator(GradReverse.apply(
        torch.cat([source_features, target_features], dim=0), alpha))
    domain_y = torch.tensor([0] * len(source) + [1] * len(target), dtype=torch.long, device=source_logits.device)
    domain_ce = model.criterion(domain_logits, domain_y)
    return dict(loss=ce + domain_ce*weight, source_ce=ce, domain_ce=domain_ce,
                source_logits=source_logits, target_logits=target_logits, domain_logits=domain_logits)


def dataset(graphs, ids, labels=None):
    if labels is not None and labels.shape != (len(ids),):
        raise ValueError('Source labels differ from the registered source IDs')
    result = []
    for index, gid in enumerate(ids):
        record = graphs[gid]
        if set(record) != {'x', 'edge_index'}:
            raise PermissionError('Unexpected field in graph record')
        graph = Data(x=record['x'], edge_index=record['edge_index'], graph_id=torch.tensor([gid]))
        if labels is not None:
            graph.y = labels[index:index+1]
        result.append(graph)
    return result


def _order(batch, wanted):
    observed = batch.graph_id.cpu().numpy()
    if not np.array_equal(np.sort(observed), np.asarray(sorted(wanted))):
        raise ValueError('Full-batch loader omitted, duplicated, or changed IDs')
    return hashlib.sha256(observed.astype('<i8').tobytes()).hexdigest()


def fit_model(source, target, source_ids, adapt_ids, plan, seed, in_dim, classes, log):
    cfg, device = plan['model'], plan['execution']['device']
    if cfg['batch_size'] != 0 or cfg['disc'] != 'JS' or cfg['mode'] != 'graph':
        raise ValueError('R10 uses the registered native full-batch graph JS path')
    seed_process(seed)
    # The original fit constructs native shuffled DataLoaders before model
    # construction and uses the global Torch RNG for their iterators.
    s_loader = DataLoader(source, batch_size=len(source), shuffle=True)
    a_loader = None if target is None else DataLoader(target, batch_size=len(target), shuffle=True)
    model = new_model(cfg, in_dim, classes, device)
    initial_sha = state_hash(model)
    optimizer = optimizer_for(model, cfg)
    start = time.perf_counter()
    history = []
    log(dict(event='fit_started', initial_parameter_sha256=initial_sha,
             source_graphs=len(source), target_adapt_graphs=len(adapt_ids),
             source_nodes=sum(x.num_nodes for x in source),
             adapt_nodes=sum(x.num_nodes for x in target) if target else 0,
             source_edges=sum(x.num_edges for x in source),
             adapt_edges=sum(x.num_edges for x in target) if target else 0))
    for epoch in range(cfg['epochs']):
        epoch_start = time.perf_counter()
        alpha = alpha_at(epoch, cfg['epochs'])
        updates = []
        batches = zip(s_loader, a_loader) if a_loader is not None else ((s, None) for s in s_loader)
        for s, a in batches:
            model.train()
            source_order = _order(s, source_ids)
            target_order = _order(a, adapt_ids) if a is not None else None
            s = s.to(device)
            if a is not None:
                a = a.to(device)
            values = js_forward(model, s, a, alpha, cfg['domain_weight'])
            if not bool(torch.isfinite(values['loss'])):
                raise FloatingPointError(f'Nonfinite GRADE loss in epoch {epoch+1}')
            optimizer.zero_grad()
            values['loss'].backward()
            optimizer.step()
            if device.startswith('cuda'):
                torch.cuda.synchronize(device)
            updates.append(dict(source_graphs=len(s), target_adapt_graphs=len(a) if a is not None else 0,
                                source_order_sha256=source_order, adapt_order_sha256=target_order,
                                source_ce=float(values['source_ce'].detach()),
                                domain_ce=float(values['domain_ce'].detach()) if a is not None else None,
                                loss=float(values['loss'].detach()),
                                source_train_accuracy=float((values['source_logits'].argmax(1)==s.y).float().mean())))
        if len(updates) != 1:
            raise ValueError(f'Epoch {epoch+1} made {len(updates)} rather than one full-batch update')
        row = dict(event='epoch', epoch=epoch+1, alpha=alpha, updates=1,
                   duration_seconds=time.perf_counter()-epoch_start, elapsed_seconds=time.perf_counter()-start,
                   peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                   **updates[0])
        history.append(row)
        log(row)
    return model, initial_sha, history


@torch.inference_mode()
def infer(model, graphs, ids, plan):
    cfg = plan['execution']
    loader = DataLoader(dataset(graphs, ids), batch_size=cfg['inference_batch_size'], shuffle=False)
    model.eval()
    outputs, indices = [], []
    for batch in loader:
        if 'y' in batch:
            raise PermissionError('H inference batch contains graph labels')
        indices.extend(batch.graph_id.tolist())
        z, _ = model(batch.to(cfg['device']))
        outputs.append(z.cpu())
    if indices != ids:
        raise ValueError('H inference order differs from the registered graph IDs')
    logits = torch.cat(outputs)
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Nonfinite epoch-200 logits')
    return dict(ids=torch.tensor(indices), logits=logits, decisions=logits.argmax(1))
