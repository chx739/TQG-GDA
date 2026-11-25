"""Train the shared source checkpoint, adapt on A and ensemble predictions on H."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import math

from .adaptation import fit_modes
from .artifacts import digest, output_file, read_json, write_json
from .data import load_task
from .metrics import equal_positive_corrections, metrics
from .model import Model, edge_log_probs, source_loss
import torch


@dataclass(frozen=True)
class Settings:
    seed: int = 3405
    hidden: int = 64
    blocks: int = 4
    source_orders: int = 1
    adapt_orders: int = 4
    inference_orders: int = 8
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    joint_weight: float = 1.0
    ridge: float = .01
    max_iter: int = 100
    tolerance: float = 1e-6
    calibrate_prior: bool = True
    bootstrap_samples: int = 0

    def __post_init__(self):
        for name in ('hidden', 'blocks', 'source_orders', 'adapt_orders', 'inference_orders',
                     'epochs', 'batch_size', 'max_iter'):
            val = getattr(self, name)
            if type(val) is not int or val < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.blocks < 2:
            raise ValueError('at least two blocks are required for a final singleton')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0,2^63)')
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 0:
            raise ValueError('invalid bootstrap_samples')
        if type(self.calibrate_prior) is not bool:
            raise ValueError('calibrate_prior must be boolean')
        for name in ('learning_rate', 'joint_weight', 'ridge', 'tolerance'):
            val = getattr(self, name)
            if not math.isfinite(val) or val < 0 or (val == 0 and name in ('learning_rate', 'tolerance')):
                raise ValueError(f'invalid {name}')


def orders(graph, seed, phase, count):
    # Stable per graph, independent of batching, role size and query enumeration.
    key = f'{seed}/{phase}/{graph.id}'.encode()
    generator = torch.Generator().manual_seed(int.from_bytes(sha256(key).digest()[:8], 'big'))
    for _ in range(count):
        yield torch.randperm(len(graph.edges), generator=generator)


def ensemble(traces, eta, components, nu):
    """Average per-order posteriors; also return MC likelihood-weighted sensitivity."""
    log_joint = []
    for trace in traces:
        prior = torch.log_softmax(trace.prior_logits.double() + nu, dim=-1)
        log_joint.append(prior + edge_log_probs(trace, eta, components).double().sum(0))
    joint = torch.stack(log_joint)
    per_order = joint.log_softmax(dim=1)
    log_p = torch.logsumexp(per_order, dim=0) - math.log(len(traces))
    log_p = log_p.log_softmax(0)  # remove roundoff without exponentiating saturated tails
    log_gen = torch.logsumexp(joint, dim=0).log_softmax(dim=0)
    return log_p, log_gen, per_order.exp().std(dim=0, unbiased=False)


def make_model(pre, settings):
    return Model(pre['input_dim'], pre['num_classes'], pre['num_edge_types'],
                 pre['common_types'], hidden=settings.hidden, blocks=settings.blocks,
                 typed_edges=pre['typed_edges'])


@torch.no_grad()
def validation_ce(model, graphs, labels, cfg):
    model.eval()
    nll = []
    for graph, y in zip(graphs, labels, strict=True):
        traces = [model.trace(graph, o) for o in orders(graph, cfg.seed, 'validation', cfg.inference_orders)]
        log_p, _, _ = ensemble(traces, None, None, 0.)
        nll.append(-log_p[y].item())
    return sum(nll) / len(nll)


def train(config, output):
    cfg = Settings(**config.get('settings', {}))
    torch.manual_seed(cfg.seed)
    task = load_task(config['data'], roles=('train', 'val'))
    gs, ys = task.graphs['train'], task.labels['train']
    model = make_model(task.preprocessing, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    mean_m = sum(len(g.edges) for g in gs) / len(gs)
    shuffle = torch.Generator().manual_seed(cfg.seed)
    best_loss, best, history = math.inf, None, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(gs), generator=shuffle)
        for batch in perm.split(cfg.batch_size):
            optimizer.zero_grad(set_to_none=True)
            for i in batch.tolist():
                g, y = gs[i], int(ys[i])
                for order in orders(g, cfg.seed, f'train/{epoch}', cfg.source_orders):
                    trace = model.trace(g, order)
                    loss = source_loss(trace, y, mean_m, lambda_g=cfg.joint_weight)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite source objective')
                    (loss / (len(batch) * cfg.source_orders)).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite source gradient')
            optimizer.step()
        value = validation_ce(model, task.graphs['val'], task.labels['val'], cfg)
        if not math.isfinite(value):
            raise FloatingPointError('nonfinite validation CE')
        history.append(value)
        if value < best_loss:
            best_loss = value
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
    checkpoint = dict(version=1, kind='cela-source', settings=asdict(cfg), model=best,
                      preprocessing=task.preprocessing, dataset=config['data']['dataset'],
                      source_ids=[g.id for role in ('train', 'val') for g in task.graphs[role]],
                      best_epoch=best_epoch, validation_ce=history)
    with output_file(output) as tmp:
        with tmp.open('wb') as stream:
            torch.save(checkpoint, stream)
    return dict(best_epoch=best_epoch, validation_ce=best_loss, graphs=len(gs))


def source_model(source, config):
    checkpoint = torch.load(source, weights_only=True, map_location='cpu')
    if checkpoint['kind'] != 'cela-source' or checkpoint['version'] != 1:
        raise ValueError('incompatible source checkpoint')
    if checkpoint['dataset'] != config['data']['dataset']:
        raise ValueError('different dataset from source checkpoint')
    cfg = Settings(**checkpoint['settings'])
    if cfg != Settings(**config.get('settings', {})):
        raise ValueError('configuration differs from the fitted checkpoint')
    model = make_model(checkpoint['preprocessing'], cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval().requires_grad_(False)
    return checkpoint, model, cfg


@torch.no_grad()
def adapt(config, source, output):
    cp, model, cfg = source_model(source, config)
    pool = load_task(config['data'], roles=('A',), preprocessing=cp['preprocessing']).graphs['A']
    if set(cp['source_ids']).intersection(g.id for g in pool):
        raise ValueError('adaptation and fitted source pools overlap')

    def replay():
        for g in pool:
            yield [model.trace(g, o) for o in orders(g, cfg.seed, 'adapt', cfg.adapt_orders)]

    modes, diagnostics = fit_modes(replay, ridge=cfg.ridge, max_iter=cfg.max_iter, tol=cfg.tolerance,
                                   calibrate_prior=cfg.calibrate_prior, bootstrap_samples=cfg.bootstrap_samples,
                                   seed=cfg.seed)
    models = {}
    prior_logits = torch.stack([model.prior_logits(g).double() for g in pool])
    for name, (eta, components, nu) in modes.items():
        prior = (prior_logits + nu).softmax(1).mean(0)
        models[name] = dict(eta=eta.tolist(), components=components.tolist(), nu=nu.tolist(),
                            class_prior=prior.tolist())
    write_json(output, dict(version=1, kind='cela-adaptation', source_sha256=digest(source),
                            ids=[g.id for g in pool], modes=models, diagnostics=diagnostics))
    return dict(graphs=len(pool), modes=list(models))


@torch.no_grad()
def predict(config, source, adaptation, output):
    cp, model, cfg = source_model(source, config)
    state = read_json(adaptation)
    if (state['kind'], state['version'], state['source_sha256']) != ('cela-adaptation', 1, digest(source)):
        raise ValueError('adaptation does not belong to this source checkpoint')
    declared = read_json(config['data']['manifest'])['roles']['A']
    if set(map(str, declared)) != set(state['ids']):
        raise ValueError('declared adaptation pool differs from the fitted A pool')
    graphs = load_task(config['data'], roles=('H',), preprocessing=cp['preprocessing']).graphs['H']
    if (set(cp['source_ids']) | set(state['ids'])).intersection(g.id for g in graphs):
        raise ValueError('holdout overlaps the fitted data')
    records = []
    for g in graphs:
        traces = [model.trace(g, o) for o in orders(g, cfg.seed, 'predict', cfg.inference_orders)]
        modes = {}
        for name, m in state['modes'].items():
            eta = torch.tensor(m['eta'], dtype=torch.float64)
            components = torch.tensor(m['components'], dtype=torch.long)
            nu = torch.tensor(m['nu'], dtype=torch.float64)
            log_p, log_gen, spread = ensemble(traces, eta, components, nu)
            log_prior = torch.tensor(m['class_prior'],dtype=torch.float64).clamp_min(1e-8).log()
            modes[name] = dict(log_probability=log_p.tolist(), probability=log_p.exp().tolist(),
                               prediction=int(log_p.argmax()), ba_prediction=int((log_p-log_prior).argmax()),
                               likelihood_weighted_probability=log_gen.exp().tolist(), order_sd=spread.tolist())
        records.append(dict(id=g.id, modes=modes))
    write_json(output, dict(version=1, kind='cela-predictions', source_sha256=digest(source),
                            adaptation_sha256=digest(adaptation), class_values=cp['preprocessing']['class_values'],
                            records=records))
    return dict(graphs=len(graphs), modes=list(state['modes']))


def score(predictions, label_file, output):
    frozen = read_json(predictions)
    if (frozen['kind'], frozen['version']) != ('cela-predictions', 1):
        raise ValueError('expected frozen CELA predictions')
    entries, lookup = frozen['records'], read_json(label_file)
    ids = [r['id'] for r in entries]
    if not isinstance(lookup, dict) or len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError('scoring IDs differ from frozen predictions')
    classes = frozen['class_values']
    if any(type(y) is not int or y not in classes for y in lookup.values()):
        raise ValueError('scoring labels must be raw integer classes in the source vocabulary')
    labels = torch.tensor([classes.index(lookup[i]) for i in ids], dtype=torch.long)
    results = {}
    references = {}
    if len(classes) == 2:
        for name in ('zero', 'prior_only'):
            if name in entries[0]['modes']:
                rows = [r['modes'][name] for r in entries]
                references[name] = ([r['probability'] for r in rows], [r['log_probability'] for r in rows])
    comparisons = {name: {} for name in references}
    for name in entries[0]['modes']:
        rows = [r['modes'][name] for r in entries]
        p = torch.tensor([r['probability'] for r in rows], dtype=torch.float64)
        log_p = torch.tensor([r['log_probability'] for r in rows], dtype=torch.float64)
        results[name] = metrics(p, labels, log_probabilities=log_p)
        results[name]['ba_decision_metrics'] = metrics(p, labels, log_probabilities=log_p,
                                                      decisions=[r['ba_prediction'] for r in rows])
        for reference, (baseline, baseline_log) in references.items():
            if name != reference:
                comparisons[reference][name] = equal_positive_corrections(
                    baseline, p, labels, ids=ids, candidate_log_probabilities=log_p,
                    reference_log_probabilities=baseline_log)
    result = dict(predictions_sha256=digest(predictions), graphs=len(ids), modes=results)
    for reference, matched in comparisons.items():
        result[f'equal_positive_vs_{reference}'] = matched
    write_json(output, result)
    return dict(graphs=len(ids), modes=list(results))
