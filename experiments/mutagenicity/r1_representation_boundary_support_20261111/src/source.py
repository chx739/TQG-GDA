"""R1 source initialization, identical training mathematics to tsba.pretrain.

The research runner retains every epoch and optimizer state. It never trains
score or adapts the classifier. Diagnostic/target labels are not inputs.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch, Data

from tsba.classifier import Coordinates, ResidualClassifier
from tsba.config import Config
from tsba.data import GraphPool, TaskData
from tsba.encoder import GraphEncoder
from tsba.pipeline import source_identity
from tsba.pretrain import SourceModel, encode, graph_batches, source_metrics
from tsba.utils import finite, set_seed
from experiments.common.io import save_tensor, write_json


def train_source(cfg, data, out, log=print):
    """Keep the production order/RNG/selection; add append-only checkpoints."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    dim, edge_dim, classes = data.validate()
    set_seed(cfg.seed)
    encoder = GraphEncoder(dim, edge_dim, cfg.hidden, cfg.num_layers, cfg.emb_dim, cfg.dropout, cfg.encoder)
    model = SourceModel(encoder, classes).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.source_lr)
    rng = torch.Generator().manual_seed(cfg.seed + 1)
    best, state, selected = float('inf'), None, None
    history = []
    (out / 'epochs').mkdir()
    for epoch in range(cfg.source_epochs):
        model.train()
        order = torch.randperm(len(data.source_train.graphs), generator=rng).tolist()
        total = 0.0
        for ids in graph_batches(data.source_train.graphs, cfg.source_batch, order):
            batch = Batch.from_data_list([data.source_train.graphs[i] for i in ids]).to(cfg.device)
            if batch.num_nodes < 2:
                raise ValueError('source BatchNorm requires >=2 nodes per training batch')
            labels = data.source_train.labels[ids].to(cfg.device)
            loss = F.cross_entropy(model(batch), labels)
            finite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            finite(*(p.grad for p in model.parameters() if p.grad is not None))
            opt.step()
            finite(*(p for p in model.parameters()))
            total += float(loss.detach()) * len(ids)
        validation = source_metrics(model, data.source_validation, cfg.source_batch, cfg.device)
        row = dict(epoch=epoch, train_ce=total / len(order), source_validation=validation)
        history.append(row)
        if validation is None or validation['ce'] < best:
            best = validation['ce'] if validation is not None else total / len(order)
            selected = epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        save_tensor(out / 'epochs' / f'epoch{epoch:03d}.pt', dict(
            epoch=epoch, state=model.state_dict(), optimizer=opt.state_dict(),
            order=[data.source_train.ids[i] for i in order], metrics=row,
            torch_rng=torch.get_rng_state(), permutation_rng=rng.get_state()))
        if epoch % 10 == 0 or epoch == cfg.source_epochs - 1:
            log(f'source seed={cfg.seed} epoch={epoch} train_ce={row["train_ce"]:.6g} validation_ce={validation["ce"]:.6g}')
    model.load_state_dict(state)
    model.encoder.freeze()
    model.eval().zero_grad(set_to_none=True)
    info = dict(selected_epoch=selected, selection='source_validation_ce' if data.source_validation.graphs
                else 'last_predeclared_epoch', history=history)
    source_key = source_identity(cfg, data.fingerprints())
    save_tensor(out / 'source.pt', dict(format='tsba-source-v1', source_identity=source_key,
        encoder_spec=model.encoder.spec, state=model.state_dict(), info=info))
    write_json(out / 'history.json', info)
    return model, info


def descriptives(h, size, ids):
    from scipy.stats import spearmanr
    values = h[ids].norm(dim=1).numpy()
    result = {}
    for col, name in enumerate(('nodes', 'undirected_edges')):
        v = size[ids, col].numpy()
        corr = float(spearmanr(values, v).statistic)
        result[name] = dict(norm_spearman=corr if corr == corr else None,
            log1p_quantiles=torch.quantile(size[ids, col], torch.tensor([0., .25, .5, .75, 1.], dtype=size.dtype)).tolist())
    result['embedding_norm_quantiles'] = torch.quantile(h[ids].norm(dim=1), torch.tensor([0., .25, .5, .75, 1.], dtype=h.dtype)).tolist()
    return result


def run_source(root, seed, plan, log):
    import json
    from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.data import LabelBroker
    root = Path(root)
    roles = json.loads((root / 'roles.json').read_text())
    sr = json.loads((root / 'roles' / f'seed{seed}.json').read_text())
    records = torch.load(root / 'inputs' / 'graphs.pt', weights_only=True)
    graphs = [Data(**r) for r in records]
    broker = LabelBroker(root, plan)
    def pool(ids, role, labels=True):
        y = broker.read(ids, role=role, purpose=f'source seed={seed}', seed=seed) if labels else None
        return GraphPool([graphs[i] for i in ids], [str(i) for i in ids], y)
    fit, val = sr['fit_ids'], sr['validation_ids']
    target_ids = roles['targets']['1']['ids']
    data = TaskData(pool(fit, 'source_fit'), pool(val, 'source_validation'),
                    pool(target_ids, 'target_inputs', False), 'transductive')
    p = plan['source']
    cfg = Config(seed=seed, encoder=p['encoder'], hidden=p['hidden'], num_layers=p['layers'],
        emb_dim=p['embedding_dim'], dropout=p['dropout'], source_epochs=p['epochs'],
        source_lr=p['lr'], source_batch=p['batch'], device='cpu', adaptation_dtype='float64')
    out = root / 'source' / f'seed{seed}'
    model, info = train_source(cfg, data, out, log)
    frozen_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    # All graph inputs are public; no domain2 class labels or evaluation used.
    h = encode(model.encoder, graphs, cfg.source_batch, cfg.device).double()
    coordinates = Coordinates.fit(h[fit], cfg.coordinate_eps)
    x = coordinates(h)
    head = ResidualClassifier.from_source(model.head, coordinates, 16, seed).eval()
    with torch.no_grad():
        logits = head(x)
        prob = logits.softmax(-1)
        rawprob = (h @ model.head.weight.double().T + model.head.bias.double()).softmax(-1)
        source_float32_prob = model.head(h.float()).softmax(-1).double()
    torch.testing.assert_close(prob, rawprob, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(prob, source_float32_prob, rtol=1e-5, atol=1e-6)
    for k, v in model.encoder.state_dict().items():
        torch.testing.assert_close(v, frozen_before[k], rtol=0, atol=0)
    assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    # A held-out input batch partition check: never touches any held-out label.
    audit_ids = target_ids[:16]
    split_h = encode(model.encoder, [graphs[i] for i in audit_ids], 7, cfg.device).double()
    partition_error = float((split_h - h[audit_ids]).abs().max())
    torch.testing.assert_close(split_h, h[audit_ids], rtol=1e-5, atol=1e-6)
    save_tensor(out / 'frozen.pt', dict(x=x, h=h, mean=coordinates.mean, scale=coordinates.scale,
                initial_head=head.state_dict(), ids=torch.arange(len(graphs))))
    save_tensor(out / 'predictions.pt', dict(probabilities=prob, log_probabilities=logits.log_softmax(-1), ids=torch.arange(len(graphs))))
    size = torch.load(root / 'inputs' / 'size.pt', weights_only=True)
    domains = {'0': roles['source']['domain_ids'], **{d: r['ids'] for d, r in roles['targets'].items()}}
    write_json(out / 'descriptives.json', {d: descriptives(h, size, ids) for d, ids in domains.items()})
    write_json(out / 'checks.json', dict(frozen_parameters_and_bn=True, initial_probability_max_error=float((prob-rawprob).abs().max()),
              source_float32_probability_max_error=float((prob-source_float32_prob).abs().max()),
              batch_partition_max_error=partition_error, coordinate_fit_ids=fit, source_diagnostics_used=False,
              target_labels_used=False, score_trained=False, classifier_adapted=False))
    log(f'source seed={seed} complete selected_epoch={info["selected_epoch"]} coordinates_scale={float(coordinates.scale):.6g}')
    return info
