"""Small source-only development runner; no target input or target evaluator."""
import argparse
import copy
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from torch.nn import functional as F
from torch_geometric.loader import DataLoader

from .data import SourceGraphs, sha256
from .models import SourceClassifier
from .initialization import InitializedClassifier, fixed_scale_pairs, verify_conversion

# Package root, so the design/code fingerprints do not depend on the caller's
# working directory (run_experiment.py chdirs here; a direct `python -m
# sega_gda.train_source` or an import from elsewhere previously crashed).
PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def atomic_save(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def seed_everything(seed, threads):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0., 0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite evaluation logits')
        loss += float(F.cross_entropy(logits, batch.y, reduction='sum'))
        correct += int((logits.argmax(-1) == batch.y).sum())
        count += batch.num_graphs
    return dict(ce=loss / count, accuracy=correct / count, graphs=count)


def load_source_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['kind'] != 'sega_s1_source' or state['schema_version'] != 1:
        raise ValueError('Not an S1 source checkpoint')
    model = SourceClassifier(state['encoder_spec'], len(state['class_labels'])).double().to(device)
    model.load_state_dict(state['model'])
    return model, state


def load_initialized_checkpoint(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['kind'] != 'sega_s1_identity_initialization' or state['schema_version'] != 1:
        raise ValueError('Not an S1 initialized checkpoint')
    source = SourceClassifier(state['encoder_spec'], len(state['class_labels'])).double().to(device)
    source.load_state_dict(state['source_model'])
    initialized = InitializedClassifier(source, state['scale2'].to(device),
                                        state['temperature'], state['num_regions']).to(device)
    initialized.load_state_dict(state['model'])
    initialized.head.geometry.assert_identity()
    return initialized, state


def run(config, output_dir, resume=None):
    config = copy.deepcopy(config)
    if config.get('experiment_kind') != 'development_source_only':
        raise ValueError('This S1 runner is explicitly restricted to development source-only runs')
    if config.get('dtype') != 'float64':
        raise ValueError('S1 debug runner locks float64 for strict conversion verification')
    if min(config['epochs'], config['batch_size'], config['num_threads']) < 1:
        raise ValueError('Epochs, batch size and CPU threads must be positive')
    if config['lr'] <= 0 or not np.isfinite(config['lr']):
        raise ValueError('Invalid learning rate')
    if config['selection'] != 'source_validation_ce_min_earliest':
        raise ValueError('S1 debug model selection rule must be fixed before running')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config['seed'], config['num_threads'])
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; no silent device change')
    dataset = SourceGraphs(config['data_cache'], config['split_manifest'], config['use_node_attributes'])
    preprocessing = dataset.fit_preprocessor()
    train_graphs = dataset.materialize(dataset.train_ids, preprocessing)
    val_graphs = dataset.materialize(dataset.val_ids, preprocessing)
    generator = torch.Generator().manual_seed(config['seed'] + 1000)
    train = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=True, generator=generator)
    # Even a nonshuffling DataLoader draws an iterator base seed. Isolate it
    # from training RNG so diagnostics after resume cannot perturb training.
    train_eval = DataLoader(train_graphs, batch_size=config['batch_size'], shuffle=False,
                            generator=torch.Generator().manual_seed(config['seed'] + 2000))
    val = DataLoader(val_graphs, batch_size=config['batch_size'], shuffle=False,
                     generator=torch.Generator().manual_seed(config['seed'] + 3000))
    all_source = DataLoader(train_graphs + val_graphs, batch_size=config['batch_size'], shuffle=False,
                            generator=torch.Generator().manual_seed(config['seed'] + 4000))
    spec = dict(input_dim=dataset.input_dim, hidden_dim=config['hidden_dim'],
                num_layers=config['num_layers'], edge_dim=dataset.edge_dim, readout=config['readout'])
    model = SourceClassifier(spec, len(dataset.classes)).double().to(device)
    # CE is the ONLY learning objective; no implicit weight decay or target losses.
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.)
    identity = dict(split_sha256=sha256(config['split_manifest']),
                    cache_sha256=sha256(config['data_cache']), source_ids=dataset.manifest,
                    target_inputs_accessed=False, target_labels_accessed=False,
                    preprocessing_fit_ids=dataset.train_ids)
    config['encoder'] = spec
    start_epoch, best_ce, best_epoch, selected = 0, float('inf'), 0, None
    if resume:
        loaded_model, state = load_source_checkpoint(resume, device)
        ignored = {'epochs', 'data_cache', 'split_manifest'}  # Paths may relocate; immutable cache/split hashes and identities are checked below.
        if ({k: v for k, v in state['config'].items() if k not in ignored}
                != {k: v for k, v in config.items() if k not in ignored}):
            raise ValueError('Resume changed configuration beyond the debug epoch budget')
        if state['identity'] != identity:
            raise ValueError('Resume changed source data or split')
        model.load_state_dict(loaded_model.state_dict())
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, best_ce, best_epoch = state['epoch'], state['best_ce'], state['best_epoch']
        selected = state.get('selected_checkpoint', state)
        restore_rng(state['rng'])
        generator.set_state(state['loader_rng'])
        if config['epochs'] <= start_epoch:
            raise ValueError('Resume budget must extend beyond the stored epoch')
    write_json(output / 'config.json', config)
    write_json(output / 'data_manifest.json', identity)
    environment = dict(python=platform.python_version(), torch=torch.__version__,
                       numpy=np.__version__, pyg=torch_geometric.__version__, device=str(device),
                       num_threads=torch.get_num_threads(), deterministic=True,
                       design_sha256=sha256(PACKAGE_ROOT / 'docs' / 'SEGA_GDA_Self_Evolving_Local_Geometry_Design_CN.md'),
                       code_sha256={p.relative_to(PACKAGE_ROOT).as_posix(): sha256(p)
                                    for p in sorted((PACKAGE_ROOT / 'sega_gda').glob('*.py'))})
    write_json(output / 'environment.json', environment)
    initial_train = evaluate(model, train_eval, device)
    started = time.monotonic()
    fields = ['epoch', 'train_ce', 'train_accuracy', 'validation_ce', 'validation_accuracy',
              'optimizer_ce', 'selected', 'elapsed_seconds']
    with (output / 'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(start_epoch + 1, config['epochs'] + 1):
            model.train()
            total_loss, seen = 0., 0
            for batch in train:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(batch), batch.y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite source CE; no skipped batch')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite source gradient; update not applied')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite source parameter')
                total_loss += float(loss.detach()) * batch.num_graphs
                seen += batch.num_graphs
            tr, va = evaluate(model, train_eval, device), evaluate(model, val, device)
            improved = va['ce'] < best_ce  # Strict comparison preserves earliest exact tie.
            if improved:
                best_ce, best_epoch = va['ce'], epoch
            current = dict(schema_version=1, kind='sega_s1_source', epoch=epoch,
                           model=copy.deepcopy(model.state_dict()), optimizer=copy.deepcopy(optimizer.state_dict()),
                           config=config, encoder_spec=spec, class_labels=dataset.classes,
                           preprocessing=preprocessing, identity=identity, best_ce=best_ce,
                           best_epoch=best_epoch, rng=rng_state(), loader_rng=generator.get_state())
            if improved:
                selected = copy.deepcopy(current)
            atomic_save(output / 'best.pt', selected)
            atomic_save(output / 'last.pt', {**current, 'selected_checkpoint': selected})
            writer.writerow(dict(epoch=epoch, train_ce=tr['ce'], train_accuracy=tr['accuracy'],
                                 validation_ce=va['ce'], validation_accuracy=va['accuracy'],
                                 optimizer_ce=total_loss / seen, selected=int(improved),
                                 elapsed_seconds=time.monotonic() - started))
            stream.flush()
            print(f'epoch={epoch} train_ce={tr["ce"]:.6f} source_val_ce={va["ce"]:.6f} '
                  f'source_val_acc={va["accuracy"]:.4f} selected={improved}', flush=True)
    model, selected_state = load_source_checkpoint(output / 'best.pt', device)
    model.eval()
    with torch.no_grad():
        coordinates = torch.cat([model.encoder(b.to(device)).cpu() for b in train_eval])
    pairs, pair_ids, scale2 = fixed_scale_pairs(coordinates, dataset.train_ids,
                                               config['scale_pairs'], config['scale_seed'])
    converted = InitializedClassifier(model, scale2.to(device), config['temperature'],
                                      config['identity_regions']).to(device).eval()
    verification = verify_conversion(model, converted, all_source, device)
    initialized = dict(schema_version=1, kind='sega_s1_identity_initialization',
                       model=converted.state_dict(), source_model=model.state_dict(), encoder_spec=spec,
                       class_labels=dataset.classes, preprocessing=preprocessing, identity=identity,
                       scale_pairs=pairs, scale_pair_graph_ids=pair_ids, scale2=scale2,
                       temperature=config['temperature'], num_regions=config['identity_regions'],
                       source_epoch=best_epoch, source_checkpoint_sha256=sha256(output / 'best.pt'),
                       distance='exact_identity_geodesic_squared', local_coverage_fitted=False,
                       classifier_optimizer_state=None, verification=verification)
    atomic_save(output / 'initialization.pt', initialized)
    restored, _ = load_initialized_checkpoint(output / 'initialization.pt', device)
    restoration = verify_conversion(model, restored, all_source, device)
    result = dict(stage='S1', status='PASSED', experiment_kind=config['experiment_kind'],
                  source_train_graphs=len(train_graphs), source_validation_graphs=len(val_graphs),
                  source_train_class_counts={str(c): sum(int(g.y)==i for g in train_graphs)
                                             for i, c in enumerate(dataset.classes)},
                  source_validation_class_counts={str(c): sum(int(g.y)==i for g in val_graphs)
                                                  for i, c in enumerate(dataset.classes)},
                  initial_train=initial_train, selected_epoch=best_epoch,
                  selected_train=evaluate(model, train_eval, device),
                  selected_source_validation=evaluate(model, val, device),
                  scale2=float(scale2), equivalence=verification, restored_equivalence=restoration,
                  checkpoint=str((output / 'initialization.pt').resolve()),
                  elapsed_seconds=time.monotonic()-started, target_evaluated=False,
                  formal_paper_result=False, resumed_from=str(resume) if resume else None)
    write_json(output / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', default=None, help='Full last.pt at a completed epoch boundary')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), args.output, args.resume)


if __name__ == '__main__':
    main()
