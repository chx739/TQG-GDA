"""Standalone post-training target evaluation. Never called by the trainer."""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch_geometric.data.separate import separate

from tqg_gda.data import GraphDomainTask
from tqg_gda.metrics import classification_metrics
from tqg_gda.training import load_model
from tqg_gda.utils import write_json


@torch.no_grad()
def evaluate_target(checkpoint, cache_path, source_split, target_index, device='cpu', batch_size=32):
    model, state = load_model(checkpoint, device)
    config = state['config']
    torch.set_num_threads(config['num_threads'])
    task = GraphDomainTask(cache_path, source_split, target_index=target_index,
        target_policy='transductive-unlabeled', fit_fraction=config['split']['fit_fraction'],
        split_seed=config['split']['seed'], use_node_attributes=config['use_node_attributes'],
        preprocess_pool=config['warmup']['source_pool'])
    for key in ('cache_sha256', 'source_split_sha256', 'source_fit_ids', 'source_query_ids', 'class_labels', 'preprocessing_pool'):
        if state['identity'][key] != task.identity[key]:
            raise ValueError('Evaluation inputs differ from checkpoint: ' + key)
    parameter = next(model.parameters())
    logits, labels = [], []
    for start in range(0, len(task.target_ids), batch_size):
        ids = task.target_ids[start:start + batch_size]
        logits.append(model(task.batch(ids, 'target', parameter.device, parameter.dtype)))
        # Evaluation-only label access. Training never imports either this script
        # or run_diagnostic.py (which reads a reserved development subset only).
        for i in ids:
            graph = separate(task._raw.__class__, task._raw, i, task.slices, decrement=False)
            if graph.y is None or graph.y.numel() != 1 or not torch.isfinite(graph.y).all() or not torch.equal(graph.y, graph.y.long()):
                raise ValueError('Expected one integer target label for standalone evaluation')
            raw_label = int(graph.y.item())
            if raw_label not in task.class_map:
                raise ValueError('Target contains a class outside the declared closed-set source vocabulary')
            labels.append(task.class_map[raw_label])
    z, y = torch.cat(logits).cpu(), torch.tensor(labels)
    result = classification_metrics(z, y)
    result.update(checkpoint=str(Path(checkpoint).resolve()), protocol='post_training_target_evaluation',
                  target_index=str(Path(target_index).resolve()), used_for_training_or_selection=False)
    return result, task.target_ids, z.softmax(-1), y


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--data-path', required=True)
    p.add_argument('--source-split', default=str(ROOT / 'data/splits/PROTEINS_idx_0_source.json'))
    p.add_argument('--target-index', required=True); p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--output', required=True)
    p.add_argument('--predictions')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    result, ids, probabilities, labels = evaluate_target(args.checkpoint, args.data_path, args.source_split,
        args.target_index, args.device, args.batch_size)
    if Path(args.output).exists():
        raise FileExistsError('Evaluation output already exists')
    write_json(args.output, result)
    if args.predictions:
        with Path(args.predictions).open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['graph_id', 'label', 'prediction', *[f'p_{i}' for i in range(probabilities.shape[1])]])
            for i, y, p in zip(ids, labels.tolist(), probabilities.tolist()):
                writer.writerow([i, y, max(range(len(p)), key=p.__getitem__), *p])
    print(result)


if __name__ == '__main__':
    main()
