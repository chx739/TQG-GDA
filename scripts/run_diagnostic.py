"""E1 oracle mechanism diagnosis on a RESERVED target development subset.

This script deliberately reads diagnostic target labels, but never final target
labels. Training does not import this script or tqg_gda.diagnostics.
"""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch_geometric.data.separate import separate

from tqg_gda.data import GraphDomainTask
from tqg_gda.diagnostics import reserve_target_ids, measure_feedbacks, summarize_blocks
from tqg_gda.feedback import ProbeConfig, build_feedback
from tqg_gda.training import load_model, environment
from tqg_gda.utils import seed_everything, sha256, fingerprint, write_json


def read_diagnostic_labels(task, ids, allowed_ids):
    """Evaluation-only label reader; the final holdout is explicitly forbidden."""
    if not ids or not set(ids) <= set(allowed_ids) or not set(allowed_ids) <= set(task.target_ids):
        raise PermissionError('Only reserved diagnostic target labels may be read')
    labels = []
    for i in ids:
        graph = separate(task._raw.__class__, task._raw, i, task.slices, decrement=False)
        y = graph.y
        if y is None or y.numel() != 1 or not torch.isfinite(y).all() or not torch.equal(y, y.long()):
            raise ValueError('Expected one integer diagnostic target label')
        if int(y.item()) not in task.class_map:
            raise ValueError('Diagnostic target label is outside the closed-set source classes')
        labels.append(task.class_map[int(y.item())])
    return torch.tensor(labels, dtype=torch.long)


def validate_plan(plan):
    for key in ('blocks', 'source_query_count', 'target_query_count', 'target_eval_count'):
        v = plan[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(key + ' must be a positive integer')
    modes = plan['directions']
    if not {'target', 'random'} <= set(modes) or len(modes) != len(set(modes)):
        raise ValueError('Require unique methods including target and random')
    for mode in modes:
        ProbeConfig(**plan['probes'], direction=mode)
    if (not plan['step_fractions'] or len(set(plan['step_fractions'])) != len(plan['step_fractions'])
            or any(not math.isfinite(x) or x <= 0 for x in plan['step_fractions'])
            or not math.isfinite(plan['gradient_epsilon']) or plan['gradient_epsilon'] <= 0):
        raise ValueError('Require distinct positive step fractions and a positive gradient epsilon')


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-path', required=True)
    p.add_argument('--source-split', default=str(ROOT / 'data/splits/PROTEINS_idx_0_source.json'))
    p.add_argument('--target-index', default=str(ROOT / 'idx/idx_PROTEINS_1.npy'))
    p.add_argument('--plan', default=str(ROOT / 'configs/diagnostics/e1_target_vs_random.json'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    validate_plan(plan)
    model, state = load_model(args.checkpoint, args.device)
    config = state['config']
    if config['seed'] not in plan['source_seeds']:
        raise ValueError('Source model seed was not registered in the experiment plan')
    if (config['warmup']['source_pool'] != 'fit' or state['identity'].get('target_ids')
            or state['kind'] not in ('tqg_source', 'tqg_predictor')):
        raise ValueError('E1 requires a source fit-only warmup checkpoint, before target adaptation')
    seed_everything(config['seed'], config['num_threads'], config['deterministic'])
    task = GraphDomainTask(args.data_path, args.source_split, target_index=args.target_index,
        target_policy='transductive-unlabeled', fit_fraction=config['split']['fit_fraction'],
        split_seed=config['split']['seed'], use_node_attributes=config['use_node_attributes'],
        preprocess_pool='fit')
    source_identity, current = dict(state['identity']), dict(task.identity)
    for identity in (source_identity, current):
        for key in ('target_ids', 'target_index_sha256', 'target_policy', 'fingerprint'):
            identity.pop(key, None)
    if source_identity != current:
        raise ValueError('Source roles/cache/preprocessing differ from the checkpoint')
    diagnostic, final = reserve_target_ids(task.target_ids, plan['diagnostic_fraction'], plan['target_split_seed'])
    if len(diagnostic) < plan['target_query_count'] + plan['target_eval_count']:
        raise ValueError('Diagnostic pool is too small for DISJOINT query/evaluation batches; no silent capping')
    if len(task.query_ids) < plan['source_query_count']:
        raise ValueError('Source query pool is too small; no silent capping')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    parameter = next(model.parameters())
    prototype = ProbeConfig(**plan['probes'])
    protocol = dict(experiment=plan['experiment'], plan=plan, seed=config['seed'],
        checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256(args.checkpoint),
        plan_sha256=sha256(args.plan), data_identity=task.identity,
        diagnostic_target_ids=list(diagnostic), final_target_ids=list(final),
        final_target_labels_read=False, target_label_policy='reserved-development-oracle-diagnosis',
        used_to_train_or_select_checkpoint=False, may_inform_future_research_decisions=True,
        environment=environment(), script_sha256=sha256(__file__),
        frozen_protocol_before_first_diagnostic_label=True)
    comparison_contract = dict(plan_sha256=protocol['plan_sha256'], task_fingerprint=task.identity['fingerprint'],
        source_config={key: config[key] for key in ('dtype', 'num_threads', 'deterministic', 'model', 'split',
                                                    'use_node_attributes', 'warmup')},
        code_sha256=protocol['environment']['code_sha256'], script_sha256=protocol['script_sha256'],
        torch=protocol['environment']['torch'], pyg=protocol['environment']['pyg'])
    protocol['comparison_fingerprint'] = fingerprint(comparison_contract)
    protocol['fingerprint'] = fingerprint(protocol)
    write_json(output / 'protocol.json', protocol)  # Written BEFORE accessing target labels.
    np.save(output / 'target_diagnostic.npy', np.asarray(diagnostic, dtype=np.int64))
    np.save(output / 'target_final.npy', np.asarray(final, dtype=np.int64))
    original = {n: t.detach().clone() for n, t in model.state_dict().items()}
    blocks = []
    started = time.perf_counter()
    with (output / 'blocks.jsonl').open('x') as stream:
        for block in range(plan['blocks']):
            # Identical batches across methods AND across source checkpoint seeds.
            sampler = torch.Generator().manual_seed(plan['sampling_seed'] + block)
            source, source_ids = task.sample('query', plan['source_query_count'], sampler,
                                            parameter.device, parameter.dtype)
            order = torch.randperm(len(diagnostic), generator=sampler).tolist()
            nq, ne = plan['target_query_count'], plan['target_eval_count']
            query_ids = tuple(diagnostic[i] for i in order[:nq])
            eval_ids = tuple(diagnostic[i] for i in order[nq:nq + ne])
            query = task.batch(query_ids, 'target', parameter.device, parameter.dtype)
            evaluation = task.batch(eval_ids, 'target', parameter.device, parameter.dtype)
            modes = plan['directions']
            offset = block % len(modes)
            execution_order = modes[offset:] + modes[:offset]
            feedbacks, timings = {}, {}
            probe_seed = config['seed'] + 100000 + block
            for mode in execution_order:
                synchronize(args.device); tick = time.perf_counter()
                feedbacks[mode] = build_feedback(model, source, source.y, query,
                    replace(prototype, direction=mode), torch.Generator().manual_seed(probe_seed))
                synchronize(args.device); timings[mode] = time.perf_counter() - tick
            # All feedbacks exist before the evaluation labels are read.
            labels = read_diagnostic_labels(task, eval_ids, diagnostic)
            measured = measure_feedbacks(model, query, evaluation, labels, feedbacks,
                                         plan['step_fractions'], plan['gradient_epsilon'])
            measured.update(block=block, source_query_ids=list(source_ids), target_query_ids=list(query_ids),
                target_eval_ids=list(eval_ids), probe_seed=probe_seed,
                execution_order=execution_order, feedback_seconds=timings)
            if any(not torch.equal(t, original[n]) for n, t in model.state_dict().items()):
                raise RuntimeError('Diagnostic unexpectedly changed the model')
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Diagnostic unexpectedly accumulated training gradients')
            blocks.append(measured)
            stream.write(json.dumps(measured, allow_nan=False) + '\n'); stream.flush()
            print(json.dumps(dict(block=block, informative=measured['informative'],
                cosine={n: row['cosine'] for n, row in measured['methods'].items()})), flush=True)
    summary = summarize_blocks(blocks)
    summary.update(experiment=plan['experiment'], seed=config['seed'], protocol_fingerprint=protocol['fingerprint'],
        comparison_fingerprint=protocol['comparison_fingerprint'],
        plan_sha256=protocol['plan_sha256'], task_fingerprint=task.identity['fingerprint'],
        elapsed_seconds=time.perf_counter() - started,
        initial_model_unchanged=True, final_target_labels_read=False, formal_paper_result=False)
    write_json(output / 'summary.json', summary)
    print(json.dumps(dict(output=str(output), target_minus_random=summary['target_minus_random'])), flush=True)


if __name__ == '__main__':
    main()
