"""SEGA-style end-to-end TQG: data/pretrain -> query/adapt -> GNN prediction.

Target labels have no interface here. Evaluation remains a separate, explicitly
scoped research workflow. Completed results are reused only after hash checks.
"""
import json
from pathlib import Path
import time

from .adapt import adapt_model
from .data import load_task
from .metrics import evaluate_source
from .predict import model_checkpoint, predict_task
from .pretrain import pretrain_model
from .provenance import code_identity, runtime_identity, task_key, verify_artifacts
from .utils import atomic_save, seed_everything, sha256, write_json


def run_task(config):
    started = time.perf_counter()
    runtime, code = runtime_identity(config), code_identity()
    seed_everything(config.seed, config.num_threads, config.deterministic)
    task = load_task(config)
    key = task_key(config, task, code, runtime)
    folder = Path(config.out_dir)/'tasks'/f'{config.version}_{key}'
    if folder.exists():
        if not (folder/'result.json').is_file():
            raise ValueError('Incomplete task directory; retain it for inspection and use a new out_dir')
        result = json.loads((folder/'result.json').read_text())
        if result.get('task_key') != key or result.get('status') != 'completed':
            raise ValueError('Task identity mismatch')
        verify_artifacts(folder, result['artifacts'])
        return dict(result, reused=True, output=str(folder.resolve()))
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'config.json', config.to_dict())
    write_json(folder/'data_roles.json', task.identity)
    write_json(folder/'provenance.json', dict(code=code, runtime=runtime))
    # Stage A: source-only initialization/cache, shared by every target and version.
    model, pretrain, cache_folder = pretrain_model(config, task, code, runtime)
    before = predict_task(model, task)
    # Stages B/C: fresh snapshot/query and one joint update, repeated at new theta.
    adapt_model(model, task, config, folder/'history.jsonl')
    # Stage D: final-budget GNN only. No target-driven model selection.
    after = predict_task(model, task)
    atomic_save(folder/'model.pt', model_checkpoint(model, task, config, stage='adapted',
        task_key=key, source_key=pretrain['source_key'], completed_steps=config.adapt.steps))
    write_json(folder/'predictions.json', dict(graph_ids=list(task.target_ids),
        class_labels=task.classes, source_probabilities=before.tolist(),
        adapted_probabilities=after.tolist(), target_labels_used=False))
    files = ('config.json', 'data_roles.json', 'provenance.json', 'history.jsonl', 'model.pt', 'predictions.json')
    result = dict(status='completed', version=config.version, task_key=key, seed=config.seed,
        dataset=config.dataset, target_index_sha256=task.identity['target_index_sha256'],
        source_split_sha256=task.identity['source_split_sha256'], pretrain=pretrain,
        source_cache=str(cache_folder.resolve()), completed_steps=config.adapt.steps,
        source_validation=evaluate_source(model, task),
        target_predictions_changed=int((before.argmax(-1) != after.argmax(-1)).sum()),
        target_labels_used=False, target_evaluated=False, formal_paper_result=False,
        seconds=time.perf_counter()-started, artifacts={name: sha256(folder/name) for name in files})
    write_json(folder/'result.json', result)
    return dict(result, reused=False, output=str(folder.resolve()))
