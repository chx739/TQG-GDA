"""Verified local job reuse and checkpoint-specific evaluation metadata."""
import contextlib
import json
from pathlib import Path
import traceback

from .utils import fingerprint, sha256, write_json


def run_verified_job(directory, contract, action):
    """action(directory/artifacts) -> {name: local artifact Path}.

    Completed jobs are reusable only if their full input contract and all output
    hashes match. A marker alone never authorizes reuse. Partial jobs are retained.
    """
    directory = Path(directory).resolve(); marker = directory/'completed.json'
    expected = fingerprint(contract)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved.get('schema_version') != 1 or saved.get('status') != 'completed'
                or saved.get('input_fingerprint') != expected or not saved.get('outputs')):
            raise ValueError('Completed job has a missing or different input identity')
        for output in saved['outputs'].values():
            path = directory/output['path']
            if not path.resolve().is_relative_to(directory) or not path.is_file() or sha256(path) != output['sha256']:
                raise ValueError('Completed job output changed or is missing')
        return {name:directory/info['path'] for name,info in saved['outputs'].items()}
    if directory.exists():
        raise FileExistsError('Partial job retained; inspect it and use a new output directory')
    directory.mkdir(parents=True)
    write_json(directory/'inputs.json', dict(schema_version=1, fingerprint=expected, contract=contract))
    try:
        with (directory/'stdout.log').open('x') as stream, contextlib.redirect_stdout(stream):
            outputs = action(directory/'artifacts')
        if not outputs:
            raise ValueError('Job produced no registered outputs')
        registered = {}
        for name,value in outputs.items():
            path = Path(value).resolve()
            if not path.is_relative_to(directory) or not path.is_file():
                raise ValueError('Job output must be an existing local file')
            registered[name] = dict(path=str(path.relative_to(directory)), sha256=sha256(path))
        payload = dict(schema_version=1, status='completed', input_fingerprint=expected, outputs=registered)
        temporary = directory/'completed.json.tmp'
        write_json(temporary,payload); temporary.replace(marker)
        return {name:directory/info['path'] for name,info in registered.items()}
    except Exception:
        (directory/'failure.txt').write_text(traceback.format_exc())
        raise


def checkpoint_metadata(state, actual_source_ce, training_result=None):
    """The CE supplied here is measured on THIS checkpoint, not its best ancestor."""
    selection = state.get('selection', {})
    if state['kind'] == 'tqg_adaptation':
        step = state['step']
    else:
        step = selection.get('step', 0)
    best_step = selection.get('step', training_result.get('selected_step', 0) if training_result else 0)
    best_ce = selection.get('ce', training_result.get('selected_source_validation_ce') if training_result else state.get('best_ce'))
    return dict(model_step=int(step), selected_step=int(best_step),
        checkpoint_source_validation_ce=float(actual_source_ce), best_source_validation_ce=best_ce,
        warmup_update=state.get('completed_updates'), warmup_epoch=state.get('epoch'))
