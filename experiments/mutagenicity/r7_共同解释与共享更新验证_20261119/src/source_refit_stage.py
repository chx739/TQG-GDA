"""Schedule registered source-only checkpoints before real adaptation."""
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import multiprocessing


def source_keys(tasks):
    return sorted({(setting, source, seed) for setting, source, _, seed in tasks})


def _train(task, plan, pooling, root, repo):
    from .source_refit_inputs import train_source

    setting, source, seed = task
    train_source(plan, setting, source, seed, pooling=pooling, output_root=root, repository=repo)
    return f'{setting}-M{source}-seed-{seed}-{pooling}'


def prepare(plan, tasks, root, workers, repo, *, reuse=None):
    if 'source_refit' not in plan:
        return {}
    refit = plan['source_refit']
    if (set(refit) != {'pooling', 'recipe'} or refit['pooling'] not in ('mean', 'sum')
            or refit['recipe'] != 'r5_frozen_role_source_refit_v1'):
        raise ValueError('Unknown registered source-refit recipe')
    roots = dict(reuse or {})
    keys = source_keys(tasks)
    if not roots.keys() <= set(keys):
        raise ValueError('A prior source archive has unregistered checkpoint keys')
    new = [key for key in keys if key not in roots]
    if new:
        destination = root / 'source-refit'
        destination.mkdir(exist_ok=False)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            trained = pool.map(_train, new, repeat(plan), repeat(refit['pooling']),
                               repeat(destination), repeat(repo))
            for task in trained:
                print(f'fit frozen source checkpoint {task}', flush=True)
        roots.update({key: destination for key in new})
    return roots


def local_files(root):
    return sorted(str(path.relative_to(root)) for path in (root / 'source-refit').rglob('*') if path.is_file())


def load(plan, task, root, repo, *, include_tsba=False):
    from .source_refit_inputs import load_inputs

    return load_inputs(plan, *task, pooling=plan['source_refit']['pooling'],
                       output_root=root, repository=repo, include_tsba=include_tsba)
