"""Execute A0/A1/A2-a frozen-feature diagnostics; no backbone training."""
from workflows._paths import REPO_ROOT
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import fields
import hashlib
import importlib.metadata as md
import json
import multiprocessing
from pathlib import Path
import resource
import subprocess
import time
import traceback
import csv

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from workflows.v2.p0.p0_experiment import ResearchExperiment, ExperimentConfig, read, write
from tqg_gda.feedback_v2 import ParameterSnapshot, V2Config, signature_kernels, target_space, query_matrices
from tqg_gda.calibration_v2 import CalibrationConfig, solve_calibration
from tqg_gda.p0_diagnostics import label_free, tensor_hash, model_hash, gradient_of_ce
from tqg_gda.utils import sha256, seed_everything
from tqg_gda.assumption_diagnostics import (ProbeConfig, ProbeLabelBoundary, ordered, split_folds,
    stable_seed, fit_grid, metrics, support_statistics)


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime+r.ru_stime


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


class AssumptionExperiment:
    def __init__(self, repo, stage, config):
        self.repo, self.stage = Path(repo).resolve(), Path(stage).resolve()
        self.code = REPO_ROOT
        self.settings = read(config)
        self.config = str(Path(config).resolve())
        self.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.code, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=self.code):
            raise ValueError('Execution code must be clean')
        self.old = self.make_backend()
        self.probe_config = ProbeConfig(**{f.name: tuple(self.settings[f.name]) if f.name.endswith('_grid') else self.settings[f.name]
                                         for f in fields(ProbeConfig)})
        self.hardware = self.old.hardware
        self.plan_sha = sha256(self.stage.parent/'stage2_method/method.md')
        if self.settings['final_labels_allowed'] or self.settings['device'] != 'cpu':
            raise ValueError('CPU-only dev diagnostic protocol')

    def make_backend(self):
        return ResearchExperiment(ExperimentConfig(str(self.repo), self.settings['old_stage'], str(self.code/'configs/p0_diagnostics.json')))

    def snapshot_references(self):
        return [r for r in read(Path(self.settings['old_stage'])/'snapshot_manifest.json')['references'] if r['snapshot_step'] == 0]

    def prepare(self):
        self.stage.mkdir(exist_ok=True)
        for name in ['logs', 'features', 'D0', 'figures']:
            (self.stage/name).mkdir(exist_ok=True)
        findings = []
        for file in [self.code/'workflows/v2/assumptions/assumptions_experiment.py', self.code/'tqg_gda/assumption_diagnostics.py'] + ([self.code/'workflows/v2/assumptions/proteins_assumptions_experiment.py'] if self.settings.get('dataset') == 'PROTEINS' else []):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('exec', 'eval', 'compile'):
                    findings.append(dict(file=str(file), line=node.lineno))
                if isinstance(node, ast.Import) and any(n.name.split('.')[0] in ('requests', 'socket', 'urllib') for n in node.names):
                    findings.append(dict(file=str(file), line=node.lineno))
        assert not findings
        write(self.stage/'ast_screen.json', dict(passed=True, subprocess_justification='local git metadata and spawned CPU workers only'))
        refs = self.snapshot_references()
        assert len(refs) == 15
        for r in refs:
            assert sha256(self.repo/r['checkpoint']) == r['checkpoint_sha256']
        write(self.stage/'snapshot_manifest.json', dict(references=refs))
        write(self.stage/'experiment_manifest.json', dict(git_commit=self.commit, plan_sha256=self.plan_sha,
              settings=self.settings, hardware=self.hardware, versions={k: md.version(k) for k in ['numpy', 'scipy', 'torch', 'torch-geometric', 'tensorboard']},
              prepared_at=time.time(), expected_selection_units=2880, expected_logical_fits=34560,
              final_labels_allowed=False, no_backbone_updates=True))
        write(self.stage/'config.json', self.settings)

    def extract(self, snapshot, task, ids, basis, q):
        h, a, p = [], [], []
        for start in range(0, len(ids), self.settings['microbatch']):
            batch = task.batch(list(ids[start:start+self.settings['microbatch']]), 'target')
            h.append(snapshot.representation(batch).detach())
            a.append(snapshot.response(batch, q, basis).detach())
            p.append(snapshot.logits(batch).detach().softmax(-1))
        return torch.cat(h), torch.cat(a), torch.cat(p)

    def prepare_geometry(self, domain, seed):
        task = self.old.task(domain)
        model, row = self.old.load(domain, seed, 0)
        before = model_hash(model)
        snapshot = ParameterSnapshot(model)
        token = f'{domain}_{seed}_0_0'
        old_stage = Path(self.settings['old_stage'])
        indices = read(old_stage/f'logs/{seed}/{token}_indices.json')
        cache_path = old_stage/f'geometry_cache/{seed}/{token}.npz'
        old_files = {r['path']: r for r in read(old_stage/'local_geometry_manifest.json')['files']}
        assert sha256(cache_path) == old_files[str(cache_path.relative_to(old_stage))]['sha256']
        z = {k: torch.from_numpy(v) for k, v in dict(np.load(cache_path)).items()}
        a_ids, b_ids, dev, final = self.old.pools(domain)
        qs = indices['query']
        assert qs == sorted(task.query_ids) and not set(qs) & set(task.fit_ids)
        assert set(indices['A']) <= set(a_ids) and len(indices['A']) == 16
        boundary = ProbeLabelBoundary(task, a_ids, b_ids, final, self.stage/'D0/label_access.jsonl')
        source = label_free(task.batch(qs, 'query'))
        fresh_h = snapshot.representation(source).detach()
        fresh_p = snapshot.logits(source).detach().softmax(-1)
        fresh_a = snapshot.response(source, z['Q'], z['V']).detach()
        jac = snapshot.jacobian(task.batch(indices['A'], 'target'), z['Q'])
        fresh_v, _ = target_space(jac)
        checks = {}
        for name, left, right in [('source_h', fresh_h, z['source_h']), ('source_p', fresh_p, z['source_p']),
                                 ('source_A', fresh_a, z['source_A']), ('V', fresh_v, z['V'])]:
            checks[name] = float((left-right).abs().max())
            assert torch.allclose(left, right, atol=1e-8, rtol=1e-8), (token, name, checks[name])
        readouts = []
        query_diag = []
        for name, signature in [('H', 'representation_only'), ('HA', 'representation_response')]:
            config = V2Config(signature=signature)
            kernels = signature_kernels(fresh_h, z['target_h'], fresh_a, z['target_A'], config)
            hst, htt = query_matrices(kernels[1], kernels[2], z['target_A'])
            w, info = solve_calibration(kernels[0], hst, htt, CalibrationConfig())
            ref_path = cache_path if name == 'HA' else Path(self.settings['old_h_stage'])/f'geometry_cache/{seed}/{token}.npz'
            ref = np.load(ref_path)
            assert torch.allclose(w, torch.from_numpy(ref['W']), atol=1e-8, rtol=1e-8), (name, token)
            readouts.append((name, w, info, hst, ref))
        boundary.log(dict(event='readouts_sealed', hashes=[tensor_hash(item[1]) for item in readouts]))
        labels = torch.tensor([task.labels[i] for i in qs])
        a_labels = torch.from_numpy(boundary.read('A_train', indices['A']))
        g_a = gradient_of_ce(snapshot, [task.batch(indices['A'], 'target')], a_labels)
        residual = ((torch.nn.functional.one_hot(labels, 2).double()-fresh_p)@z['Q']).flatten()
        for name, w, info, hst, ref in readouts:
            gradient = -z['V']@(w@residual)
            old_gradient = -z['V']@(torch.from_numpy(ref['W'])@residual)
            assert torch.allclose(gradient, old_gradient, atol=1e-8, rtol=1e-8)
            ca = float(torch.dot(gradient, g_a)/(gradient.norm()*g_a.norm()))
            prior = [json.loads(line) for line in (self.repo/f'experiments/{"tqg_v2_p0_20261031" if name == "HA" else "tqg_h_only_controls_20261031"}/logs/{seed}/gradient_blocks.jsonl').read_text().splitlines()]
            prior = next(r for r in prior if r['task'] == f'0_to_{domain}' and r['snapshot_step'] == 0 and r['block_id'] == 0 and r['arm'] == 'TRUE')
            assert abs(ca-prior['c_A']) < 1e-8
            query_diag.append(dict(domain=domain, seed=seed, signature=name, c_A=ca,
                e_A=float((gradient-g_a).norm()/g_a.norm()), weight_norm=float(w.norm()),
                feedback_norm=float(gradient.norm()), mean_error=info['mean_error'], max_error=info['max_error'],
                H_ST_norm=float(hst.norm()), converged=info['converged'], W_replay_max=float((w-torch.from_numpy(ref['W'])).abs().max()),
                old_B_projection=prior.get('B_space_outside_energy_fraction'), checkpoint_sha256=row['checkpoint_sha256']))
        th, ta, tp = self.extract(snapshot, task, list(a_ids)+list(b_ids), z['V'], z['Q'])
        # Reuse the historical B-space energy scalar; no B labels are requested during feature construction.
        old_true = prior
        b_projection_fields = {k: v for k, v in old_true.items() if any(s in k.lower() for s in ['outside', 'projection', 'space'])}
        dest = self.stage/f'features/{seed}_{domain}.npz'
        np.savez_compressed(dest, ids=np.array(qs+list(a_ids)+list(b_ids)), h=torch.cat((fresh_h, th)).numpy(),
            a=torch.cat((fresh_a, ta)).numpy(), p=torch.cat((fresh_p, tp)).numpy(), q=z['Q'].numpy().flatten(),
            source_y=labels.numpy(), V=z['V'].numpy())
        identity = dict(query=qs, fit=list(task.fit_ids), validation=list(task.validation_ids), A=list(a_ids), B=list(b_ids), final=final,
            target_batch=indices['A'], checkpoint=row, cache_path=str(cache_path), cache_sha256=sha256(cache_path),
            feature_sha256=sha256(dest), model_unchanged=model_hash(model) == before, checks=checks,
            query_diagnostics=query_diag, historical_B_projection=b_projection_fields)
        assert identity['model_unchanged']
        write(dest.with_suffix('.json'), identity)
        print(json.dumps(dict(event='A0_geometry', seed=seed, domain=domain, checks=checks)), flush=True)

    def a0(self):
        started = cpu_seconds()
        for seed in self.settings['seeds']:
            for domain in self.settings['domains']:
                self.prepare_geometry(domain, seed)
        write(self.stage/'D0/replay.json', dict(passed=True, geometries=15, signatures=30, cpu_seconds=cpu_seconds()-started,
              no_B_labels_read=True, no_final_labels_read=True, snapshot_step=0, block=0))

    def worker(self, seed, pilot=False):
        seed_everything(seed, 1, True)
        out = self.stage/'logs'/(f'pilot_{seed}' if pilot else str(seed))
        out.mkdir(exist_ok=False)
        (out/'models').mkdir()
        writer = SummaryWriter(str(out/'tensorboard'))
        started_cpu, started_wall = cpu_seconds(), time.time()
        units = logical_fits = 0
        write(out/'RUN_INFO.json', dict(seed=seed, git_commit=self.commit, config_name=self.settings['config_name'],
             plan_sha256=self.plan_sha, hardware=self.hardware, gpu_hours=0., pilot=pilot, started_at=started_wall))
        metrics_file = (out/'probe_metrics.jsonl').open('a')
        history_file = (out/'history.jsonl').open('a')
        cv_file = (out/'inner_cv.jsonl').open('a')
        prediction_file = (out/'probe_predictions.csv').open('w', newline='')
        prediction_writer = None
        try:
            for domain in ([1] if pilot else self.settings['domains']):
                data_path = self.stage/f'features/{seed}_{domain}.npz'
                data = dict(np.load(data_path))
                role = read(data_path.with_suffix('.json'))
                assert sha256(data_path) == role['feature_sha256']
                ids = data['ids'].tolist()
                position = {i: j for j, i in enumerate(ids)}
                query, a_pool, b_pool = role['query'], role['A'], role['B']
                source_folds = split_folds(query, 4, 'outer')
                task = self.old.task(domain)
                boundary = ProbeLabelBoundary(task, a_pool, b_pool, role['final'], out/'label_access.jsonl')
                source_labels = dict(zip(query, data['source_y'].tolist()))
                distances, kernels = {}, {}
                ns = len(query)
                sh = torch.from_numpy(data['h'][:ns])
                sa = torch.from_numpy(data['a'][:ns].reshape(ns, -1))
                scale_h = float(torch.quantile(torch.pdist(sh).square(), .5).clamp_min(1e-16))
                scale_a = float(torch.quantile(torch.pdist(sa).square(), .5).clamp_min(1e-16))
                ht, at = torch.from_numpy(data['h']), torch.from_numpy(data['a'].reshape(len(ids), -1))
                dh = torch.cdist(ht, ht, compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/scale_h
                da = torch.cdist(at, at, compute_mode='donot_use_mm_for_euclid_dist').square().numpy()/scale_a
                for signature in self.settings['signatures']:
                    distances[signature] = dh if signature == 'H' else dh+da
                    kernels[signature] = np.exp(-distances[signature]/2)
                jobs, supports = [], []
                for fold in range(3 if pilot else 4):
                    source_audit = sorted(source_folds[fold])
                    source_train = sorted(set(query)-set(source_audit))
                    ntrain = len(source_train)
                    target_train = sorted(ordered(a_pool, 'target_train', domain, fold)[:ntrain])
                    assert len(target_train) == ntrain and ntrain+len(source_audit) == ns
                    assert not set(source_train)&set(source_audit)
                    target_y = boundary.read('A_train', target_train)
                    for signature in self.settings['signatures']:
                        source_idx = [position[i] for i in source_train]
                        b_idx = [position[i] for i in b_pool]
                        support = support_statistics(distances[signature][np.ix_(source_idx, source_idx)],
                            distances[signature][np.ix_(source_idx, b_idx)], kernels[signature][np.ix_(source_idx, b_idx)])
                        supports.append(dict(fold=fold, signature=signature, B=b_pool, threshold=support['threshold'],
                            near=support['near'].tolist(), distance=support['distance'].tolist(),
                            max_kernel=support['max_kernel'].tolist(), ess=[float(x) if np.isfinite(x) else None for x in support['ess']]))
                        x = data['h'] if signature == 'H' else np.c_[data['h'], data['a'].reshape(len(ids), -1)]
                        for train_domain in ['S', 'T']:
                            train_ids = source_train if train_domain == 'S' else target_train
                            true_y = np.array([source_labels[i] for i in train_ids]) if train_domain == 'S' else target_y
                            label_sets = [true_y]+[true_y[np.random.default_rng(stable_seed('null', seed, fold, train_domain, rep)).permutation(ntrain)] for rep in range(3)]
                            y_variants = np.asarray(label_sets)
                            train_index = [position[i] for i in train_ids]
                            eval_ids = source_audit+b_pool if train_domain == 'S' else b_pool
                            eval_index = [position[i] for i in eval_ids]
                            inner = split_folds(train_ids, 3, 'inner', fold)
                            meta = dict(seed=seed, task=f'0_to_{domain}', fold=fold, signature=signature, train_domain=train_domain)
                            write(out/f'{domain}_{fold}_{signature}_{train_domain}_ids.json', dict(train=train_ids,
                                 source_audit=source_audit, B=b_pool, inner_validation=inner, source_train=source_train))
                            for family in self.settings['families']:
                                grid = getattr(self.probe_config, {'KRR':'krr_grid', 'LR':'logistic_grid', 'MLP':'mlp_grid'}[family])
                                cv_scores = np.zeros((4, len(grid)))
                                cv_info = []
                                fit_scope = ['inner']
                                def log(record):
                                    history_file.write(json.dumps(dict(**meta, fit_scope=fit_scope[0], **record), allow_nan=False)+'\n')
                                for k, validation_ids in enumerate(inner):
                                    fit_scope[0] = f'inner_{k}'
                                    va = [train_ids.index(i) for i in validation_ids]
                                    tr = sorted(set(range(ntrain))-set(va))
                                    tr_idx, va_idx = np.array(train_index)[tr], np.array(train_index)[va]
                                    pred, _, info = fit_grid(family, x[tr_idx], y_variants[:, tr], data['p'][tr_idx, 1],
                                        x[va_idx], data['p'][va_idx, 1], kernels[signature][np.ix_(tr_idx, tr_idx)],
                                        kernels[signature][np.ix_(va_idx, tr_idx)], data['q'], grid,
                                        stable_seed('init', seed, fold, k), self.probe_config, log)
                                    cv_info += info
                                    logical_fits += 4*len(grid)
                                    for variant in range(4):
                                        for j, penalty in enumerate(grid):
                                            score = metrics(y_variants[variant, va], pred[variant, j])
                                            cv_scores[variant, j] += score['nll']/3
                                            cv_file.write(json.dumps(dict(**meta, family=family, variant=variant, inner_fold=k,
                                                penalty=penalty, **score, optimization=info[variant*len(grid)+j]), allow_nan=False)+'\n')
                                selected = [min(range(len(grid)), key=lambda j: (cv_scores[i, j], -grid[j])) for i in range(4)]
                                fit_scope[0] = 'refit'
                                # Refit exactly one independently selected penalty per label variant.
                                pred, states, info = fit_grid(family, x[train_index], y_variants, data['p'][train_index, 1],
                                    x[eval_index], data['p'][eval_index, 1], kernels[signature][np.ix_(train_index, train_index)],
                                    kernels[signature][np.ix_(eval_index, train_index)], data['q'], [grid[j] for j in selected],
                                    stable_seed('init', seed, fold, 'refit'), self.probe_config, log, paired=True)
                                logical_fits += 4  # protocol-selected refits
                                for variant, j in enumerate(selected):
                                    key = f'{domain}_{fold}_{signature}_{train_domain}_{family}_{variant}'
                                    np.savez_compressed(out/f'models/{key}.npz', **states[variant, 0], train_ids=np.array(train_ids),
                                                        penalty=grid[j], training_y=y_variants[variant])
                                    jobs.append(dict(**meta, family=family, variant=variant, penalty=grid[j], key=key,
                                        eval_ids=eval_ids, prediction=pred[variant, 0], train_y=y_variants[variant],
                                        train_ids=train_ids, prior=float(y_variants[variant].mean()), cv_nll=float(cv_scores[variant, j]),
                                        optimization=info[variant], inner_nonconverged=sum(r['converged'] is False for r in cv_info if r['variant'] == variant)))
                                units += 4
                                writer.add_scalar(f'{family}/cv_nll_true', cv_scores[0, selected[0]], units)
                                used_cpu = cpu_seconds()-started_cpu
                                write(out/'progress.json', dict(seed=seed, domain=domain, fold=fold, selection_units=units,
                                      logical_fits=logical_fits, cpu_seconds=used_cpu, elapsed=time.time()-started_wall,
                                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                                allocation = read(self.stage/'compute_estimate.json')['formal_budget_cpu_seconds']/5 if not pilot else self.settings['cpu_hour_cap']*3600/5
                                if used_cpu > allocation or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > self.settings['memory_cap_gib']*1024**2/5:
                                    raise TimeoutError('Per-worker allocation exhausted; save incomplete run')
                    print(json.dumps(dict(event='probe_progress', pilot=pilot, seed=seed, domain=domain, fold=fold, selection_units=units,
                        cpu_seconds=cpu_seconds()-started_cpu)), flush=True)
                write(out/f'{domain}_support.json', dict(groups=supports, scale_h_squared=scale_h, scale_a_squared=scale_a))
                pred_path = out/f'{domain}_sealed_predictions.npz'
                np.savez_compressed(pred_path, **{j['key']: j['prediction'] for j in jobs})
                history_file.flush(); cv_file.flush()
                artifacts = dict(domain=domain, prediction_sha256=sha256(pred_path),
                    support_sha256=sha256(out/f'{domain}_support.json'), feature_sha256=sha256(data_path),
                    inner_cv_prefix_bytes=(out/'inner_cv.jsonl').stat().st_size,
                    inner_cv_prefix_sha256=sha256(out/'inner_cv.jsonl'),
                    models=[dict(path=f'models/{job["key"]}.npz', sha256=sha256(out/f'models/{job["key"]}.npz')) for job in jobs])
                artifact_path = out/f'{domain}_sealed_artifacts.json'
                write(artifact_path, artifacts)
                boundary.log(dict(event='probe_artifacts_sealed', domain=domain, sha256=sha256(artifact_path)))
                boundary.seal(sha256(pred_path), len(jobs))
                if pilot:
                    continue
                b_y = boundary.read('B_evaluation', b_pool)
                self.after_B_labels(domain, seed, data, b_pool, b_y, out)
                b_labels = dict(zip(b_pool, b_y.tolist()))
                for job in jobs:
                    meta = {k: job[k] for k in ['seed','task','fold','signature','train_domain','family','variant','penalty','key','cv_nll','inner_nonconverged']}
                    group = next(g for g in supports if g['fold'] == job['fold'] and g['signature'] == job['signature'])
                    mapping = {i: j for j, i in enumerate(job['eval_ids'])}
                    evaluations = [('B', b_pool, b_y, np.ones(len(b_pool), bool)),
                                   ('B_near', b_pool, b_y, np.array(group['near'])),
                                   ('B_far', b_pool, b_y, ~np.array(group['near']))]
                    if job['train_domain'] == 'S':
                        audit_ids = [i for i in job['eval_ids'] if i in source_labels]
                        evaluations.append(('S_audit', audit_ids, np.array([source_labels[i] for i in audit_ids]), np.ones(len(audit_ids), bool)))
                    for pool, pool_ids, truth, mask in evaluations:
                        raw = job['prediction'][[mapping[i] for i in pool_ids]]
                        pp = data['p'][[position[i] for i in pool_ids], 1]
                        row = dict(**meta, pool=pool, **metrics(truth[mask], raw[mask]),
                            prior_nll=metrics(truth[mask], np.full(mask.sum(), job['prior']))['nll'],
                            model_nll=metrics(truth[mask], pp[mask])['nll'], optimization=job['optimization'],
                            git_commit=self.commit, plan_sha256=self.plan_sha, config_name=self.settings['config_name'], gpu_hours=0.,
                            hardware=self.hardware, event_flags='clean')
                        metrics_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                    for i, graph_id in enumerate(job['eval_ids']):
                        is_b = graph_id in b_labels
                        record = {k: meta[k] for k in ['seed','task','fold','signature','train_domain','family','variant']}
                        record.update(graph_id=graph_id, pool='B' if is_b else 'S_audit', label=b_labels[graph_id] if is_b else source_labels[graph_id],
                                      probability=float(job['prediction'][i]), near=group['near'][b_pool.index(graph_id)] if is_b else '')
                        if prediction_writer is None:
                            prediction_writer = csv.DictWriter(prediction_file, fieldnames=list(record)); prediction_writer.writeheader()
                        prediction_writer.writerow(record)
            result = dict(seed=seed, pilot=pilot, completed=True, selection_units=units, logical_fits=logical_fits,
                          physical_fits=logical_fits,
                          cpu_seconds=cpu_seconds()-started_cpu, elapsed=time.time()-started_wall,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            write(out/'done.json', result)
            return result
        except Exception:
            (out/'error.txt').write_text(traceback.format_exc())
            raise
        finally:
            writer.close()
            for f in [metrics_file, history_file, cv_file, prediction_file]:
                f.close()

    def after_B_labels(self, domain, seed, data, b_pool, b_y, out):
        pass


def experiment_class(config):
    if read(config).get('dataset') == 'PROTEINS':
        from workflows.v2.assumptions.proteins_assumptions_experiment import ProteinsAssumptionExperiment
        return ProteinsAssumptionExperiment
    return AssumptionExperiment


def worker(args):
    repo, stage, config, seed, pilot = args
    return experiment_class(config)(repo, stage, config).worker(seed, pilot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'a0', 'pilot', 'run'])
    parser.add_argument('--repo', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--config', default='configs/assumptions_diagnostics.json')
    args = parser.parse_args()
    seed_everything(3405, 1, True)
    exp = experiment_class(args.config)(args.repo, args.stage, args.config)
    if args.action == 'prepare':
        exp.prepare()
    elif args.action == 'a0':
        exp.a0()
    elif args.action == 'pilot':
        result = exp.worker(3405, True)
        cost = result['cpu_seconds']/result['selection_units']*2880*exp.settings['compute_buffer']
        remaining = exp.settings['cpu_hour_cap']*3600-read(exp.stage/'D0/replay.json')['cpu_seconds']-result['cpu_seconds']-360
        write(exp.stage/'compute_estimate.json', dict(pilot=result, estimated_formal_cpu_seconds=cost,
              cap_cpu_seconds=exp.settings['cpu_hour_cap']*3600, formal_budget_cpu_seconds=remaining, feasible=cost < remaining))
        if cost >= remaining:
            raise TimeoutError('Pilot budget gate failed')
    else:
        assert read(exp.stage/'compute_estimate.json')['feasible']
        results = []
        with ProcessPoolExecutor(max_workers=5, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(worker, (args.repo,args.stage,args.config,seed,False)):seed for seed in exp.settings['seeds']}
            while futures:
                done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result()); del futures[future]
                progress = []
                for seed in exp.settings['seeds']:
                    p = exp.stage/f'logs/{seed}/progress.json'
                    if p.exists(): progress.append(read(p))
                print(json.dumps(dict(event='heartbeat', completed=len(results), selection_units=sum(p['selection_units'] for p in progress),
                      cpu_seconds=sum(p['cpu_seconds'] for p in progress))), flush=True)
        write(exp.stage/'execution_done.json', dict(completed=True, workers=results))


if __name__ == '__main__':
    main()
