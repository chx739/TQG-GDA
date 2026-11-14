"""Replay static checkpoints, audit controls, and summarize all five fixed settings."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import torch
from tsba.config import Config
from experiments.common.synthetic.archive import digest, write_json
from experiments.common.synthetic.static_score import SETTINGS, audit_scores, evaluate, new_model, schedules, tensor_hash
from experiments.common.synthetic.summarize_replay import close_tree, flat, stats, write_csv


def read(p):
    return json.loads(p.read_text())


def check_manifest(root):
    items = read(root/'manifest.json')
    for name, info in items.items():
        p = root/name
        if digest(p) != info['sha256'] or p.stat().st_size != info['bytes']:
            raise ValueError('sealed file changed: '+str(p))
    return len(items)


def audit(root):
    verified = check_manifest(root)
    parent = Path(read(root/'parent-integrity-before.json')['path'])
    parent_files = check_manifest(parent)
    metrics, signals, fits, gates = [], [], [], []
    replays, oracle_probes, recomputed, baseline_checks = 0, 0, 0, 0
    for directory in sorted(root.glob('d*_seed*')):
        cfg = Config(**read(directory/'config.json'))
        fixed = torch.load(directory/'fixed-inputs.pt', weights_only=True)
        old = torch.load(parent/directory.name/'inputs.pt', weights_only=True)
        old_queries = torch.load(parent/directory.name/'queries.pt', weights_only=True)
        for k in fixed['initial_head']:
            torch.testing.assert_close(fixed['initial_head'][k], old['initial_head'][k], atol=0, rtol=0)
        torch.testing.assert_close(fixed['target'][:512], old['target'], atol=0, rtol=0)
        for name, bank in fixed['queries'].items():
            for k, x in bank.items():
                torch.testing.assert_close(x, old_queries[name][k], atol=0, rtol=0)
        schedule = torch.load(directory/'schedule.pt', weights_only=True)
        expected = schedules(cfg.emb_dim, cfg.seed)
        for key in ('noise', 'validation_noise'):
            torch.testing.assert_close(schedule[key], expected[key], atol=0, rtol=0)
        for n in (512, 2048, 8192):
            torch.testing.assert_close(schedule['ids'][n], expected['ids'][n], atol=0, rtol=0)
            assert schedule['ids'][n].shape == (400, 128)
        masks = torch.load(directory/'common-masks.pt', weights_only=True)
        scale, mean = fixed['coordinates']['scale'], fixed['coordinates']['mean']
        primary_results = {}
        for setting, n, multiplier in SETTINGS:
            dest = directory/setting
            row = read(dest/'summary.json')
            data = torch.load(dest/'per-query.pt', weights_only=True)
            checkpoint = torch.load(dest/'checkpoint.pt', weights_only=True)
            saved_metrics = read(dest/'metrics.json')
            training = read(dest/'training.json')
            sigma = fixed['baseline_sigma']*multiplier
            assert row['sigma'] == sigma and row['updates'] == checkpoint['updates'] == 400
            assert len(training['train_history']) == 400 and len(training['validation_history']) == 100
            assert all(int(x['step']) == 400 for x in checkpoint['optimizer']['state'].values())
            best = min(training['validation_history'], key=lambda x: x['noise_validation_loss'])
            assert checkpoint['selected_update'] == best['update'] == row['selected_update']
            assert row['noise_hash'] == tensor_hash(schedule['noise'])
            assert row['pool_hash'] == tensor_hash(fixed['target'][:n])
            initial = new_model(cfg.emb_dim, cfg)
            for k, x in initial.state_dict().items():
                torch.testing.assert_close(x, checkpoint['initial_state'][k], atol=0, rtol=0)
            if setting == 'n512_s1':
                old_state = torch.load(parent/directory.name/'score.pt', weights_only=True)['state_dict']
                for k in old_state:
                    torch.testing.assert_close(checkpoint['selected_state'][k], old_state[k], atol=0, rtol=0)
                baseline_checks += 1
            oracle = torch.load(dest/'oracle.pt', weights_only=True)
            torch.testing.assert_close(oracle['empirical']['means'], fixed['target'][:n], atol=0, rtol=0)
            for bank_name, bank in fixed['queries'].items():
                points, normals = bank['points'], bank.get('normals')
                arrays = data[bank_name]
                for label, state in (('learned', 'final_state'), ('selected', 'selected_state')):
                    restored = new_model(cfg.emb_dim, cfg).requires_grad_(False).eval()
                    restored.load_state_dict(checkpoint[state])
                    torch.testing.assert_close(evaluate(restored, points), arrays['score_'+label], atol=0, rtol=0)
                    replays += 1
                # Independent autograd log-mixture probes verify the saved oracle states.
                probe_ids = torch.linspace(0, len(points)-1, min(8, len(points))).long().unique()
                for label in ('empirical', 'population'):
                    state = oracle[label]
                    probe = points[probe_ids].clone().requires_grad_(True)
                    difference = probe[:, None]-state['means'][None]
                    logs = -.5*(difference.square()/state['variance']+state['variance'].log()).sum(-1)
                    gradient = torch.autograd.grad(logs.logsumexp(-1).sum(), probe)[0]
                    torch.testing.assert_close(gradient, arrays['score_'+label][probe_ids], atol=1e-10, rtol=1e-10)
                    oracle_probes += len(probe_ids)
                values = {k: arrays['score_'+k] for k in ('learned','empirical','population','zero','selected')}
                result, rebuilt = audit_scores(points, normals, values, sigma, float(scale), cfg,
                                               masks.get(bank_name), points[:, 0]*scale+mean[0])
                close_tree(result, saved_metrics[bank_name])
                for k, value in rebuilt.items():
                    torch.testing.assert_close(value, arrays[k], atol=0, rtol=0)
                recomputed += 1
                identity = dict(dim=cfg.emb_dim, seed=cfg.seed, setting=setting, bank=bank_name)
                for comparison, value in result['comparisons'].items():
                    metrics.append(dict(**identity, comparison=comparison, count=result['count'], **flat(value)))
                for estimator, value in result['signals'].items():
                    signals.append(dict(**identity, estimator=estimator, **value))
            primary_results[setting] = saved_metrics
            fits.append({k: row[k] for k in ('dim','seed','setting','samples','sigma_multiplier','sigma','raw_sigma',
                                            'updates','selected_update','seconds','peak_rss_bytes')})
        # Report every criterion separately; the conjunction is a descriptive screen.
        for setting, results in primary_results.items():
            gate = dict(dim=cfg.emb_dim, seed=cfg.seed, setting=setting)
            for bank in ('initial_boundaries','auxiliary_planes'):
                c = results[bank]['comparisons']
                gate[bank+'_normal_better_zero'] = c['learned_vs_population']['normal_rmse_scaled'] < c['zero_vs_population']['normal_rmse_scaled']
                gate[bank+'_request_better_zero'] = c['learned_vs_population']['request_rmse_sigma'] < c['zero_vs_population']['request_rmse_sigma']
                signal = results[bank]['signals']
                gate[bank+'_oracle_toward_target'] = signal['population']['toward_target_boundary'] == 1.
                gate[bank+'_learned_majority_toward_target'] = signal['learned']['toward_target_boundary'] > .5
            signs = results['auxiliary_planes']['comparisons']['learned_vs_population']['common_sign']
            gate['both_sign_recalls_above_half'] = all(signs[k] is not None and signs[k] > .5
                                                     for k in ('balanced','positive_recall','negative_recall'))
            gate['passes_all'] = all(v for k, v in gate.items() if k not in ('dim','seed','setting'))
            gates.append(gate)
    return dict(files_verified=verified, parent_files_verified=parent_files, score_fits=len(fits),
                optimizer_updates=sum(r['updates'] for r in fits), checkpoint_bank_replays=replays,
                independent_oracle_gradient_probe_points=oracle_probes, metric_banks_recomputed=recomputed,
                historical_baselines_bitwise=baseline_checks, frozen_queries_exact=True,
                nested_pools_exact=True, fixed_noise_and_initialization_exact=True,
                classifier_updates=0, target_evaluation_label_access=False), metrics, signals, fits, gates


def aggregate(rows, keys, ignored=()):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    output = []
    for identity, members in groups.items():
        fields = [k for k in members[0] if k not in (*keys, 'seed', *ignored)]
        values = {k: stats([r.get(k) for r in members]) for k in fields}
        output.append(dict(zip(keys, identity), **values))
    return output


def paired(rows):
    pairs = []
    identities = ('dim','seed','bank','comparison')
    base = {tuple(r[k] for k in identities): r for r in rows if r['setting'] == 'n512_s1'}
    for row in rows:
        if row['setting'] == 'n512_s1':
            continue
        old = base[tuple(row[k] for k in identities)]
        diff = {k: (v-old[k] if v is not None and old[k] is not None else None)
                for k, v in row.items() if k not in (*identities, 'setting')}
        pairs.append(dict(**{k: row[k] for k in (*identities, 'setting')}, **diff))
    return pairs


def figures(rows, signals, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['512 / 1x','2048 / 1x','8192 / 1x','512 / 0.5x','512 / 2x']
    settings = [s[0] for s in SETTINGS]
    comparisons = [('learned_vs_population','Learned'),('empirical_vs_population','Empirical'),('zero_vs_population','Zero')]
    for bank, name in (('initial_boundaries','initial-normal'),('auxiliary_planes','auxiliary-normal')):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout='constrained')
        for i, dim in enumerate((2,16)):
            for j, metric in enumerate(('normal_rmse_scaled','normal_rmse_raw','request_rmse_raw')):
                for comparison, label in comparisons:
                    groups = [[r[metric] for r in rows if (r['dim'], r['bank'], r['setting'], r['comparison']) ==
                               (dim, bank, s, comparison)] for s in settings]
                    axes[i,j].errorbar(range(5), [stats(g)['mean'] for g in groups],
                                      yerr=[stats(g)['sd'] for g in groups], marker='o', label=label, capsize=3)
                axes[i,j].set_title(f'd={dim}: {metric}')
                axes[i,j].set_xticks(range(5), labels, rotation=20)
                axes[i,j].set_yscale('log')
                axes[i,j].grid(alpha=.2)
        axes[0,0].legend()
        fig.suptitle(bank+' | five-seed mean +/- sample SD; fixed queries')
        fig.savefig(output/(name+'.png'), dpi=150)
        fig.savefig(output/(name+'.svg'))
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for i, dim in enumerate((2,16)):
        for comparison, label in comparisons[:2]:
            groups = [[r['common_sign.balanced'] for r in rows if (r['dim'],r['bank'],r['setting'],r['comparison']) ==
                       (dim,'auxiliary_planes',s,comparison)] for s in settings]
            axes[i,0].plot(range(5), [stats(g)['mean'] for g in groups], marker='o', label=label)
        for estimator in ('learned','empirical','population'):
            groups = [[r['toward_target_boundary'] for r in signals if (r['dim'],r['bank'],r['setting'],r['estimator']) ==
                       (dim,'auxiliary_planes',s,estimator)] for s in settings]
            axes[i,1].plot(range(5), [stats(g)['mean'] for g in groups], marker='o', label=estimator)
        for j in (0,1):
            axes[i,j].axhline(.5, color='gray', linestyle='--', linewidth=1)
            axes[i,j].set_ylim(-.04,1.04)
            axes[i,j].set_xticks(range(5), labels, rotation=20)
            axes[i,j].grid(alpha=.2)
        axes[i,0].set_title(f'd={dim}: agreement with SAME-sigma oracle')
        axes[i,1].set_title(f'd={dim}: direction toward target boundary 0')
    axes[0,0].legend(); axes[0,1].legend()
    fig.suptitle('Fixed two-sided auxiliary queries | five-seed mean')
    fig.savefig(output/'direction.png', dpi=150)
    fig.savefig(output/'direction.svg')
    plt.close(fig)
