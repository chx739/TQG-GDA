"""Post-OBS metrics and paired summaries; no fitting or new neighborhoods."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from tsba.classifier import ResidualClassifier
from experiments.common.io import save_tensor, write_json
from experiments.common.statistics import classification_metrics, clean, write_csv; from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.evaluate import seed_summary, strict_fold_summary
from experiments.mutagenicity.r1_representation_boundary_support_20261111.src.probes import Probe
from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.common import verify_seal


def replay(root, plan):
    root = Path(root)
    if (root / 'seals/OBS.json').exists():
        raise PermissionError('Replay precedes OBS')
    features = {s: torch.load(root / f'deps/source/seed{s}/frozen.pt', weights_only=True)['x'] for s in plan['seeds']}
    maximum, warm_pools, step_pools = 0., 0, 0
    with torch.no_grad():
        for mp in sorted((root / 'warm').rglob('metadata.json')):
            meta = json.loads(mp.read_text())
            if meta['status'] != 'success':
                continue
            predictions = torch.load(mp.parent / 'probabilities.pt', weights_only=True)
            for endpoint, pools in predictions.items():
                state = torch.load(mp.parent / meta['checkpoints'][endpoint], weights_only=False)
                with torch.random.fork_rng(devices=[]):
                    model = Probe(state['mean'], state['scale'], 'linear')
                model.load_state_dict(state['model_state'])
                model.eval()
                for pool in pools.values():
                    logits = model(features[meta['seed']][pool['ids']])
                    actual = logits.softmax(-1)
                    maximum = max(maximum, float((actual-pool['probabilities']).abs().max()))
                    torch.testing.assert_close(actual, pool['probabilities'], rtol=0, atol=0)
                    torch.testing.assert_close(logits.log_softmax(-1), pool['log_probabilities'], rtol=0, atol=0)
                    warm_pools += 1
        for row in json.loads((root / 'steps/index.json').read_text()):
            if row['status'] != 'success':
                continue
            item = torch.load(root / row['path'], weights_only=False)
            for view in item['views'].values():
                if view.get('status','success') != 'success':
                    continue
                state = view['model_state']
                with torch.random.fork_rng(devices=[]):
                    model = ResidualClassifier(features[row['seed']].shape[1], 2, state['U1.weight'].shape[0], dtype=torch.float64)
                model.load_state_dict(state)
                model.eval()
                for pool in view['probabilities'].values():
                    logits = model(features[row['seed']][pool['ids']])
                    actual = logits.softmax(-1)
                    maximum = max(maximum, float((actual-pool['probabilities']).abs().max()))
                    torch.testing.assert_close(actual, pool['probabilities'], rtol=1e-10, atol=1e-12)
                    torch.testing.assert_close(logits.log_softmax(-1), pool['log_probabilities'], rtol=1e-10, atol=1e-12)
                    step_pools += 1
    result = dict(warm_endpoint_pools=warm_pools, step_view_pools=step_pools,
                  maximum_probability_error=maximum, exact_match=maximum == 0., step_rtol=1e-10,
                  step_atol=1e-12, warm_exact_match=True, labels_used=False)
    write_json(root / 'prediction-replay.json', result)
    return result


def _metrics(pool, labels):
    ids = list(map(int, pool['ids']))
    return classification_metrics(pool['probabilities'], [labels[i] for i in ids], pool['log_probabilities'])


def _subset(global_pool, ids):
    return dict(ids=ids, probabilities=global_pool['probabilities'][ids], log_probabilities=global_pool['log_probabilities'][ids])


def evaluate_warm(root, plan, b_labels, source_labels):
    root = Path(root)
    rows = []
    roles = json.loads((root / 'deps/roles.json').read_text())
    source_predictions = {}
    for seed in plan['seeds']:
        source_predictions[seed] = torch.load(root / f'deps/source/seed{seed}/predictions.pt', weights_only=True)
        for domain in plan['domains']:
            pool = _subset(source_predictions[seed], roles['targets'][str(domain)]['B'])
            rows.append(dict(seed=seed, domain=domain, fold=None, method='source_initial', endpoint='selected',
                             pool=f'B{domain}', selected_step=0, **_metrics(pool, b_labels[domain])))
    for mp in sorted((root / 'warm').rglob('metadata.json')):
        meta = json.loads(mp.read_text())
        if meta['status'] != 'success':
            rows.append(dict(seed=meta['seed'], domain=meta['domain'], fold=meta['fold'], status=meta['status'],
                             failure=meta.get('failure'), method='warm' if 'lr' in meta else 'random_corrected'))
            continue
        method = f"warm_lr{meta['lr']:.0e}" if meta.get('initialization_kind') == 'source_affine' else 'random_corrected'
        saved = torch.load(mp.parent / 'probabilities.pt', weights_only=True)
        for endpoint, pools in saved.items():
            for name, pool in pools.items():
                labels = b_labels[meta['domain']] if name.startswith('B') else source_labels[meta['seed']]
                rows.append(dict(seed=meta['seed'], domain=meta['domain'], fold=meta['fold'], method=method,
                    endpoint=endpoint, pool=name, selected_step=meta['selected_step'], **_metrics(pool, labels)))
    # Preserve the previously published random MLP as contextual baseline, not a new fit.
    for seed in plan['seeds']:
        for domain in plan['domains']:
            for fold in range(4):
                base = root / f'deps/probes/target{domain}/seed{seed}/fold{fold}/H/mlp/true'
                meta = json.loads((base / 'metadata.json').read_text())
                pool = torch.load(base / 'probabilities.pt', weights_only=True)['selected'][f'B{domain}']
                rows.append(dict(seed=seed, domain=domain, fold=fold, method='old_random_mlp', endpoint='selected',
                    pool=f'B{domain}', selected_step=meta['selected_step'], **_metrics(pool, b_labels[domain])))
    metrics = ('nll', 'balanced_accuracy', 'accuracy', 'recall0', 'recall1')
    grouped = defaultdict(list)
    for row in rows:
        if row.get('nll') is not None:
            grouped[(row['domain'], row['method'], row['endpoint'], row['pool'], row['seed'])].append(row)
    by_seed = []
    for domain in plan['domains']:
        pools = (f'B{domain}','source_fit','source_validation','source_diagnostic','source_all')
        for method in ('source_initial','warm_lr1e-03','warm_lr1e-04','random_corrected','old_random_mlp'):
            endpoints = ('selected',) if method in ('source_initial','old_random_mlp') else (
                ('0','selected','original_selected') if method == 'random_corrected' else ('0','selected','200','800'))
            for endpoint in endpoints:
                for pool in ((f'B{domain}',) if method in ('source_initial','old_random_mlp') else pools):
                    for seed in plan['seeds']:
                        items = grouped[(domain,method,endpoint,pool,seed)]
                        expected = 1 if method == 'source_initial' else 4
                        if len(items) > expected:
                            raise ValueError('Duplicate fold metrics')
                        by_seed.append(dict(domain=domain,method=method,endpoint=endpoint,pool=pool,seed=seed,
                            available_folds=len(items),expected_folds=expected,
                            **{m:strict_fold_summary([r[m] for r in items]+[None]*(expected-len(items)),expected)['mean'] for m in metrics},
                            selected_step0_count=sum(r['selected_step'] == 0 for r in items)))
    summary = []
    grouped = defaultdict(list)
    for row in by_seed:
        grouped[(row['domain'], row['method'], row['endpoint'], row['pool'])].append(row)
    for (domain, method, endpoint, pool), items in sorted(grouped.items()):
        summary.append(dict(domain=domain, method=method, endpoint=endpoint, pool=pool,
                            **{m: seed_summary([r[m] for r in items], len(plan['seeds'])) for m in metrics},
                            selected_step0_count=sum(r['selected_step0_count'] for r in items)))
    paired = []
    index = {(r['domain'],r['method'],r['endpoint'],r['pool'],r['seed']):r for r in by_seed}
    for domain in plan['domains']:
        for method in ('warm_lr1e-03','warm_lr1e-04','random_corrected','old_random_mlp'):
            for reference in ('source_initial','random_corrected'):
                if method == reference:
                    continue
                differences = {m: [] for m in metrics}
                for seed in plan['seeds']:
                    left = index[(domain,method,'selected',f'B{domain}',seed)]
                    right = index[(domain,reference,'selected',f'B{domain}',seed)]
                    for m in metrics:
                        differences[m].append(left[m]-right[m] if left[m] is not None and right[m] is not None else None)
                paired.append(dict(domain=domain, method=method, reference=reference,
                    **{m: seed_summary(v,len(plan['seeds'])) for m,v in differences.items()},
                    nll_improved_seeds=sum(v is not None and v<0 for v in differences['nll']),
                    ba_improved_seeds=sum(v is not None and v>0 for v in differences['balanced_accuracy'])))
    output = dict(rows=rows, by_seed=by_seed, summary=summary, paired=paired)
    for key, values in output.items():
        write_json(root / 'evaluation' / f'warm_{key}.json', clean(values))
        write_csv(root / 'evaluation' / f'warm_{key}.csv', values)
    return output


def evaluate_steps(root, plan, b_labels, source_labels):
    root = Path(root)
    rows, comparisons = [], []
    source = {s: torch.load(root / f'deps/source/seed{s}/predictions.pt', weights_only=True) for s in plan['seeds']}
    predictions = {}
    for record in json.loads((root / 'steps/index.json').read_text()):
        if record['status'] != 'success':
            for view_name in ('small_step','production_backtrack'):
                for pool_name in (f"B{record['domain']}",'source_fit','source_validation','source_diagnostic','source_all'):
                    base = {k:record[k] for k in ('seed','domain','estimator','bandwidth_multiplier','arm')}
                    row = dict(**base,view=view_name,pool=pool_name,status='failed',failure=record.get('failure',record.get('common_alpha_reason')),
                        **{m:None for m in ('nll','balanced_accuracy','nll_delta_source','ba_delta_source','corrected','newly_wrong','prediction_flips')})
                    rows.append(row)
                    predictions[tuple(base[k] for k in ('seed','domain','estimator','bandwidth_multiplier','arm'))+(view_name,pool_name)] = row
            continue
        item = torch.load(root / record['path'], weights_only=False)
        for view_name, view in item['views'].items():
            for pool_name, pool in view['probabilities'].items():
                if view.get('status','success') != 'success':
                    base = {k:record[k] for k in ('seed','domain','estimator','bandwidth_multiplier','arm')}
                    row = dict(**base,view=view_name,pool=pool_name,status='failed',failure=view.get('reason'),
                        **{m:None for m in ('nll','balanced_accuracy','nll_delta_source','ba_delta_source','corrected','newly_wrong','prediction_flips')})
                    rows.append(row)
                    predictions[tuple(base[k] for k in ('seed','domain','estimator','bandwidth_multiplier','arm'))+(view_name,pool_name)] = row
                    continue
                labels = b_labels[record['domain']] if pool_name.startswith('B') else source_labels[record['seed']]
                ids = list(map(int, pool['ids']))
                y = torch.tensor([labels[i] for i in ids])
                pred = pool['probabilities'].argmax(-1)
                old_pred = source[record['seed']]['probabilities'][ids].argmax(-1)
                base = {k: record[k] for k in ('seed','domain','estimator','bandwidth_multiplier','arm')}
                metrics = _metrics(pool, labels)
                baseline = _metrics(_subset(source[record['seed']], ids), labels)
                row = dict(**base, view=view_name, pool=pool_name, **metrics,
                    nll_delta_source=metrics['nll']-baseline['nll'],
                    ba_delta_source=metrics['balanced_accuracy']-baseline['balanced_accuracy'],
                    corrected=int(((old_pred != y)&(pred == y)).sum()),
                    newly_wrong=int(((old_pred == y)&(pred != y)).sum()),
                    prediction_flips=int((old_pred != pred).sum()),
                    status='success',execution=record['views'][view_name])
                rows.append(row)
                predictions[tuple(base[k] for k in ('seed','domain','estimator','bandwidth_multiplier','arm'))+(view_name,pool_name)] = row
    for seed in plan['seeds']:
        for domain in plan['domains']:
            for estimator in ('empirical','learned'):
                for b in (1.,2.):
                    for view in ('small_step','production_backtrack'):
                        normal = predictions[(seed,domain,estimator,b,'normal',view,f'B{domain}')]
                        for control in ('reverse','zero','matched_source'):
                            other = predictions[(seed,domain,estimator,b,control,view,f'B{domain}')]
                            comparisons.append(dict(seed=seed,domain=domain,estimator=estimator,bandwidth_multiplier=b,
                                view=view,control=control,
                                nll_difference=normal['nll']-other['nll'] if normal['nll'] is not None and other['nll'] is not None else None,
                                ba_difference=normal['balanced_accuracy']-other['balanced_accuracy'] if normal['balanced_accuracy'] is not None and other['balanced_accuracy'] is not None else None))
    grouped = defaultdict(list)
    for r in comparisons:
        grouped[tuple(r[k] for k in ('domain','estimator','bandwidth_multiplier','view','control'))].append(r)
    paired = []
    for key, group in sorted(grouped.items()):
        paired.append(dict(zip(('domain','estimator','bandwidth_multiplier','view','control'),key),
            nll_difference=seed_summary([r['nll_difference'] for r in group],len(plan['seeds'])),
            ba_difference=seed_summary([r['ba_difference'] for r in group],len(plan['seeds'])),
            nll_improved_seeds=sum(r['nll_difference'] is not None and r['nll_difference']<0 for r in group),
            ba_improved_seeds=sum(r['ba_difference'] is not None and r['ba_difference']>0 for r in group)))
    grouped = defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in ('domain','estimator','bandwidth_multiplier','view','arm','pool'))].append(r)
    summary = []
    for key, group in sorted(grouped.items()):
        summary.append(dict(zip(('domain','estimator','bandwidth_multiplier','view','arm','pool'),key),
            **{m:seed_summary([r[m] for r in group],len(plan['seeds'])) for m in
               ('nll','balanced_accuracy','nll_delta_source','ba_delta_source','corrected','newly_wrong','prediction_flips')}))
    for key, values in dict(rows=rows,comparisons=comparisons,paired=paired,summary=summary).items():
        write_json(root / 'evaluation' / f'steps_{key}.json', clean(values))
        write_csv(root / 'evaluation' / f'steps_{key}.csv', values)
    return dict(rows=rows,summary=summary,paired=paired)


def evaluate_all(root, plan, broker):
    from experiments.mutagenicity.r1b_density_and_single_step_diagnostics_20261111.src.scales import evaluate_scales
    root = Path(root)
    verify_seal(root,'OBS')
    roles = json.loads((root / 'deps/roles.json').read_text())
    b = {d:broker.read(roles['targets'][str(d)]['B'],'target_B','Frozen R1b evaluation only',domain=d) for d in plan['domains']}
    source = {}
    for seed in plan['seeds']:
        split = json.loads((root / f'deps/roles/seed{seed}.json').read_text())
        source[seed] = {}
        for role in ('fit','validation','diagnostic'):
            source[seed].update(broker.read(split[role+'_ids'],'source_'+role,'Frozen R1b source evaluation only',seed=seed))
    save_tensor(root / 'evaluation/restricted-labels.pt',dict(B=b,source=source))
    evaluate_warm(root,plan,b,source)
    evaluate_scales(root,plan,b)
    evaluate_steps(root,plan,b,source)
    verify_seal(root,'OBS')
    write_json(root / 'evaluation/audit.json',dict(observations_unchanged=True, final_values_parsed=False,
        B_scope='previously observed development data; no setting/model selection',
        primary='NLL',secondary='balanced accuracy',fold_then_seed_aggregation=True))
