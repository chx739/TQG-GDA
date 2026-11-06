"""Pre-registered, seed-weighted M1 gates. No threshold fitting."""
import statistics


def average(values):
    finite = [x for x in values if x is not None]
    return statistics.mean(finite) if finite else None


def describe(values):
    good = [x for x in values if x is not None]
    return dict(mean=average(good), sd=statistics.stdev(good) if len(good)>1 else None,
                n=len(good), values=values)


def paired_gain(control, true, baseline):
    difference = control['B_ce']-true['B_ce']
    error = max(control['numeric_floor'], true['numeric_floor'],
                abs(difference-(control['B_ce64_actual']-true['B_ce64_actual'])))
    return difference/baseline, error/baseline


def summarize(blocks, source_rows):
    seed_rows, tasks = [], []
    source_usable = sum(r['source_usable'] for r in source_rows)>=2 and len(source_rows)==3
    for version in ('v1','v2'):
        for domain in (1,3):
            relevant = [r for r in blocks if r['version']==version and r['domain']==domain]
            if len(relevant)!=12:
                raise ValueError('Each version/task needs all 12 registered blocks')
            for seed in (3405,3406,3407):
                rows = [r for r in relevant if r['seed']==seed]
                if len(rows)!=4:
                    raise ValueError('Missing seed blocks')
                raw = []
                for block in rows:
                    g = {r['arm']:r for r in block['gradients']}
                    s = {r['arm']:r for r in block['steps']}
                    ct, cp, cr = g['TRUE']['cos_A'], average([g[f'PERM-{i}']['cos_A'] for i in range(1,4)]), average([g[f'RAND-{i}']['cos_A'] for i in range(1,4)])
                    one = dict(cos_true=ct, cos_perm=cp, cos_rand=cr,
                        cos_adv_perm=ct-cp if ct is not None and cp is not None else None,
                        cos_adv_rand=ct-cr if ct is not None and cr is not None else None,
                        paired_valid=int(all(g[k]['cos_A'] is not None for k in
                            ['TRUE']+[f'{a}-{i}' for a in ('PERM','RAND') for i in range(1,4)])))
                    controls = {'fit':[s['FIT-CE']], 'query':[s['FIT+QRY-CE']],
                        'mean':[s['FIT+MEAN-Y']], 'perm':[s[f'FIT+PERM-{i}'] for i in range(1,4)],
                        'rand':[s[f'FIT+RAND-{i}'] for i in range(1,4)]}
                    for name, arms in controls.items():
                        differences = [paired_gain(arm,s['FIT+TRUE'],s['FROZEN']['B_ce']) for arm in arms]
                        one['gain_'+name] = average([d[0] for d in differences])
                        one['error_'+name] = average([d[1] for d in differences])
                    one['oracle_only_gain'] = (s['FROZEN']['B_ce']-s['ORACLE-only']['B_ce'])/s['FROZEN']['B_ce']
                    one['true_only_gain'] = (s['FROZEN']['B_ce']-s['TRUE-only']['B_ce'])/s['FROZEN']['B_ce']
                    one['query_gain_over_fit'] = (s['FIT-CE']['B_ce']-s['FIT+QRY-CE']['B_ce'])/s['FROZEN']['B_ce']
                    raw.append(one)
                seed_rows.append(dict(version=version,domain=domain,seed=seed,
                    **{k:average([r[k] for r in raw]) for k in raw[0]},
                    cos_valid_blocks=sum(x['cos_true'] is not None for x in raw),blocks=4))
            seeds = [r for r in seed_rows if r['version']==version and r['domain']==domain]
            means = {k:describe([r[k] for r in seeds]) for k in seeds[0] if k not in ('version','domain','seed')}
            def positive(key, errors=False):
                return sum(r[key] is not None and r[key]>(r['error_'+key[5:]] if errors else 0) for r in seeds)
            health = [r['health']['query'] for r in relevant]
            fallbacks = sum('numerical' in q['status'] for q in health)
            attempted = [q['calibration'] for q in health if 'calibration' in q]
            nonconverged = sum(not x['converged'] for x in attempted)
            numerical_ok = fallbacks/12<=.1 and (version=='v1' or bool(attempted) and nonconverged/len(attempted)<=.1)
            def mean_at_least(key, threshold, strict=False):
                value = means[key]['mean']
                return value is not None and (value>threshold if strict else value>=threshold)
            gates = dict(source_initialization=source_usable, numerical=numerical_ok,
                positive_cosine=mean_at_least('cos_true',0,True),
                cosine_vs_perm=mean_at_least('cos_adv_perm',.05) and positive('cos_adv_perm')>=2,
                cosine_vs_rand=mean_at_least('cos_adv_rand',.05) and positive('cos_adv_rand')>=2,
                coverage=mean_at_least('paired_valid',.8),
                gain_vs_fit=mean_at_least('gain_fit',1e-4) and positive('gain_fit',True)>=2)
            for control in ('query','mean','perm','rand'):
                gates['gain_vs_'+control] = mean_at_least('gain_'+control,0,True) and positive('gain_'+control,True)>=2
            tasks.append(dict(version=version,domain=domain,passed=all(gates.values()),gates=gates,
                metrics=means,numerical_fallback_blocks=fallbacks,calibration_attempts=len(attempted),
                nonconverged=nonconverged,positive_seed_counts={k:positive('gain_'+k,True) for k in ('fit','query','mean','perm','rand')}))
    selected = [v for v in ('v1','v2') if all(t['passed'] for t in tasks if t['version']==v)]
    return dict(source_initialization_passed=source_usable,tasks=tasks,selected_M2_versions=selected,
                M2_trajectories=6+12*len(selected) if selected else 0),seed_rows
