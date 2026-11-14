"""Fixed twelve-direction denominators; descriptive conditional values are explicit."""
from fractions import Fraction
import math
import statistics
from .evaluate import POOLS,CONTRASTS

TARGET_GROUPS=('initially_correct','initially_wrong','generators','complement')


def summaries(plan,tables):
    identity=dict(setting=plan['setting']['id'])
    seeds,summary=[],[]
    families=(('paired',[f'{a}−{b}' for a,b in CONTRASTS],POOLS+TARGET_GROUPS,('nll','accuracy','balanced_accuracy','macro_f1')),
              ('interactions',['I'],POOLS+TARGET_GROUPS,('interaction',)),
              ('derivatives',['s/u'],POOLS,('d_s','d_u','d_u_after_source')))
    scopes=[('Avg',plan['directions'])]+[(f'M{s}→M{t}',[[s,t]]) for s,t in plan['directions']]
    for family,contrasts,pools,metrics in families:
        indexed={}
        for row in tables[family]:
            key=(row['attempt'],row['k'],row['arm'],row['pool'],row['source'],row['target'],row['seed'])
            if key in indexed:raise ValueError('Duplicate evaluation cell')
            indexed[key]=row
        for attempt in plan['states']['attempts_before']:
            for k in plan['steps']['multipliers']:
                for contrast in contrasts:
                    for pool in pools:
                        for subset in ('all_finite','common_motion_feasible'):
                            for scope,directions in scopes:
                                per_seed=[]
                                for seed in plan['seeds']:
                                    candidates=[indexed.get((attempt,k,contrast,pool,s,t,seed)) for s,t in directions]
                                    found=[r for r in candidates if r is not None and (subset=='all_finite' or r['common_motion_feasible'])]
                                    row=dict(**identity,family=family,attempt=attempt,k=k,arm=contrast,pool=pool,subset=subset,scope=scope,
                                             seed=seed,planned_blocks=len(directions),available_blocks=len(found))
                                    for metric in metrics:
                                        valid=[r for r in found if r.get(metric) is not None]
                                        values=[Fraction(r['net_correct'],r['n']) if metric=='accuracy' else r[metric] for r in valid]
                                        full=len(values)==len(directions)
                                        total=sum(values) if metric=='accuracy' else math.fsum(values)
                                        avg=total/len(values) if values else None
                                        row[metric]=float(avg) if full else None
                                        row[metric+'_conditional_mean']=float(avg) if avg is not None else None
                                        row[metric+'_valid_blocks']=len(values)
                                        row[metric+'_sign']=((avg>0)-(avg<0)) if full else None
                                    per_seed.append(row)
                                seeds.extend(per_seed)
                                for metric in metrics:
                                    complete=[r[metric] for r in per_seed if r[metric] is not None]
                                    signs=[r[metric+'_sign']*(-1 if metric not in ('accuracy','balanced_accuracy','macro_f1') else 1)
                                           for r in per_seed if r[metric] is not None]
                                    conditional=[r[metric+'_conditional_mean'] for r in per_seed if r[metric+'_conditional_mean'] is not None]
                                    summary.append(dict(**identity,family=family,attempt=attempt,k=k,arm=contrast,pool=pool,subset=subset,scope=scope,metric=metric,
                                        mean=statistics.mean(complete) if len(complete)==len(plan['seeds']) else None,
                                        sd=statistics.stdev(complete) if len(complete)==len(plan['seeds']) and len(complete)>1 else None,
                                        complete_seeds=len(complete),planned_seeds=len(plan['seeds']),wins=signs.count(1),ties=signs.count(0),losses=signs.count(-1),
                                        available_blocks=sum(r[metric+'_valid_blocks'] for r in per_seed),planned_blocks=len(directions)*len(plan['seeds']),
                                        conditional_available_seed_mean=statistics.mean(conditional) if conditional else None,
                                        conditional_seed_count=len(conditional)))
    return seeds,summary
