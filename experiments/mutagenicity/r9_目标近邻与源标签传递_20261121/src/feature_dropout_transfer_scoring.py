"""Score the frozen source-internal supplement and report the registered decision."""
import csv
import json
from fractions import Fraction as F

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run, roc_score
from . import feature_dropout_transfer as exp, residual_inputs as data, true_rate_replay


def methods(setting):
    return (*exp.ORIGINAL,'Count',*exp.PROTEIN) if setting=='proteins_edge' else (*exp.ORIGINAL,'Count')


def mean(values):
    rows = list(values)
    return sum(rows,F(0))/len(rows)


def initial_summary(screen,cfg):
    source = screen.root/'directions.csv'
    if digest(source)!=cfg['published_endpoints_sha256']:
        raise PermissionError('Published initial screen summary differs')
    rows = list(csv.DictReader(source.read_text().splitlines()))
    results = []
    for setting in cfg['supplement_counts']:
        for arm in (*exp.ORIGINAL,'Count'):
            chosen = [r for r in rows if r['setting']==setting and r['method']==arm]
            if len(chosen)!=8:
                raise PermissionError('Published initial E missed a direction')
            counts = [int(r['n']) for r in chosen]
            correct = [round(float(r['accuracy'])*n) for r,n in zip(chosen,counts,strict=True)]
            gains = [F(int(r['fixed'])-int(r['harmed']),n) for r,n in zip(chosen,counts,strict=True)]
            results.append(dict(setting=setting,method=arm,directions=8,n=sum(counts),
                accuracy=float(mean(F(k,n) for k,n in zip(correct,counts,strict=True))),
                target_minus_method_pp=float(100*mean(gains)),
                corrected=sum(int(r['fixed']) for r in chosen),harmed=sum(int(r['harmed']) for r in chosen)))
    return results


def summary(cfg,rows,pairs):
    ends,contrasts,leave = [],[],[]
    for setting in cfg['supplement_counts']:
        for arm in methods(setting):
            endpoint = [r for r in rows if r['setting']==setting and r['method']==arm]
            a,f = [mean(F(r[key]) for r in endpoint) for key in ('accuracy_exact','macro_f1_exact')]
            ends.append(dict(setting=setting,method=arm,n=sum(r['n'] for r in endpoint),
                directions=len(endpoint),accuracy=float(a),macro_f1=float(f),accuracy_exact=str(a)))
            if arm=='Target':
                continue
            paired = [p for p in pairs if p['setting']==setting and p['right']==arm]
            values = [F(r['accuracy_exact']) for r in paired]
            delta = mean(values)
            contrasts.append(dict(setting=setting,left='Target',right=arm,directions=len(paired),
                n=sum(r['n'] for r in paired),accuracy_exact=str(delta),accuracy_pp=float(100*delta),
                macro_f1_pp=float(100*mean(F(r['macro_f1_exact']) for r in paired)),
                **{k:sum(r[k] for r in paired) for k in ('corrected','harmed','net','changed')},
                wins=sum(v>0 for v in values),ties=sum(v==0 for v in values),losses=sum(v<0 for v in values)))
            if arm=='Static':
                for source in range(4):
                    kept = [r for r in paired if r['source']!=source]
                    gain = mean(F(r['accuracy_exact']) for r in kept)
                    leave.append(dict(setting=setting,omitted_source=source,n=sum(r['n'] for r in kept),
                        directions=len(kept),accuracy_exact=str(gain),accuracy_pp=float(100*gain)))
    return ends,contrasts,leave


def decide(cfg,contrasts,leave):
    setting = 'proteins_edge'
    gaps = {r['right']:F(r['accuracy_exact']) for r in contrasts if r['setting']==setting}
    static = gaps['Static']
    flags = dict(at_least_half_pp=static>=F(*cfg['minimum_accuracy_fraction']),
                 leaves_positive=all(F(r['accuracy_exact'])>0 for r in leave if r['setting']==setting))
    flags.update({arm:gaps[arm]>0 for arm in ('Source','Perm','Count',*exp.PROTEIN)})
    return dict(eligibility='consider_real_proteins_only' if all(flags.values()) else (
                'stop_fixed_dropout' if static<=0 else 'local_effect_only'),checks=flags,
                accuracy_pp={m:float(100*v) for m,v in gaps.items()})


def score(out,stage,screen,cfg,sha,execution):
    pred = exp.observed(out,screen,cfg,sha)
    # Only after OBS replay, give the existing evaluation port the replacement
    # role. Its role name E is the reader slot; the IDs are the sealed U complement.
    eval_roles = {k:r | {'roles':{'E':r['supplement']}} for k,r in screen.roles.items()}
    endpoints,pairs = [],[]
    with data.ScopedLabels(screen.archive.by_setting,eval_roles,'replay_E') as port:
        for key,row in screen.roles.items():
            x = pred[key]
            with port.direction(key):
                y = port.load(key,'E',row['supplement'])
            if y.shape!=x['ids'].shape or not np.isin(y,[0,1]).all():
                raise ValueError('Supplemental label scope does not match predictions')
            ident = {k:row[k] for k in data.IDENTITY} | {'n':len(y)}
            scores = {}
            for arm in methods(row['setting']):
                confusion = roc_score.confusion(x[arm],y)
                a,f = F(confusion[0]+confusion[-1],len(y)),roc_score.f1(confusion)
                scores[arm] = a,f
                endpoints.append(dict(**ident,method=arm,accuracy_exact=str(a),macro_f1_exact=str(f),
                                      accuracy=float(a),macro_f1=float(f),**dict(zip(
                                          ('tn','fp','fn','tp'),confusion,strict=True))))
            for arm in methods(row['setting'])[1:]:
                counts = true_rate_replay.paired_counts(x['Target'],x[arm],y)
                a,f = (u-v for u,v in zip(scores['Target'],scores[arm],strict=True))
                if a!=F(counts['net'],len(y)):
                    raise ArithmeticError('Paired FLDA excess error is inconsistent')
                pairs.append(dict(**ident,left='Target',right=arm,accuracy_exact=str(a),
                                  macro_f1_exact=str(f),accuracy_pp=float(100*a),macro_f1_pp=float(100*f),**counts))
    data.validate_receipt(eval_roles,port.actual,'replay_E')
    absolute,paired,leave = summary(cfg,endpoints,pairs)
    initial = initial_summary(screen,cfg)
    tables = {'endpoints':endpoints,'paired-blocks':pairs,'summary':absolute,'paired-summary':paired,
              'leave-source':leave,'initial-summary':initial}
    for n,rows in tables.items():
        write_table(stage/f'{n}.csv',rows)
    write_json(stage/'gate.json',decide(cfg,paired,leave))
    write_json(stage/'score-execution.json',execution)
    write_json(stage/'label-access.json',dict(obs_sha256=sha,scored_supplement={k:r['E'] for k,r in port.actual.items()},
        original_E_labels_read=False,used_reference_labels_read=False,real_target_labels_read=False))
    (stage/'OBS.json').write_bytes((out/'OBS.json').read_bytes())
    return prior_run.seal(stage,'FINAL',['OBS.json','gate.json','score-execution.json','label-access.json',
                          *[f'{n}.csv' for n in tables]],dict(obs_sha256=sha))
