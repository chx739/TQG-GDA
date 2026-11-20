"""Post hoc same-seed R14 versus source-only WL on identical sealed H graphs."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np

from experiments.common.io import write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run as storage
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src.run import summarize_pairs
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src import historical
from experiments.mutagenicity.r13_目标图归一化与分类排序_20261121.src.scoring import auc
from experiments.mutagenicity.r14_节点类型与分类更新_20261121.src import scoring as r14_score

SEED = 3405
IDENTITY = ('setting','source','target','seed')
R14_FINAL = '04823138216678a643b6c2cba22ff2550844ded9aab05f1d5d795a56b9bde40a'
R14_ARMS = ('Base','Binding')
WL_ARMS = ('WL_fixed','WL_selected')
PAIRS = tuple((a,b) for a in R14_ARMS for b in WL_ARMS)


def read_archives(root, sha, role_name):
    raw = (root/'FINAL.json').read_bytes()
    final = json.loads(raw)
    if r11.sha(raw)!=sha or final['stage']!='FINAL':
        raise PermissionError('Matched-control FINAL identity differs')
    obs_raw = storage.read_member(root,final['files'],'OBS.json')
    obs = json.loads(obs_raw)
    if obs['stage']!='OBS' or final['predecessors']['obs_sha256']!=r11.sha(obs_raw):
        raise PermissionError('Matched-control OBS differs from FINAL')
    fit = storage.read_member(root,obs['files'],'FIT.json')
    if obs['predecessors']['fit_sha256']!=r11.sha(fit):
        raise PermissionError('Matched-control FIT differs from OBS')
    roles = json.loads(storage.read_member(root,obs['files'],role_name))
    return json.loads(fit),obs,roles


def read_h_role(repo, archive, key, primary, control):
    task = {k:primary[k] for k in IDENTITY}
    m,checksum,context = historical.context(archive,task,repo)
    if (any(primary[k]!=control[k] for k in IDENTITY) or task['seed']!=SEED or r11.key(**task)!=key
            or checksum!=primary['manifest_sha256'] or checksum!=control['manifest_sha256']
            or primary['held_ids']!=control['holdout_ids'] or primary['held_ids']!=m['ids']['holdout']):
        raise PermissionError(f'R14 and structural H role differ: {key}')
    return task,r11.raw_grant(context,m,'holdout')


def compare(repo, output, r14, wl, structural_final):
    if output.exists():
        raise FileExistsError(output)
    r14_fit,r14_obs,r14_roles = read_archives(r14,R14_FINAL,'prediction-roles.json')
    wl_fit,wl_obs,wl_roles = read_archives(wl,structural_final,'roles.json')
    actual = json.loads(storage.read_member(r14,r14_fit['files'],'executed-config.json'))
    plan = actual['historical_role_plan']
    c = historical.CArchive(plan)
    root = {'r7_final_digest':plan['r7_final_digest']}
    if r14_fit['predecessors']!=root or wl_fit['predecessors']!=root or actual['seeds']!=[SEED]:
        raise PermissionError('Expected one common authentic R7 C archive and seed')
    keys = {r11.key(s,i,j,SEED) for s in plan['settings'] for i in plan['domains'] for j in plan['domains'] if i!=j}
    if set(r14_roles)!=keys or {k for k,v in wl_roles.items() if v['seed']==SEED}!=keys:
        raise PermissionError('Expected exactly 36 common direction keys')
    frozen, grants = {}, {}
    for key in sorted(keys):
        role = r14_roles[key]
        task, grant = read_h_role(repo,c,key,role,wl_roles[key])
        group = task['setting'],task['target']
        if grants.setdefault(group,grant)!=grant:
            raise PermissionError('A shared H cohort has inconsistent label grants')
        relative = f'predictions/{key}.npz'
        x = r11.npz_member(storage.read_member(r14,r14_obs['files'],relative))
        h = r11.npz_member(storage.read_member(wl,wl_obs['files'],relative))
        r14_score.checked_predictions(role,x)
        np.testing.assert_array_equal(h['ids'],x['ids'])
        for endpoint in ('B0','SourceCal'):
            np.testing.assert_array_equal(x[f'logits_{endpoint}'],h[f'logits_{endpoint}'])
            np.testing.assert_array_equal(x[f'decision_{endpoint}'],h[f'action_{endpoint}'])
        score = {m:x[f'logits_{m}'] for m in R14_ARMS} | {m:h[f'logits_{m}'] for m in WL_ARMS}
        actions = {m:x[f'decision_{m}'] for m in R14_ARMS} | {m:h[f'action_{m}'] for m in WL_ARMS}
        for arm,z in score.items():
            if z.shape!=(len(x['ids']),2) or not np.isfinite(z).all():
                raise ValueError('Malformed matched logits')
            np.testing.assert_array_equal(actions[arm],z.argmax(axis=1))
        frozen[key] = task,x['ids'],actions,score
    reps,seed,minimum,_ = paired._settings(c.plan)
    points,draws,cohorts,receipts = {},{},{},{}
    endpoints,rows = [],[]
    with r11.GraphLabelSession('holdout') as labels:
        for key,(task,ids,actions,score) in frozen.items():
            s,i,j,model_seed = (task[k] for k in IDENTITY)
            group = (s,j)
            if group not in cohorts:
                y = labels.load(grants[group],ids.tolist())
                order = np.argsort(ids,kind='stable')
                cohorts[group] = y,order,paired._cohort(s,j,ids[order],y[order],reps,seed,minimum)
                receipts[key] = ids.tolist()
            y,order,cohort = cohorts[group]
            np.testing.assert_array_equal(ids[order],cohort['ids'])
            flags = {name:p==y for name,p in actions.items()}
            for name,z in score.items():
                endpoints.append(dict(**task,method=name,auroc=auc(z,y),
                    **evaluate.classification(z,y,actions['Base'],decisions=actions[name])))
            for a,b in PAIRS:
                p,q = flags[a],flags[b]
                corrected,harmed = int((p&~q).sum()),int((~p&q).sum())
                net = corrected-harmed
                contrast = (s,i,j,model_seed,a,b)
                points[contrast] = Fraction(net,len(y))
                if cohort['weights'] is not None:
                    draws[contrast] = cohort['weights']@(p[order].astype(float)-q[order])
                rows.append(dict(**task,left=a,right=b,n=len(y),accuracy_pp=float(100*points[contrast]),
                                 corrected=corrected,harmed=harmed,net_correct=net))
    output.mkdir(parents=True,exist_ok=False)
    write_table(output/'endpoints.csv',endpoints)
    write_table(output/'paired-blocks.csv',rows)
    write_table(output/'paired-summary.csv',summarize_pairs(plan|{'seeds':[SEED]},points,draws,PAIRS))
    write_json(output/'h-label-access.json',receipts)
    files = ['endpoints.csv','paired-blocks.csv','paired-summary.csv','h-label-access.json']
    parents = {'r14_final_sha256':R14_FINAL,'structural_final_sha256':structural_final}
    return storage.seal(output,'COMPARISON',files,parents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r14',required=True,type=Path)
    parser.add_argument('--structural',required=True,type=Path)
    parser.add_argument('--structural-final',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    result = compare(repo,args.output,args.r14,args.structural,args.structural_final)
    print('R14 versus WL seed 3405 comparison sealed',result,flush=True)


if __name__=='__main__':
    main()
