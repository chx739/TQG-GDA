"""R6 §5–7: supervised diagnostics only after this setting's R6_OBS."""
from collections import Counter
import traceback
from dataclasses import asdict
import torch
from torch.func import grad
from torch.nn.functional import cross_entropy
from tsba.classifier import ParameterMap
from tsba.utils import finite
from .shared import io, output, p3, p4, t3, subset_metrics, correction
from .protocol import blocks, block_path, restore, old_plan

POOLS=('source_batch','source_fit','source_validation','target')
CONTRASTS=(('U+','N'),('U-','N'),('U+','U-'),('SU+','S'),('SU-','S'),('SU+','SU-'),('S','N'),('SU+','U+'),('SU-','U-'))


def evaluation_labels(root,plan):
    seal=io.verify(root,'R6_OBS');io.verify(root,'R6_INPUT')
    replay=p3.read_json(root/'replay.json')
    required={'seals/R6_INPUT.json','registry.json','input-manifest.json','replay.json','protocol/plan.json'}
    required.update(str(block_path(root,t,s).relative_to(root)) for t,s in blocks(plan))
    required.update(str(p.relative_to(root)) for p in (root/'deps').rglob('*') if p.is_file())
    if not required<=seal['files'].keys() or replay['blocks']!=len(blocks(plan)):
        raise PermissionError('Incomplete R6 observation')
    if replay['candidate_cells']!=18*replay['statuses'].get('available',0) or replay['replayed_cells']!=replay['candidate_cells']:
        raise PermissionError('Candidate replay incomplete')
    meta=p3.read_json(root/'deps/inputs/label_records.json')
    targets={}
    for d in plan['domains']:
        values=p3.read_values(meta['path'],meta['offsets'],p4.domain(root/'deps',d),meta['sha256'])
        targets[d]=torch.tensor([meta['mapping'][str(v)] for v in values])
    output.write_json(root/'evaluation/label-access.json',dict(observation=seal['digest'],target_domains=plan['domains'],
                    purpose='fixed-candidate NLL and direction diagnostics; no feedback to candidates'))
    return targets


def directional(params,theta,x,y,s,u,plan):
    def loss(vector):return cross_entropy(params.logits(vector,x),y)
    gradient=grad(loss)(theta).detach();finite(gradient)
    finite(gradient.norm());finite(s.norm());finite(u.norm())
    result=dict(gradient_norm=float(gradient.norm()))
    for name,v in (('s',s),('u',u)):
        value=float(gradient@v)
        threshold=plan['numerics']['derivative_sign_atol']+plan['numerics']['derivative_sign_rtol']*float(gradient.norm())*float(v.norm())
        result.update({f'd_{name}':value,f'threshold_{name}':threshold,
                       f'sign_{name}':'near_zero' if abs(value)<=threshold else ('negative' if value<0 else 'positive'),
                       f'zero_direction_{name}':not bool(v.any())})
    combined=float(gradient@(s+u));finite(gradient.new_tensor([combined,result['d_s'],result['d_u']]))
    error=abs(combined-result['d_s']-result['d_u'])
    tolerance=plan['numerics']['derivative_sign_atol']+plan['numerics']['derivative_sign_rtol']*float(gradient.norm())*float((s+u).norm())
    if error>tolerance:raise ValueError('Directional derivative additivity failed')
    result['additivity_error']=error
    return result


def metrics(logp,y,baseline):
    result=subset_metrics(logp,y)
    if len(y):result.update(correction(logp,baseline,y))
    return result


def evaluate_block(model,pools,labels,record,plan):
    identity={key:record[key] for key in ('setting','source','target','seed','attempt')}
    if record['status']!='available':
        return dict(candidates=[],paired=[],derivatives=[],interactions=[],coverage=[dict(**identity,k=k,status=record['status'],
                 candidate_cells=0,common_motion_feasible=False) for k in plan['steps']['multipliers']])
    params=ParameterMap(model);theta,s,u=record['theta'],record['s'],record['u']
    candidates,paired,derivatives,interactions,coverage=[],[],[],[],[]
    derivative={pool:directional(params,theta,pools[pool],labels[pool],s,u,plan) for pool in POOLS}
    # Subsets are fixed using theta and generator IDs, after the observation gate.
    target_y=labels['target'];origin=torch.zeros(len(target_y),dtype=torch.bool);origin[record['origins']]=True
    correct=record['baseline']['target'].argmax(1)==target_y
    masks={pool:(pool,torch.ones(len(labels[pool]),dtype=torch.bool)) for pool in POOLS}
    masks.update({name:('target',mask) for name,mask in dict(initially_correct=correct,initially_wrong=~correct,
                                                          generators=origin,complement=~origin).items()})
    for k in plan['steps']['multipliers']:
        cells={c['arm']:c for c in record['candidates'] if c['k']==k}
        beta=k*record['alpha'];joint=all(c['motion_feasible'] for c in cells.values())
        coverage.append(dict(**identity,k=k,status='available',candidate_cells=len(cells),common_motion_feasible=joint,
                             feasible_arms=sum(c['motion_feasible'] for c in cells.values())))
        for pool in POOLS:
            after=directional(params,theta+beta*s,pools[pool],labels[pool],s,u,plan)
            derivatives.append(dict(**identity,k=k,pool=pool,arm='s/u',status='available',beta=beta,
                common_motion_feasible=joint,**derivative[pool],d_u_after_source=after['d_u'],
                threshold_u_after_source=after['threshold_u'],sign_u_after_source=after['sign_u']))
        for pool,(parent,mask) in masks.items():
            y=labels[parent][mask];baseline=record['baseline'][parent][mask]
            measured={arm:metrics(c['predictions'][parent][mask],y,baseline) for arm,c in cells.items()}
            for arm,c in cells.items():
                row=dict(**identity,k=k,arm=arm,pool=pool,status='available',alpha=record['alpha'],beta=beta,
                         motion_feasible=c['motion_feasible'],common_motion_feasible=joint,**measured[arm],
                         max_movement=c['max_movement'],movement_rms_sigma=c['movement_rms_sigma'],
                         source_linear_rms_sigma=c['source_linear_rms_sigma'],request_linear_rms_sigma=c['request_linear_rms_sigma'],
                         combined_linear_rms_sigma=c['combined_linear_rms_sigma'],norm_s=record['norm_s'],norm_u=record['norm_u'],
                         cosine_s_u=record['cosine_s_u'],single_root_count=c['single_root_count'],
                         relocation_anchors=len(c['relocation']['intersections']),
                         relocation_statuses=dict(Counter(r['status'] for r in c['relocation']['intersections'])),
                         root_rmse_vs_proxy_sigma=c['root_rmse_vs_proxy_sigma'])
                if len(y):
                    delta=measured[arm]['nll']-measured['N']['nll'];row['delta_nll']=delta
                    row['weighted_delta_nll']=delta*len(y)/len(labels[parent])
                    if pool in derivative:
                        d=derivative[pool];linear=beta*(c['a']*d['d_s']+c['q']*d['d_u'])
                        row.update(first_order_nll=linear,nonlinear_residual=delta-linear)
                if parent=='target':
                    geo=c['normalized_margin'];old=record['baseline_normalized_margin']
                    valid=mask & geo['valid'] & old['valid'];row['normalized_margin_valid']=int(valid.sum())
                    row['normalized_margin_change_rms']=float((geo['value'][valid]-old['value'][valid]).square().mean().sqrt()) if valid.any() else None
                candidates.append(row)
            for left,right in CONTRASTS:
                a,b=measured[left],measured[right]
                row=dict(**identity,k=k,pool=pool,arm=f'{left}−{right}',contrast=f'{left}−{right}',status='available',
                         common_motion_feasible=joint,n=len(y))
                if len(y):
                    row.update({metric:a[metric]-b[metric] if a.get(metric) is not None and b.get(metric) is not None else None
                                for metric in ('nll','accuracy','balanced_accuracy','macro_f1')})
                    row.update(correction(cells[left]['predictions'][parent][mask],cells[right]['predictions'][parent][mask],y))
                    row['accuracy']=row['net_correct']/len(y)
                    row['weighted_nll']=row['nll']*len(y)/len(labels[parent])
                    for metric in ('accuracy','balanced_accuracy','macro_f1'):
                        row[metric+'_pp']=100*row[metric] if row[metric] is not None else None
                    if row['net_correct']!=a['correct']-b['correct']:raise ValueError('Correction conservation failed')
                paired.append(row)
            value=(measured['SU+']['nll']-measured['S']['nll']-measured['U+']['nll']+measured['N']['nll']) if len(y) else None
            interactions.append(dict(**identity,k=k,pool=pool,arm='I',status='available',common_motion_feasible=joint,
                                     n=len(y),interaction=value))
        # The two fixed subgroup partitions must add exactly to the full target change.
        for left,right in CONTRASTS:
            chosen={r['pool']:r for r in paired if r['k']==k and r['contrast']==f'{left}−{right}'}
            for groups in (('initially_correct','initially_wrong'),('generators','complement')):
                difference=sum(chosen[g].get('weighted_nll',0.) for g in groups)-chosen['target']['nll']
                if abs(difference)>1e-10+1e-8*abs(chosen['target']['nll']):raise ValueError('Subgroup NLL conservation failed')
    return dict(candidates=candidates,paired=paired,derivatives=derivatives,interactions=interactions,coverage=coverage)


def worker(root,plan,item):
    task,step,labels=item
    out=root/'evaluation/blocks'/task.key/f'before{step:02d}.json'
    if out.exists():return
    rec=io.load(block_path(root,task,step))
    try:
        model,pools,*_=restore(root,task,step) if rec['status']=='available' else (None,None)
        # No target gradients are saved or passed back to candidate generation.
        result=evaluate_block(model,pools,labels,rec,plan)
    except (*t3.NUMERICAL_ERRORS,ValueError,AssertionError,KeyError) as error:
        status='numerical_failure' if isinstance(error,t3.NUMERICAL_ERRORS) else 'engineering_failure'
        failed=dict(rec,status=status)
        result=evaluate_block(None,None,None,failed,plan)
        result['error']=traceback.format_exc()
        result['candidate_status']=rec['status']
    io.write_json(out,result)
