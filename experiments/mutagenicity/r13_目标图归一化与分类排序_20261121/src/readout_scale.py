"""Intervene once at the frozen sum-pooled readout and audit unlabeled Q actions."""
from fractions import Fraction
from hashlib import sha256
from statistics import mean, median
import json

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
import torch
from torch_geometric.data import Batch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs as r11, prior_run as storage
from tsba.pretrain import graph_batches
from . import historical, observation

ARMS = ('B0','Own','Individual','Other1','Other2','Proportional','Intercept','Frozen','EqualPositive')
GATE = ('B0','Individual','EqualPositive','OtherMean')


def identity(ids):
    return sha256(json.dumps(ids,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def order(setting,ids):
    return sorted(ids,key=lambda i:(sha256(f'r9-local-v1|{setting}|{i}'.encode('utf-8')).digest(),i))


def matched_roles(setting, source, manifests, old):
    # Validate S/V/A/H partitions and a common source. Only chosen V/R/Q
    # enter the graph reader and no H identifiers are exported in this audit.
    observation.roles_for_group(setting,manifests)
    a = {str(r['target']):r['ids']['adapt'] for r in manifests}
    ordered = {d:order(setting,ids) for d,ids in a.items()}
    query = {d:ids[::2] for d,ids in ordered.items()}
    references = {d:ids[1::2] for d,ids in ordered.items()}
    val = manifests[0]['ids']['validation']
    k = min(len(val),*(len(ids) for ids in references.values()))
    if k < 1:
        raise ValueError('R13 requires four nonempty equal-budget pools')
    references = {d:ids[:k] for d,ids in references.items()}
    references['V'] = order(setting,val)[:k]
    granted = {'S':manifests[0]['ids']['source'],'V':val,**a}
    read = {'S':granted['S'],'V':references['V'],
            **{d:query[d]+references[d] for d in query}}
    if (old['setting']!=setting or old['source']!=source or old['k']!=k
            or {d:identity(ids) for d,ids in granted.items()}!=old['sha256_granted_role_ids']
            or {d:identity(ids) for d,ids in read.items()}!=old['sha256_graph_ids_read']):
        raise PermissionError('R13 Q/R/V roles differ from the registered R9 audit')
    return query,references,k


class GraphScope:
    """Record only reads granted by this source group's disjoint input roles."""
    def __init__(self, store, query, references):
        self.store = store
        self.roles = {'V':references['V'],**{f'R{t}':v for t,v in references.items() if t!='V'},
                      **{f'Q{t}':v for t,v in query.items()}}
        flat = [i for ids in self.roles.values() for i in ids]
        if len(set(flat)) != len(flat):
            raise PermissionError('Q/R/V overlap')
        self.read = {}

    def take(self, role):
        if role not in self.roles or role in self.read:
            raise PermissionError('Unregistered or repeated graph read')
        ids = self.roles[role]
        rows = self.store.take(ids)
        self.read[role] = identity(ids)
        return rows


def b0_adapt(archive,task,all_ids,query):
    key = r11.key(**task)
    observed = archive.read('R7_OBS',f'blocks/{key}/tau-{archive.tau:.3f}-observed.npz')
    z = r11.npz_member(observed,['adapt_logits_B0'])['adapt_logits_B0']
    if z.shape!=(len(all_ids),2) or z.dtype!=np.float64 or not np.isfinite(z).all():
        raise ValueError('Invalid historical A logits')
    index = {i:n for n,i in enumerate(all_ids)}
    return z[[index[i] for i in query]]


def centered(z):
    return z - z.max(axis=1,keepdims=True)


@torch.inference_mode()
def native(model, pooled, scale):
    inputs = pooled * torch.as_tensor(scale,dtype=pooled.dtype).reshape(-1,1)
    u = model.encoder.readout[0](inputs)
    h = model.encoder.readout[2](torch.relu(u))
    z = model.head(h)
    return z.double().numpy(),u.double().numpy(),h


def margin(z):
    return z[:,1] - z[:,0]


def frozen_margin(baseline, preactivation, linear_input, output_weight, scale):
    delta = (scale-1.) * linear_input
    # Exact zero at the native input takes the one-sided branch for this step.
    change = np.where(preactivation>0.,delta,np.where(preactivation==0.,np.maximum(delta,0.),0.))
    return baseline + change@output_weight


@torch.inference_mode()
def interventions(model,graphs,batch_size,scales):
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise PermissionError('Readout audit requires the original frozen eval model')
    if not graphs or any(g.y is not None for g in graphs):
        raise PermissionError('Readout only accepts unlabeled graphs')
    captured = []
    prehook = model.encoder.readout.register_forward_pre_hook(lambda _m, a:captured.append(a[0]))
    ret = {key:[] for key in ('P','native_B0','historical_head_B0','crossings','strict_crossings','zero_departures','intercept_at_zero',
                               *[name for name in ARMS if name!='EqualPositive'])}
    last = model.encoder.readout[2].weight.double().numpy()
    w = model.encoder.readout[0].weight.double().numpy()
    v = (model.head.weight[1].double()-model.head.weight[0].double()).numpy()@last
    try:
        for ids in graph_batches(graphs,batch_size):
            batch = Batch.from_data_list([graphs[i] for i in ids])
            captured.clear()
            original = model.encoder(batch)
            if len(captured)!=1:
                raise ValueError('Expected one native readout of the concatenated node sums')
            p = captured[0]
            b0,u1,_ = native(model,p,np.ones(len(ids)))
            np.testing.assert_array_equal(b0,model.head(original).double().numpy())
            hist = (original.double()@model.head.weight.double().T+model.head.bias.double()).numpy()
            z1 = margin(b0)
            ret['P'].append(p.numpy())
            ret['native_B0'].append(b0)
            ret['historical_head_B0'].append(hist)
            ret['B0'].append(z1)
            t = float(scales['Own'])
            for arm in ('Own','Individual','Other1','Other2'):
                scale = np.asarray(scales[arm])
                z,u,_ = native(model,p,scale if scale.ndim==0 else scale[ids])
                ret[arm].append(margin(z))
                if arm=='Own':
                    ret['crossings'].append(((u>0.)!=(u1>0.)).sum(axis=1))
                    ret['strict_crossings'].append((((u<0.)&(u1>0.))|((u>0.)&(u1<0.))).sum(axis=1))
                    ret['zero_departures'].append(((u1==0.)&(u>0.)).sum(axis=1))
            intercept = margin(native(model,p,0.)[0])
            ret['intercept_at_zero'].append(intercept)
            ret['Proportional'].append(t*z1)
            ret['Intercept'].append(intercept+t*(z1-intercept))
            ret['Frozen'].append(frozen_margin(z1,u1,p.double().numpy()@w.T,v,t))
    finally:
        prehook.remove()
    output = {name:np.concatenate(parts) for name,parts in ret.items()}
    if not all(np.isfinite(x).all() for x in output.values()):
        raise ValueError('Nonfinite readout manipulation')
    return output


def compare_b0(arrays,expected,cfg):
    native = centered(arrays['native_B0'])
    historical_head = centered(arrays['historical_head_B0'])
    for candidate,key in ((native,'native_logit_tolerance'),(historical_head,'historical_head_tolerance')):
        np.testing.assert_allclose(candidate,expected,**cfg[key])
        np.testing.assert_array_equal(candidate.argmax(1),expected.argmax(1))
    return dict(native_max_abs_logit_error=float(np.max(np.abs(native-expected))),
                historical_head_max_abs_logit_error=float(np.max(np.abs(historical_head-expected))),
                decision_disagreements=0)


def spearman(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:
        return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def summarize_direction(arrays):
    z1 = arrays['B0']
    decision = {arm:(arrays[arm]>0.) for arm in ARMS if arm!='EqualPositive'}
    tie_z = np.column_stack((np.zeros(len(z1)),z1))
    decision['EqualPositive'] = observation.equal_positive(arrays['ids'],tie_z,int(decision['Own'].sum())).astype(bool)
    arrays.update({f'action_{name}':v for name,v in decision.items()})
    np.testing.assert_array_equal(decision['Proportional'],decision['B0'])
    rows = []
    n = len(z1)
    for arm in ARMS:
        measures = dict(median_abs_margin=None,p95_abs_margin=None,spearman_vs_b0=None,mean_abs_probability_change=None)
        if arm!='EqualPositive':
            z = arrays[arm]
            absolute = np.abs(z)
            measures = dict(median_abs_margin=float(np.median(absolute)),
                            p95_abs_margin=float(np.quantile(absolute,.95,method='linear')),
                            spearman_vs_b0=spearman(z,z1),
                            mean_abs_probability_change=float(np.abs(expit(z)-expit(z1)).mean()))
        c1 = int((decision['Own']!=decision[arm]).sum())
        c0 = int((decision['B0']!=decision[arm]).sum())
        rows.append(dict(arm=arm,n=n,positive_rate=float(decision[arm].mean()),
                         decisions_different_from_own=c1,decision_disagreement_from_own=c1/n,
                         decisions_different_from_b0=c0,decision_disagreement_from_b0=c0/n,**measures))
    return rows


def gate(rows,plan,cfg):
    exact = {}
    for r in rows:
        if r['arm'] in (*GATE[:-1],'Other1','Other2'):
            key = r['setting'],r['source'],r['target'],r['arm']
            exact[key] = Fraction(r['decisions_different_from_own'],r['n'])
    dims = [(s,t) for s in plan['domains'] for t in plan['domains'] if s!=t]
    cutoff = Fraction(*cfg['minimum_disagreement_fraction'])
    settings = {}
    for s in plan['settings']:
        values = {a:mean(exact[s,x,y,a] for x,y in dims) for a in GATE[:-1]}
        values['OtherMean'] = mean((exact[s,x,y,'Other1']+exact[s,x,y,'Other2'])/2 for x,y in dims)
        settings[s] = dict(consider=all(v>=cutoff for v in values.values()),
                           percent={name:float(100*v) for name,v in values.items()})
    return dict(consider_classification=settings['proteins_edge']['consider'] and any(
        row['consider'] for n,row in settings.items() if n.startswith('mutagenicity_')), settings=settings)


def run(out,repo,plan,cfg):
    archive = historical.CArchive(plan)
    checks = repo/cfg['r9_checks']
    if digest(checks)!=cfg['r9_checks_sha256']:
        raise PermissionError('Original R9 Q/R read receipts changed')
    original = json.loads(checks.read_bytes())
    if original['r7_c_final']!=plan['r7_final_digest']:
        raise PermissionError('R9 receipt is from a different R7 C')
    groups = {(r['setting'],r['source']):r for r in original['groups']}
    (out/'predictions').mkdir()
    stores, roles, directions, table = {}, {}, [], []
    for setting, source, seed, tasks in observation.group_tasks(plan,[plan['phase1_seed']]):
        first, sha, ctx = historical.context(archive,tasks[0],repo)
        model, lineage = historical.checkpoint(first,ctx)
        manifests = []
        for task in tasks:
            mf, auth = observation.bind(task,archive,repo,ctx,lineage,sha if task==tasks[0] else None)
            manifests.append(task|{'ids':mf['ids'],'manifest_sha256':auth})
        q, refs, k = matched_roles(setting,source,manifests,groups[setting,source])
        if setting not in stores:
            stores[setting] = historical.GraphStore(ctx)
        store = stores[setting]
        scope = GraphScope(store,q,refs)
        median_v = median(g.num_nodes for g in scope.take('V'))
        medians = {t:median(g.num_nodes for g in scope.take(f'R{t}')) for t in q}
        group = f'{setting}-M{source}-seed-{seed}'
        for row in manifests:
            task = {k:row[k] for k in observation.IDENTITY}
            t = str(task['target'])
            ids = q[t]
            graphs = scope.take(f'Q{t}')
            count = np.asarray([g.num_nodes for g in graphs],dtype=np.int64)
            others = sorted(set(q)-{t},key=int)
            if median_v<=0 or any(m<=0 for m in medians.values()) or np.any(count<=0):
                raise ValueError('Readout scale requires positive native node counts')
            scale = dict(Own=median_v/medians[t],Individual=median_v/count,
                         Other1=median_v/medians[others[0]],Other2=median_v/medians[others[1]])
            arrays = interventions(model,graphs,lineage['batch_size'],scale)
            expected = b0_adapt(archive,task,row['ids']['adapt'],ids)
            errors = compare_b0(arrays,expected,cfg)
            arrays.update(ids=np.asarray(ids),node_count=count,archived_B0=expected)
            table.extend(task|r for r in summarize_direction(arrays))
            key = r11.key(**task)
            storage.save_predictions(out,f'predictions/{key}.npz',arrays)
            tval = scale['Own']
            directions.append(dict(**task,n=len(ids),k=k,m_v=median_v,m_a=medians[t],own_t=tval,
                  own_t_sign='less' if tval<1. else 'greater' if tval>1. else 'equal',
                  other1_domain=int(others[0]),other1_median=medians[others[0]],other1_t=scale['Other1'],
                  other2_domain=int(others[1]),other2_median=medians[others[1]],other2_t=scale['Other2'],
                  activation_mask_changes=int(arrays['crossings'].sum()),
                  strict_crossings=int(arrays['strict_crossings'].sum()),
                  zero_departures=int(arrays['zero_departures'].sum()),
                  graphs_with_mask_change=int(np.count_nonzero(arrays['crossings'])),**errors))
            roles[key] = dict(**task,group=group,query_ids=ids,manifest_sha256=row['manifest_sha256'],
                              checkpoint=lineage,graph_archive_sha256=store.graph_sha)
        for row in manifests:
            roles[r11.key(**{k:row[k] for k in observation.IDENTITY})].update(
                reference_ids=refs,role_read_hashes=scope.read)
        print(f'R13-readout {group} K={k} V-median={median_v} A-medians={medians}',flush=True)
    if len(directions)!=36 or len(table)!=36*len(ARMS):
        raise ValueError('R13 readout did not cover the registered 36 blocks')
    fields = ('median_abs_margin','p95_abs_margin','positive_rate','spearman_vs_b0',
              'decision_disagreement_from_own','decision_disagreement_from_b0','mean_abs_probability_change')
    summary = []
    for s in plan['settings']:
        for arm in ARMS:
            subset = [r for r in table if r['setting']==s and r['arm']==arm]
            values = {k:[r[k] for r in subset if r[k] is not None] for k in fields}
            summary.append(dict(setting=s,arm=arm,directions=len(subset),
                                spearman_directions=len(values['spearman_vs_b0']),
                                **{k:mean(v) if v else None for k,v in values.items()}))
    write_table(out/'directions.csv',directions)
    write_table(out/'arms-by-direction.csv',table)
    write_table(out/'arms-by-setting.csv',summary)
    result = gate(table,plan,cfg)
    write_json(out/'gate.json',result)
    write_json(out/'roles.json',roles)
    write_json(out/'r7-access.json',archive.digests)
    files = ['executed-config.json','directions.csv','arms-by-direction.csv','arms-by-setting.csv',
             'gate.json','roles.json','r7-access.json',*storage.inventory(roles,'predictions')]
    checksum = storage.seal(out,'FINAL',files,dict(r7_final_digest=plan['r7_final_digest'],
                                                   r9_checks_sha256=cfg['r9_checks_sha256']))
    return checksum,result
