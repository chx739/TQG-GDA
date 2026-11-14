"""R2 immutable dependencies, atomic progress and source-only training inputs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import torch

from experiments.common.synthetic.archive import digest
from experiments.common.real_data.data import canonical_sha
from experiments.common.real_data.next_round.common import NextLabelBroker
from tsba.classifier import ResidualClassifier


FROZEN = ('deps','scores','trajectories','diagnostics','batches','protocol','source-labels')


def write_json(path,value):
    """Publish a complete JSON record atomically and exclusively."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    temp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    with temp.open('xb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
    os.link(temp,path);temp.unlink()


def seal(root,stage,paths):
    root=Path(root);files={}
    for item in paths:
        path=root/item
        if not path.exists():raise FileNotFoundError(path)
        for p in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if p.is_file():files[str(p.relative_to(root))]=digest(p)
    value=dict(stage=stage,files=dict(sorted(files.items())))
    value['digest']=canonical_sha(value)
    target=root/'seals'/f'{stage}.json'
    if target.exists():
        if json.loads(target.read_text())!=value:raise PermissionError('Existing seal differs')
    else:write_json(target,value)
    return value


def load(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def atomic_tensor(path,value):
    """Exclusive atomic publication; orphan partial files remain auditable."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    with temp.open('xb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.link(temp,path)  # Fails if a completed record already exists.
    temp.unlink()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify(root,stage):
    root=Path(root)
    record=json.loads((root/'seals'/f'{stage}.json').read_text())
    if record['stage']!=stage or canonical_sha({k:record[k] for k in ('stage','files')})!=record['digest']:
        raise PermissionError('Invalid seal identity')
    for rel,expected in record['files'].items():
        p=(root/rel).resolve()
        if not p.is_relative_to(root.resolve()) or digest(p)!=expected:
            raise PermissionError('Sealed evidence changed: '+rel)
    directories=('deps','batches','protocol') if stage=='INPUT' else FROZEN if stage=='OBS' else ()
    for name in directories:
        actual={str(p.relative_to(root)) for p in (root/name).rglob('*') if p.is_file()}
        expected={p for p in record['files'] if p.startswith(name+'/')}
        if actual!=expected:raise PermissionError('Frozen directory membership changed: '+name)
    return record


class Broker(NextLabelBroker):
    def _check(self,stage):
        # Only 22 scoped accesses are required, so favor transparent full checks.
        return verify(self.root,stage)['digest']

    def read(self,ids,role,purpose,**kwargs):
        if role=='target_A':
            with (self.root/'logs/label-access.jsonl').open('a') as f:
                f.write(json.dumps(dict(role=role,ids=list(ids),purpose=purpose,permitted=False,
                    reason='R2 does not train target-supervised references'))+'\n')
            raise PermissionError('R2 does not train target-supervised references')
        return super().read(ids,role,purpose,**kwargs)


def model_from_frozen(frozen):
    state=frozen['initial_head']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model=ResidualClassifier(frozen['x'].shape[1],2,state['U1.weight'].shape[0],dtype=torch.float64)
    model.load_state_dict(state);model.eval();model.requires_grad_(False)
    return model


def batches(fit_ids,target_ids,seed,rounds=30,size=128):
    rng=torch.Generator().manual_seed(seed+201)
    result=[]
    for attempt in range(1,rounds+1):
        s=torch.randperm(len(fit_ids),generator=rng)[:size].tolist()
        t=torch.randperm(len(target_ids),generator=rng)[:size].tolist()
        result.append(dict(attempt=attempt,source_positions=s,target_positions=t,
            source_ids=[fit_ids[i] for i in s],target_ids=[target_ids[i] for i in t],rng_after=rng.get_state().clone()))
    return result


def validate_roles(root,plan):
    root=Path(root);roles=json.loads((root/'deps/roles.json').read_text())
    if sorted(i for ids in roles['domains'].values() for i in ids)!=list(range(4337)):
        raise ValueError('Domain IDs do not partition all graphs')
    if sorted(roles['source']['domain_ids'])!=sorted(roles['domains']['0']) or len(roles['source']['domain_ids'])!=1084:
        raise ValueError('Source domain changed')
    for d in plan['domains']:
        t=roles['targets'][str(d)]
        if [len(t[k]) for k in ('A','B','final')]!=[271,271,542+(d==3)]:raise ValueError('Target counts changed')
        if sorted(t['A']+t['B']+t['final'])!=sorted(t['ids']) or sorted(t['ids'])!=sorted(roles['domains'][str(d)]):raise ValueError('Target partition changed')
        if len(set(t['query_ids']))!=64 or not set(t['query_ids']).issubset(t['A']):raise ValueError('Old queries changed')
    for seed in plan['seeds']:
        s=json.loads((root/f'deps/roles/seed{seed}.json').read_text())
        groups=[s[k+'_ids'] for k in ('fit','validation','diagnostic')]
        if list(map(len,groups))!=[779,109,196] or sorted(sum(groups,[]))!=sorted(roles['source']['domain_ids']):
            raise ValueError('Source role overlap/coverage/count mismatch')
    return roles


def prepare(root,plan):
    root=Path(root);old=Path(plan['dependency']['root'])
    seal=json.loads((old/'seals/OBS.json').read_text())
    if seal['digest']!=plan['dependency']['OBS_digest'] or canonical_sha({k:seal[k] for k in ('stage','files')})!=seal['digest']:
        raise ValueError('R1b observation identity mismatch')
    if digest(plan['dependency']['archive'])!=plan['dependency']['archive_sha256']:
        raise ValueError('R1b complete archive SHA mismatch')
    selected={'deps/roles.json','deps/inputs/broker.json','deps/inputs/class_mapping.json'}
    for seed in plan['seeds']:
        selected.add(f'deps/roles/seed{seed}.json')
        selected.update(f'deps/source/seed{seed}/{n}' for n in ('frozen.pt','predictions.pt','source.pt'))
        for d in plan['domains']:
            selected.add(f'deps/geometry/seed{seed}_domain{d}/banks/source.pt')
            for b in plan['bandwidth_multipliers']:
                prefix=f'scores/seed{seed}_domain{d}/b{b}'
                selected.update(prefix+'/'+n for n in ('epochs/epoch098.pt','epochs/epoch099.pt','selected.pt','metadata.json','history.json','static.pt'))
                selected.add(f'steps/seed{seed}_domain{d}/learned_b{b}/proposal.pt')
    records={}
    for rel in sorted(selected):
        if rel not in seal['files'] or digest(old/rel)!=seal['files'][rel]:raise ValueError('Historical dependency changed: '+rel)
        destrel=rel if rel.startswith('deps/') else 'deps/r1b/'+rel
        dst=root/destrel;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():raise FileExistsError(dst)
        shutil.copyfile(old/rel,dst)
        if digest(dst)!=seal['files'][rel]:raise ValueError('Dependency copy mismatch')
        records[destrel]=dict(original=rel,sha256=seal['files'][rel],bytes=dst.stat().st_size)
    roles=validate_roles(root,plan)
    checks=[]
    with torch.no_grad():
        for seed in plan['seeds']:
            f=load(root/f'deps/source/seed{seed}/frozen.pt');model=model_from_frozen(f)
            p=load(root/f'deps/source/seed{seed}/predictions.pt')
            torch.testing.assert_close(model(f['x']).softmax(-1),p['probabilities'],rtol=0,atol=0)
            torch.testing.assert_close((f['h'].double()-f['mean'])/f['scale'],f['x'],rtol=0,atol=0)
            s=json.loads((root/f'deps/roles/seed{seed}.json').read_text())
            for d in plan['domains']:
                schedule=batches(s['fit_ids'],roles['targets'][str(d)]['ids'],seed)
                atomic_tensor(root/f'batches/seed{seed}_domain{d}.pt',schedule)
            checks.append(dict(seed=seed,source_probability_exact=True,coordinates_exact=True,dimension=f['x'].shape[1]))
    write_json(root/'dependency-manifest.json',dict(original_root=str(old),OBS_digest=seal['digest'],files=records))
    write_json(root/'input-checks.json',dict(source_models=checks,source_role_counts=[779,109,196],target_A=271,target_B=271,role_partitions_verified=True,final_values_parsed=False))
    return roles


def jobs(plan):
    result=[]
    for seed in plan['seeds']:
        for domain in plan['domains']:
            for lam in plan['lambdas']:
                result.append(dict(seed=seed,domain=domain,lam=lam,kind='geometry',bandwidth=0,updates=0,sign=0))
                for b in plan['bandwidth_multipliers']:
                    for sign in (1,-1):
                        result.append(dict(seed=seed,domain=domain,lam=lam,kind='kde',bandwidth=b,updates=0,sign=sign))
                        for updates in plan['score_updates']:
                            result.append(dict(seed=seed,domain=domain,lam=lam,kind='learned',bandwidth=b,updates=updates,sign=sign))
    return result


def job_key(spec):
    return f"seed{spec['seed']}_domain{spec['domain']}/{spec['kind']}_b{spec['bandwidth']}_u{spec['updates']}_lambda{spec['lam']}_{'normal' if spec['sign']==1 else 'reverse' if spec['sign']==-1 else 'zero'}"


def pools(root,seed,domain):
    root=Path(root);r=json.loads((root/'deps/roles.json').read_text());s=json.loads((root/f'deps/roles/seed{seed}.json').read_text())
    return {f'B{domain}':r['targets'][str(domain)]['B'],**{'source_'+k:s[k+'_ids'] for k in ('fit','validation','diagnostic')}}


def predictions(model,features,ids):
    result={}
    with torch.no_grad():
        for name,which in ids.items():
            logits=model(features[which]);lp=logits.log_softmax(-1)
            if not torch.isfinite(lp).all():raise ValueError('Nonfinite predictions')
            result[name]=dict(ids=which,probabilities=logits.softmax(-1),log_probabilities=lp)
    return result
