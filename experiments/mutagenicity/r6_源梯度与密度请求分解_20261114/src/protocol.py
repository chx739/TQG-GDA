"""Authenticate R5 and copy only fixed dependencies; never load evaluation values."""
import copy
from pathlib import Path
import shutil
from .shared import io, p3, p4, t4, digest, assert_nested_equal


def resolve(request, setting):
    plan=copy.deepcopy(request)
    plan['setting']=copy.deepcopy(next(s for s in request['settings'] if s['id']==setting))
    plan['workers']=request['execution']['max_workers']
    if plan['steps']['multipliers'] != [1.,.5,.25] or len(plan['factorial']['arms']) != 6:
        raise ValueError('R6 requires the registered six arms and three fixed steps')
    if {(a['source_gradient_coefficient'],a['request_sign']) for a in plan['factorial']['arms']} != {(a,q) for a in (0,1) for q in (-1,0,1)}:
        raise ValueError('Incomplete factorial design')
    p4.tasks(plan)
    return plan


def blocks(plan):
    return [(task,step) for task in p4.tasks(plan) for step in plan['states']['attempts_before']]


def block_path(root,task,step,folder='blocks'):
    return root/folder/task.key/f'before{step:02d}.pt'


def old_plan(root):
    return p3.read_json(root/'deps/protocol/plan.json')


def restore(root,task,step):
    old=old_plan(root);dep=root/'deps'
    model,pools=t4.inputs(dep,old,task)
    record=io.load(p4.job_dir(dep,old,task,'A0')/f'round{step:02d}.pt')
    previous=io.load(p4.job_dir(dep,old,task,'A0')/f'round{step-1:02d}.pt')
    schedule=io.load(p4.batch_path(dep,task))
    assert_nested_equal(record['batch'],schedule[step-1])
    assert_nested_equal(record['before'],previous['after'])
    pools['source_batch']=pools['source_fit'][record['batch']['source_positions']]
    y=io.load(p4.labels_path(dep,task.source,task.seed,'fit'))
    sigma=io.load(p4.source_dir(dep,old,task.source,task.seed)/f'geometry_M{task.target}.pt')['sigma']
    return model,pools,y,record,previous,sigma,p4.config(old,task.seed),old


def prepare(root,repo,plan):
    spec=plan['setting'];oldroot=Path(spec['r5_archive_root'])
    for item in plan['reference_artifacts']:
        if digest(repo/item['path']) != item['sha256']:
            raise ValueError(f"Reference changed: {item['path']}")
    seals={}
    for name in ('INPUT','OBS','FINAL'):
        path=oldroot/f'seals/R5_{name}.json'
        if digest(path) != spec['r5_seal_file_sha256'][name]:
            raise ValueError('Historical seal file mismatch')
        seal=io.verify(oldroot,f'R5_{name}')
        if seal['digest'] != spec['r5_seal_digests'][name]:
            raise ValueError('Historical seal digest mismatch')
        seals[name]=seal
    source_plan=oldroot/'protocol/plan.json'
    if digest(source_plan) != spec['r5_execution_plan_sha256']:
        raise ValueError('R5 execution plan changed')
    old=p3.read_json(source_plan)
    if p3.read_json(oldroot/'protocol/provenance.json')['git_commit'] != spec['r5_execution_commit']:
        raise ValueError('R5 execution version changed')
    for key in ('model','score'):
        if old[key] != spec[f'inherited_{key}_config']:
            raise ValueError(f'Inherited configuration differs: {key}')
    for key in ('seeds','domains','directions'):
        if old[key] != plan[key]:raise ValueError('Task registry changed')
    if old['setting']['domain_indices'] != spec['domain_indices'] or old['input'] != spec['input']:
        raise ValueError('Dataset setting differs')
    selected={'protocol/plan.json','protocol/provenance.json','inputs/domains.json','inputs/label_records.json','inputs/graph_hashes.json'}
    for source in plan['domains']:
        for seed in plan['seeds']:
            selected.add(f'roles/M{source}/seed{seed}.json')
            for split in ('fit','validation'):
                selected.add(str(p4.labels_path(oldroot,source,seed,split).relative_to(oldroot)))
            for name in ('source.pt','frozen.pt'):
                selected.add(str((p4.source_dir(oldroot,old,source,seed)/name).relative_to(oldroot)))
    for task in p4.tasks(plan):
        selected.add(str(p4.batch_path(oldroot,task).relative_to(oldroot)))
        selected.add(str((p4.source_dir(oldroot,old,task.source,task.seed)/f'geometry_M{task.target}.pt').relative_to(oldroot)))
        selected.add(str((p4.score_dir(oldroot,old,task)/f"epoch{old['model']['score_epochs']:04d}.pt").relative_to(oldroot)))
        for step in plan['states']['attempts_before']:
            for n in (step-1,step):
                selected.add(str((p4.job_dir(oldroot,old,task,'A0')/f'round{n:02d}.pt').relative_to(oldroot)))
    manifest={}
    for relative in sorted(selected):
        expected=seals['OBS']['files'][relative]
        source=oldroot/relative;target=root/'deps'/relative
        if digest(source)!=expected:raise ValueError('Dependency SHA mismatch')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if digest(target)!=expected:raise ValueError('Copied dependency SHA mismatch')
        manifest[relative]=dict(source=str(source),sha256=expected,bytes=source.stat().st_size)
    # Opaque label-file copying/hashing above does not deserialize evaluation labels.
    statuses=[]
    for task,step in blocks(plan):
        rec=io.load(p4.job_dir(root/'deps',old,task,'A0')/f'round{step:02d}.pt')
        statuses.append(dict(source=task.source,target=task.target,seed=task.seed,attempt_before=step,status=rec['side']['status']))
    missing=[dict(source=r['source'],target=r['target'],seed=r['seed'],attempt_before=r['attempt_before'],reason=r['status'])
             for r in statuses if r['status']!='available']
    key=lambda r:(r['source'],r['target'],r['seed'],r['attempt_before'])
    if sorted(missing,key=key)!=sorted(spec['r5_missing_blocks'],key=key):
        raise ValueError('Historical missing-block registry differs')
    if len(statuses)!=spec['planned_state_blocks'] or len(statuses)-len(missing)!=spec['expected_available_blocks_from_r5']:
        raise ValueError('Historical state count mismatch')
    io.write_json(root/'input-manifest.json',dict(root=str(oldroot),seals={k:v['digest'] for k,v in seals.items()},files=manifest))
    io.write_json(root/'registry.json',statuses)
