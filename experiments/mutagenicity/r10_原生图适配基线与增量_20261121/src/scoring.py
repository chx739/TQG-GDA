"""Seal all R10 predictions before opening the matched H labels."""
import json
from pathlib import Path
import subprocess
from uuid import uuid4

import numpy as np
import torch

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import evaluate, paired, r5_inputs, seals
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src import inputs as r8_inputs, pilot
from . import data, jobs, training

HISTORICAL = ('B0','SourceCal','SourceFusion','J_argmax','Joint','TargetSlope')
METHODS = ('GRADE','SourceOnly',*HISTORICAL)
COMPARISONS = (('GRADE','SourceOnly'),*((arm,ref) for arm in ('GRADE','SourceOnly') for ref in HISTORICAL))


def _hash_members(root, members):
    return {rel:digest(root/rel) for rel in members}


def _upstream_role_arrays(repo, plan, roles):
    parent = data.parent_dir(repo)
    donor_plan = json.loads((parent/'executed-config.json').read_bytes())['plan']
    frozen = pilot.verified_observations(parent, plan['parent_r8_obs_sha256'], donor_plan)
    mass = parent.parent/'fixed-mass-diagnostic'
    final = (mass/'FINAL.json').read_bytes()
    if r8_inputs.sha(final) != plan['fixed_mass_final_sha256']:
        raise PermissionError('Fixed-mass comparison has the wrong FINAL identity')
    final = json.loads(final)
    for file, expected in final['files'].items():
        if digest(mass/file) != expected:
            raise PermissionError(f'Fixed-mass FINAL evidence differs: {file}')
    obs_blob = (mass/'OBS.json').read_bytes()
    if r8_inputs.sha(obs_blob) != final['observation_sha256']:
        raise PermissionError('Fixed-mass OBS does not match its FINAL')
    obs = json.loads(obs_blob)
    if obs['parent_final_sha256'] != digest(parent/'FINAL.json'):
        raise PermissionError('Fixed-mass comparison does not refer to this R8 attempt')
    data_out, stored = {}, {}
    for key, item in roles['tasks'].items():
        rel = f'predictions/{key}-H.npz'
        blob = (mass/rel).read_bytes()
        if r8_inputs.sha(blob) != obs['files'][rel]:
            raise PermissionError(f'Fixed-mass H predictions differ: {key}')
        historical_r8 = r8_inputs.npz_member(frozen[rel], ['ids', *(f'{stem}_{arm}' for stem in ('z','decision')
                                                                 for arm in HISTORICAL if arm != 'TargetSlope')])
        alternative = r8_inputs.npz_member(blob, ['ids','z_TargetSlope','decision_TargetSlope'])
        np.testing.assert_array_equal(historical_r8['ids'], item['h_ids'])
        np.testing.assert_array_equal(alternative['ids'], item['h_ids'])
        stored[key] = dict(r8_sha256=r8_inputs.sha(frozen[rel]), fixed_mass_sha256=r8_inputs.sha(blob),
                           h_ids=item['h_ids'])
        data_out[key] = historical_r8 | alternative
    return data_out, stored


def complete_training(root, roles, plan):
    complete = []
    epochs = plan['model']['epochs']
    expected_jobs = [key for key,_,_ in jobs.grid(plan)]
    for name in expected_jobs:
        if not (root/'runs'/name/'DONE.json').exists():
            raise PermissionError(f'All 15 models must finish before R10 H evaluation. Missing {name}.')
    for name, seed, target in jobs.grid(plan):
        attempt, done = jobs.read_done(root,name)
        summary = json.loads((attempt/'summary.json').read_bytes())
        if (done['seed'] != seed or done['target'] != target or summary['updates'] != epochs
                or summary['epochs'] != epochs or summary['input_sha256'] != digest(root/'INPUT.json')):
            raise ValueError(f'Invalid matched training identity or step count: {name}')
        s = roles['source_fit'][str(seed)]
        a = roles['tasks'][jobs.target_key(seed,target)]['a_ids'] if target is not None else []
        trace = [json.loads(line) for line in (attempt/'epochs.jsonl').read_text().splitlines()]
        trace = [row for row in trace if row['event']=='epoch']
        if ([row['epoch'] for row in trace] != list(range(1,epochs+1)) or
            any((row['updates'],row['source_graphs'],row['target_adapt_graphs']) != (1,len(s),len(a)) for row in trace)):
            raise ValueError(f'Incomplete full-batch coverage: {name}')
        complete.append((name, seed, target, attempt, summary))
    for seed in plan['seeds']:
        group = [summary for _,s,_,_,summary in complete if s==seed]
        if (len({x['initial_parameter_sha256'] for x in group}) != 1
                or len({x['source_role_hash'] for x in group}) != 1):
            raise ValueError(f'Source role or initialization not matched: seed {seed}')
    return complete


def make_observation(root, roles, plan, repo):
    complete = complete_training(root,roles,plan)
    held_members = [f'data/H-M{target}.pt' for _,target in plan['directions']]
    data.read_inputs(root, requested=held_members)
    held = {target:torch.load(root/f'data/H-M{target}.pt',map_location='cpu',weights_only=True)
            for _,target in plan['directions']}
    historical, lineage = _upstream_role_arrays(repo,plan,roles)
    destination = root/'observation'
    if destination.exists():
        destination.rename(root/f'unsealed-observation-{uuid4().hex}')
    results = root/f'.observation-{uuid4().hex}'
    results.mkdir()
    combined = {}
    max_error = 0.
    all_files = ['INPUT.json','roles.json','executed-config.json',*held_members]
    for name,seed,target,attempt,summary in complete:
        model = training.new_model(plan['model'],roles['node_features'],roles['classes'],plan['execution']['device'])
        state = torch.load(attempt/'model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state)
        if training.state_hash(model) != summary['final_parameter_sha256']:
            raise ValueError(f'Reloaded checkpoint parameter hash differs: {name}')
        arm = 'SourceOnly' if target is None else 'GRADE'
        for t in summary['prediction_targets']:
            key = jobs.target_key(seed,t)
            ids = roles['tasks'][key]['h_ids']
            archived = torch.load(attempt/f'prediction-M{t}.pt',map_location='cpu',weights_only=True)
            replay = training.infer(model,held[t],ids,plan)
            if archived['ids'].tolist() != ids or not torch.equal(archived['decisions'],replay['decisions']):
                raise ValueError(f'{name}: reloaded H identities or decisions canonical order differ')
            error = float((replay['logits']-archived['logits']).abs().max())
            max_error = max(max_error,error)
            torch.testing.assert_close(replay['logits'],archived['logits'],atol=1e-7,rtol=1e-6)
            values = combined.setdefault(key,{})
            values['logits_'+arm] = archived['logits'].numpy().astype(np.float64)
            values['actions_'+arm] = archived['decisions'].numpy()
        base = attempt.parent
        all_files.append(str((base/'DONE.json').relative_to(root)))
        all_files.extend(str((attempt/rel).relative_to(root)) for rel in json.loads((base/'DONE.json').read_bytes())['files'])
    if set(combined) != set(roles['tasks']):
        raise ValueError('R10 ten-block predictions are incomplete')
    for key, observed in combined.items():
        expected = {'logits_GRADE','actions_GRADE','logits_SourceOnly','actions_SourceOnly'}
        if observed.keys() != expected:
            raise ValueError(f'{key}: missing paired GRADE and source-only predictions')
        for name in HISTORICAL:
            z = historical[key]['z_'+name]
            observed['logits_'+name] = np.column_stack((np.zeros_like(z),z))
            observed['actions_'+name] = historical[key]['decision_'+name]
        path = results/f'{key}.npz'
        observed['ids'] = np.asarray(roles['tasks'][key]['h_ids'])
        with path.open('xb') as stream:
            np.savez_compressed(stream,**observed)
        all_files.append(f'observation/{path.name}')
    write_json(results/'lineage.json', dict(models=15,blocks=10,checkpoint_replay_max_abs_logit_error=max_error,
                                             historical=lineage, evaluation_worktree=str(repo)))
    all_files.append('observation/lineage.json')
    results.rename(destination)
    write_json(root/'OBS.json', dict(stage='R10_OBS', input_sha256=digest(root/'INPUT.json'),
                                     blocks=len(combined), models=len(complete), files=_hash_members(root,all_files)))


def verify_observation(root,roles):
    obs_blob = (root/'OBS.json').read_bytes()
    obs = json.loads(obs_blob)
    expected = {f'observation/{key}.npz' for key in roles['tasks']}
    if (obs['stage'] != 'R10_OBS' or obs['input_sha256'] != digest(root/'INPUT.json')
            or obs['blocks'] != 10 or obs['models'] != 15 or not expected <= obs['files'].keys()
            or obs['files'] != _hash_members(root,obs['files'])):
        raise PermissionError('Invalid R10 observation seal')
    return r8_inputs.sha(obs_blob), {key:r8_inputs.npz_member((root/f'observation/{key}.npz').read_bytes())
                                    for key in roles['tasks']}


def score(root, repo, roles, plan, output):
    obs_sha, frozen = verify_observation(root,roles)
    endpoint_rows, pairs, cohorts, label_ids = [],[],[],{}
    archive = Path(roles['r7_archive'])
    original = roles['r7_plan']
    bootstrap = {}
    reps, bs_seed, minimum, quantiles = paired._settings(original)
    with seals.verified_seal_chain(archive,'R7_FINAL') as input_scope:
        if json.loads((archive/'seals/R7_FINAL.json').read_bytes())['digest'] != roles['r7_final_digest']:
            raise PermissionError('R10 referenced a different R7 evaluation identity')
        with r5_inputs.evaluation_labels(archive,repository=repo) as labels:
            y_by_target = {}
            for key, task in roles['tasks'].items():
                ids = frozen[key]['ids']
                np.testing.assert_array_equal(ids,task['h_ids'])
                target = task['target']
                manifest = input_scope.read_bytes(task['r7_manifest'], 'R7_INPUT')
                if r8_inputs.sha(manifest) != task['r7_manifest_sha256']:
                    raise PermissionError('R10 holdout role manifest changed')
                if target not in y_by_target:
                    y_by_target[target] = labels.load(task['r7_manifest'],'holdout')
                    label_ids[str(target)] = task['h_ids']
                    order = np.argsort(ids,kind='stable')
                    bootstrap[target] = (order,paired._cohort(task['setting'],target,ids[order],
                        y_by_target[target][order],reps,bs_seed,minimum))
                elif label_ids[str(target)] != task['h_ids']:
                    raise ValueError('A shared H role changed across seeds')
                y = y_by_target[target]
                identity = {field:task[field] for field in ('setting','source','target','seed')}
                x = frozen[key]
                logits = {name:x['logits_'+name] for name in METHODS}
                actions = {name:x['actions_'+name] for name in METHODS}
                cohorts.append(dict(**identity,ids=ids,y=y,logits=logits,decisions=actions))
                by_method = {}
                for name in METHODS:
                    metrics = evaluate.classification(logits[name],y,actions['SourceOnly'],decisions=actions[name])
                    by_method[name] = metrics
                    endpoint_rows.append(dict(**identity,method=name,**metrics))
                order,weights = bootstrap[target]
                for left,right in COMPARISONS:
                    l,r = actions[left]==y,actions[right]==y
                    corrected,harmed = int((l&~r).sum()),int((~l&r).sum())
                    interval = [None,None]
                    if weights['eligible']:
                        change = l.astype(float)-r.astype(float)
                        interval = np.percentile(100*weights['weights']@change[order],quantiles).tolist()
                    pairs.append(dict(**identity,left=left,right=right,n=len(y),corrected=corrected,harmed=harmed,
                                      net=corrected-harmed,accuracy_pp=100*(corrected-harmed)/len(y),
                                      accuracy_ci_low=interval[0],accuracy_ci_high=interval[1],
                                      nll_delta=by_method[left]['sampled_nll']-by_method[right]['sampled_nll'],
                                      ba_delta_pp=100*(by_method[left]['balanced_accuracy']-by_method[right]['balanced_accuracy'])))
    if verify_observation(root,roles)[0]!=obs_sha:
        raise PermissionError('R10 observation changed during label evaluation')
    intervals = paired.paired_bootstrap(original,cohorts,phase='P',comparisons=COMPARISONS)
    intervals['interval_rows'] = [row for row in intervals['interval_rows'] if row['metric']!='directional_risk_gap']
    for field in ('directional_gap','candidate_max_recomputed_each_replicate'):
        intervals['metadata'].pop(field,None)
    pilot.write_table(output/'endpoints.csv',endpoint_rows)
    pilot.write_table(output/'paired-seeds.csv',pairs)
    write_json(output/'paired-intervals.json',intervals)
    write_json(output/'label-access.json',dict(obs_sha256=obs_sha,target_ids=label_ids,evaluated_blocks=len(cohorts)))
    return obs_sha


def run(root,repo):
    roles,config = data.read_inputs(root)
    plan = config['plan']
    torch.set_num_threads(plan['execution']['torch_threads_per_worker'])
    torch.set_num_interop_threads(plan['execution']['interop_threads'])
    torch.use_deterministic_algorithms(True)
    with jobs.lock(root/'supervisor.lock'):
        if (root/'FINAL.json').exists():
            final = json.loads((root/'FINAL.json').read_bytes())
            if final['files'] != _hash_members(root,final['files']):
                raise PermissionError('R10 final artifacts changed')
            verify_observation(root,roles)
            print('R10 FINAL verified',flush=True)
            return
        if not (root/'OBS.json').exists():
            make_observation(root,roles,plan,repo)
            print('R10 OBS sealed',digest(root/'OBS.json'),flush=True)
        if (root/'evaluation').exists():
            raise FileExistsError('An existing R10 evaluation needs a separate recovery directory')
        staging = root/f'.evaluation-{uuid4().hex}'
        staging.mkdir()
        commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
        write_json(staging/'execution.json',dict(code_commit=commit,scoring_sha256=digest(Path(__file__)),
                     trained_with=config['code_commit'],started_utc=jobs.now()))
        obs_sha = score(root,repo,roles,plan,staging)
        staging.rename(root/'evaluation')
        output = ['OBS.json',*(f'evaluation/{p.name}' for p in (root/'evaluation').iterdir())]
        write_json(root/'FINAL.json',dict(stage='R10_FINAL',observation_sha256=obs_sha,blocks=10,
                                         score_label_pools=['holdout'],files=_hash_members(root,output)))
        print('R10 FINAL sealed',digest(root/'FINAL.json'),flush=True)
