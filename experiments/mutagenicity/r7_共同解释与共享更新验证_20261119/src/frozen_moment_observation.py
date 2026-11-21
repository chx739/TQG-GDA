"""Verify the original B0/E/J predictions and seal candidate indices before reading A/H truth."""
import csv
import io
import json

import numpy as np
from scipy.special import softmax

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import inputs, prior_run
from . import fit, frozen_moment as attribution
from .p_risk_replay import read_npz
from .seals import verified_seal_chain


METHODS = ('B0', 'E', 'J')
POOLS = ('adapt', 'holdout')
MOVES = METHODS[1:]


def source_metadata(root, cfg):
    values = ('R7_OBS', 'R7_FINAL')
    hashes = {name: digest(root / 'seals' / f'{name}.json') for name in values}
    for name, pin in zip(values, ('source_obs_digest','source_final_digest'), strict=True):
        data = json.loads((root/'seals'/f'{name}.json').read_bytes())
        if data['digest'] != cfg[pin]:
            raise PermissionError(f'Original {name} version differs')
    return hashes


def tasks(cfg):
    return {
        f"{cfg['setting']}-M{cfg['source']}_to_M{t}-seed-{s}":dict(setting=cfg['setting'],source=cfg['source'],target=t,seed=s)
        for t in cfg['targets'] for s in cfg['seeds']
    }


def parents(root, cfg):
    return {f'{k}_sha256':v for k,v in source_metadata(root, cfg).items()}


def members(cfg):
    return ['executed-config.json','source-seals.json','roles.json','selections.json','candidate-risk-bounds.csv',
            *prior_run.inventory(tasks(cfg),'predictions')]


def read_plan(scope, cfg):
    provenance = json.loads(scope.read_bytes('provenance.json','R7_INPUT'))
    plan = json.loads(scope.read_bytes('executed-plan.json','R7_INPUT'))
    if (provenance['git_commit'] != cfg['source_execution_commit']
            or plan['shared_update']['primary_tau'] != cfg['tau']
            or plan['features']['output_dimension_real'] != cfg['dimensions']
            or plan.get('separation_variant') != 'restore_trust_v1' or plan.get('decision_policy') is not None):
        raise PermissionError('Historical fitting version, dimension, threshold or action rule differs')
    return plan


def frozen_block(scope, cfg, original, key, task):
    root = f'blocks/{key}'
    width = f"tau-{cfg['tau']:.3f}"
    manifest = json.loads(scope.read_bytes(f'{root}/manifest.json','R7_INPUT'))
    if any(manifest[k] != v for k,v in task.items()):
        raise PermissionError(f'Original R7 task identity differs: {key}')
    source = read_npz(scope,f'{root}/fit_input.npz','R7_INPUT')
    standard = read_npz(scope,f'{root}/standardization.npz','R7_OBS')
    observed = read_npz(scope,f'{root}/{width}-observed.npz','R7_OBS')
    records = json.loads(scope.read_bytes(f'{root}/{width}-observed.json','R7_OBS'))
    if not all(records[name]['valid'] for name in METHODS) or np.any(observed['w_B0']):
        raise PermissionError(f'Historical B0/E/J is invalid: {key}')
    held = read_npz(scope,f'{root}/{width}-heldout.npz','R7_OBS')
    held_standard = read_npz(scope,f'{root}/holdout_standardization.npz','R7_OBS')
    holdout = read_npz(scope,f'{root}/holdout_input.npz','R7_INPUT')
    if (source['adapt_h'].shape[1] != cfg['dimensions'] or holdout['holdout_h'].shape[1] != cfg['dimensions']
            or standard['source_u'].shape[1] != cfg['dimensions']+1
            or len(observed['theta']) != records['J']['candidate_count']):
        raise PermissionError(f'Historical feature or explanation dimensions differ: {key}')
    retained = observed['theta']
    candidate_records = records['J']['candidates']
    np.testing.assert_array_equal(retained, np.array([c['parameters'] for c in candidate_records]))
    np.testing.assert_array_equal(observed['stats'], np.array([c['statistic'] for c in candidate_records]))
    ref = observed['ref_theta']
    theta = np.vstack((ref,retained))
    params = fit.config(original, task['seed'], cfg['tau'])
    atol = cfg['absolute_replay_tolerance']
    arrays = dict(theta=theta, **{f'w_{name}':observed[f'w_{name}'] for name in METHODS})
    errors, banks = [],{}
    # Index -1 denotes the separately archived reference; the others retain their original zero-based order.
    for pool, f, st, saved in (('adapt',source,standard,observed),('holdout',holdout,held_standard,held)):
        ids,h,z0 = (f[f'{pool}_{k}'] for k in ('ids','h','z0'))
        np.testing.assert_array_equal(ids,manifest['ids'][pool])
        if len(np.unique(ids)) != len(ids):
            raise PermissionError(f'Duplicate graph ID: {key}/{pool}')
        u = fit.features(h,standard['mean'],standard['scale'])
        np.testing.assert_allclose(u,st[f'{pool}_u'],rtol=0.,atol=atol)
        np.testing.assert_array_equal(z0,st[f'{pool}_z0'])
        np.testing.assert_allclose(h @ source['source_head_weight'].T + source['source_head_bias'],z0,rtol=0.,atol=atol)
        fields = fit.explanation_family(original,standard['source_u'][:,:-1],source['source_y'],u[:,:-1],
                                        params,standard['v0'],source_mean=standard['mean'],source_scale=standard['scale'])
        q = np.stack([fields._target_terms(x)[-1] for x in theta])
        if not np.isfinite(q).all() or (q<0.).any() or (q>1.).any():
            raise ArithmeticError(f'Invalid frozen explanation probability: {key}/{pool}')
        np.testing.assert_allclose(q.sum(axis=2),1.,rtol=0.,atol=atol)
        if pool=='adapt':
            np.testing.assert_allclose(np.stack([fit.moment(v,u) for v in q[1:]]),observed['stats'],rtol=0.,atol=atol)
        arrays[f'{pool}_ids'], arrays[f'{pool}_q1'] = ids,q[:,:,1]
        result = {}
        for name in METHODS:
            predicted = fit.predict(name,z0,u,observed[f'w_{name}'])
            logits = saved[f'{pool}_logits_{name}']
            np.testing.assert_allclose(predicted,logits,rtol=0.,atol=atol)
            np.testing.assert_array_equal(predicted.argmax(1),logits.argmax(1))
            np.testing.assert_allclose(softmax(logits,axis=1),saved[f'{pool}_probability_{name}'],rtol=0.,atol=atol)
            errors.append(float(np.max(np.abs(predicted-logits))))
            arrays[f'{pool}_logits_{name}'] = logits
            result[name] = logits[:,1]-logits[:,0]
        banks[pool] = {name: attribution.risk_vector(result['B0'],result[name]) for name in MOVES}
    if np.intersect1d(arrays['adapt_ids'], arrays['holdout_ids']).size:
        raise PermissionError(f'A and H overlap: {key}')

    candidate_rows = []
    estimates = {p:{m:attribution.soft_risks(*vectors, arrays[f'{p}_q1']) for m,vectors in bank.items()}
                 for p,bank in banks.items()}
    vals = estimates['adapt']
    np.testing.assert_allclose(vals['E'][0],records['E']['known_candidate_risks'][0],rtol=0.,atol=atol)
    np.testing.assert_allclose(vals['J'][1:],records['J']['known_candidate_risks'],rtol=0.,atol=atol)
    pick,tied = attribution.most_adverse(vals['J'][1:],cfg['worst_tie_tolerance'])
    selection = dict(E=-1,J=pick,J_same_worst_ids=tied,maximum_prediction_error=max(errors),
                     archived_reference_retained_ids=np.flatnonzero(np.all(retained==ref,axis=1)).tolist())
    for pool, bank in banks.items():
        masses = [attribution.sum_mass(q) for q in arrays[f'{pool}_q1']]
        for name,vectors in bank.items():
            for i,mass in enumerate(masses):
                lo,hi = attribution.fixed_mass_bounds(*vectors,mass)
                risk = float(estimates[pool][name][i])
                if hi+atol < risk or lo-atol > risk or lo > hi+atol:
                    raise ArithmeticError(f'Candidate soft risk lies outside its exact mass bounds: {key}')
                candidate_rows.append(task | dict(key=key,pool=pool,method=name,candidate_id=i-1,
                    selected=i-1==selection[name],a_numeric_tie=name=='J' and i-1 in tied,
                    n=len(arrays[f'{pool}_ids']),soft_mass=float(mass),soft_mass_exact=str(mass),
                    candidate_prior=float(mass/len(arrays[f'{pool}_ids'])),risk=risk,bound_low=lo,bound_high=hi))
    cohort = task | {p:manifest['ids'][p] for p in POOLS}
    return arrays,selection,cohort,candidate_rows


def reconstruct(root, cfg):
    with verified_seal_chain(root,'R7_OBS') as scope:
        original = read_plan(scope,cfg)
        return {key:frozen_block(scope,cfg,original,key,task) for key,task in tasks(cfg).items()}


def freeze(out, root, cfg):
    (out/'predictions').mkdir()
    seals = parents(root,cfg)
    selections, roles, risks = {},{},[]
    for key,(arrays,selection,role,rows) in reconstruct(root,cfg).items():
        selections[key],roles[key] = selection,role
        prior_run.save_predictions(out,f'predictions/{key}.npz',arrays)
        risks.extend(rows)
        print(key,'J worst A',selection['J'],'tied',selection['J_same_worst_ids'],flush=True)
    for name, value in (('roles',roles),('selections',selections),('source-seals',seals)):
        write_json(out/f'{name}.json',value)
    write_table(out/'candidate-risk-bounds.csv',risks)
    return prior_run.seal(out,'OBS',members(cfg),seals)


def observed(out, root, cfg, obs_sha):
    seals = parents(root,cfg)
    frozen = prior_run.verified_members(out,'OBS',obs_sha,members(cfg),seals)
    cases = reconstruct(root,cfg)
    for name,value in (('roles',{k:r[2] for k,r in cases.items()}),
                       ('selections',{k:r[1] for k,r in cases.items()}),('source-seals',seals)):
        if json.loads(frozen[f'{name}.json']) != value:
            raise PermissionError(f'Sealed original-risk {name} does not reproduce')
    if json.loads(frozen['executed-config.json'])['plan'] != cfg:
        raise PermissionError('Sealed risk plan differs')
    actual = list(csv.DictReader(io.StringIO(frozen['candidate-risk-bounds.csv'].decode('utf-8'))))
    expected = [{k:str(v) for k,v in row.items()} for entry in cases.values() for row in entry[3]]
    if actual != expected:
        raise PermissionError('Original-risk candidate bounds do not reproduce')
    for key,(z,*_) in cases.items():
        saved = inputs.npz_member(frozen[f'predictions/{key}.npz'])
        if set(saved)!=set(z) or any(v.dtype!=saved[n].dtype or not np.array_equal(v,saved[n]) for n,v in z.items()):
            raise PermissionError(f'Original-risk margins or candidate probabilities differ: {key}')
    return cases
