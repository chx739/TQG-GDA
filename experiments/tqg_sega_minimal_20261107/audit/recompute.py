"""Independent verification of completed registered M0/M1 records (no dataset labels)."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from collections import defaultdict
import torch

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(values):return statistics.mean(values)
def audit(root):
 root=Path(root)
 reg=json.loads((root/'registration.json').read_text())
 decision=json.loads((root/'decision.json').read_text())
 schedule=json.loads((root/'block_manifest.json').read_text())
 expected={(r['seed'],r['domain'],r['block']):r for r in schedule['blocks']}
 sources=json.loads((root/'source_models.json').read_text())
 assert len(sources)==3
 for s in sources:
  checkpoint=root/'encoders'/s['source_key']/'best.pt'
  assert digest(checkpoint)==s['checkpoint_sha256']
  history=[json.loads(x) for x in (checkpoint.parent/'history.jsonl').read_text().splitlines()]
  best=min(history,key=lambda r:r['source_validation']['ce'])
  assert len(history)==100 and history[-1]['updates']==700
  assert best['epoch']==s['selected_epoch']
  assert best['source_validation']==s['source_validation']
  for name,sha in json.loads((checkpoint.parent/'result.json').read_text())['artifacts'].items():
   assert digest(checkpoint.parent/name)==sha
 access=[json.loads(line) for line in (root/'label_access.jsonl').read_text().splitlines()]
 sealed=set(); accessed=0
 for event in access:
  token=event['token']
  folder=root/'M1'/token
  seal=json.loads((folder/'seal.json').read_text())
  row=seal['batches'];allowed=set(row['A']+row['B'])
  assert not set(row['A'])&set(row['B'])
  if event['event']=='unlabeled_query_sealed':
   assert set(event['ids'])==allowed
   for version,sha in seal['payloads'].items():assert digest(folder/(version+'_sealed.pt'))==sha
   sealed.add(token)
  else:
   assert event['event']=='label_access' and token in sealed and set(event['ids'])<=allowed
   accessed+=len(event['ids'])
 assert len(sealed)==24 and accessed==6888
 allrows=[];physical=0;logical=0;matrix_checks=0
 by_key=defaultdict(list)
 for folder in sorted((root/'M1').iterdir()):
  result=json.loads((folder/'result.json').read_text())
  for name,sha in result['artifacts'].items():assert digest(folder/name)==sha
  versions=result['versions'];assert len(versions)==2
  seal=json.loads((folder/'seal.json').read_text())
  br=seal['batches'];assert br==expected[(br['seed'],br['domain'],br['block'])]
  assert len(br['fit'])==16 and len(br['query'])==196 and len(br['A'])==16 and len(br['B'])==271
  for r in versions:
   q=torch.load(folder/(r['version']+'_sealed.pt'),weights_only=False,map_location='cpu')
   assert q['model_hash']==r['source_model_hash']==result['source_model_hash']
   f=q['feedback']
   if r['version']=='v1':
    x=f['source_features'];res=f['source_residual'];m=x.shape[-1]
    gram=torch.einsum('bcm,bcn->mn',x,x)/len(res)+.01*torch.eye(m,dtype=x.dtype)
    a=torch.linalg.solve(gram,torch.einsum('bcm,bc->m',x,res)/len(res))
    torch.testing.assert_close(a,f['coefficients'],rtol=0,atol=0)
    torch.testing.assert_close(f['target_features']@a,f['correction'],rtol=0,atol=0)
   else:
    torch.testing.assert_close(f['readout']@f['source_residual'],f['coefficients'],rtol=0,atol=0)
    expected_g=-(f['basis'].double()@f['coefficients']).to(q['snapshot'].dtype)
    torch.testing.assert_close(expected_g,q['gradients']['TRUE'],rtol=0,atol=0)
   for name,g in q['gradients'].items():
    h=hashlib.sha256(f'{g.dtype}|{tuple(g.shape)}|'.encode()+g.contiguous().numpy().tobytes()).hexdigest()
    assert h==r['health']['gradient_hashes'][name]
   matrix_checks+=1
   logical+=sum(s['arm']!='FROZEN' for s in r['steps'])
   physical+=sum(s['arm']!='FROZEN' and not s['reused_shared_step'] for s in r['steps'])
   steps={s['arm']:s for s in r['steps']};g={g['arm']:g for g in r['gradients']}
   values={'cos_true':g['TRUE']['cos_A'],'cos_adv_perm':g['TRUE']['cos_A']-mean([g[f'PERM-{i}']['cos_A'] for i in (1,2,3)]),
           'cos_adv_rand':g['TRUE']['cos_A']-mean([g[f'RAND-{i}']['cos_A'] for i in (1,2,3)]),
           'paired_valid':int(all(g[n]['cos_A'] is not None for n in ['TRUE']+[f'{k}-{i}' for k in ('PERM','RAND') for i in (1,2,3)]))}
   for name,controls in {'fit':['FIT-CE'],'query':['FIT+QRY-CE'],'mean':['FIT+MEAN-Y'],
                         'perm':[f'FIT+PERM-{i}' for i in (1,2,3)],'rand':[f'FIT+RAND-{i}' for i in (1,2,3)]}.items():
    true=steps['FIT+TRUE'];base=steps['FROZEN']['B_ce'];diffs=[];errs=[]
    for n in controls:
     c=steps[n];delta=c['B_ce']-true['B_ce'];diffs.append(delta/base)
     errs.append(max(c['numeric_floor'],true['numeric_floor'],abs(delta-(c['B_ce64_actual']-true['B_ce64_actual'])))/base)
    values['gain_'+name]=mean(diffs);values['error_'+name]=mean(errs)
   by_key[(r['version'],r['domain'],r['seed'])].append(values)
   allrows.append(r)
 assert logical==624 and physical==576 and matrix_checks==48
 seeds={k:{m:mean([x[m] for x in vs]) for m in vs[0]} for k,vs in by_key.items()}
 for task in decision['tasks']:
  vs=[seeds[(task['version'],task['domain'],s)] for s in (3405,3406,3407)]
  for metric in vs[0]:
   assert math.isclose(mean([v[metric] for v in vs]),task['metrics'][metric]['mean'],rel_tol=1e-12,abs_tol=1e-15)
  for control in ('fit','query','mean','perm','rand'):
   assert sum(v['gain_'+control]>v['error_'+control] for v in vs)==task['positive_seed_counts'][control]
  gates=dict(source_initialization=sum(s['source_usable'] for s in sources)>=2,numerical=True,
     positive_cosine=mean([v['cos_true'] for v in vs])>0,
     cosine_vs_perm=mean([v['cos_adv_perm'] for v in vs])>=.05 and sum(v['cos_adv_perm']>0 for v in vs)>=2,
     cosine_vs_rand=mean([v['cos_adv_rand'] for v in vs])>=.05 and sum(v['cos_adv_rand']>0 for v in vs)>=2,
     coverage=mean([v['paired_valid'] for v in vs])>=.8)
  relevant=[r for r in allrows if r['version']==task['version'] and r['domain']==task['domain']]
  fallbacks=sum('numerical' in r['health']['query']['status'] for r in relevant)
  solves=[r['health']['query']['calibration'] for r in relevant if 'calibration' in r['health']['query']]
  gates['numerical']=fallbacks/12<=.1 and (task['version']=='v1' or bool(solves) and sum(not x['converged'] for x in solves)/len(solves)<=.1)
  for control in ('fit','query','mean','perm','rand'):
   val=mean([v['gain_'+control] for v in vs])
   gates['gain_vs_'+control]=(val>=1e-4 if control=='fit' else val>0) and sum(v['gain_'+control]>v['error_'+control] for v in vs)>=2
  assert gates==task['gates'] and all(gates.values())==task['passed']
 assert decision['selected_M2_versions']==[] and json.loads((root/'M2_status.json').read_text())['updates']==0
 result=dict(status='passed',review_type='independent recomputation; no target dataset label access',
   execution_commit=reg['commit'],source_models=3,source_updates=2100,paired_blocks=24,version_blocks=48,
   matrix_identities_verified=48,temporary_logical_steps=logical,temporary_physical_steps=physical,
   development_label_accesses=accessed,development_tokens=len(sealed),final_label_accesses=0,
   gate_recomputation_matches=True,artifact_hashes_match=True,M2_updates=0)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run_directory');p.add_argument('--output',required=True);args=p.parse_args()
 result=audit(args.run_directory);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
