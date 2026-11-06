"""Derive omitted diagnostic columns from already sealed vectors; no labels/data read."""
import argparse,csv,json
from pathlib import Path
import torch
parser=argparse.ArgumentParser();parser.add_argument('run_directory');parser.add_argument('--output',required=True);args=parser.parse_args()
p=Path(args.run_directory)
out=Path(args.output)
rows=[]
def cos(a,b):
 den=float(a.norm()*b.norm());return float(a@b)/den if den>1e-24 else None
for folder in sorted((p/'M1').iterdir()):
 result=json.loads((folder/'result.json').read_text())
 oracle=torch.load(folder/'oracle_gradients.pt',weights_only=False,map_location='cpu')
 ga,gb=oracle['A'].double(),oracle['B'].double()
 for record in result['versions']:
  payload=torch.load(folder/(record['version']+'_sealed.pt'),weights_only=False,map_location='cpu')
  gf,gq=payload['source_fit_gradient'].double(),payload['query_ce_gradient'].double()
  space=payload['space'].double();projected=space@(space.T@gb)
  arms={k:g.double() for k,g in payload['gradients'].items()}
  arms.update({'CENTERED-Y':arms['TRUE']-arms['MEAN-Y'],'QUERY-CE':gq,'ORACLE-A':ga,'FIT-CE':gf})
  for arm,g in arms.items():
   rows.append(dict(seed=record['seed'],domain=record['domain'],block=record['block'],version=record['version'],arm=arm,
    norm=float(g.norm()),oracle_A_norm=float(ga.norm()),oracle_B_norm=float(gb.norm()),source_fit_norm=float(gf.norm()),
    cos_A=cos(g,ga),cos_B=cos(g,gb),cos_source_fit=cos(g,gf),cos_B_projected=cos(g,projected),
    B_space_energy=float(projected.square().sum()/gb.square().sum()),
    relative_error_A=float((g-ga).norm()/ga.norm()),weighted_norm_over_source=float(.1*g.norm()/gf.norm()),
    gradient_execution_space_dim=space.shape[1]))
with out.open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print({'derived_rows':len(rows),'new_training':False,'new_label_access':False})
