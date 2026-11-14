"""Independent from-zero CPU smoke test; ALL generated files stay outside this package."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch_geometric.data import Data,InMemoryDataset
from sega_gda.data import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output).resolve()
    if output.is_relative_to(ROOT):raise ValueError('Smoke output must be OUTSIDE the package')
    output.mkdir(parents=True,exist_ok=False)
    inputs=output/'synthetic_inputs';inputs.mkdir();idx=inputs/'idx';idx.mkdir()
    graphs=[]
    for i in range(18):
        x=torch.tensor([[.2+.12*i,.4+.15*(i%3)],[.4+.08*i,.3+.1*(i%2)],[.3+.07*i,.5+.06*(i%4)]],dtype=torch.float64)
        y=i%2 if i<12 else 999  # Poison target labels: never used by adaptation.
        graphs.append(Data(x=x,edge_index=torch.tensor([[0,1,1,2,2,0],[1,0,2,1,0,2]]),y=torch.tensor([y])))
    data,slices=InMemoryDataset.collate(graphs);cache=inputs/'data.pt'
    torch.save((data.to_dict(),slices,dict(num_node_attributes=2,num_node_labels=0,num_edge_attributes=0,num_edge_labels=0),Data),cache)
    np.save(idx/'idx_fixture_0.npy',np.arange(12));np.save(idx/'idx_fixture_1.npy',np.arange(12,18))
    split=inputs/'source_split.json'
    split.write_text(json.dumps(dict(dataset='fixture',source_train_indices=list(range(8)),source_validation_indices=list(range(8,12)),
                                    dataset_cache_sha256=sha256(cache),domain_index_sha256=sha256(idx/'idx_fixture_0.npy'))))
    source=json.loads((ROOT/'configs/source_development.json').read_text())
    source.update(epochs=2,hidden_dim=4,num_layers=1,batch_size=4,use_node_attributes=True,scale_pairs=3)
    source_file=inputs/'source.json';source_file.write_text(json.dumps(source))
    config=json.loads((ROOT/'configs/adaptation_development.json').read_text())
    config.update(scale_pairs=3,query_count=4,target_count=3,references_per_class=1,trust_pairs=dict(ss=2,tt=2,st=2))
    config_file=inputs/'adaptation.json';config_file.write_text(json.dumps(config))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    common=['--data-path',str(cache),'--idx-dir',str(idx),'--dataset','fixture','--source-split',str(split),
            '--source-config',str(source_file),'--adapt-config',str(config_file),'--device','cpu']
    def run(stage,extra,name):
        with (output/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),stage,*common,*extra],cwd=ROOT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    run('all',['--target-policy','transductive-unlabeled','--output',str(output/'fresh')],'fresh')
    run('adapt',['--target-policy','transductive-unlabeled','--source-checkpoint',str(output/'fresh/source/best.pt'),
                 '--resume-adapt',str(output/'fresh/adaptation/round_1.pt'),'--output',str(output/'resumed')],'resume_adapt')
    first=torch.load(output/'fresh/adaptation/last.pt',map_location='cpu',weights_only=False)
    second=torch.load(output/'resumed/adaptation/last.pt',map_location='cpu',weights_only=False)
    assert first['round']==second['round']==2
    for name,value in first['model'].items():assert torch.equal(value,second['model'][name]),name
    assert torch.equal(first['pi_state']['result'].pi,second['pi_state']['result'].pi)
    assert first['portable_selection']['round']==second['portable_selection']['round']
    source['epochs']=1;source_file.write_text(json.dumps(source))
    run('source',['--output',str(output/'source_first')],'source_first')
    source['epochs']=2;source_file.write_text(json.dumps(source))
    run('source',['--resume-source',str(output/'source_first/source/last.pt'),'--output',str(output/'source_resumed')],'resume_source')
    a=torch.load(output/'fresh/source/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(output/'source_resumed/source/last.pt',map_location='cpu',weights_only=False)
    for name,value in a['model'].items():assert torch.equal(value,b['model'][name]),name
    proof=json.loads((output/'fresh/adaptation/initialization.json').read_text())
    assert proof['equivalence']['prediction_mismatches']==0
    result=dict(status='PASSED',from_zero_source_and_adaptation=True,adaptation_rounds=2,
                exact_source_resume=True,exact_adaptation_resume=True,target_labels_poisoned_and_unused=True,
                package_root=str(ROOT),outputs=str(output))
    (output/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
