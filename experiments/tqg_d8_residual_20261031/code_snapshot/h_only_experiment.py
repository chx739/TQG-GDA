"""H-only candidate: frozen D1/D2 null controls, P0 geometry, D4 H readout."""
import argparse
import ast
import json
from pathlib import Path
import torch

from p0_experiment import ExperimentConfig, ResearchExperiment as P0, read, write, run_d0
from d34_experiment import ResearchExperiment as D34
from tqg_gda.d34_diagnostics import kernel_family, solve_observed
from tqg_gda.p0_diagnostics import tensor_hash
from tqg_gda.utils import sha256


class ResearchExperiment(P0):
    def __init__(self, config):
        super().__init__(config)
        self.p0=Path(self.settings['p0_stage'])
        self.d34=Path(self.settings['d34_stage'])
        self.cache_index={r['path']:r for r in read(self.p0/'local_geometry_manifest.json')['files']}
        self.expected={}
        for seed in self.plan['seeds']:
            for line in (self.d34/'logs'/str(seed)/'signature_target_blocks.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['arm']=='H':
                    self.expected[(int(row['task'][-1]),seed,row['snapshot_step'],row['block_id'])]=row
        if len(self.expected)!=360: raise ValueError('Incomplete D4 reference')

    def base(self,*args):
        row=super().base(*args)
        row.update(candidate_id='V2_H_ONLY',signature='H',candidate_protocol_sha256=sha256(self.stage.parent/'candidate_protocol.md'))
        return row

    def ast_screen(self):
        super().ast_screen()
        findings=[]
        for name in ['h_only_experiment.py','h_only_report.py']:
            for node in ast.walk(ast.parse((self.code/name).read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('eval','exec','compile'):
                    findings.append({'file':name,'line':node.lineno})
        write(self.stage/'candidate_ast_screen.json',dict(passed=not findings,findings=findings))
        if findings: raise ValueError('Candidate AST screen failed')

    def prepare(self):
        super().prepare()
        manifest=read(self.stage/'experiment_manifest.json')
        manifest.update(candidate_id='V2_H_ONLY',candidate_protocol_sha256=sha256(self.stage.parent/'candidate_protocol.md'),
            reference_runs=dict(P0=str(self.p0),D34=str(self.d34)),
            scope='H-only rerun of original D1/D2 criteria, exploratory development evidence')
        write(self.stage/'experiment_manifest.json',manifest)
        files=[]
        for rel,row in self.cache_index.items():
            if sha256(self.p0/rel)!=row['sha256']: raise ValueError('P0 geometry changed')
            files.append(row)
        write(self.stage/'input_cache_manifest.json',dict(source=str(self.p0),files=files))
        for seed in self.plan['seeds']:
            for domain in (1,2,3):
                for step in self.plan['snapshots']['times']:
                    for block in range(8):
                        from tqg_gda.p0_diagnostics import sample_ids
                        ids=read(self.p0/'logs'/str(seed)/f'{domain}_{seed}_{step}_{block}_indices.json')
                        bseed=61000000+domain*100000+(seed-3405)*10000+step*10+block
                        a,b,_,_=self.pools(domain);task=self.task(domain)
                        assert ids==dict(A=list(sample_ids(a,16,bseed)),B=list(b),fit=list(sample_ids(task.fit_ids,16,bseed+1000)),query=sorted(task.query_ids),block_seed=bseed)
        print('Verified all 360 input blocks and frozen P0 caches',flush=True)

    def geometry(self,model,source_inputs,target_inputs,domain,seed,step,block):
        geo,rel=D34.cached_geometry(self,model,domain,seed,step,block)
        permutation=torch.arange(len(geo.source_probabilities))
        kernels,_=kernel_family(geo.source_representation,geo.target_representation,
            geo.source_response,geo.target_response,permutation,scale_epsilon=self.v2.scale_epsilon)
        geo.readout,info=solve_observed(kernels['H'],geo.target_response,'JOINT',.01,self.v2.calibration)
        expected=self.expected[(domain,seed,step,block)]
        if tensor_hash(geo.readout)!=expected['W_hash']: raise ValueError('H readout differs from D4 frozen arm')
        original=read(self.p0/'logs'/str(seed)/f'{domain}_{seed}_{step}_{block}_indices.json')
        # Reuse the exact input-only geometry, changing only kernel/readout.
        geo.statistics=dict(calibration=info,P0_cache=rel,P0_cache_sha256=self.cache_index[rel]['sha256'],
            H_readout_matches_D4=True,rank=geo.basis.shape[1],block_seed=original['block_seed'])
        return geo

    def worker_entrypoint(self):
        return run_worker


def run_worker(config,seed):
    return ResearchExperiment(config).worker(seed)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','d0','pilot','run'])
    parser.add_argument('--repo',required=True);parser.add_argument('--stage',required=True)
    parser.add_argument('--config',default='configs/h_only_diagnostics.json')
    args=parser.parse_args()
    e=ResearchExperiment(ExperimentConfig(args.repo,args.stage,str(Path(args.config).resolve())))
    torch.set_num_threads(e.settings['threads_per_worker'])
    if args.action=='prepare':e.prepare()
    elif args.action=='d0':
        run_d0(e)
        checks=[]
        for domain in (1,2,3):
            model,_=e.load(domain,3405,0)
            geo=e.geometry(model,None,None,domain,3405,0,0)
            checks.append(dict(domain=domain,H_readout_matches_D4=True,converged=geo.statistics['calibration']['converged']))
        write(e.stage/'D0/H_replay_checks.json',dict(passed=all(c['converged'] for c in checks),checks=checks))
    elif args.action=='pilot':
        assert read(e.stage/'D0/H_replay_checks.json')['passed']
        e.worker(3405,pilot=True)
    else:e.run()


if __name__=='__main__':main()
