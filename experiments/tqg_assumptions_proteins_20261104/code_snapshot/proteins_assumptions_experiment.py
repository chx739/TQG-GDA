"""PROTEINS backend and independent fresh-geometry A0 for assumption probes."""
from pathlib import Path
import hashlib
import numpy as np
import torch

from assumptions_experiment import AssumptionExperiment, read, write, cpu_seconds
from tqg_gda.data import GraphDomainTask
from tqg_gda.training import load_model
from tqg_gda.feedback_v2 import (ParameterSnapshot, V2Config, build_gradient_feedback, parameter_feedback_loss, query_matrices)
from tqg_gda.p0_diagnostics import (unlabeled_geometry, gradient_of_ce, label_free, tensor_hash, model_hash)
from tqg_gda.d34_diagnostics import solve_observed
from tqg_gda.assumption_diagnostics import ordered, stable_seed, ProbeLabelBoundary
from tqg_gda.utils import sha256


class ProteinsBackend:
    def __init__(self,repo,settings,roles_manifest=None):
        self.repo,self.settings=Path(repo),settings;self.tasks={}
        self.hardware='cpu:Intel Xeon;threads=1;FP64'
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                self.hardware='cpu:'+line.split(':',1)[1].strip()+';threads=1;FP64';break
        if roles_manifest is None:
            self.role_base=Path('/')
            self.roles=read(self.repo/settings['legacy_roles_manifest'])['domains']
            self.roles['1']=read(self.repo/settings['domain1_roles'])
        else:
            self.role_base=Path(roles_manifest).parent
            self.roles=read(roles_manifest)['domains']
        for domain in [1,2,3]:
            role=self.roles[str(domain)]
            for key in ['development','final']:
                assert sha256(self.role_base/role[key]['path'])==role[key]['sha256']
            assert sha256(self.repo/f'idx/idx_PROTEINS_{domain}.npy')==role['index_sha256']
    def task(self,domain):
        if domain not in self.tasks:
            task=GraphDomainTask(self.settings['cache'],self.repo/self.settings['source_split'],
                target_index=self.repo/f'idx/idx_PROTEINS_{domain}.npy',target_policy='transductive-unlabeled')
            assert [len(task.fit_ids),len(task.query_ids),len(task.validation_ids)]==[199,51,28]
            assert task.input_dim==3 and task.edge_dim==0
            self.tasks[domain]=task
        return self.tasks[domain]
    def pools(self,domain):
        role=self.roles[str(domain)]
        dev=np.load(self.role_base/role['development']['path']).tolist();final=np.load(self.role_base/role['final']['path']).tolist()
        sequence=ordered(dev,'proteins_assumptions_20261104',domain)
        a,b=sequence[:69],sorted(sequence[69:])
        assert len(dev)==139 and len(a)==69 and len(b)==70 and not set(dev)&set(final)
        assert set(dev)|set(final)==set(self.task(domain).target_ids)
        return a,b,dev,final
    def load(self,domain,seed,step):
        assert step==0
        checkpoint=f"{self.settings['archive']}/checkpoints/source_pretraining/fit/{seed}/best.pt"
        model,state=load_model(self.repo/checkpoint);task=self.task(domain)
        assert state['encoder_spec']==dict(input_dim=3,hidden_dim=32,num_layers=2,edge_dim=0,readout='mean')
        for field,ids in [('source_fit_ids',task.fit_ids),('source_query_ids',task.query_ids),('source_validation_ids',task.validation_ids)]:
            assert state['identity'][field]==list(ids)
        assert state['identity']['cache_sha256']==self.settings['cache_sha256']
        return model.double().eval(),dict(domain=domain,seed=seed,snapshot_step=0,checkpoint=checkpoint,
                 checkpoint_sha256=sha256(self.repo/checkpoint),historical_training='PROTEINS fit-only 700-update budget, source validation CE selected')


class ProteinsAssumptionExperiment(AssumptionExperiment):
    def make_backend(self):return ProteinsBackend(self.repo,self.settings,self.stage.parent/'stage2_method/data_roles/manifest.json')
    def snapshot_references(self):return [self.old.load(d,s,0)[1] for d in self.settings['domains'] for s in self.settings['seeds']]
    def prepare_geometry(self,domain,seed):
        task=self.old.task(domain);model,row=self.old.load(domain,seed,0);before=model_hash(model)
        qs=sorted(task.query_ids);a,b,dev,final=self.old.pools(domain)
        target_ids=ordered(a,'target_geometry',domain,seed)[:16]
        source=label_free(task.batch(qs,'query'));target=task.batch(target_ids,'target')
        geometry=unlabeled_geometry(model,source,target,V2Config())
        snapshot=geometry.snapshot
        boundary=ProbeLabelBoundary(task,a,b,final,self.stage/'D0/label_access.jsonl')
        kernels={};readouts={}
        from tqg_gda.feedback_v2 import signature_kernels
        for name,signature in [('H','representation_only'),('HA','representation_response')]:
            blocks=signature_kernels(geometry.source_representation,geometry.target_representation,
                geometry.source_response,geometry.target_response,V2Config(signature=signature))
            w,info=solve_observed(blocks[:3],geometry.target_response,'JOINT',.01,{})
            kernels[name]=blocks[:3];readouts[name]=(w,info)
        boundary.log(dict(event='readouts_sealed',hashes=[tensor_hash(w) for w,_ in readouts.values()]))
        sy=torch.tensor([task.labels[i] for i in qs]);ay=torch.from_numpy(boundary.read('A_train',target_ids))
        ga=gradient_of_ce(snapshot,[target],ay)
        residual=((torch.nn.functional.one_hot(sy,2).double()-geometry.source_probabilities)@geometry.contrasts).flatten()
        ry=(torch.nn.functional.one_hot(ay,2).double()-snapshot.logits(target).detach().softmax(-1))@geometry.contrasts
        reconstructed=-geometry.basis@torch.einsum('bqr,bq->r',geometry.target_response,ry)/16
        checks={}
        def verify(name,left,right):
            error=float((left-right).abs().max());checks[name]=error
            assert torch.allclose(left,right,atol=1e-8,rtol=1e-8),(name,error)
        verify('CE_gradient_reconstruction',reconstructed,ga)
        tiny=label_free(task.batch(qs[:3],'query'))
        explicit=(snapshot.jacobian(tiny,geometry.contrasts)@geometry.basis).reshape_as(geometry.source_response[:3])
        verify('explicit_source_JVP',explicit,geometry.source_response[:3])
        diagnostics=[]
        cache={}
        for name,signature in [('H','representation_only'),('HA','representation_response')]:
            w,info=readouts[name];gradient=-geometry.basis@(w@residual)
            production=build_gradient_feedback(model,source,sy,target,V2Config(signature=signature))
            verify(name+'_production_W',w,production.readout)
            verify(name+'_production_gradient',gradient,production.gradient)
            changed=build_gradient_feedback(model,source,sy.roll(1),target,V2Config(signature=signature))
            verify(name+'_label_blind_W',w,changed.readout)
            loss=parameter_feedback_loss(model,production)
            actual=torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss,tuple(model.parameters()))])
            verify(name+'_parameter_loss_gradient',gradient,actual)
            diagnostics.append(dict(domain=domain,seed=seed,signature=name,c_A=float(torch.dot(gradient,ga)/(gradient.norm()*ga.norm())),
                e_A=float((gradient-ga).norm()/ga.norm()),weight_norm=float(w.norm()),feedback_norm=float(gradient.norm()),
                mean_error=info['mean_error'],max_error=info['max_error'],H_ST_norm=float(torch.linalg.norm(query_matrices(kernels[name][1],kernels[name][2],geometry.target_response)[0])),
                converged=info['converged'],W_replay_max=0.,old_B_projection=None,checkpoint_sha256=row['checkpoint_sha256']))
            cache[name+'_W']=w.numpy()
        replay=self.stage/f'D0/{seed}_{domain}_fresh_readouts.npz';np.savez_compressed(replay,**cache)
        for name in readouts:verify(name+'_serialized_replay',torch.from_numpy(np.load(replay)[name+'_W']),readouts[name][0])
        th,ta,tp=self.extract(snapshot,task,a+b,geometry.basis,geometry.contrasts)
        dest=self.stage/f'features/{seed}_{domain}.npz'
        np.savez_compressed(dest,ids=np.array(qs+a+b),h=torch.cat((geometry.source_representation,th)).numpy(),
            a=torch.cat((geometry.source_response,ta)).numpy(),p=torch.cat((geometry.source_probabilities,tp)).numpy(),
            q=geometry.contrasts.numpy().flatten(),source_y=sy.numpy(),V=geometry.basis.numpy())
        assert model_hash(model)==before
        write(dest.with_suffix('.json'),dict(query=qs,fit=list(task.fit_ids),validation=list(task.validation_ids),A=a,B=b,final=final,
              target_batch=target_ids,checkpoint=row,feature_sha256=sha256(dest),model_unchanged=True,checks=checks,
              query_diagnostics=diagnostics,historical_B_projection={},A0_type='new geometry, explicit and production cross-checks; no historical v2 cache replay'))
        print(f'A0 PROTEINS seed={seed} domain={domain} rank={geometry.basis.shape[1]} max_error={max(checks.values()):.3e}',flush=True)
    def a0(self):
        start=cpu_seconds()
        for seed in self.settings['seeds']:
            for domain in self.settings['domains']:self.prepare_geometry(domain,seed)
        write(self.stage/'D0/replay.json',dict(passed=True,geometries=15,signatures=30,cpu_seconds=cpu_seconds()-start,
             no_B_labels_read=True,no_final_labels_read=True,snapshot_step=0,block=0,
             type='fresh explicit-Jacobian / CE / production / label-blind / parameter-loss / serialization checks'))
    def after_B_labels(self,domain,seed,data,b_pool,b_y,out):
        task=self.old.task(domain);model,_=self.old.load(domain,seed,0);snapshot=ParameterSnapshot(model)
        chunks=[task.batch(b_pool[i:i+32],'target') for i in range(0,len(b_pool),32)]
        gb=gradient_of_ce(snapshot,chunks,torch.from_numpy(b_y))
        v=torch.from_numpy(data['V']);inside=v@(v.mT@gb)
        fraction=float((gb-inside).square().sum()/gb.square().sum()) if gb.norm()>1e-12 else None
        write(out/f'{domain}_B_projection.json',dict(task=f'0_to_{domain}',seed=seed,B_graphs=len(b_pool),
             B_space_outside_energy_fraction=fraction,scope='B empirical gradient after prediction seal; never used in probe selection'))
