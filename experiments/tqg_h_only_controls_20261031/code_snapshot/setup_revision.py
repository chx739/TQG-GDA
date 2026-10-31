"""Lock the explicitly requested revision execution without altering old plans."""
import ast
from pathlib import Path
import time

import yaml

from revision_common import CODE, STAGE, RUN, DESIGN, OLD, read, atomic_json, git_identity, hardware
from tqg_gda.utils import sha256


def main():
    commit=git_identity()
    if (STAGE/'execution_lock.json').exists():
        raise FileExistsError('Execution is already locked; use resume, not overwrite')
    plan=yaml.safe_load((DESIGN/'experiment_plan.yaml').read_text())
    inputs={}
    def add(key,path):
        path=Path(path)
        inputs[key]=dict(path=str(path),sha256=sha256(path))
    for name in ('method.md','experiment_plan.yaml','resolved_arms_preview.json','pseudocode.py'):
        add('design_'+name,DESIGN/name)
    add('human_method',plan['human_document'])
    add('data',plan['datasets'][0]['cache'])
    add('source_protocol','/home/hxchen/TQG_data/PROTEINS_official_20261023/source_protocol_official.json')
    add('roles_manifest',plan['datasets'][0]['roles_manifest'])
    add('old_training_manifest',OLD/'training_manifest.json')
    roles=read(plan['datasets'][0]['roles_manifest'])
    for domain in ('2','3'):
        role=roles['domains'][domain]
        add('domain_'+domain,role['index_path'])
        for name in ('development','final'):
            add(f'{domain}_{name}_ids',role[name]['path'])
    for name,row in enumerate(read(OLD/'training_manifest.json')['records']):
        if row['arm'] in ('original_TQG_target_probes','full_source_CE_700plus200'):
            add(f'source_{name}',row['warmup_checkpoint'])
        if row['arm']=='original_TQG_target_probes':
            add(f'historical_final_{name}',row['final_checkpoint'])
    checks=[]
    # This is an inspection screen, not a claim that AST proves safety.
    new_files=['revision_common.py','revision_evaluation.py','revision_execution_diagnostics.py',
               'diagnostics_sample_size.py','diagnostics_local.py','evaluate_sample_size.py',
               'setup_revision.py','training_revision.py','experiment_revision.py',
               'summarize_revision.py','run_d0.py','run_revision_pipeline.py','plot_revision.py',
               'tqg_gda/revision.py','tqg_gda/matching.py','tqg_gda/residual_solver.py']
    new_files += [str(p.relative_to(CODE)) for p in sorted((CODE/'tests').glob('test_revision*.py'))]
    new_files += ['tests/test_d1_design.py']
    for name in new_files:
        path=CODE/name
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('exec','eval'):
                raise ValueError('Dynamic code execution forbidden in '+name)
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
                if any(x.split('.')[0] in ('requests','socket','urllib','paramiko') for x in modules):
                    raise ValueError('Unneeded network import in '+name)
        checks.append(dict(path=str(path),sha256=sha256(path),AST_screen_passed=True))
    lock=dict(revision='v1',commit=commit,created_unix=time.time(),inputs=inputs,
              code_screen=checks,hardware=hardware(),
              user_instruction='按照 /home/hxchen/TQG_GDA/docs/TQG_Improvement_Experiment_Plan_v1.md 开始改进和实验',
              execution_scope=['D0','D1','D2','A','B_only_if_D2_gate_passes'],
              cpu_serial_wall_hour_cap=4.,gpu_hour_cap=0.,final_label_evaluation_allowed=False)
    atomic_json(STAGE/'execution_lock.json',lock)
    atomic_json(STAGE/'compute_ledger.json',dict(cap_seconds=4*3600,consumed_seconds=0.,records=[],gpu_hours=0))
    atomic_json(STAGE/'run_state.json',dict(status='implementation_complete_pending_D0',
                                         phases={},started_unix=time.time(),goal_complete=False))
    print('Revision locked at '+commit)


if __name__=='__main__':
    main()
