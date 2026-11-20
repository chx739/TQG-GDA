from pathlib import Path
import json, hashlib, pickle, subprocess, sys
import numpy as np
import torch
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import residual_inputs, wl
ROOT = Path('/home/hxchen/research/GDA/TSBA/repository')
DIR = Path(__file__).parent
ROUND = Path(residual_inputs.__file__).parents[1]
OLD = Path('/home/hxchen/research/GDA/TSBA/experiments/r9/attempt-002-weighted-residuals')
CFGFILE = ROUND/'configs/weighted-source-residuals.json'
SHA = lambda b:hashlib.sha256(b).hexdigest()
COMMIT = '03dd388b55c92eb949919314570da6485230e8db'
def frozen(file):
    target = str(file.relative_to(ROOT))
    expected = subprocess.check_output(['git','show',f'{COMMIT}:{target}'],cwd=ROOT)
    if file.read_bytes()!=expected: raise RuntimeError(f'Changed: {target}')
    return SHA(expected)
def inputs():
    old_obs=(OLD/'OBS.json').read_bytes()
    assert SHA(old_obs)=='6298811e840521a07d4134131e02130240649e4ef6fdf71d5b204034ca039617'
    old_roles=(OLD/'roles.json').read_bytes()
    assert SHA(old_roles)==json.loads(old_obs)['files']['roles.json']
    return json.loads(CFGFILE.read_bytes()),json.loads(old_roles)
if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg,expected=inputs()
    files=[CFGFILE,ROUND/'docs/reference-geometry.md',Path(wl.__file__),Path(residual_inputs.__file__)]
    archive = residual_inputs.Archive(ROOT, cfg)
    print('Authenticated D/C source and unlabeled structure containers',flush=True)
    roles,features,top=archive.prepare()
    assert roles == expected
    for (setting,source), group in features.items():
        with (DIR/f'{setting}-M{source}.pkl').open('wb') as f: pickle.dump(group,f)
    prov={str(f.relative_to(ROOT)):frozen(f) for f in files}
    (DIR/'inputs.json').write_text(json.dumps(dict(commit=COMMIT,registered_obs=SHA((OLD/'OBS.json').read_bytes()),roles=SHA((OLD/'roles.json').read_bytes()),sources=prov),indent=2)+'\n')
    print('Verified all 24 historical role orders, generated features',flush=True)
