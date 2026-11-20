"""Authenticate C's frozen graph roles and B0 margins without fitting."""
import io
import hashlib
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.inputs import npz_member, sha


def registered(plan):
    for setting in plan['settings']:
        for source in plan['domains']:
            for seed in plan['seeds']:
                yield setting,source,seed,[t for t in plan['domains'] if t!=source]


def key(setting,source,target,seed):
    return f'{setting}-M{source}_to_M{target}-seed-{seed}'


def margin(logits):
    if logits.dtype!=np.float64 or logits.ndim!=2 or logits.shape[1]!=2 or not np.isfinite(logits).all():
        raise ValueError('B0 logits must be finite float64 binary arrays')
    return logits[:,1]-logits[:,0]


def prepare_group(scope, repo, plan, setting, source, seed, targets):
    group, validated = {}, None
    for target in targets:
        block = key(setting,source,target,seed)
        prefix = f'blocks/{block}'
        manifest_blob = scope.read_bytes(f'{prefix}/manifest.json','R7_INPUT')
        manifest = json.loads(manifest_blob)
        task = dict(setting=setting,source=source,target=target,seed=seed)
        if any(manifest[k]!=v for k,v in task.items()):
            raise ValueError('Historical block identity differs')
        context = r5_inputs._context(manifest['r7_plan'],**task,repository=repo)
        if context['ids']!=manifest['ids']:
            raise ValueError('Archived fit, validation, A or H IDs changed')
        ids = manifest['ids']
        old_files = {name:checksum for name,checksum in manifest['provenance']['r5_files'].items()
                     if name.endswith('/source.pt')}
        if len(old_files)!=1:
            raise ValueError('Historical manifest lacks an unambiguous source checkpoint')
        checkpoint, checkpoint_sha = next(iter(old_files.items()))
        val = npz_member(scope.read_bytes(f'{prefix}/validation_input.npz','R7_INPUT'))
        np.testing.assert_array_equal(val['validation_ids'],ids['validation'])
        v_margin = margin(val['validation_z0'])
        lineage = dict(path=checkpoint,sha256=checkpoint_sha, selected_epoch=manifest['provenance']['selected_source_epoch'])
        if validated is None:
            blob = context['read'](checkpoint)
            if sha(blob)!=checkpoint_sha:
                raise PermissionError('Referenced R5 source checkpoint differs from R7 identity')
            state = torch.load(io.BytesIO(blob),weights_only=True,map_location='cpu')
            features = manifest['r7_plan']['features']
            head = state[features['source_model_state_key']]
            w,b = (head[name].double().numpy() for name in features['source_head_state_keys'])
            if state['selected_epoch']!=lineage['selected_epoch']:
                raise ValueError('R5 selected source epoch does not match the manifest')
            selection = min(state['history'],key=lambda row:row['source_validation']['ce'])['epoch']
            if selection != state['selected_epoch']:
                raise ValueError('Archive selected epoch does not match source-validation CE')
            np.testing.assert_allclose(val['validation_h']@w.T+b,val['validation_z0'],atol=plan['replay_atol'],rtol=0)
            label_file = f'source_labels/M{source}/seed{seed}_validation.pt'
            v_port = ValidationPort(context['read'](label_file,'INPUT'),len(ids['validation']),
                                    tuple(context['roles']['class_counts']['validation']))
            validated = dict(ids=ids['validation'],margin=v_margin,lineage=lineage,v_port=v_port)
        else:
            if validated['lineage']!=lineage or validated['ids']!=ids['validation']:
                raise ValueError('Same-source target blocks do not share checkpoint and validation IDs')
            np.testing.assert_array_equal(validated['margin'],v_margin)
        width = f"tau-{plan['tau']:.3f}"
        obs = npz_member(scope.read_bytes(f'{prefix}/{width}-observed.npz','R7_OBS'),
                         ['adapt_logits_B0','adapt_probability_B0','w_B0'])
        held = npz_member(scope.read_bytes(f'{prefix}/{width}-heldout.npz','R7_OBS'),
                          ['holdout_logits_B0','holdout_probability_B0'])
        if np.any(obs['w_B0']!=0):
            raise ValueError('Archived B0 has a nonzero update')
        margins = {}
        for name, source_array,file in (('adapt',obs,'fit_input.npz'),('holdout',held,'holdout_input.npz')):
            prefix_keys = [f'{name}_ids',f'{name}_z0']
            original = npz_member(scope.read_bytes(f'{prefix}/{file}','R7_INPUT'),prefix_keys)
            np.testing.assert_array_equal(original[prefix_keys[0]],ids[name])
            logits = source_array[f'{name}_logits_B0']
            reported = source_array[f'{name}_probability_B0']
            x = margin(logits)
            np.testing.assert_allclose(x,margin(original[prefix_keys[1]]),atol=plan['replay_atol'],rtol=0)
            np.testing.assert_array_equal(logits.argmax(1),decisions_from_margin(x))
            np.testing.assert_allclose(softmax(logits,axis=1),reported,atol=plan['replay_atol'],rtol=0)
            margins[name]=x
        group[target]=dict(task=task,key=block,ids=ids,manifest_sha256=sha(manifest_blob),
                           margins=margins,dataset=context['spec']['tu_dataset'],
                           adapt_port=raw_grant(context,manifest,'adapt'))
    return validated,group


def decisions_from_margin(x):
    return (x>0).astype(np.int64)


@dataclass(frozen=True)
class ValidationPort:
    payload: bytes
    n: int
    class_counts: tuple

    def load(self):
        y = torch.load(io.BytesIO(self.payload),weights_only=True,map_location='cpu').numpy()
        if y.shape!=(self.n,) or y.dtype!=np.int64 or not np.isin(y,[0,1]).all():
            raise ValueError('Historical source validation label identity differs')
        if tuple(np.bincount(y,minlength=2))!=self.class_counts:
            raise ValueError('Source validation class counts differ from sealed roles')
        return y


@dataclass(frozen=True)
class RawLabelPort:
    role: str
    path: Path
    sha256: str
    ids: tuple
    offsets: dict
    mapping: dict


def raw_grant(context,manifest,role):
    if role not in ('adapt','holdout') or manifest['ids'][role]!=context['ids'][role]:
        raise PermissionError('Raw label grant role differs from the authenticated R5 cohort')
    relative=context['label_relative']
    meta=context['label_meta']
    expected=dict(relative_path=relative,sha256=context['provenance']['r5_files'][relative])
    if manifest['label_records']!=expected:
        raise PermissionError('The R7 label manifest does not bind the authentic R5 file')
    ids=tuple(context['ids'][role])
    return RawLabelPort(role,Path(meta['path']),meta['sha256'],ids,
                        {i:tuple(meta['offsets'][i]) for i in ids},dict(meta['mapping']))


class GraphLabelSession:
    """Authenticate file bytes for integrity, then parse only granted offsets."""

    def __init__(self, role):
        if role not in ('adapt','holdout'):
            raise ValueError('Unknown target label session role')
        self.role=role
        self.streams={}

    def __enter__(self):
        return self

    @staticmethod
    def _hash(stream):
        stream.seek(0)
        return hashlib.file_digest(stream,'sha256').hexdigest()

    def load(self,port,selected):
        if port.role!=self.role or not set(selected)<=set(port.ids) or len(set(selected))!=len(selected):
            raise PermissionError('Label request is outside the submitted read grant')
        if self.role=='holdout' and tuple(selected)!=port.ids:
            raise PermissionError('Holdout scoring requires the exact frozen H order')
        key=port.path,port.sha256
        if key not in self.streams:
            stream=port.path.open('rb',buffering=0)
            if self._hash(stream)!=port.sha256:
                stream.close()
                raise PermissionError('Raw labels differ from the sealed R5 record')
            self.streams[key]=stream
        stream=self.streams[key]
        y=[]
        for i in selected:
            offset,length=port.offsets[i]
            stream.seek(offset)
            raw=stream.read(length)
            if len(raw)!=length:
                raise ValueError('Label offset is outside the authenticated file')
            y.append(port.mapping[str(int(raw))])
        return np.asarray(y,dtype=np.int64)

    def __exit__(self,*exc):
        changed=False
        for (_,checksum),stream in self.streams.items():
            try:
                changed |= self._hash(stream)!=checksum
            finally:
                stream.close()
        self.streams.clear()
        if changed:
            raise PermissionError('Raw label bytes changed during R11 evaluation')
