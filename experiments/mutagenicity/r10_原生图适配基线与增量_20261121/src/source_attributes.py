"""Authenticated TU node attributes and the registered three-median model."""
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import logit

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import scores
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import geometry_models as geom, wl
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src.inputs import sha
from . import structural

NODE_TYPES = (0, 1, 2)
RAW_SUFFIXES = ('graph_indicator', 'node_labels', 'node_attributes')
METHOD = 'Attribute'


def raw_columns(plan, count):
    """Hash the exact three byte strings before parsing any node records."""
    spec = plan['input']
    if plan['dataset']!='PROTEINS' or spec['node_label_values']!=list(NODE_TYPES):
        raise PermissionError('Unexpected node type coding in the authenticated PROTEINS input')
    columns, hashes = [], {}
    for suffix, parse in zip(RAW_SUFFIXES,(int,int,float),strict=True):
        name = f'PROTEINS_{suffix}.txt'
        payload = (Path(spec['raw_directory'])/name).read_bytes()
        checksum = sha(payload)
        if checksum!=spec['files'][name]['sha256'] or len(payload)!=spec['files'][name]['bytes']:
            raise PermissionError(f'Raw PROTEINS file differs from R5: {name}')
        hashes[name] = checksum
        columns.append(np.array([parse(row) for row in payload.splitlines()]))
    indicators, types, values = columns
    if (indicators.shape!=types.shape or types.shape!=values.shape or
            not np.array_equal(np.unique(indicators), np.arange(1,count+1)) or
            not np.array_equal(np.unique(types), NODE_TYPES) or not np.isfinite(values).all()):
        raise ValueError('Invalid TU indicator, scalar attribute or type inventory')
    return indicators-1, types, values, hashes


def graph_medians(types, values):
    if (types.ndim!=1 or types.shape!=values.shape or not np.isin(types,NODE_TYPES).all()
            or not np.isfinite(values).all()):
        raise ValueError('Expected one finite scalar and one registered type per node')
    result = np.full(len(NODE_TYPES), np.nan)
    for j in NODE_TYPES:
        same = values[types==j]
        if same.size:
            result[j] = np.median(same)
    return result


def graph_attributes(graph_id, graph, owners, types, values):
    mask = owners==graph_id
    ordered_types = types[mask]
    x = graph['x'].numpy()
    if not ordered_types.size or not np.array_equal(x,np.eye(len(NODE_TYPES))[ordered_types]):
        raise PermissionError(f'TU node order or type differs from the sealed R5 graph: {graph_id}')
    selected_values = values[mask]
    median = graph_medians(ordered_types, selected_values)
    # Reverse values within each type while keeping all original node types fixed.
    permuted = selected_values.copy()
    for j in NODE_TYPES:
        slot = ordered_types==j
        permuted[slot] = selected_values[slot][::-1]
    if not np.array_equal(median,graph_medians(ordered_types,permuted),equal_nan=True):
        raise ArithmeticError(f'Within-type permutation changed a median: {graph_id}')
    return median, len(ordered_types)


def scale_attributes(d):
    mean, scale, counts = np.zeros(3), np.zeros(3), np.zeros(3,dtype=int)
    for j in NODE_TYPES:
        known = d[~np.isnan(d[:,j]),j]
        counts[j] = len(known)
        if known.size:
            mean[j] = known.mean()
            # Detect a genuinely constant coordinate before summation rounding.
            if np.ptp(known)>0.:
                scale[j] = known.std()
    return dict(mean=mean.tolist(), scale=scale.tolist(), present=counts.tolist())


def transform_attributes(x, scaler):
    z = np.zeros_like(x,dtype=float)
    sd = np.asarray(scaler['scale'])
    np.divide(x-scaler['mean'],sd,out=z,where=~np.isnan(x)&(sd>0.))
    return z


def source_design(features, roles):
    d = roles['D']
    lexicon = wl.vocabulary({i:features[i][0] for i in d},d)
    size = np.stack([features[i][1] for i in d])
    mean,sd = size.mean(axis=0),size.std(axis=0)
    sd[np.ptp(size,axis=0)==0.] = 1.
    scaler = scale_attributes(np.stack([features[i][2] for i in d]))
    return dict(vocabulary=[[depth,color.hex()] for depth,color in lexicon],
                size_mean=mean.tolist(),size_scale=sd.tolist(),attribute=scaler)


def design_for(features, ids, model):
    lexicon = {(depth,bytes.fromhex(color)):i for i,(depth,color) in enumerate(model['vocabulary'])}
    rows,size,attributes = zip(*(features[i] for i in ids),strict=True)
    f = wl.sparse_rows(rows,range(len(ids)),lexicon)
    z = (np.stack(size)-model['size_mean'])/model['size_scale']
    a = transform_attributes(np.stack(attributes),model['attribute'])
    base = geom.matrix(f,z)
    return dict(WL=f, Size=z, Base=base, Attribute=sparse.hstack((base,a),format='csr'))


def fit_new(design_d, design_c, yd, yc, solver):
    if len(set(yd))==1:
        p = (int(yd.sum())+.5)/(len(yd)+1)
        return dict(status='single_class_D',constant=float(logit(p)),selected_C=None)
    candidates = structural.fit_candidates(design_d[METHOD],design_c[METHOD],yd,yc,
                                           dict(C=structural.STRENGTHS,solver=solver))
    return dict(status='ok',**candidates)


def predict_new(model, matrix):
    if model['status']=='single_class_D':
        return np.full(matrix.shape[0],model['constant'])
    return scores(matrix,structural.select_model(model))
