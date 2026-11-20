"""Authenticate R10's source roles and grant exact per-direction R5 label offsets."""
from contextlib import ExitStack, contextmanager
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from torch_geometric.data import Data

from experiments.mutagenicity.r7_共同解释与共享更新验证_20261119.src import r5_inputs
from experiments.mutagenicity.r10_原生图适配基线与增量_20261121.src import structural, structural_observation
from experiments.mutagenicity.r11_分类阈值的跨域检验_20261121.src import prior_run, roc_transfer
from experiments.mutagenicity.r12_三种图读数与阈值识别_20261121.src import source_inputs


DIRECTIONS = ('small_to_large','large_to_small')
ROLES = ('D','C','U','E')
PHASE_ROLES = {'fit':('D','C'),'score':('U','E'),'replay_E':('E',)}
IDENTITY = ('setting','source','direction','seed')


def key(setting, source, direction, seed):
    return f'{setting}-M{source}-{direction}-seed-{seed}'


def grid(cfg):
    return {key(setting,source,direction,cfg['seed']):dict(
        setting=setting,source=source,direction=direction,seed=cfg['seed'])
        for setting in cfg['settings'] for source in cfg['domains'] for direction in DIRECTIONS}


def split(s, v, sizes):
    ids = [*s,*v]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(sizes):
        raise ValueError('Source S/V and size inventory must agree and be disjoint')
    distinct, counts = np.unique([sizes[i] for i in ids],return_counts=True)
    cut = None
    if len(distinct)>1:
        balance = np.abs(2*np.cumsum(counts)[:-1]-len(ids))
        cut = int(distinct[np.argmin(balance)])
    lower = {i for i in ids if cut is not None and sizes[i]<=cut}
    s_low,s_high = [i for i in s if i in lower],[i for i in s if i not in lower]
    v_low,v_high = [i for i in v if i in lower],[i for i in v if i not in lower]
    first = dict(D=s_low,C=v_low,U=s_high,E=v_high)
    second = dict(D=s_high,C=v_high,U=s_low,E=v_low)
    return cut, dict(zip(DIRECTIONS,(first,second),strict=True))


def source_graphs(repo, archive, source, s, v):
    """Verify S/V by their identity and topology, without opening S/V class counts."""
    ids = [*s,*v]
    if any(type(i)!=int or not 0<=i<archive.count for i in ids) or len(ids)!=len(set(ids)):
        raise PermissionError('Invalid or overlapping source fit and validation IDs')
    entry = archive.plan['setting']['domain_indices'][source]
    path = archive.spec['domain_index_pattern'].format(domain=source)
    payload = r5_inputs._inside(repo,path).read_bytes()
    authorized = np.load(io.BytesIO(payload),allow_pickle=False)
    if (entry['domain']!=source or entry['path']!=path or hashlib.sha256(payload).hexdigest()!=entry['sha256']
            or authorized.ndim!=1 or not np.issubdtype(authorized.dtype,np.integer)
            or len(ids)!=entry['count'] or len(ids)!=archive.spec['domain_counts'][source]
            or sorted(authorized.tolist())!=sorted(ids)):
        raise PermissionError('The R10 source S/V roles do not partition the authenticated domain')
    archive.receipt[f'repository:{path}'] = entry['sha256']
    result = {}
    for i in ids:
        graph,record = archive.graphs[i],archive.graph_hashes[i]
        if type(graph)!=dict or set(graph)!={'x','edge_index','edge_attr'}:
            raise PermissionError('Only label-free R5 graph records are allowed')
        hashes = {k:None if v is None else hashlib.sha256(v.numpy().tobytes()).hexdigest()
                  for k,v in graph.items()}
        if record['id']!=i or record['input_sha256']!=hashes:
            raise PermissionError(f'R5 graph bytes differ: {i}')
        result[i] = graph
    return result


class Archive:
    def __init__(self, repo, cfg, *, load_graphs=True):
        self.repo = repo
        root = Path(cfg['r10'])
        record = roc_transfer.stage(root,'FIT',cfg['r10_fit_sha256'])
        files = record['files']
        self.plan = json.loads(prior_run.read_member(root,files,'executed-config.json'))['plan']
        structural.validate_plan(self.plan)
        if ((cfg['settings'],cfg['domains'],cfg['seed']) !=
                (self.plan['settings'],self.plan['domains'],self.plan['seeds'][0])
                or record['predecessors'] != {'r7_final_digest':self.plan['r7_final_digest']}):
            raise PermissionError('Source-size screen scope differs from R10 FIT')
        auth = source_inputs.ValidationArchive(dict(r7_archive=self.plan['r7_archive'],
            r7_final_digest=self.plan['r7_final_digest'],settings=cfg['settings'],domains=cfg['domains'],seed=cfg['seed']))
        if auth.plan != json.loads(prior_run.read_member(root,files,'historical-plan.json')):
            raise PermissionError('R10 FIT belongs to a different R7 C input')
        provenance = auth.plan['provenance']
        index = json.loads((repo/provenance['r5_archive_index']).read_bytes())
        actual = json.loads((repo/provenance['r5_actual_config']).read_bytes())
        specs = {entry['id']:entry for entry in auth.plan['real_settings']}
        self.fit_access = json.loads(prior_run.read_member(root,files,'fit-access.json'))
        if len(index) != len({x['setting'] for x in index}):
            raise PermissionError('Ambiguous R5 archive registry')
        by_setting = {}
        for setting in cfg['settings']:
            registration = next(x for x in index if x['setting']==setting)
            by_setting[setting] = source_inputs.SourceArchive(
                repo,registration,actual[setting]['config'],specs[setting],cfg['domains'],cfg['seed'],
                load_graphs=load_graphs)
        self.by_setting = by_setting
        self.scope = cfg

    def prepare(self):
        ordered = {}
        features, sizes_by_source = {},{}
        for setting, archive in self.by_setting.items():
            for source in self.scope['domains']:
                group = structural_observation.group_key(setting,source,self.scope['seed'])
                old = self.fit_access[group]
                s_ids,v_ids = old['source_ids'],old['validation_ids']
                if old['graph_container_sha256']!=archive.receipt['inputs/graphs.pt']:
                    raise PermissionError('Historical R5 graph container differs from R10 FIT')
                graphs = source_graphs(self.repo,archive,source,s_ids,v_ids)
                feat = {}
                sizes = {}
                for gid, record in graphs.items():
                    g = Data(**record)
                    feat[gid] = structural.features(g,setting)
                    if setting=='mutagenicity_node':
                        sizes[gid] = len(record['x'])
                    else:
                        sizes[gid] = len({(min(u,v),max(u,v)) for u,v in record['edge_index'].T.tolist() if u!=v})
                cut, directions = split(s_ids,v_ids,sizes)
                features[setting,source] = feat
                sizes_by_source[group] = dict(source_ids=s_ids,validation_ids=v_ids,
                                               sizes=[sizes[i] for i in s_ids+v_ids],cut=cut)
                for direction, roles in directions.items():
                    key_ = key(setting,source,direction,self.scope['seed'])
                    absent = [r for r in ROLES if not roles[r]]
                    status = 'ok' if cut is not None and not absent else 'unavailable'
                    ordered[key_] = dict(setting=setting,source=source,direction=direction,seed=self.scope['seed'],
                                         cut=cut,roles=roles,status=status,
                                         reason='no_size_cut' if cut is None else
                                                ('empty_'+'_'.join(absent) if absent else None))
        if set(ordered)!=set(grid(self.scope)):
            raise PermissionError('Missing R9 direction roles')
        return ordered, features, sizes_by_source


class ScopedLabels:
    """Parse only the active direction's authorized role, never all S/V labels."""
    def __init__(self, by_setting, active_roles, phase):
        self.archives = by_setting
        self.grants = {key_:(row['setting'],{r:tuple(row['roles'][r]) for r in PHASE_ROLES[phase]})
                       for key_,row in active_roles.items() if row['status']=='ok'}
        self.actual = {}
        self.active = None
        self.phase = phase
        self.streams = {}
        self.stack = ExitStack()

    def __enter__(self):
        self.stack.__enter__()
        return self

    def label_file(self, setting):
        archive = self.archives[setting]
        meta = json.loads(archive._read('inputs/label_records.json'))
        expected = Path(archive.plan['input']['raw_directory'])/f"{archive.plan['dataset']}_graph_labels.txt"
        file_spec = archive.plan['input']['files'][expected.name]
        registered = file_spec['sha256'] if isinstance(file_spec,dict) else file_spec
        if (Path(meta['path']).resolve()!=expected.resolve() or registered!=meta['sha256']
                or len(meta['offsets'])!=archive.count or sorted(meta['mapping'].values())!=[0,1]):
            raise PermissionError('Raw labels differ from R5 identity or binary map')
        stream = self.stack.enter_context(expected.open('rb',buffering=0))
        if hashlib.file_digest(stream,'sha256').hexdigest()!=registered:
            raise PermissionError('Raw labels differ from their authenticated bytes')
        return stream,meta

    @contextmanager
    def direction(self, task):
        if self.active is not None or task not in self.grants:
            raise PermissionError('Only one registered direction may request labels at a time')
        self.active = task
        try:
            yield
        finally:
            self.active = None

    def load(self, task, role, ids):
        if task!=self.active:
            raise PermissionError('Label request is outside the active direction')
        setting, approved = self.grants[task]
        authorized = approved.get(role)
        if not authorized or tuple(ids)!=authorized or role in self.actual.get(task,{}):
            raise PermissionError('Label request differs from the exact phase, direction or role grant')
        if setting not in self.streams:
            self.streams[setting] = self.label_file(setting)
        stream,meta = self.streams[setting]
        labels = []
        for gid in ids:
            start,length = meta['offsets'][gid]
            if any(type(n)!=int for n in (start,length)) or start<0 or length<1:
                raise PermissionError('Unusable registered label offset')
            stream.seek(start)
            record = stream.read(length)
            if len(record)!=length:
                raise PermissionError('Incomplete registered label record')
            labels.append(meta['mapping'][str(int(record))])
        self.actual.setdefault(task,{})[role] = list(ids)
        return np.asarray(labels,dtype=np.int64)

    def __exit__(self, *exc):
        with self.stack:
            for stream,meta in self.streams.values():
                stream.seek(0)
                if hashlib.file_digest(stream,'sha256').hexdigest()!=meta['sha256']:
                    raise PermissionError('Registered label bytes changed during the phase')
        return False


def validate_receipt(rows, actual, phase):
    expected = {k:{r:row['roles'][r] for r in PHASE_ROLES[phase]}
                for k,row in rows.items() if row['status']=='ok'}
    if actual != expected:
        raise PermissionError(f'Actual {phase} label reads differ from exact directional grants')
