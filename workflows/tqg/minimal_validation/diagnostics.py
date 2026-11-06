"""Stopped, paired controls and functional one-step audits; no label reader."""
import copy
from dataclasses import replace
import hashlib
import time
from unittest.mock import patch

import torch
from torch.nn import functional as F

from tqg_gda.p0_diagnostics import model_hash, tensor_hash, gradient_of_ce, cosine
from tqg_gda.v1.query import build_feedback, feedback_loss
from tqg_gda.v2.query import (ParameterSnapshot, build_gradient_feedback,
                             parameter_feedback_loss, target_space)


def derived_seed(seed, domain, block, role, rep=0):
    text = f'tqg-sega-minimal-20261106|seed={seed}|domain={domain}|block={block}|role={role}|rep={rep}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') % (2**63-1)


def generator(seed):
    return torch.Generator().manual_seed(seed)


def permuted_labels(query_ids, labels, seed, rep):
    """Global graph-ID mapping, independent of current batch ordering."""
    order = sorted(range(len(query_ids)), key=lambda i: query_ids[i])
    sorted_labels = labels[order]
    index = torch.randperm(len(labels), generator=generator(derived_seed(seed, 0, 0, 'perm', rep)))
    mapping = {int(query_ids[j]): int(sorted_labels[index[i]]) for i, j in enumerate(order)}
    return torch.tensor([mapping[int(i)] for i in query_ids], dtype=torch.long), mapping


def flat_gradient(loss, model, retain_graph=False):
    parameters = tuple(model.parameters())
    parts = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, parts)]).detach()


def norm_match(vector, norm):
    length = float(vector.double().norm())
    return vector * (norm/length) if length > 1e-12 and norm > 1e-12 else torch.zeros_like(vector)


def relative_error(actual, reference):
    difference = float((actual.double()-reference.double()).norm())
    return dict(absolute=difference, relative=difference/max(float(reference.double().norm()), 1e-12))


def construct(model, query, labels, target, config, seed, domain, block, replay=False):
    """All label controls reuse the exact TRUE geometry; never reads target y."""
    if query.y is not None or target.y is not None:
        raise PermissionError('Diagnostics require stripped input labels')
    model.eval()
    original_hash = model_hash(model)
    snapshot = ParameterSnapshot(model)
    query_ids = tuple(query.graph_ids)  # plain metadata installed by the runner
    started = time.perf_counter()
    if config.version == 'v1':
        probe_seed = derived_seed(seed, domain, block, 'v1/probe')
        try:
            feedback = build_feedback(model, query, labels, target, config.v1, generator(probe_seed))
        except (FloatingPointError, torch.linalg.LinAlgError) as error:
            if model_hash(model)!=original_hash:
                raise AssertionError('Failed query modified snapshot') from error
            names=['TRUE','MEAN-Y']+[f'{a}-{i}' for a in ('PERM','RAND') for i in range(1,4)]
            gradients={name:torch.zeros_like(snapshot.flat) for name in names}
            return dict(gradients=gradients,space=snapshot.flat.new_zeros((len(snapshot.flat),0)),
                feedback=None,permutations={str(i):permuted_labels(query_ids,labels,seed,i)[1] for i in (1,2,3)},
                health=dict(query=dict(status='numerical_fallback_source_only',error=str(error)),
                    query_and_execution_seconds=time.perf_counter()-started,
                    gradient_identity=dict(absolute=0.,relative=0.),space_dim=0,
                    deterministic_replay=False,model_hash=original_hash,state_unchanged=True,
                    gradient_hashes={k:tensor_hash(v) for k,v in gradients.items()}))
        build_seconds=time.perf_counter()-started
        solve_seconds=None
        true = flat_gradient(feedback_loss(model, target, feedback), model)
        reference = flat_gradient(-(model(target)*feedback.correction).sum()/target.num_graphs, model)
        # C maps response coefficients to executable all-parameter gradients.
        z = model(target)
        columns = [flat_gradient((z*feedback.target_features[..., j]).sum()/target.num_graphs,
                                 model, retain_graph=True)
                   for j in range(feedback.target_features.shape[-1])]
        c = torch.stack(columns, 1) if columns else snapshot.flat.new_zeros((len(snapshot.flat), 0))
        space = target_space(c.T)[0] if columns else c
        features = feedback.source_features
        gram = torch.einsum('bcm,bcn->mn', features, features)/len(labels)
        gram = gram + config.v1.ridge*torch.eye(gram.shape[0], dtype=gram.dtype)
        target_base = snapshot.logits(target).detach()

        def read(residual):
            corr = torch.einsum('bcm,bc->m', features, residual)/len(labels)
            a = torch.linalg.solve(gram, corr) if len(corr) else corr
            correction = feedback.target_features@a
            stopped = replace(feedback, coefficients=a.detach(), correction=correction.detach(),
                              reference=(target_base+correction).detach())
            return flat_gradient(feedback_loss(model, target, stopped), model)
        if replay:
            repeated = build_feedback(model, query, labels, target, config.v1, generator(probe_seed))
            if not torch.equal(repeated.correction, feedback.correction):
                raise AssertionError('v1 query replay changed')
    else:
        from tqg_gda.v2 import query as query_module
        solver=query_module.solve_calibration
        timings=[]
        def timed_solver(*args,**kwargs):
            begin=time.perf_counter()
            try:
                return solver(*args,**kwargs)
            finally:
                timings.append(time.perf_counter()-begin)
        with patch.object(query_module,'solve_calibration',side_effect=timed_solver):
            feedback = build_gradient_feedback(model, query, labels, target, config.v2)
        build_seconds=time.perf_counter()-started
        solve_seconds=sum(timings)
        true = flat_gradient(parameter_feedback_loss(model, feedback), model)
        reference = feedback.gradient
        space = feedback.basis

        def read(residual):
            if feedback.statistics['status'] != 'gradient_feedback':
                return torch.zeros_like(snapshot.flat)
            r = (residual@feedback.contrasts).double().reshape(-1)
            return -(feedback.basis.double()@(feedback.readout@r)).to(snapshot.flat.dtype)
        if replay:
            repeated = build_gradient_feedback(model, query, labels, target, config.v2)
            if not torch.equal(repeated.gradient, feedback.gradient):
                raise AssertionError('v2 query replay changed')
    query_seconds = time.perf_counter()-started
    identity = relative_error(true, reference)
    if identity['relative'] > 1e-4:
        raise AssertionError(f'{config.version} gradient identity: {identity}')
    probabilities = snapshot.logits(query).detach().softmax(-1)
    one_hot = F.one_hot(labels, probabilities.shape[1]).to(probabilities)
    gradients = {'TRUE': true, 'MEAN-Y': read(one_hot.mean(0).expand_as(probabilities)-probabilities)}
    maps = {}
    for rep in range(1, 4):
        changed, mapping = permuted_labels(query_ids, labels, seed, rep)
        maps[str(rep)] = mapping
        gradients[f'PERM-{rep}'] = read(F.one_hot(changed, probabilities.shape[1]).to(probabilities)-probabilities)
        noise = torch.randn(space.shape[1], generator=generator(
            derived_seed(seed, domain, block, config.version+'/rand', rep)), dtype=space.dtype)
        gradients[f'RAND-{rep}'] = norm_match(space@noise, float(true.double().norm()))
    if model_hash(model) != original_hash:
        raise AssertionError('Query/control changed model parameters or buffers')
    if any(not torch.isfinite(g).all() for g in gradients.values()):
        raise FloatingPointError('Nonfinite control gradient')
    health = dict(query=feedback.statistics, query_and_execution_seconds=query_seconds,
                  primary_query_seconds=build_seconds,calibration_seconds=solve_seconds,
                  gradient_identity=identity, space_dim=space.shape[1], deterministic_replay=replay,
                  model_hash=original_hash, state_unchanged=True,
                  gradient_hashes={k:tensor_hash(v) for k,v in gradients.items()})
    if config.version == 'v1':
        health['source_column_rms'] = feedback.source_features.square().sum(1).mean(0).sqrt().tolist()
        health['target_column_rms'] = feedback.target_features.square().sum(1).mean(0).sqrt().tolist()
    return dict(gradients=gradients, space=space.detach(), feedback=feedback,
                health=health, permutations=maps)


def input_batches(task, ids, size=64, dtype=torch.float32):
    return [task.batch(ids[i:i+size], 'target', dtype=dtype) for i in range(0, len(ids), size)]


@torch.no_grad()
def logits_at(snapshot, flat, batches):
    return torch.cat([snapshot.logits(batch, flat) for batch in batches])


def gradient_rows(gradients, ga, gb, gf, space, beta):
    projected = space.double()@(space.double().T@gb.double())
    bnorm = float(gb.double().norm())
    result = []
    for name, g in dict(gradients, **{'CENTERED-Y':gradients['TRUE']-gradients['MEAN-Y']}).items():
        gd = g.double()
        result.append(dict(arm=name, norm=float(gd.norm()), cos_A=cosine(gd, ga.double(), 1e-12),
            cos_B=cosine(gd, gb.double(), 1e-12), relative_error_A=relative_error(g, ga)['relative'],
            beta_norm_over_fit=float(beta*gd.norm()/gf.double().norm()) if float(gf.norm())>1e-12 else None,
            B_space_energy=float(projected.square().sum()/bnorm**2) if bnorm>1e-12 else None))
    return result


def step_audit(model, task, a_ids, b_ids, ya, yb, ga, gb, gf, gq, gradients, config, common=None):
    """Evaluate stored FP32 steps and FP64 rounding audits without re-querying."""
    initial_hash = model_hash(model)
    snapshot = ParameterSnapshot(model)
    double_model = copy.deepcopy(model).double().eval()
    double_snapshot = ParameterSnapshot(double_model)
    a_batches, b_batches = input_batches(task, a_ids), input_batches(task, b_ids)
    base_a = logits_at(snapshot, snapshot.flat, a_batches)
    base_b = logits_at(snapshot, snapshot.flat, b_batches)
    base_b64 = logits_at(double_snapshot, snapshot.flat.double(), b_batches)
    ce0 = float(F.cross_entropy(base_b, yb))
    ce064 = float(F.cross_entropy(base_b64, yb))
    beta, lr = config.adapt.feedback_weight, config.adapt.lr
    oracle = norm_match(ga, float(gradients['TRUE'].double().norm()))
    directions = {'FROZEN':torch.zeros_like(gf), 'FIT-CE':gf, 'FIT+QRY-CE':gf+beta*gq,
                  'FIT+ORACLE':gf+beta*oracle, 'TRUE-only':beta*gradients['TRUE'],
                  'ORACLE-only':beta*oracle}
    directions.update({'FIT+'+k:gf+beta*v for k,v in gradients.items()})
    rows, changes = [], {}
    common = {} if common is None else common
    for arm, d in directions.items():
        if arm in common:
            rows.append(dict(common[arm], reused_shared_step=True))
            continue
        # Same fused add_ as momentum-free SGD, preserving FP32 writeback.
        actual = snapshot.flat.clone().add_(d, alpha=-lr)
        ideal = snapshot.flat.double()-lr*d.double()
        error = float((actual.double()-ideal.to(actual.dtype).double()).abs().max())
        scale = max(1., float(snapshot.flat.abs().max()))
        if error > 2*torch.finfo(actual.dtype).eps*scale:
            raise AssertionError('SGD writeback differs from rounded prescribed step')
        za, zb = logits_at(snapshot, actual, a_batches), logits_at(snapshot, actual, b_batches)
        zb64 = logits_at(double_snapshot, actual.double(), b_batches)
        ideal_b = logits_at(double_snapshot, ideal, b_batches)
        ce, ce64 = float(F.cross_entropy(zb,yb)), float(F.cross_entropy(zb64,yb))
        change, change64 = ce-ce0, ce64-ce064
        inc, ideal_inc = actual.double()-snapshot.flat.double(), -lr*d.double()
        ideal_norm = float(ideal_inc.norm())
        row = dict(arm=arm, B_ce=ce, B_ce_change=change, B_ce64_actual=ce64,
            B_ce64_change=change64, B_ce64_ideal=float(F.cross_entropy(ideal_b,yb)),
            A_ce=float(F.cross_entropy(za,ya)), A_ce_change=float(F.cross_entropy(za,ya)-F.cross_entropy(base_a,ya)),
            predicted_B_ce_change=float(-lr*torch.dot(gb.double(),d.double())),
            numeric_floor=max(10*torch.finfo(torch.float32).eps*max(1.,ce0),abs(change-change64)),
            parameter_change_norm=float(inc.norm()), ideal_parameter_change_norm=ideal_norm,
            actual_over_ideal_norm=float(inc.norm())/ideal_norm if ideal_norm>1e-30 else None,
            rounded_sgd_max_error=error, B_logit_change_norm=float((zb-base_b).double().norm()),
            actual_parameters_hash=tensor_hash(actual), direction_hash=tensor_hash(d),
            reused_shared_step=False)
        rows.append(row)
        changes[arm] = d.detach()
        if arm in ('FROZEN','FIT-CE','FIT+QRY-CE'):
            common[arm] = row
    if model_hash(model)!=initial_hash:
        raise AssertionError('Temporary updates changed the source snapshot')
    return rows, changes, common
