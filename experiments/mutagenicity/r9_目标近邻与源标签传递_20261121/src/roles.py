"""Label-free splits on edge count and complete WL fingerprint groups."""
from fractions import Fraction
import hashlib
import struct

from .wl import by_fingerprint


def order_key(namespace, key, seed=None):
    prefix = namespace.encode('ascii') + b'\0'
    if seed is not None:
        prefix += struct.pack('>I', seed)
    return hashlib.sha256(prefix+key).digest()


def edge_split(edges):
    ordered = sorted(set(edges.values()))
    if len(ordered) < 2:
        raise ValueError('Need at least two distinct edge sizes for the registered split')
    from collections import Counter
    counts = Counter(edges.values())
    lower, candidates = 0, []
    for cut in ordered[:-1]:
        lower += counts[cut]
        candidates.append((abs(2*lower-len(edges)), cut))
    _, cut = min(candidates)
    return cut, {0: sorted(i for i, n in edges.items() if n <= cut),
                 1: sorted(i for i, n in edges.items() if n > cut)}


def nearest_prefix(groups, fraction, *, reserve=False):
    stop = len(groups) - int(reserve)
    if stop < 1:
        raise ValueError('Need a nonempty prefix and the registered remainder')
    wanted = Fraction(*fraction)*sum(map(len, groups))
    cumulative, choices = 0, []
    for n, group in enumerate(groups[:stop], start=1):
        cumulative += len(group)
        choices.append((abs(cumulative-wanted), n))
    return min(choices)[1]


def role_direction(source, target, all_groups, counts, plan):
    source_ids, target_ids = all_groups[source], all_groups[target]
    groups = by_fingerprint(counts, target_ids)
    if len(groups) < 2:
        return dict(source=source, target=target, source_ids=source_ids, holdout_ids=[],
                    unavailable_reason='target_has_fewer_than_two_normalized_fingerprint_groups',
                    groups=[dict(group_sha256=hashlib.sha256(key).hexdigest(), graph_ids=ids, pool='unassigned')
                            for key, ids in sorted(groups.items())],
                    perturbations={str(seed): dict(adapt=[], source_middle=[]) for seed in plan['adaptation_seeds']})
    ns = plan['hash_namespaces']
    ordered = sorted(groups, key=lambda key: (order_key(ns['holdout'], key), key))
    end = nearest_prefix([groups[key] for key in ordered], plan['h_fraction'], reserve=True)
    fixed_h, pool = ordered[:end], ordered[end:]
    holdout = sorted(i for key in fixed_h for i in groups[key])
    perturbations = {}
    for seed in plan['adaptation_seeds']:
        order = sorted(pool, key=lambda key: (order_key(ns['adaptation'], key, seed), key))
        end = nearest_prefix([groups[key] for key in order], plan['a_fraction'])
        chosen = order[:end]
        original_a = [i for key in chosen for i in groups[key]]
        n = min(len(original_a), len(source_ids))
        adapt = original_a[:n]
        mids = sorted(source_ids,
                      key=lambda i: (order_key(ns['source_bridge'], struct.pack('>I', i), seed), i))[:n]
        perturbations[str(seed)] = dict(adapt=adapt, source_middle=mids, full_a=original_a,
            a_group_order=[order_key(ns['adaptation'], key, seed).hex() for key in order],
            chosen_groups=len(chosen), a_pool_size=sum(len(groups[key]) for key in pool))
    return dict(source=source, target=target, source_ids=source_ids, holdout_ids=holdout,
                groups=[dict(group_sha256=hashlib.sha256(key).hexdigest(),
                             holdout_order_hash=order_key(ns['holdout'], key).hex(),
                             graph_ids=groups[key], pool='H' if key in fixed_h else 'U')
                        for key in ordered], perturbations=perturbations)


def build(edges, counts, plan):
    cut, all_groups = edge_split(edges)
    specs = [role_direction(source, target, all_groups, counts, plan) for source, target in plan['directions']]
    return dict(max_small_edges=cut, pseudo_domains=all_groups, directions=specs)
