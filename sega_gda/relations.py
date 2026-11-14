"""§3.6: fixed graph identities; current encodings and the shared S2 distance."""
from dataclasses import dataclass, asdict
import math
import torch
from torch_geometric.data import Batch, Data


@dataclass(frozen=True)
class RelationRules:
    # Explicit development settings are passed by config; no formal defaults.
    source_positive_neighbors: int
    target_near_neighbors: int
    target_far_rank_low: int  # 1-based, excluding anchor
    target_far_rank_high: int
    source_per_anchor: int
    target_per_anchor: int
    zeta: float
    epsilon: float

    def validate(self):
        counts = [self.source_positive_neighbors, self.target_near_neighbors,
                  self.target_far_rank_low, self.target_far_rank_high,
                  self.source_per_anchor, self.target_per_anchor]
        if any(not isinstance(n, int) or n < 1 for n in counts):
            raise ValueError('Positive integer relation budgets/ranks required')
        if self.target_far_rank_low <= self.target_near_neighbors or self.target_far_rank_high < self.target_far_rank_low:
            raise ValueError('Target near and far rank pools must be disjoint')
        if any(not math.isfinite(x) or x < 0 for x in (self.zeta, self.epsilon)):
            raise ValueError('Relation separation parameters must be finite and nonnegative')


@dataclass(frozen=True)
class RelationMemory:
    domain: str
    triplets: tuple
    metadata: dict

    def __post_init__(self):
        if self.domain not in ('source', 'target'):
            raise ValueError('Unknown relation domain')
        if any(len(t) != 3 or len(set(t)) != 3 for t in self.triplets):
            raise ValueError('A relation must have three distinct graph IDs')

    def to_dict(self):
        return dict(domain=self.domain, triplets=[list(t) for t in self.triplets], metadata=self.metadata)

    @classmethod
    def from_dict(cls, value):
        return cls(value['domain'], tuple(tuple(t) for t in value['triplets']), value['metadata'])


def _reservoir(candidates, budget, generator):
    """Uniform combination sampling without storing all possible triplets."""
    selected, count = [], 0
    for candidate in candidates:
        count += 1
        if count <= budget:
            selected.append(candidate)
        else:
            index = int(torch.randint(count, (1,), generator=generator))
            if index < budget:
                selected[index] = candidate
    return selected, count


@torch.no_grad()
def _build(domain, graph_ids, initial_coordinates, distance, rules, seed, provenance, source_labels=None):
    rules.validate()
    ids = tuple(graph_ids)
    if (not ids or any(not isinstance(g, str) for g in ids) or len(set(ids)) != len(ids)
            or len(ids) != len(initial_coordinates) or not torch.isfinite(initial_coordinates).all()):
        raise ValueError('Need unique namespaced string graph IDs and aligned initial coordinates')
    if bool((distance.solver.metric.B.detach() != 0).any()):
        raise ValueError('Relation memory must be built with initial identity geometry')
    if not provenance or not provenance.get('initial_encoder_sha256'):
        raise ValueError('Record the initial encoder identity in relation provenance')
    if domain == 'source' and (source_labels is None or set(source_labels) != set(ids)):
        raise ValueError('Only source train ID -> real label map may define source relations')
    generator = torch.Generator().manual_seed(seed)
    # Sort anchors by ID as well: tie-breaking and RNG consumption ignore caller order.
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    triplets, skipped, candidates_per_anchor = [], [], {}
    for i in order:
        # One row only; solver handles pair chunks. No full dataset N x N matrix.
        row = distance(initial_coordinates[i], initial_coordinates).detach().cpu().tolist()
        ranked = sorted((j for j in range(len(ids)) if j != i), key=lambda j: (row[j], ids[j]))
        if domain == 'source':
            near = [j for j in ranked if source_labels[ids[j]] == source_labels[ids[i]]][:rules.source_positive_neighbors]
            far = [k for k in ranked if source_labels[ids[k]] != source_labels[ids[i]]]
            # Source negatives: full other-class pool; uniform COMBINATIONS, no hard mining.
            candidates = ((ids[i], ids[j], ids[k]) for j in near for k in far)
            budget = rules.source_per_anchor
        else:
            near = ranked[:rules.target_near_neighbors]
            far = ranked[rules.target_far_rank_low-1:rules.target_far_rank_high]
            candidates = ((ids[i], ids[j], ids[k]) for j in near for k in far
                          if row[k] >= (1+rules.zeta)*row[j]+rules.epsilon)
            budget = rules.target_per_anchor
        selected, count = _reservoir(candidates, budget, generator)
        triplets.extend(selected)
        candidates_per_anchor[ids[i]] = count
        if not selected:
            skipped.append(ids[i])
    metadata = dict(rules=asdict(rules), seed=seed, provenance=dict(provenance),
                    graph_ids=sorted(ids), skipped_anchors=skipped, candidate_counts=candidates_per_anchor,
                    sorting='(D0, graph_id); anchors by graph_id', sampling='uniform combinations without replacement',
                    source_negative_pool='all other classes' if domain == 'source' else None,
                    ranks='1-based excluding anchor; truncate to available pool', distance='S2 D at B=0',
                    training_reduction='full fixed memory mean per domain')
    return RelationMemory(domain, tuple(triplets), metadata)


def build_source_memory(graph_ids, initial_coordinates, source_train_labels, distance, rules, seed, provenance):
    return _build('source', graph_ids, initial_coordinates, distance, rules, seed, provenance, source_train_labels)


def build_target_memory(graph_ids, initial_coordinates, distance, rules, seed, provenance):
    # Deliberately no label, predicted class, confidence, or class-prior argument.
    return _build('target', graph_ids, initial_coordinates, distance, rules, seed, provenance)


def encode_graph_ids(graph_ids, graph_lookup, encoder):
    """Load original graphs each call; only graph inputs are placed into PyG Batch.

    graph_lookup must enforce allowed source-train / target-unlabeled IDs. This
    helper does not read y, even if an external Data happens to carry that field.
    """
    if not graph_ids:
        raise ValueError('Cannot encode an empty ID list')
    parameter = next(encoder.parameters())
    graphs = []
    for graph_id in graph_ids:
        graph = graph_lookup(graph_id)
        inputs = Data(x=graph.x.to(parameter), edge_index=graph.edge_index.to(parameter.device))
        if graph.edge_attr is not None:
            inputs.edge_attr = graph.edge_attr.to(parameter)
        graphs.append(inputs)
    return encoder(Batch.from_data_list(graphs))


def relation_loss(source_memory, target_memory, graph_lookup, encoder, distance, margin):
    """Exactly .5 * (source mean + target mean), including an empty side as zero."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Finite nonnegative relation margin required')
    if source_memory.domain != 'source' or target_memory.domain != 'target':
        raise ValueError('Source/target memories are reversed')
    # A connected zero makes the both-empty case usable in a combined objective.
    zero = next(encoder.parameters()).sum()*0 + distance.solver.metric.B.sum()*0
    losses = []
    for memory in (source_memory, target_memory):
        if not memory.triplets:
            losses.append(zero)
            continue
        ids = sorted({g for t in memory.triplets for g in t})
        coordinates = encode_graph_ids(ids, graph_lookup, encoder)
        losses.append(_relation_mean(memory, dict(zip(ids,coordinates)), distance, margin))
    return dict(total=(losses[0]+losses[1])/2, source=losses[0], target=losses[1],
                source_count=len(source_memory.triplets), target_count=len(target_memory.triplets))


def _relation_mean(memory, coordinates, distance, margin):
    anchor,positive,negative = [torch.stack([coordinates[t[column]] for t in memory.triplets]) for column in range(3)]
    # One current differentiable scale for both terms of every triplet.
    pairs = distance(torch.cat((anchor,anchor)),torch.cat((positive,negative)))
    near,far = pairs.chunk(2)
    return torch.relu(margin+near-far).mean()


def relation_loss_from_coordinates(source_memory, target_memory, coordinates, distance, margin):
    """Same S3 reduction on current coordinates; spatial callers pass detached probes.

    This function never creates or persists an embedding memory. Content callers
    must supply freshly encoded original graphs, spatial callers frozen feedback.
    """
    if source_memory.domain != 'source' or target_memory.domain != 'target':
        raise ValueError('Source/target memories reversed')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Invalid relation margin')
    zero = distance.solver.metric.B.sum()*0
    losses = [_relation_mean(memory,coordinates,distance,margin) if memory.triplets else zero
              for memory in (source_memory,target_memory)]
    return dict(total=(losses[0]+losses[1])/2,source=losses[0],target=losses[1],
                source_count=len(source_memory.triplets),target_count=len(target_memory.triplets))
