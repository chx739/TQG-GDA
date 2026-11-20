"""Node-aligned R9 WL colors and per-fingerprint votes from disjoint neighborhoods."""
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from math import fsum
from statistics import fmean
import struct

from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.composition import discrete_types, graph_counts
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import wl


@dataclass
class GraphRecord:
    layers: list
    neighbors: list
    edges: int

    @property
    def counts(self):
        return [Counter(layer) for layer in self.layers]


def encode(graph, plan):
    """Retain the center order of the original typed R9 color recursion."""
    nw, ew = plan['node_types'], plan['edge_types']
    _, edges = graph_counts(graph, nw, ew)
    node_types = discrete_types(graph['x'], nw, len(graph['x']), 'node')
    edge_types = discrete_types(graph['edge_attr'], ew, graph['edge_index'].shape[1], 'edge')
    labeled = [[] for _ in node_types]
    for (source, target), label in zip(graph['edge_index'].T.tolist(), edge_types):
        labeled[source].append((label, target))

    colors = [wl.hash_color(b'R9-WL0\0' + struct.pack('>I', label)) for label in node_types]
    layers = [colors]
    for radius in (1, 2):
        updated = []
        for center, adjacent in enumerate(labeled):
            terms = sorted((label, colors[target]) for label, target in adjacent)
            body = b''.join(struct.pack('>I', label) + color for label, color in terms)
            payload = (b'R9-WL\0' + struct.pack('>I', radius) + colors[center]
                       + struct.pack('>I', len(terms)) + body)
            updated.append(wl.hash_color(payload))
        layers.append(updated)
        colors = updated
    return GraphRecord(layers, [[target for _, target in row] for row in labeled], edges.total())


def distances(neighbors, centers):
    """Return the shortest undirected distance to any center, with -1 for unreachable."""
    distance = [-1] * len(neighbors)
    queue = deque(centers)
    for node in centers:
        distance[node] = 0
    while queue:
        node = queue.popleft()
        for target in neighbors[node]:
            if distance[target] == -1:
                distance[target] = distance[node] + 1
                queue.append(target)
    return distance


def context_score(record, centers, coefficients):
    distance = distances(record.neighbors, centers)
    counts = [Counter(color for color, d in zip(layer, distance) if d > 2 + r)
              for r, layer in enumerate(record.layers)]
    # R9 normalization includes all anchors, including colors absent from D.
    features = wl.normalize(counts)
    if not features:
        return None
    return fsum(value * coefficients.get(key, 0.) for key, value in features.items())


def collect_votes(u_records, coefficients):
    """Each record must be the one registered representative of its U fingerprint."""
    parents, votes = {}, defaultdict(list)
    for graph_id, record in sorted(u_records.items()):
        centers = defaultdict(list)
        for index, (parent, word) in enumerate(zip(record.layers[1], record.layers[2])):
            if (2, word) in coefficients or (1, parent) not in coefficients:
                continue
            if parents.setdefault(word, parent) != parent:
                raise ValueError('One WL two-hop color has multiple center parents')
            centers[word].append(index)
        for word, nodes in centers.items():
            score = context_score(record, nodes, coefficients)
            if score is not None:
                votes[word].append(dict(graph_id=graph_id, edges=record.edges, score=score))
    return parents, votes


def fit_dictionary(u_records, coefficients, plan):
    parents, votes = collect_votes(u_records, coefficients)
    candidates, groups = {}, defaultdict(list)
    for word, parent in sorted(parents.items()):
        voting = votes[word]
        n = len(voting)
        edge_sum = sum(vote['edges'] for vote in voting)
        candidate = dict(parent=parent.hex(), n=n, m=(edge_sum / n if n else None),
                         u=(fmean(vote['score'] for vote in voting) if n else None),
                         votes=voting, weight=0., swapped_weight=0.)
        candidates[word] = candidate
        if n >= plan['minimum_votes']:
            # floor(log2(1 + m)) equals floor(log2(floor(1 + m))).
            group = parent, n.bit_length() - 1, ((n + edge_sum) // n).bit_length() - 1
            groups[group].append(word)

    pairs = []
    for group, words in sorted(groups.items()):
        ordered = sorted(words, key=lambda word: (candidates[word]['n'], candidates[word]['m'], word))
        for left, right in zip(ordered[::2], ordered[1::2]):
            pair = len(pairs)
            pairs.append(dict(words=[left.hex(), right.hex()], parent=group[0].hex(),
                              log2_support=group[1], log2_one_plus_mean_edges=group[2]))
            # These are the two residuals around the shared pair mean. Form
            # each as half a difference to preserve the sign reversal exactly.
            for recipient, donor in ((left, right), (right, left)):
                row = candidates[recipient]
                u, other = row['u'], candidates[donor]['u']
                shrink = row['n'] / (row['n'] + plan['shrinkage_pseudocount'])
                row.update(pair=pair, weight=shrink * ((u - other) / 2),
                           swapped_weight=shrink * ((other - u) / 2))
                if row['swapped_weight'] != -row['weight']:
                    raise ValueError('Swapped residual did not preserve recipient shrinkage')
    return dict(words={word.hex(): info for word, info in candidates.items()}, pairs=pairs)


def predict_delta(rows, dictionary):
    """Use the original full-graph R9 radius-2 denominator for every E graph."""
    weights = {bytes.fromhex(word): (info['weight'], info['swapped_weight'])
               for word, info in dictionary['words'].items() if 'pair' in info}
    outputs = []
    for row in rows:
        terms = [(scale * real, scale * swapped)
                 for (depth, color), scale in row.items() if depth == 2 and color in weights
                 for real, swapped in (weights[color],)]
        outputs.append((fsum(t[0] for t in terms), fsum(t[1] for t in terms)))
    return outputs
