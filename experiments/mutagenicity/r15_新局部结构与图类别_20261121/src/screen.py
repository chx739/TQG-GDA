"""Write and authenticate both dictionaries and E actions before requesting any E labels."""
import io
import json

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r9_目标近邻与源标签传递_20261121.src import inputs

from . import scoring, source_inputs, structure


def seal(output, name, stage, members, **references):
    write_json(output / name, dict(stage=stage, **references,
                                  files={path: digest(output / path) for path in sorted(members)}))
    return digest(output / name)


def write_arrays(output, filename, arrays):
    with (output / filename).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def freeze(output, plan, original):
    records, rows = original.records()
    supports, lineage = {}, {}
    members = ['executed-config.json', 'source-roles.json', 'unlabeled-summary.json']
    for direction in original.roles['directions']:
        name = source_inputs.direction_name(direction)
        role = source_inputs.direction_roles(direction)
        margin, coefficients, parity = original.model(name, direction, rows)
        dictionary = structure.fit_dictionary({i: records[i] for i in role['U_representatives']}, coefficients, plan)
        e_ids = np.array(role['E'], dtype=np.int64)
        delta, swapped = np.asarray(structure.predict_delta([rows[i] for i in e_ids], dictionary)).T
        prediction = scoring.predictions(e_ids, margin, delta, swapped)
        supports[name] = scoring.support(role, dictionary, prediction, parity)
        lineage[name] = role
        dict_path, pred_path = f'{name}-dictionary.json', f'{name}-predictions.npz'
        write_json(output / dict_path, dictionary)
        write_arrays(output, pred_path, prediction)
        members.extend((dict_path, pred_path))

    write_json(output / 'source-roles.json', dict(r9_source_provenance=original.provenance, directions=lineage))
    write_json(output / 'unlabeled-summary.json', dict(directions=supports, **scoring.identifiability(supports),
                                                      label_values_loaded=False))
    return seal(output, 'OBS.json', 'R15_OBS', members,
                r9_observation_sha256=plan['r9_observation_sha256'])


def verify_frozen(output, obs_sha, original):
    raw = (output / 'OBS.json').read_bytes()
    if inputs.sha(raw) != obs_sha:
        raise PermissionError('R15 OBS seal identity differs')
    obs = json.loads(raw)
    if obs['stage'] != 'R15_OBS' or obs['r9_observation_sha256'] != original.obs_sha256:
        raise ValueError('R15 OBS stage or parent differs')
    directions = original.roles['directions']
    names = [source_inputs.direction_name(d) for d in directions]
    expected = {'executed-config.json', 'source-roles.json', 'unlabeled-summary.json'}
    for name in names:
        expected.update((f'{name}-dictionary.json', f'{name}-predictions.npz'))
    if set(obs['files']) != expected:
        raise PermissionError('R15 OBS membership differs')
    verified = {}
    for path, sha in obs['files'].items():
        payload = (output / path).read_bytes()
        if inputs.sha(payload) != sha:
            raise PermissionError(f'R15 frozen member differs: {path}')
        verified[path] = payload
    lineage = json.loads(verified['source-roles.json'])
    if lineage['r9_source_provenance'] != original.provenance:
        raise PermissionError('R15 frozen source provenance differs')
    summary = json.loads(verified['unlabeled-summary.json'])
    rebuilt = {}
    predictions = {}
    for name, direction in zip(names, directions):
        role = source_inputs.direction_roles(direction)
        if lineage['directions'][name] != role:
            raise PermissionError(f'R15 frozen role differs for {name}')
        dictionary = json.loads(verified[f'{name}-dictionary.json'])
        with np.load(io.BytesIO(verified[f'{name}-predictions.npz']), allow_pickle=False) as archive:
            p = dict(archive)
        np.testing.assert_array_equal(p['graph_ids'], role['E'])
        replay = scoring.predictions(p['graph_ids'], p['B'], p['delta'], p['swapped_delta'])
        for key, values in replay.items():
            np.testing.assert_array_equal(values, p[key])
        parity = summary['directions'][name]['r9_B_prediction_parity']
        rebuilt[name] = scoring.support(role, dictionary, p, parity)
        predictions[name] = p
    if summary != dict(directions=rebuilt, **scoring.identifiability(rebuilt), label_values_loaded=False):
        raise ValueError('R15 unlabeled summary differs from frozen evidence')
    return summary, predictions


def finish(output, obs_sha, plan, original):
    summary, predictions = verify_frozen(output, obs_sha, original)
    members = ['OBS.json', 'verdict.json']
    receipt = dict(fit_label_ids_parsed=[], verified_obs_sha256=obs_sha,
                   score_labels_granted=summary['scoring_permitted'], score_label_ids_parsed={})
    if not summary['scoring_permitted']:
        verdict = dict(passed=False, status='stopped_without_labels',
                       reasons=summary['unlabeled_stop_reasons'])
    else:
        reader = inputs.RoleLabels(original.evidence, original.roles, phase='score')
        rng = np.random.default_rng(plan['bootstrap_seed'])
        results = {}
        for direction in original.roles['directions']:
            name = source_inputs.direction_name(direction)
            ids = predictions[name]['graph_ids'].tolist()
            labels = reader.load(direction['source'], direction['target'], ids)
            receipt['score_label_ids_parsed'][name] = ids
            label_path = f'{name}-scored-labels.npz'
            write_arrays(output, label_path, dict(graph_ids=ids, labels=labels))
            members.append(label_path)
            results[name] = scoring.score_direction(predictions[name], labels, plan['bootstrap_replicates'], rng)
        write_json(output / 'scores.json', dict(
            directions=results, descriptive_interval=dict(strata=['direction', 'class'], replicates=plan['bootstrap_replicates'],
                                                          seed=plan['bootstrap_seed'], percentiles=[2.5, 97.5],
                                                          same_graph_draws_across_arms=True)))
        members.append('scores.json')
        verdict = dict(status='scored', **scoring.promotion(results, plan))

    write_json(output / 'label-access.json', receipt)
    write_json(output / 'verdict.json', verdict)
    final_sha = seal(output, 'FINAL.json', 'R15_FINAL', [*members, 'label-access.json'],
                     observation_sha256=obs_sha)
    return final_sha, verdict
