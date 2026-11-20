"""Run the registered eight-source screen and preserve each stratum and counterfactual."""
import json

import numpy as np

from experiments.common.io import digest, write_json
from experiments.mutagenicity.r8_图结构观察量与类别约束_20261121.src.pilot import write_table
from . import identification, readouts, source_inputs, worlds


def specifications(plan):
    specs = plan['worlds']
    if (len(specs) != 2 or len({row['name'] for row in specs}) != 2
            or any(len(row['positive_percent']) != 4 or sum(row['positive_percent']) != 200
                   or any(type(q) is not int or not 0 < q < 100 for q in row['positive_percent']) for row in specs)):
        raise ValueError('Invalid R12 world registration')
    true = [worlds.true_accuracy(row['positive_percent']) for row in specs]
    best = [int(np.argmax(row)) for row in true]
    if best != [row['optimal_j'] for row in specs]:
        raise ValueError('Registered oracles do not match exact enumerated thresholds')
    return list(zip(specs, true, best))


def analyze(counts, plan):
    n = counts.sum(axis=(2, 3))
    status = 'supported' if np.all(n > 0) else 'insufficient_support'
    specs = specifications(plan)
    diagnostics, strata, outcomes, estimates = [], [], [], []
    fits = {}
    for (spec, true, best), (_, _, opposite_best) in (specs, specs[::-1]):
        name = spec['name']
        concentrations = spec['positive_percent']
        if status == 'supported':
            real, product = worlds.tables(counts, concentrations)
            audit, oracle_j = worlds.stratum_oracle(counts, real.sum(axis=0), concentrations,
                                                    plan['estimator']['feasibility_tolerance'])
            if oracle_j is not None and oracle_j != best:
                raise AssertionError('Within-stratum oracle chose an unexpected threshold')
            strata.extend(dict(world=name, **row) for row in audit)
            for arm, class_joint in (('paired', real), ('product', product)):
                observed = class_joint.sum(axis=0)
                estimate = identification.recover(observed, **plan['estimator'])
                associations = worlds.conditional_stats(class_joint)
                for row in associations['class_rows']:
                    diagnostics.append(dict(world=name, arm=arm, **row))
                j = estimate['j']
                fits[name, arm] = j
                outcomes.append(dict(world=name, arm=arm, support=status, selected_j=j,
                    optimal_j=best, status=estimate['status'], true_accuracy_fraction=float(true[j]),
                    accuracy_gain_vs_middle_pp=float(100 * (true[j] - true[2])),
                    gain_vs_exchanged_oracle_pp=float(100 * (true[j] - true[opposite_best])),
                    exact_optimum=bool(j == best),
                    **{k: estimate[k] for k in ('prior', 'reasons', 'unfolding_rank', 'reconstructed_table_max_error')},
                    actual_L_order=associations['L_order'], actual_L_contrast=associations['L_contrast'],
                    actual_T_contrast=associations['T_contrast'], actual_D_contrast=associations['D_contrast'],
                    actual_view_class_ranks=associations['views_rank'],
                    stratum_oracle_j=oracle_j if arm == 'paired' else None))
                estimate.update(associations=associations, observed=observed.tolist(), class_joint=class_joint.tolist(),
                                true_accuracy_exact=list(map(str, true)))
                estimates.append(dict(world=name, arm=arm, estimate=estimate))
        else:
            strata.extend(dict(world=name, bin=k, identifiable=None, reason=status,
                               recovered_positive_rate=None, assigned_positive_rate=concentrations[k-1]/100)
                          for k in range(1, 5))
            for arm in ('paired', 'product'):
                outcomes.append(dict(world=name, arm=arm, support=status, selected_j=None, optimal_j=best,
                                     status=status, exact_optimum=False, reasons=['empty_U_Y_stratum']))
    outcomes_by_world = {spec['name']: true for spec, true, _ in specs}
    exchanged = {specs[0][0]['name']: specs[1][0]['name'], specs[1][0]['name']: specs[0][0]['name']}
    if status == 'supported':
        for row in outcomes:
            name, arm, j = (row[k] for k in ('world', 'arm', 'selected_j'))
            opposite_j = fits[exchanged[name], arm]
            true = outcomes_by_world[name]
            row.update(exchanged_selected_j=opposite_j,
                       gain_vs_exchanged_fit_pp=float(100 * (true[j] - true[opposite_j])))
    return status, outcomes, estimates, diagnostics, strata


def run(output, repo, plan, index, actual, options, *, authenticate_only=False):
    r7 = source_inputs.ValidationArchive(plan)
    specs = {spec['id']: spec for spec in r7.plan['real_settings']}
    seed = plan['seed']
    if len(index) != len({x['setting'] for x in index}):
        raise ValueError('R5 registry has duplicate settings')
    archives = {setting: source_inputs.SourceArchive(
        repo, next(x for x in index if x['setting'] == setting), actual[setting]['config'],
        specs[setting], plan['domains'], seed) for setting in plan['settings']}
    sources, support, outcomes, estimates, dependencies, oracles = [], [], [], [], [], []
    if not authenticate_only:
        (output / 'views').mkdir()
        (output / 'probes').mkdir()
    for setting, archive in archives.items():
        for source in plan['domains']:
            task = dict(setting=setting, source=source, seed=seed)
            key = f'{setting}-M{source}-seed-{seed}'
            roles = archive.role(source)
            v = r7.validation(setting, source, seed)
            margins, checkpoint = archive.b0(source, roles, v, r7.plan['features'])
            s_ids, v_ids = roles['fit'], roles['validation']
            graphs = archive.source_graphs(source)
            if authenticate_only:
                print(f'{key}: authenticated {len(s_ids)} S and {len(v_ids)} V graphs and B0; no labels decoded', flush=True)
                continue
            y_s = archive.labels(source, 'fit', s_ids)
            cuts, u = readouts.quantile_bins(margins)
            probes, frozen = readouts.fit(graphs, s_ids, v_ids, y_s, archive.plan['input']['node_feature_dim'], options)
            write_json(output / f'probes/{key}.json', probes)
            with (output / f'views/{key}-frozen.npz').open('xb') as stream:
                np.savez_compressed(stream, validation_ids=np.array(v_ids), margin=margins, cuts=cuts, u=u, **frozen)
            # V labels are only needed after freezing the readouts and cutpoints.
            y_v = archive.labels(source, 'validation', v_ids)
            counts = worlds.cell_counts(y_v, u, frozen['l'], frozen['t'])
            with (output / f'views/{key}-known-answers.npz').open('xb') as stream:
                np.savez_compressed(stream, validation_ids=np.array(v_ids), y=y_v, cell_counts=counts)
            state, rows, observations, diagnostic, strata = analyze(counts, plan)
            sources.append(dict(**task, support=state, fit_ids=s_ids, validation_ids=v_ids, b0_checkpoint=checkpoint,
                                u_cutpoints=cuts.tolist(), observed_bins=sorted(set(u.tolist()))))
            support.extend(dict(**task, y=y, u=k+1, n=int(counts[y, k].sum()))
                           for y in (0, 1) for k in range(4))
            outcomes.extend(dict(**task, **row) for row in rows)
            estimates.extend(dict(**task, **row) for row in observations)
            dependencies.extend(dict(**task, **row) for row in diagnostic)
            oracles.extend(dict(**task, **row) for row in strata)
            print(f'{key}: {state}, paired selections ' +
                  str([row['selected_j'] for row in rows if row['arm'] == 'paired']), flush=True)
    if authenticate_only:
        return None
    write_json(output / 'source-roles.json', sources)
    write_json(output / 'observations-and-estimates.json', estimates)
    write_table(output / 'support.csv', support)
    write_table(output / 'outcomes.csv', outcomes)
    write_table(output / 'within-class.csv', dependencies)
    write_table(output / 'stratum-oracle.csv', oracles)
    receipts = {setting: dict(archive=str(ar.root), members=ar.receipt,
                            source_label_reads={f'M{src}-{role}': ids for (src, role), ids in ar.label_reads.items()})
                for setting, ar in archives.items()}
    write_json(output / 'input-access.json', dict(r7_source_reads=r7.source_reads, r5_sources=receipts))
    gate = verdict(plan, outcomes)
    write_json(output / 'gate.json', gate)
    files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'FINAL.json', dict(stage='R12_SOURCE_SCREEN_FINAL', files=files))
    return gate


def verdict(plan, outcomes):
    by_setting = {}
    for setting in plan['settings']:
        by_source = {}
        for source in plan['domains']:
            rows = [row for row in outcomes if row['setting'] == setting and row['source'] == source]
            if len(rows) != 4:
                raise ValueError('Expected both worlds and both mechanisms for all eight registered sources')
            by_source[str(source)] = dict(supported=rows[0]['support'] == 'supported',
                paired_both=all(row['exact_optimum'] for row in rows if row['arm'] == 'paired'),
                product_both=all(row['exact_optimum'] for row in rows if row['arm'] == 'product'))
        supported = sum(row['supported'] for row in by_source.values())
        paired = sum(row['paired_both'] for row in by_source.values())
        product = sum(row['product_both'] for row in by_source.values())
        by_setting[setting] = dict(supported=supported, paired_both=paired, product_both=product,
                                   denominator=len(plan['domains']), domains=by_source,
                                   passed=paired >= plan['minimum_correct_domains'])
    return dict(passed=all(row['passed'] for row in by_setting.values()), settings=by_setting)
