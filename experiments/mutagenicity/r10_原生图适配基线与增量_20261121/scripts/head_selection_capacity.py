"""Compute retrospective head-selection ceilings from the two published endpoint tables."""

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path
from statistics import mean


R10_FINAL = '174c0b5601291b6dc0e29a7b756b7f6b7131b25b3f171657bc93831bab0b667b'
R11_FINAL = 'c9e2763c05e8029516f1f304dbbdb3c17b4b748b5dfc77b63a0331650042e8b9'
SETTINGS = ('mutagenicity_edge', 'mutagenicity_node', 'proteins_edge')
DOMAINS = range(4)
SEEDS = (3405, 3406, 3407, 3408, 3409)
THREE = ('B0', 'WL_selected', 'Size')
FIVE = (*THREE, 'WL_fixed', 'SourceCal')


def authenticated_rows(directory, final_hash):
    manifest = (directory / 'FINAL.json').read_bytes()
    record = json.loads(manifest)
    if hashlib.sha256(manifest).hexdigest() != final_hash or record['stage'] != 'FINAL':
        raise ValueError(f'Unexpected FINAL in {directory}')
    data = (directory / 'endpoints.csv').read_bytes()
    if hashlib.sha256(data).hexdigest() != record['files']['endpoints.csv']:
        raise ValueError(f'Endpoint digest mismatch in {directory}')
    return csv.DictReader(io.StringIO(data.decode('utf-8')))


def load(directory, final_hash, settings, methods):
    records = {}
    for row in authenticated_rows(directory, final_hash):
        method = row['method']
        if method not in methods:
            continue
        key = (row['setting'], int(row['source']), int(row['target']), int(row['seed']), method)
        if key in records:
            raise ValueError(f'Duplicate endpoint {key}')
        n = int(row['n'])
        published = Fraction(row['accuracy'])
        accuracy = Fraction(round(n * published), n)
        if not 0 <= accuracy <= 1 or abs(accuracy - published) > Fraction(1, 10**12):
            raise ValueError(f'Accuracy disagrees with integer graph counts for {key}')
        if 'accuracy_exact' in row:
            positives = int(row['fn']) + int(row['tp'])
            if (Fraction(row['accuracy_exact']) != accuracy or
                    int(row['tn']) + int(row['tp']) != accuracy * n or
                    sum(int(row[k]) for k in ('tn', 'fp', 'fn', 'tp')) != n):
                raise ValueError(f'R11 confusion counts disagree for {key}')
        else:
            positives = int(row['class_1_count'])
        records[key] = (n, positives, accuracy)
    expected = {k for k in product(settings, DOMAINS, DOMAINS, SEEDS, methods) if k[1] != k[2]}
    if set(records) != expected:
        raise ValueError(f'Incomplete or unexpected grid in {directory}')
    return records


def oracle(table, methods, group_key):
    """Maximize mean Accuracy within each group, then give each block equal weight."""
    groups = defaultdict(list)
    for key, scores in table.items():
        groups[group_key(*key)].append(scores)
    return sum(max(sum(block[h] for block in group) for h in methods)
               for group in groups.values()) / len(table)


def percent(value):
    return f'{100 * float(value):.9f}'


def summaries(source, curve):
    spans = [SEEDS, SEEDS[1:], *((seed,) for seed in SEEDS)]
    for setting, methods, seeds in product(SETTINGS, (THREE, FIVE), spans):
        table = {(s, t, i): {h: source[setting, s, t, i, h][2] for h in methods}
                 for s, t, i in product(DOMAINS, DOMAINS, seeds) if s != t}
        fixed = oracle(table, methods, lambda s, t, i: ())
        pool = None
        if setting == 'proteins_edge':
            pool = mean(curve[setting, s, t, i, 'CurvePool'][2] for s, t, i in table)
        for mode in (('per_seed', 'shared') if len(seeds) > 1 else ('per_seed',)):
            def seed_group(i):
                return (i,) if mode == 'per_seed' else ()

            best_setting = oracle(table, methods, lambda s, t, i: seed_group(i))
            best_source = oracle(table, methods, lambda s, t, i: (s, *seed_group(i)))
            best_direction = oracle(table, methods, lambda s, t, i: (s, t, *seed_group(i)))
            yield dict(
                setting=setting, candidates='|'.join(methods), seeds='|'.join(map(str, seeds)),
                across_seeds=mode, blocks=len(table), overall_fixed_pct=percent(fixed),
                setting_best_pct=percent(best_setting), source_best_pct=percent(best_source),
                direction_best_pct=percent(best_direction),
                source_minus_setting_pp=percent(best_source - best_setting),
                direction_minus_source_pp=percent(best_direction - best_source),
                direction_minus_setting_pp=percent(best_direction - best_setting),
                direction_minus_overall_fixed_pp=percent(best_direction - fixed),
                curve_pool_pct='' if pool is None else percent(pool),
                direction_minus_curve_pool_pp='' if pool is None else percent(best_direction - pool))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('r10', type=Path, help='R10 attempt-003-structural directory')
    parser.add_argument('r11', type=Path, help='R11 attempt-004-protein-size-exposure/score directory')
    args = parser.parse_args()
    source = load(args.r10, R10_FINAL, SETTINGS, FIVE)
    curve = load(args.r11, R11_FINAL, ('proteins_edge',), ('B0', 'Size', 'CurvePool'))
    for key, record in curve.items():
        if key[-1] in ('B0', 'Size') and source[key] != record:
            raise ValueError(f'R10/R11 block size, positives or shared endpoint differs for {key}')
        if curve[(*key[:-1], 'B0')][:2] != record[:2]:
            raise ValueError(f'R11 cohorts differ for {key}')
    rows = list(summaries(source, curve))
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
