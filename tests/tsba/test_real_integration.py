"""Synthetic staged artifact integration; no real graph labels are opened."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch

from tsba.classifier import ResidualClassifier
from workflows.tsba.validation.archive import digest, save_tensor, write_json
from workflows.tsba.validation.real_data.data import hash_order, label_offsets, probe_folds
from workflows.tsba.validation.real_data.runtime import seal
from workflows.tsba.validation.real_data.probes import run_probes
from workflows.tsba.validation.real_data.geometry import run_source_coverage, run_cell
from workflows.tsba.validation.real_data.evaluate import evaluate_run
from workflows.tsba.validation.real_data.audit import replay_predictions


class RealIntegrationTests(unittest.TestCase):
    def test_synthetic_staged_probabilities_geometry_and_evaluation(self):
        torch.set_num_threads(1)
        repo = Path(__file__).resolve().parents[2]
        plan = json.loads((repo / 'experiments/mutagenicity/r1-20261111/stage1-plan.json').read_text())
        plan = copy.deepcopy(plan)
        plan.update(seeds=[3405], tasks=[[0, 1]])
        plan['probes'].update(updates=20, supplementary_endpoints=[20])
        plan['density']['bootstrap_repeats'] = 4
        plan['geometry']['scan_grid'] = 17
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ('inputs', 'roles', 'logs', 'source/seed3405', 'probes', 'geometry', 'evaluation'):
                (root / directory).mkdir(parents=True, exist_ok=True)
            source_ids, target_ids = list(range(220)), list(range(220, 440))
            target = dict(ids=target_ids, A=target_ids[:147], B=target_ids[147:211], final=target_ids[211:])
            target['query_ids'] = hash_order(target['A'], f'{plan["protocol_id"]}|geometry|1')[:64]
            roles = dict(source=dict(domain_ids=source_ids), targets={'1': target})
            write_json(root / 'roles.json', roles)
            sr = dict(fit_ids=source_ids[:16], validation_ids=source_ids[16:24], diagnostic_ids=source_ids[24:])
            sr['folds'] = probe_folds(sr['diagnostic_ids'], roles['targets'], plan['protocol_id'])
            write_json(root / 'roles/seed3405.json', sr)
            generator = torch.Generator().manual_seed(929)
            x = torch.randn((440, 16), dtype=torch.float64, generator=generator)
            labels = (x[:, 0] < 0).long()
            label_file = root / 'synthetic-labels.txt'
            label_file.write_text(''.join('FINAL_SENTINEL\n' if i in target['final'] else f'{int(y)}\n'
                                         for i, y in enumerate(labels)))
            write_json(root / 'inputs/broker.json', dict(path=str(label_file), offsets=label_offsets(label_file), sha256=digest(label_file)))
            write_json(root / 'inputs/class_mapping.json', dict(raw_to_class={'0': 0, '1': 1}))
            size = torch.log1p(x[:, :2].abs() * 20 + 5)
            save_tensor(root / 'inputs/size.pt', size)
            write_json(root / 'inputs/graphs.json', [dict(id=i, nodes=int(size[i, 0].exp()),
                       undirected_edges=int(size[i, 1].exp())) for i in range(len(x))])
            head = ResidualClassifier(16, 2, 16, dtype=torch.float64)
            with torch.no_grad():
                head.linear.weight.zero_()
                head.linear.bias.zero_()
                head.linear.weight[0, 0] = 1
                head.linear.weight[1, 0] = -1
                logits = head(x)
            save_tensor(root / 'source/seed3405/frozen.pt', dict(x=x, h=x, mean=x.mean(0),
                        scale=torch.tensor(1., dtype=torch.float64), initial_head=head.state_dict()))
            save_tensor(root / 'source/seed3405/predictions.pt', dict(probabilities=logits.softmax(-1),
                        log_probabilities=logits.log_softmax(-1), ids=torch.arange(len(x))))
            run_source_coverage(root, 3405, 1, plan)
            s1 = seal(root, 'S1', ['inputs', 'roles', 'roles.json', 'source', 'geometry'])
            run_probes(root, plan)
            run_cell(root, 3405, 1, plan)
            checked = replay_predictions(root, plan)
            self.assertTrue(checked['exact_match'])
            seal(root, 'S2', ['seals/S1.json', 'inputs', 'roles', 'roles.json', 'source', 'geometry', 'probes'])
            evaluate_run(root, plan)
            for path, expected in s1['files'].items():
                self.assertEqual(digest(root / path), expected)
            audit = [json.loads(line) for line in (root / 'logs/label-access.jsonl').read_text().splitlines()]
            self.assertFalse(any(set(r['ids']) & set(target['final']) for r in audit))
            self.assertTrue(any(r['role'] == 'target_B' and r['permitted'] for r in audit))
            self.assertEqual(len(list((root / 'probes').rglob('metadata.json'))), 64)


if __name__ == '__main__':
    unittest.main()
