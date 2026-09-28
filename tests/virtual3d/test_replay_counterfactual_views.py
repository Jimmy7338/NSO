"""Accounting/integrity failures the independent replay must reject."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from scripts.replay_counterfactual_views import (
    area_accounting, check_manifest, compare, compare_npz, completion_status,
    extract_sources, file_hash, joint_auc, json_hash, validate_predictions, validate_route,
)


def route():
    return {'candidate_id': 0, 'cost': 4, 'arrival_action': 2, 'pose': [2, 2, 3],
            'actions': ['right'] * 4,
            'states': [[2, 2, 1], [2, 2, 2], [2, 2, 3], [2, 2, 0], [2, 2, 1]]}


def predictions(routes):
    rows = [{'candidate_id': r['candidate_id'], 'cost': r['cost'],
             'states_sha256': json_hash(r['states'], compact=True),
             'scores': {name: {'score': 1., 'final_score': 1.} for name in ('G', 'O', 'S', 'X', 'M', 'N')}}
            for r in routes]
    ids = [r['candidate_id'] for r in routes]
    return {'scorers': ['G', 'O', 'S', 'X', 'M', 'N'], 'candidates': rows,
            'rankings': {name: ids for name in ('G', 'O', 'S', 'X', 'M', 'N')},
            'selected': {name: ids[0] if ids else None for name in ('G', 'O', 'S', 'X', 'M', 'N')},
            'invariants': {'validated': True, 'candidate_routes_sha256': json_hash(
                [{key: r[key] for key in ('candidate_id', 'states', 'cost')} for r in routes], compact=True)}}


class ReplayAccountingTests(unittest.TestCase):
    def test_closed_paid_route_and_unknown_action(self):
        safe = np.ones((5, 5), bool)
        validate_route(route(), (2, 2, 1), safe, 4)
        bad = route(); bad['actions'][0] = 'stop'
        with self.assertRaisesRegex(ValueError, 'unpaid'):
            validate_route(bad, (2, 2, 1), safe, 4)

    def test_route_teleport_return_and_cost_rejected(self):
        for mutate in (lambda r: r['states'][1].__setitem__(0, 1),
                       lambda r: r['states'][-1].__setitem__(2, 3),
                       lambda r: r.__setitem__('cost', 3)):
            bad = route(); mutate(bad)
            with self.assertRaises(ValueError):
                validate_route(bad, (2, 2, 1), np.ones((5, 5), bool), 4)

    def test_semantic_cannot_change_pool_or_tie_selection(self):
        routes = [route(), route()]; routes[1]['candidate_id'] = 1
        pred = predictions(routes)
        validate_predictions(pred, routes)
        bad = copy.deepcopy(pred); bad['selected']['S'] = 1
        with self.assertRaisesRegex(ValueError, 'selection'):
            validate_predictions(bad, routes)
        bad = copy.deepcopy(pred); bad['candidates'][0]['states_sha256'] = 'fake'
        with self.assertRaisesRegex(ValueError, 'path hash'):
            validate_predictions(bad, routes)

    def test_union_ledger_preserves_physical_weights_and_negative_f1(self):
        prefix = np.array([True, False, False, False])
        masks = {'outbound': np.array([True, True, False, False]),
                 'endpoint': np.array([False, True, True, False]),
                 'return': np.array([True, True, True, True])}
        union, parts, total = area_accounting(prefix, masks, np.array([.2, .3, .4, .5]))
        self.assertTrue(union.all())
        self.assertEqual(parts, {'outbound': .3, 'endpoint': .4, 'return': .5})
        self.assertAlmostEqual(total, 1.2)
        self.assertTrue(prefix[0]); self.assertEqual(prefix.sum(), 1)
        compare({'f1_gain': -.1}, {'f1_gain': -.1}, 'negative gain')
        with self.assertRaises(ValueError):
            area_accounting(prefix, masks, np.array([.2, -.3, .4, .5]))

    def test_auc_pays_prefix_and_holds_failure_terminal_value(self):
        metrics = [{'action_index': 0, 'joint_05cm': .4}, {'action_index': 2, 'joint_05cm': .2}]
        self.assertAlmostEqual(joint_auc(metrics, 4, '05cm'), .25)
        with self.assertRaises(ValueError):
            joint_auc(metrics + [metrics[-1]], 4, '05cm')

    def test_smoke_never_claims_full_even_if_all_branches_fit(self):
        self.assertEqual(completion_status(100, 2, 2)['status'], 'partial')
        self.assertFalse(completion_status(0, 0, 2)['passed_full'])
        self.assertEqual(completion_status(None, 2, 2)['status'], 'passed_full')

    def test_comparison_rejects_missing_fields_bool_counts_and_mesh_permutation(self):
        with self.assertRaises(ValueError): compare({'a': 1}, {'a': 1, 'b': 2}, 'missing')
        with self.assertRaises(ValueError): compare(True, 1, 'boolean count')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'mesh.npz'
            points = np.array([[1., 2., 3.], [4., 5., 6.]])
            np.savez(path, vertices=points)
            compare_npz(path, {'vertices': points})
            with self.assertRaises(AssertionError): compare_npz(path, {'vertices': points[::-1]})


class ReplayIntegrityTests(unittest.TestCase):
    def test_source_archive_hash_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); run = root / 'run'; run.mkdir(); target = root / 'source'; target.mkdir()
            payload = b'# frozen dependency\n'
            sha = hashlib.sha256(payload).hexdigest()
            with zipfile.ZipFile(run / 'sources.zip', 'w') as archive:
                archive.writestr('utils/example.py', payload)
            extract_sources(run, target, {'utils/example.py': sha})
            self.assertEqual((target / 'utils/example.py').read_bytes(), payload)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                extract_sources(run, target, {'utils/example.py': 'fake'})
            with zipfile.ZipFile(run / 'sources.zip', 'w') as archive:
                archive.writestr('../escape.py', payload)
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                extract_sources(run, target, {'../escape.py': sha})

    def test_manifest_rejects_mutated_or_added_raw_and_allows_separate_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            for name in ('metadata.json', 'config.json', 'sources.zip'):
                (run / name).write_text(name)
            manifest = {p.name: file_hash(p) for p in run.iterdir()}
            (run / 'artifact_hashes.json').write_text(json.dumps(manifest))
            (run / 'verification.json').write_text('{}')
            (run / 'analysis').mkdir(); (run / 'analysis/review.json').write_text('{}')
            check_manifest(run, manifest)
            (run / 'extra_frame.npz').write_text('added')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                check_manifest(run, manifest)
            (run / 'extra_frame.npz').unlink()
            (run / 'config.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                check_manifest(run, manifest)


if __name__ == '__main__':
    unittest.main()
