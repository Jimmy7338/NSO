import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.fit_response_v7 import (
    Inputs, context_inventory, execute_calibration, file_hash, guarded_predictions,
    load_outcomes, paired_information_oracle, structure_diagnostics, write_json,
)
from nso.conditional_response_v7 import HistoryFeatures


class EntryGateTests(unittest.TestCase):
    def test_missing_T_batch_stops_before_reading_any_file(self):
        inputs = Inputs()
        with patch.object(inputs, 'read', side_effect=AssertionError('must not read incomplete cohort')):
            with self.assertRaisesRegex(ValueError, 'exactly all 8'):
                context_inventory(inputs, [Path('unused')], 'train', {'pipeline': {'train_contexts': [f'T{i}' for i in range(8)]}})
        self.assertEqual(inputs.outcome_tables_opened, 0)

    def test_complete_inventory_accepts_physical_failures_but_rejects_partial_replay(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); families = ['storage_shelves', 'ventilation_baffles']
            contract = {'pipeline': {'train_contexts': [f'T{i}' for i in range(8)]}, 'contexts': {}, 'families': families}
            paths = []
            for i in range(8):
                cid = f'T{i}'; folder = root / cid; folder.mkdir(); paths.append(folder)
                context = {'context_id': cid, 'role': 'train', 'outer_seed': 751}
                contract['contexts'][cid] = context
                write_json(folder / 'artifact_hashes.json', {})
                (folder / 'sources.zip').write_bytes(b'not opened by inventory preflight')
                metadata = {'status': 'complete', 'context': context, 'physical_branches': 12,
                            'failures': 2 if i == 1 else 0, 'source_sha256': {'synthetic': 'a' * 64}}
                write_json(folder / 'metadata.json', metadata)
                report = {'status': 'passed_full', 'passed_full': True, 'partial': False, 'max_branches': None,
                    'branches_checked': 12, 'branches_total': 12, 'run': str(folder.resolve()), 'context_id': cid, 'role': 'train',
                    'raw_hashes_rechecked_after_replay': True, 'paired_geometric_prefix_exact': True,
                    'paired_route_pools_exact': True, 'paired_nonsemantic_features_exact': True,
                    'artifact_manifest_sha256': file_hash(folder / 'artifact_hashes.json'),
                    'source_archive_sha256': file_hash(folder / 'sources.zip'), 'source_sha256': metadata['source_sha256'],
                    'verifier_sha256': 'b' * 64, 'base_verifier_sha256': 'c' * 64,
                    'families': [{'family': f, 'branches_total': 6, 'checked_branches': [{}] * 6} for f in families]}
                write_json(folder / 'verification.json', report)
            inventory = context_inventory(Inputs(), paths, 'train', contract)
            self.assertEqual(inventory['T1']['meta']['failures'], 2)
            report['partial'] = True; write_json(paths[-1] / 'verification.json', report)
            with self.assertRaisesRegex(ValueError, 'replay'):
                context_inventory(Inputs(), paths, 'train', contract)

    def test_direct_outcome_reads_rejected_and_explicit_loader_aligns_failed_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            rows = [dict(candidate_id=1, paid_actions=1, planned_actions=4, failure='collision', new_area_m2=3.,
                         area_per_action=3., f1_gain_05cm=-.1),
                    dict(candidate_id=0, paid_actions=2, planned_actions=2, failure=None, new_area_m2=1.,
                         area_per_action=.5, f1_gain_05cm=.02)]
            write_json(folder / 'outcomes.json', rows)
            write_json(folder / 'candidate_001/actions.json', [{'position': [1, 2], 'heading': 0, 'stage': 'outbound'}])
            inputs = Inputs()
            with self.assertRaisesRegex(ValueError, 'explicit outcome loader'):
                inputs.read(folder / 'outcomes.json')
            packet = {'folder': folder, 'history': HistoryFeatures('T0/f', 'T0', {}),
                'routes': [{'candidate_id': 0, 'cost': 2, 'states': [[0, 0, 0]]},
                           {'candidate_id': 1, 'cost': 4, 'states': [[0, 0, 0]]}]}
            targets = load_outcomes(inputs, [packet])
            np.testing.assert_array_equal(targets['T0/f'], [[.5, .02], [3., -.1]])
            self.assertFalse(packet['outcomes'][1]['returned_to_origin'])
            self.assertEqual(inputs.outcome_tables_opened, 1)

    def test_hash_checker_rejects_unsealed_missing_assets(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            write_json(folder / 'artifact_hashes.json', {'raw_depth.npz': 'a' * 64})
            with self.assertRaisesRegex(ValueError, 'missing asset'):
                Inputs().manifest(folder)

    def test_whole_history_guard_preserves_raw_and_uses_frozen_N(self):
        features = {'G': np.zeros((2, 12))}; features['G'][1, 0] = 2.
        history = HistoryFeatures('C0/f', 'C0', features)
        values = {'G': np.array([[1., 0.], [-1., 0.]]), 'S': np.array([[2., 0.], [-2., 0.]]),
                  'N': np.array([[-1., 0.], [1., 0.]])}
        bank = SimpleNamespace(predict=lambda h: values)
        guard = {'columns': list(range(8)), 'lower': [-1.] * 8, 'upper': [1.] * 8}
        raw, guarded, decision = guarded_predictions(bank, history, guard)
        self.assertTrue(decision['fallback'])
        np.testing.assert_array_equal(raw['S'], [[2., 0.], [-2., 0.]])
        for array in guarded.values():
            np.testing.assert_array_equal(array, values['N'])

    def test_marked_direction_support_cannot_be_supplied_by_background(self):
        packets = []
        for sign, family in ((-1, 'storage_shelves'), (1, 'ventilation_baffles')):
            matrix = np.zeros((6, 12)); matrix[:, 4:6] = 1.; matrix[:, 8] = sign * np.arange(6)
            features = {'S': matrix, 'O': matrix.copy()}
            patches = [{'marked_points': 3, 'class_vote': sign}, {'marked_points': 0, 'class_vote': 0}]
            descriptors = [[[0., 0., 0., 0.], [1., 1., 1., 1.]]] * 6
            packets.append({'history': HistoryFeatures(f'T0/{family}', 'T0', features),
                'feature_audit': {'patches': patches, 'routes': [{'unscaled_patch_descriptors': d} for d in descriptors]},
                'candidate_audit': {'available_roles': [{'role': 'front', 'patch': 0}, {'role': 'left', 'patch': 0}]},
                'routes': [{'group': 'front', 'cost': i + 2} for i in range(6)]})
        report = structure_diagnostics(packets)
        self.assertFalse(report['passed'])
        self.assertFalse(report['class_support']['1']['front'])
        self.assertFalse(report['class_support']['-1']['side'])

    def test_paired_oracle_has_positive_value_only_for_conflicting_optima(self):
        routes = [{'candidate_id': 0}, {'candidate_id': 1}]
        packets = []
        for family, values in (('storage_shelves', [3., 1.]), ('ventilation_baffles', [1., 3.])):
            packets.append({'history': HistoryFeatures('T0/' + family, 'T0', {}), 'family': family,
                            'routes': routes, 'outcomes': [{'area_per_action': x} for x in values]})
        report = paired_information_oracle(packets)['contexts'][0]
        self.assertEqual(report['value_of_family_information_upper_bound'], 1.)
        self.assertFalse(report['maximizer_sets_intersect_exactly'])
        packets[1]['outcomes'] = copy.deepcopy(packets[0]['outcomes'])
        report = paired_information_oracle(packets)['contexts'][0]
        self.assertEqual(report['value_of_family_information_upper_bound'], 0.)
        self.assertTrue(report['maximizer_sets_intersect_exactly'])

    def test_execution_guard_cannot_be_attached_after_C_acquisition(self):
        with tempfile.TemporaryDirectory() as raw:
            args = SimpleNamespace(output=Path(raw))
            with self.assertRaisesRegex(ValueError, 'preexisting C run'):
                execute_calibration(args)


if __name__ == '__main__':
    unittest.main()
