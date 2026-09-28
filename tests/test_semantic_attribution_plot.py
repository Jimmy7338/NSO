"""Reporting invariants; synthetic endpoints never execute a scientific slot."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'attribution_plot', ROOT / 'scripts/plot_semantic_attribution_results.py')
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class AttributionReportingTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / 'configs/virtual3d/semantic_development_acquisition_20260923.json').read_text())
        rows = []
        for run_id, slot in self.protocol['slots'].items():
            parent, condition = slot['asset_id'].split('__', 1)
            rows.append(dict(slot, run_id=run_id, parent_id=parent, condition=condition,
                             state='not_attempted', metrics=None, C_nav=None, Q=None,
                             J_nav=None, evaluation_execution=None))
        self.data = dict(schema='semantic.scene_matrix.progress_summary.v1',
                         states={'integrity_error': 0}, episodes=rows,
                         all_declared_endpoints_reviewed=False)

    def measured(self, run_id, coverage, quality):
        row = next(r for r in self.data['episodes'] if r['run_id'] == run_id)
        row.update(state='complete_episode', metrics={'C_nav': coverage, 'Q': quality,
                                                     'J_nav': coverage * quality},
                   C_nav=coverage, Q=quality, J_nav=coverage * quality,
                   evaluation_execution='recomputed')
        return row

    def test_missing_ablation_remains_missing_despite_known_baseline(self):
        self.measured('core_P00_nom_S_b120_lexicographic', .8, .5)
        rows = PLOT.validate_snapshot(self.data, self.protocol)
        table, _ = PLOT.build_tables(rows)
        row = next(r for r in table if r['candidate_run_id'] == 'ablation_P00_nom_NF_b120_lexicographic')
        self.assertFalse(row['complete_reviewed_pair'])
        self.assertIsNone(row['delta_J_nav'])
        self.assertIsNone(row['relative_delta_J_percent'])
        self.assertAlmostEqual(row['baseline_J_nav'], .4)

    def test_zero_baseline_is_not_a_relative_gain(self):
        self.measured('core_P00_nom_S_b120_lexicographic', 0, .5)
        self.measured('ablation_P00_nom_NF_b120_lexicographic', .8, .5)
        table, _ = PLOT.build_tables(PLOT.validate_snapshot(self.data, self.protocol))
        row = next(r for r in table if r['candidate_run_id'] == 'ablation_P00_nom_NF_b120_lexicographic')
        self.assertTrue(row['complete_reviewed_pair'])
        self.assertAlmostEqual(row['delta_J_nav'], .4)
        self.assertIsNone(row['relative_delta_J_percent'])

    def test_shared_belief_contrast_uses_nc_and_b(self):
        self.measured('core_P00_nom_B_b120_lexicographic', .8, .5)
        self.measured('ablation_P00_nom_NC_b120_lexicographic', .9, .5)
        table, _ = PLOT.build_tables(PLOT.validate_snapshot(self.data, self.protocol))
        row = next(r for r in table if r['contrast'] == 'NC_vs_B'
                   and r['parent_id'] == 'SEM_P00' and r['condition'] == 'nominal_relationship')
        self.assertEqual(row['baseline_run_id'], 'core_P00_nom_B_b120_lexicographic')
        self.assertAlmostEqual(row['relative_delta_J_percent'], 12.5)
        self.assertAlmostEqual(row['delta_C_nav'], .1)

    def test_mismatched_budget_cannot_be_an_ablation_pair(self):
        rows = PLOT.validate_snapshot(self.data, self.protocol)
        target = rows['ablation_P00_nom_NF_b120_lexicographic']
        baseline = copy.deepcopy(rows['core_P00_nom_S_b120_lexicographic'])
        baseline['budget'] = 80
        with self.assertRaisesRegex(ValueError, 'non-method'):
            PLOT.paired_row(target, baseline, 'NF_vs_S')

    def test_all_predeclared_rows_and_reused_anchors_retained(self):
        pairs, boundary = PLOT.build_tables(PLOT.validate_snapshot(self.data, self.protocol))
        self.assertEqual(len(pairs), 16)
        self.assertEqual({r['parent_id'] for r in pairs}, {'SEM_P00', 'SEM_P04'})
        self.assertEqual(len(boundary), 30)
        self.assertEqual(sum(r['reused_core_anchor'] for r in boundary), 6)
        self.assertEqual(len({r['run_id'] for r in boundary}), 30)
        self.assertTrue(all(r['J_nav'] is None for r in boundary))

    def test_plotted_values_must_match_embedded_review(self):
        row = self.measured('core_P00_nom_S_b120_lexicographic', .8, .5)
        row.update(C_nav=.5, Q=.5, J_nav=.25)
        with self.assertRaisesRegex(ValueError, 'embedded reviewed'):
            PLOT.validate_snapshot(self.data, self.protocol)

    def test_incomplete_snapshot_cannot_claim_completion(self):
        self.measured('core_P00_nom_S_b120_lexicographic', .8, .5)
        self.data['all_declared_endpoints_reviewed'] = True
        with self.assertRaisesRegex(ValueError, 'Completion flag'):
            PLOT.validate_snapshot(self.data, self.protocol)

    def test_imputation_inconsistent_scores_and_unreviewed_metrics_rejected(self):
        pristine = copy.deepcopy(self.data)
        for fault in ('imputation', 'identity', 'provenance', 'metadata'):
            with self.subTest(fault=fault):
                self.data = copy.deepcopy(pristine)
                row = self.data['episodes'][0]
                if fault == 'imputation':
                    row['J_nav'] = 0.
                elif fault == 'metadata':
                    row['budget'] = 81
                else:
                    row = self.measured(row['run_id'], .8, .5)
                    if fault == 'identity':
                        row['J_nav'] = .9
                    else:
                        row['evaluation_execution'] = 'predicted'
                with self.assertRaises(ValueError):
                    PLOT.validate_snapshot(self.data, self.protocol)


if __name__ == '__main__':
    unittest.main()
