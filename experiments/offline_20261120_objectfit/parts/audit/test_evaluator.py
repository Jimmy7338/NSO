#!/usr/bin/env python3
"""Check B6 evaluator against an already frozen OF self-comparison fixture."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('parts_eval_fixture',HERE/'evaluate_parts.py')
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
np=evaluation.np

class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory=evaluation.B5/'replay/shelf_left_24_attempt_001'
        manifest=json.loads((directory/'manifest.json').read_text())
        cls.row=next(row for row in manifest['stages']if row['variant']=='object_fit')
        cls.row=copy.deepcopy(cls.row)
        cls.row['baseline_predictions']=copy.deepcopy(cls.row['predictions'])
        cls.row['baseline_prediction_receipt']=copy.deepcopy(cls.row['prediction_receipt'])
        cls.e=evaluation.Evaluator()
        cls.e.support_arrays=dict(np.load(manifest['observed_support']['path'],allow_pickle=False))
        cls.e.support_metadata=json.loads(Path(manifest['observed_support_metadata']['path']).read_text())

    def test_frozen_OF_self_comparison_is_exact(self):
        result=self.e.stage(self.row,'shelf_left')
        for name in ('G','S'):
            data=result['observed_object_attribution'][name]
            self.assertEqual(data['current'],data['B5_object_fit_same_track_baseline'])
            self.assertTrue(all(all(check.values())for check in data['same_track_forecast_parity'].values()))
            self.assertEqual(data['support_validation']['basis'],'full')
            self.assertFalse(data['lost_baseline_tracks'])

    def test_baseline_digest_mismatch_rejected(self):
        row=copy.deepcopy(self.row)
        row['baseline_predictions']['sha256']='0'*64
        with self.assertRaises(AssertionError):self.e.stage(row,'shelf_left')

if __name__=='__main__':unittest.main(verbosity=2)
