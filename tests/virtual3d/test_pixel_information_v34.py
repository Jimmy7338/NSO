"""Saved reward adapter and sensor identity checks; zero world/DP calls."""
import json
from pathlib import Path
import unittest
import numpy as np
from nso.pixel_information_v34 import SavedPotentialModelV34, arrays_sha256, information_pair_v34

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'audit_results/v33_direction_information_r1_20260917'


class PixelInformationAdapterTests(unittest.TestCase):
    def test_saved_prefix_and_all_witness_reward_values(self):
        checks=0
        for parent in ('P00','P01'):
            geometry=json.loads((OLD/(parent+'_geometry.json')).read_text())
            policies=json.loads((OLD/(parent+'_policies.json')).read_text())
            for h,t in enumerate(geometry['tables']):
                model=SavedPotentialModelV34(t,t['signature_sha256'])
                cases=[(model.initial_mask,t['prefix_terminal'])]
                cases += [(int(w['final_observed_mask_hex'],16),w['terminal'])
                    for w in policies['witnesses'] if w['actual_hypothesis']==h]
                for mask,expected in cases:
                    actual=model.terminal(mask)
                    for key in ('coverage','surface','joint'):
                        self.assertAlmostEqual(actual[key],expected[key],places=12)
                    self.assertEqual(actual['feasible'],expected['feasible']);checks+=1
        self.assertEqual(checks,16)

    def test_information_partition_does_not_modify_rewards(self):
        for parent in ('P00','P01'):
            geometry=json.loads((OLD/(parent+'_geometry.json')).read_text())
            original=[SavedPotentialModelV34(t,t['signature_sha256']) for t in geometry['tables']]
            paired=information_pair_v34(original)
            self.assertEqual(paired['first_information_action_layer'],geometry['pair']['first_information_action_layer'])
            changed=[SavedPotentialModelV34(t,[str(h)]*len(t['poses'])) for h,t in enumerate(geometry['tables'])]
            self.assertEqual(information_pair_v34(changed)['first_information_action_layer'],0)
            self.assertFalse(information_pair_v34(changed)['prefix_equal'])
            for a,b in zip(original,changed):
                self.assertEqual(a.observed_masks,b.observed_masks)
                self.assertEqual(a.terminal(a.initial_mask),b.terminal(b.initial_mask))

    def test_array_identity_includes_field_dtype_shape_and_actual_bytes(self):
        x=np.array([1.,2.],np.float32)
        self.assertEqual(arrays_sha256(a=x,b=x+1),arrays_sha256(b=x+1,a=x.copy()))
        for other in (arrays_sha256(b=x),arrays_sha256(a=x.astype(np.float64)),
                      arrays_sha256(a=x.reshape(1,2)),arrays_sha256(a=x+1)):
            self.assertNotEqual(arrays_sha256(a=x),other)
        with self.assertRaises(ValueError):arrays_sha256(a=np.array([{}],object))


if __name__=='__main__':unittest.main()
