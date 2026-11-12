import unittest
import numpy as np
from audit_module import _costs, _raw_geometry_groups
from types import SimpleNamespace

class NativeCostChecks(unittest.TestCase):
    def test_final_heading_is_paid(self):
        cells,out,ret=_costs(np.ones((1,1),bool),(0,0,0),(0,0,0))
        self.assertEqual(out.tolist(),[0,1,2,1])
        self.assertEqual((out+ret).tolist(),[0,2,4,2])
    def test_forward_only_full_return(self):
        cells,out,ret=_costs(np.ones((1,2),bool),(0,0,1),(0,0,1))
        # East one cell costs1; returning needs two turns, west forward,
        # then two turns to restore east at original anchor:5.
        self.assertEqual(out[5],1)
        self.assertEqual(ret[5],5)
        self.assertEqual(out[5]+ret[5],6)
    def test_known_safe_disconnection(self):
        cells,out,ret=_costs(np.array([[1,0,1]],bool),(0,0,0),(0,0,0))
        self.assertTrue(np.isinf(out[4:]).all())
        self.assertTrue(np.isinf(ret[4:]).all())
    def test_unsafe_current_rejected(self):
        with self.assertRaisesRegex(ValueError,'current_footprint_not_known_safe'):
            _costs(np.array([[0,1]],bool),(0,0,0),(0,1,0))

class OriginalGeometryChecks(unittest.TestCase):
    def test_boundary_and_large_marked_group_excluded(self):
        # Synthetic exact original-threshold fixture, not robot observations.
        points=[]
        for cols in [[0],list(range(5,10)),list(range(15,36))]:
            points.extend([[c*.2+.1,3.9,z] for c in cols for z in [.1,.6]])
        q=dict(point=np.array(points),label=np.full(len(points),3))
        mapper=SimpleNamespace(shape=(50,50),measured_extent_profile='measured_extent_3m_v1',
            quality_evidence=lambda max_points:q)
        assets=[dict(group=1,support_points=10,marked_points=10)]
        r=_raw_geometry_groups(mapper,assets)
        self.assertEqual(r['quality_sample_class3'],54)
        self.assertTrue(r['passed_groups_equal_native'])
        self.assertTrue(r['passed_support_and_marked_counts_equal_native'])
        self.assertEqual(r['groups'][0]['first_failed'],'group0_excluded_before_geometry_loop')
        rejected=[g for g in r['groups'] if g['first_failed']=='max_horizontal_extent']
        self.assertEqual(rejected[0]['marked_class3'],42)
        self.assertEqual(r['thresholds']['max_horizontal_extent_m'],3.)

if __name__=='__main__':unittest.main()
