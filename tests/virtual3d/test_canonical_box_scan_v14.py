import unittest
import numpy as np
from env.canonical_box_scan_v14 import box_union_ranges


class CanonicalBoxScanTests(unittest.TestCase):
    def test_closed_form_hits_parallel_misses_and_range_cap(self):
        directions = [[1,0,0],[0,1,0],[-1,0,0]]
        actual = box_union_ranges([0,0,.5],directions,[[2,-1,0,1,2,1]],5.)
        np.testing.assert_array_equal(actual,np.array([2.,5.,5.],np.float32))

    def test_equal_front_face_and_primitive_order_have_identical_ranges(self):
        angles = np.linspace(-.2,.2,23)
        rays = np.column_stack([np.cos(angles),np.sin(angles),np.zeros(len(angles))])
        full = [[2.9,-2,0,.8,4,1]]
        panel = [[2.9,-2,0,.08,4,1],[3.62,-2,0,.08,4,1]]
        expected = box_union_ranges([0,0,.5],rays,full,8.)
        np.testing.assert_array_equal(expected,box_union_ranges([0,0,.5],rays,panel,8.))
        np.testing.assert_array_equal(expected,box_union_ranges([0,0,.5],rays,panel[::-1],8.))

    def test_nearest_union_surface_and_empty_scene(self):
        rays = [[1.,0.,0.]]
        self.assertEqual(box_union_ranges([0,0,.5],rays,[[3,-1,0,2,2,1],[2,-1,0,2,2,1]],10.)[0],2.)
        self.assertEqual(box_union_ranges([0,0,.5],rays,[],10.)[0],10.)

    def test_inside_origin_and_invalid_direction_rejected(self):
        with self.assertRaisesRegex(ValueError,'inside'):
            box_union_ranges([.5,.5,.5],[[1,0,0]],[[0,0,0,1,1,1]],5.)
        with self.assertRaisesRegex(ValueError,'unit'):
            box_union_ranges([0,0,.5],[[2,0,0]],[],5.)


if __name__=='__main__':
    unittest.main()
