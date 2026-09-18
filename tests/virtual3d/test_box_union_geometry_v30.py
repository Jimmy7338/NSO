"""Exact analytic counterexamples, without World, sensor, TSDF, or mesh dependencies."""
import unittest
from nso.box_union_geometry_v30 import BoxV30, union_exterior_faces_v30, union_volume_v30, surface_audit_v30


class BoxUnionGeometryV30Tests(unittest.TestCase):
    def assert_closed(self,boxes,volume):
        audit=surface_audit_v30(union_exterior_faces_v30(boxes))
        self.assertTrue(audit['closed_oriented_two_manifold'],audit)
        self.assertAlmostEqual(union_volume_v30(boxes),volume,places=11)
        self.assertAlmostEqual(audit['signed_volume'],volume,places=11)

    def test_body_exact_area_and_volume(self):
        boxes=[BoxV30((-1,1,2,5,0,1.6),0)]
        self.assertAlmostEqual(sum(f.area for f in union_exterior_faces_v30(boxes,vertical_only=True)),16.)
        self.assert_closed(boxes,9.6)

    def test_partial_contact_is_clipped_exactly(self):
        boxes=[BoxV30((-1,1,2,5,0,1.6),0),BoxV30((-.6,.6,5,5.7,.4,1.2),0)]
        faces=union_exterior_faces_v30(boxes,vertical_only=True)
        self.assertAlmostEqual(sum(f.area for f in faces),17.12,places=11)
        contact_plane=[f for f in faces if f.axis==1 and f.bounds[2]==5]
        self.assertAlmostEqual(sum(f.area for f in contact_plane),3.2-.96,places=11)
        for f in contact_plane:
            self.assertFalse(f.bounds[0]>=-.6 and f.bounds[1]<=.6 and f.bounds[4]>=.4 and f.bounds[5]<=1.2)
        self.assert_closed(boxes,10.272)

    def test_overlap_duplicate_and_contained_boxes_do_not_double_count(self):
        box=BoxV30((0,1,0,1,0,1),7)
        boxes=[box,box,BoxV30((.2,.8,.2,.8,.2,.8),7),BoxV30((.5,1.5,0,1,0,1),7)]
        self.assertAlmostEqual(sum(f.area for f in union_exterior_faces_v30(boxes)),8.)
        self.assert_closed(boxes,1.5)

    def test_nonconvex_l_shape_is_closed(self):
        boxes=[BoxV30((0,2,0,1,0,1)),BoxV30((0,1,1,2,0,1))]
        self.assertAlmostEqual(sum(f.area for f in union_exterior_faces_v30(boxes)),14.)
        self.assert_closed(boxes,3.)

    def test_owner_is_only_metadata_and_conflicting_overlap_rejected(self):
        a=[BoxV30((0,1,0,1,0,1),0),BoxV30((1,2,0,1,0,1),1)]
        b=[BoxV30(x.bounds,1-x.owner) for x in a]
        identity=lambda boxes:[(f.bounds,f.axis,f.sign) for f in union_exterior_faces_v30(boxes)]
        self.assertEqual(identity(a),identity(b)); self.assert_closed(a,2.)
        with self.assertRaises(ValueError):
            union_exterior_faces_v30([a[0],BoxV30((.5,1.5,0,1,0,1),1)])

    def test_edge_and_corner_touch_not_falsely_called_watertight(self):
        first=BoxV30((0,1,0,1,0,1),0)
        for bounds in ((1,2,1,2,0,1),(1,2,1,2,1,2)):
            boxes=[first,BoxV30(bounds,0)]
            self.assertFalse(surface_audit_v30(union_exterior_faces_v30(boxes))['closed_oriented_two_manifold'])
            self.assertAlmostEqual(union_volume_v30(boxes),2.)

    def test_common_refinement_preserves_geometry_and_volume(self):
        boxes=[BoxV30((0,1,0,1,0,1),0)]
        faces=union_exterior_faces_v30(boxes,split_planes=([.2,.6],[.3],[.4,.8]))
        self.assertAlmostEqual(sum(f.area for f in faces),6.)
        audit=surface_audit_v30(faces)
        self.assertTrue(audit['closed_oriented_two_manifold']);self.assertAlmostEqual(audit['signed_volume'],1.)

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(union_exterior_faces_v30([]),());self.assertEqual(union_volume_v30([]),0.)
        for bounds in ((0,0,0,1,0,1),(1,0,0,1,0,1),(0,float('inf'),0,1,0,1)):
            with self.assertRaises(ValueError):BoxV30(bounds)
        with self.assertRaises(ValueError):BoxV30((0,1,0,1,0,1),'complex')
        with self.assertRaises(TypeError):union_exterior_faces_v30([(0,1,0,1,0,1)])


if __name__=='__main__':unittest.main()
