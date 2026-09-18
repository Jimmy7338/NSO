"""Implementation contracts; no project world, saved mesh, or TSDF access."""
from types import SimpleNamespace
import unittest
import numpy as np
from shapely.geometry import Polygon
from nso.supported_outline_v31 import supported_projection_v31, compare_supported_projection_v31, FullSupportedOutlineEvaluatorV31


def vertical_panels(edges):
    vertices=[];triangles=[]
    for a,b in edges:
        offset=len(vertices)
        vertices.extend([[*a,0.],[*b,0.],[*b,1.],[*a,1.]])
        triangles.extend([[offset,offset+1,offset+2],[offset,offset+2,offset+3]])
    return SimpleNamespace(vertices=np.asarray(vertices,float).reshape(-1,3),
        triangles=np.asarray(triangles,np.int32).reshape(-1,3))


class SupportedOutlineContracts(unittest.TestCase):
    def test_exact_closed_lines_have_area_without_cap(self):
        mesh=vertical_panels([((0,0),(1,0)),((1,0),(1,1)),((1,1),(0,1)),((0,1),(0,0))])
        before=mesh.vertices.copy()
        result=supported_projection_v31(mesh,(0,1))
        self.assertAlmostEqual(result['area'].area,1.)
        self.assertEqual(result['receipt']['added_connections'],0)
        np.testing.assert_array_equal(before,mesh.vertices)

    def test_positive_gap_is_not_closed(self):
        mesh=vertical_panels([((0,0),(1,0)),((1,0),(1,1)),((1,1),(0,1)),((0,1),(0,.001))])
        result=supported_projection_v31(mesh,(0,1))
        self.assertTrue(result['area'].is_empty)
        self.assertAlmostEqual(result['residual_lines'].length,3.999)

    def test_external_dangling_line_is_not_discarded(self):
        edges=[((0,0),(1,0)),((1,0),(1,1)),((1,1),(0,1)),((0,1),(0,0)),((1,1),(3,1))]
        result=supported_projection_v31(vertical_panels(edges),(0,1))
        self.assertAlmostEqual(result['area'].area,1.)
        self.assertAlmostEqual(result['residual_lines'].length,2.)
        score=compare_supported_projection_v31(result,Polygon([(0,0),(1,0),(1,1),(0,1)]))
        self.assertEqual(score['iou'],1.)
        self.assertLess(score['05cm']['precision'],.7)

    def test_duplicate_edges_are_a_geometric_set(self):
        edges=[((0,0),(1,0)),((1,0),(1,1)),((1,1),(0,1)),((0,1),(0,0))]
        a=supported_projection_v31(vertical_panels(edges),(0,1))
        b=supported_projection_v31(vertical_panels(edges*3),(0,1))
        self.assertTrue(a['support'].equals(b['support']))

    def test_three_dimensional_wire_triangle_is_rejected(self):
        mesh=SimpleNamespace(vertices=np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),
            triangles=np.array([[0,1,2]]))
        with self.assertRaisesRegex(ValueError,'zero-area 3D'):
            supported_projection_v31(mesh,(0,1))

    def test_unassigned_invalid_output_is_still_rejected(self):
        import open3d as o3d
        ref=o3d.geometry.TriangleMesh.create_box(1.,1.,1.)
        evaluator=FullSupportedOutlineEvaluatorV31([dict(id=0,mesh=ref,bounds=[[-.1]*3,[1.1]*3])])
        invalid=SimpleNamespace(vertices=np.array([[3.,0.,0.],[4.,0.,0.],[5.,0.,0.]]),
            triangles=np.array([[0,1,2]]))
        with self.assertRaisesRegex(ValueError,'zero-area 3D'):
            evaluator.evaluate([ref,invalid],[dict(observed_seed_xyz=[0.,0.,0.]),None],2,1.,
                returned=True,paid_actions=0,budget=0)


if __name__=='__main__':unittest.main()
