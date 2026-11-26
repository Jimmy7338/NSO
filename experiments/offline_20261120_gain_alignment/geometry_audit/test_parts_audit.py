import unittest
import numpy as np
import parts_audit as p
class BoundaryTests(unittest.TestCase):
 def test_single_box_has_no_internal_or_duplicate(self):
  parts=[([0,0,0],[1,1,1])];xyz,w,n,ids=p.quad(parts);i,d,m=p.boundary_classification(parts,xyz,n,ids)
  self.assertFalse(i.any());self.assertFalse(d.any());self.assertAlmostEqual(w.sum(),6)
 def test_identical_components_are_same_outer_face_duplicates(self):
  parts=[([0,0,0],[1,1,1])]*2;xyz,w,n,ids=p.quad(parts);i,d,m=p.boundary_classification(parts,xyz,n,ids)
  self.assertFalse(i.any());self.assertTrue(d.all());self.assertTrue((m==2).all());self.assertAlmostEqual((w/m).sum(),6)
 def test_joined_boxes_internal_contact_and_external_area(self):
  parts=[([0,0,0],[1,1,1]),([1,0,0],[1,1,1])];xyz,w,n,ids=p.quad(parts);i,d,m=p.boundary_classification(parts,xyz,n,ids)
  self.assertAlmostEqual(w[i].sum(),2);self.assertFalse(d.any());self.assertAlmostEqual(w[~i].sum(),10)
 def test_alternative_hypotheses_must_not_be_unioned(self):
  for origin in [[0,0,0],[.2,0,0]]:
   parts=[(origin,[1,1,1])];xyz,w,n,ids=p.quad(parts);i,d,m=p.boundary_classification(parts,xyz,n,ids);self.assertFalse(i.any())
if __name__=='__main__':unittest.main()
