import unittest
import numpy as np
from unseen_world import SPECS,LAYOUTS,make_world,canonical_digest,sealed_prefix_actions
class SceneTests(unittest.TestCase):
 def test_two_distinct_graph_topologies(self):
  self.assertEqual([len(SPECS[x]['room_graph']['edges'])for x in LAYOUTS],[3,3,4,4])
  self.assertEqual([len(SPECS[x]['room_graph']['edges'])-4+1 for x in LAYOUTS],[0,0,1,1])
 def test_all_architecture_geometries_differ(self):
  self.assertEqual(len({canonical_digest(SPECS[x]['wall_boxes_xyz_size'])for x in LAYOUTS}),4)
  self.assertEqual(len({tuple(SPECS[x]['shape'])for x in LAYOUTS}),4)
 def test_prefix_is_predeclared_and_returned_copy(self):
  for x in LAYOUTS:
   p=sealed_prefix_actions(x,'p1');self.assertEqual(len(p),17);p.append('stop');self.assertEqual(len(sealed_prefix_actions(x,'p1')),17)
   self.assertEqual(sealed_prefix_actions(x,'p2')[:17],sealed_prefix_actions(x,'p1'))
 def test_sensor_packet_contains_only_measured_contract(self):
  w=make_world('chain_a');f=w.sense();scan=w.scan();self.assertEqual(set(vars(f)),{'timestamp_s','depth_m','color_rgb','intrinsic','world_from_camera','semantic'});self.assertTrue(np.all(w.sense_visible_instances()[f.depth_m==0]==0));self.assertEqual(scan.ranges_m.ndim,1);self.assertTrue(np.isfinite(scan.ranges_m).all())
 def test_actual_room_connectivity_and_all_prefixes(self):
  from prepare import room_of
  for x in LAYOUTS:
   w=make_world(x);q=SPECS[x];edges=set();rooms=set()
   for rr,cc in np.argwhere(w.reachable):
    cell=(int(rr),int(cc));a=room_of(w.xy(cell),q);rooms.add(a)
    for h,(dr,dc)in enumerate([(-1,0),(0,1),(1,0),(0,-1)]):
     if w._edge_valid[cell+(h,)]:
      b=room_of(w.xy((cell[0]+dr,cell[1]+dc)),q)
      if a!=b:edges.add(tuple(sorted([a,b])))
   self.assertEqual(len(rooms),4);self.assertEqual(edges,{tuple(sorted(e))for e in q['room_graph']['edges']})
   for p in q['sealed_prefixes']:
    w.position=w.start;w.heading=q['heading'];w.step_count=w.collisions=w.moves=0
    for action in p['actions']:self.assertFalse(w.step(action)[1])
    np.testing.assert_allclose(w.xy(w.position),p['expected_xy']);self.assertEqual(w.heading,p['expected_heading'])
if __name__=='__main__':unittest.main()
