from pathlib import Path
import unittest,copy,importlib.util,json
from types import SimpleNamespace as N
base=Path(__file__).resolve().parent
fixture=Path(__file__).with_name('test_rtabmap_mapping_backend.py')
src=fixture.read_text().replace("p=Path(__file__).parents[1]/'runtime/rtabmap_mapping_backend.py'","p=Path("+repr(str(base.parent/'runtime/rtabmap_mapping_backend.py'))+")")
ns=dict(__name__='fixture',__file__=str(fixture));exec(compile(src,str(fixture),'exec'),ns)
m=ns['m'];graph=ns['graph'];grid=ns['grid'];header=ns['header'];transform=ns['transform']
class DeltaTests(unittest.TestCase):
 def adapter(self):
  a=m.EpochAdapter();a.receive('grid',grid());out=a.receive('graph',graph());return a,out
 def data(self,nodeid=1,x=7,n=100):
  node=N(id=nodeid,map_id=0,stamp=.00000009,pose=N(position=N(x=x,y=0.,z=0.),orientation=transform().rotation))
  return N(header=header(n),graph=graph(n),nodes=[node],_connection_header={'callerid':'/nso_loop/rtabmap'})
 def test_identical_metadata_snapshot_exactly_unchanged(self):
  a,out=self.adapter();a.node_metadata(self.data());before=a.graph_snapshot(out[1],out[2]);g,changed=a.node_metadata(self.data(),include_changed=True)
  self.assertEqual(changed,set());self.assertEqual(before,a.graph_snapshot(out[1],out[2]))
 def test_off_graph_metadata_snapshot_exactly_unchanged(self):
  a,out=self.adapter();before=a.graph_snapshot(out[1],out[2]);g,changed=a.node_metadata(self.data(2),include_changed=True)
  self.assertEqual(changed,{2});self.assertFalse(changed.intersection(a.epoch[2].posesId));self.assertEqual(before,a.graph_snapshot(out[1],out[2]));self.assertIn(2,a.nodes)
 def test_new_in_graph_metadata_requires_snapshot_refresh(self):
  a,out=self.adapter();before=a.graph_snapshot(out[1],out[2]);g,changed=a.node_metadata(self.data(),include_changed=True)
  self.assertEqual(changed,{1});self.assertTrue(changed.intersection(a.epoch[2].posesId));self.assertNotEqual(before,a.graph_snapshot(out[1],out[2]))
 def test_changed_in_graph_metadata_requires_refresh(self):
  a,out=self.adapter();a.node_metadata(self.data());before=a.graph_snapshot(out[1],out[2]);g,changed=a.node_metadata(self.data(x=8),include_changed=True)
  self.assertEqual(changed,{1});self.assertTrue(changed.intersection(a.epoch[2].posesId));self.assertNotEqual(before,a.graph_snapshot(out[1],out[2]))
 def test_new_epoch_already_receives_publication(self):
  a,out=self.adapter();previous=a.epoch;a.receive('grid',grid(200));g,changed=a.node_metadata(self.data(n=200),include_changed=True);new=a.receive('graph',g)
  self.assertIsNotNone(new);self.assertIsNot(a.epoch,previous);self.assertEqual(new[2]['grid_source_stamp_ns'],200)
 def test_default_metadata_api_preserved(self):
  a,out=self.adapter();g=a.node_metadata(self.data());self.assertEqual(g.header.stamp.to_nsec(),100)
if __name__=='__main__':
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ns['Tests']),unittest.defaultTestLoader.loadTestsFromTestCase(DeltaTests)])
 result=unittest.TextTestRunner(verbosity=1).run(suite)
 (base/'metadata_snapshot_tests.local.json').write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()),indent=2)+'\n')
 raise SystemExit(not result.wasSuccessful())
