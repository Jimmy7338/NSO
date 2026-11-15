"""Safety/provenance tests with fake process/master state; never executes reset."""
import contextlib,copy,importlib.util,io,json,pathlib,tempfile,types,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('reset_driver',P/'root_reset_fresh_rtab.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class DriverTests(unittest.TestCase):
 def test_snapshot_requires_fresh_exact_source_and_no_bootstrap(self):
  spec=dict(epoch_id='new',new_database_path='/new/rtabmap.db',epoch_start_s=10.)
  valid=dict(schema='rtabmap_mapping_backend/1',source_epoch='new',source_database_path='/new/rtabmap.db',map_frame='nso_loop_map',alias_frame='nso_pair_map',source_caller='/nso_loop/rtabmap',map_stamp_ns=10000000001,graph_source_stamp_ns=10000000001,grid_source_stamp_ns=10000000001,initial_metadata_receipt=None,recorded_epoch_graph=None,nodes=[{'odom_from_base':[[1.]]}])
  self.assertTrue(m.snapshot_valid(valid,spec))
  for diff in [dict(source_epoch='old'),dict(source_database_path='/old/rtabmap.db'),dict(graph_source_stamp_ns=10000000002),dict(initial_metadata_receipt={'old':True}),dict(recorded_epoch_graph={'old':True}),dict(nodes=[]),dict(map_stamp_ns=9),dict(source_caller='/other')]:
   self.assertFalse(m.snapshot_valid(dict(valid,**diff),spec),diff)
 def test_process_identity_changes_cannot_be_signalled_as_owned(self):
  q=dict(pid=123,uid=10,start_ticks=100,argv=['owned'])
  with patch.object(m,'proc',return_value=q):self.assertTrue(m.same(q))
  for changed in [dict(q,start_ticks=101),dict(q,uid=11),dict(q,argv=['unrelated'])]:
   with patch.object(m,'proc',return_value=changed):self.assertFalse(m.same(q))
 def test_stop_signals_only_captured_parent_without_killpg(self):
  root=dict(pid=123,uid=10,start_ticks=100,argv=['owned']);child=dict(pid=124,uid=10,start_ticks=101,argv=['owned_child']);alive={123:True,124:True};calls=[]
  def identity(pid):return (root if pid==123 else child) if alive.get(pid) else None
  def send(pid,sig):
   calls.append((pid,sig));alive[123]=alive[124]=False
  with patch.object(m,'proc',side_effect=identity),patch.object(m.os,'kill',side_effect=send),patch.object(m.os,'killpg',side_effect=AssertionError('group signal')):
   m.stop_target(dict(name='owned',root=root,descendants=[child]))
  self.assertEqual(calls,[(123,m.signal.SIGINT)])
 def test_default_readonly_path_never_writes_or_launches(self):
  with tempfile.TemporaryDirectory() as td:
   reg=pathlib.Path(td)/'registry.json';reg.write_text(json.dumps({'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
   spec=dict(epoch_id='new',new_database_path='/absent/new/rtabmap.db')
   args=types.SimpleNamespace(epoch_id='new',execute=False,environment_from=None)
   with patch.object(m,'REGISTRY',reg),patch.object(m,'assert_no_actuators'),patch.object(m.B,'build',return_value=spec),patch.object(m,'owned_targets',return_value=([],'/old/rtabmap.db','old')),patch.object(m,'save',side_effect=AssertionError('write')),patch.object(m.subprocess,'Popen',side_effect=AssertionError('launch')),patch.object(m.os,'kill',side_effect=AssertionError('signal')),contextlib.redirect_stdout(io.StringIO()) as stream:
    m.reset(args)
   self.assertTrue(json.loads(stream.getvalue())['prepared_only'])
if __name__=='__main__':unittest.main()
