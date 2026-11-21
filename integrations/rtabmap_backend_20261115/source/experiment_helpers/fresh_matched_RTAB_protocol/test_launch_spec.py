import importlib.util,json,pathlib,unittest,yaml
P=pathlib.Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('spec_builder',P/'build_launch_spec.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class SpecTests(unittest.TestCase):
 def setUp(self):self.s=m.build('offline_spec_test_ONLY',1794740000.,environment_from=P/'example_launch_spec.json')
 def test_preparation_does_not_create_database_session(self):
  self.assertFalse(pathlib.Path(self.s['new_database_path']).parent.exists());self.assertFalse(self.s['actuator_launched'])
 def test_primary_readers_precede_fresh_source(self):
  c={r['name']:r for r in self.s['launch_commands']}
  for n in ['mapping_backend','tf_hold','assembler','unified_viewer']:self.assertLess(c[n]['phase'],c['rtabmap']['phase'])
  self.assertGreater(c['history_relay']['phase'],c['rtabmap']['phase'])
 def test_no_bootstrap_and_no_native_or_mission_launch(self):
  backend=next(r for r in self.s['launch_commands'] if r['name']=='mapping_backend')
  for key in ['_initial_metadata_path','_initial_epoch_graph_path','_initial_epoch_graph_manifest_path']:
   value=next(a.split(':=',1)[1] for a in backend['argv'] if a.startswith(key+':='));self.assertEqual(yaml.safe_load(value),'')
  for r in self.s['launch_commands']:
   self.assertFalse(any('mission.py' in a or 'nso_scout_attended_node' in a for a in r['argv']))
 def test_epoch_paths_and_sources_bound(self):
  self.assertIn(self.s['epoch_id'],pathlib.Path(self.s['new_database_path']).parts)
  for p in self.s['source_bindings']:self.assertTrue(pathlib.Path(p).is_file(),p)
  self.assertEqual(self.s['first_data_contract']['expected_epoch'],self.s['epoch_id'])
 def test_same_guards_and_no_runtime_limits_in_pair_protocol(self):
  p=json.loads((P/'paired_protocol_template.json').read_text())
  self.assertIsNone(p['runtime_overall_time_limit']);self.assertFalse(p['automatic_return']);self.assertFalse(p['bit_identical_initial_information_claim']);self.assertFalse(p['G_S_gain_established'])
 def test_saved_environment_works_without_reading_dead_process(self):
  self.assertEqual(self.s['navigation_environment_source']['kind'],'reviewed_saved_launch_spec')
  self.assertIn('/teb-transformwait-20261115/devel/lib',self.s['environment']['LD_LIBRARY_PATH'])
  self.assertEqual(self.s['new_trial_configuration_bindings']['rtabmap_graph_snapshot_path'],self.s['new_snapshot_path'])
if __name__=='__main__':unittest.main()
