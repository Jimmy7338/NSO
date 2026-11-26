import sys,types,unittest
from pathlib import Path
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');O=Path(__file__).parent
sys.path.insert(0,str(W/'runtime'))
m=types.ModuleType('mission_primitive_candidate');m.__file__=str(W/'runtime/mission.py');exec(compile((O/'candidate/mission.py').read_text(),m.__file__,'exec'),m.__dict__)
class Tests(unittest.TestCase):
 def test_worker_primitive_arrivals_not_ROS_metric_count(self):
  s=m.result_summary(dict(primitive_feedback_profile='primitive_feedback_unbounded_v1',primitive_verified_arrivals=1,successful_arrivals=[{},{}],primitive_feedback_attempts=[{},{}],macro_verified_arrivals=0),'/software')
  self.assertEqual(s['verified_arrivals'],1);self.assertEqual(s['ROS_metric_arrivals'],2);self.assertEqual(s['macro_verified_arrivals'],0)
 def test_count_cannot_silently_apply_to_old_profile(self):
  for profile in ['ros_checked_exploration_v2','ros_viewpoint_exploration_v4_unbounded',None]:
   with self.assertRaises(ValueError):m.validate_primitive_pilot_harness_args(profile,3)
  m.validate_primitive_pilot_harness_args('primitive_feedback_unbounded_v1',3)
  m.validate_primitive_pilot_harness_args('ros_viewpoint_exploration_v4_unbounded',None)
 def test_explicit_controller_keeps_unbounded_profile(self):
  settings=dict(configuration=dict(policy_variant='ros_viewpoint_exploration_v4_unbounded',total_budget=None))
  value,receipt=m.select_execution_controller(settings,'primitive_feedback')
  self.assertEqual(value['configuration']['policy_variant'],'primitive_feedback_unbounded_v1');self.assertIsNone(value['configuration']['total_budget']);self.assertEqual(settings['configuration']['policy_variant'],'ros_viewpoint_exploration_v4_unbounded')
if __name__=='__main__':unittest.main()
