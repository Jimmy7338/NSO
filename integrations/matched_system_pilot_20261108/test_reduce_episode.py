import json,pathlib,tempfile,unittest
from reduce_episode import reduce_episode

class ReducerTests(unittest.TestCase):
 def test_startup_failure_has_no_invented_native_success(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td);(p/'result.json').write_text(json.dumps(dict(status='execution_interrupted',reason='prearm_fault')))
   r=reduce_episode(p);self.assertEqual(r['accounting']['accepted_observations'],0);self.assertIsNone(r['accounting']['native_paid_from_last_accepted_observe']);self.assertIsNone(r['return_audit']['recorded_returned_to_anchor']);self.assertIsNone(r['fixed_support_quality']['arithmetic_delta'])
 def test_failure_counts_negative_quality_and_same_state_rankings(self):
  with tempfile.TemporaryDirectory() as td:
   p=pathlib.Path(td);(p/'joint').mkdir();(p/'result.json').write_text(json.dumps(dict(status='execution_interrupted',reason='source_loss',arrival_evidence=[dict(verified=True)],joint_actions=[{},{}])));(p/'joint_configuration.json').write_text(json.dumps(dict(configuration=dict(return_anchor=[2,2,0]))))
   def observe(i,Q):
    return dict(request=dict(method='observe',params=dict(action_id=i,action='left' if i else None)),response=dict(ok=True,result=dict(accepted=True,duplicate=False,state=dict(pose=[2,3,0]),summary=dict(actual_paid_actions=i,runtime=dict(arrived_count=i,modules=dict(feedback=dict(fixed_old_support_count=10,fixed_old_support_sha256='same',initial_old_quality_proxy=.5,old_quality_proxy=Q,old_quality_proxy_total_change=Q-.5)))))))
   selection=dict(plan_number=1,mode='S',selected=dict(candidate_id=1),candidates=[dict(candidate_id=0,cost=3),dict(candidate_id=1,cost=2)],scores={m:[1,1] for m in ['C','G','S','X']},score_audit=[dict(candidate_id=1,incremental_aperture=[.2],prior_potentials=dict(G=1,S=.5,X=1.5))],candidate_audit={})
   rows=[observe(0,.5),dict(request=dict(method='next_action'),response=dict(ok=True,result=dict(selection=selection))),observe(2,.4)]
   (p/'joint/decisions.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n');r=reduce_episode(p)
   self.assertEqual(r['accounting']['native_paid_from_last_accepted_observe'],2);self.assertEqual(r['accounting']['native_arrived_count'],2);self.assertEqual(r['accounting']['ROS_verified_arrival_records'],1);self.assertAlmostEqual(r['fixed_support_quality']['arithmetic_delta'],-.1);self.assertFalse(r['return_audit']['exact_discrete_pose_equals_anchor']);self.assertEqual(r['global_states'][0]['shadow_modes']['G']['rank'],[1,0]);self.assertEqual(len(r['global_states'][0]['active_aperture_candidates']),1)
 def test_missing_terminal_rejected_without_polling(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(ValueError):reduce_episode(td)

if __name__=='__main__':unittest.main()
