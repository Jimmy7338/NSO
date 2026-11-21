import json,tempfile,unittest
from pathlib import Path
from ros_viewpoint_coordinator import final_viewpoint_observation_audit,PROFILE
class AuditTests(unittest.TestCase):
    def fixture(self,root,credit=False):
        (root/'joint').mkdir();(root/'joint_inputs').mkdir();rows=[dict(request=dict(id=1,method='init',params={'configuration':dict(policy_variant=PROFILE,scene_id='s',episode_id='e')}),response=dict(id=1,ok=True,result={}))]
        for aid,status in [(0,None),(1,4)]:
            packet=root/'joint_inputs'/('joint-f'+str(aid)+'.npz');packet.touch()
            summary=dict(accounting_unit='actual_waypoint_attempt',waypoint_attempts=aid,waypoint_ROS_successes=0,primitive_arrivals_credited=0,runtime=dict(scene_id='s',episode_id='e'))
            state=dict(action_id=aid,step=aid,pose=[2,3,1],frame_id='f'+str(aid))
            outcome=dict(attempt_id=aid,ros_status=status,waypoint_attempt_charged=aid>0,primitive_arrival_credit=credit)
            rows.append(dict(request=dict(id=aid+2,method='observe_viewpoint_result',params=dict(path=str(packet),attempt_id=aid,ros_status=status)),response=dict(id=aid+2,ok=True,result=dict(accepted=True,duplicate=False,state=state,summary=summary,outcome=outcome))))
        (root/'joint/decisions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));return rows[-1]['response']['result']['summary']
    def test_bootstrap_failed_actual_result_counted_without_primitive_credit(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);summary=self.fixture(root);stale=dict(summary,waypoint_attempts=0)
            x=final_viewpoint_observation_audit(root,stale,worker_closed=True,stop_confirmed=True)
            r=x['final_worker_observation_receipt'];self.assertEqual(r['status'],'verified')
            self.assertEqual((r['accepted_bootstrap_observations'],r['accepted_waypoint_result_observations']),(1,1))
            self.assertEqual(r['worker_waypoint_attempts'],1);self.assertEqual(r['primitive_arrivals_credited'],0)
            self.assertIsNone(x['supervisor_worker_count_difference'])
            self.assertEqual(x['supervisor_worker_viewpoint_count_difference']['worker_minus_supervisor'],1)
            self.assertEqual(x['final_worker_summary'],summary)
    def test_reject_fake_primitive_credit_and_require_closed_stopped(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);summary=self.fixture(root,credit=True)
            x=final_viewpoint_observation_audit(root,summary,worker_closed=True,stop_confirmed=True)
            self.assertEqual(x['final_worker_observation_receipt']['status'],'audit_error');self.assertIsNone(x['final_worker_summary'])
            x=final_viewpoint_observation_audit(root,summary,worker_closed=False,stop_confirmed=True)
            self.assertEqual(x['final_worker_observation_receipt']['status'],'audit_error')
if __name__=='__main__':unittest.main()
