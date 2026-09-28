"""Pure routing/contract tests; no ROS, World, sensing, mapping or scores."""
import math
from types import SimpleNamespace
import unittest

from scripts.run_tare_routes_v39 import TareWaypointPlannerV39


def graph():
    poses=tuple((x,0.,h*90) for x in (0.,1.) for h in range(4))
    edges=[]
    for i in range(8):
        base=i//4*4; h=i%4
        row=[('right',base+(h+1)%4),('left',base+(h-1)%4)]
        if i==0: row.append(('forward',4))
        if i==6: row.append(('forward',2))
        edges.append(tuple(row))
    return SimpleNamespace(poses=poses,edges=tuple(edges),anchor=0)


class AdapterTests(unittest.TestCase):
    def planner(self):
        model=graph(); p=TareWaypointPlannerV39((model,model),(3.5,.5,0.))
        p.step=18
        p.response=dict(unique_paid_observations=19,published_registered_scans=19,expected_native_planning_due=False,
            exploration_finished=False,fresh_waypoint=False)
        return p

    def test_world_xy_projection_ignores_native_z(self):
        p=self.planner(); p.last_waypoint=dict(xyz=[4.5,.5,0.],frame_id='map',output_counter=1)
        r=p.select(0,24,None,.999)
        self.assertEqual(r['action'],'forward'); self.assertEqual(r['native_goal_node'],4)
        self.assertFalse(r['posterior_used_to_choose_action']); self.assertFalse(r['projection']['z_used_to_plan'])

    def test_far_goal_is_rejected_without_substitute(self):
        p=self.planner(); p.last_waypoint=dict(xyz=[100.,.5,.9],frame_id='map',output_counter=1)
        r=p.select(4,24,None,.01)
        self.assertIn('projection_rejected',r['decision_reason']); self.assertEqual(r['native_goal_node'],0)

    def test_real_scan_wait_bounded_by_next_native_due(self):
        p=self.planner()
        r=p.select(0,24,None,.5)
        self.assertEqual(r['action'],'right'); self.assertEqual(r['adapter_wait_turn'],1)
        p.step=19; p.response.update(unique_paid_observations=20,published_registered_scans=20,expected_native_planning_due=True)
        r=p.select(1,23,None,.5)
        self.assertEqual(r['action'],'left'); self.assertIn('cadence_exhausted',r['decision_reason'])

    def test_never_five_wait_turns_after_due(self):
        p=self.planner(); p.step=19
        p.response.update(unique_paid_observations=20,published_registered_scans=20,expected_native_planning_due=True)
        self.assertIsNone(p.select(0,24,None,.5)['action']); self.assertEqual(p.wait_turns,0)

    def test_wait_preserves_exact_return_budget(self):
        p=self.planner()
        self.assertIsNone(p.select(0,1,None,.5)['action'])

    def test_empty_paid_observation_does_not_advance_native_cadence(self):
        p=self.planner(); p.response.update(unique_paid_observations=20,published_registered_scans=19)
        self.assertEqual(p.select(0,24,None,.5)['scans_until_next_native_keypose'],1)

    def test_finished_means_return_without_wait(self):
        p=self.planner(); p.response['exploration_finished']=True
        self.assertIsNone(p.select(0,24,None,.5)['action']); self.assertEqual(p.wait_turns,0)

    def test_goal_requires_return_affordability(self):
        p=self.planner(); p.last_waypoint=dict(xyz=[4.5,.5,.9])
        r=p.select(0,1,None,.5)
        self.assertEqual(r['decision_reason'],'native_goal_not_return_affordable'); self.assertIsNone(r['action'])

    def test_invalid_new_native_goal_clears_old(self):
        p=self.planner(); p.last_waypoint=dict(xyz=[4.5,.5,.9]); p.last_output_counter=1
        reply=dict(status='observed',action_id=19,unique_paid_observations=20,
            waypoint=dict(xyz=[4.5,.5,.9],frame_id='odom',output_counter=2))
        p.client=SimpleNamespace(request=lambda message:reply)
        p.observe(SimpleNamespace(step=19),'saved.npz')
        self.assertIsNone(p.last_waypoint); self.assertEqual(len(p.rejected_waypoints),1)

    def test_no_new_waypoint_keeps_old_valid_native_target(self):
        p=self.planner(); p.last_waypoint=dict(xyz=[4.5,.5,.9]); p.last_output_counter=1
        reply=dict(status='observed',action_id=19,unique_paid_observations=20,waypoint=None)
        p.client=SimpleNamespace(request=lambda message:reply)
        p.observe(SimpleNamespace(step=19),'saved.npz')
        self.assertEqual(p.last_waypoint['xyz'],[4.5,.5,.9])

    def test_replay_reexecutes_exact_saved_action_index(self):
        model=graph(); p=TareWaypointPlannerV39((model,model),(0.,0.),recorded_actions=['right'])
        p.observe(SimpleNamespace(step=0))
        self.assertEqual(p.select(0,42,None,.5)['action'],'right')
        p.observe(SimpleNamespace(step=1))
        self.assertIsNone(p.select(1,41,None,.5)['action'])


if __name__=='__main__': unittest.main()
