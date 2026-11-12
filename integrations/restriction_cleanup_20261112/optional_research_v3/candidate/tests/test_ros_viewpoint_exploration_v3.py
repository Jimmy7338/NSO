"""Macro endpoint/resource/actual-pose contracts with synthetic acquired packets."""
import unittest
from nso_bridge.ros_viewpoint_exploration_v3 import ViewpointSession,ViewpointCPU,PROFILE
from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules
from test_joint_engine import JointEngineTests

class ViewpointTests(unittest.TestCase):
    def session(self,attempts=3):
        fixture=JointEngineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        cfg=dict(fixture.config,planner_revision='v10_3_1',policy_variant=PROFILE,total_budget=attempts,
            nominal_planning_horizon=10,max_viewpoint_attempts=attempts)
        fixture.engine=ViewpointSession(cfg,execution_allowed=False)
        fixture.engine.observe_viewpoint_result(str(fixture.observation()),0)
        return fixture,fixture.engine
    def test_full_selected_macro_endpoint_and_forecast_cost_are_not_a_primitive(self):
        f,s=self.session();d=s.next_viewpoint(remaining_time_s=600)
        self.assertEqual(d['target']['pose'],d['target']['selected']['pose'])
        self.assertGreater(d['target']['predicted_nominal_outbound_cost'],1)
        self.assertEqual(s.summary()['waypoint_attempts'],0)
        self.assertIn('no forward/left/right primitive fallback',s.identity()['receipt']['viewpoint_execution']['admission']['no_option_fallback'])
        self.assertEqual(d['selection']['candidate_audit']['remaining_budget'],10)
        self.assertIs(ViewpointCPU._scores_v10_1,ExplorationJointCPUFourModules._scores_v10_1)
        with self.assertRaisesRegex(ValueError,'no primitive'):s.next_action()
    def test_failed_macro_accepts_actual_partial_pose_once_without_primitive_gain(self):
        f,s=self.session();d=s.next_viewpoint()
        path=str(f.observation(1,base=[3.3,3.3,1.],timestamp=2.))
        result=s.observe_viewpoint_result(path,1,4,'ABORTED')
        self.assertTrue(result['accepted']);self.assertEqual(result['outcome']['ros_status'],4)
        self.assertFalse(result['outcome']['primitive_arrival_credit'])
        self.assertEqual(result['summary']['waypoint_attempts'],1)
        self.assertEqual(result['summary']['waypoint_ROS_successes'],0)
        self.assertEqual(s.mapper.frames,2)
        self.assertEqual(s.backend.scenes[0]['gain'].snapshot()['observed_actions'],0)
        self.assertFalse(s.backend.scenes[0]['gain']._pending)
        self.assertNotEqual(result['state']['pose'],d['target']['pose'])
        self.assertEqual(s.runtime.states[0]['arrived_count'],0)
        with self.assertRaisesRegex(ValueError,'pending'):s.observe_viewpoint_result(path,1,4)
        again=s.next_viewpoint()
        self.assertEqual(again['target']['attempt_id'],2)
        self.assertFalse(result['outcome']['candidate_exclusion_applied'])
        self.assertEqual(again['target']['selected'],s.backend.scenes[0]['selected'])
        self.assertEqual(again['selection']['candidate_audit']['remaining_budget'],10)
    def test_attempt_cap_and_cancelled_result_preserve_actual_resources(self):
        f,s=self.session(attempts=2)
        for aid,status in [(1,3),(2,2)]:
            s.next_viewpoint();r=s.observe_viewpoint_result(str(f.observation(aid,base=[3.3,3.3,1.] if aid==1 else None,timestamp=1.+aid)),aid,status,'actual terminal')
        self.assertEqual(s.summary()['waypoint_attempts'],2)
        self.assertEqual(s.summary()['waypoint_ROS_successes'],1)
        self.assertIsNone(s.next_viewpoint()['target'])
        self.assertEqual(s.mapper.frames,3)
    def test_explicit_best_effort_return_and_time_budget(self):
        f,s=self.session()
        s.next_viewpoint();s.observe_viewpoint_result(str(f.observation(1,base=[3.3,3.3,1.],timestamp=2.)),1,4)
        goal=s.next_viewpoint(return_requested=True)['target']
        self.assertEqual(goal['pose'],list(s.anchor));self.assertEqual(goal['goal_role'],'best_effort_return')
        self.assertIsNone(goal['predicted_nominal_grid_cost'])

    def test_partial_fusion_exception_poisoned_worker_cannot_be_reused(self):
        from unittest.mock import patch
        f,s=self.session();s.next_viewpoint()
        path=str(f.observation(1,timestamp=2.))
        update=s.mapper.update
        def partial_update(*args,**kwargs):
            update(*args,**kwargs)
            raise RuntimeError('injected_after_actual_map_fusion')
        with patch.object(s.mapper,'update',side_effect=partial_update):
            with self.assertRaisesRegex(RuntimeError,'injected_after_actual_map_fusion'):
                s.observe_viewpoint_result(path,1,4,'ABORTED')
        self.assertEqual(s.mapper.frames,2)
        self.assertTrue(s.runtime.states[0]['closed'])
        self.assertEqual(s.runtime.states[0]['termination']['reason'],'waypoint_fusion_or_feedback_exception')
        self.assertFalse(s.runtime.states[0]['termination']['rollback_performed'])
        self.assertIsNone(s.next_viewpoint()['target'])
        with self.assertRaisesRegex(ValueError,'worker closed'):
            s.observe_viewpoint_result(path,1,4,'retry')
        self.assertEqual(s.mapper.frames,2)
        self.assertEqual(s.summary()['waypoint_attempts'],0)
