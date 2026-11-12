"""Admission behavior, independent from ROS/hardware; no actuator connections."""
from types import SimpleNamespace
import unittest
import numpy as np
from nso_bridge.joint_engine import VENDOR
from env.virtual3d import VirtualConfig
from nso_bridge.exploration_joint_policy import generate_options
from nso_bridge.ros_checked_exploration_v2 import ROSCheckedGuard,PROFILE,identity

class ROSCheckedTests(unittest.TestCase):
    def test_unknown_admission_does_not_change_measurements(self):
        belief=np.full((9,9),-1,np.int8);old=belief.copy();g=ROSCheckedGuard(.2,.72)
        a=g.assess(belief,(4,4),0,'forward',3)
        self.assertTrue(a.allowed);np.testing.assert_array_equal(belief,old)
        self.assertEqual(int((belief==0).sum()),0)
    def test_coarse_occupied_start_allows_rotation_escape_not_occupied_successor(self):
        belief=np.full((9,9),-1,np.int8);belief[4,4]=1;g=ROSCheckedGuard(.2,.72);g.current=(4,4)
        self.assertTrue(g.assess(belief,(4,4),0,'left',3).allowed)
        self.assertTrue(g.assess(belief,(4,4),0,'forward',3).allowed)
        belief[3,4]=1
        self.assertFalse(g.assess(belief,(4,4),0,'forward',3).allowed)
        self.assertEqual(int((belief==1).sum()),2)
    def test_no_certificate_or_anchor_admission_required_for_options(self):
        belief=np.full((9,9),-1,np.int8);belief[4,4]=1;belief[1,1]=1
        m=SimpleNamespace(belief=belief,shape=belief.shape,camera_seen=np.zeros_like(belief,bool),keyframes=[],
            config=VirtualConfig(max_depth_m=.6,robot_radius_m=.72),quality_evidence=lambda **kw:None)
        old=belief.copy()
        strict,a=generate_options(m,(4,4),0,3,(1,1,0),coverage_strategy='total_diverse')
        self.assertEqual(strict,[]);self.assertEqual(a['status'],'current_footprint_not_known_safe')
        routes,a=generate_options(m,(4,4),0,3,(1,1,0),coverage_strategy='total_diverse',admission_profile=PROFILE)
        self.assertTrue(routes);self.assertFalse(a['known_free_return_certificate'])
        self.assertTrue(all(r['cost']<=3 and not r['return_plan_available_within_budget'] for r in routes))
        np.testing.assert_array_equal(belief,old)
    def test_return_best_effort_preserves_budget_and_obstacle(self):
        b=np.full((9,9),-1,np.int8);g=ROSCheckedGuard(.2,.72);g.current=(4,4)
        r=g.return_plan(b,(4,4),0,(1,1,0),2)
        self.assertTrue(r.available);self.assertLessEqual(len(r.actions),2);self.assertGreater(r.paid_cost,2)
        b[1,1]=1;self.assertFalse(g.return_plan(b,(4,4),0,(1,1,0),20).available)
    def test_identity_and_resource_budget(self):
        d=identity();self.assertTrue(d['research_policy_changed']);self.assertFalse(d['original_method_equivalence'])
        self.assertFalse(d['radius_changed']);g=ROSCheckedGuard(.2,.72)
        self.assertFalse(g.assess(np.zeros((3,3),np.int8),(1,1),0,'forward',0).allowed)
if __name__=='__main__':unittest.main()

class FailedAttemptContractTests(unittest.TestCase):
    def test_failed_partial_pose_charges_attempt_without_arrival_and_replans(self):
        from test_joint_engine import JointEngineTests
        from nso_bridge.joint_engine import JointSession
        fixture=JointEngineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        config=dict(fixture.config,planner_revision='v10_3_1',policy_variant=PROFILE,total_budget=8)
        fixture.engine=JointSession(config,execution_allowed=False);engine=fixture.engine
        engine.observe(str(fixture.observation()),0);decision=engine.next_action()
        self.assertIsNotNone(decision['action'])
        # Failed execution has the actual unchanged initial pose, not the goal.
        result=engine.observe_failed_action(str(fixture.observation(1,timestamp=2.)),1,decision['action'],'ABORTED:blocked')
        self.assertTrue(result['accepted']);self.assertFalse(result['failed_execution_attempt']['arrival_credited'])
        self.assertEqual(engine.summary()['actual_paid_actions'],1)
        self.assertEqual(engine.runtime.states[0]['arrived_count'],0)
        self.assertFalse(engine.runtime.states[0]['closed']);self.assertEqual(engine.runtime.states[0]['active_actions'],[])
        self.assertIsNotNone(engine.next_action()['action'])
    def test_no_scored_option_and_no_return_does_not_hard_stop(self):
        from test_joint_engine import JointEngineTests
        from unittest.mock import patch
        from nso_bridge.joint_engine import JointSession
        fixture=JointEngineTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.engine=JointSession(dict(fixture.config,planner_revision='v10_3_1',policy_variant=PROFILE,total_budget=3))
        engine=fixture.engine;engine.observe(str(fixture.observation()),0)
        with patch.object(engine.backend,'select_target',return_value=None):
            result=engine.next_action()
        self.assertIsNotNone(result['action']);self.assertFalse(engine.runtime.states[0]['closed'])
        self.assertTrue(any(e['event']=='ROS_best_effort_without_scored_option' for e in result['runtime_events']))
