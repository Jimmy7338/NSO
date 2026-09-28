"""Handwritten observed maps/packets only; zero world, sensors or TSDF."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

from nso.persistent_observation_option_v29 import PersistentObservationOptionV29
from nso.observed_runtime_v29 import ObservedANSRuntimeV29
from utils.grid_geometry import DIRECTIONS
from test_observed_planner_v26 import state_for, cue_for
from test_observed_runtime_v26 import NonFusingMapper, SHAPE, START, config, packet


def target(pose=(26,21,0), semantic=True):
    return dict(pose=list(pose),group='unit_observed_target',semantic_evidence=[
        dict(cue_id=cue_for().cue_id,hypothesis_gain=.05,sector=2)] if semantic else [])


def moved(state, action, **changes):
    r,c=state.position;h=state.heading
    if action == 'forward':
        dr,dc=DIRECTIONS[h];r,c=r+dr,c+dc
    else:
        h=(h+(1 if action=='right' else -1))%4
    values=dict(position=(r,c),heading=h,action_id=state.action_id+1,
        remaining_budget=state.remaining_budget-1,geometry_sha256=f'unit-{state.action_id+1}')
    values.update(changes)
    return replace(state,**values)


class PersistentOptionTests(unittest.TestCase):
    def initialized(self, budget=70):
        option=PersistentObservationOptionV29();state=state_for(budget=budget)
        option.initialize(state,state)
        return option,state

    def consume_move(self, option, state, **changes):
        action=option.next_action()
        self.assertIsNotNone(action)
        after=moved(state,action,**changes)
        option.consume(after,after,action=action)
        return after

    def test_whole_option_keeps_early_improvement_and_identity_beyond_five_steps(self):
        option,state=self.initialized()
        chosen=target();option.open(chosen,(cue_for(),))
        opening=option.active
        chosen['pose'][0]=8;chosen['semantic_evidence'][0]['hypothesis_gain']=999
        mutated=option.active;mutated['target_pose'][0]=9
        for i in range(7):
            action=option.next_action();self.assertEqual(action,'forward')
            patches=tuple(replace(p,bits=5,best_range=2.) for p in state.patches)
            after=moved(state,action,patches=patches)
            option.consume(after,after,action=action);state=after
            if i<6:
                self.assertEqual(option.active,opening)
        self.assertIsNone(option.active)
        self.assertEqual(len(option.closed),1)
        closed=option.closed[0]
        self.assertEqual(closed['actual_paid_actions'],7)
        self.assertEqual(closed['opened_at_action'],0)
        self.assertEqual(closed['closed_at_action'],7)
        self.assertEqual(closed['outcome_features'][0]['improved_common_patches'],6)
        self.assertEqual(closed['route_refreshes'],7)
        self.assertEqual(closed['distinct_observed_base_poses_including_start'],8)
        self.assertAlmostEqual(closed['max_translation_baseline_from_start_m'],1.4)
        self.assertEqual(closed['frozen_prediction'][0]['hypothesis_gain'],.05)
        self.assertIsNone(closed['learning_reward'])

    def test_complete_route_budget_includes_paid_return_turns(self):
        for budget,valid in ((18,True),(17,False)):
            option,state=self.initialized(budget)
            if valid:
                option.open(target(),(cue_for(),))
                self.assertEqual(option.active['initial_outbound_cost'],7)
                self.assertEqual(option.active['initial_return_reserve'],11)
            else:
                with self.assertRaisesRegex(ValueError,'complete observation'):
                    option.open(target(),(cue_for(),))

    def test_multi_waypoint_task_closes_only_after_last_heading(self):
        option,state=self.initialized()
        option.open(target((29,21,1)),(cue_for(),),waypoints=((31,21,0),(29,21,1)))
        while option.active:
            old=state;state=self.consume_move(option,state)
            if state.position==(31,21):
                self.assertIsNotNone(option.active)
            if state.position==(29,21) and state.heading==0:
                self.assertIsNotNone(option.active)
            self.assertLess(state.action_id,10)
        self.assertEqual(option.closed[0]['completed_waypoints'],2)
        self.assertEqual(option.closed[0]['actual_paid_actions'],5)

    def test_zero_action_option_and_pending_mutation_rejected(self):
        option,state=self.initialized()
        with self.assertRaises(ValueError):
            option.open(target((*state.position,state.heading)),(cue_for(),))
        option.open(target(),(cue_for(),))
        option.next_action()
        for operation in (lambda:option.next_action(),lambda:option.open(target(),(cue_for(),)),
                          lambda:option.begin_return()):
            with self.assertRaises(ValueError):operation()
        self.assertEqual(option.paid_ledger,[])

    def test_observed_and_intervened_classes_are_frozen_separately(self):
        for planning_class in (2,None):
            option,state=self.initialized()
            selected=target();selected['semantic_evidence'][0]['class_id']=planning_class
            option.open(selected,(cue_for(3),))
            selected['semantic_evidence'][0]['class_id']=3
            frozen=option.active['frozen_prediction'][0]
            self.assertEqual(frozen['observed_class_id'],3)
            self.assertEqual(frozen['planning_class_id'],planning_class)
            self.assertEqual(frozen['outward'],list(cue_for().outward))

    def test_changed_map_can_detour_without_cancelling_target(self):
        option,state=self.initialized();option.open(target(),(cue_for(),))
        action=option.next_action();belief=state.belief.copy();belief[29,21]=1
        after=moved(state,action,belief=belief);option.consume(after,after,action=action)
        self.assertIsNotNone(option.next_action())
        self.assertEqual(option.active['option_id'],1)
        self.assertEqual(option.closed,[])

    def test_unreachable_target_cancels_once_and_accounts_return_separately(self):
        option,state=self.initialized();option.open(target(),(cue_for(),))
        action=option.next_action();belief=state.belief.copy();belief[26,21]=-1
        state=moved(state,action,belief=belief);option.consume(state,state,action=action)
        while True:
            action=option.next_action()
            if action is None:break
            state=moved(state,action);option.consume(state,state,action=action)
            self.assertLess(state.action_id,10)
        self.assertEqual(len(option.closed),1)
        self.assertEqual(option.closed[0]['status'],'cancelled')
        self.assertEqual(option.closed[0]['actual_paid_actions'],1)
        self.assertEqual(len(option.paid_ledger),6)
        self.assertEqual(sum(r['option_id'] is None for r in option.paid_ledger),5)
        self.assertTrue(option.return_only)
        self.assertEqual((*state.position,state.heading),state.anchor)
        with self.assertRaises(ValueError):option.open(target(),(cue_for(),))

    def test_no_observed_return_never_clears_map_to_escape(self):
        option,state=self.initialized();option.open(target(),(cue_for(),))
        action=option.next_action();belief=state.belief.copy();belief[state.anchor[:2]]=-1
        state=moved(state,action,belief=belief);option.consume(state,state,action=action)
        self.assertIsNone(option.next_action())
        self.assertEqual(option.closed[0]['status'],'cancelled')
        self.assertEqual(state.belief[state.anchor[:2]],-1)

    def test_collision_and_sensor_stop_are_paid_without_reward(self):
        for collision in (True,False):
            option,state=self.initialized();option.open(target(),(cue_for(),))
            action=option.next_action()
            after=moved(state,action,**({'position':state.position} if collision else {}))
            option.consume(after,after,action=action,collision=collision,
                           stop_reason=None if collision else 'sensor_end')
            self.assertEqual(option.closed[0]['actual_paid_actions'],1)
            self.assertEqual(option.closed[0]['status'],'cancelled')
            self.assertIsNone(option.closed[0]['learning_reward'])
            self.assertIsNone(option.next_action())
            with self.assertRaises(ValueError):option.consume(after,after,action=action)

    def test_budget_end_after_exact_paid_return_is_terminal_and_counted(self):
        option,state=self.initialized(6);option.open(target((32,21,0)),(cue_for(),))
        state=self.consume_move(option,state)
        option.begin_return()
        for _ in range(5):state=self.consume_move(option,state)
        self.assertEqual(state.remaining_budget,0)
        self.assertEqual((*state.position,state.heading),state.anchor)
        self.assertEqual(len(option.paid_ledger),6)
        self.assertEqual(len(option.closed),1)
        self.assertEqual(option.closed[0]['status'],'completed')
        self.assertIsNone(option.next_action())

    def test_arrival_on_terminal_frame_completes_observation_but_not_mission(self):
        option,state=self.initialized();option.open(target((32,21,0)),(cue_for(),))
        action=option.next_action();state=moved(state,action)
        option.consume(state,state,action=action,stop_reason='sensor_end')
        self.assertEqual(option.closed[0]['status'],'completed')
        self.assertNotEqual((*state.position,state.heading),state.anchor)
        self.assertIsNone(option.next_action())

    def test_exact_budget_multi_waypoint_arrival_at_anchor_is_completed(self):
        option,state=self.initialized(6)
        option.open(target(state.anchor),(cue_for(),),waypoints=((32,21,0),state.anchor))
        for _ in range(6):state=self.consume_move(option,state)
        self.assertEqual(state.remaining_budget,0)
        self.assertEqual((*state.position,state.heading),state.anchor)
        self.assertEqual(option.closed[0]['status'],'completed')
        self.assertEqual(option.closed[0]['actual_paid_actions'],6)
        self.assertIsNone(option.next_action())

    def test_new_keys_are_not_quality_and_changed_contract_or_future_cue_is_rejected(self):
        option,state=self.initialized()
        with self.assertRaises(ValueError):option.open(target(),(cue_for(action=1),))
        option.open(target((32,21,0)),(cue_for(),))
        action=option.next_action()
        bad=moved(state,action,resolution_m=.4)
        with self.assertRaises(ValueError):option.consume(bad,bad,action=action)
        self.assertEqual(option.paid_ledger,[])
        state=moved(state,action,patches=tuple(replace(p,key=(99+i,1,1)) for i,p in enumerate(state.patches)))
        option.consume(state,state,action=action)
        outcome=option.closed[0]['outcome_features'][0]
        self.assertEqual(outcome['comparable_patches'],0)
        self.assertEqual(outcome['improved_common_patches'],0)
        self.assertEqual(outcome['new_keys'],6)
        self.assertIsNone(option.closed[0]['learning_reward'])


class RuntimeAdapterTests(unittest.TestCase):
    def runtime(self,mode='G'):
        with patch('nso.observed_runtime_v29.ObservedRuntimeMapperV10',NonFusingMapper):
            return ObservedANSRuntimeV29(SHAPE,config(),40,mode)

    def test_periodic_check_preserves_task_and_last_frame_before_new_task(self):
        runtime=self.runtime();runtime.accept(packet())
        selected=target((8,15,0),semantic=False)
        with patch.object(runtime,'_plan',return_value=dict(selected=selected,candidates=[selected])) as choose:
            for i in range(1,8):
                self.assertEqual(runtime.next_action(),'forward')
                runtime.accept(packet(i,position=(15-i,15),action='forward'))
            self.assertEqual(choose.call_count,1)
        closed=runtime.options.closed[0]
        self.assertEqual(closed['actual_paid_actions'],7)
        self.assertEqual(closed['closed_at_action'],7)
        self.assertTrue(any(r['operation']=='preserve_option_at_periodic_check' for r in runtime.calls))
        selected=target((7,15,0),semantic=False)
        with patch.object(runtime,'_plan',return_value=dict(selected=selected,candidates=[selected])):
            runtime.next_action()
        self.assertEqual(runtime.options.active['opened_at_action'],7)
        self.assertEqual(runtime.options.paid_ledger[-1]['option_id'],1)
        self.assertEqual({r['module'] for r in runtime.calls},{'OV-SDF','STGHP','RPN-UQ','IGCR'})

    def test_bad_packets_do_not_consume_or_fuse_pending_instruction(self):
        for bad in (packet(1,position=(14,15),action='forward',frame_id='unit-frame-0'),
                    packet(1,position=START,action='forward'),
                    replace(packet(1,position=(14,15),action='forward'),episode_id='other'),
                    packet(2,position=(14,15),action='forward')):
            runtime=self.runtime();runtime.accept(packet())
            selected=target((8,15,0),semantic=False)
            with patch.object(runtime,'_plan',return_value=dict(selected=selected,candidates=[selected])):
                runtime.next_action()
            with self.assertRaises(ValueError):runtime.accept(bad)
            self.assertEqual(runtime.mapper.frames,1)
            self.assertEqual(runtime.pending,'forward')
            self.assertEqual(runtime.options.paid_ledger,[])
            with self.assertRaises(ValueError):runtime.next_action()

    def test_actual_planner_modes_can_open_observed_safe_option_without_world(self):
        for mode in ('G','O','S','X'):
            runtime=self.runtime(mode);runtime.accept(packet(marker=True))
            action=runtime.next_action()
            self.assertIn(action,('forward','left','right'))
            self.assertIsNotNone(runtime.options.active)
            if mode=='G':self.assertEqual(runtime.cues,())
            self.assertFalse(runtime.summary()['full_method_efficacy_proven'])


if __name__=='__main__':
    unittest.main()
