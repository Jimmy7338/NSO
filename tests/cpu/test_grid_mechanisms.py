"""Mechanism isolation, observation privacy, topology and label semantics."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from env.grid_exploration import GridConfig, GridExplorationEnv
from env.grid_layouts import generate_layout
from env.grid_semantics import synthetic_semantics
from nso.grid_topology import ObservedTopology
from nso.grid_mechanisms import MechanismPolicy, action_cost
from nso.cpu_reachability import CandidateRPN, FEATURES
from scripts.eval_cpu import run_episode


class MechanismTests(unittest.TestCase):
    def test_door_cuts_four_rooms_stable_ids_and_reset(self):
        world=generate_layout('multi_room',64,31,5).astype(np.int8)
        g=ObservedTopology().update(world)
        self.assertEqual(len(g.nodes),4)
        self.assertEqual(len(g.edges),4)
        previous=g.labels.copy();g.update(world)
        np.testing.assert_array_equal(previous,g.labels)
        self.assertEqual(len(set(g.nodes)),4)
        g.reset();self.assertEqual(g.updates,0);self.assertEqual(g.nodes,{})

    def test_unknown_boundary_is_not_a_door_wall(self):
        belief=np.full((30,30),-1,np.int8);belief[10:15,2:28]=0
        graph=ObservedTopology().update(belief)
        self.assertEqual(len(graph.nodes),1)
        self.assertFalse(graph.edges)
        self.assertTrue(np.all(graph.labels[belief == -1] == 0))

    def test_semantic_sensor_hidden_changes_do_not_affect_decision(self):
        world=generate_layout('multi_room',64,31,5)
        field=synthetic_semantics(world.shape,1)
        a=GridExplorationEnv(world,semantic_field=field);obs=a.reset(seed=1)
        self.assertTrue(np.isnan(obs.semantic[~obs.visible]).all())
        self.assertFalse(obs.semantic.flags.writeable)
        changed=field.copy();changed[~obs.visible]=1-changed[~obs.visible]
        b=GridExplorationEnv(world,semantic_field=changed);other=b.reset(seed=1)
        p=MechanismPolicy(a.config,'gain_semantic_structure')
        q=MechanismPolicy(a.config,'gain_semantic_structure')
        self.assertEqual(p.act(obs),q.act(other))
        self.assertEqual(p.selection_audit,q.selection_audit)
        self.assertTrue(np.all(p.semantic[obs.belief == -1] == 0))
        p.reset();self.assertEqual(p.diagnostics()['semantic_updates'],0)
        self.assertIsNone(p.semantic);self.assertFalse(p.attempts)

    def test_zero_weights_match_control_and_calls_respect_switches(self):
        world=generate_layout('multi_room',64,2,5)
        env=GridExplorationEnv(world,semantic_field=synthetic_semantics(world.shape,4))
        obs=env.reset(seed=1)
        control=MechanismPolicy(env.config,'gain_control')
        both=MechanismPolicy(env.config,'gain_semantic_structure',options={'semantic_weight':0,'structure_weight':0})
        for _ in range(45):
            a,b=control.act(obs),both.act(obs)
            self.assertEqual(a,b)
            obs,done=env.step(a.action)
            control.observe_outcome(obs,done);both.observe_outcome(obs,done)
            if done:break
        self.assertEqual(control.diagnostics()['topo_updates'],0)
        self.assertEqual(control.diagnostics()['semantic_updates'],0)
        self.assertGreater(both.diagnostics()['topo_updates'],0)
        self.assertGreater(both.diagnostics()['semantic_updates'],0)

    def test_scoring_terms_change_ranking_when_enabled(self):
        # Controlled equal-geometry candidates: a visible semantic hotspot breaks tie.
        world=np.zeros((31,31),bool);world[[0,-1]]=1;world[:,[0,-1]]=1
        env=GridExplorationEnv(world,semantic_field=np.ones(world.shape))
        obs=env.reset(start=(15,15),heading=0)
        p=MechanismPolicy(env.config,'gain_semantic');p.act(obs)
        rows=p.selection_audit
        self.assertTrue(any(r['semantic_term']>0 and r['score']>r['gain_per_cost'] for r in rows))
        self.assertTrue(all(r['structure_term']==0 for r in rows))

    def test_censored_targets_are_not_failures_and_budget_is_failure(self):
        env=GridExplorationEnv(generate_layout('multi_room',64,0,5))
        obs=env.reset(seed=1)
        p=MechanismPolicy(env.config,'gain_control',options={'goal_budget':2})
        active=dict(start_step=0,goal=[2,2],goal_heading=0,features=[0]*5,geometric_connected=True,budget=2)
        p._active=active.copy();obs,_=env.step('right');p.observe_outcome(obs,True)
        self.assertIsNone(p.attempts[-1]['controller_success'])
        p._active=active.copy();obs,_=env.step('right');p.observe_outcome(obs)
        self.assertEqual(p.attempts[-1]['controller_success'],0)
        self.assertEqual(p.attempts[-1]['elapsed_actions'],2)

    def test_all_masked_candidates_bounded_fallback(self):
        env=GridExplorationEnv(generate_layout('single_room',64,0,5))
        obs=env.reset(seed=1);p=MechanismPolicy(env.config,'gain_control')
        p._blocked_goals=set(map(tuple,np.argwhere(np.ones(obs.belief.shape,bool))))
        actions=[p.act(obs).action for _ in range(6)]
        self.assertEqual(actions[-1],'stop')
        self.assertEqual(p.goal_changes,0)

    def test_action_budget_includes_turns_and_terminal_heading(self):
        self.assertEqual(action_cost([(2,2),(2,3),(3,3)],0,3),5)

    def test_noisy_semantics_terminate_with_finite_budget(self):
        config=GridConfig(max_steps=60)
        world=generate_layout('dead_end',64,3,5)
        env=GridExplorationEnv(world,config,np.random.default_rng(12).random(world.shape))
        obs=env.reset(seed=1);policy=MechanismPolicy(config,'gain_semantic_structure')
        for _ in range(config.max_steps):
            decision=policy.act(obs)
            self.assertTrue(np.isfinite(decision.score))
            obs,done=env.step(decision.action);policy.observe_outcome(obs,done)
            if done:break
        self.assertTrue(done)
        self.assertIsNone(policy._active)

    def test_frozen_candidate_rpn(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.npz'
            np.savez(path,w1=np.zeros((5,16)),b1=np.zeros(16),w2=np.zeros((16,1)),b2=np.zeros(1),
                     features=np.array(FEATURES),goal_budget=32)
            model=CandidateRPN(path)
            np.testing.assert_array_equal(model.predict([[0]*5]),[.5])
            model.verify_frozen()
            with self.assertRaises(ValueError):model.weights['w1'][0,0]=1
            model.weights['w1']=np.ones((5,16))
            with self.assertRaises(RuntimeError):model.verify_frozen()

    def test_archived_attempts_counters_and_untried_labels(self):
        with tempfile.TemporaryDirectory() as d:
            world=generate_layout('multi_room',64,0,5)
            result=run_episode(world,GridConfig(max_steps=40),'gain_semantic_structure',101,
                               Path(d)/'episode',{},semantic_field=synthetic_semantics(world.shape,3))
            self.assertGreater(result['semantic_updates'],0)
            self.assertGreater(result['topo_updates'],0)
            self.assertEqual(result['rpn_calls'],0)
            attempts=[json.loads(s) for s in (Path(d)/'episode/goal_attempts.jsonl').read_text().splitlines()]
            candidates=[json.loads(s) for s in (Path(d)/'episode/candidates.jsonl').read_text().splitlines()]
            self.assertEqual(len(attempts),result['goal_changes'])
            self.assertTrue(all(r['controller_success'] is None for r in candidates))
            self.assertTrue(all(r['geometric_connected'] for r in candidates))
            self.assertTrue(all(r['elapsed_actions'] <=32 for r in attempts))


if __name__ == '__main__':unittest.main()
