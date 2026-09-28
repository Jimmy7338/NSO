"""Protocol integrity and deterministic rule comparator regressions."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from env.grid_layouts import generate_layout
from env.grid_exploration import GridExplorationEnv
from nso.grid_mechanisms import MechanismPolicy
from utils.cpu_protocol import geometry_hash,digest_json,file_hash,verify_protocol,runtime_versions


class ProtocolTests(unittest.TestCase):
    def test_rotations_reflections_canonicalize(self):
        world=generate_layout('multi_room',64,15,5)
        for k in range(4):
            self.assertEqual(geometry_hash(world),geometry_hash(np.rot90(world,k)))
            self.assertEqual(geometry_hash(world),geometry_hash(np.rot90(world.T,k)))
        changed=world.copy();changed[10,10]=1-changed[10,10]
        self.assertNotEqual(geometry_hash(world),geometry_hash(changed))

    def test_modified_config_code_runtime_or_leaked_map_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'policy.py';source.write_text('frozen')
            config=dict(frozen_protocol='protocol.json',maps=[dict(layout='multi_room',seed=15)],map_size=64,door_width=5)
            value=geometry_hash(generate_layout('multi_room',64,15,5))
            manifest=dict(config_sha256=digest_json(config),runtime=runtime_versions(),
                          files_sha256={'policy.py':file_hash(source)},maps={'multi_room_seed15':value},excluded_geometry_hashes=[])
            path=root/'protocol.json';path.write_text(json.dumps(manifest));verify_protocol(config,root)
            with self.assertRaises(ValueError):verify_protocol(dict(config,door_width=7),root)
            source.write_text('changed')
            with self.assertRaises(ValueError):verify_protocol(config,root)
            source.write_text('frozen');manifest['excluded_geometry_hashes']=[value];path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_protocol(config,root)
            manifest['excluded_geometry_hashes']=[];manifest['runtime']={};path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_protocol(config,root)

    def test_hard_budget_rule_excludes_only_over_budget_or_failed(self):
        env=GridExplorationEnv(generate_layout('multi_room',64,0,5));obs=env.reset(seed=1)
        policy=MechanismPolicy(env.config,'gain_budget_rule',options={'goal_budget':1})
        policy.act(obs)
        self.assertTrue(policy.selection_audit)
        for row in policy.selection_audit:
            self.assertEqual(row['excluded'],row['estimated_actions']>1)
        self.assertEqual(policy.rpn_calls,0)

    def test_soft_budget_rule_keeps_candidates_and_matches_formula(self):
        env=GridExplorationEnv(generate_layout('multi_room',64,0,5));obs=env.reset(seed=1)
        policy=MechanismPolicy(env.config,'gain_semantic_structure_budget_soft',options={'goal_budget':1})
        policy.act(obs)
        for row in policy.selection_audit:
            self.assertFalse(row['excluded'])
            expected=1/(1+np.exp(np.clip((row['estimated_actions']-1)/4,-40,40)))
            self.assertAlmostEqual(row['reach_probability'],expected)
            self.assertGreater(row['reach_probability'],0)
        self.assertEqual(policy.rpn_calls,0)

if __name__=='__main__':unittest.main()
