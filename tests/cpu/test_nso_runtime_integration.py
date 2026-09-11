"""ANS adapter contracts using real NSO components, without Habitat or downloads."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from nso.components import NSO_Components
from nso.runtime_integration import NSORuntimeIntegration, infer_uq_channels
from nso.topo_graph import RoomNode
from nso.reachability_uq import ReachabilityHeadUQ


def arguments(**overrides):
    values=dict(eval=True,train_global=False,use_semantic=False,
        use_open_vocab_semantic=False,use_topo_graph=False,use_rpn_uq=False,
        use_igcr=False,paper_rewards=0,map_resolution=100,robot_radius_m=0.,
        structural_reward_coeff=0.,semantic_reward_coeff=0.,frontier_reward_coeff=0.,
        rpn_mc_samples=2)
    values.update(overrides)
    return SimpleNamespace(**values)


def make(args, shape=(25,35), scenes=2, channels=4):
    components=NSO_Components(args)
    components.initialize('cpu',scenes,*shape,*shape,rpn_in_channels=channels)
    return components,NSORuntimeIntegration(components,scenes,shape)


class RuntimeTests(unittest.TestCase):
    def test_disabled_is_noop_even_without_valid_observation(self):
        components,runtime=make(arguments())
        self.assertIsNone(runtime.observe(0,0,None,None,None))
        self.assertEqual(runtime.choose_goal(0,[7,11],None),[7,11])
        self.assertEqual(runtime.states,[None,None])

    def test_real_reward_snapshots_and_components_are_scene_local(self):
        components,runtime=make(arguments(use_igcr=True),shape=(12,12))
        self.assertIsNot(components._reward_computers[0],components._reward_computers[1])
        frame=np.zeros((4,12,12),np.float32);frame[1,:3,:3]=1
        for scene in (0,1):
            runtime.observe(scene,0,frame,[0,12,0,12],[1.,1.,0.])
        expanded=frame.copy();expanded[1,3:6,:3]=1
        changed=runtime.observe(0,1,expanded,[0,12,0,12],[1.,1.,0.])
        unchanged=runtime.observe(1,1,frame,[0,12,0,12],[1.,1.,0.])
        self.assertGreater(changed['parts']['r_ig'],unchanged['parts']['r_ig'])
        self.assertEqual(unchanged['parts']['r_ig'],0.)
        runtime.reset_scene(0)
        self.assertIsNone(runtime.states[0])
        np.testing.assert_array_equal(runtime.states[1]['map'],frame)

    def test_real_topology_goal_uses_connected_path_not_coordinate_clip(self):
        components,runtime=make(arguments(use_topo_graph=True))
        frame=np.zeros((4,25,35),np.float32);frame[1]=1
        frame[0,:,10]=1;frame[0,2,10]=0
        runtime.observe(0,0,frame,[0,25,0,35],[4.,12.,0.])
        graph=components._topo[0]
        graph.nodes={0:RoomNode(0,np.array([12,4]),[(12,4)]),
                     1:RoomNode(1,np.array([12,28]),[(12,28)],frontier_length=20)}
        graph.current_agent_node=0
        chosen=runtime.choose_goal(0,[9,9],[0,25,0,18])
        self.assertEqual(chosen,[2,17])
        self.assertNotEqual(chosen,[12,17])
        self.assertEqual(runtime.states[0]['cell'],(12,4))
        runtime.reset_scene(0)
        self.assertEqual(graph.nodes,{})

    def test_disconnected_or_occupied_target_has_no_projection(self):
        safe=np.ones((8,12),bool);safe[:,5]=False
        self.assertIsNone(NSORuntimeIntegration.project_path_to_window(safe,(4,2),(4,9),(0,8,0,6)))
        self.assertIsNone(NSORuntimeIntegration.project_path_to_window(safe,(4,2),(4,5),(0,8,0,6)))

    def test_topology_target_rejects_footprint_over_unknown_neighbor(self):
        for radius,expected in ((0.,[12,28]),(.5,[8,8])):
            with self.subTest(robot_radius_m=radius):
                components,runtime=make(arguments(use_topo_graph=True,robot_radius_m=radius))
                frame=np.zeros((4,25,35),np.float32);frame[1]=1
                # Target center is observed; its adjacent cell is unknown.
                frame[1,12,29]=0
                runtime.observe(0,0,frame,[0,25,0,35],[4.,12.,0.])
                graph=components._topo[0]
                graph.nodes={0:RoomNode(0,np.array([12,4]),[(12,4)]),
                             1:RoomNode(1,np.array([12,28]),[(12,28)],frontier_length=20)}
                graph.current_agent_node=0
                self.assertEqual(runtime.choose_goal(0,[8,8],[0,25,0,35]),expected)

    def test_missing_rpn_weights_never_predict_or_change_actor_goal(self):
        components,runtime=make(arguments(use_rpn_uq=True))
        self.assertFalse(components._rpn_ready)
        self.assertEqual(components.predict_reachability_uq(torch.zeros(1,4,25,35)),(None,None))
        frame=np.zeros((4,25,35),np.float32);frame[1]=1
        runtime.observe(0,0,frame,[0,25,0,35],[4.,12.,0.])
        with patch.object(components._rpn_uq,'forward',side_effect=AssertionError('random UQ used')):
            self.assertEqual(runtime.choose_goal(0,[8,8],[0,25,0,35]),[8,8])
        self.assertFalse(runtime.audit[-1]['uq_active'])

    def test_loaded_fixture_uq_changes_real_topology_gating(self):
        # Synthetic fixed tensors test wiring, not a claim of trained accuracy.
        network=ReachabilityHeadUQ(in_channels=4,hidden=64,t_mc=2)
        for parameter in network.parameters():parameter.data.zero_()
        network.mean_head.bias.data.fill_(-10.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.pt';torch.save(network.state_dict(),path)
            args=arguments(use_rpn_uq=True,use_topo_graph=True,goal_reachability_model_path=str(path))
            self.assertEqual(infer_uq_channels(args),4)
            components,runtime=make(args)
            self.assertTrue(components._rpn_ready)
            frame=np.zeros((4,25,35),np.float32);frame[1]=1
            runtime.observe(0,0,frame,[0,25,0,35],[4.,12.,0.])
            graph=components._topo[0]
            graph.nodes={0:RoomNode(0,np.array([12,4]),[(12,4)]),
                         1:RoomNode(1,np.array([12,28]),[(12,28)],frontier_length=20)}
            graph.current_agent_node=0
            self.assertEqual(runtime.choose_goal(0,[8,8],[0,25,0,35]),[8,8])
            self.assertTrue(runtime.audit[-1]['uq_active'])
            self.assertFalse(runtime.audit[-1]['topology_proposed'])

    def test_lfs_pointer_is_reported_unavailable_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pointer';path.write_text('version https://git-lfs.github.com/spec/v1\n')
            args=arguments(use_rpn_uq=True,goal_reachability_model_path=str(path))
            self.assertEqual(infer_uq_channels(args),2)
            components,_=make(args)
            self.assertFalse(components._rpn_ready)
            self.assertIn('unavailable',components.capabilities['reachability_uq'])

    def test_missing_clip_does_not_construct_detector_or_constant_semantic_field(self):
        with patch('nso.clip_semantic_map._try_import_clip',return_value=None), \
             patch('nso.clip_semantic_map.OVSemanticDensityField',side_effect=AssertionError('detector constructed')):
            components,runtime=make(arguments(use_open_vocab_semantic=True))
        self.assertIsNone(components.get_sem_density(0))
        self.assertIn('unavailable',components.capabilities['semantic'])
        self.assertIsNone(runtime.semantic_window(0,[0,25,0,35]))

    def test_training_does_not_postprocess_ppo_action(self):
        components,runtime=make(arguments(use_topo_graph=True,eval=False,train_global=True))
        with patch.object(components,'select_topo_target',side_effect=AssertionError('PPO action changed')):
            self.assertEqual(runtime.choose_goal(0,[8,9],[0,25,0,35]),[8,9])

    def test_main_contains_initial_periodic_reset_and_reward_adapter_calls(self):
        # Structural check complements real interface tests; not Habitat proof.
        tree=ast.parse(Path('main.py').read_text())
        calls=[node.func.attr for node in ast.walk(tree)
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and isinstance(node.func.value,ast.Name) and node.func.value.id=='nso_runtime']
        self.assertEqual(calls.count('observe'),2)
        self.assertEqual(calls.count('choose_goal'),2)
        self.assertEqual(calls.count('reset_scene'),1)


if __name__=='__main__':unittest.main()
