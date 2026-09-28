"""Critical finite-ledger, factory and same-driver replay checks; no live World."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nso import semantic_experiment as m
from nso import semantic_experiment_replay as r
from nso.evidence_writer_v44 import CompressedStepWriterV44
from nso.episode_driver_v43 import execute_episode_v43
from tests.test_episode_driver_v43 import FixedPackets


PROTOCOL_PATH=m.ROOT/'configs/virtual3d/semantic_integration_20260923.json'


class FiniteMapper:
    def __init__(self):self.frames=[]
    def update(self,observation,scan):
        self.frames.append(observation)
        return dict(frame_id=observation.frame_id,observation_sha256=observation.sha256())
    def mesh_arrays(self):
        return dict(vertices=np.empty((0,3)),triangles=np.empty((0,3),np.int32),vertex_colors=np.empty((0,3)))
    def occupancy_arrays(self):return np.full((2,2),-1,np.int8),np.zeros((2,2),bool)
    def snapshot(self):return dict(frames=len(self.frames),tsdf_integration_count=len(self.frames))


class FiniteController:
    def __init__(self,first='forward'):self.index=-1;self.first=first
    def accept(self,observation,mapper,execution_outcome):
        self.index+=1
        return dict(observation_sha256=observation.sha256(),execution_outcome=execution_outcome)
    def choose(self):
        return dict(action=self.first if self.index==0 else 'blocked',reason='finite test only',paid_step=self.index)


def saved_fixture(root,mapper=None):
    writer=CompressedStepWriterV44(root)
    result=execute_episode_v43(FixedPackets([]),FiniteController(),FiniteMapper() if mapper is None else mapper,writer,budget=1)
    writer.json('encoding.json',dict(schema='v44.step_encoding.v1',steps=writer.step_encoding),terminal=True)
    return dict(root=Path(root),protocol=dict(maximum_file_bytes=32*1024**2,maximum_elapsed_s=600.),
        slot=dict(budget=1),result=result,manifest=dict(files=deepcopy(writer.files)),bundle={},
        current_sources_match=True)


class SemanticExperimentTests(unittest.TestCase):
    def test_gate_blocks_before_assets_ledger_and_world(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(m,'storage_report_v41',return_value={'passed':False}), \
             patch.object(m,'load_bundle') as bundle,patch.object(m,'PhaseLedger') as ledger, \
             patch.object(m,'create_development_sensor') as factory:
            output=Path(directory)/'episodes'
            result=m.run_experiment(PROTOCOL_PATH,'integration_A_G',output_root=output)
            self.assertEqual(result['status'],'blocked_before_world_creation')
            self.assertFalse(output.exists())
            bundle.assert_not_called();ledger.assert_not_called();factory.assert_not_called()

    def test_phase_cannot_expand_or_change_ledger_to_restart(self):
        p=deepcopy(m.read_json(PROTOCOL_PATH));sha='a'*64
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)):
            old=Path(directory)/'audit_results/v43_development_batch_20260921/start_ledger.json'
            old.parent.mkdir(parents=True);old.write_text('historical sentinel')
            ledger=m.PhaseLedger(p,sha)
            meta=dict(output='/first',source_sha256={'module.py':'b'*64})
            ledger.reserve('integration_A_G',meta)
            ledger.finish('integration_A_G',status='failed',world_created=False,result_sha256='c'*64)
            with self.assertRaisesRegex(ValueError,'already reserved'):
                ledger.reserve('integration_A_G',dict(meta,output='/different'))
            changed=deepcopy(p);changed['ledger_relative_path']='audit_results/semantic_elsewhere/start_ledger.json'
            with self.assertRaisesRegex(ValueError,'already bound'):
                m.PhaseLedger(changed,sha).reserve('integration_A_G',meta)
            expanded=deepcopy(p);expanded['slots']['extra']=deepcopy(p['slots']['integration_A_G'])
            with self.assertRaisesRegex(ValueError,'already bound'):
                m.PhaseLedger(expanded,'d'*64).reserve('extra',meta)
            with self.assertRaisesRegex(ValueError,'source closure differs'):
                ledger.reserve('integration_A_B',dict(meta,source_sha256={'module.py':'e'*64}))
            self.assertEqual(old.read_text(),'historical sentinel')
            self.assertEqual(len(m.read_json(ledger.path)['entries']),1)

    def test_protocol_rejects_old_ledger_and_unlisted_slots(self):
        p=deepcopy(m.read_json(PROTOCOL_PATH));p['ledger_relative_path']='audit_results/mechanism_development_20260923/start_ledger.json'
        with self.assertRaisesRegex(ValueError,'old ledgers forbidden'):m.validate_protocol(p)
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'ROOT',Path(directory)):
            ledger=m.PhaseLedger(m.read_json(PROTOCOL_PATH),'a'*64)
            with self.assertRaisesRegex(ValueError,'undeclared run ID'):
                ledger.reserve('not_declared',dict(source_sha256={}))

    def test_geometry_factory_never_constructs_private_structure_matcher(self):
        p=m.read_json(PROTOCOL_PATH);slot=p['slots']['integration_A_G']
        bundle=dict(graph=object(),home_state=object(),public_spec=dict(task=dict(max_actions=160),
            structure_prior=dict(abstract_structures=['planar'],probability_by_category={'x':[1.]})),
            workspace=dict(marker_palette={'x':[1,2,3]},bounds_xy_m=[[0.,0.],[2.,2.]]))
        with patch('nso.controller_semantic_mechanism.SemanticMechanismController') as controller, \
             patch('nso.observed_mapper_v42.ObservedMapperV42'), \
             patch('nso.known_structure_reference.PaidInstanceStructureMatcher.from_asset') as private:
            m.make_components(p,slot,bundle)
            private.assert_not_called()
            self.assertEqual(controller.call_args.kwargs['observation_frontend'],'bounded_local')
            self.assertEqual(controller.call_args.kwargs['method'],'G')

    def test_saved_packet_and_writer_adapters_replay_entire_driver_without_world(self):
        with tempfile.TemporaryDirectory() as directory:
            episode=saved_fixture(Path(directory))
            with patch.object(r,'make_components',return_value=(FiniteController(),FiniteMapper())), \
                 patch.object(m,'create_development_sensor') as factory:
                result=r.replay_saved_episode(episode)
            self.assertEqual(result['status'],'verified')
            self.assertEqual(result['frames_verified'],2)
            self.assertEqual(result['replayed_tsdf_integrations'],2)
            self.assertEqual(result['new_worlds'],0)
            factory.assert_not_called()

    def test_replay_rejects_changed_decision_and_source_version(self):
        with tempfile.TemporaryDirectory() as directory:
            episode=saved_fixture(Path(directory))
            with patch.object(r,'make_components',return_value=(FiniteController(first='left'),FiniteMapper())):
                with self.assertRaises(ValueError):r.replay_saved_episode(episode)
            episode['current_sources_match']=False
            with patch.object(r,'make_components') as factory:
                with self.assertRaisesRegex(ValueError,'matching executed sources'):r.replay_saved_episode(episode)
                factory.assert_not_called()

    def test_analytic_packets_reconstruct_nonempty_real_open3d_tsdf_twice(self):
        from nso.observed_mapper_v42 import ObservedMapperV42
        def mapper():return ObservedMapperV42(shape=(40,40),resolution_m=.1,origin_xy_m=(-.5,-.5))
        with tempfile.TemporaryDirectory() as directory:
            episode=saved_fixture(Path(directory),mapper())
            with np.load(Path(directory)/'prediction/mesh.npz',allow_pickle=False) as mesh:
                self.assertGreater(len(mesh['triangles']),0)
            with patch.object(r,'make_components',return_value=(FiniteController(),mapper())), \
                 patch.object(m,'create_development_sensor') as factory:
                result=r.replay_saved_episode(episode)
            self.assertEqual(result['status'],'verified')
            self.assertTrue(result['prediction_verified'])
            self.assertEqual(result['replayed_tsdf_integrations'],2)
            self.assertEqual(result['terminal_status_verified'],'controller_blocked')
            factory.assert_not_called()


if __name__=='__main__':unittest.main()
