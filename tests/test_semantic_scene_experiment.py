"""New schema/pins and archive orchestration, using explicit 0-World fixtures."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from env.development_sensor_v41 import runtime_counts_v41
from nso import semantic_experiment as base
from nso import semantic_experiment_replay as replay
from nso import semantic_scene_experiment as m
from nso.observed_mapper_v42 import ObservedMapperV42
from tests.test_episode_driver_v43 import FixedPackets
from tests.test_semantic_experiment import FiniteController


PROTOCOL_PATH=m.ROOT/'configs/virtual3d/semantic_development_20260923.json'


def protocol_fixture():
    p=deepcopy(m.read_json(PROTOCOL_PATH));slot=deepcopy(next(iter(p['slots'].values())))
    slot.update(method='G',budget=1)
    p.update(status='frozen',phase_id='semantic_analytic_test',maximum_new_world_slots=1,
        slots={'analytic_fixture':slot},ledger_relative_path='audit_results/semantic_analytic_test/start_ledger.json',
        output_relative_path='audit_results/semantic_analytic_test/episodes',primary_experiment_started=False,
        scope='Temporary analytic unit fixture, not physical experiment data.')
    return p


class AnalyticContext:
    """All fabricated provenance is confined to an auto-deleted temporary tree."""
    def __enter__(self):
        self.stack=ExitStack();self.root=Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(base,'ROOT',self.root))
        self.stack.enter_context(patch.object(m,'ROOT',self.root))
        self.protocol=protocol_fixture();identity=self.protocol['slots']['analytic_fixture']['asset_id']
        payloads={
            'asset_manifest.json':{'fixture':'asset'},'navigation_manifest.json':{'fixture':'navigation'},
            'reference_manifest.json':{'fixture':'reference'},
        }
        for name,value in payloads.items():(self.root/name).write_bytes(m.canonical_bytes(value))
        ref=dict(root='reference',manifest_sha256=m.file_sha256(self.root/'reference_manifest.json'))
        (self.root/'reference_index.json').write_bytes(m.canonical_bytes(dict(references={identity:ref})))
        for config,name in (('asset_manifest_sha256','asset_manifest.json'),('navigation_manifest_sha256','navigation_manifest.json'),
                            ('reference_index_sha256','reference_index.json')):
            self.protocol[config]=m.file_sha256(self.root/name)
        self.path=self.root/'protocol.json';self.path.write_bytes(m.canonical_bytes(self.protocol))
        for name in m.ENTRY_SOURCES:
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('# temporary analytic fixture source; not an executable experiment\n')
        self.inputs=dict(bundle=dict(graph=SimpleNamespace(input_sha256='a'*64),graph_spec={'fixture':'graph'},
            public_spec={'fixture':'public'},workspace={'fixture':'workspace'}),reference=ref,
            archived_inputs={name:(self.root/name,m.file_sha256(self.root/name)) for name in
                ('asset_manifest.json','navigation_manifest.json','reference_manifest.json','reference_index.json')},
            declared_structure_reference_inputs={})
        self.stack.enter_context(patch.object(m,'load_inputs',return_value=self.inputs))
        self.stack.enter_context(patch.object(m,'source_names',return_value=list(m.ENTRY_SOURCES)))
        self.stack.enter_context(patch.object(m,'storage_report_v41',return_value={'passed':True,'fixture':True}))
        self.counts=dict(worlds_created=0,rgbd_frames=0,scans=0,paid_actions=0,blocked_before_world_creation=0)
        self.stack.enter_context(patch.object(m,'runtime_counts_v41',side_effect=lambda:deepcopy(self.counts)))
        self.stack.enter_context(patch.object(m,'make_components',side_effect=lambda *a:(FiniteController(),self.mapper())))
        outer=self
        class AnalyticSensor(FixedPackets):
            def initial_observation(self):
                outer.counts['rgbd_frames']+=1;outer.counts['scans']+=1
                return super().initial_observation()
            def step(self,action):
                outer.counts['rgbd_frames']+=1;outer.counts['scans']+=1;outer.counts['paid_actions']+=1
                return super().step(action)
        def sensor(*args,**kwargs):
            # A test-only local counter, never the real physics counter.
            self.counts['worlds_created']+=1
            return AnalyticSensor([])
        self.factory=self.stack.enter_context(patch.object(m,'create_semantic_scene_sensor',side_effect=sensor))
        return self

    @staticmethod
    def mapper():return ObservedMapperV42(shape=(40,40),resolution_m=.1,origin_xy_m=(-.5,-.5))

    def run(self):return m.run_experiment(self.path,'analytic_fixture')

    def __exit__(self,*args):return self.stack.__exit__(*args)


class SemanticSceneExperimentTests(unittest.TestCase):
    def test_real_matrix_schema_and_new_source_roots(self):
        p=m.validate_protocol(m.read_json(PROTOCOL_PATH));self.assertEqual(len(p['slots']),96)
        sources=m.source_names()
        for name in ('env/semantic_scene_sensor.py','nso/semantic_scene_assets.py','nso/semantic_scene_navigation.py',
                     'nso/semantic_scene_evaluation.py','nso/complete_surface_evaluation.py',
                     'nso/known_structure_reference.py',*m.ENTRY_SOURCES):
            self.assertIn(name,sources)
        p=deepcopy(p);p['slots'][next(iter(p['slots']))]['asset_id']='DEV_A_00'
        with self.assertRaises(ValueError):m.validate_protocol(p)

    def test_draft_run_and_failed_resource_gate_never_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            p=protocol_fixture();p['status']='draft';path=Path(directory)/'p.json';path.write_bytes(m.canonical_bytes(p))
            with patch.object(m,'PhaseLedger') as ledger,patch.object(m,'create_semantic_scene_sensor') as sensor:
                with self.assertRaisesRegex(ValueError,'freeze'):m.run_experiment(path,'analytic_fixture')
                ledger.assert_not_called();sensor.assert_not_called()
            p['status']='frozen';path.write_bytes(m.canonical_bytes(p))
            with patch.object(m,'storage_report_v41',return_value={'passed':False}), \
                 patch.object(m,'PhaseLedger') as ledger,patch.object(m,'load_inputs') as inputs:
                result=m.run_experiment(path,'analytic_fixture')
                self.assertEqual(result['status'],'blocked_before_world_creation')
                ledger.assert_not_called();inputs.assert_not_called()

    def test_preflight_pins_but_does_not_mutate_phase(self):
        before=runtime_counts_v41()
        with AnalyticContext() as fixture:
            result=m.run_experiment(fixture.path,'analytic_fixture',preflight_only=True)
            self.assertEqual(result['status'],'ready_without_world_creation')
            self.assertFalse(result['start_slot_reserved']);fixture.factory.assert_not_called()
            self.assertFalse((fixture.root/'audit_results').exists())
        self.assertEqual(before,runtime_counts_v41())

    def test_full_analytic_driver_archive_inspect_and_same_driver_tsdf_replay(self):
        before=runtime_counts_v41()
        with AnalyticContext() as fixture:
            result=fixture.run();self.assertEqual(result['status'],'controller_blocked')
            episode=m.inspect_experiment(result['output'],result['artifact_manifest_sha256'])
            self.assertTrue(episode['current_sources_match'])
            self.assertEqual(episode['result']['mapper_frames'],2)
            with patch.object(replay,'make_components',return_value=(FiniteController(),fixture.mapper())):
                verified=replay.replay_saved_episode(episode)
            self.assertTrue(verified['prediction_verified']);self.assertEqual(verified['replayed_tsdf_integrations'],2)
            with patch.object(replay,'make_components',return_value=(FiniteController('left'),fixture.mapper())):
                with self.assertRaises(ValueError):replay.replay_saved_episode(episode)
            (fixture.root/m.ENTRY_SOURCES[0]).write_text('# changed later\n')
            historic=m.inspect_experiment(result['output'],result['artifact_manifest_sha256'])
            self.assertFalse(historic['current_sources_match']);self.assertIsNone(historic['bundle'])
            with self.assertRaisesRegex(ValueError,'matching executed sources'):replay.replay_saved_episode(historic)
            with self.assertRaises(FileExistsError):fixture.run()
        self.assertEqual(before,runtime_counts_v41())

    def test_sensor_construction_failure_is_terminal_no_retry(self):
        before=runtime_counts_v41()
        with AnalyticContext() as fixture:
            fixture.factory.side_effect=ValueError('analytic injected construction failure')
            failure=fixture.run();self.assertEqual(failure['status'],'experiment_attempt_failed')
            self.assertFalse(failure['world_created'])
            ledger=m.read_json(fixture.root/fixture.protocol['ledger_relative_path'])
            self.assertEqual(ledger['entries'][0]['status'],'experiment_attempt_failed')
            self.assertTrue((fixture.root/fixture.protocol['output_relative_path']/'analytic_fixture/attempt_failure.json').is_file())
            with self.assertRaises(FileExistsError):fixture.run()
        self.assertEqual(before,runtime_counts_v41())

    def test_archive_tamper_is_rejected(self):
        with AnalyticContext() as fixture:
            result=fixture.run();path=Path(result['output'])/'public_workspace.json';path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                m.inspect_experiment(result['output'],result['artifact_manifest_sha256'])

    def test_complete_endpoint_keeps_tiny_positive_face_and_checks_prediction_pin(self):
        from nso.offline_evaluation_v44 import _array_sha
        from nso.surface_evaluation_v40 import CandidateViewV40,freeze_reference_v40
        vertices=np.array([[0.,0.,2.],[0.,1.,2.],[1.,1.,2.],[1.,0.,2.]])
        triangles=np.array([[0,1,2],[0,2,3]])
        view=CandidateViewV40(intrinsic=[[10.,0.,30.],[0.,10.,30.],[0.,0.,1.]],
            world_from_camera=np.eye(4),width=100,height=100)
        reference=freeze_reference_v40(vertices,triangles,np.zeros(2,dtype=int),[view],
            sample_spacing_m=.3,seed=4001,max_samples=50000)
        domain=np.ones((2,2),bool);belief=np.zeros((2,2),np.int8);workspace={'analytic_fixture':True}
        descriptor=dict(shape=[2,2],resolution_m=.1,origin_xy_m=[0.,0.],grid_convention='analytic fixture',
            denominator_cells=4,denominator_mask_sha256=_array_sha(domain))
        record=dict(coverage=descriptor,public_workspace=workspace)
        before=runtime_counts_v41()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prediction=root/'prediction';prediction.mkdir()
            (root/'public_workspace.json').write_bytes(m.canonical_bytes(workspace))
            (root/'artifact_manifest.json').write_bytes(m.canonical_bytes({'analytic_fixture':True}))
            (prediction/'mapper.json').write_bytes(m.canonical_bytes(dict(descriptor,backend_poisoned=False,
                frames=1,occupancy_sha256=_array_sha(belief))))
            np.savez(prediction/'occupancy.npz',belief=belief,observed=domain)
            submitted=np.vstack((vertices,[[3.,0.,2.],[3.,1e-7,2.],[3.+1e-7,0.,2.]]))
            np.savez(prediction/'mesh.npz',vertices=submitted,triangles=np.vstack((triangles,[[4,5,6]])),
                vertex_colors=np.zeros_like(submitted))
            files={name:dict(sha256=m.file_sha256(root/name)) for name in
                ('public_workspace.json','prediction/mapper.json','prediction/occupancy.npz','prediction/mesh.npz')}
            episode=dict(root=root,current_sources_match=True,protocol=protocol_fixture(),
                slot=protocol_fixture()['slots']['analytic_fixture'],reference=dict(root='reference',manifest_sha256='a'*64),
                manifest_sha256=m.file_sha256(root/'artifact_manifest.json'),manifest=dict(files=files),
                result=dict(acquired_and_saved_packets=1,status='controller_stop',sensor_status={'returned_xy_and_yaw':True}))
            with patch.object(m,'load_semantic_scene_reference',return_value=(reference,domain,record)):
                result=m.evaluate_endpoint(episode)
                self.assertEqual(result['metrics']['positive_subthreshold_faces_preserved'],1)
                self.assertEqual(result['metrics']['submitted_prediction_triangles'],3)
                self.assertGreater(result['metrics']['extra_false_positive_area_m2'],0)
                self.assertEqual(result['metrics']['C_nav'],1.)
                (prediction/'mapper.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'prediction differs'):m.evaluate_endpoint(episode)
        self.assertEqual(before,runtime_counts_v41())


if __name__=='__main__':unittest.main()
