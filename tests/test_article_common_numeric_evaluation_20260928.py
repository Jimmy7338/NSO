"""Synthetic posthoc lifecycle tests: never evaluate a real mesh or a World."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import evaluate_article_common_numeric_20260928 as api
from nso.article_prediction_mesh_adapter_v1 import prepare_prediction_mesh_v1


class CommonNumericLifecycleTests(unittest.TestCase):
    def fixture(self,root,*,tiny=False):
        episode=root/'original/run';(episode/'prediction').mkdir(parents=True)
        v=np.array([[0.,0,0],[1.,0,0],[0.,1.,0],[0.,1e-13,0]])
        t=np.array([[0,1,2],[0,1,3]] if tiny else [[0,1,2]],dtype=np.int32)
        np.savez(episode/'prediction/mesh.npz',vertices=v,triangles=t)
        np.savez(episode/'prediction/occupancy.npz',belief=np.zeros((2,2)))
        _,_,adapter=prepare_prediction_mesh_v1(v,t)
        prepared=root/'supplement/run/prepared';prepared.mkdir(parents=True)
        snapshot=dict(run_id='run',frozen_plan_sha256='frozen',original_episode='original/run',
            original_end_to_end_status='attempt_failed' if tiny else 'controller_stop',
            original_end_to_end_qualified=not tiny,sealed_prediction_available=True,
            original_prediction_seal={'mesh.npz':api.sha(episode/'prediction/mesh.npz')},
            motion_completion={'motion_completion_verified':True},adapter=adapter,input_files=api.inventory(episode))
        api.save(prepared/'input_snapshot.json',snapshot)
        api.save(prepared/'manifest.json',{'files':api.inventory(prepared)})
        plan=dict(output_root='supplement',slots={'run':{'scene_id':'scene'}},asset_manifest_sha256='asset',
            references={'scene':{'root':'reference','manifest_sha256':'reference-pin'}},evaluation=api.EVALUATION)
        reference=SimpleNamespace(fingerprint='fp')
        return episode,prepared,plan,reference,adapter

    def environment(self,stack,root,plan,reference):
        stack.enter_context(patch.object(api,'validated_plan',return_value=(plan,{})))
        stack.enter_context(patch.object(api,'rooted',side_effect=lambda p:root/p))
        stack.enter_context(patch.object(api,'load_reference',return_value=(reference,np.ones((2,2)),{'asset_manifest_sha256':'asset','coverage':{}})))
        stack.enter_context(patch.object(api,'measure_navigation_coverage_v44',return_value={'C_nav':.75}))

    def test_reuse_does_not_call_evaluator_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);episode,_,plan,reference,_=self.fixture(root)
            self.environment(stack,root,plan,reference)
            old={'metrics':{'Q':.5,'C_nav':.75,'J_nav':.375},'source_evaluation_sha256':'old'}
            stack.enter_context(patch.object(api,'previous_quality',return_value=old))
            evaluate=stack.enter_context(patch.object(api,'evaluate_surface_v40',side_effect=AssertionError('must not evaluate')))
            before=api.inventory(episode)
            result=api.measure(Path('plan'),'frozen','run')
            self.assertEqual(result['mode'],'verified_exact_input_reuse');self.assertEqual(result['new_surface_evaluations'],0)
            self.assertEqual(api.inventory(episode),before);evaluate.assert_not_called()
            with self.assertRaises(FileExistsError):api.measure(Path('plan'),'frozen','run')

    def test_failed_online_state_survives_new_mocked_derived_measurement(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);episode,_,plan,reference,_=self.fixture(root,tiny=True)
            self.environment(stack,root,plan,reference)
            stack.enter_context(patch.object(api,'previous_quality',return_value=None))
            def evaluated(ref,v,t,**kw):
                self.assertEqual(len(t),1);self.assertEqual(len(v),4)
                np.testing.assert_array_equal(t,[[0,1,2]])
                self.assertEqual(kw,dict(C_map=.75,**api.EVALUATION))
                return {'Q':.5,'C_map':.75,'J':.375}
            evaluator=stack.enter_context(patch.object(api,'evaluate_surface_v40',side_effect=evaluated))
            original=api.inventory(episode);result=api.measure(Path('plan'),'frozen','run')
            self.assertTrue(result['quality_measurement_available']);self.assertTrue(result['motion_completion_verified'])
            self.assertEqual(result['original_end_to_end_status'],'attempt_failed')
            self.assertFalse(result['original_end_to_end_qualified']);self.assertTrue(result['original_failed_attempt_remains_failed'])
            self.assertEqual(result['new_surface_evaluations'],1);self.assertEqual(evaluator.call_count,1)
            self.assertEqual(api.inventory(episode),original)

    def test_evaluator_failure_keeps_attempt_and_refuses_second_call(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);_,_,plan,reference,_=self.fixture(root,tiny=True)
            self.environment(stack,root,plan,reference)
            stack.enter_context(patch.object(api,'previous_quality',return_value=None))
            evaluate=stack.enter_context(patch.object(api,'evaluate_surface_v40',side_effect=ValueError('synthetic failure')))
            with self.assertRaisesRegex(ValueError,'synthetic failure'):api.measure(Path('plan'),'frozen','run')
            result=api.read(root/'supplement/run/measurement/result.json')
            self.assertEqual(result['status'],'derived_measurement_failed');self.assertFalse(result['quality_measurement_available'])
            with self.assertRaises(FileExistsError):api.measure(Path('plan'),'frozen','run')
            self.assertEqual(evaluate.call_count,1)

    def test_changed_forensic_input_refuses_before_creating_measurement(self):
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);episode,_,plan,reference,_=self.fixture(root)
            self.environment(stack,root,plan,reference)
            with (episode/'prediction/mesh.npz').open('ab') as stream:stream.write(b'corruption')
            evaluate=stack.enter_context(patch.object(api,'evaluate_surface_v40'))
            with self.assertRaisesRegex(ValueError,'forensic inventory'):api.measure(Path('plan'),'frozen','run')
            evaluate.assert_not_called();self.assertFalse((root/'supplement/run/measurement').exists())

    def test_wrong_explicit_plan_hash_and_path_escape_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=Path(directory)/'plan.json';plan.write_text('{}')
            with self.assertRaisesRegex(ValueError,'plan SHA'):api.validated_plan(plan,'wrong')
        for name in ('../escape','/tmp/escape'):
            with self.subTest(name=name),self.assertRaises(ValueError):api.rooted(name)

    def test_real_reuse_gate_requires_equal_review_prediction_reference_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);episode=root/'run';episode.mkdir()
            api.save(episode/'artifact_manifest.json',{'synthetic':True})
            metrics=dict(reference_fingerprint='fp',C_nav=.75,Q=.5,J_nav=.375,
                threshold_m=.05,prediction_sample_spacing_m=.3,prediction_seed=4002)
            evaluation=dict(qualified=True,metrics=metrics,source_prediction_sha256={'mesh.npz':'sealed'},
                reference_manifest_sha256='reference',coverage={'C_nav':.75},no_roi_crop=True,semantic_weights_used=False)
            api.save(episode/'evaluation.json',evaluation)
            review_root=root/'reviews/run';review_root.mkdir(parents=True)
            api.save(review_root/'review.json',dict(all_checks_passed=True,qualified=True,
                input_manifest_sha256=api.sha(episode/'artifact_manifest.json'),protocol_sha256='protocol',metrics=metrics))
            api.save(review_root/'manifest.json',dict(reviewer_source_sha256='reviewer',files={'review.json':api.record(review_root/'review.json')}))
            plan=dict(reviews_root='reviews',protocol_sha256='protocol',slots={'run':{'scene_id':'scene'}},
                references={'scene':{'manifest_sha256':'reference'}},evaluation=api.EVALUATION,
                source_sha256={'scripts/review_article_episode_20260928.py':'reviewer'})
            snapshot=dict(adapter={'removed_faces':0},motion_completion={'motion_completion_verified':True},
                original_end_to_end_qualified=True,original_prediction_seal={'mesh.npz':'sealed'})
            reference=SimpleNamespace(fingerprint='fp');coverage={'C_nav':.75}
            with patch.object(api,'ROOT',root),patch.object(api,'rooted',side_effect=lambda p:root/p):
                self.assertIsNotNone(api.previous_quality(episode,plan,snapshot,reference,coverage))
                with self.assertRaisesRegex(ValueError,'unequal reference'):
                    api.previous_quality(episode,plan,snapshot,SimpleNamespace(fingerprint='changed'),coverage)
                with self.assertRaisesRegex(ValueError,'unequal reference'):
                    api.previous_quality(episode,plan,snapshot,reference,{'C_nav':.74})
                with self.assertRaisesRegex(ValueError,'unequal reference'):
                    api.previous_quality(episode,plan,{**snapshot,'original_prediction_seal':{'mesh.npz':'changed'}},reference,coverage)
                self.assertIsNone(api.previous_quality(episode,plan,{**snapshot,'adapter':{'removed_faces':1}},reference,coverage))


if __name__=='__main__':unittest.main()
