"""Exposure review attack fixtures: no World, mapping or performance trial."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from env.development_sensor_v41 import runtime_counts_v41
from nso.analytic_fixture_v42 import analytic_forward_sequence_v42, analytic_plane_observation_v42
from nso.controller_article_v1 import ArticleObservedResidualV1
from nso.view_quality_article_exposure_v3 import ArticleExposureViewPredictorV3
from scripts import review_article_exposure_episode_20260928 as review
from tests.test_view_quality_v42 import ledger, public_view


def protocols():
    baseline=json.loads((review.ROOT/'configs/virtual3d/article_ground_ablation_v2_20260928.json').read_text())
    current=deepcopy(baseline);current['schema']=review.SCHEMA
    current['controller']['same_center_exposure_dedup']=True
    current['slots']={'exposure_'+key:{**value,'paired_baseline_run_id':key} for key,value in baseline['slots'].items()}
    return current,baseline


def fixture(multi=True):
    controller=dict(multi_view_planes=multi,maximum_instances=8)
    observed=ledger(mode='G');residual=ArticleObservedResidualV1(multi_view_planes=multi)
    predictor=ArticleExposureViewPredictorV3(residual=residual,maximum_instances=8)
    auditor=review.ExposureAudit(controller)
    for packet in analytic_forward_sequence_v42():
        association=observed.observe(packet);plane={}
        if multi:plane['article_plane_evidence']=residual.article_planes.observe(packet,association['accepted'])
        evidence=dict(association=association,observed_residual=plane,view_evidence=predictor.observe(packet,association['accepted']),
            paid_step=packet.paid_step,observation_sha256=packet.sha256(),geometry_feedback=[],
            structure_belief=dict(instances=[dict(row,geometry_log_evidence=row['geometry_log_scores']) for row in deepcopy(association['instances'])]))
        auditor.observe(packet,evidence)
    instance=observed.snapshot()['instances'][0]
    views=[public_view('target:1',x=1.,yaw=np.pi/6),public_view('second:0',x=.25,y=1.25)]
    forecasts=[predictor.forecast(instance,views)]
    selection=dict(forecasts=forecasts,selected=dict(target=dict(node='target',heading=1)))
    return auditor,predictor,observed,residual,evidence,selection


def coherent_reallocate(row,probabilities):
    component=next(c for c in row['components'] if c['excluded_previously_exposed_unseen_area_m2']>.01)
    delta=.001
    component['excluded_previously_exposed_unseen_area_m2']-=delta
    component['new_surface_area_m2']+=delta
    new=[float(np.mean([c['new_surface_area_m2'] for c in row['components'] if c['structure']==name])) for name in review.STRUCTURES]
    row['structure_new_surface_area_m2']=row['unexcluded_structure_new_surface_area_m2']=new
    row['expected_new_surface_area_m2']=row['unexcluded_expected_new_surface_area_m2']=float(np.asarray(probabilities)@new)


class ExposureEpisodeReviewTests(unittest.TestCase):
    def setUp(self):self.before=runtime_counts_v41()
    def tearDown(self):self.assertEqual(runtime_counts_v41(),self.before)

    def test_strict_twelve_matrix_and_unchanged_common_runtime(self):
        current,baseline=protocols();self.assertTrue(review.verify_pairing(current,baseline))
        for key in ('maximum_elapsed_s','maximum_episode_bytes','reserve_bytes'):
            bad=deepcopy(current);bad[key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):review.verify_pairing(bad,baseline)

    def test_unknown_baseline_and_method_specific_overrides_rejected(self):
        current,baseline=protocols();key=next(iter(current['slots']))
        for kind in ('unknown','override','method','global','partial','schema'):
            bad=deepcopy(current)
            if kind=='unknown':bad['slots'][key]['paired_baseline_run_id']='missing'
            elif kind=='override':bad['slots'][key]['controller_overrides']={'same_center_exposure_dedup':False}
            elif kind=='method':bad['slots'][key]['method']='S_no_future'
            elif kind=='global':bad['controller']['inspection_weight']=2.
            elif kind=='partial':bad['slots'].pop(key)
            else:bad['schema']='article.experiment_protocol.ground_v2'
            with self.subTest(kind=kind),self.assertRaises(ValueError):review.verify_pairing(bad,baseline)

    def test_old_shared_source_cannot_be_changed_or_dropped(self):
        baseline={'scripts/run_article_ground_experiment_20260928.py':'a'*64,'nso/old.py':'b'*64}
        current={'nso/old.py':'b'*64,'nso/new.py':'c'*64}
        review.verify_source_relationship(current,baseline)
        for bad in ({'nso/old.py':'x'*64},{'nso/new.py':'c'*64}):
            with self.assertRaises(ValueError):review.verify_source_relationship(bad,baseline)

    def test_actual_controller_sole_switch_and_predictor_pin(self):
        protocol,_=protocols();audit=review.ExposureAudit(protocol['controller'])
        mapping={'cache_feedback_repair':'common_cache_feedback_repair','multi_view_planes':'common_multiview_plane_estimation',
            'ground_association':'common_ground_association'}
        baseline={mapping.get(k,k):v for k,v in protocol['controller'].items() if k!='same_center_exposure_dedup'}
        baseline.update(article_version='article_ground_v2',ground_association_policy=dict(review.GROUND_POLICY))
        config=dict(baseline,article_exposure_version=review.EXPOSURE_VERSION,same_center_exposure_dedup=True,
            exposure_predictor_configuration_sha256=audit.configuration_sha)
        review.verify_actual_configuration(config,baseline,protocol,audit.configuration)
        for key,value in [('same_center_exposure_dedup',False),('inspection_weight',2.),('exposure_predictor_configuration_sha256','0'*64)]:
            bad=dict(config);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):review.verify_actual_configuration(bad,baseline,protocol,audit.configuration)

    def test_full_paid_support_plane_and_all_candidate_math_then_subset_geometry(self):
        audit,predictor,_,_,evidence,selection=fixture()
        audit.verify_forecasts(evidence,selection)
        self.assertEqual(audit.all_candidates,2);self.assertEqual(audit.geometry_candidates,2)
        self.assertEqual(audit.first_geometry_replan,1)
        self.assertTrue(all(r['independent_geometry_recomputed'] for r in audit.rows))

    def test_planning_snapshot_uses_this_frame_feedback_after_association(self):
        _,_,_,_,evidence,_=fixture(False)
        previous=evidence['association']['instances'][0];after=deepcopy(previous)
        after['geometric_feedback_frames']+=1;after['geometry_log_scores']=[0.,-1.,-2.,-3.]
        prior=np.asarray(after['active_structure_prior']);p=prior*np.exp(after['geometry_log_scores']);p/=p.sum()
        after['structure_probabilities']=p.tolist()
        evidence['geometry_feedback']=[dict(applied=True,instance_id=after['instance_id'],instance=after,
            paid_step=evidence['paid_step'],observation_sha256=evidence['observation_sha256'])]
        belief=evidence['structure_belief']['instances'][0];belief['geometry_log_evidence']=after['geometry_log_scores']
        belief['structure_probabilities']=after['structure_probabilities']
        got=review.planning_snapshots(evidence)[after['instance_id']]
        self.assertEqual(got['geometric_feedback_frames'],after['geometric_feedback_frames'])
        self.assertEqual(got['geometry_log_scores'],after['geometry_log_scores'])
        self.assertEqual(previous['geometric_feedback_frames'],0)
        evidence['geometry_feedback'][0]['instance']['support_sha256']='0'*64
        with self.assertRaises(ValueError):review.planning_snapshots(evidence)

    def test_later_replan_geometry_scope_remains_selected_target_only(self):
        audit,predictor,observed,_,evidence,selection=fixture(False)
        audit.verify_forecasts(evidence,selection)
        packet=analytic_plane_observation_v42(2,(1.25,.75,0.));association=observed.observe(packet)
        evidence=dict(association=association,observed_residual={},view_evidence=predictor.observe(packet,association['accepted']),
            paid_step=2,observation_sha256=packet.sha256(),geometry_feedback=[],
            structure_belief=dict(instances=[dict(r,geometry_log_evidence=r['geometry_log_scores']) for r in deepcopy(association['instances'])]))
        audit.observe(packet,evidence)
        views=[public_view('target:1',x=1.25,yaw=np.pi/6),public_view('second:0',x=.25,y=1.25)]
        selection['forecasts']=[predictor.forecast(observed.snapshot()['instances'][0],views)]
        audit.verify_forecasts(evidence,selection)
        self.assertEqual(audit.all_candidates,4);self.assertEqual(audit.geometry_candidates,3)
        self.assertEqual([r['independent_geometry_recomputed'] for r in audit.rows[-2:]],[True,False])

    def test_negative_missing_or_inconsistent_area_receipts_rejected(self):
        audit,_,_,_,evidence,selection=fixture(False)
        forecast=selection['forecasts'][0];original=forecast['candidates'][0]
        for kind in ('negative','missing','partition','observed','mean','beforemeaning','count','gt'):
            row=deepcopy(original);component=row['components'][0]
            if kind=='negative':component['excluded_previously_exposed_unseen_area_m2']=-1.
            elif kind=='missing':component.pop('new_surface_area_before_exposure_m2')
            elif kind=='partition':component['new_surface_area_before_exposure_m2']+=.1
            elif kind=='observed':component['already_observed_area_m2']+=.1
            elif kind=='mean':row['expected_new_surface_area_m2']+=.1
            elif kind=='beforemeaning':row['unexcluded_expected_new_surface_area_m2']=999.
            elif kind=='count':component['same_center_distinct_paid_views']+=1
            else:row['future_depth']=[[1.]]
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                review.verify_candidate_arithmetic(row,forecast['structure_probabilities'],audit.legacy._paid_cameras,None,audit.legacy.bank)

    def test_coherent_forged_partition_caught_by_independent_geometry(self):
        audit,_,_,_,evidence,selection=fixture(False)
        forecast=selection['forecasts'][0];row=forecast['candidates'][0]
        coherent_reallocate(row,forecast['structure_probabilities'])
        review.verify_candidate_arithmetic(row,forecast['structure_probabilities'],audit.legacy._paid_cameras,None,audit.legacy.bank)
        with self.assertRaisesRegex(ValueError,'independent_exposure_geometry'):audit.verify_forecasts(evidence,selection)

    def test_future_or_forged_paid_camera_history_hash_caught(self):
        audit,_,_,_,evidence,selection=fixture(False)
        forecast=selection['forecasts'][0]
        forecast['observed_geometry']['paid_camera_history_sha256']='0'*64
        forecast['geometry_evidence_sha256']=review._digest(forecast['observed_geometry'])
        with self.assertRaises(ValueError):audit.verify_forecasts(evidence,selection)

    def test_raw_support_is_checked_and_cannot_use_private_geometry(self):
        packet=analytic_forward_sequence_v42()[0];observed=ledger(mode='G');association=observed.observe(packet)
        for field in ('points_world_m','true_structure'):
            audit=review.ExposureAudit(dict(multi_view_planes=False,maximum_instances=8));bad=deepcopy(association)
            if field=='points_world_m':bad['accepted'][0][field][0][0]+=.01
            else:bad['accepted'][0][field]='planar'
            with self.subTest(field=field),self.assertRaises(ValueError):audit.observe(packet,dict(association=bad))

    def test_exact_paid_pose_exclusion_links_actual_frame_identity(self):
        audit,predictor,observed,_,evidence,_=fixture(False)
        instance=observed.snapshot()['instances'][0]
        forecast=predictor.forecast(instance,[public_view('same:0',x=1.)]);row=forecast['candidates'][0]
        self.assertTrue(row['repeated_view_excluded'])
        review.verify_candidate_arithmetic(row,forecast['structure_probabilities'],audit.legacy._paid_cameras,None,audit.legacy.bank)
        row['matching_paid_views'][0]['observation_sha256']='0'*64
        with self.assertRaises(ValueError):review.verify_candidate_arithmetic(row,forecast['structure_probabilities'],audit.legacy._paid_cameras,None,audit.legacy.bank)

    def test_running_and_unstarted_not_read_as_corrupt_or_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            phase=Path(temp)/'phase';episode=phase/'episodes'/'one';output=Path(temp)/'review'
            self.assertEqual(review.review(episode,output)['status'],'unstarted');self.assertFalse(output.exists())
            episode.mkdir(parents=True);(phase/'start_ledger.json').write_text(json.dumps(dict(entries=[dict(run_id='one',status='reserved')])))
            (episode/'protocol.json').write_text('{incomplete')
            self.assertEqual(review.review(episode,output)['status'],'pending');self.assertFalse(output.exists())

    def test_common_mesh_metric_keeps_v3_schema_and_unchanged_denominator(self):
        protocol=dict(evaluation=review.EVALUATION,source_sha256={
            'nso/surface_evaluation_v40.py':'a'*64,'nso/article_prediction_mesh_adapter_v1.py':'b'*64})
        evaluation=dict(schema='article.exposure_v3.canonical_evaluation.v1',metric_version=review.METRIC_VERSION,
            evaluation_settings=review.EVALUATION,evaluation_source_sha256=protocol['source_sha256'],
            metric_preprocessing=dict(schema=review.PREPROCESSING_SCHEMA,receipt='prediction_preprocessing.json',receipt_sha256='c'*64),
            original_prediction_files_unchanged=True,no_roi_crop=True,semantic_weights_used=False,fixed_target_instances=4)
        review.verify_evaluation_metadata(evaluation,protocol,'c'*64)
        for key,value in [('fixed_target_instances',2),('schema','article.ground_v2.canonical_evaluation.v1'),('no_roi_crop',False)]:
            bad=dict(evaluation);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):review.verify_evaluation_metadata(bad,protocol,'c'*64)


if __name__=='__main__':unittest.main()
