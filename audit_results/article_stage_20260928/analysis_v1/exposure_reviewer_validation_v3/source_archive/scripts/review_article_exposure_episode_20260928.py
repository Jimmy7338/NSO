#!/usr/bin/env python3
"""Independent read-only Exposure V3 episode contract and receipt audit.

The unchanged ArticleV1 reviewer verifies compatible physics/packet/map/route
receipts in a nested report. Extra checks bind the new protocol, common ground
frontend, paired GroundV2 baseline, paid exposure arithmetic and mesh adapter.
No World, controller execution, TSDF integration or quality remeasurement.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import review_article_episode_20260928 as base
from scripts import review_article_ground_episode_20260928 as ground_review
from nso.controller_article_v1 import ArticleObservedResidualV1, ArticleViewPredictorV1, _without_article_audit
from nso.view_quality_article_exposure_v3 import ArticleExposureViewPredictorV3
from nso.view_quality_v42 import _digest, _candidate_receipt
from nso.observed_instances_v41 import support_digest
from nso.development_geometry_v40 import STRUCTURES
from nso.observed_residual_v41 import _ray_box_depths
from nso.surface_evaluation_v40 import CandidateViewV40
from scipy.spatial import cKDTree
from nso.observed_instances_article_ground_v2 import GROUND_POLICY, ground_mask_from_observation

SCHEMA='article.experiment_protocol.exposure_v3'
PREPROCESSING_SCHEMA='article.prediction_numeric_face_adapter.v1'
METRIC_VERSION='article.common_numeric_face_evaluation.v1'
EVALUATION=dict(threshold_m=.05,sample_spacing_m=.3,seed=4002,max_samples=1000000)
CAPS=dict(development=12,main=96,ablation=24)
GROUND_FIELDS={'common_ground_association','ground_association_policy','ground_removal_scope'}
EXPOSURE_VERSION='article.same_center_paid_exposure.v3'
EXPOSURE_FIELDS={'article_exposure_version','same_center_exposure_dedup','exposure_predictor_configuration_sha256'}
METHODS={'G','B','S','NBV'}
GEOMETRY_SCOPE='all selected-target instance candidates, plus all candidates at first replan with any nonfallback forecast; all other candidates receive full provenance and receipt-arithmetic checks'
EXPOSURE_SOURCE='nso/view_quality_article_exposure_v3.py'


def verify_actual_configuration(configuration,baseline_configuration,protocol,predictor_configuration):
    require(configuration.get('article_version')=='article_ground_v2'
        and configuration.get('common_ground_association') is True,'inherited GroundV2 controller retained')
    require(configuration.get('ground_association_policy')==dict(GROUND_POLICY),'unchanged common ground constants')
    require(configuration.get('article_exposure_version')==EXPOSURE_VERSION
        and configuration.get('same_center_exposure_dedup') is True,'actual common exposure predictor enabled')
    require(configuration.get('exposure_predictor_configuration_sha256')==_digest(predictor_configuration),
        'actual predictor configuration pin')
    if baseline_configuration is not None:
        require({k:v for k,v in configuration.items() if k not in EXPOSURE_FIELDS}==baseline_configuration,
            'entire paired configuration differs only by declared common exposure fields')
    for key,value in protocol['controller'].items():
        mapped={'cache_feedback_repair':'common_cache_feedback_repair',
            'multi_view_planes':'common_multiview_plane_estimation','ground_association':'common_ground_association'}.get(key,key)
        require(configuration.get(mapped)==value,'actual controller option '+key)


def visible_samples(points,normals,boxes,rotation,translation,camera):
    """Independent nominal pinhole/self-occlusion calculation, no scene input."""
    world=points@rotation.T+translation
    transform=np.asarray(camera['world_from_camera']);origin=transform[:3,3]
    xyz=(world-origin)@transform[:3,:3];depth=xyz[:,2]
    projected=xyz@np.asarray(camera['intrinsic']).T
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=projected[:,:2]/projected[:,2,None]
    mask=(depth>=.1)&(depth<=4.)&(uv[:,0]>=-.5)&(uv[:,0]<camera['width']-.5)&(uv[:,1]>=-.5)&(uv[:,1]<camera['height']-.5)
    mask &= np.sum((normals@rotation.T)*(origin-world),axis=1)>1e-10
    ids=np.flatnonzero(mask);answer=np.zeros(len(points),bool)
    if len(ids):
        local_origin=(origin-translation)@rotation
        rays=(points[ids]-local_origin)/depth[ids,None]
        answer[ids]=np.abs(_ray_box_depths(local_origin,rays,boxes)-depth[ids])<=1e-6
    return answer


def paid_matches(history,candidate,*,whole_pose=False):
    found=[];seen=set()
    for paid in history:
        a=np.asarray(paid['world_from_camera']);b=np.asarray(candidate['world_from_camera'])
        position_match=np.allclose(a if whole_pose else a[:3,3],b if whole_pose else b[:3,3],atol=1e-9,rtol=0)
        if (paid['width']==candidate['width'] and paid['height']==candidate['height'] and position_match
                and np.allclose(paid['intrinsic'],candidate['intrinsic'],atol=1e-9,rtol=0)):
            if whole_pose or paid['camera_geometry_sha256'] not in seen:
                found.append(paid);seen.add(paid['camera_geometry_sha256'])
    return found


def independent_components(legacy,state,support,candidate,history):
    """Frozen old area integral plus independently constructed exposure union."""
    original=legacy._areas(state,support,candidate)
    current=_candidate_receipt(candidate);matched=paid_matches(history,current)
    plane=state['plane'];outward=np.asarray(plane['outward_normal_world'])
    rotation=np.column_stack((np.cross([0.,0.,1.],outward),-outward,[0.,0.,1.]))
    anchor=np.asarray(plane['anchor_world_m']);tree=cKDTree(support)
    for row,prototype,sampled in zip(original,legacy.bank.candidates,legacy._samples):
        _,_,boxes,local_anchor=prototype;points,weights,normals=sampled
        translation=anchor-rotation@local_anchor;old=row['new_surface_area_m2'];removed=0.
        if matched:
            union=np.zeros(len(points),bool)
            for camera in matched:union|=visible_samples(points,normals,boxes,rotation,translation,camera)
            current_visible=visible_samples(points,normals,boxes,rotation,translation,current)
            unseen=tree.query(points@rotation.T+translation,k=1,workers=1)[0]>legacy.observed_distance_m
            removed=float(weights[current_visible&unseen&union].sum())
            row['new_surface_area_m2']=float(weights[current_visible&unseen&~union].sum())
        row.update(new_surface_area_before_exposure_m2=old,
            excluded_previously_exposed_unseen_area_m2=removed,same_center_distinct_paid_views=len(matched),
            exposure_exclusion_is_not_measured_surface=True)
    return original


def verify_candidate_arithmetic(row,probabilities,history,reason,bank):
    """Every candidate: strict finite partition, posterior and paid-source links."""
    fields={'view_id','candidate','candidate_sha256','expected_new_surface_area_m2','structure_new_surface_area_m2',
        'structure_visible_area_m2','unexcluded_structure_new_surface_area_m2','unexcluded_expected_new_surface_area_m2',
        'repeated_view_excluded','matching_paid_views','component_areas_before_view_exclusion','components','fallback','fallback_reason'}
    require(set(row)==fields,'exact candidate receipt fields')
    camera=row['candidate'];require(set(camera)=={'view_id','intrinsic','world_from_camera','width','height','near_m','far_m'},'strict public candidate camera')
    candidate=CandidateViewV40(np.asarray(camera['intrinsic']),np.asarray(camera['world_from_camera']),camera['width'],camera['height'],
        near_m=camera['near_m'],far_m=camera['far_m'],view_id=camera['view_id'])
    require(_candidate_receipt(candidate)==camera and _digest(camera)==row['candidate_sha256']
        and camera['view_id']==row['view_id'],'candidate camera geometry/hash binding')
    require(type(row['fallback']) is bool and row['fallback']==(reason is not None) and row['fallback_reason']==reason,'unchanged plane/association fallback')
    matches=paid_matches(history,camera,whole_pose=True)
    expected_matches=[{k:p[k] for k in ('frame_id','paid_step','observation_sha256','camera_geometry_sha256')} for p in matches]
    require(row['matching_paid_views']==expected_matches and type(row['repeated_view_excluded']) is bool
        and row['repeated_view_excluded']==bool(matches),'exact-view gate uses actual paid cameras only')
    require(row['component_areas_before_view_exclusion'] is True,'component values precede exact-view gate')
    if reason is not None:
        require(row['components']==[],'fallback has no fabricated area components');new=visible=[0.]*4
    else:
        require(len(row['components'])==len(bank.candidates),'all fixed structure/scale components required')
        count=len(paid_matches(history,camera));new=[];visible=[]
        for component,(name,scale,_,_) in zip(row['components'],bank.candidates):
            require(set(component)=={'structure','scale','visible_area_m2','new_surface_area_m2','already_observed_area_m2',
                'total_prototype_area_m2','sampled_surface_points','visible_surface_points','new_surface_area_before_exposure_m2',
                'excluded_previously_exposed_unseen_area_m2','same_center_distinct_paid_views','exposure_exclusion_is_not_measured_surface'},'strict area component fields')
            require(component['structure']==name and component['scale']==scale,'shared fixed structure/scale order')
            for key in ('visible_area_m2','new_surface_area_m2','already_observed_area_m2','total_prototype_area_m2',
                        'new_surface_area_before_exposure_m2','excluded_previously_exposed_unseen_area_m2'):
                require(type(component[key]) is float and math.isfinite(component[key]) and component[key]>=0.,'finite nonnegative '+key)
            before=component['new_surface_area_before_exposure_m2'];after=component['new_surface_area_m2'];excluded=component['excluded_previously_exposed_unseen_area_m2']
            require(abs(before-after-excluded)<=1e-12,'before/excluded/after area partition')
            require(abs(component['visible_area_m2']-component['already_observed_area_m2']-before)<=1e-12,'visible/support/unseen area partition')
            require(component['visible_area_m2']<=component['total_prototype_area_m2']+1e-12,'visible area within prototype area')
            require(type(component['sampled_surface_points']) is int and type(component['visible_surface_points']) is int
                and 0<=component['visible_surface_points']<=component['sampled_surface_points']<=20000,'bounded surface point counts')
            require(type(component['same_center_distinct_paid_views']) is int and component['same_center_distinct_paid_views']==count
                and component['exposure_exclusion_is_not_measured_surface'] is True,'same-center provenance and eligibility meaning')
        new=[float(np.mean([c['new_surface_area_m2'] for c in row['components'] if c['structure']==name])) for name in STRUCTURES]
        visible=[float(np.mean([c['visible_area_m2'] for c in row['components'] if c['structure']==name])) for name in STRUCTURES]
    effective=[0.]*4 if matches else new
    for key,expected in dict(structure_new_surface_area_m2=effective,structure_visible_area_m2=visible,
        unexcluded_structure_new_surface_area_m2=new,expected_new_surface_area_m2=float(np.asarray(probabilities)@np.asarray(effective)),
        unexcluded_expected_new_surface_area_m2=float(np.asarray(probabilities)@np.asarray(new))).items():
        exact_or_numeric(row[key],expected,'candidate/'+key)
    return candidate


def planning_snapshots(evidence):
    """accept() stores association before feedback; choose() sees its result."""
    snapshots={r['instance_id']:deepcopy(r) for r in evidence['association']['instances']}
    for feedback in evidence.get('geometry_feedback',[]):
        if not feedback['applied']:continue
        key=feedback['instance_id'];updated=feedback['instance']
        require(key in snapshots and updated['instance_id']==key,'feedback instance identity')
        require(feedback['paid_step']==evidence['paid_step'] and feedback['observation_sha256']==evidence['observation_sha256'],
            'feedback current paid frame binding')
        mutable={'geometry_log_scores','geometric_feedback_frames','structure_probabilities','active_structure_prior'}
        require({k:v for k,v in snapshots[key].items() if k not in mutable}==
            {k:v for k,v in updated.items() if k not in mutable},'geometry feedback cannot replace measured geometry or category identity')
        require(updated['geometric_feedback_frames']==snapshots[key]['geometric_feedback_frames']+1,'one accepted feedback increment')
        snapshots[key]=deepcopy(updated)
    beliefs={r['instance_id']:r for r in evidence['structure_belief']['instances']}
    for key,instance in snapshots.items():
        posterior=beliefs[key]
        require(instance['geometry_log_scores']==posterior['geometry_log_evidence'],'planning evidence is post-feedback')
        for name in ('active_structure_prior','structure_probabilities','semantic_conditioning_used'):instance[name]=posterior[name]
    return snapshots


class ExposureAudit:
    """Rebuild measured support/plane provenance; no controller or residual scores."""
    def __init__(self,controller,graph=None):
        self.residual=ArticleObservedResidualV1(multi_view_planes=controller['multi_view_planes'])
        # Frozen controller_v43 default is 8 when the shared protocol omits it.
        maximum_instances=controller.get('maximum_instances',8)
        self.legacy=ArticleViewPredictorV1(residual=self.residual,maximum_instances=maximum_instances)
        expected=ArticleExposureViewPredictorV3(residual=self.residual,maximum_instances=maximum_instances)
        self.configuration=expected.configuration;self.configuration_sha=_digest(self.configuration)
        self.first_geometry_replan=None;self.rows=[];self.all_candidates=0;self.geometry_candidates=0;self.steps=0
        self.graph=graph

    def observe(self,packet,evidence):
        # This invokes only the frozen measured-marker fitter and support
        # validator, not ArticleObservedResidualV1.observe or its shape solver.
        accepted=evidence['association']['accepted']
        if self.residual.article_planes is not None:
            plane_receipt=self.residual.article_planes.observe(packet,_without_article_audit(accepted))
            exact_or_numeric(evidence['observed_residual']['article_plane_evidence'],plane_receipt,'paid_marker_plane')
        observed=self.legacy.observe(packet,accepted)
        exact_or_numeric(evidence['view_evidence'],observed,'paid_view_evidence')
        self.steps+=1

    def verify_forecasts(self,evidence,selection):
        forecasts=selection.get('forecasts',[])
        selected=selection.get('selected');target=None if not selected else selected['target']
        target_id=None if target is None else target['node']+':'+str(target['heading'])
        if self.first_geometry_replan is None and any(not c['fallback'] for f in forecasts for c in f['candidates']):
            self.first_geometry_replan=evidence['paid_step']
        full=evidence['paid_step']==self.first_geometry_replan
        snapshots=planning_snapshots(evidence)
        require(len({f['instance_id'] for f in forecasts})==len(forecasts),'unique instance forecasts')
        for forecast in forecasts:
            key=forecast['instance_id'];instance=snapshots[key]
            state,support,probabilities=self.legacy._snapshot(instance)
            reason=('association_uncertain' if instance['association_uncertain'] else
                'conflicting_observed_label_planes' if state['plane_conflict'] else
                'no_reliable_observed_label_plane' if state['plane'] is None else None)
            geometry=dict(instance_id=key,support_sha256=support_digest(support),plane_fit=state['plane'],
                source_frames=state['source_frames'],configuration_sha256=self.configuration_sha,
                paid_camera_history_sha256=_digest(self.legacy._paid_cameras))
            expected=dict(schema='v42.observed_posterior_view_quality.v1',ans_module='STGHP',instance_id=key,
                paid_step=evidence['paid_step'],structure_names=list(STRUCTURES),structure_probabilities=probabilities.tolist(),
                semantic_conditioning_used=bool(instance['semantic_conditioning_used']),instance_snapshot_sha256=_digest(instance),
                geometry_evidence_sha256=_digest(geometry),observed_geometry=geometry,
                prototype_bank_sha256=self.legacy.bank.sha256,configuration=self.configuration,configuration_sha256=self.configuration_sha,
                known_surface_rule='distance to accumulated measured support <= observed_distance_m',
                repeated_view_exclusion_is_not_measured_surface_evidence=True,scale_posterior_calibrated=False,
                pose_uncertainty_marginalized=False,external_occlusion_modeled=False,candidate_navigation_checked=False,
                observation_resolution_or_noise_gain_modeled=False,ground_truth_used=False,future_observation_used=False,
                class_importance_weights_used=False,tsdf_quality_measured=False,
                exposure_version=EXPOSURE_VERSION,
                metric='posterior expected nominal exterior area eligible after same-center paid exposure de-duplication',
                measured_support_rule_unchanged=True,exposure_exclusion_is_not_measured_surface=True,actual_depth_and_tsdf_modified=False,
                limitation='common upright label mount and nominal shape family; external occlusion, pose uncertainty, and repeated-view accuracy gain are not modeled; same-center attempted exposure is only an eligibility proxy; missing-depth recovery is not predicted')
            require(set(forecast)==set(expected)|{'candidates'},'strict forecast fields; no hidden private or method override')
            exact_or_numeric({k:v for k,v in forecast.items() if k!='candidates'},expected,'forecast_header')
            require(1<=len(forecast['candidates'])<=256 and len({r['view_id'] for r in forecast['candidates']})==len(forecast['candidates']),
                'bounded unique candidates')
            for row in forecast['candidates']:
                candidate=verify_candidate_arithmetic(row,probabilities,self.legacy._paid_cameras,reason,self.legacy.bank)
                if self.graph is not None:
                    node,heading=row['view_id'].rsplit(':',1);heading=int(heading)
                    require(node in self.graph.positions and 0<=heading<12 and row['view_id']==f'{node}:{heading}',
                        'candidate lies on public navigation graph')
                    yaw=heading*math.pi/6.;transform=np.eye(4)
                    transform[:3,:3]=[[math.sin(yaw),0.,math.cos(yaw)],[-math.cos(yaw),0.,math.sin(yaw)],[0.,-1.,0.]]
                    transform[:3,3]=[*self.graph.positions[node],self.graph.camera_height_m]
                    paid=self.legacy._paid_cameras[-1]
                    require(np.array_equal(candidate.world_from_camera,transform) and candidate.width==paid['width']
                        and candidate.height==paid['height'] and np.array_equal(candidate.intrinsic,paid['intrinsic']),
                        'candidate camera derived from graph pose and actual paid calibration')
                for component,(_,weights,_) in zip(row['components'],self.legacy._samples):
                    require(component['sampled_surface_points']==len(weights)
                        and abs(component['total_prototype_area_m2']-float(weights.sum()))<=1e-12,
                        'declared sampling count and total area match shared public quadrature')
                recompute=reason is None and (full or row['view_id']==target_id)
                if recompute:
                    independently=independent_components(self.legacy,state,support,candidate,self.legacy._paid_cameras)
                    exact_or_numeric(row['components'],independently,'independent_exposure_geometry')
                    self.geometry_candidates+=1
                self.all_candidates+=1
                self.rows.append(dict(paid_step=evidence['paid_step'],instance_id=key,view_id=row['view_id'],
                    all_provenance_and_arithmetic_checked=True,independent_geometry_recomputed=recompute,
                    sampling_reason='first reliable replan' if recompute and full else 'selected target' if recompute else 'arithmetic only',
                    fallback_reason=reason,expected_eligible_area_m2=row['expected_new_surface_area_m2'],
                    component_excluded_sum_m2=sum(c['excluded_previously_exposed_unseen_area_m2'] for c in row['components'])))


def sha(path):return base.sha(path)
def read(path):return base.read(path)
def require(value,message):
    if not value:raise ValueError(message)


def array_sha(value):
    value=np.asarray(value)
    header=json.dumps(dict(dtype=value.dtype.str,shape=list(value.shape)),sort_keys=True,separators=(',',':')).encode()+b'\n'
    return hashlib.sha256(header+np.ascontiguousarray(value).tobytes()).hexdigest()


def exact_or_numeric(actual,expected,path='receipt',*,atol=1e-12):
    """Exact discrete fields; bounded scalar roundoff, never altered gates."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),path+': dictionary keys')
        for key,value in expected.items():exact_or_numeric(actual[key],value,path+'/'+key,atol=atol)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),path+': list length')
        for index,(a,b) in enumerate(zip(actual,expected)):exact_or_numeric(a,b,path+'/'+str(index),atol=atol)
    elif type(expected) is float:
        require(type(actual) is float and math.isfinite(actual) and abs(actual-expected)<=atol,path+': finite float')
    else:
        require(type(actual) is type(expected) and actual==expected,path+': exact value')


def verify_pairing(protocol,baseline):
    """Pure independent protocol comparison; no runner validator imported."""
    require(protocol.get('schema')==SCHEMA and protocol.get('status')=='frozen','strict frozen exposure protocol schema')
    require(protocol.get('phase')=='ablation','this reviewer is for declared front-end ablation only')
    require(protocol.get('frozen_before_world_creation') is True,'pre-world freeze required')
    require(baseline.get('schema')=='article.experiment_protocol.ground_v2' and baseline.get('status')=='frozen'
            and baseline.get('phase')=='ablation','original GroundV2 baseline schema')
    require(protocol.get('metric_preprocessing')==dict(schema=PREPROCESSING_SCHEMA,
        applied_equally_to_paired_baseline=True),'explicit common derived preprocessing version')
    require(protocol['evaluation']==baseline['evaluation']==EVALUATION,'common frozen evaluation settings')
    for key in ('asset_root','asset_manifest_sha256','mapper','references','maximum_elapsed_s','maximum_episode_bytes','reserve_bytes'):
        require(protocol[key]==baseline[key],'paired common '+key)
    common=dict(protocol['controller'])
    require(common.pop('same_center_exposure_dedup',None) is True,'common exposure switch required')
    require(common.get('ground_association') is True,'unchanged common ground switch required')
    require(common==baseline['controller'],'only exposure deduplication changes controller options')
    slots=protocol['slots'];expected={(f'ART1_{family}_DEV',method) for family in ('AISLE','CELL','LOOP') for method in METHODS}
    require(len(slots)==12 and {(s['scene_id'],s['method']) for s in slots.values()}==expected,'complete declared twelve-cell matrix')
    paired=[]
    for slot in slots.values():
        require(set(slot)=={'scene_id','method','budget','noise_seed','paired_baseline_run_id'},'exact paired slot fields; no override')
        require(type(slot['budget']) is int and slot['budget']==160 and type(slot['noise_seed']) is int
                and slot['noise_seed']==92801,'fixed paid budget and noise seed')
        key=slot['paired_baseline_run_id'];paired.append(key)
        require(isinstance(key,str) and key in baseline['slots'],'unknown paired baseline run')
        target=baseline['slots'][key]
        require(set(target)=={'scene_id','method','budget','noise_seed','paired_baseline_run_id'},'baseline has no per-slot override')
        require(all(slot[k]==target[k] for k in ('scene_id','method','budget','noise_seed')),'paired method/scene/budget/noise')
    require(len(set(paired))==12 and set(paired)==set(baseline['slots']),'one-to-one full baseline pairing')
    runtime=protocol.get('numerical_runtime')
    require(isinstance(runtime,dict) and all(runtime.get(k) for k in
        ('python_executable','python_version','numpy_version','numpy_path','numpy_build','scipy_version','scipy_path','scipy_build')),
        'complete declared numerical runtime build and path fingerprint')
    require(runtime.get('thread_environment')==dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        'three single-thread numerical controls')
    require(runtime==baseline.get('numerical_runtime'),'paired numerical backend identical')
    return True


def verify_preprocessing_arrays(vertices,triangles,receipt):
    """Independent face gate and SHA arithmetic, without calling the adapter."""
    v=np.asarray(vertices);t=np.asarray(triangles)
    require(v.ndim==2 and v.shape[1]==3 and v.dtype.kind in 'fiu' and len(v)<=2000000
        and np.isfinite(v).all(),'bounded finite raw vertices')
    require(t.ndim==2 and t.shape[1]==3 and t.dtype.kind in 'iu' and len(t)<=1000000,'bounded integer raw faces')
    require(not t.size or (t.min()>=0 and t.max()<len(v)),'raw face indices in range')
    xyz=np.asarray(v,dtype=np.float64)[t]
    with np.errstate(over='ignore',invalid='ignore'):
        twice=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    require(np.isfinite(twice).all(),'nonfinite face geometry rejected, not removed')
    area=twice*.5;remove=np.flatnonzero(twice<=1e-12);keep=np.flatnonzero(twice>1e-12)
    retained=t[keep];removed_xyz=xyz[remove]
    edge=np.linalg.norm(removed_xyz-np.roll(removed_xyz,1,axis=1),axis=2)
    expected=dict(schema=PREPROCESSING_SCHEMA,status='geometry_prepared_no_quality_evaluation',
        fixed_maximum_removed_face_area_m2=5e-13,equivalent_frozen_v40_rule='reject twice_area <= 1e-12',
        original_vertices=len(v),original_faces=len(t),retained_faces=len(keep),removed_faces=len(remove),
        removed_original_face_indices=remove.tolist(),removed_total_area_m2=float(area[remove].sum()),
        removed_maximum_face_area_m2=float(area[remove].max()) if len(remove) else 0.,
        removed_maximum_edge_m=float(edge.max()) if edge.size else 0.,
        original_total_area_m2=float(area.sum()),retained_total_area_m2=float(area[keep].sum()),
        original_vertices_array_sha256=array_sha(v),original_triangles_array_sha256=array_sha(t),
        adapted_vertices_array_sha256=array_sha(v),adapted_triangles_array_sha256=array_sha(retained),
        retained_original_face_indices_sha256=array_sha(keep),
        original_vertices_values_order_and_dtype_unchanged=True,retained_faces_values_order_and_dtype_unchanged=True,
        prediction_arrays_exactly_unchanged=not len(remove),gt_reference_read=False,semantic_labels_read=False,roi_applied=False,
        new_surface_evaluations=0,new_tsdf_integrations=0,new_worlds=0,
        quality_reuse_eligibility='requires independent equality of prediction, reference and all evaluator settings' if not len(remove)
            else 'new derived measurement required; cannot reuse original score',
        sampling_warning='Removing a face may shift later random samples in the frozen evaluator; no metric-equivalence claim is made.')
    allowed=set(expected)|{'source_prediction_sha256','original_prediction_seal_sha256','comparison_rule'}
    require(set(expected)<=set(receipt)<=allowed,'strict adapter receipt fields')
    # Exact masks, counts, threshold and hash bindings; small aggregate sums
    # can differ only by ordinary arithmetic roundoff, not face eligibility.
    for key,value in expected.items():
        tolerance=(0. if key=='fixed_maximum_removed_face_area_m2' else
            max(abs(value)*1e-12,1e-30) if type(value) is float else 0.)
        exact_or_numeric(receipt[key],value,'preprocessing/'+key,atol=tolerance)
    return dict(raw_vertices=len(v),raw_faces=len(t),removed_faces=len(remove),retained_faces=len(keep),
        removed_original_face_indices=remove.tolist(),removed_total_area_m2=float(area[remove].sum()),
        adapted_vertices_array_sha256=array_sha(v),adapted_triangles_array_sha256=array_sha(retained))


def verify_evaluation_metadata(evaluation,protocol,receipt_sha):
    require(evaluation.get('schema')=='article.exposure_v3.canonical_evaluation.v1'
        and evaluation.get('metric_version')==METRIC_VERSION,'explicit common derived metric version')
    require(evaluation.get('evaluation_settings')==protocol['evaluation']==EVALUATION,'evaluation actual/settings declaration')
    expected_pins={name:protocol['source_sha256'][name] for name in
        ('nso/surface_evaluation_v40.py','nso/article_prediction_mesh_adapter_v1.py')}
    require(evaluation.get('evaluation_source_sha256')==expected_pins,'derived evaluator and adapter source pins')
    require(evaluation.get('metric_preprocessing')==dict(schema=PREPROCESSING_SCHEMA,
        receipt='prediction_preprocessing.json',receipt_sha256=receipt_sha),'derived preprocessing artifact binding')
    require(evaluation.get('original_prediction_files_unchanged') is True and evaluation.get('no_roi_crop') is True
        and evaluation.get('semantic_weights_used') is False and evaluation.get('fixed_target_instances')==4,
        'unchanged whole prediction, no ROI/semantic weighting, four-instance denominator')


def verify_archive(protocol,checks,label):
    pins=protocol['source_sha256'];archive=base.safe_file(ROOT,protocol['source_archive'])
    checks.require(sha(archive)==protocol['source_archive_sha256'],label+' source archive SHA')
    with zipfile.ZipFile(archive) as stream:
        checks.require(len(stream.namelist())==len(set(stream.namelist())) and set(stream.namelist())==set(pins),label+' complete source closure')
        for name,pin in pins.items():
            checks.require(hashlib.sha256(stream.read(name)).hexdigest()==pin,label+' archived source '+name)
            path=base.safe_file(ROOT,name)
            checks.require(path.is_file() and sha(path)==pin,label+' live source '+name)


def verify_source_relationship(current,baseline):
    overlap=set(current)&set(baseline)
    require(all(current[name]==baseline[name] for name in overlap)
        and set(baseline)-set(current)<={'scripts/run_article_ground_experiment_20260928.py'},
        'shared execution sources unchanged; only replaced Ground CLI may leave new dependency closure')


def verify_registry(phase,protocol_sha,entry,checks):
    ledger_path=phase/'start_ledger.json';path=phase.parent/'execution_registry.json';registry=read(path)
    checks.require(registry['schema']=='article.execution_registry.v1' and registry['phase_caps']==CAPS,'shared cumulative registry schema/caps')
    counts=Counter(r['phase'] for r in registry['entries'])
    checks.require(all(k in CAPS and v<=CAPS[k] for k,v in counts.items()),'cumulative starts obey all phase caps')
    matches=[r for r in registry['entries'] if r['run_id']==entry['run_id'] and r['ledger']==str(ledger_path.resolve())]
    checks.require(len(matches)==1 and matches[0]['phase']=='ablation' and
        matches[0]['protocol_sha256']==protocol_sha,'this attempt consumes one global ablation slot')
    identities=[(r['ledger'],r['run_id']) for r in registry['entries']]
    checks.require(len(set(identities))==len(identities),'no duplicated global reservation identity')
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=sha(path),counts=dict(counts),phase_caps=CAPS,total_reserved_attempts=len(identities))


def paired_configuration(baseline_phase,entry,checks):
    """Use the original closed Ground seal; retain failures without fabrication."""
    episode=baseline_phase/'episodes'/entry['run_id']
    if (episode/'artifact_manifest.json').is_file():
        checks.require(sha(episode/'artifact_manifest.json')==entry['artifact_manifest_sha256'],'baseline closed manifest seal')
        row=read(episode/'artifact_manifest.json')['files']['controller_final.json'];path=episode/'controller_final.json'
        checks.require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'baseline controller configuration original seal')
        return read(path)['configuration'],dict(kind='original_complete_artifact_seal',
            manifest_sha256=sha(episode/'artifact_manifest.json'),controller_final_sha256=sha(path))
    checks.require(entry['status']=='attempt_failed','unsealed baseline explicitly remains failed')
    failure=episode/'attempt_failure.json'
    if not failure.is_file():failure=baseline_phase/'failures'/(entry['run_id']+'.json')
    checks.require(failure.is_file() and sha(failure)==entry['result_sha256'],'paired Ground failure retained and ledger-bound')
    return None,dict(kind='terminal_failure_without_complete_configuration_seal',original_status='attempt_failed',
        original_qualified=False,failure_sha256=sha(failure),actual_paired_configuration_comparable=False,
        limitation='No forged controller snapshot; new-arm configuration still verified against frozen common protocol, but paired actual configuration unavailable.')


def review(episode,output,*,protocol_path=None):
    episode=Path(episode).resolve();output=Path(output).resolve();phase=episode.parent.parent
    require(output!=episode and not output.is_relative_to(episode),'review output outside episode')
    ledger_path=phase/'start_ledger.json'
    if not ledger_path.exists():return dict(status='unstarted',run_id=episode.name,reason='no ledger reservation',new_worlds=0)
    ledger=read(ledger_path);matches=[r for r in ledger['entries'] if r['run_id']==episode.name]
    if not matches:return dict(status='unstarted',run_id=episode.name,reason='no ledger reservation',new_worlds=0)
    require(len(matches)==1,'exactly one ledger row')
    entry=matches[0]
    if entry['status']=='reserved':return dict(status='pending',run_id=episode.name,reason='active reservation; no incomplete artifacts inspected',new_worlds=0)
    require(not output.exists(),'exclusive review output')
    start=time.monotonic();checks=base.Checks();before=base.runtime_counts_v41();base_report=None;manifest_sha=None;protocol_sha=None
    summary=dict(schema='article.saved_episode_review.exposure_v3',run_id=episode.name,online_status=entry['status'],
        new_worlds=0,new_sensor_queries=0,new_policy_runs=0,new_tsdf_integrations=0,new_surface_evaluations=0,
        scope='All saved physics/ground/camera/source/area-arithmetic receipts; explicitly sampled independent exposure geometry; no new quality measurement')
    ground_rows=[];exposure_audit=None
    try:
        archived=episode/'protocol.json'
        selected=archived if archived.is_file() else Path(protocol_path) if protocol_path else None
        checks.require(selected is not None and selected.is_file(),'known protocol even for terminal failure')
        protocol=read(selected);protocol_sha=sha(selected)
        checks.require(protocol_sha==ledger['protocol_sha256'],'protocol bound to episode ledger')
        if protocol_path is not None:checks.require(sha(protocol_path)==protocol_sha,'external and archived protocol SHA')
        pin=protocol['paired_baseline_protocol']
        checks.require(set(pin)=={'path','sha256'},'strict nested paired baseline protocol pin')
        baseline_path=base.safe_file(ROOT,pin['path'])
        checks.require(sha(baseline_path)==pin['sha256'],'original baseline protocol SHA')
        baseline=read(baseline_path);verify_pairing(protocol,baseline);checks.check(True,'independent single-component paired design')
        checks.require(ledger['schema']=='article.start_ledger.v1' and ledger['phase']=='ablation' and ledger['slots']==protocol['slots'],
                       'ledger phase and complete slot binding')
        checks.require(len(ledger['entries'])<=12 and len({r['run_id'] for r in ledger['entries']})==len(ledger['entries'])
            and all(r['run_id'] in protocol['slots'] for r in ledger['entries']),'twelve named starts, no undeclared retry')
        slot=protocol['slots'][episode.name]
        checks.require((ROOT/protocol['output_relative_path']).resolve()==phase,'episode belongs to declared phase output')
        verify_source_relationship(protocol['source_sha256'],baseline['source_sha256'])
        checks.check(True,'shared execution sources unchanged; Ground CLI separately preserved in baseline seal')
        checks.require({'nso/article_experiment_exposure_v3.py','nso/controller_article_exposure_v3.py',
            EXPOSURE_SOURCE,'nso/controller_article_ground_v2.py','nso/observed_instances_article_ground_v2.py',
            'nso/article_prediction_mesh_adapter_v1.py','scripts/run_article_exposure_experiment_20260928.py'}<=set(protocol['source_sha256']),
            'all new execution modules included in source closure')
        verify_archive(protocol,checks,'exposure');verify_archive(baseline,checks,'Ground baseline')
        summary['source_archive_sha256']=protocol['source_archive_sha256']
        summary['baseline_source_archive_sha256']=baseline['source_archive_sha256']
        summary['cumulative_registry']=verify_registry(phase,protocol_sha,entry,checks)
        baseline_phase=base.safe_file(ROOT,baseline['output_relative_path'])
        baseline_ledger=read(baseline_phase/'start_ledger.json')
        checks.require(baseline_ledger['protocol_sha256']==pin['sha256'],'baseline ledger protocol binding')
        pairs=[r for r in baseline_ledger['entries'] if r['run_id']==slot['paired_baseline_run_id']]
        checks.require(len(pairs)==1 and pairs[0]['status']!='reserved','known terminal baseline attempt; no outcome eligibility filter')
        summary['paired_baseline']=dict(protocol=pin,run_id=slot['paired_baseline_run_id'],online_status=pairs[0]['status'],
            quality_comparison_requires_common_derived_evaluation=True,baseline_failure_retained=pairs[0]['status']=='attempt_failed')
        if not (episode/'artifact_manifest.json').exists():
            failure=episode/'attempt_failure.json'
            if not failure.is_file():failure=phase/'failures'/(episode.name+'.json')
            checks.require(failure.is_file() and sha(failure)==entry['result_sha256'],'terminal failure retained and bound')
            summary.update(status='failed_attempt_preserved',qualified=False,failure=read(failure),
                limitation='No complete artifact manifest; no incomplete per-frame analysis or quality claim.')
        else:
            manifest_sha=sha(episode/'artifact_manifest.json')
            checks.require(manifest_sha==entry['artifact_manifest_sha256'],'terminal complete artifact seal')
            # The original reader is actually executed. Its detailed compatible
            # checks remain visible in a separately hashed nested report.
            base_report=base.review(episode,output/'compatible_v1_receipts')
            checks.require(base_report['status']=='reviewed' and base_report['all_checks_passed'],
                'original physics/packet/map/route reader passes this new episode',base_report.get('failures'))
            manifest=read(episode/'artifact_manifest.json');started=read(episode/'started.json')
            checks.require(manifest['schema']=='article.episode_artifacts.v1','compatible artifact inventory schema')
            checks.require(started['numerical_runtime']==protocol['numerical_runtime'],'actual numerical backend bound before execution')
            expected_pair=dict(protocol_relative_path=pin['path'],protocol_sha256=pin['sha256'],original_sources_preserved=True,
                old_scores_not_overwritten=True,comparison_requires_canonical_metric_on_both_arms=True,
                baseline_version='article_ground_v2',sole_common_component_change='same_center_exposure_dedup')
            checks.require(started['paired_baseline']==expected_pair,'started paired comparison declaration')
            final=read(episode/'controller_final.json');configuration=final['configuration']
            exposure_audit=ExposureAudit(protocol['controller'],base.PublicPrimitiveGraphV41(read(episode/'public_graph.json')))
            checks.require(exposure_audit.configuration['source_sha256']==protocol['source_sha256'][EXPOSURE_SOURCE],
                'actual exposure predictor source pin')
            digest=hashlib.sha256(json.dumps(configuration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            checks.require(digest==final['configuration_sha256'],'actual controller configuration digest')
            original_configuration,provenance=paired_configuration(baseline_phase,pairs[0],checks)
            verify_actual_configuration(configuration,original_configuration,protocol,exposure_audit.configuration)
            checks.check(True,'whole actual configuration and common single exposure component')
            summary['paired_baseline']['configuration_provenance']=provenance
            count=base_report['rgbd_frames'];ground_counts=Counter()
            for i in range(count):
                with np.load(episode/f'packets/{i:03d}_rgbd.npz',allow_pickle=False) as raw:
                    checks.require(set(raw.files)==set(base.PaidRGBDObservationV40.__dataclass_fields__),
                        'strict saved public RGBD field whitelist:'+str(i))
                sensor,_=base.packet_at(episode,i)
                log=json.loads(gzip.decompress((episode/f'steps/{i:03d}.json.gz').read_bytes()))
                evidence=log['controller_evidence'];decision=log['decision']
                checks.require(decision['configuration_sha256']==evidence['configuration_sha256']==final['configuration_sha256'],
                    'per-frame unchanged actual controller configuration:'+str(i))
                ground=evidence['association']['article_ground_association']
                _,computed=ground_mask_from_observation(sensor.rgbd)
                exact_or_numeric(ground,computed,'current_paid_ground/'+str(i))
                checks.check(True,'current raw paid packet recomputes exact ground mask/eligibility:'+str(i))
                ground_counts[ground['reason']]+=1
                ground_rows.append(dict(paid_step=i,accepted=ground['accepted'],reason=ground['reason'],
                    excluded_pixels=ground['excluded_pixels'],observation_sha256=ground['observation_sha256'],mask_sha256=ground['mask_sha256']))
                exposure_audit.observe(sensor.rgbd,evidence)
                checks.check(True,'actual paid pixel support and marker-plane history:'+str(i))
                if decision['global_replanned']:
                    exposure_audit.verify_forecasts(evidence,decision['global_selection'])
                    checks.check(True,'all forecast provenance/area arithmetic and declared geometric subset:'+str(i))
            evaluation=read(episode/'evaluation.json');receipt_path=episode/'prediction_preprocessing.json'
            verify_evaluation_metadata(evaluation,protocol,sha(receipt_path))
            checks.check(True,'derived metric version/settings/source/preprocessing links')
            seal=read(episode/'prediction_seal.json');receipt=read(receipt_path)
            checks.require(receipt['source_prediction_sha256']==seal==evaluation['source_prediction_sha256'],'raw sealed array file pins')
            checks.require(receipt['original_prediction_seal_sha256']==sha(episode/'prediction_seal.json'),'original prediction seal provenance')
            checks.require(receipt['comparison_rule']=='same numeric-face adapter and frozen metric on both paired arms','common metric declaration')
            with np.load(episode/'prediction/mesh.npz',allow_pickle=False) as arrays:
                preprocessing=verify_preprocessing_arrays(arrays['vertices'],arrays['triangles'],receipt)
            checks.check(True,'independent raw-array geometry threshold/removed-index/hash audit')
            checks.require(sha(episode/'artifact_manifest.json')==manifest_sha,'episode seal unchanged during combined review')
            summary.update(status='reviewed',qualified=base_report['qualified'],method=slot['method'],scene_id=slot['scene_id'],
                phase='ablation',metrics=evaluation['metrics'],metric_version=METRIC_VERSION,
                metric_settings=protocol['evaluation'],ground_reason_counts=dict(ground_counts),mesh_preprocessing=preprocessing,
                exposure_audit=dict(scope=GEOMETRY_SCOPE,all_candidates=exposure_audit.all_candidates,
                    independent_geometry_candidates=exposure_audit.geometry_candidates,
                    first_geometry_replan=exposure_audit.first_geometry_replan,paid_support_plane_frames=exposure_audit.steps,
                    all_candidates_geometrically_recomputed=exposure_audit.all_candidates==exposure_audit.geometry_candidates),
                compatible_reader_checks=base_report['checks'],physics_metrics={k:base_report[k] for k in
                    ('paid_actions','budget','translation_m','action_counts','rgbd_frames','collisions','returned_xy_and_yaw')})
    except Exception as exc:
        checks.check(False,'exposure review execution completed',dict(type=type(exc).__name__,message=str(exc)))
        summary.update(status='review_error',qualified=False,error=dict(type=type(exc).__name__,message=str(exc)))
    checks.check(base.runtime_counts_v41()==before,'no World or paid action created by review')
    summary.update(elapsed_s=time.monotonic()-start,checks=checks.count,all_checks_passed=not checks.failures,failures=checks.failures,
        input_manifest_sha256=manifest_sha,protocol_sha256=protocol_sha,
        reviewer_source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__),
            str(Path(base.__file__).relative_to(ROOT)):sha(base.__file__),
            str(Path(ground_review.__file__).relative_to(ROOT)):sha(ground_review.__file__)})
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('review.json',summary),('ground_receipts.json',ground_rows),
        ('exposure_candidate_checks.json',[] if exposure_audit is None else exposure_audit.rows)]:
        with (output/name).open('xb') as stream:stream.write(base.canonical(value))
    files={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in output.rglob('*') if p.is_file()}
    with (output/'manifest.json').open('xb') as stream:
        stream.write(base.canonical(dict(schema='article.exposure_episode_review_manifest.v3',files=files,
            input_episode_manifest_sha256=manifest_sha,protocol_sha256=protocol_sha,
            reviewer_source_sha256=summary['reviewer_source_sha256'],new_worlds=0,new_surface_evaluations=0,new_tsdf_integrations=0)))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--protocol',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'audit_results/article_stage_20260928/episode_reviews_exposure_v3'/args.episode.name
    result=review(args.episode,output,protocol_path=args.protocol)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    if result['status'] not in ('pending','unstarted','reviewed','failed_attempt_preserved') or result.get('all_checks_passed') is False:
        raise SystemExit(1)
