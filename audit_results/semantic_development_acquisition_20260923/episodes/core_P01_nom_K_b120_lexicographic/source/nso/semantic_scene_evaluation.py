"""Pinned static references and whole-prediction scoring for new scene assets.

The asset binding is new; the V40 surface and V44 C_nav arithmetic is reused
unchanged. Private geometry is confined to this offline evaluator. No sensor,
World, mapper, controller, route or observed-instance selection is constructed.
"""
from pathlib import Path
import re

import numpy as np

from nso.offline_evaluation_v44 import (
    REFERENCE_ARRAYS, REFERENCE_FILES, _array_sha, _json, _write_json,
    conservative_floor_domain_v44, measure_navigation_coverage_v44, sha256,
)
from nso.semantic_scene_assets import load_semantic_scene_asset, parse_asset_id
from nso.surface_evaluation_v40 import (
    CandidateViewV40, ReferenceSurfaceV40, evaluate_surface_v40, freeze_reference_v40,
)
from scripts.verify_development_surface_pipeline_v40 import reachable_candidates


ROOT=Path(__file__).resolve().parents[1]
EVALUATION_PARAMETERS=dict(threshold_m=.05,sample_spacing_m=.3,seed=4002,max_samples=1000000)
SOURCE_FILES=('nso/semantic_scene_evaluation.py','nso/semantic_scene_assets.py',
    'nso/offline_evaluation_v44.py','nso/surface_evaluation_v40.py',
    'scripts/verify_development_surface_pipeline_v40.py')
PREDICTION_FILES=('mapper.json','occupancy.npz','mesh.npz')


def _pin(value, name):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('explicit SHA256 pin required: '+name)


def _source_hashes():
    return {name:sha256(ROOT/name) for name in SOURCE_FILES}


def _asset(asset_id,asset_root,asset_manifest_sha256):
    parse_asset_id(asset_id)
    return load_semantic_scene_asset(Path(asset_root)/asset_id,
        expected_manifest_sha256=asset_manifest_sha256)


def prepare_semantic_scene_reference(asset_id,output,*,asset_root,asset_manifest_sha256):
    """Freeze a route-independent reference; fail if any facility is invisible."""
    asset=_asset(asset_id,asset_root,asset_manifest_sha256)
    arrays,metadata=asset['arrays'],asset['metadata']
    target_ids=sorted(row['instance_id'] for row in metadata['private_instances'])
    owners=arrays['triangle_instance_id']
    if target_ids!=sorted(set(owners.tolist())-{-1}) or len(set(target_ids))!=len(target_ids):
        raise ValueError('complete unique fixed target inventory required')
    sources=_source_hashes()
    # Both original functions retain their original conservative AABB rule,
    # start position, radius, spacings, view headings and visibility algorithm.
    views,candidates=reachable_candidates(metadata,{'public_defaults':asset['public_spec']})
    reference=freeze_reference_v40(arrays['vertices'],arrays['triangles'],owners,views,
        sample_spacing_m=.3,seed=4001,max_samples=50000)
    surface=reference.manifest()
    if surface['target_instances']!=target_ids:
        raise ValueError('a facility cannot be removed from the macro denominator')
    domain,coverage=conservative_floor_domain_v44(metadata)
    if sources!=_source_hashes():
        raise ValueError('reference generator sources changed during preparation')
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'surface.npz',**{key:getattr(reference,key) for key in REFERENCE_ARRAYS})
    np.savez_compressed(output/'coverage_domain.npz',domain=domain)
    _write_json(output/'candidate_views.json',candidates)
    record=dict(schema='semantic_scene.offline_reference.v1',asset_id=asset_id,
        parent_id=metadata['parent_id'],condition=metadata['condition'],
        asset_manifest_sha256=asset_manifest_sha256,source_sha256=sources,
        surface=surface,coverage=coverage,public_workspace=asset['public_workspace'],
        target_instance_inventory=target_ids,evaluation_parameters=dict(EVALUATION_PARAMETERS),
        reference_sampling=dict(sample_spacing_m=.3,seed=4001,max_samples=50000,
            candidate_lattice_spacing_m=1.5,yaw_degrees=[0,90,180,270]),
        old_v40_v44_math_unchanged=True,development_only=True,
        formal_performance_evidence=False,primary_matrix_started=False,
        prediction_selection_or_trajectory_used=False,new_worlds=0,new_sensor_packets=0,
        new_trajectories=0,new_tsdf_integrations=0,
        predeclaration_requirement='Pin this manifest before corresponding policy episodes.',
        limitation='Inherited finite area quadrature, sparse 1.5 m visibility lattice and conservative AABB C_nav; this is not a sensor episode or formal benchmark result.')
    _write_json(output/'reference.json',record)
    _write_json(output/'manifest.json',dict(schema='semantic_scene.offline_reference_manifest.v1',
        asset_id=asset_id,asset_manifest_sha256=asset_manifest_sha256,
        files={name:dict(sha256=sha256(output/name),bytes=(output/name).stat().st_size)
               for name in sorted(REFERENCE_FILES)}))
    return dict(asset_id=asset_id,output=str(output),manifest_sha256=sha256(output/'manifest.json'),
        asset_manifest_sha256=asset_manifest_sha256,target_instances=len(target_ids),
        observable_samples=len(reference.points),candidate_views=len(views),coverage=coverage,
        reference_fingerprint=reference.fingerprint,new_worlds=0,new_sensor_packets=0,
        new_tsdf_integrations=0,formal_performance_evidence=False)


def load_semantic_scene_reference(root,*,manifest_sha256,asset_id,asset_root,asset_manifest_sha256):
    """Verify reference/asset/source identity before returning immutable inputs."""
    _pin(manifest_sha256,'reference manifest')
    root=Path(root).resolve()
    manifest_path=root/'manifest.json'
    if manifest_path.is_symlink() or sha256(manifest_path)!=manifest_sha256:
        raise ValueError('reference manifest differs from explicit pin')
    manifest=_json(manifest_path)
    if (manifest.get('schema')!='semantic_scene.offline_reference_manifest.v1'
            or manifest.get('asset_id')!=asset_id or manifest.get('asset_manifest_sha256')!=asset_manifest_sha256
            or set(manifest.get('files',{}))!=REFERENCE_FILES):
        raise ValueError('reference asset binding or exact file inventory differs')
    for name,row in manifest['files'].items():
        path=root/name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size!=row['bytes']
                or sha256(path)!=row['sha256']):
            raise ValueError('reference artifact changed: '+name)
    asset=_asset(asset_id,asset_root,asset_manifest_sha256)
    record=_json(root/'reference.json')
    target_ids=sorted(row['instance_id'] for row in asset['metadata']['private_instances'])
    if (record.get('schema')!='semantic_scene.offline_reference.v1'
            or record.get('asset_id')!=asset_id or record.get('asset_manifest_sha256')!=asset_manifest_sha256
            or record.get('source_sha256')!=_source_hashes()
            or record.get('public_workspace')!=asset['public_workspace']
            or record.get('target_instance_inventory')!=target_ids
            or record.get('evaluation_parameters')!=EVALUATION_PARAMETERS):
        raise ValueError('reference source, workspace, inventory or metric binding differs')
    candidates=_json(root/'candidate_views.json')
    views=tuple(CandidateViewV40(**row) for row in candidates['candidate_views'])
    with np.load(root/'surface.npz',allow_pickle=False) as data:
        if set(data.files)!=set(REFERENCE_ARRAYS):
            raise ValueError('complete reference surface arrays required')
        arrays={key:data[key].copy() for key in REFERENCE_ARRAYS}
    surface=record['surface']
    reference=ReferenceSurfaceV40(**arrays,candidate_views=views,
        sample_spacing_m=surface['sample_spacing_m'],seed=surface['seed'],
        full_target_sample_count=surface['full_target_sample_count'],fingerprint=surface['fingerprint'])
    if reference.manifest()!=surface or surface['target_instances']!=target_ids:
        raise ValueError('reference surface content or complete facility denominator differs')
    with np.load(root/'coverage_domain.npz',allow_pickle=False) as data:
        if data.files!=['domain']:
            raise ValueError('coverage domain inventory differs')
        domain=data['domain'].copy()
    measure_navigation_coverage_v44(np.full(domain.shape,-1,dtype=np.int8),domain,record['coverage'])
    return reference,domain,record


def evaluate_semantic_scene_prediction(prediction_root,reference,domain,record,*,
        expected_prediction_sha256,expected_frames,public_workspace):
    """Score pinned whole saved mesh/occupancy, without declaring episode success.

    The caller must inspect the episode protocol, source archive, counters and
    durable ledger first. This layer independently rechecks all three original
    prediction file pins and the complete mapper coordinate/history contract.
    """
    if (type(expected_frames) is not int or expected_frames<1
            or set(expected_prediction_sha256)!=set(PREDICTION_FILES)):
        raise ValueError('positive inspected frame count and complete prediction pins required')
    root=Path(prediction_root).resolve()
    for name,digest in expected_prediction_sha256.items():
        _pin(digest,'prediction '+name)
        path=root/name
        if path.is_symlink() or not path.is_file() or sha256(path)!=digest:
            raise ValueError('saved prediction differs from inspected artifact manifest: '+name)
    descriptor=record['coverage']
    if public_workspace!=record['public_workspace']:
        raise ValueError('saved episode and fixed reference workspaces differ')
    mapper=_json(root/'mapper.json')
    if (mapper.get('backend_poisoned') is not False or mapper.get('frames')!=expected_frames
            or any(mapper.get(key)!=descriptor[key] for key in
                   ('shape','resolution_m','origin_xy_m','grid_convention'))):
        raise ValueError('prediction mapper coordinates/history do not match reference')
    with np.load(root/'occupancy.npz',allow_pickle=False) as data:
        if set(data.files)!={'belief','observed'}:
            raise ValueError('complete original occupancy arrays required')
        belief,observed=data['belief'].copy(),data['observed'].copy()
    if (_array_sha(belief)!=mapper.get('occupancy_sha256') or observed.dtype!=np.bool_
            or observed.shape!=belief.shape or np.any(observed&(belief<0))):
        raise ValueError('saved occupancy disagrees with mapper or observation mask')
    coverage=measure_navigation_coverage_v44(belief,domain,descriptor)
    with np.load(root/'mesh.npz',allow_pickle=False) as data:
        if set(data.files)!={'vertices','triangles','vertex_colors'}:
            raise ValueError('complete original mesh required; no ROI or prediction-owner array accepted')
        vertices,triangles,colors=(data[key].copy() for key in ('vertices','triangles','vertex_colors'))
    if colors.shape!=vertices.shape or not np.isfinite(colors).all():
        raise ValueError('mesh colors must align with complete finite vertices')
    metrics=evaluate_surface_v40(reference,vertices,triangles,C_map=coverage['C_nav'],**EVALUATION_PARAMETERS)
    metrics.pop('C_map');metrics.pop('J')
    metrics.update(C_nav=coverage['C_nav'],J_nav=coverage['C_nav']*metrics['Q'])
    for name,digest in expected_prediction_sha256.items():
        if sha256(root/name)!=digest:
            raise ValueError('prediction changed during evaluation: '+name)
    return dict(metrics=metrics,coverage=coverage,input_prediction_sha256=dict(expected_prediction_sha256),
        all_task_instances_in_macro_denominator=True,prediction_roi_cropped=False,
        prediction_reintegrated=False,new_worlds=0,new_sensor_packets=0,new_tsdf_integrations=0,
        evaluation_source_sha256=sha256(Path(__file__)),formal_performance_evidence=False,
        independent_replay_performed_here=False)


def evaluate_semantic_scene_endpoint(episode,reference_root,*,reference_manifest_sha256,
        asset_root,asset_manifest_sha256):
    """Adapt an inspected new-runner episode; its loader remains upstream."""
    if episode.get('current_sources_match') is not True:
        raise ValueError('evaluation requires executed sources or isolated archived execution')
    root=Path(episode['root'])
    _pin(episode['manifest_sha256'],'episode manifest')
    if sha256(root/'artifact_manifest.json')!=episode['manifest_sha256']:
        raise ValueError('inspected episode manifest changed')
    slot=episode['slot']
    reference,domain,record=load_semantic_scene_reference(reference_root,
        manifest_sha256=reference_manifest_sha256,asset_id=slot['asset_id'],
        asset_root=asset_root,asset_manifest_sha256=asset_manifest_sha256)
    manifest=episode['manifest']
    workspace_path=root/'public_workspace.json'
    if sha256(workspace_path)!=manifest['files']['public_workspace.json']['sha256']:
        raise ValueError('inspected public workspace changed')
    result=evaluate_semantic_scene_prediction(root/'prediction',reference,domain,record,
        expected_prediction_sha256={name:manifest['files']['prediction/'+name]['sha256']
                                    for name in PREDICTION_FILES},
        expected_frames=episode['result']['acquired_and_saved_packets'],
        public_workspace=_json(workspace_path))
    result.update(schema='semantic_scene.saved_prediction_offline_evaluation.v1',
        asset_id=slot['asset_id'],asset_manifest_sha256=asset_manifest_sha256,
        episode_manifest_sha256=episode['manifest_sha256'],reference_manifest_sha256=reference_manifest_sha256,
        task_success=episode['result']['status']=='controller_stop'
            and episode['result']['sensor_status'].get('returned_xy_and_yaw') is True,
        original_episode_status=episode['result']['status'])
    return result
