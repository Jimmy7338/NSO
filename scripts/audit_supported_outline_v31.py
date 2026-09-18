#!/usr/bin/env python3
"""One frozen analytic applicability run, no physical observations or mapping."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import open3d as o3d
import shapely
from nso.research_evidence_v31 import ROOT,read,write,freeze,verify_sources,seal,sha
from nso.supported_outline_v31 import FullSupportedOutlineEvaluatorV31
from nso.facility_outline_v30 import FullInstanceOutlineEvaluatorV30
from nso.outline_fixtures_v31 import (MeshFixtureV31,reference_meshes_v31,geometry_fixtures_v31,
    association_fixtures_v31,translated_mesh_v31,empty_mesh_v31,mesh_from_faces_v31,BOX_BOUNDS)
from nso.box_union_geometry_v30 import BoxV30,RectFaceV30,union_exterior_faces_v30

OUT=ROOT/'audit_results/v31_supported_outline_20260917'
PROTOCOL=ROOT/'docs/research/V31_SUPPORTED_OUTLINE_PROTOCOL_20260917.md'
ARGS=dict(returned=True,collisions=0,failed=False,paid_actions=0,budget=0)


def no_acquisition_guards():
    from env.information_pixel_v30 import InformationPixelWorldV30
    from env.virtual3d import VirtualWorld
    from nso.mapping3d import SensorMapper
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    import env.canonical_rgbd_v15 as rgbd
    import env.canonical_box_scan_v14 as scan
    def forbidden(*args,**kwargs):raise AssertionError('new physical acquisition or mapping forbidden')
    for cls in (InformationPixelWorldV30,VirtualWorld,SensorMapper,ObservedRuntimeMapperV10):cls.__init__=forbidden
    rgbd.render_axial_depth=forbidden;scan.box_union_ranges=forbidden
    o3d.pipelines.integration.ScalableTSDFVolume=forbidden


def as_o3d(mesh):
    result=o3d.geometry.TriangleMesh()
    result.vertices=o3d.utility.Vector3dVector(np.asarray(mesh.vertices).copy())
    result.triangles=o3d.utility.Vector3iVector(np.asarray(mesh.triangles,np.int32).copy())
    return result


def asset(reference,identifier=0):
    vertices=np.asarray(reference.vertices)
    return dict(id=identifier,mesh=as_o3d(reference),bounds=[vertices.min(axis=0)-.1,vertices.max(axis=0)+.1])


def rotation(mesh):
    theta=np.deg2rad(37.)
    matrix=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
    return MeshFixtureV31(mesh.vertices@matrix.T+[4.25,-2.75,.3],mesh.triangles)


def run():
    if OUT.exists():raise FileExistsError('audit already started; no overwrite or implicit retry')
    OUT.mkdir();started=perf_counter();counts=dict(candidate_evaluations=0,legacy_evaluations=0,
        expected_invalid_rejections=0,worlds=0,sensor_packets=0,mapper_updates=0,TSDF_integrations=0,new_main_tasks=0)
    rows={};gates={}
    try:
        no_acquisition_guards()
        sources=[Path(__file__),PROTOCOL,ROOT/'docs/research/V31_ANALYTIC_FIXTURES_PROTOCOL_20260917.md',
            ROOT/'docs/research/V31_OUTLINE_CONTRACT_REVIEW_20260917.md',
            ROOT/'tests/virtual3d/test_supported_outline_v31.py',ROOT/'tests/virtual3d/test_outline_fixtures_v31.py']
        manifest=freeze(OUT,sources,status='frozen_before_formal_analytic_scoring',main_tasks_used=16,
            versions=dict(numpy=np.__version__,open3d=o3d.__version__,shapely=shapely.__version__),
            kernel_contract_tests_passed=6,fixture_geometry_tests_passed=8,
            kernel_test_projection_comparison_calls=1,physical_execution_forbidden=True)
        refs=reference_meshes_v31();fixtures=geometry_fixtures_v31()
        def evaluate(name,assets,meshes,seeds,track_count):
            assets=[dict(item,mesh=as_o3d(item['mesh'])) for item in assets]
            before=[(sha_arrays(m)) for m in meshes]
            candidate=FullSupportedOutlineEvaluatorV31(assets)
            counts['candidate_evaluations']+=1
            result=candidate.evaluate(meshes,seeds,track_count,1.,**ARGS)
            legacy=FullInstanceOutlineEvaluatorV30(assets)
            counts['legacy_evaluations']+=1
            old=legacy.evaluate(meshes,seeds,track_count,1.,**ARGS)
            if before!=[sha_arrays(m) for m in meshes]:raise AssertionError('prediction mutated')
            rows[name]=dict(candidate=result,legacy_full_instance=old,inputs_unchanged=True,
                input_mesh_hashes=before,qualification_fields_are_analytic_test_inputs=True)
        for name,fixture in fixtures.items():
            ref=refs[fixture.reference_name]
            seed=dict(observed_seed_xyz=np.asarray(ref.vertices).mean(axis=0),source='analytic_not_sensor')
            evaluate(name,[asset(ref)],[fixture.prediction],[seed],1)
        for name,pred in [('rotated_closed',refs['box']),('rotated_vertical',fixtures['box_vertical'].prediction)]:
            ref=rotation(refs['box']);seed=dict(observed_seed_xyz=ref.vertices.mean(axis=0))
            evaluate(name,[asset(ref)],[rotation(pred)],[seed],1)
        faces=[]
        for face in union_exterior_faces_v30([BoxV30(BOX_BOUNDS,0)],vertical_only=True):
            if face.axis==0 and face.sign==-1:
                bounds=list(face.bounds);bounds[2]=.001
                face=RectFaceV30(tuple(bounds),face.axis,face.sign,face.owner)
            faces.append(face)
        evaluate('one_mm_gap',[asset(refs['box'])],[mesh_from_faces_v31(faces)],
            [dict(observed_seed_xyz=[1.,0.,.8])],1)
        association=association_fixtures_v31()
        for name,case in association.items():
            evaluate(name,case['assets'],case['observed_meshes'],case['seeds'],case['observed_track_count'])
        present=association['both_present']
        evaluate('extra_unassigned_instance',present['assets'],
            (*present['observed_meshes'],translated_mesh_v31(refs['box'],(10.,0.,0.))),
            (*present['seeds'],dict(observed_seed_xyz=[11.,0.,.8])),3)
        evaluate('empty_extra_duplicate_seed',present['assets'],(*present['observed_meshes'],empty_mesh_v31()),
            (*present['seeds'],present['seeds'][0]),3)
        invalid=MeshFixtureV31(np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]]),np.array([[0,1,2]]))
        counts['candidate_evaluations']+=1
        try:
            FullSupportedOutlineEvaluatorV31([asset(refs['box'])]).evaluate([invalid],
                [dict(observed_seed_xyz=[1.,0.,.8])],1,1.,**ARGS)
        except ValueError as error:
            if 'zero-area 3D triangle' not in str(error):raise
            counts['expected_invalid_rejections']+=1
        counts['candidate_evaluations']+=1
        try:
            FullSupportedOutlineEvaluatorV31([asset(refs['box'])]).evaluate([refs['box'],invalid],
                [dict(observed_seed_xyz=[1.,0.,.8]),None],2,1.,**ARGS)
        except ValueError as error:
            if 'zero-area 3D triangle' not in str(error):raise
            counts['expected_invalid_rejections']+=1
        quality=lambda name:rows[name]['candidate']['05cm']['outline_macro_quality']
        positives=['box_closed','box_vertical','box_retriangulated','box_vertical_retriangulated',
            'box_duplicate','box_vertical_duplicate','l_closed','l_vertical','rotated_closed','rotated_vertical']
        for name in positives:gates[name+'_near_one']=quality(name)>=.99
        gates['equivalent_box_geometry_equal']=max(quality(n) for n in positives[:6])-min(quality(n) for n in positives[:6])<=1e-8
        gates['rotated_closed_vertical_equal']=abs(quality('rotated_closed')-quality('rotated_vertical'))<=1e-8
        gates['l_closed_vertical_equal']=abs(quality('l_closed')-quality('l_vertical'))<=1e-8
        gates['thin_walls_near_one']=quality('box_four_thin_walls')>=.98
        negatives=['box_vertical_missing_side','box_expanded_030','box_shifted_030','l_notch_filled',
            'l_convex_shortcut','box_extra_detached_box','box_extra_open_strip']
        for name in negatives:
            correct='l_closed' if name.startswith('l_') else 'box_closed'
            gates[name+'_penalized']=quality(correct)-quality(name)>=.01
        gates['all_missing_zero']=quality('empty')==quality('all_instances_missing')==0.
        gates['one_missing_fixed_denominator']=quality('one_instance_missing')<=.5
        gates['duplicate_seed_zero']=quality('duplicate_seed')==0.
        extra=rows['extra_unassigned_instance']['candidate']
        gates['extra_nonempty_output_penalty']=abs(quality('extra_unassigned_instance')-2/3)<=1e-12 and not extra['eligible']
        gates['empty_extra_not_penalized']=abs(quality('empty_extra_duplicate_seed')-quality('both_present'))<=1e-12
        gates['one_mm_gap_not_closed']=rows['one_mm_gap']['candidate']['instances'][0]['projections']['xy']['predicted_area_m2']==0.
        gates['invalid_wire_triangle_rejected_including_unassigned']=counts['expected_invalid_rejections']==2
        gates['expected_evaluation_counts']=counts['candidate_evaluations']==28 and counts['legacy_evaluations']==26
        write(OUT,OUT/'scores.json',rows)
        verify_sources(OUT)
        result=dict(status='passed' if all(gates.values()) else 'failed_gates',gates=gates,
            all_gates_passed=all(gates.values()),failed_gates=[k for k,v in gates.items() if not v],
            quality05={name:quality(name) for name in rows},
            legacy_quality05={name:row['legacy_full_instance']['05cm']['outline_macro_quality'] for name,row in rows.items()},
            counts=counts,main_tasks_used=16,source_count=len(manifest['source_sha256']),
            elapsed_seconds=perf_counter()-started,semantic_efficacy_proven=False,
            scope='analytic task-applicability fixtures; no new physical mapping accuracy')
        write(OUT,OUT/'result.json',result);seal(OUT)
        print(result,flush=True)
    except BaseException:
        write(OUT,OUT/'failure.json',dict(error=traceback.format_exc(),counts=counts,completed_cases=list(rows)))
        if not (OUT/'scores.json').exists():write(OUT,OUT/'partial_scores.json',rows)
        if not (OUT/'artifact_hashes.json').exists():seal(OUT)
        raise


def sha_arrays(mesh):
    import hashlib
    return {key:hashlib.sha256(np.asarray(getattr(mesh,key)).tobytes()).hexdigest() for key in ('vertices','triangles')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    else:parser.print_help()
