#!/usr/bin/env python3
"""One preregistered analytic outline audit; no world, sensor, mapper or TSDF."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import open3d as o3d
from nso.box_union_geometry_v30 import BoxV30,union_exterior_faces_v30
from nso.facility_outline_v30 import FullInstanceOutlineEvaluatorV30

OUT=ROOT/'audit_results/v30_outline_applicability_20260917'
PROTOCOL=ROOT/'docs/research/V30_MEASUREMENT_PROTOCOL_REVIEW_20260917.md'
CAP=1024**2
RESERVE=64*1024**2
CASE_NAMES=('closed_self','vertical_only','missing_one','missing_all',
            'shifted_0p3m','scaled_1p6','duplicate_seeds')


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def store(name,payload):
    if not isinstance(payload,bytes):
        payload=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    require(used+len(payload)+16384<=CAP,'1 MiB evidence limit')
    require(shutil.disk_usage(OUT).free-len(payload)>=RESERVE,'64 MiB free reserve')
    target=OUT/name
    require(not target.exists(),'audit refuses to overwrite '+name)
    target.write_bytes(payload)


def source_files():
    paths={Path(__file__).resolve(),PROTOCOL}
    for module in tuple(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path is None:
            continue
        path=Path(path).resolve()
        try:
            relative=path.relative_to(ROOT)
        except ValueError:
            continue
        if path.suffix=='.py' and relative.parts[0] in ('env','nso','utils','scripts'):
            paths.add(path)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def mesh_from_faces(faces):
    vertices=[];triangles=[]
    for face in faces:
        start=len(vertices);vertices.extend(face.vertices())
        triangles.extend(((start,start+1,start+2),(start,start+2,start+3)))
    mesh=o3d.geometry.TriangleMesh()
    mesh.vertices=o3d.utility.Vector3dVector(np.asarray(vertices,float).reshape(-1,3))
    mesh.triangles=o3d.utility.Vector3iVector(np.asarray(triangles,np.int32).reshape(-1,3))
    return mesh


def fixtures():
    references=[];vertical=[];seeds=[]
    for identifier,cx in enumerate((-3.,3.)):
        body=[cx-1,cx+1,2.,5.,0.,1.6]
        attachment=[cx-.6,cx+.6,5.,5.7,.4,1.2]
        bounds=[body]+([attachment] if identifier==1 else [])
        boxes=[]
        for raw in bounds:
            bounds=tuple(value+(5.5 if axis<2 else .5 if axis<4 else 0.)
                         for axis,value in enumerate(raw))
            boxes.append(BoxV30(bounds,identifier))
        mesh=mesh_from_faces(union_exterior_faces_v30(boxes))
        vertical.append(mesh_from_faces(union_exterior_faces_v30(boxes,vertical_only=True)))
        lo=[cx-2.25+5.5,1.5,0.];hi=[cx+2.25+5.5,7.5,2.]
        references.append(dict(id=identifier,mesh=mesh,bounds=[lo,hi]))
        seeds.append(dict(observed_seed_xyz=[cx+5.5,2.5,.8],
            source='analytic_evaluator_fixture_not_an_observed_sensor_seed'))
    return references,vertical,seeds


def run():
    require(not OUT.exists(),'output exists; no second run or overwrite')
    require(shutil.disk_usage(ROOT).free>=RESERVE+CAP,'audit disk reserve')
    OUT.mkdir();started=perf_counter();sources=source_files()
    try:
        archive=io.BytesIO()
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
            for name in sources:bundle.writestr(name,(ROOT/name).read_bytes())
        store('sources.zip',archive.getvalue())
        store('manifest.json',dict(status='frozen_before_any_projection_evaluation',
            source_sha256=sources,source_archive_sha256=sha(OUT/'sources.zip'),
            protocol_sha256=sha(PROTOCOL),planned_cases=list(CASE_NAMES),
            bounds_role='reference and first-seed association only; no prediction clipping',
            map_or_sensor_creation_authorized=False,main_tasks_used_before=12))
        assets,vertical,seeds=fixtures()
        closed=[item['mesh'] for item in assets]
        shifted=[deepcopy(mesh).translate((.3,.3,.3)) for mesh in closed]
        scaled=[]
        for mesh in closed:
            points=np.asarray(mesh.vertices);center=(points.min(axis=0)+points.max(axis=0))/2
            scaled.append(deepcopy(mesh).scale(1.6,center))
        empty=lambda:o3d.geometry.TriangleMesh()
        inputs=dict(closed_self=(closed,seeds,2),vertical_only=(vertical,seeds,2),
            missing_one=([closed[0],empty()],[seeds[0],None],1),
            missing_all=([empty(),empty()],[None,None],0),
            shifted_0p3m=(shifted,seeds,2),scaled_1p6=(scaled,seeds,2),
            duplicate_seeds=(closed,[seeds[0],seeds[0]],2))
        evaluator=FullInstanceOutlineEvaluatorV30(assets)
        scores={}
        for name in CASE_NAMES:
            meshes,cues,count=inputs[name]
            scores[name]=evaluator.evaluate(meshes,cues,count,1.,returned=True,
                collisions=0,failed=False,paid_actions=0,budget=0)
        # Preserve all scores before the assertions, including low results.
        store('scores.json',dict(cases=scores,scope='analytic mesh fixtures; not collected geometry',
            artificial_qualification_fields_are_test_inputs_not_mission_evidence=True))
        quality=lambda name:scores[name]['05cm']['outline_macro_quality']
        require(abs(quality('closed_self')-1.)<1e-12,'closed reference self score')
        require(abs(quality('missing_one')-.5)<1e-12,'fixed two-asset missing denominator')
        require(quality('missing_all')==0. and scores['missing_all']['missing_asset_count']==2,'all missing zero')
        require(quality('duplicate_seeds')==0. and not scores['duplicate_seeds']['unique_complete_seed_association'],
                'duplicate association must not select best mesh')
        require(quality('shifted_0p3m')<1. and quality('scaled_1p6')<1.,'no hidden alignment/size crop')
        require(all(row['outside_window']['outside_vertices']>0 for row in scores['scaled_1p6']['instances']),
                'oversized prediction retained outside task window')
        vertical_rows=scores['vertical_only']['instances']
        require(all(row['projections']['xy']['missing_projection'] for row in vertical_rows),
                'exact vertical XY degeneracy must be visible')
        require(all(not row['projections'][name]['missing_projection'] and
                    row['projections'][name]['iou']>1.-1e-12 for row in vertical_rows for name in ('xz','yz')),
                'two nondegenerate exact exterior projections')
        require(all(abs(row['05cm']['outline_quality']-2/3)<1e-12 for row in vertical_rows),
                'vertical-only expected formula, no artificial thickness')
        for name,expected in sources.items():require(sha(ROOT/name)==expected,'frozen source changed: '+name)
        store('result.json',dict(status='passed_with_declared_vertical_projection_limitation',
            geometry_cases=len(CASE_NAMES),source_count=len(sources),reference_signature=evaluator.reference_signature,
            quality_05cm={name:quality(name) for name in CASE_NAMES},
            vertical_only_q_is_not_full_observation_failure=True,
            vertical_only_missing_projection='xy',vertical_only_evaluable_projections=['xz','yz'],
            at_least_two_projections_evaluable=True,closed_self_and_missing_gates=True,
            original_V23_headline_not_replaced=True,companion_prediction_clipping=False,
            main_tasks_used=12,new_main_tasks=0,world_instances=0,sensor_packets=0,
            mapper_updates=0,TSDF_integrations=0,
            analytic_projection_evaluator_calls=len(CASE_NAMES),robot_or_sensor_Q_evaluations=0,
            elapsed_seconds=perf_counter()-started,physical_trial_efficacy_proven=False))
    except BaseException as error:
        store('failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            new_main_tasks=0,world_instances=0,mapper_updates=0,TSDF_integrations=0))
        raise
    finally:
        store('artifact_hashes.json',{p.name:sha(p) for p in sorted(OUT.iterdir())
            if p.is_file() and p.name!='artifact_hashes.json'})
    print((OUT/'result.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Run the only frozen analytic applicability audit')
    args=parser.parse_args()
    if args.run:run()
    else:parser.print_help()
