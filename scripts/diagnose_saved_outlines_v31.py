#!/usr/bin/env python3
"""Once-only V30 saved-mesh diagnostic after the V31 analytic gates pass."""
import argparse
import hashlib
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
import traceback
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import (ROOT,read,write,freeze,verify_sources,
    verify_inventory,seal,sha)
from nso.supported_outline_v31 import FullSupportedOutlineEvaluatorV31
from nso.box_union_geometry_v30 import union_exterior_faces_v30
from env.information_pixel_v30 import scene_boxes,mesh_from_faces,SHIFT
from scripts.audit_supported_outline_v31 import no_acquisition_guards

OUT=ROOT/'audit_results/v31_saved_outline_diagnostic_20260917'
SOURCE=ROOT/'audit_results/v30_pixel_routes_20260917'
GATE=ROOT/'audit_results/v31_supported_outline_20260917'
PROTOCOL=ROOT/'docs/research/V31_SUPPORTED_OUTLINE_PROTOCOL_20260917.md'


def legacy_sources():
    manifest=read(SOURCE/'manifest.json')
    if sha(SOURCE/'sources.zip')!=manifest['source_zip_sha256']:raise ValueError('old archive changed')
    with zipfile.ZipFile(SOURCE/'sources.zip') as archive:
        if set(archive.namelist())!=set(manifest['source_sha256']):raise ValueError('old archive set differs')
        for name,expected in manifest['source_sha256'].items():
            if sha(ROOT/name)!=expected or hashlib.sha256(archive.read(name)).hexdigest()!=expected:
                raise ValueError('old source bytes changed: '+name)
    return manifest


def reference_assets(hypothesis):
    # Pure frozen configuration/geometry functions, no World instance and no
    # reachable-floor calculation. Coverage is inherited from actual V30 data.
    cfg,_,facilities=scene_boxes(hypothesis)
    assets=[]
    for index,station in enumerate(cfg['stations']):
        x=station['center_x']+SHIFT[0]
        assets.append(dict(id=index,mesh=mesh_from_faces(union_exterior_faces_v30(facilities[index])),
            bounds=[[x-1.75,2.,.01],[x+1.75,7.,2.1]]))
    return assets


def run():
    if OUT.exists():raise FileExistsError('saved diagnostic already started; no implicit retry')
    verify_inventory(GATE);verify_sources(GATE)
    if not read(GATE/'result.json')['all_gates_passed']:raise ValueError('analytic gate failed; saved diagnostic forbidden')
    verify_inventory(SOURCE);old_manifest=legacy_sources()
    for i in range(4):verify_inventory(SOURCE/f'case{i:02d}')
    OUT.mkdir();started=perf_counter();rows=[]
    counts=dict(candidate_evaluations=0,old_quality_evaluations=0,mesh_inputs_read=0,
        worlds=0,sensor_packets=0,mapper_updates=0,TSDF_integrations=0,mapper_mesh_extractions=0,new_main_tasks=0)
    try:
        no_acquisition_guards()
        inputs=[SOURCE/'manifest.json',SOURCE/'artifact_hashes.json',SOURCE/'result.json',GATE/'result.json',
            GATE/'artifact_hashes.json',GATE/'manifest.json']
        for i in range(4):
            folder=SOURCE/f'case{i:02d}'
            inputs.extend([folder/'result.json',folder/'artifact_hashes.json',folder/'replay.json'])
            for stage in ('prefix','final'):
                inputs.append(folder/f'{stage}_metadata.json')
                inputs.extend(folder/f'{stage}_slot{slot}.npz' for slot in range(2))
        # Inputs are already sealed under the V30 inventory. Archive compact
        # metadata/source only; preserve unique NPZ files in their old location.
        code=[Path(__file__),PROTOCOL,ROOT/'docs/research/V31_ANALYTIC_FIXTURES_PROTOCOL_20260917.md']
        manifest=freeze(OUT,code,status='frozen_before_saved_mesh_scoring',
            input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
            source_main_quota=16,old_source_count=len(old_manifest['source_sha256']),
            posthoc_development_diagnostic=True,old_metrics_not_recomputed=True)
        for i in range(4):
            folder=SOURCE/f'case{i:02d}';old=read(folder/'result.json')
            evaluator=FullSupportedOutlineEvaluatorV31(reference_assets(old['case']['hypothesis']))
            stages={}
            for stage in ('prefix','final'):
                metadata=read(folder/f'{stage}_metadata.json');saved=old['stages'][stage]
                if evaluator.reference.reference_signature!=saved['window']['reference_signature']:
                    raise ValueError('reference differs from original V30 reference')
                meshes=[];geometry_hashes=[]
                for slot in range(2):
                    with np.load(folder/f'{stage}_slot{slot}.npz',allow_pickle=False) as npz:
                        vertices=npz['vertices'].copy();triangles=npz['triangles'].copy()
                    vertices.setflags(write=False);triangles.setflags(write=False)
                    mesh=SimpleNamespace(vertices=vertices,triangles=triangles)
                    h=hashlib.sha256(vertices.tobytes()+triangles.tobytes()).hexdigest()
                    expected=saved['full_instance']['submitted_instance_support'][slot]['submitted_mesh']['sha256']
                    if h!=expected:raise ValueError('saved instance geometry differs from old full-instance evaluation')
                    meshes.append(mesh);geometry_hashes.append(h);counts['mesh_inputs_read']+=1
                old_window=saved['window']['main_observed']
                counts['candidate_evaluations']+=1
                new=evaluator.evaluate(meshes,[x['seed'] for x in metadata['instances']],
                    len(metadata['observed_track_summary']),old_window['coverage_2d'],
                    returned=old_window['returned'],collisions=old_window['collisions'],failed=old_window['failed'],
                    paid_actions=18 if stage=='prefix' else 48,budget=48)
                stages[stage]=dict(candidate=new,original_window=saved['window']['main_observed'],
                    original_full_instance=saved['full_instance'],prediction_geometry_sha256=geometry_hashes,
                    candidate_minus_original_full_Q05=new['05cm']['outline_macro_quality']-
                        saved['full_instance']['05cm']['outline_macro_quality'],
                    coverage_and_physical_qualification_inherited=True)
                print(f'case {i} {stage}: saved geometry evaluated',flush=True)
            rows.append(dict(case=old['case'],stages=stages))
        source_analysis=read(SOURCE/'result.json')
        selection={}
        ids=source_analysis['observed_rule_case_indices']
        for tag in ('02cm','05cm','10cm'):
            selection[tag]={}
            for key in ('outline_macro_quality','joint_outline'):
                values=[row['stages']['final']['candidate'][tag][key] for row in rows]
                mean=lambda indices:float(np.mean([values[j] for j in indices]))
                fixedA,fixedB=mean([0,2]),mean([1,3]);best=max(fixedA,fixedB)
                selection[tag][key]=dict(always_A=fixedA,always_B=fixedB,best_fixed=best,
                    complex_rule=mean(ids['complex']),swapped_rule=mean(ids['swapped']),
                    oracle=float((max(values[:2])+max(values[2:]))/2),
                    complex_relative=mean(ids['complex'])/best-1 if best else None,
                    scope='ineligible, previously inspected fixed-route development data')
        verify_sources(OUT);verify_sources(GATE);verify_inventory(SOURCE);legacy_sources()
        for rel,expected in manifest['input_sha256'].items():
            if sha(ROOT/rel)!=expected:raise ValueError('saved input changed during diagnostic')
        write(OUT,OUT/'scores.json',rows)
        changes=[stage['candidate_minus_original_full_Q05'] for row in rows for stage in row['stages'].values()]
        result=dict(status='complete',counts=counts,main_tasks_used=16,
            maximum_absolute_change_Q05=float(max(map(abs,changes))),
            final_candidate_Q05=[row['stages']['final']['candidate']['05cm']['outline_macro_quality'] for row in rows],
            final_original_full_Q05=[row['stages']['final']['original_full_instance']['05cm']['outline_macro_quality'] for row in rows],
            final_candidate_eligible=[row['stages']['final']['candidate']['eligible'] for row in rows],
            physical_qualification_unchanged=True,selection=selection,
            observed_geometry_changed=False,semantic_efficacy_proven=False,
            source_count=len(manifest['source_sha256']),input_count=len(manifest['input_sha256']),
            elapsed_seconds=perf_counter()-started)
        write(OUT,OUT/'result.json',result);seal(OUT);print(result,flush=True)
    except BaseException:
        write(OUT,OUT/'failure.json',dict(error=traceback.format_exc(),counts=counts,complete_cases=len(rows)))
        if not (OUT/'scores.json').exists():write(OUT,OUT/'partial_scores.json',rows)
        if not (OUT/'artifact_hashes.json').exists():seal(OUT)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    else:parser.print_help()
