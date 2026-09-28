#!/usr/bin/env python3
"""Verify the fixed batch; characterize failures from saved observations only.

No planning, sensor queries, new motion trials, or changes to acquisition files.
Unsuccessful partial maps are measured with failed=True and never become successes.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_thesis_expansion_20260928 as batch


def csv_write(path, rows):
    if not rows: return
    buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);batch.write_bytes(path,buffer.getvalue().encode())


def partial_endpoint(case, folder, output):
    import numpy as np
    from nso.decision_replay_v13 import load_packet
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from nso.surface_measurement_v34 import extract_observed_asset_mesh, SurfaceMeasurementV34
    started=time.monotonic()
    parent, models, _, info=batch.load_scene(case['scene'])
    # This object supplies configuration and offline references only. No packet/step call.
    reference=batch.world_for(parent,case['hypothesis'],'offline-failure-reference',case['budget'],case['noise_seed'])
    mapper=ObservedRuntimeMapperV10(reference.shape,reference.config,truncation_m=.12)
    packets=sorted((folder/'packets').glob('*.npz'))
    if not packets:
        return {'measurement':None,'reason':'no recorded sensor observation','new_sensor_queries':0,'new_planning_calls':0}
    collisions=0;last=None
    for path in packets:
        packet=load_packet(path);mapper.update(packet.frame,packet.scan)
        collisions+=int(packet.collision);last=packet
    raw=mapper.mesh();mesh,crop=extract_observed_asset_mesh(raw,info['public_bounds'])
    batch.arrays(output/'partial_mesh.npz',vertices=np.asarray(mesh.vertices),triangles=np.asarray(mesh.triangles))
    batch.arrays(output/'partial_map.npz',belief=mapper.belief)
    batch.write(output/'partial_crop.json',crop)
    batch.write(output/'prediction_freeze.json',dict(before_evaluation_reference=True,
        source_acquisition_seal_sha256=batch.sha(folder/'seal.json'),
        artifacts={p.name:batch.sha(p) for p in output.iterdir() if p.is_file()}))
    floor=reference.evaluation_floor(); coverage=float(np.mean(mapper.belief[floor['reachable']]!=-1))
    pose=tuple(map(int,np.rint(last.frame.world_from_camera[:2,3]-reference.shift[:2])))+(int(last.heading),)
    evaluator=SurfaceMeasurementV34(reference.instance_mesh(0),reference.instance_mesh(0,vertical_only=True))
    measure=evaluator.evaluate(mesh,coverage,returned=pose==models[0].poses[models[0].anchor],
        collisions=collisions,failed=True,paid_actions=int(last.action_id),budget=case['budget'])
    result=dict(measurement=measure,case=case,source_acquisition_seal_sha256=batch.sha(folder/'seal.json'),
        scope='offline reconstruction of recorded partial failure trajectory; never a successful endpoint',
        saved_frames_reintegrated=len(packets),new_sensor_queries=0,new_planning_calls=0,new_paid_actions=0,
        sensor_counts=dict(reference.counts),elapsed_seconds=time.monotonic()-started)
    batch.write(output/'result.json',result)
    return result


def main():
    analysis_started=time.monotonic()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        if os.environ.get(name)!='1': raise ValueError(name+' must equal 1')
    config=batch.verify_sources();out=batch.OUT
    review=out/'analysis_review';review.mkdir()
    import numpy as np
    import scipy
    import open3d
    cpu_models=sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')})
    batch.write(review/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        open3d=open3d.__version__,cpu_models=cpu_models,logical_cpu_count=os.cpu_count(),
        threads_per_case=1,simultaneous_cases=1,gpu_used=False,TSDF_voxel_m=.04,TSDF_truncation_m=.12))
    batch.write(review/'analysis_freeze.json',dict(script_sha256=batch.sha(Path(__file__)),
        runner_sha256=batch.sha(Path(batch.__file__)),source_freeze_sha256=batch.sha(out/'source_freeze.json'),
        acquisition_inputs={c['id']:batch.sha(out/'cases'/c['id']/'seal.json') for c in config['cases']},
        corrections=['layout count describes distinct base geometries, never independent test layouts',
                     'qualification requires completed status AND eligible measurement',
                     'failed partial endpoints kept separately; no new online trials']))
    # Preserve the acquisition runner's own original arithmetic report as an audit source.
    batch.analyze()
    rows=[]; per_condition=defaultdict(dict); failure_reason_counts=defaultdict(int)
    full_prefix_groups=defaultdict(list)
    for case in config['cases']:
        folder=out/'cases'/case['id']
        result=batch.read(folder/('result.json' if (folder/'result.json').exists() else 'failure.json'))
        trace_path=folder/('trace.json' if (folder/'trace.json').exists() else 'partial_trace.json')
        if trace_path.exists():
            trace=batch.read(trace_path)
            digest=hashlib.sha256(batch.payload([dict(nonsemantic=r['nonsemantic'],pose=r['pose']) for r in trace[:19]])).hexdigest()
            full_prefix_groups[(case['scene'],case['budget'],case['noise_seed'])].append(dict(case=case['id'],
                prefix_frames=min(len(trace),19),nonsemantic_prefix_sha256=digest))
        measure=result.get('measurements',{}).get('final')
        scope='completed_online_endpoint'
        if result['status']!='completed':
            scope='partial_failure_endpoint';failure_reason_counts[result['exception']]+=1
            target=review/'failure_endpoints'/case['id'];target.mkdir(parents=True)
            supplement=partial_endpoint(case,folder,target)
            measure=supplement.get('measurement')
        measure=measure or {}
        qualified=result['status']=='completed' and bool(measure.get('eligible',False))
        row=dict(case,status=result['status'],measurement_scope=scope,qualified=qualified,
                 paid_actions=result['paid_actions'],returned=measure.get('returned'),
                 collisions=measure.get('collisions'),C_map=measure.get('C_map'),
                 elapsed_online_seconds=result['elapsed_seconds'],failure=result.get('exception',''))
        for threshold in ('02cm','05cm','10cm'):
            for field in ('precision','recall','f1','joint'):
                row[threshold+'_'+field]=measure.get(threshold,{}).get(field)
        rows.append(row)
        per_condition[(case['scene'],case['hypothesis'],case['budget'],case['noise_seed'])][case['mode']]=row
    csv_write(review/'complete_endpoint_metrics.csv',rows)
    full_prefix_checks=[dict(scene=key[0],budget=key[1],noise_seed=key[2],declared_arms=8,
        recorded_arms=len(values),all_prefix_frames_present=all(v['prefix_frames']==19 for v in values),
        all_nonsemantic_prefixes_identical=len({v['nonsemantic_prefix_sha256'] for v in values})==1)
        for key,values in full_prefix_groups.items()]
    csv_write(review/'all_success_and_failure_prefix_checks.csv',full_prefix_checks)
    if not all(r['recorded_arms']==r['declared_arms'] and r['all_prefix_frames_present'] and r['all_nonsemantic_prefixes_identical'] for r in full_prefix_checks):
        raise ValueError('full declared paired prefix check failed; no final aggregate accepted')
    paired=[]
    for key,modes in per_condition.items():
        for a,b in [('S','G'),('X','Xnf'),('X','G')]:
            left,right=modes[a],modes[b]
            qualified=left['qualified'] and right['qualified']
            p=dict(scene=key[0],hypothesis=key[1],budget=key[2],noise_seed=key[3],treatment=a,control=b,
                   both_qualified=qualified,treatment_status=left['status'],control_status=right['status'])
            for threshold in ('02cm','05cm','10cm'):
                x,y=left[threshold+'_joint'],right[threshold+'_joint']
                p[threshold+'_delta']=x-y if x is not None and y is not None else None
            p['J5_treatment']=left['05cm_joint'];p['J5_control']=right['05cm_joint']
            p['interpretation']='completed online contrast' if qualified else 'failure/qualification boundary; partial map score not successful performance'
            paired.append(p)
    csv_write(review/'complete_paired_metrics.csv',paired)
    aggregates=[]
    for family,scenes in [('originals',('P00','P01')),('same_family_variants',('P00_rib_shift','P01_lower_ribs'))]:
        for budget in (30,42,54):
            for a,b in [('S','G'),('X','Xnf'),('X','G')]:
                group=[p for p in paired if p['scene'] in scenes and p['budget']==budget and p['treatment']==a and p['control']==b]
                if not group:continue
                all_qualified=all(p['both_qualified'] for p in group)
                row=dict(scene_family=family,budget=budget,treatment=a,control=b,declared_pairs=len(group),
                         qualified_pairs=sum(p['both_qualified'] for p in group),all_pairs_qualified=all_qualified,
                         distinct_base_parent_count=2,scene_variant_count=2 if family=='same_family_variants' else 0,
                         noise_replicates_per_hypothesis=2)
                if all_qualified:
                    base=sum(p['J5_control'] for p in group)/len(group)
                    delta=sum(p['05cm_delta'] for p in group)/len(group)
                    row.update(mean_J5_control=base,mean_J5_treatment=base+delta,mean_delta_J5=delta,
                               relative_percent=100*delta/base if base else None,
                               wins=sum(p['05cm_delta']>1e-12 for p in group),ties=sum(abs(p['05cm_delta'])<=1e-12 for p in group),
                               losses=sum(p['05cm_delta'] < -1e-12 for p in group))
                else:row['quality_comparison']='not aggregated as completed performance; all failures retained separately'
                aggregates.append(row)
    stratified=[]
    scopes=[('originals',('P00','P01')),('same_family_variants',('P00_rib_shift','P01_lower_ribs'))]
    scopes += [(scene,(scene,)) for scene in ('P00','P01','P00_rib_shift','P01_lower_ribs')]
    for scope,scenes in scopes:
        for budget in (30,42,54):
            for mode in ('G','S','X','Xnf'):
                group=[r for r in rows if r['scene'] in scenes and r['budget']==budget and r['mode']==mode]
                if not group:continue
                all_qualified=all(r['qualified'] for r in group)
                for threshold in ('02cm','05cm','10cm'):
                    stratified.append(dict(scope=scope,budget=budget,mode=mode,threshold=threshold,
                        declared_episodes=len(group),completed_episodes=sum(r['status']=='completed' for r in group),
                        qualified_episodes=sum(r['qualified'] for r in group),all_qualified=all_qualified,
                        mean_C_map=sum(r['C_map'] for r in group)/len(group) if all_qualified else None,
                        mean_precision=sum(r[threshold+'_precision'] for r in group)/len(group) if all_qualified else None,
                        mean_recall=sum(r[threshold+'_recall'] for r in group)/len(group) if all_qualified else None,
                        mean_f1=sum(r[threshold+'_f1'] for r in group)/len(group) if all_qualified else None,
                        mean_joint=sum(r[threshold+'_joint'] for r in group)/len(group) if all_qualified else None,
                        mean_paid_actions=sum(r['paid_actions'] for r in group)/len(group),
                        mean_online_seconds=sum(r['elapsed_online_seconds'] for r in group)/len(group)))
    csv_write(review/'stratified_metrics.csv',stratified)
    detailed_pairs=[]
    for scope,scenes in scopes:
        for budget in (30,42,54):
            for a,b in [('S','G'),('X','Xnf'),('X','G')]:
                group=[p for p in paired if p['scene'] in scenes and p['budget']==budget and p['treatment']==a and p['control']==b]
                if not group:continue
                valid=all(p['both_qualified'] for p in group)
                for threshold in ('02cm','05cm','10cm'):
                    controls=[r[threshold+'_joint'] for r in rows if r['scene'] in scenes and r['budget']==budget and r['mode']==b]
                    mean_control=sum(controls)/len(controls) if valid else None
                    delta=sum(p[threshold+'_delta'] for p in group)/len(group) if valid else None
                    detailed_pairs.append(dict(scope=scope,budget=budget,treatment=a,control=b,threshold=threshold,
                        declared_pairs=len(group),qualified_pairs=sum(p['both_qualified'] for p in group),
                        mean_control=mean_control,mean_delta=delta,
                        relative_percent=100*delta/mean_control if valid and mean_control else None,
                        wins=sum(p[threshold+'_delta']>1e-12 for p in group) if valid else None,
                        ties=sum(abs(p[threshold+'_delta'])<=1e-12 for p in group) if valid else None,
                        losses=sum(p[threshold+'_delta'] < -1e-12 for p in group) if valid else None))
    csv_write(review/'stratified_pair_statistics.csv',detailed_pairs)
    mixtures=[]
    for key,modes in per_condition.items():
        valid=all(modes[m]['qualified'] for m in ('G','S','X'))
        if not valid:continue
        for reliability in (0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.):
            mixed=reliability*modes['S']['05cm_joint']+(1-reliability)*modes['X']['05cm_joint']
            mixtures.append(dict(scene=key[0],hypothesis=key[1],budget=key[2],noise_seed=key[3],
                probability_correct_episode_category=reliability,mixture_J5=mixed,
                G_J5=modes['G']['05cm_joint'],delta_J5=mixed-modes['G']['05cm_joint'],
                new_reliability_trials=0))
    csv_write(review/'analytic_category_mixture.csv',mixtures)
    break_even=[]
    for scope,scenes in scopes:
        for budget in (30,42,54):
            groups=[modes for key,modes in per_condition.items() if key[0] in scenes and key[2]==budget]
            if not groups or not all(all(m[meth]['qualified'] for meth in ('G','S','X')) for m in groups):continue
            means={meth:sum(m[meth]['05cm_joint'] for m in groups)/len(groups) for meth in ('G','S','X')}
            slope=means['S']-means['X']
            crossing=(means['G']-means['X'])/slope if abs(slope)>1e-12 else None
            break_even.append(dict(scope=scope,budget=budget,matched_conditions=len(groups),
                G_mean=means['G'],S_correct_mean=means['S'],X_wrong_mean=means['X'],
                mixing_slope=slope,algebraic_crossing_correct_category_probability=crossing,
                crossing_inside_probability_interval=crossing is not None and 0<=crossing<=1,
                semantic_prior_in_controller=.9,new_reliability_trials=0,
                interpretation='descriptive linear mixture of observed complete endpoints; not calibration or guarantee'))
    csv_write(review/'analytic_category_break_even.csv',break_even)
    summary=dict(declared_cases=len(rows),completed=sum(r['status']=='completed' for r in rows),
        failed=sum(r['status']!='completed' for r in rows),qualified=sum(r['qualified'] for r in rows),
        failure_reason_counts=dict(failure_reason_counts),all_failure_records_retained=True,
        all_original_frozen_59_sources_unchanged=True,aggregate=aggregates,
        all_128_success_and_failure_prefixes_verified=True,
        source_freeze_sha256=batch.sha(out/'source_freeze.json'),
        raw_runner_summary_sha256=batch.sha(out/'summary.json'),
        raw_summary_field_correction='independent_layout_count in raw summary means distinct base_parent_count; no independence inference permitted',
        all_new_data_bytes=batch.used_bytes(),disk_free_GiB=__import__('shutil').disk_usage(ROOT).free/batch.GIB,
        statistical_significance_claimed=False,unseen_family_generalization_claimed=False,
        total_online_episode_seconds=sum(r['elapsed_online_seconds'] for r in rows),
        analysis_and_partial_fusion_seconds=time.monotonic()-analysis_started,
        analytic_mixture_file='analytic_category_mixture.csv',
        analytic_mixture_scope='completed measured G/S/X conditions only, deterministic mixture of correct/wrong category endpoints; not extra trials or calibrated reliability')
    batch.write(review/'summary.json',summary)
    batch.write(review/'seal.json',{str(p.relative_to(review)):batch.sha(p) for p in review.rglob('*') if p.is_file()})
    batch.verify_sources()
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
