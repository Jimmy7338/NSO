#!/usr/bin/env python3
"""One exact slow/fast comparison of a fixed saved observation history.

Loads the archived pre-acceleration frontend. The two independent ledgers
process every paid packet and are compared before downstream computation.
Identical association inputs share residual/predictor computation, whose
outputs are additionally compared against the original saved-history replay.
No simulator, new action, TSDF or evaluation truth is accessed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_mechanism_marker_chain import load_packet
from nso.observed_instances_local import ObservedInstancesLocal
from nso.observed_residual_v41 import ObservedResidualV41
from nso.surface_evaluation_v40 import CandidateViewV40
from nso.view_quality_v42 import ViewQualityPredictorV42


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def timed(function):
    wall,cpu=time.perf_counter(),time.process_time()
    value=function()
    return value,time.perf_counter()-wall,time.process_time()-cpu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=ROOT/'audit_results/mechanism_development_20260923'
    parser.add_argument('--episode',type=Path,default=base/'episodes/pilot_A_S')
    parser.add_argument('--output',type=Path,default=base/'diagnostics/exact_acceleration_validation')
    parser.add_argument('--maximum-seconds',type=float,default=600.)
    args=parser.parse_args()
    if not 0<args.maximum_seconds<=900:
        raise ValueError('verification must have a finite (0,900] second limit')
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    began,cpu_began=time.perf_counter(),time.process_time()
    source_names=('nso/observed_instances_local.py','nso/exact_support_distance.py',
        'nso/observed_instances_v41.py','nso/instance_belief_v40.py',
        'nso/observed_residual_v41.py','nso/view_quality_v42.py',
        'scripts/verify_local_exact_acceleration.py','scripts/diagnose_mechanism_marker_chain.py')
    source_hashes={name:file_sha(ROOT/name) for name in source_names}
    archived=base/'diagnostics/source_before_exact_acceleration/observed_instances_local.py'
    archived_manifest=base/'diagnostics/source_before_exact_acceleration/manifest.json'
    old_replay_path=base/'diagnostics/local_frontend_saved_history_replay.json'
    archive_record=json.loads(archived_manifest.read_text())
    old_replay=json.loads(old_replay_path.read_text())
    if (file_sha(archived)!=archive_record['sha256'] or
            file_sha(old_replay_path)!=archive_record['original_replay_sha256'] or
            file_sha(archived)!=old_replay['source_sha256']['nso/observed_instances_local.py']):
        raise ValueError('archived slow source/original replay binding differs')
    module_spec=importlib.util.spec_from_file_location('archived_local_frontend_before_exact',archived)
    archived_module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(archived_module)
    public_paths=[args.episode/'public_spec.json',args.episode/'public_workspace.json']
    spec,workspace=(json.loads(path.read_text()) for path in public_paths)
    prior=spec['structure_prior']
    kwargs=dict(palette=workspace['marker_palette'],structure_names=prior['abstract_structures'],
                class_structure_prior=prior['probability_by_category'],mode='S',maximum_instances=8)
    ledgers={'slow':archived_module.ObservedInstancesLocal(**kwargs),
             'fast':ObservedInstancesLocal(exact_distance_acceleration=True,**kwargs)}
    residual,predictor=ObservedResidualV41(),ViewQualityPredictorV42(maximum_instances=8)
    paths=sorted((args.episode/'packets').glob('*_rgbd.npz'))
    old_rows={row['paid_step']:row for row in old_replay['frames']}
    old_forecasts={row['instance_id']:row for row in old_replay['first_reliable_plane_forecasts']}
    configuration=dict(schema='local_frontend.exact_acceleration_validation.v1',
        episode_path=str(args.episode.resolve()),maximum_seconds=args.maximum_seconds,
        slow_archived_source_sha256=file_sha(archived),source_sha256=source_hashes,
        original_replay_sha256=file_sha(old_replay_path),original_replay_source_binding_verified=True,
        fast_constructor={'exact_distance_acceleration':True},expected_packets=len(paths),
        query_backend='scipy.spatial.cKDTree; eps=0,p=2,workers=1; NumPy metric recomputation',
        downstream_shared=True,downstream_sharing_condition='complete association and full ledger snapshots exactly equal',
        downstream_also_compared_with_original_replay=True,
        world_calls=0,tsdf_integrations=0,new_actions=0,evaluation_truth_read=False,
        full_raw_pixel_masks_compared=True,whole_association_receipts_compared=True,
        full_public_and_geometry_snapshots_compared=True,
        input_files_sha256={str(path.relative_to(args.episode)):file_sha(path) for path in public_paths})
    write_json(output/'started.json',configuration)
    totals={f'{name}_{kind}_{unit}':0. for name in ('slow','fast')
            for kind in ('observe','feedback') for unit in ('wall_s','cpu_s')}
    totals.update(shared_residual_wall_s=0.,shared_residual_cpu_s=0.,
                  shared_predictor_wall_s=0.,shared_predictor_cpu_s=0.)
    rows,forecasted,error=[],set(),None
    status='running'
    def equal(a,b,label,step):
        if a!=b:
            raise AssertionError(f'{label} differs at paid step {step}; slow_sha={canonical_sha(a)}, fast_sha={canonical_sha(b)}')
    try:
        for path in paths:
            if time.perf_counter()-began>args.maximum_seconds:
                status='time_limit_partial'
                break
            obs=load_packet(path); step=obs.paid_step
            order=('slow','fast') if step%2==0 else ('fast','slow')
            associations={}
            for name in order:
                value,wall,cpu=timed(lambda:ledgers[name].observe(obs))
                associations[name]=value
                totals[f'{name}_observe_wall_s']+=wall
                totals[f'{name}_observe_cpu_s']+=cpu
            equal(associations['slow'],associations['fast'],'complete association receipt',step)
            for slow,fast in zip(associations['slow']['accepted'],associations['fast']['accepted']):
                if not np.array_equal(slow['pixel_indices'],fast['pixel_indices']):
                    raise AssertionError('raw associated pixel mask differs')
                if not np.array_equal(slow['marker_pixel_indices'],fast['marker_pixel_indices']):
                    raise AssertionError('raw marker pixel mask differs')
            before={name:ledger.snapshot() for name,ledger in ledgers.items()}
            equal(before['slow'],before['fast'],'complete pre-feedback public snapshot',step)
            equal(ledgers['slow'].geometry_snapshot(),ledgers['fast'].geometry_snapshot(),
                  'complete pre-feedback geometry snapshot',step)
            scores,wall,cpu=timed(lambda:residual.observe(obs,associations['slow']['accepted']))
            totals['shared_residual_wall_s']+=wall;totals['shared_residual_cpu_s']+=cpu
            equal(scores['results'],old_rows[step]['residual_results'],'original residual/plane results',step)
            feedbacks={}
            for name in order:
                def apply():
                    return [ledgers[name].apply_geometry_feedback(r['instance_id'],frame_id=obs.frame_id,
                        observation_sha256=obs.sha256(),log_likelihoods=r['log_likelihoods'])
                        for r in scores['results'] if r['accepted']]
                feedbacks[name],wall,cpu=timed(apply)
                totals[f'{name}_feedback_wall_s']+=wall;totals[f'{name}_feedback_cpu_s']+=cpu
            equal(feedbacks['slow'],feedbacks['fast'],'full geometry feedback receipt',step)
            applied=[r['instance_id'] for r in feedbacks['slow'] if r['applied']]
            equal(applied,old_rows[step]['geometry_feedback_applied'],'original applied feedback IDs',step)
            after={name:ledger.snapshot() for name,ledger in ledgers.items()}
            equal(after['slow'],after['fast'],'complete post-feedback public snapshot',step)
            geometry={name:ledger.geometry_snapshot() for name,ledger in ledgers.items()}
            equal(geometry['slow'],geometry['fast'],'complete post-feedback geometry snapshot',step)
            views,wall,cpu=timed(lambda:predictor.observe(obs,associations['slow']['accepted']))
            totals['shared_predictor_wall_s']+=wall;totals['shared_predictor_cpu_s']+=cpu
            equal(views,old_rows[step]['view_evidence'],'original predictor evidence/planes',step)
            reliable={r['instance_id'] for r in views['results'] if r['reliable_cached_plane']}
            for instance in after['slow']['instances']:
                key=instance['instance_id']
                if key not in reliable or key in forecasted:
                    continue
                transform=np.array(obs.world_from_camera,copy=True)
                yaw=np.arctan2(transform[1,2],transform[0,2])+np.pi/6
                transform[:3,:3]=[[np.sin(yaw),0.,np.cos(yaw)],[-np.cos(yaw),0.,np.sin(yaw)],[0.,-1.,0.]]
                candidate=CandidateViewV40(obs.intrinsic,transform,obs.depth_m.shape[1],obs.depth_m.shape[0],
                                          view_id='public_turn_plus_30')
                forecast=predictor.forecast(instance,[candidate])
                equal(forecast,old_forecasts[key],'original first reliable-plane forecast',step)
                forecasted.add(key)
            row=dict(paid_step=step,packet_file_sha256=file_sha(path),observation_sha256=obs.sha256(),
                backend_order=list(order),full_association_equal=True,raw_pixel_masks_equal=True,
                public_snapshot_equal=True,geometry_snapshot_equal=True,feedback_equal=True,
                original_residual_planes_and_predictor_equal=True,
                association_sha256=canonical_sha(associations['slow']),
                public_snapshot_sha256=canonical_sha(after['slow']),
                geometry_snapshot_sha256=canonical_sha(geometry['slow']),
                feedback_sha256=canonical_sha(feedbacks['slow']),
                accepted_masks=[dict(instance_id=r['instance_id'],pixels=len(r['pixel_indices']),
                    pixel_indices_sha256=canonical_sha(r['pixel_indices']),
                    marker_pixel_indices_sha256=canonical_sha(r['marker_pixel_indices']))
                    for r in associations['slow']['accepted']],
                support_point_counts={r['instance_id']:len(r['support_points_world_m']) for r in after['slow']['instances']},
                accepted_residuals=sum(r['accepted'] for r in scores['results']),applied_feedback_ids=applied,
                cumulative_timings=dict(totals),elapsed_s=time.perf_counter()-began)
            rows.append(row)
            with (output/'frames.jsonl').open('a') as stream:
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
            if step%10==0 or step==len(paths)-1:
                print(json.dumps(dict(step=step,frames=len(rows),elapsed_s=row['elapsed_s'],
                    slow_observe_s=totals['slow_observe_wall_s'],fast_observe_s=totals['fast_observe_wall_s'],
                    support_point_counts=row['support_point_counts'])),flush=True)
        else:
            status='passed_exact'
        for name,expected in source_hashes.items():
            if file_sha(ROOT/name)!=expected:
                raise AssertionError('source changed during validation: '+name)
        if status=='passed_exact' and forecasted!=set(old_forecasts):
            raise AssertionError('not all original forecast probes reproduced')
    except Exception as exc:
        status='failed_retained'
        error=dict(type=type(exc).__name__,message=str(exc),last_completed_step=rows[-1]['paid_step'] if rows else None)
        write_json(output/'failure.json',error)
    elapsed=time.perf_counter()-began
    result=dict(configuration,status=status,error=error,frames_compared=len(rows),
        elapsed_s=elapsed,total_process_cpu_s=time.process_time()-cpu_began,
        combined_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        timings=totals,exact_masks_and_all_snapshots=(status=='passed_exact'),
        verified_forecast_instance_ids=sorted(forecasted),
        total_accepted_associations=sum(len(r['accepted_masks']) for r in rows),
        final_support_point_counts=rows[-1]['support_point_counts'] if rows else {},
        max_total_support_points=max((sum(r['support_point_counts'].values()) for r in rows),default=0),
        observe_wall_speedup=totals['slow_observe_wall_s']/max(totals['fast_observe_wall_s'],1e-12),
        downstream_scope='One shared deterministic residual/predictor computation after proving equal complete inputs; also checked against unchanged original replay.',
        no_automatic_rerun=True)
    write_json(output/'result.json',result)
    print(json.dumps(dict(status=status,frames=len(rows),elapsed_s=elapsed,timings=totals,
        observe_wall_speedup=result['observe_wall_speedup'],result=str(output/'result.json'))),flush=True)
    return 0 if status=='passed_exact' else 1


if __name__=='__main__':
    raise SystemExit(main())
