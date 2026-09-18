#!/usr/bin/env python3
"""One frozen 416-pose pixel-information gate; no mapping or main tasks."""
import argparse
import io
from pathlib import Path
import signal
import sys
from time import perf_counter
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from nso.research_evidence_v31 import ROOT,read,sha,write,write_bytes,freeze,verify_sources,verify_inventory,seal
from nso.pixel_information_v34 import SavedPotentialModelV34, arrays_sha256, information_pair_v34
from nso.finite_belief_solver_v33 import ExactCoverageBeliefSolverV33,SolverLimitV33,NEG
from env.information_pixel_v34 import InformationPixelWorldV34,nonsemantic_rgb,cue_from_rgb,sensor_counts_v34

OLD=ROOT/'audit_results/v33_direction_information_r1_20260917'
CONFIG=ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json'
PROTOCOL=ROOT/'docs/research/V34_PIXEL_INFORMATION_PROTOCOL_20260918.md'
OUT=ROOT/'audit_results/v34_pixel_information_20260918'


def raw_fields(packet):
    return dict(depth=packet.frame.depth_m,rgb=packet.frame.color_rgb,
        intrinsic=packet.frame.intrinsic,camera_pose=packet.frame.world_from_camera,
        ranges=packet.scan.ranges_m,laser_pose=packet.scan.world_from_laser,
        scan_calibration=np.asarray([packet.scan.angle_min_rad,packet.scan.angle_increment_rad,packet.scan.range_max_m]),
        cell=np.asarray(packet.position,dtype=np.int64),heading=np.asarray(packet.heading,dtype=np.int64))


def signature_fields(fields):
    return {**{key:value for key,value in fields.items() if key!='rgb'},'rgb_nonsemantic':nonsemantic_rgb(fields['rgb'])}


def run():
    if OUT.exists(): raise FileExistsError('single frozen run already exists; no implicit retry')
    verify_sources(OLD);verify_inventory(OLD)
    cfg=read(CONFIG)
    if any(sensor_counts_v34().values()): raise ValueError('formal batch needs a fresh process')
    OUT.mkdir(); started=perf_counter(); solvers=[]; worlds=[]
    counts=dict(worlds=0,sensor_packets=0,clean_depth_queries=0,scan_queries=0,
        saved_reward_models=0,exact_solver_instances=0,memo_states=0,
        mapper_updates=0,TSDF_integrations=0,Q_evaluations=0,new_main_tasks=0)
    def sync_sensor_counts():
        actual=dict(sensor_counts_v34())
        for key in ('worlds','clean_depth_queries','scan_queries'): counts[key]=actual[key]
        counts['sensor_packets']=actual['packets'];counts['paid_step_calls']=actual['step_calls']
    def time_limit(*_): raise SolverLimitV33('300 second gate cap exceeded; no certified result')
    signal.signal(signal.SIGALRM,time_limit);signal.alarm(300)
    try:
        inputs={str(p.relative_to(ROOT)):sha(p) for p in OLD.iterdir() if p.is_file()}
        sources=[Path(__file__),CONFIG,PROTOCOL,ROOT/'tests/virtual3d/test_information_pixel_v34.py',
            ROOT/'tests/virtual3d/test_pixel_information_v34.py']
        manifest=freeze(OUT,sources,status='before_first_formal_world_or_query',
            input_sha256=inputs,maximum_formal_pose_queries=416,maximum_memo_states=1500000,
            forced_prefix_actions=18,remaining_actions=24,main_tasks_used=16,
            precision_thresholds_cm={'2':None,'5':None,'10':None})
        parent_records=[]; model_pairs=[]
        for parent in cfg['parents']:
            prior=read(OLD/(parent['id']+'_geometry.json'))
            rows=[]; models=[]
            for h,table in enumerate(prior['tables']):
                world=InformationPixelWorldV34(parent['id'],h,f'v34-pixel-{parent["id"]}-h{h}',noise_model='clean')
                worlds.append(world);counts['worlds']+=1
                columns={}; frames=[]; signatures=[]
                for node,pose in enumerate(table['poses']):
                    if counts['sensor_packets']>=416: raise RuntimeError('query quota exhausted')
                    packet=world.packet_at(tuple(pose),step=node,noise_model='clean')
                    counts['sensor_packets']+=1;counts['clean_depth_queries']+=1;counts['scan_queries']+=1
                    if not np.array_equal(packet.frame.depth_m,world.last_clean_depth): raise ValueError('formal gate must be clean')
                    fields=raw_fields(packet)
                    for name,value in fields.items(): columns.setdefault(name,[]).append(np.array(value,copy=True))
                    sig=signature_fields(fields);signatures.append(arrays_sha256(**sig))
                    frames.append(dict(node=node,pose=pose,geometric_sha256=signatures[-1],
                        fields_sha256={name:arrays_sha256(value=value) for name,value in sig.items()},
                        raw_sha256=arrays_sha256(**fields),cue=cue_from_rgb(packet.frame.color_rgb)))
                archive={key:np.stack(value) for key,value in columns.items()}
                buffer=io.BytesIO();np.savez_compressed(buffer,**archive)
                name=f'{parent["id"]}_h{h}_pixels.npz';write_bytes(OUT,OUT/name,buffer.getvalue())
                # This is evaluator geometry only; no mapper is instantiated.
                reachable=world.evaluation_floor()['reachable']
                rows.append(dict(hypothesis=h,pixels_file=name,frames=frames,
                    raster_shape=world.shape,static_reachable_floor_cells=int(reachable.sum()),
                    static_reachable_sha256=arrays_sha256(reachable=reachable),
                    actual_map_coverage=None,world_counts=dict(world.counts)))
                models.append(SavedPotentialModelV34(table,signatures));counts['saved_reward_models']+=1
            pair=information_pair_v34(models)
            cues=[]
            for h,row in enumerate(rows):
                observations=[row['frames'][n] for n in models[h].prefix_nodes]
                recognized=[o for o in observations if o['cue']['pixel_count']>=16 and o['cue']['type']==('type_A','type_B')[h]]
                locations={tuple(o['pose'][:2]) for o in recognized}
                contradictions=[o for o in observations if o['cue']['pixel_count']>=16 and o['cue']['type'] not in (None,('type_A','type_B')[h])]
                cues.append(dict(hypothesis=h,prefix_recognized_frames=len(recognized),
                    distinct_positions=len(locations),positions=sorted(locations),contradictions=len(contradictions),
                    passed=len(locations)>=2 and not contradictions))
            prior_first=prior['pair']['first_information_action_layer']
            record=dict(parent_id=parent['id'],hypotheses=rows,pair=pair,prefix_cue_checks=cues,
                previous_ideal_first_information_layer=prior_first,
                information_earlier_than_v33=pair['first_information_action_layer'] is not None and pair['first_information_action_layer']<prior_first,
                paired_raster_denominator=rows[0]['static_reachable_sha256']==rows[1]['static_reachable_sha256'],
                prerequisite_passed=pair['prefix_equal'] and pair['all_poses_connected'] and all(c['passed'] for c in cues)
                    and rows[0]['static_reachable_sha256']==rows[1]['static_reachable_sha256'])
            parent_records.append(record);model_pairs.append(models)
            write(OUT,OUT/(parent['id']+'_observations.json'),record)
            print(dict(stage='pixels_complete',parent=parent['id'],prefix_equal=pair['prefix_equal'],
                first=pair['first_information_action_layer'],cue_checks=cues,packets=counts['sensor_packets']),flush=True)
        if counts['sensor_packets']!=416: raise AssertionError('incomplete declared pose matrix')
        policies=[]
        if all(r['prerequisite_passed'] for r in parent_records):
            for parent,models in zip(cfg['parents'],model_pairs):
                initial=[m.initial_mask for m in models]
                left=1500000-sum(s.states for s in solvers)
                if left<=0: raise SolverLimitV33('memo cap exhausted')
                solver=ExactCoverageBeliefSolverV33(models,initial,left);solvers.append(solver)
                counts['exact_solver_instances']+=1
                g=solver.uncertain(solver.anchor,24,*initial)
                known=[solver.known(h,solver.anchor,24,initial[h]) for h in (0,1)]
                if min(g,*known)<=NEG/2: raise ValueError('no qualified finite policy')
                witnesses=[]
                for h in (0,1):
                    for policy in ('G','class_oracle','swapped_class_with_correction'):
                        w=solver.witness(h,24,policy,hint=1-h if policy.startswith('swapped') else None)
                        w['total_paid_actions']=18+w['suffix_paid_actions'];witnesses.append(w)
                for name,expected in (('G',g),('class_oracle',sum(known)/2)):
                    actual=sum(w['terminal']['joint'] for w in witnesses if w['policy']==name)/2
                    if abs(actual-expected)>1e-10: raise AssertionError('witness differs from finite optimum')
                prior_policy=read(OLD/(parent['id']+'_policies.json'))
                # Extra image information cannot make the known-model optimum
                # change: actions and reward tables are deliberately identical.
                if any(abs(a-b)>1e-10 for a,b in zip(known,prior_policy['class_optimal_by_hypothesis'])):
                    raise AssertionError('reward/action contract changed')
                left=1500000-sum(s.states for s in solvers)
                if left<=0: raise SolverLimitV33('memo cap exhausted')
                ordinary=ExactCoverageBeliefSolverV33([models[0],models[0]],[initial[0],initial[0]],left)
                solvers.append(ordinary);counts['exact_solver_instances']+=1
                og=ordinary.uncertain(ordinary.anchor,24,initial[0],initial[0]);ok=ordinary.known(0,ordinary.anchor,24,initial[0])
                control=og>NEG/2 and abs(og-ok)<=1e-10
                mean=sum(known)/2;relative=(mean-g)/g
                row=dict(parent_id=parent['id'],G_optimal=g,class_optimal_by_hypothesis=known,
                    class_optimal_mean=mean,information_absolute=mean-g,information_relative=relative,
                    exact_search_completed=True,screening_passed=control and relative>.05,
                    main_memo_states=solver.states,ordinary_memo_states=ordinary.states,
                    identical_structure_control=dict(G=og,known=ok,passed=control),
                    independent_cue_value=g,free_geometry_reveal_value=mean,
                    algebraic_controls_not_extra_measurement=True,witnesses=witnesses,
                    actual_reconstruction_metric=False)
                policies.append(row);write(OUT,OUT/(parent['id']+'_policies.json'),row)
                print(dict(stage='policy_complete',parent=parent['id'],G=g,oracle=mean,relative=relative),flush=True)
                solver.clear();ordinary.clear()
                for m in models:m.terminal.cache_clear()
        counts['memo_states']=sum(s.states for s in solvers)
        for world in worlds:
            if world.counts['packets']!=world.counts['clean_depth_queries'] or world.counts['packets']!=world.counts['scan_queries']:
                raise AssertionError('unaccounted sensor query')
        sync_sensor_counts()
        if counts['sensor_packets']!=416 or counts['worlds']!=4 or counts['paid_step_calls']:
            raise AssertionError('formal acquisition counts differ')
        verify_sources(OUT);verify_sources(OLD);verify_inventory(OLD)
        for rel,expected in inputs.items():
            if sha(ROOT/rel)!=expected: raise ValueError('old input changed')
        result=dict(status='complete',all_pixel_prerequisites_passed=all(r['prerequisite_passed'] for r in parent_records),
            all_information_gates_passed=len(policies)==2 and all(r['screening_passed'] for r in policies),
            parents=[{k:v for k,v in r.items() if k!='hypotheses'} for r in parent_records],
            policies=[{k:v for k,v in r.items() if k!='witnesses'} for r in policies],
            counts=counts,main_tasks_used=16,source_count=len(manifest['source_sha256']),
            physical_experiment_ready=False,actual_C80_verified=False,semantic_efficacy_proven=False,
            full_architecture_advantage_proven=False,precision_thresholds_cm={'2':None,'5':None,'10':None},
            scope='clean pixel information with saved ideal V33 area/coverage reward; forced18+24 finite policy',
            elapsed_seconds=perf_counter()-started)
        write(OUT,OUT/'result.json',result);seal(OUT);print(result,flush=True)
    except BaseException as error:
        sync_sensor_counts()
        counts['memo_states']=sum(s.states for s in solvers)
        write(OUT,OUT/'failure.json',dict(status='censored' if isinstance(error,SolverLimitV33) else 'failed',
            traceback=traceback.format_exc(),counts=counts,main_tasks_used=16,elapsed_seconds=perf_counter()-started,
            exact_optimum_certified=False))
        if not (OUT/'artifact_hashes.json').exists():seal(OUT)
        raise
    finally:
        signal.alarm(0)
        for solver in solvers:solver.clear()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    else:parser.print_help()
