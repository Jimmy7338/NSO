#!/usr/bin/env python3
"""Saved V34 fixed-route audit: numpy only; no fusion, extraction or scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile
import numpy as np

from verify_pixel_information_v34 import same, calibration

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'audit_results/v34_fixed_measurement_20260918'
OUTPUT=ROOT/'audit_results/v34_fixed_measurement_review_20260918'
FRAME=('timestamp_s','depth_m','color_rgb','intrinsic','world_from_camera','semantic')
SCAN=('timestamp_s','ranges_m','angle_min_rad','angle_increment_rad','range_max_m','world_from_laser')
PACKET=('scene_id','episode_id','frame_id','action_id','frame','scan','position','heading','sensor_source','pose_source','action','collision','done')
COLORS={2:(40,100,220),3:(220,60,40)}
TAGS=('02cm','05cm','10cm')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()


def seal(folder,name,exclude=None):
    inventory=read(folder/name)
    paths={str(p.relative_to(folder)):p for p in folder.rglob('*') if p.is_file() and p.name!=name
        and (exclude is None or p.relative_to(folder).parts[0]!=exclude)}
    same(sorted(paths),sorted(inventory),'seal names')
    for key,h in inventory.items():same(sha(paths[key]),h,key)
    return inventory


def sources():
    manifest=read(SOURCE/'manifest.json')
    for name,h in manifest['source_sha256'].items():same(sha(ROOT/name),h,name)
    same(sha(SOURCE/'sources.zip'),manifest['source_archive_sha256'])
    with zipfile.ZipFile(SOURCE/'sources.zip') as z:
        same(sorted(z.namelist()),sorted(manifest['source_sha256']))
        for name,h in manifest['source_sha256'].items():same(hashlib.sha256(z.read(name)).hexdigest(),h,name)
    for name,h in manifest['input_sha256'].items():same(sha(ROOT/name),h,name)
    for name,h in read(SOURCE/'prepare_seal.json').items():same(sha(SOURCE/name),h,name)
    assert not (SOURCE/'prepare_failure.json').exists()
    return manifest


def packet_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        expected={'metadata'}|{f'{outer}__{inner}' for outer,fields in (('frame',FRAME),('scan',SCAN)) for inner in fields}
        same(sorted(z.files),sorted(expected),'packet field set')
        arrays={k:z[k].copy() for k in z.files if k!='metadata'};metadata=json.loads(z['metadata'].item())
    same(sorted(metadata),sorted(set(PACKET)-{'frame','scan'}))
    h=hashlib.sha256()
    def append(b):h.update(len(b).to_bytes(8,'big'));h.update(b)
    for key in PACKET:
        append(key.encode())
        if key in ('frame','scan'):
            for field in FRAME if key=='frame' else SCAN:
                a=np.ascontiguousarray(arrays[key+'__'+field]);assert not a.dtype.hasobject
                for b in (field.encode(),a.dtype.str.encode(),str(a.shape).encode(),a.tobytes()):append(b)
        else:append(digest(metadata[key]).encode())
    return metadata,arrays,h.hexdigest()


def observed_cue(rgb,depth):
    counts={k:int(np.all(rgb==c,axis=-1).sum()) for k,c in COLORS.items()}
    valid={str(k):int((np.all(rgb==c,axis=-1)&(depth>0)).sum()) for k,c in COLORS.items()}
    candidates=[k for k,v in counts.items() if v>=16]
    selected=candidates[0] if len(candidates)==1 and sum(n>0 for n in counts.values())==1 else None
    return dict(class_id=selected,type={2:'type_A',3:'type_B'}.get(selected),pixel_count=counts.get(selected,0),
        pixel_counts={str(k):v for k,v in counts.items()},valid_depth_pixel_counts=valid,minimum_pixels=16,
        source='public_exact_artificial_RGB_colors_only',natural_semantic_network=False)


def geometry_metadata(mesh_path):
    with np.load(mesh_path,allow_pickle=False) as z:
        same(sorted(z.files),['triangles','vertex_colors','vertices'])
        vertices=z['vertices'];triangles=z['triangles']
        assert vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all()
        assert triangles.ndim==2 and triangles.shape[1]==3 and np.issubdtype(triangles.dtype,np.integer)
        if len(triangles):assert triangles.min()>=0 and triangles.max()<len(vertices)
        faces=np.asarray(vertices[triangles],np.float64)
    areas=np.linalg.norm(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]),axis=1)/2
    valid=faces[areas>1e-12]
    canonical=np.asarray(sorted({tuple(sorted(map(tuple,f))) for f in valid}),np.float64).reshape(-1,3,3)
    c_area=np.linalg.norm(np.cross(canonical[:,1]-canonical[:,0],canonical[:,2]-canonical[:,0]),axis=1).sum()/2
    return dict(raw_triangles=len(faces),raw_area_m2=float(areas.sum()),
        degenerate_triangles=int((areas<=1e-12).sum()),exact_duplicate_triangles=len(valid)-len(canonical),
        canonical_triangles=len(canonical),canonical_area_m2=float(c_area)),hashlib.sha256(canonical.tobytes()).hexdigest()


def verify_complete_case(folder,case,config,parent):
    seal(folder,'main_seal.json','replay')
    start=read(folder/'started.json');same(start,read(SOURCE/'starts'/f'main_{case["index"]:02d}.json'))
    same(start['cumulative_main_used'],17+case['index']);assert start['counted_on_directory_creation']
    r=read(folder/'result.json');same(r['physical_case'],case)
    trace=read(folder/'trace.json');total=len(case['actions']);assert 18<=total<=42
    same(len(trace),total+1)
    same(sorted(p.name for p in (folder/'packets').iterdir()),[f'{i:03d}.npz' for i in range(total+1)])
    for paid,row in enumerate(trace):
        meta,a,packet_hash=packet_arrays(folder/'packets'/f'{paid:03d}.npz')
        pose=case['poses'][paid];action=None if paid==0 else case['actions'][paid-1]
        same(row['paid'],paid);same(row['action'],action);same(row['pose_v33'],pose)
        same(meta['action_id'],paid);same(meta['action'],action);same(meta['position'],row['position']);same(meta['heading'],row['heading'])
        same(meta['scene_id'],'v34-P00');same(meta['episode_id'],case['episode_id'])
        same(meta['frame_id'],f'{case["episode_id"]}:{paid}:{pose[0]}:{pose[1]}:{pose[2]}')
        same(meta['sensor_source'],'v34-first-hit-pixels-euclidean4-laser8-iid_025px')
        same(meta['pose_source'],'simulator-exact-discrete-pose')
        assert not meta['collision'] and not row['collision'];same(meta['done'],paid==42)
        same(float(a['frame__timestamp_s']),float(paid));same(float(a['scan__timestamp_s']),float(paid))
        assert not np.any(a['frame__semantic'])
        same(packet_hash,row['packet_sha256']);same(row['collisions_so_far'],0)
        rgb=a['frame__color_rgb'];neutral=rgb.copy()
        marker=np.logical_or.reduce([np.all(rgb==color,axis=-1) for color in COLORS.values()]);neutral[marker]=(127,127,127)
        fields=dict(depth=a['frame__depth_m'],rgb_nonsemantic=neutral,intrinsic=a['frame__intrinsic'],
            camera_pose=a['frame__world_from_camera'],ranges=a['scan__ranges_m'],laser_pose=a['scan__world_from_laser'],
            scan_calibration=np.array([a['scan__angle_min_rad'].item(),a['scan__angle_increment_rad'].item(),a['scan__range_max_m'].item()]),
            cell=np.asarray(meta['position'],np.int64),heading=np.asarray(meta['heading'],np.int64))
        calibration({**fields,'rgb':rgb},pose,parent)
        same({k:array_hash(v) for k,v in fields.items()},row['nonsemantic'])
        same(observed_cue(rgb,fields['depth']),row['cue'])
        details=row['footprint_conflict_details'];same(details['map_version'],paid+1)
        same(details['timestamp_s'],float(paid));same(details['current_cell'],meta['position'])
        same(details['conflict'],row['footprint_conflict'])
    same(digest(trace),r['trace_sha256'])
    same(digest([{k:t[k] for k in ('paid','action','pose_v33')} for t in trace]),r['trajectory_sha256'])
    same(r['footprint_conflict_count'],sum(t['footprint_conflict'] for t in trace))
    freeze=read(folder/'prediction_freeze.json')
    expected={'started.json','trace.json'}|{f'packets/{i:03d}.npz' for i in range(total+1)}|{
        f'{stage}_{name}' for stage in ('prefix','final') for name in ('raw.npz','extracted.npz','maps.npz','crop.json')}
    same(sorted(freeze['artifact_sha256']),sorted(expected),'complete pre-GT prediction freeze')
    for name,h in freeze['artifact_sha256'].items():same(sha(folder/name),h,name)
    assert freeze['before_first_quality_or_GT_reference_access'];same(freeze['stages'],['prefix','final'])
    same(freeze['packet_count'],total+1);same(freeze['public_roi_sha256'],digest(config['acquisition']['public_bounds']))
    with np.load(folder/'evaluation_floor.npz',allow_pickle=False) as z:
        same(sorted(z.files),['declared_grid_cells','reachable','safe'])
        reachable,safe,grid=z['reachable'],z['safe'],z['declared_grid_cells']
    same(list(reachable.shape),config['acquisition']['raster_shape'])
    assert reachable.dtype==bool and safe.dtype==bool and np.all(~reachable|safe) and reachable.any()
    shift=np.asarray(config['acquisition']['translation']);height=reachable.shape[0]
    expected_grid=[]
    for x,y in parent['nav_cells']:
        expected_grid.append([height-1-int(np.floor((y+shift[1])/.2)),int(np.floor((x+shift[0])/.2))])
    np.testing.assert_array_equal(grid,np.asarray(expected_grid));assert np.all(reachable[grid[:,0],grid[:,1]])
    stages={}
    for stage,paid in (('prefix',18),('final',total)):
        saved=r['stages'][stage];m=saved['measurement'];crop=read(folder/f'{stage}_crop.json')
        with np.load(folder/f'{stage}_maps.npz',allow_pickle=False) as z:belief=z['belief']
        assert np.isin(belief,[-1,0,1]).all();same(array_hash(belief),saved['belief_sha256'])
        C=float(np.mean(belief[reachable]!=-1));Cg=float(np.mean(belief[grid[:,0],grid[:,1]]!=-1))
        same(C,m['C_map']);same(Cg,saved['C_grid_measured']);same(len(grid),saved['C_grid_denominator'])
        same(int(reachable.sum()),saved['full_reachable_raster_denominator']);same(digest(crop),saved['crop_audit_sha256'])
        raw_info,_=geometry_metadata(folder/f'{stage}_raw.npz');pred_info,pred_hash=geometry_metadata(folder/f'{stage}_extracted.npz')
        same(raw_info,crop['raw']);same(pred_info,m['prediction']);same(pred_hash,m['prediction_geometry_sha256'])
        same(pred_hash,crop['kept_geometry_sha256']);same(crop['kept_area_m2'],pred_info['canonical_area_m2'])
        same(crop['public_bounds'],config['acquisition']['public_bounds'])
        same(crop['padded_roi'],(np.asarray(crop['public_bounds'])+np.array([[-.2]*3,[.2]*3])).tolist())
        same(crop['raw']['canonical_area_m2'],crop['clipped_area_m2']+crop['outside_area_m2'])
        same(crop['clipped_area_m2'],crop['ground_removed_area_m2']+crop['output_normalization']['raw_area_m2'])
        assert not crop['outside_roi_geometry_penalized_by_precision'] and not crop['truth_input']
        returned=trace[paid]['pose_v33']==case['poses'][0]
        eligible=C>=.8 and returned and paid<=42
        same(m['returned'],returned);same(m['collisions'],0);assert not m['failed']
        same(m['paid_actions'],paid);same(m['budget'],42);same(m['budget_compliant'],paid<=42);same(m['eligible'],eligible)
        same(m['reference']['sample_count'],32768);same(m['reference']['prediction_seed'],340918);same(m['reference']['reference_seed'],340919)
        same(m['main_threshold_m'],.05)
        for tag in TAGS:
            p,rec=m[tag]['precision'],m[tag]['recall'];assert 0<=p<=1 and 0<=rec<=1
            f=2*p*rec/(p+rec) if p+rec else 0.
            same(m[tag]['f1'],f);same(m[tag]['joint'],C*f)
        same(m['main_joint'],m['05cm']['joint'])
        stages[stage]=dict(C_map=C,C_grid=Cg,F1_05=m['05cm']['f1'],J05=m['05cm']['joint'],eligible=eligible,
            ROI_outside_area_m2=crop['outside_area_m2'],ground_removed_area_m2=crop['ground_removed_area_m2'])
    same(r['paid_actions'],total);same(r['budget'],42);same(r['collisions'],0);assert r['returned']
    same(r['cumulative_main_used'],17+case['index']);same(r['original_quota'],16)
    for flag in ('inference_performed','adaptive_policy_executed','full_architecture_executed'):assert r[flag] is False
    recognized={tuple(t['pose_v33'][:2]) for t in trace[:19] if t['cue']['class_id']==case['hypothesis']+2}
    wrong=[t['paid'] for t in trace[:19] if t['cue']['class_id'] not in (None,case['hypothesis']+2)]
    same(r['prefix_cue'],dict(distinct_recognized_positions=[list(p) for p in sorted(recognized)],
        contradictory_actions=wrong,passed=len(recognized)>=2 and not wrong))
    expected_counts=dict(worlds=1,sensor_packets=total+1,clean_depth_queries=total+1,scan_queries=total+1,
        paid_actions=total,mapper_updates=total+1,TSDF_integrations=total+1,mapper_mesh_extractions=2,evaluation_stages=2,learned_reconstruction_calls=0)
    same(r['counts'],expected_counts)
    stop=not stages['final']['eligible'] or not r['prefix_cue']['passed'];same((folder/'stop.json').exists(),stop)
    if stop:
        stopped=read(folder/'stop.json');assert stopped['stop_remaining_acquisitions'] and stopped['negative_result_retained']
        same(stopped['result_sha256'],sha(folder/'result.json'));same(stopped['actual_C80_passed'],stages['final']['C_map']>=.8)
    return r,trace,dict(index=case['index'],main_pid=start['pid'],stages=stages,counts=expected_counts,
        prefix_cue_passed=r['prefix_cue']['passed'],prediction_artifacts_before_GT=len(expected),stop=stop)


def verify_config(config,manifest):
    scene=read(ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json')
    parent=next(p for p in scene['parents'] if p['id']=='P00')
    policies=read(ROOT/'audit_results/v34_pixel_information_20260918/P00_policies.json')
    same(sha(ROOT/'audit_results/v34_pixel_information_20260918/P00_policies.json'),config['source_policy_sha256'])
    a=config['acquisition'];same(a['parent'],'P00');same(a['budget'],42);same(a['forced_prefix'],18)
    same(a['noise_model'],'iid_025px');same(a['noise_seed'],1901)
    bounds=np.asarray([b for h in parent['hypotheses'] for asset in h['assets'] for b in asset['boxes']]).reshape(-1,3,2)
    nav=np.asarray(parent['nav_cells']);shift=np.r_[-(nav.min(axis=0)-.5),0.]
    public=(np.stack([bounds[:,:,0].min(axis=0),bounds[:,:,1].max(axis=0)])+shift).tolist()
    same(a['public_bounds'],public);same(a['translation'],shift.tolist())
    identities={};physical=config['physical_cases'];cells=config['cells'];assert len(cells)==4 and 1<=len(physical)<=4
    for i,cell in enumerate(cells):
        h,policy=((0,'G'),(0,'class_oracle'),(1,'G'),(1,'class_oracle'))[i]
        same(cell['cell_index'],i);same(cell['hypothesis'],h);same(cell['policy'],policy)
        witness=next(w for w in policies['witnesses'] if w['actual_hypothesis']==h and w['policy']==policy)
        same(cell['witness_sha256'],digest(witness));actions=parent['prefix_actions']+witness['suffix_actions']
        identity=digest(dict(hypothesis=h,actions=actions,acquisition=a));duplicate=identity in identities
        if not duplicate:identities[identity]=len(identities)
        same(cell['alias'],duplicate);same(cell['independent_sample'],not duplicate);same(cell['physical_case'],identities[identity])
        case=physical[cell['physical_case']];same(case['hypothesis'],h);same(case['actions'],actions)
        same(case['acquisition_identity_sha256'],identity);same(case['poses'][18:],witness['suffix_poses'])
        pose=list(parent['anchor']);path=[pose.copy()]
        for action in actions:
            if action=='forward':
                dx,dy=((0,1),(1,0),(0,-1),(-1,0))[pose[2]];pose[0]+=dx;pose[1]+=dy
            else:assert action in ('left','right');pose[2]=(pose[2]+(1 if action=='right' else -1))%4
            path.append(pose.copy())
        same(path,case['poses']);same(path[0],path[-1]);same(path[0],path[18]);assert len(actions)<=42
    same(len(identities),len(physical));same(len(physical),manifest['physical_case_count'])
    same(manifest['comparison_cell_count'],4);same(manifest['aliases'],[c for c in cells if c['alias']])
    same(config['historical_main_used'],16);same(config['total_main_limit'],36)
    same(read(SOURCE/'static_routes.json'),[dict(**c,actions=physical[c['physical_case']]['actions'],poses=physical[c['physical_case']]['poses']) for c in cells])
    return parent


def inspect():
    seal(SOURCE,'final_seal.json');manifest=sources();config=read(SOURCE/'config.json');parent=verify_config(config,manifest)
    analysis=read(SOURCE/'result.json');assert analysis['status'] in ('complete','stopped')
    folders=sorted(p for p in SOURCE.glob('case*') if p.is_dir())
    same([p.name for p in folders],[f'case{i:02d}' for i in range(len(folders))]);assert 0<len(folders)<=4
    complete={};traces={};summaries=[];main_counts={};replay_counts={};replayed={};pids=[]
    for i,folder in enumerate(folders):
        if not (folder/'main_seal.json').exists():
            assert any(f['case']==i and f['kind']=='main_unsealed' for f in analysis['failures']);continue
        seal(folder,'main_seal.json','replay')
        if (folder/'failure.json').exists():
            f=read(folder/'failure.json');assert f['stop_remaining_acquisitions'] and f['retry_allowed'] is False
            main_counts[i]=f['counts'];assert i==len(folders)-1;continue
        r,trace,summary=verify_complete_case(folder,config['physical_cases'][i],config,parent)
        complete[i]=r;traces[i]=trace;main_counts[i]=r['counts'];summaries.append(summary);pids.append(summary['main_pid'])
        replay=folder/'replay';seal(replay,'seal.json')
        rs=read(replay/'started.json');same(rs['main_pid'],summary['main_pid']);assert rs['pid']!=rs['main_pid'];pids.append(rs['pid'])
        if (replay/'failure.json').exists():
            f=read(replay/'failure.json');replay_counts[i]=f['counts'];assert i==len(folders)-1;continue
        rr=read(replay/'result.json');assert rr['passed'] and rr['all_packet_field_bytes_equal'] and rr['all_raw_extracted_mesh_map_array_bytes_equal'] and rr['complete_result_equal']
        same(rr['main_pid'],summary['main_pid']);same(rr['replay_pid'],rs['pid']);same(rr['main_result_sha256'],sha(folder/'result.json'))
        same(rr['counts'],r['counts']);same(rr['packet_count'],len(trace));replayed[i]=rr;replay_counts[i]=rr['counts']
        pc=read(replay/'prediction_comparison.json');assert pc['passed'] and pc['both_raw_and_extracted_meshes_equal'] and pc['all_mapper_arrays_equal'] and pc['all_trace_fields_equal']
        same(pc['main_prediction_freeze_sha256'],sha(folder/'prediction_freeze.json'));same(pc['packet_count'],len(trace))
        summary['replay_pid']=rs['pid'];summary['independent_replay_passed']=True
        if summary['stop']:assert i==len(folders)-1,'acquisition continued after stop'
        if i>0:assert i-1 in replayed and not summaries[-2]['stop'],'previous replay/qualification gate'
    # exec invocations use separate PID namespaces and can reuse numeric PIDs.
    # Each bound main/replay pair was checked above; global uniqueness is not
    # a valid independence criterion across these isolated invocations.
    rejected=[]
    for path in sorted(SOURCE.glob('replay_launch_rejected_*.json')):
        entry=read(path)
        same(entry['status'],'rejected_before_replay_start')
        for key in ('new_worlds','new_packets','new_main_tasks','new_replays_started'):same(entry[key],0)
        assert entry['source_changed'] is False
        index=entry['case'];assert index in complete
        same(entry['main_local_pid'],read(SOURCE/f'case{index:02d}'/'started.json')['pid'])
        same(entry['rejected_local_pid'],entry['main_local_pid'])
        rejected.append(dict(path=str(path.relative_to(SOURCE)),sha256=sha(path),receipt=entry))
    full=len(complete)==len(config['physical_cases']);same(analysis['full_matrix_collected'],full)
    for cell,saved in zip(config['cells'],analysis['table']):
        r=complete.get(cell['physical_case'])
        for k,v in cell.items():same(v,saved[k])
        same(saved['measured'],r is not None);same(saved['stages'],None if r is None else r['stages'])
        delta=None if r is None else {t:r['stages']['final']['measurement'][t]['joint']-r['stages']['prefix']['measurement'][t]['joint'] for t in TAGS}
        same(saved['prefix_to_final_joint_change'],delta);same(saved['independent_replay_passed'],cell['physical_case'] in replayed)
    pairings=[]
    for policy in ('G','class_oracle'):
        ids=[next(c['physical_case'] for c in config['cells'] if c['hypothesis']==h and c['policy']==policy) for h in (0,1)]
        first=None
        if all(i in traces for i in ids):
            left,right=[traces[i] for i in ids]
            same([r['pose_v33'] for r in left[:19]],[r['pose_v33'] for r in right[:19]])
            same([r['action'] for r in left[:19]],[r['action'] for r in right[:19]])
            first={k:next((n for n,(a,b) in enumerate(zip(left,right)) if a['nonsemantic'][k]!=b['nonsemantic'][k]),None) for k in left[0]['nonsemantic']}
        pairings.append(dict(policy=policy,physical_cases=ids,first_observed_field_difference=first,
            actual_prefix_nonsemantic_equal=None if first is None else all(v is None or v>18 for v in first.values())))
    same(pairings,analysis['actual_noisy_prefix_pairings'])
    comparisons={}
    if full:
        for tag in TAGS:
            vals={(c['hypothesis'],c['policy']):complete[c['physical_case']]['stages']['final']['measurement'][tag]['joint'] for c in config['cells']}
            g=sum(vals[h,'G'] for h in (0,1))/2;o=sum(vals[h,'class_oracle'] for h in (0,1))/2
            delta=[vals[h,'class_oracle']-vals[h,'G'] for h in (0,1)]
            comparisons[tag]=dict(G_mean_joint=g,class_oracle_mean_joint=o,paired_mean_joint_difference=sum(delta)/2,
                paired_joint_differences=delta,relative_mean_joint_difference=(sum(delta)/2)/g if g else None)
    same(comparisons,analysis['prescribed_paired_comparison'])
    eligible=full and all(r['stages']['final']['measurement']['eligible'] for r in complete.values())
    cue_ok=full and all(r['prefix_cue']['passed'] for r in complete.values());paired=all(p['actual_prefix_nonsemantic_equal'] is True for p in pairings)
    replay_ok=len(replayed)==len(config['physical_cases'])
    positive=bool(comparisons and comparisons['05cm']['paired_mean_joint_difference']>0 and all(comparisons[t]['paired_mean_joint_difference']>=0 for t in ('02cm','10cm')))
    same(analysis['all_final_eligible'],eligible);same(analysis['all_replays_passed'],replay_ok)
    same(analysis['controlled_marker_prerequisite_passed'],cue_ok);same(analysis['all_prefix_nonsemantic_pairings_passed'],paired)
    gate=eligible and replay_ok and cue_ok and paired and positive and not analysis['failures']
    same(analysis['measured_fixed_route_gate_passed'],gate)
    same(analysis['actual_C80_verified'],full and all(r['stages']['final']['measurement']['C_map']>=.8 for r in complete.values()))
    stopped=bool(analysis['failures'] or any(not r['stages']['final']['measurement']['eligible'] or not r['prefix_cue']['passed'] for r in complete.values()))
    same(analysis['status'],'stopped' if stopped else 'complete');assert full or stopped
    def total(rows):return {k:sum(r[k] for r in rows.values()) for k in next(iter(rows.values()))} if rows else {}
    same(analysis['acquisition_counts'],total(main_counts));same(analysis['replay_counts'],total(replay_counts))
    same(analysis['new_main_tasks'],len(folders));same(analysis['total_main_used'],16+len(folders));same(analysis['main_limit'],36)
    for flag in ('adaptive_G_or_ANS_advantage_proven','full_architecture_advantage_proven','natural_semantic_innovation_proven'):assert analysis[flag] is False
    same(analysis['autonomous_trials_started'],0);assert analysis['aliases_are_not_independent_samples']
    return dict(status='passed',saved_evidence_verified=True,source_result_sha256=sha(SOURCE/'result.json'),
        source_final_seal_sha256=sha(SOURCE/'final_seal.json'),source_manifest_sha256=sha(SOURCE/'manifest.json'),
        observed_batch_status=analysis['status'],full_matrix_collected=full,measured_fixed_route_gate_passed=gate,
        new_main_tasks=len(folders),total_main_used=16+len(folders),physical_summaries=summaries,
        declared_comparison_cells=4,aliases_not_independent=True,comparison= comparisons,
        pairings=pairings,source_acquisition_counts=total(main_counts),source_replay_counts=total(replay_counts),
        prestart_replay_rejections=rejected,prestart_rejections_are_not_executed_replays=True,
        this_audit_counts=dict(worlds=0,sensor_queries=0,mapper_updates=0,TSDF_integrations=0,surface_evaluations=0,new_main_tasks=0),
        limits=['Surface distances/P/R are not rescored; mesh bytes, F1/J arithmetic and independent replay receipts verified.',
        'Clean depth has saved hash only: no independent clean-render recomputation.',
        'GT floor masks are loaded to recompute coverage, not recomputed from geometry.',
        'Prediction freeze checks inventory and frozen execution ordering; no independent trusted wall-clock attestation.',
        'Saved fixed branches are not adaptive measured G or ANS execution.',
        'Numeric PIDs may repeat across exec namespaces; bound main/replay pair IDs differ and receipts identify the exact source result.'])


def write(path,value):
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())+len(payload)>2*1024**2 or shutil.disk_usage(ROOT).free-len(payload)<64*1024**2:raise RuntimeError('audit cap/reserve')
    with path.open('xb') as f:f.write(payload)


def run():
    if OUTPUT.exists():raise FileExistsError('review exists; no implicit rerun')
    if not (SOURCE/'final_seal.json').is_file():raise ValueError('wait for final analyzed seal')
    if shutil.disk_usage(ROOT).free<64*1024**2:raise RuntimeError('64 MiB reserve')
    OUTPUT.mkdir()
    try:
        deps=[Path(__file__),ROOT/'scripts/verify_pixel_information_v34.py']
        frozen={str(p.relative_to(ROOT)):sha(p) for p in deps}
        write(OUTPUT/'manifest.json',dict(status='before_saved_evidence_review',source_sha256=frozen,
            source_final_seal_sha256=sha(SOURCE/'final_seal.json')))
        result=inspect();sources();seal(SOURCE,'final_seal.json')
        for name,h in frozen.items():same(sha(ROOT/name),h)
        write(OUTPUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('physical_summaries','pairings','limits')},allow_nan=False),flush=True)
    except BaseException:
        write(OUTPUT/'failure.json',dict(status='failed',traceback=traceback.format_exc(),no_new_measurement_or_score=True));raise
    finally:
        write(OUTPUT/'artifact_hashes.json',{str(p.relative_to(OUTPUT)):sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:run()
    else:parser.print_help()
