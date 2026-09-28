#!/usr/bin/env python3
"""Independently verify saved V34 pixels and witnesses; no renderer or DP."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'audit_results/v34_pixel_information_20260918'
OLD=ROOT/'audit_results/v33_direction_information_r1_20260917'
OUTPUT=ROOT/'audit_results/v34_pixel_information_review_20260918'
FIELDS={'depth','rgb','intrinsic','camera_pose','ranges','laser_pose','scan_calibration','cell','heading'}
COLORS={2:(40,100,220),3:(220,60,40)}
NEUTRAL=(127,127,127)
CAP=2*1024**2
RESERVE=64*1024**2


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())


def array_digest(**arrays):
    h=hashlib.sha256()
    for name in sorted(arrays):
        a=np.ascontiguousarray(arrays[name])
        if a.dtype.hasobject: raise ValueError('object array forbidden')
        for part in (name.encode(),a.dtype.str.encode(),str(a.shape).encode(),a.tobytes()):
            h.update(len(part).to_bytes(8,'big'));h.update(part)
    return h.hexdigest()


def same(actual,expected,label=''):
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected),label
        for k,v in expected.items(): same(actual[k],v,label+'/'+str(k))
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected),label
        for i,(a,b) in enumerate(zip(actual,expected)):same(a,b,label+'/'+str(i))
    elif isinstance(expected,float):
        assert np.isfinite(expected) and np.isfinite(actual) and abs(actual-expected)<=1e-10,label
    else: assert actual==expected,(label,actual,expected)


def verify_package(folder):
    inventory=read(folder/'artifact_hashes.json')
    paths={str(p.relative_to(folder)):p for p in folder.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    assert set(paths)==set(inventory),'inventory member set'
    for name,h in inventory.items():assert sha(paths[name])==h,name
    manifest=read(folder/'manifest.json')
    for name,h in manifest['source_sha256'].items():assert sha(ROOT/name)==h,name
    assert sha(folder/'sources.zip')==manifest['source_archive_sha256']
    with zipfile.ZipFile(folder/'sources.zip') as z:
        assert set(z.namelist())==set(manifest['source_sha256'])
        for name,h in manifest['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    return dict(manifest_sha256=sha(folder/'manifest.json'),inventory_sha256=sha(folder/'artifact_hashes.json'),
        result_sha256=sha(folder/'result.json'),files=len(inventory),sources=len(manifest['source_sha256']))


def cue(rgb,depth,report_depth):
    counts={code:int(np.all(rgb==color,axis=-1).sum()) for code,color in COLORS.items()}
    supported=[k for k,n in counts.items() if n>=16]
    selected=supported[0] if len(supported)==1 and sum(n>0 for n in counts.values())==1 else None
    valid={str(k):int((np.all(rgb==v,axis=-1)&(depth>0)).sum()) for k,v in COLORS.items()}
    for k,n in counts.items():assert valid[str(k)]==n,'clean marker without valid depth'
    return dict(class_id=selected,type={2:'type_A',3:'type_B'}.get(selected),pixel_count=counts.get(selected,0),
        pixel_counts={str(k):n for k,n in counts.items()},valid_depth_pixel_counts=valid if report_depth else None,
        minimum_pixels=16,source='public_exact_artificial_RGB_colors_only',natural_semantic_network=False)


def calibration(fields,pose,parent):
    x,y,heading=pose
    cells=np.asarray(parent['nav_cells']);low=cells.min(axis=0)-.5;high=cells.max(axis=0)+.5
    height,width=map(int,np.rint((high-low)[::-1]/.2))
    xy=np.array([x,y])-low
    expected_cell=np.array([height-1-int(np.floor(xy[1]/.2)),int(np.floor(xy[0]/.2))],np.int64)
    assert np.array_equal(fields['cell'],expected_cell)
    assert int(fields['heading'])==heading
    np.testing.assert_array_equal(fields['intrinsic'],[[48.,0.,47.5],[0.,48.,35.5],[0.,0.,1.]])
    front=np.array([(0,1,0),(1,0,0),(0,-1,0),(-1,0,0)][heading],float)
    camera=np.eye(4);camera[:3,:3]=np.column_stack((np.cross(front,[0.,0.,1.]),[0.,0.,-1.],front))
    camera[:3,3]=[(expected_cell[1]+.5)*.2,(height-expected_cell[0]-.5)*.2,.9]
    np.testing.assert_allclose(fields['camera_pose'],camera,rtol=0,atol=1e-10)
    laser=np.eye(4);laser[:3,0]=front;laser[:3,1]=-camera[:3,0];laser[:3,3]=camera[:3,3];laser[2,3]=.25
    np.testing.assert_allclose(fields['laser_pose'],laser,rtol=0,atol=1e-10)
    np.testing.assert_allclose(fields['scan_calibration'],[-np.pi,2*np.pi/180,8.],rtol=0,atol=1e-10)
    d=fields['depth'];r=fields['ranges'];rgb=fields['rgb']
    assert d.shape==(72,96) and d.dtype==np.float32 and np.isfinite(d).all() and np.min(d)>=0
    assert rgb.shape==(72,96,3) and rgb.dtype==np.uint8
    assert r.shape==(180,) and r.dtype==np.float32 and np.isfinite(r).all() and np.min(r)>0 and np.max(r)<=8.
    v,u=np.mgrid[:72,:96];length=np.sqrt(1+((u-47.5)/48)**2+((v-35.5)/48)**2)
    assert np.max(d*length)<=4.+1e-6 and np.all(d[(d>0)]>.15)
    return [height,width]


def pair_info(tables,signatures):
    a,b=tables
    for key in ('poses','edges','anchor','prefix_nodes'):same(a[key],b[key],key)
    anchor=a['anchor'];dist={anchor:0};previous={};q=deque([anchor])
    while q:
        node=q.popleft()
        for action,nxt in a['edges'][node]:
            if nxt not in dist:dist[nxt]=dist[node]+1;previous[nxt]=(node,action);q.append(nxt)
    different=[n for n in range(len(a['poses'])) if signatures[0][n]!=signatures[1][n]]
    first=min((dist[n] for n in different if n in dist),default=None);witnesses=[]
    for node in different:
        if dist.get(node)!=first:continue
        cursor=node;actions=[]
        while cursor!=anchor:cursor,action=previous[cursor];actions.append(action)
        witnesses.append(dict(node=node,pose=a['poses'][node],actions=actions[::-1]))
    bad=[i for i,n in enumerate(a['prefix_nodes']) if signatures[0][n]!=signatures[1][n]]
    return dict(prefix_equal=not bad,prefix_different_indices=bad,differing_pose_count=len(different),
        first_information_action_layer=first,first_information_witnesses=witnesses,
        all_poses_connected=len(dist)==len(a['poses']))


def reward(table,mask):
    bits=table['target_bits'];refs=table['exact_reference_vertical_areas'];area={k:0. for k in refs}
    assert len(table['targets'])==bits and 0<=mask<(1<<(bits+len(table['floor_cells'])))
    for i,t in enumerate(table['targets']):
        if mask&(1<<i):area[str(t['owner'])]+=t['area']
    fractions={k:min(1.,max(0.,v/refs[k])) for k,v in area.items()}
    coverage=(mask>>bits).bit_count()/len(table['floor_cells']);surface=sum(fractions.values())/len(fractions)
    return dict(coverage=coverage,surface=surface,joint=coverage*surface,feasible=coverage>=.8-1e-12,
        per_asset_surface=fractions)


def verify_witness(w,tables,signatures):
    h=w['actual_hypothesis'];table=tables[h];node=table['anchor'];mask=0
    for n in table['prefix_nodes']:mask|=int(table['observed_masks_hex'][n],16)
    same(hex(mask),table['prefix_mask_hex'],'initial mask')
    assert len(table['prefix_nodes'])==19
    actions=w['suffix_actions'];assert len(actions)<=24
    same(w['suffix_paid_actions'],len(actions));same(w['total_paid_actions'],18+len(actions))
    nodes=[node];first=0 if signatures[0][node]!=signatures[1][node] else None
    sources=[];events=[]
    for step,action in enumerate(actions):
        sources.append('revealed_geometry' if first is not None else 'geometry_belief' if w['policy']=='G' else 'class_hypothesis')
        transitions=dict(table['edges'][node]);assert action in transitions
        node=transitions[action];nodes.append(node);mask|=int(table['observed_masks_hex'][node],16)
        revealed=signatures[0][node]!=signatures[1][node]
        if first is None and revealed:first=step+1
        events.append(dict(action=action,node=node,remaining=23-step,geometric_hypothesis_revealed=revealed))
    same(nodes,w['suffix_nodes']);same([table['poses'][n] for n in nodes],w['suffix_poses'])
    same(events,w['events']);same(sources,w['decision_sources']);same(first,w['first_information_suffix_action'])
    same(hex(mask),w['final_observed_mask_hex']);assert node==table['anchor']
    assert w['returned_exact_pose'] and w['budget_compliant']
    result=reward(table,mask)
    for k,v in result.items():same(v,w['terminal'][k],k)
    same(result['feasible'],w['actual_coverage_qualified'])
    if w['policy'] in ('G','class_oracle'):assert result['feasible'] and w['initial_hint'] is None
    else:assert w['policy']=='swapped_class_with_correction' and w['initial_hint']==1-h
    return result


def verify_policies(row,tables,signatures,prior):
    witnesses=row['witnesses'];assert len(witnesses)==6
    expected={(h,p) for h in (0,1) for p in ('G','class_oracle','swapped_class_with_correction')}
    assert {(w['actual_hypothesis'],w['policy']) for w in witnesses}==expected
    values={(w['actual_hypothesis'],w['policy']):verify_witness(w,tables,signatures) for w in witnesses}
    g=sum(values[h,'G']['joint'] for h in (0,1))/2
    known=[values[h,'class_oracle']['joint'] for h in (0,1)];mean=sum(known)/2
    for k,v in dict(G_optimal=g,class_optimal_by_hypothesis=known,class_optimal_mean=mean,
        information_absolute=mean-g,information_relative=(mean-g)/g,independent_cue_value=g,
        free_geometry_reveal_value=mean).items():same(v,row[k],k)
    same(known,prior['class_optimal_by_hypothesis'],'known model reward unchanged')
    control=row['identical_structure_control'];same(control['G'],control['known'])
    assert control['passed'];same(bool((mean-g)/g>.05),row['screening_passed'])
    # A G history cannot choose different actions while all paid observations agree.
    gpaths=[next(w for w in witnesses if w['policy']=='G' and w['actual_hypothesis']==h) for h in (0,1)]
    pos=[t['anchor'] for t in tables];common=0
    for step in range(max(len(w['suffix_actions']) for w in gpaths)+1):
        if signatures[0][pos[0]]!=signatures[1][pos[1]]:break
        following=[w['suffix_actions'][step] if step<len(w['suffix_actions']) else None for w in gpaths]
        same(following[0],following[1],'G pre-revelation action independence')
        if following[0] is None:break
        pos=[dict(t['edges'][n])[following[h]] for h,(t,n) in enumerate(zip(tables,pos))];common+=1
    return dict(G_mean=g,oracle_mean=mean,information_relative=(mean-g)/g,
        witnesses_checked=6,G_common_paid_actions_before_revelation=common,
        optimum_independently_resolved=False)


def inspect_saved():
    receipts={'V34':verify_package(INPUT),'V33r1':verify_package(OLD)}
    result=read(INPUT/'result.json');assert result['status']=='complete'
    manifest=read(INPUT/'manifest.json')
    for name,h in manifest['input_sha256'].items():assert sha(ROOT/name)==h,name
    cfg=read(ROOT/'configs/virtual3d/v33_direction_scene_r1_20260917.json')
    summaries=[];frames_total=0;policies_total=0
    for parent in cfg['parents']:
        name=parent['id'];record=read(INPUT/(name+'_observations.json'));old=read(OLD/(name+'_geometry.json'))
        tables=old['tables'];signatures=[];cues=[];frame_cues=[]
        for h,(saved,table) in enumerate(zip(record['hypotheses'],tables)):
            assert saved['hypothesis']==table['hypothesis']==h
            assert saved['pixels_file']==f'{name}_h{h}_pixels.npz'
            n=len(table['poses']);assert n==[96,112][name=='P01'] and len(saved['frames'])==n
            with np.load(INPUT/saved['pixels_file'],allow_pickle=False) as z:
                assert set(z.files)==FIELDS
                arrays={k:z[k] for k in z.files};assert all(len(a)==n for a in arrays.values())
                sigs=[];cs=[]
                for node,pose in enumerate(table['poses']):
                    raw={k:a[node] for k,a in arrays.items()};frame=saved['frames'][node]
                    same(frame['node'],node);same(frame['pose'],pose)
                    shape=calibration(raw,pose,parent);same(shape,saved['raster_shape'])
                    neutral=raw['rgb'].copy()
                    marker=np.logical_or.reduce([np.all(raw['rgb']==color,axis=-1) for color in COLORS.values()])
                    neutral[marker]=NEUTRAL
                    geom={k:v for k,v in raw.items() if k!='rgb'};geom['rgb_nonsemantic']=neutral
                    sig=array_digest(**geom);same(sig,frame['geometric_sha256'])
                    same(array_digest(**raw),frame['raw_sha256'])
                    same({k:array_digest(value=v) for k,v in geom.items()},frame['fields_sha256'])
                    observed=cue(raw['rgb'],raw['depth'],frame['cue']['valid_depth_pixel_counts'] is not None)
                    same(observed,frame['cue']);sigs.append(sig);cs.append(observed);frames_total+=1
            signatures.append(sigs);frame_cues.append(cs)
            recognized=[i for i in table['prefix_nodes'] if cs[i]['pixel_count']>=16 and cs[i]['type']==('type_A','type_B')[h]]
            xy=sorted({tuple(table['poses'][i][:2]) for i in recognized})
            contradictions=sum(cs[i]['pixel_count']>=16 and cs[i]['type'] not in (None,('type_A','type_B')[h]) for i in table['prefix_nodes'])
            cues.append(dict(hypothesis=h,prefix_recognized_frames=len(recognized),distinct_positions=len(xy),
                positions=[list(p) for p in xy],contradictions=contradictions,passed=len(xy)>=2 and not contradictions))
            assert saved['actual_map_coverage'] is None
            same(saved['world_counts'],dict(worlds=1,packets=n,clean_depth_queries=n,scan_queries=n,step_calls=0))
        pairing=pair_info(tables,signatures);same(pairing,record['pair']);same(cues,record['prefix_cue_checks'])
        same(old['pair']['first_information_action_layer'],record['previous_ideal_first_information_layer'])
        first=pairing['first_information_action_layer']
        same(first is not None and first<old['pair']['first_information_action_layer'],record['information_earlier_than_v33'])
        raster_equal=record['hypotheses'][0]['static_reachable_sha256']==record['hypotheses'][1]['static_reachable_sha256']
        same(raster_equal,record['paired_raster_denominator'])
        prereq=pairing['prefix_equal'] and pairing['all_poses_connected'] and all(c['passed'] for c in cues) and raster_equal
        same(prereq,record['prerequisite_passed'])
        entry=dict(parent=name,frames=sum(len(x) for x in signatures),pair=pairing,prefix_cue_checks=cues,
            prerequisites_passed=prereq,static_raster_hash_recomputed=False,
            static_raster_limit='NPZ omits mask; saved count/hash are source-backed assertions, not independently recomputed')
        policy_path=INPUT/(name+'_policies.json')
        if policy_path.exists():
            p=read(policy_path);entry['policy_checks']=verify_policies(p,tables,signatures,read(OLD/(name+'_policies.json')))
            matching=next(x for x in result['policies'] if x['parent_id']==name)
            same({k:v for k,v in p.items() if k!='witnesses'},matching);policies_total+=1
        same({k:v for k,v in record.items() if k!='hypotheses'},next(x for x in result['parents'] if x['parent_id']==name))
        summaries.append(entry)
    assert frames_total==416
    for key,value in dict(worlds=4,sensor_packets=416,clean_depth_queries=416,scan_queries=416,
        saved_reward_models=4,mapper_updates=0,TSDF_integrations=0,Q_evaluations=0,new_main_tasks=0,paid_step_calls=0).items():same(value,result['counts'][key],key)
    assert result['counts']['memo_states']<=1500000 and result['counts']['exact_solver_instances']==2*policies_total
    same(all(p['prerequisites_passed'] for p in summaries),result['all_pixel_prerequisites_passed'])
    gates=policies_total==2 and all(p['policy_checks']['information_relative']>.05 for p in summaries)
    same(gates,result['all_information_gates_passed'])
    assert result['main_tasks_used']==16
    for flag in ('physical_experiment_ready','actual_C80_verified','semantic_efficacy_proven','full_architecture_advantage_proven'):assert result[flag] is False
    same(result['precision_thresholds_cm'],{'2':None,'5':None,'10':None})
    return dict(status='passed',saved_evidence_verified=True,
        source_result_sha256=sha(INPUT/'result.json'),source_manifest_sha256=sha(INPUT/'manifest.json'),
        source_inventory_sha256=sha(INPUT/'artifact_hashes.json'),
        input_receipts=receipts,frames_checked=frames_total,policy_parents_checked=policies_total,
        parents=summaries,all_information_gates_passed=gates,
        this_verification_counts=dict(worlds=0,renderer_queries=0,DP_instances=0,mapper_updates=0,TSDF_integrations=0,Q_evaluations=0,new_main_tasks=0),
        limits=['Byte/data/math verification only; no independent renderer reproduction or exact-optimality solve.',
                'Saved static raster masks unavailable: count/hash equality checked but mask geometry not recomputed.'])


def write(path,value):
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    total=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
    if total+len(payload)>CAP or shutil.disk_usage(ROOT).free-len(payload)<RESERVE:raise RuntimeError('cap/reserve')
    with path.open('xb') as f:f.write(payload)


def run():
    if OUTPUT.exists():raise FileExistsError('unique verifier output already exists')
    if not (INPUT/'artifact_hashes.json').exists() or read(INPUT/'result.json')['status']!='complete':
        raise ValueError('wait for sealed complete main batch')
    if shutil.disk_usage(ROOT).free<RESERVE:raise RuntimeError('64 MiB reserve required')
    OUTPUT.mkdir()
    try:
        source_sha=sha(Path(__file__))
        write(OUTPUT/'manifest.json',dict(status='frozen_before_saved_data_verification',source_sha256={str(Path(__file__).relative_to(ROOT)):source_sha},
            input_inventory_sha256=sha(INPUT/'artifact_hashes.json'),old_inventory_sha256=sha(OLD/'artifact_hashes.json')))
        result=inspect_saved()
        verify_package(INPUT);verify_package(OLD);assert sha(Path(__file__))==source_sha
        write(OUTPUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='parents'},allow_nan=False),flush=True)
    except BaseException:
        write(OUTPUT/'failure.json',dict(status='failed',traceback=traceback.format_exc(),no_physical_or_DP_work=True))
        raise
    finally:
        write(OUTPUT/'artifact_hashes.json',{str(p.relative_to(OUTPUT)):sha(p) for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:run()
    else:parser.print_help()
