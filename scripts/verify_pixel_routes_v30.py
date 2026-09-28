#!/usr/bin/env python3
"""Read sealed V30 numbers/arrays only. No project imports or Q recomputation."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
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

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'audit_results/v30_pixel_routes_20260917'
OUT=ROOT/'audit_results/v30_pixel_routes_review_20260917'
REPORT=ROOT/'docs/research/V30_PIXEL_ROUTE_INDEPENDENT_REVIEW_20260917.md'
CAP=1024**2
COLORS={2:np.array([40,100,220],np.uint8),3:np.array([220,60,40],np.uint8)}
FRAME=('timestamp_s','depth_m','color_rgb','intrinsic','world_from_camera','semantic')
SCAN=('timestamp_s','ranges_m','angle_min_rad','angle_increment_rad','range_max_m','world_from_laser')
PACKET=('scene_id','episode_id','frame_id','action_id','frame','scan','position','heading',
        'sensor_source','pose_source','action','collision','done')


def require(value,reason):
    if not value:raise ValueError(reason)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def arrhash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()
def close(a,b,why):require(abs(float(a)-float(b))<=1e-11,why)


def inventory(folder):
    listed=read(folder/'artifact_hashes.json')
    # The producer deliberately excludes every nested artifact_hashes.json.
    # Verify root and all four nested inventories independently.
    actual={str(p.relative_to(folder)) for p in folder.rglob('*')
            if p.is_file() and p.name!='artifact_hashes.json'}
    require(set(listed)==actual,'complete artifact file set: '+str(folder))
    for name,expected in listed.items():require(sha(folder/name)==expected,'artifact SHA: '+name)
    return dict(files=len(listed),inventory_sha256=sha(folder/'artifact_hashes.json'))


def packet(path):
    with np.load(path,allow_pickle=False) as file:data={key:file[key].copy() for key in file.files}
    meta=json.loads(str(data.pop('metadata').item()))
    require(set(data)=={group+'__'+key for group,names in (('frame',FRAME),('scan',SCAN)) for key in names},
            'complete saved sensor array schema')
    h=hashlib.sha256()
    def append(blob):h.update(len(blob).to_bytes(8,'big'));h.update(blob)
    for key in PACKET:
        append(key.encode())
        if key in ('frame','scan'):
            for field in FRAME if key=='frame' else SCAN:
                a=np.ascontiguousarray(data[key+'__'+field])
                require(not a.dtype.hasobject,'object array forbidden')
                append(field.encode());append(a.dtype.str.encode());append(str(a.shape).encode());append(a.tobytes())
        else:append(digest(meta[key]).encode())
    return meta,data,h.hexdigest()


def components(mask):
    visited=np.zeros(mask.shape,bool);rows=[];height,width=mask.shape
    for y,x in zip(*np.nonzero(mask)):
        if visited[y,x]:continue
        queue=deque([(int(y),int(x))]);visited[y,x]=True;points=[]
        while queue:
            r,c=queue.popleft();points.append((r,c))
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    rr,cc=r+dr,c+dc
                    if 0<=rr<height and 0<=cc<width and mask[rr,cc] and not visited[rr,cc]:
                        visited[rr,cc]=True;queue.append((rr,cc))
        if len(points)>=16:rows.append(np.asarray(sorted(points),int))
    return rows


class MarkerAudit:
    def __init__(self):self.tracks=[];self.seeds={}
    def consume(self,meta,data,receipt):
        depth=data['frame__depth_m'];rgb=data['frame__color_rgb'];k=data['frame__intrinsic'];pose=data['frame__world_from_camera']
        mask=np.zeros(depth.shape,bool)
        for color in COLORS.values():mask|=(rgb==color).all(axis=2)
        hits=[]
        for pixels in components(mask&(depth>.15)&np.isfinite(depth)):
            rr,cc=pixels.T;z=depth[rr,cc]
            local=np.column_stack(((cc-k[0,2])*z/k[0,0],(rr-k[1,2])*z/k[1,1],z))
            points=local@pose[:3,:3].T+pose[:3,3];center=np.median(points,axis=0)
            distances=[np.linalg.norm(center-t['center']) for t in self.tracks]
            if distances and min(distances)<=.75:slot=int(np.argmin(distances))
            else:
                slot=len(self.tracks)
                self.tracks.append(dict(center=center.copy(),frames=0,total=0,maximum=0,
                    first=meta['action_id'],last=meta['action_id'],poses=set()))
            track=self.tracks[slot];track['frames']+=1
            track['center']+=(center-track['center'])/track['frames']
            track['total']+=len(rr);track['maximum']=max(track['maximum'],len(rr));track['last']=meta['action_id']
            if meta['action_id']>0:track['poses'].add((*meta['position'],meta['heading']))
            nearest=int(np.argmin(np.linalg.norm(points-center,axis=1)))
            code=next(code for code,color in COLORS.items() if np.array_equal(rgb[rr[nearest],cc[nearest]],color))
            if slot not in self.seeds:
                self.seeds[slot]=dict(action_id=meta['action_id'],pixel=[int(rr[nearest]),int(cc[nearest])],
                    observed_seed_xyz=points[nearest].tolist(),actual_marker_code=code,component_pixels=len(rr))
            hits.append(dict(slot=slot,pixels=len(rr),center=center))
        rows=receipt['marker_components'];require(len(rows)==len(hits),'actual binary marker component count')
        for hit,row in zip(hits,rows):
            require(hit['slot']==row['observed_track_index'] and hit['pixels']==row['valid_rgb_marker_depth_pixels'],
                    'actual marker track order/pixel count')
            require(np.allclose(hit['center'],row['observed_world_centroid'],rtol=0,atol=1e-12),'marker centroid from saved depth')
        return mask
    def check_metadata(self,metadata):
        saved={row['observed_track_index']:row for row in metadata['observed_track_summary']}
        require(len(saved)==len(self.tracks),'track total')
        for index,track in enumerate(self.tracks):
            row=saved[index]
            for key,value in dict(frames=track['frames'],first_action=track['first'],last_action=track['last'],
                total_valid_pixels=track['total'],max_valid_pixels=track['maximum'],distinct_paid_poses=len(track['poses'])).items():
                require(row[key]==value,'observed track summary '+key)
            require(row['paid_poses']==[list(p) for p in sorted(track['poses'])],'distinct actual paid marker poses')
            require(np.allclose(row['centre'],track['center'],atol=1e-12,rtol=0),'final track center')
        supported=len(saved)==2 and all(t['maximum']>=16 and len(t['poses'])>=2 for t in self.tracks)
        require(supported==metadata['observed_marker_support_gate_passed'],'marker support gate')
        for slot,row in enumerate(metadata['instances']):
            seed=row['seed'];expected=self.seeds.get(slot)
            require((seed is None)==(expected is None),'missing actual seed')
            if seed is None:continue
            for key in ('action_id','pixel','actual_marker_code','component_pixels'):
                require(seed[key]==expected[key],'actual first seed '+key)
            require(np.allclose(seed['observed_seed_xyz'],expected['observed_seed_xyz'],atol=1e-12,rtol=0),'actual seed depth coordinates')
        return supported


def score_arithmetic(score,coverage,paid,returned,collisions):
    require(score['mission_asset_count']==2 and len(score['instances'])==2,'fixed two-asset macro')
    close(score['coverage_2d'],coverage,'actual C from saved belief/reachable')
    require(score['returned']==returned and score['collisions']==collisions,'score actual return/collision fields')
    require(score['primitive_budget_compliant']==(paid<=48),'actual paid budget qualification')
    eligible=bool(coverage>=.8 and returned and collisions==0 and not score['failed'] and paid<=48)
    require(score['eligible']==eligible,'C80/safety/return eligibility')
    require(score['missing_asset_count']==sum(row['missing'] for row in score['instances']),'missing count')
    for tag in ('02cm','05cm','10cm'):
        qualities=[];f1s=[]
        for row in score['instances']:
            projections=list(row['projections'].values());require(len(projections)==3,'three fixed projections')
            for projection in projections:
                p,r,f=map(float,(projection[tag][name] for name in ('precision','recall','f1')))
                require(all(0<=v<=1+1e-12 for v in (p,r,f,projection['iou'])),'projection score bounds')
                close(f,2*p*r/(p+r) if p+r else 0.,'boundary F1 arithmetic only')
                if projection['missing_projection']:
                    require(p==r==f==projection['iou']==0.,'missing projection zero')
            quality=sum(min(p[tag]['f1'],p['iou']) for p in projections)/3
            f1=sum(p[tag]['f1'] for p in projections)/3
            close(row[tag]['outline_quality'],quality,'three-projection quality arithmetic')
            close(row[tag]['outline_f1'],f1,'three-projection F1 arithmetic')
            if row['missing']:require(quality==0.,'missing facility remains zero')
            qualities.append(quality);f1s.append(f1)
        close(score[tag]['outline_macro_quality'],sum(qualities)/2,'fixed macro quality')
        close(score[tag]['outline_macro_f1'],sum(f1s)/2,'fixed macro F1')
        close(score[tag]['joint_outline'],coverage*sum(qualities)/2,'J=C*Q arithmetic')


def check_mesh(folder,stage,metadata,companion):
    hashes=[]
    for name,expected in [('raw',metadata['raw_mesh_sha256']),
        *[(f'slot{i}',row['observed_mesh_sha256']) for i,row in enumerate(metadata['instances'])]]:
        with np.load(folder/f'{stage}_{name}.npz',allow_pickle=False) as file:
            arrays={key:file[key] for key in ('vertices','triangles','vertex_colors')}
            for key,value in expected.items():require(arrhash(arrays[key])==value,'saved mesh array SHA')
            if name!='raw':
                receipt=hashlib.sha256(np.asarray(arrays['vertices'],float).tobytes()+arrays['triangles'].tobytes()).hexdigest()
                slot=int(name[4:]);saved=companion['submitted_instance_support'][slot]['submitted_mesh']
                require(receipt==saved['sha256'] and len(arrays['vertices'])==saved['vertices']
                    and len(arrays['triangles'])==saved['triangles'],'full companion used complete saved mesh')
            hashes.append(dict(name=name,array_sha256={key:arrhash(a) for key,a in arrays.items()}))
    require(companion['prediction_crop'] is False and companion['prediction_alignment'] is False,'no companion crop/alignment')
    return hashes


def verify_case(index,config):
    folder=SOURCE/f'case{index:02d}';inv=inventory(folder);result=read(folder/'result.json')
    main=read(folder/'started.json');replay=read(folder/'replay.json');began=read(folder/'replay_started.json')
    require(replay['passed'] and replay['all49_packets_byte_equal'] and replay['full_result_equal'],'completed exact replay')
    require(replay['result_sha256']==sha(folder/'result.json'),'replay result hash')
    require(main['pid']==replay['main_pid'] and began['pid']==replay['replay_pid'] and main['pid']!=began['pid'],'fresh replay PID')
    require(result['case']==config['cases'][index] and main['cumulative_main_used']==13+index,'fixed case/quota')
    counts=dict(worlds=1,sensor_packets=49,paid_actions=48,mapper_updates=49,
        mapper_mesh_extractions=2,measurement_snapshots=4,backend_observe_calls=136,evaluation_stages=2)
    require(result['counts']==counts==replay['counts'],'actual main/replay call receipts')
    require(not any('failure' in p.name or 'mismatch' in p.name for p in folder.iterdir()),'failure retained means review cannot claim all passed')
    trace=read(folder/'trace.json');require(len(trace)==49 and digest(trace)==result['trace_sha256'],'all saved trace rows/hash')
    require(digest([{k:r[k] for k in ('paid','action','position','heading')} for r in trace])==result['trajectory_sha256'],'trajectory hash')
    require(set(p.name for p in (folder/'packets').iterdir())=={f'{i:03d}.npz' for i in range(49)},'49 raw packets exact set')
    action_sequence=[None]+config['prefix']+config['suffixes'][result['case']['arm']]
    marker=MarkerAudit();states=[];computed=[];stage_metadata={};stage_rules={};collisions=0;first_pose=None
    for paid,row in enumerate(trace):
        meta,data,packet_sha=packet(folder/'packets'/f'{paid:03d}.npz')
        require(meta['action_id']==paid==row['paid'] and meta['action']==row['action']==action_sequence[paid], 'paid chronology/action')
        require(meta['position']==row['position'] and meta['heading']==row['heading'],'pose metadata')
        require(packet_sha==row['packet_sha256']==row['marker_receipt']['packet_sha256'],'all raw packet bytes/provenance hash')
        require(meta['episode_id']==f'v30-case{index:02d}' and meta['frame_id']==f'v30-case{index:02d}:{paid:03d}', 'frame/episode identity')
        require(meta['done']==(paid==48) and meta['collision']==row['collision'],'terminal/collision flags')
        close(float(data['frame__timestamp_s']),paid,'actual frame time');close(float(data['scan__timestamp_s']),paid,'scan time')
        pose=(*meta['position'],meta['heading']);states.append(pose);first_pose=first_pose or pose
        if paid:
            r,c,h=states[-2]
            if meta['action']=='forward' and not meta['collision']:
                dr,dc=((-5,0),(0,5),(5,0),(0,-5))[h];r+=dr;c+=dc
            elif meta['action']!='forward':h=(h+(1 if meta['action']=='right' else -1))%4
            require(pose==(r,c,h),'actual 1m forward or paid heading transition')
        collisions+=int(meta['collision'])
        require(not np.any(data['frame__semantic']),'no class field fed to mapper')
        depth=data['frame__depth_m'];rgb=data['frame__color_rgb']
        require(depth.shape==(72,96) and rgb.shape==(72,96,3),'actual sensor dimensions')
        actual_marker=marker.consume(meta,data,row['marker_receipt'])
        generic=np.any(rgb!=rgb[...,0,None],axis=-1)
        require(np.array_equal(actual_marker,generic),'binary marker audit uses only known physical colors')
        stripped=rgb.copy();stripped[actual_marker]=[153,153,153]
        nonsem=dict(depth=arrhash(depth),scan=arrhash(data['scan__ranges_m']),rgb_without_class=arrhash(stripped),
            binary_marker=arrhash(actual_marker),pose=arrhash(data['frame__world_from_camera']))
        require(nonsem==row['nonsemantic'],'raw recomputed class-stripped evidence')
        computed.append(nonsem)
        if paid in (18,48):
            stage='prefix' if paid==18 else 'final';metadata=read(folder/f'{stage}_metadata.json')
            require(metadata['observed_frames']==paid+1 and metadata['last_action_id']==paid,'snapshot full history')
            require(digest(metadata)==result['stages'][stage]['measurement_metadata_sha256'],'frozen snapshot metadata digest')
            supported=marker.check_metadata(metadata);stage_metadata[stage]=metadata
            cues={}
            for seed in (row['seed'] for row in metadata['instances']):
                if seed is not None:cues.setdefault(seed['actual_marker_code'],[]).append('A' if seed['observed_seed_xyz'][0]<5.5 else 'B')
            valid=supported and set(cues)=={2,3} and all(len(v)==1 for v in cues.values()) and cues[2]!=cues[3]
            stage_rules[stage]=dict(complex=cues[3][0],swapped=cues[2][0]) if valid else None
    require(states[18]==states[48]==first_pose,'prefix/final actual exact return')
    require(result['steps']==48 and result['collisions']==collisions and result['returned'],'actual terminal cost/safety')
    stages={}
    with np.load(folder/'maps.npz',allow_pickle=False) as maps:
        require(set(maps.files)=={'prefix','final','reachable'},'saved evaluation maps')
        reachable=maps['reachable'];require(reachable.dtype==bool and reachable.any(),'fixed reachable mask')
        for stage,paid in (('prefix',18),('final',48)):
            entry=result['stages'][stage];belief=maps[stage];coverage=float(np.mean(belief[reachable]!=-1))
            require(arrhash(belief)==entry['belief_sha256'] and int(reachable.sum())==entry['safe_floor_cells'],'saved map/hash/C denominator')
            for score in (entry['window']['main_observed'],entry['window']['raw_secondary'],entry['full_instance']):
                score_arithmetic(score,coverage,paid,states[paid]==first_pose,collisions)
            mesh_receipts=check_mesh(folder,stage,stage_metadata[stage],entry['full_instance'])
            main_score=entry['window']['main_observed'];full=entry['full_instance']
            stages[stage]=dict(C=coverage,Q=main_score['05cm']['outline_macro_quality'],J=main_score['05cm']['joint_outline'],
                Q_per_instance=[r['05cm']['outline_quality'] for r in main_score['instances']],
                raw_Q=entry['window']['raw_secondary']['05cm']['outline_macro_quality'],
                full_instance_Q=full['05cm']['outline_macro_quality'],
                full_instance_J=full['05cm']['joint_outline'],eligible=main_score['eligible'],
                marker_gate=entry['window']['observed_marker_support_gate_passed'],
                instance_gate=entry['window']['evaluation_association_and_minimum_separation_passed'],
                full_seed_gate=full['unique_complete_seed_association'],
                outside_vertices=[r['outside_window']['outside_vertices'] for r in full['instances']],
                full_xy_saved_diagnostics=[dict(id=r['id'],boundary_f1=r['projections']['xy']['05cm']['f1'],
                    iou=r['projections']['xy']['iou'],predicted_area_m2=r['projections']['xy']['predicted_area_m2'],
                    reference_area_m2=r['projections']['xy']['reference_area_m2']) for r in full['instances']],
                mesh_receipts=mesh_receipts)
    selected_index=0 if result['case']['arm']=='A' else 1
    gain=[b-a for a,b in zip(stages['prefix']['Q_per_instance'],stages['final']['Q_per_instance'])]
    summary=dict(case=result['case'],inventory=inv,main_pid=main['pid'],replay_pid=began['pid'],counts=counts,
        stages=stages,observed_prefix_rule=stage_rules['prefix'],actual_packets_checked=49,
        supplemental_paid_actions=30,window_Q_gain=stages['final']['Q']-stages['prefix']['Q'],
        full_instance_Q_gain=stages['final']['full_instance_Q']-stages['prefix']['full_instance_Q'],
        selected_asset_Q_gain=gain[selected_index],other_asset_Q_gain=gain[1-selected_index],
        sensor_or_TSDF_regenerated_by_review=False)
    return summary,computed,trace


def analyze_fixed(cases,rules):
    available=all(rule is not None for rule in rules)
    selected={key:[2*h+int(rules[h][key]=='B') for h in range(2)] for key in ('complex','swapped')} if available else {}
    output={}
    for metric in ('Q','J'):
        y=[c['stages']['final'][metric] for c in cases];mean=lambda ids:sum(y[i] for i in ids)/len(ids)
        a,b=mean([0,2]),mean([1,3]);best=max(a,b);oracle=(max(y[:2])+max(y[2:]))/2
        output[metric]=dict(always_A=a,always_B=b,best_fixed=best,uniform_random=mean(range(4)),
            complex_rule=mean(selected['complex']) if available else None,
            swapped_rule=mean(selected['swapped']) if available else None,two_option_oracle=oracle,
            oracle_relative_to_best_fixed=oracle/best-1 if best else None,
            complex_relative_to_best_fixed=mean(selected['complex'])/best-1 if best and available else None,
            complex_minus_simple_per_assignment=[y[a]-y[b] for a,b in zip(selected['complex'],selected['swapped'])] if available else None,
            difference_of_differences=(y[0]-y[1])-(y[2]-y[3]))
    return selected,output


def verify():
    manifest=read(SOURCE/'manifest.json');require(manifest['status']=='complete','must await four complete tasks and four replays plus analyze')
    root_inventory=inventory(SOURCE);analysis=read(SOURCE/'result.json')
    require(manifest['new_main_started']==4 and manifest['replay_started']==4 and manifest['historical_main_used']==12,'actual quota receipts')
    sources=manifest['source_sha256'];require(len(sources)==54,'declared 54 frozen dependencies')
    require(sha(SOURCE/'sources.zip')==manifest['source_zip_sha256'],'frozen ZIP bytes')
    with zipfile.ZipFile(SOURCE/'sources.zip') as archive:
        require(len(archive.namelist())==len(sources) and set(archive.namelist())==set(sources),'complete source ZIP entries')
        for name,expected in sources.items():
            require(sha(ROOT/name)==expected and hashlib.sha256(archive.read(name)).hexdigest()==expected,'frozen source/ZIP entry: '+name)
    config=read(ROOT/'configs/virtual3d/v30_pixel_routes_20260917.json');require(config['budget']==48 and len(config['prefix'])==18,'frozen full budget')
    cases=[];computed=[];traces=[]
    for index in range(4):
        c,p,t=verify_case(index,config);cases.append(c);computed.append(p);traces.append(t)
    require(len(analysis['table'])==4,'all four aggregate table cases retained')
    for case,table in zip(cases,analysis['table']):
        for key,value in case['case'].items():require(table[key]==value,'aggregate case identity')
        original=read(SOURCE/f"case{case['case']['index']:02d}"/'result.json')
        for stage in ('prefix','final'):
            for key in ('C','Q','J','eligible','Q_per_instance','marker_gate','instance_gate'):
                require(table[stage][key]==case['stages'][stage][key],'aggregate table field '+stage+'.'+key)
            require(table[stage]['full_instance']==original['stages'][stage]['full_instance'],
                'aggregate retains full unmodified companion results')
    pids=[c[key] for c in cases for key in ('main_pid','replay_pid')]
    require(len(set(pids))==8,'eight actual independent process identities')
    pairings=[]
    for left,right in ((0,2),(1,3)):
        first={key:next((i for i in range(49) if computed[left][i][key]!=computed[right][i][key]),None) for key in computed[left][0]}
        clean=next((i for i in range(49) if traces[left][i]['clean_depth_sha256']!=traces[right][i]['clean_depth_sha256']),None)
        saved=dict(cases=[left,right],first_difference_action=first,clean_depth_first_difference_action=clean,
            prefix_nonsemantic_equal=all(i is None or i>18 for i in first.values()))
        require(saved==analysis['pairings'][len(pairings)],'aggregate prefix pairing/first difference')
        row=dict(**saved,clean_depth_scope='saved producer hash only; clean depth not regenerated')
        if first['depth'] is not None:
            action=first['depth'];_,a,_=packet(SOURCE/f'case{left:02d}'/'packets'/f'{action:03d}.npz');_,b,_=packet(SOURCE/f'case{right:02d}'/'packets'/f'{action:03d}.npz')
            row['first_actual_depth_different_pixels']=int((a['frame__depth_m']!=b['frame__depth_m']).sum())
        pairings.append(row)
    rules=[cases[a]['observed_prefix_rule'] if cases[a]['observed_prefix_rule']==cases[b]['observed_prefix_rule'] else None
           for a,b in ((0,1),(2,3))]
    selected,selection=analyze_fixed(cases,rules)
    require(rules==analysis['observed_class_rules'] and selected==analysis['observed_rule_case_indices'],'actual RGB observed rules')
    for metric,values in selection.items():
        for name,value in values.items():
            old=analysis['fixed_route_selection'][metric][name]
            if value is None:require(old is None,'unavailable rule preserved')
            elif isinstance(value,list):
                for a,b in zip(value,old):close(a,b,'policy table vector arithmetic')
            else:close(value,old,'policy table arithmetic '+metric+'.'+name)
    eligible=all(c['stages']['final']['eligible'] for c in cases)
    gate=bool(eligible and all(rules) and all(p['prefix_nonsemantic_equal'] for p in pairings)
        and all(c['stages']['prefix']['marker_gate'] and c['stages']['final']['instance_gate'] for c in cases)
        and selection['J']['oracle_relative_to_best_fixed'] is not None
        and selection['J']['oracle_relative_to_best_fixed']>.05
        and min(selection['J']['complex_minus_simple_per_assignment'])>0)
    require(analysis['all_replays_passed'] and eligible==analysis['all_main_eligible'] and gate==analysis['progress_to_learning_gate_passed'],'terminal qualification/learning gate')
    totals={key:sum(c['counts'][key] for c in cases)*2 for key in cases[0]['counts']}
    require(totals==analysis['counts_with_replays'],'exact successful main plus replay counts')
    require(analysis['new_main_tasks']==4 and analysis['total_main_used']==16 and analysis['main_limit']==36,'main quota remains16/36')
    return dict(status='independent_saved_evidence_verified',source_root=str(SOURCE),root_inventory=root_inventory,
        source_count=len(sources),source_archive_sha256=manifest['source_zip_sha256'],source_sha256=sources,
        input_result_sha256=sha(SOURCE/'result.json'),cases=cases,pairings=pairings,observed_class_rules=rules,
        fixed_route_selection=selection,all_main_eligible=eligible,learning_gate_passed=gate,
        actual_saved_packets_verified=196,all_fresh_replays_receipt_verified=True,process_ids=pids,
        main_plus_replay_counts=totals,new_review_worlds=0,new_review_sensors=0,new_review_mapper_updates=0,
        new_review_Q_evaluations=0,metric_scope='saved metric arithmetic and mesh identity; no independent geometric Q recomputation',
        clean_depth_scope='saved trace hash only',historical_main_used=12,total_main_used=16,
        additional_view_quality_and_semantic_selection_must_be_interpreted_separately=True,
        ideal_vertical_projection_Q_two_thirds_does_not_prove_semantics_useless=True)


def protected(name,value):
    payload=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    used=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    require(used+len(payload)+16384<=CAP and shutil.disk_usage(OUT).free-len(payload)>=64*1024**2,'audit storage bound')
    target=OUT/name;require(not target.exists(),'no reviewer artifact overwrite');target.write_bytes(payload)


def report_text(result):
    lines=['# V30 像素路线终态独立复核','',
        '只读取封存证据、保存 packet/mesh/map 数组与数值；未生成传感、重建 TSDF 或重新评价几何 Q。',
        '54 份源及 ZIP、根/四案例完整产物集合、196 个原始包、4 主进程 + 4 独立回放进程、实际 marker 和政策算术已核验。',
        '回放真实性依据不同 PID 与完整哈希收据；本复核不再执行回放。clean depth 仅核保存 hash，不能据此声称重新测量了理想深度。','',
        '| case | 选臂 | C终点 | 窗口Q前缀→终点 | 完整实例Q前缀→终点 | 所选设施窗口Q增量 | 资格 |',
        '| --- | --- | ---: | --- | --- | ---: | --- |']
    for c in result['cases']:
        a,b=c['stages']['prefix'],c['stages']['final']
        lines.append(f"| {c['case']['index']} | {c['case']['arm']} | {b['C']:.6f} | {a['Q']:.6f} → {b['Q']:.6f} | {a['full_instance_Q']:.6f} → {b['full_instance_Q']:.6f} | {c['selected_asset_Q_gain']:+.6f} | {b['eligible']} |")
    for pair in result['pairings']:
        lines.append(f"两排列配对 {pair['cases']}：实际深度首差动作 {pair['first_difference_action']['depth']}，不同像素 {pair['first_actual_depth_different_pixels']}；共同前缀19包完全一致。")
    j=result['fixed_route_selection']['J']
    lines+=['',f"最佳固定臂 mean J={j['best_fixed']:.9f}；两选项事后 oracle={j['two_option_oracle']:.9f}；观测复杂类规则={j['complex_rule']}。",
        f"预声明学习推进门：{result['learning_gate_passed']}。补看的实测质量增量与类别选臂相对最佳固定臂的净收益必须分别判断；前者不能替代后者。",'',
        '四案 C 均未过原80%门槛，表中 J 与类别规则差值只作不合格任务的诊断值；不解释为成功任务收益。',
        '完整实例 XY 的 F1/IoU差异来自保存数值；它提示轮廓与面积闭合的表示差别，不足以定位唯一TSDF根因，也不证明去掉IoU合理。',
        '原 V23 窗口 Q 为固定 headline，完整实例未裁剪 Q 为 companion；已核 companion 使用完整保存 mesh 的 hash，未独立重算投影。',
        '人工 marker、单场景单种子与固定脚本路线限制保持。最佳固定臂不是可主动取信息并改选的强几何策略；结果不是当前 ANS 自主比较。',
        '完整精确竖面在旧三投影定义下 Q=2/3 的解析限制需要披露，但它本身不能推出语义无用。',
        '主任务累计16/36；本次独立审计新增主任务0。完整逐项数值、hash、缺失与归属诊断见同目录 result.json。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:parser.print_help();return
    require(read(SOURCE/'manifest.json')['status']=='complete','await explicit terminal completion; no intermediate analysis')
    require(not OUT.exists() and not REPORT.exists(),'no overwrite')
    require(shutil.disk_usage(ROOT).free>=CAP+64*1024**2,'audit free reserve')
    OUT.mkdir()
    try:
        protected('review_script.py',Path(__file__).read_bytes())
        result=verify();result['review_script_sha256']=sha(__file__)
        protected('result.json',result);text=report_text(result)
        protected('REVIEW.md',text.encode());REPORT.write_text(text)
        print(json.dumps(dict(status=result['status'],all_eligible=result['all_main_eligible'],learning_gate=result['learning_gate_passed'],output=str(OUT))))
    except BaseException as error:
        protected('failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),new_main_tasks=0))
        raise
    finally:
        protected('artifact_hashes.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':main()
