#!/usr/bin/env python3
"""Independent saved-record audit; no controller/World/mapper/evaluator run."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.replay_final_local_saved_case_20260929 import BATCH,read,sha,plain,payload,require,verify_batch,write
OUTPUT=ROOT/'audit_results/final_delivery_20260929/validation_review.json'


def array_hash(value,receipt=False):
    import numpy as np
    a=np.ascontiguousarray(value)
    if not receipt:return hashlib.sha256(f'{a.dtype.str}:{a.shape}:'.encode()+a.tobytes()).hexdigest()
    h=hashlib.sha256()
    for part in (a.dtype.str.encode(),str(a.shape).encode(),a.tobytes()):
        h.update(len(part).to_bytes(8,'big'));h.update(part)
    return h.hexdigest()


def close(a,b,label):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=1e-12,label)


def residual_score(depth,scan,td,ts):
    """Independent arithmetic of the fixed V35 public-template pseudo-likelihood."""
    import numpy as np
    depth=np.asarray(depth,dtype=np.float64);scan=np.asarray(scan,dtype=np.float64)
    scores=[];depth_losses=None;scan_losses=None
    use=(np.abs(td[0]-td[1])>1e-5)&(depth>0);nd=int(use.sum())
    if nd>=8:
        observed=(480.*.12)/depth[use];depth_losses=[]
        for h in (0,1):
            expected=td[h][use];valid=expected>0;loss=np.full(nd,12.5,np.float64)
            r=(observed[valid]-(480.*.12)/expected[valid])/.25
            loss[valid]=np.minimum(.5*r*r,12.5);depth_losses.append(float(loss.mean()))
        scores.append(depth_losses[1]-depth_losses[0])
    use=(np.abs(ts[0]-ts[1])>1e-5)&(scan>0);ns=int(use.sum())
    if ns>=3:
        scan_losses=[]
        for h in (0,1):
            r=(scan[use]-ts[h][use])/.02;scan_losses.append(float(np.minimum(.5*r*r,12.5).mean()))
        scores.append(scan_losses[1]-scan_losses[0])
    score=float(np.clip(np.mean(scores),-6.,6.)) if scores else 0.
    return score,dict(depth_samples=nd,scan_samples=ns,depth_mean_losses=depth_losses,
        scan_mean_losses=scan_losses,log_likelihood_ratio=score,usable_modalities=len(scores))


def semantic_prior(rgb,pose,mode,positions):
    import numpy as np
    if mode=='G':return 0.,None,None
    counts={code:int(np.all(rgb==color,axis=-1).sum()) for code,color in [(2,(40,100,220)),(3,(220,60,40))]}
    nonzero=[c for c,n in counts.items() if n>0];supported=[c for c,n in counts.items() if n>=16]
    chosen=supported[0] if len(supported)==len(nonzero)==1 else None
    if len(supported)==2:
        for p in positions.values():p.add(pose[:2])
    elif chosen is not None:positions[chosen].add(pose[:2])
    present=[c for c,p in positions.items() if p];qualified=[c for c in present if len(positions[c])>=2]
    sem=(1. if qualified[0]==2 else -1.)*math.log(.9/(1.-.9)) if len(present)==len(qualified)==1 else 0.
    return sem,chosen,{str(k):v for k,v in counts.items()}


def back_distances(edges,anchor):
    reverse={n:[] for n in range(len(edges))}
    for n,row in enumerate(edges):
        for _,target in row:reverse[target].append(n)
    d={anchor:0};q=deque([anchor])
    while q:
        n=q.popleft()
        for prev in reverse[n]:
            if prev not in d:d[prev]=d[n]+1;q.append(prev)
    return d


def review_case(case,inputs):
    import numpy as np
    from nso.decision_replay_v13 import load_packet
    from nso.sensor_contract_v34 import validate_sensor_packet_v34
    from nso.cpu_sensor_contract_v10 import GridTransform
    from env.information_pixel_v34 import InformationConfigV34
    folder=BATCH/'cases'/case['id'];seal=read(folder/'seal.json')
    actual_files={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name!='seal.json'}
    require(actual_files==set(seal),'case inventory differs')
    for rel,pin in seal.items():require(sha(folder/rel)==pin,'case artifact changed: '+rel)
    inputs[str((folder/'seal.json').relative_to(ROOT))]=sha(folder/'seal.json')
    r=read(folder/'result.json');require(r['case']==case and r['status']=='completed','case result mismatch')
    start=read(folder/'started.json')
    require(start['source_freeze_sha256']==sha(BATCH/'source_freeze.json') and start['preparation_sha256']==sha(BATCH/'preparation_seal.json'),'case source/preparation link differs')
    freeze=read(folder/'prediction_freeze.json');require(freeze['before_first_reference_access'] is True,'prediction not declared sealed before reference')
    for rel,pin in freeze['artifacts'].items():require(sha(folder/rel)==pin,'pre-reference prediction changed')
    template=BATCH/'public_templates'/case['scene'];geometry=read(template/'geometry.json');info=read(template/'metadata.json')
    tables=geometry['tables'];a,b=tables
    for key in ('poses','edges','anchor','prefix_nodes'):require(a[key]==b[key],'public hypothesis graph differs')
    poses=list(map(tuple,a['poses']));node_for={p:i for i,p in enumerate(poses)};return_cost=back_distances(a['edges'],a['anchor'])
    require(len(return_cost)==len(poses)==92,'public pose reachability differs')
    with np.load(template/'sensor_templates.npz',allow_pickle=False) as data:td=data['depth'].copy();ts=data['ranges'].copy()
    require(np.array_equal(td[0,a['prefix_nodes']],td[1,a['prefix_nodes']]) and np.array_equal(ts[0,a['prefix_nodes']],ts[1,a['prefix_nodes']]),'clean nonsemantic prefix differs')
    config=InformationConfigV34(width_m=7.,height_m=5.,max_steps=42);transform=GridTransform((25,35),.2)
    trace=read(folder/'trace.json');controller=read(folder/'controller.json');history=controller['history']
    require(len(trace)==len(history)==r['paid_actions']+1<=43,'frame/action budget mismatch')
    require(len(list((folder/'packets').glob('*.npz')))==len(trace),'packet inventory differs')
    require(controller['truth_or_selected_hypothesis_passed_to_controller'] is False,'controller truth boundary differs')
    plans={p['step']:p for p in controller['plans']};require(len(plans)==len(controller['plans']),'duplicate planning step')
    seen=set();positions={2:set(),3:set()};logodds=0.;masks=[0,0];rgb_receipts=[];checks=0
    first_semantic=first_geometry=None
    for step,(t,h) in enumerate(zip(trace,history)):
        packet=load_packet(folder/'packets'/f'{step:03d}.npz');validate_sensor_packet_v34(packet,transform,config)
        require(packet.sha256()==t['packet_sha256'],'raw packet sha mismatch')
        require(packet.episode_id==case['id'] and packet.scene_id=='v34-P00','packet namespace/episode differs')
        require(packet.action_id==step and packet.action==t['action'],'paid action mismatch')
        require(not np.any(packet.frame.semantic),'semantic GT array not empty')
        pose=(*map(int,np.rint(packet.frame.world_from_camera[:2,3]-np.asarray(info['shift'][:2]))),packet.heading)
        require(list(pose)==t['pose']==h['state']['pose']==h['posterior']['pose'],'pose log mismatch')
        require(step==t['paid']==h['step']==h['posterior']['step']==h['state']['step'],'step mismatch')
        require(packet.collision==t['collision']==False,'unexpected collision')
        rgb=packet.frame.color_rgb.copy();raw_rgb_hash=array_hash(rgb)
        for color in ((40,100,220),(220,60,40)):rgb[np.all(rgb==color,axis=-1)]=(127,127,127)
        fields=dict(depth=packet.frame.depth_m,rgb_nonsemantic=rgb,intrinsic=packet.frame.intrinsic,
            camera_pose=packet.frame.world_from_camera,ranges=packet.scan.ranges_m,laser_pose=packet.scan.world_from_laser,
            scan_calibration=np.asarray([packet.scan.angle_min_rad,packet.scan.angle_increment_rad,packet.scan.range_max_m]),
            cell=np.asarray(packet.position,dtype=np.int64),heading=np.asarray(packet.heading,dtype=np.int64))
        require({k:array_hash(v) for k,v in fields.items()}==t['nonsemantic'],'nonsemantic fields not derived from saved packet')
        rgb_receipts.append(raw_rgb_hash)
        post=h['posterior'];node=node_for[pose]
        require(node==h['state']['node'] and h['state']['remaining_budget']==42-step,'public node/budget state differs')
        for k,value in [('depth',packet.frame.depth_m),('rgb',packet.frame.color_rgb),('ranges',packet.scan.ranges_m)]:
            require(array_hash(value,receipt=True)==post['observation_sha256'][k],'belief sensor binding differs')
        score,residual=residual_score(packet.frame.depth_m,packet.scan.ranges_m,td[:,node],ts[:,node])
        for key,value in residual.items():
            if value is None:require(post['residuals'][key] is None,'residual null differs')
            elif isinstance(value,list):
                for x,y in zip(value,post['residuals'][key]):close(x,y,'per-template residual differs')
            else:close(value,post['residuals'][key],'residual arithmetic differs')
        fresh=pose not in seen;seen.add(pose);applied=score if fresh else 0.;logodds=min(24.,max(-24.,logodds+applied))
        sem,chosen,counts=semantic_prior(packet.frame.color_rgb,pose,case['mode'],positions)
        require(post['new_geometry_pose']==fresh and post['geometry_unique_poses']==len(seen),'geometry novelty accounting differs')
        require(post['geometry_feedback_enabled'] is True and post['mode']==case['mode'],'G/S feedback/mode differs')
        close(applied,post['geometry_applied_log_odds'],'applied evidence differs');close(logodds,post['geometry_log_odds'],'geometry accumulation differs');close(sem,post['semantic_log_odds'],'RGB prior differs')
        require(post['observed_class_after_intervention']==chosen and post['rgb_pixel_counts']==counts,'RGB class receipt differs')
        require(post['class_distinct_xy']=={str(k):len(v) for k,v in positions.items()} and post['class_conflict']==all(positions.values()),'class support receipt differs')
        p0=1/(1+math.exp(-(sem+logodds)));close(p0,post['probabilities'][0],'posterior differs');close(1-p0,post['probabilities'][1],'posterior complement differs')
        require(h['state']['probabilities']==post['probabilities'],'planner did not consume saved posterior')
        if sem and first_semantic is None:first_semantic=step
        if applied and first_geometry is None:first_geometry=step
        for j in (0,1):masks[j]|=int(tables[j]['observed_masks_hex'][node],16)
        require(h['state']['masks_hex']==[hex(v) for v in masks],'public surface union differs')
        require(t['measured_map_sha256']==h['state']['measured_map_sha256'],'per-frame map/controller linkage differs')
        for module,operation in [('OV-SDF','observed_semantic_and_map_state'),('IGCR','paid_geometry_residual_feedback'),('STGHP','update_public_topology_state')]:
            require(sum(c['step']==step and c['module']==module and c['operation']==operation for c in controller['calls'])==1,'module update missing/duplicate')
        if step:
            prev=tuple(trace[step-1]['pose']);edge=dict(a['edges'][node_for[prev]])
            require(packet.action==history[step-1]['next_action'] and edge[packet.action]==node,'executed primitive differs from public graph/issued action')
        else:require(pose==poses[a['anchor']] and packet.action is None,'initial state differs')
        action=h['next_action'];require(action==t['next_action'],'next action trace differs')
        if action is not None:
            plan=plans[step];require(plan['action']==action and plan['from_node']==node and plan['probabilities']==post['probabilities'],'selected action/belief differs')
            require(1+return_cost[plan['node']]<=42-step,'return budget violated')
            local=[c for c in controller['calls'] if c['module']=='RPN-UQ' and c['step']==step]
            require(len(local)==1 and local[0]['allowed'] and local[0]['action']==action,'local validation differs')
            if step<18:require(action==info['prefix'][step],'forced prefix differs')
            else:
                pred=plan['planning'];require(pred['phase']=='online_belief_global_planning' and pred['geometry_feedback'] is True,'post-prefix planning/feedback differs')
                close(pred['probability0'],p0,'plan posterior differs')
                best=pred['expected_proxy'];close(max([pred['stop_value']]+[c['expected_proxy'] for c in pred['candidates']]),best,'candidate max differs')
                chosen_action=next(c['action'] for c in pred['candidates'] if c['expected_proxy']>=best-1e-12)
                require(chosen_action==action and pred['stop_value']<best-1e-12,'frozen tie-breaking/stop decision differs')
        checks+=1
    require(controller['actions']==[t['action'] for t in trace[1:]],'full executed action list differs')
    require(tuple(trace[-1]['pose'])==poses[a['anchor']] and trace[-1]['next_action'] is None,'terminal exact return differs')
    require(set(c['module'] for c in controller['calls'])=={'OV-SDF','STGHP','RPN-UQ','IGCR'},'four-module coverage differs')
    with np.load(folder/'evaluation_floor.npz',allow_pickle=False) as d:floor=d['reachable']
    for stage,m in r['measurements'].items():
        with np.load(folder/f'{stage}_map.npz',allow_pickle=False) as d:
            belief=d['belief'];coverage=float(np.mean(belief[floor]!=-1))
        index=18 if stage=='prefix' else len(trace)-1
        require(array_hash(belief)==trace[index]['measured_map_sha256'],'actual saved snapshot map hash differs')
        close(coverage,m['C_map'],'measured coverage differs')
        for threshold in ('02cm','05cm','10cm'):
            q=m[threshold];p,recall=q['precision'],q['recall'];f1=2*p*recall/(p+recall) if p+recall else 0.
            require(all(0<=x<=1 for x in (p,recall,q['f1'],q['joint'])),'invalid metric bound')
            close(f1,q['f1'],'F1 differs');close(coverage*f1,q['joint'],'J differs')
        close(m['main_joint'],m['05cm']['joint'],'main threshold differs')
        require(m['eligible']==bool(coverage>=.8 and m['returned'] and not m['collisions'] and not m['failed'] and m['paid_actions']<=42),'qualification differs')
        require(m['contract']=='v34-area-surface-1' and m['main_threshold_m']==.05,'frozen surface metric differs')
        crop=read(folder/f'{stage}_crop.json');require(crop['public_bounds']==info['public_bounds'] and crop['padding_m']==.2,'ROI contract differs')
        for kind in ('raw_mesh','mesh'):
            with np.load(folder/f'{stage}_{kind}.npz',allow_pickle=False) as d:
                vertices,triangles=d['vertices'],d['triangles']
                require(np.isfinite(vertices).all() and triangles.ndim==2 and triangles.shape[1]==3,'invalid saved mesh arrays')
                require(not len(triangles) or triangles.min()>=0 and triangles.max()<len(vertices),'invalid saved mesh index')
    return (dict(case=case,status='completed',qualified=bool(r['measurements']['final']['eligible']),
        checked_frames=checks,raw_geometry_likelihoods_recomputed=True,first_semantic_prior_step=first_semantic,
        first_geometry_feedback_step=first_geometry,measurements=r['measurements'],costs=r['execution_costs'],
        elapsed_seconds=r['elapsed_seconds'],timings=r['timings'],case_seal_sha256=sha(folder/'seal.json')),
        dict(trace=trace,controller=controller,rgb_hashes=rgb_receipts))


def compare_pair(g,s,gr,sr):
    left=g['controller']['history'];right=s['controller']['history']
    differing=next((i for i,(x,y) in enumerate(zip(left,right)) if x['next_action']!=y['next_action']),None)
    candidates_changed=[];rank_changed=[];witnesses=[]
    gp={x['step']:x for x in g['controller']['plans']};sp={x['step']:x for x in s['controller']['plans']}
    prefix_equal=True
    for i,(x,y) in enumerate(zip(g['trace'],s['trace'])):
        prefix_equal=prefix_equal and x['nonsemantic']==y['nonsemantic'] and g['rgb_hashes'][i]==s['rgb_hashes'][i]
        if i<18 or not prefix_equal or i not in gp or i not in sp:continue
        a,b=gp[i]['planning'],sp[i]['planning'];ca={c['action']:c for c in a['candidates']};cb={c['action']:c for c in b['candidates']}
        require(set(ca)==set(cb),'same observation state has different action candidate set')
        changed=any(abs(ca[k]['expected_proxy']-cb[k]['expected_proxy'])>1e-12 for k in ca)
        if changed:candidates_changed.append(i)
        if gp[i]['action']!=sp[i]['action']:rank_changed.append(i)
        if i==18 or gp[i]['action']!=sp[i]['action']:
            witnesses.append(dict(step=i,actual_paid_inputs_equal=True,G_action=gp[i]['action'],S_action=sp[i]['action'],
                G_posterior=left[i]['posterior']['probabilities'],S_posterior=right[i]['posterior']['probabilities'],
                G_geometry_log_odds=left[i]['posterior']['geometry_log_odds'],S_geometry_log_odds=right[i]['posterior']['geometry_log_odds'],
                G_semantic_log_odds=left[i]['posterior']['semantic_log_odds'],S_semantic_log_odds=right[i]['posterior']['semantic_log_odds'],
                G_candidates=a['candidates'],S_candidates=b['candidates']))
    gm,sm=gr['measurements']['final'],sr['measurements']['final']
    return dict(scene=gr['case']['scene'],hypothesis=gr['case']['hypothesis'],
        G_case=gr['case']['id'],S_case=sr['case']['id'],both_qualified=gr['qualified'] and sr['qualified'],
        actual_frames_G=len(left),actual_frames_S=len(right),all_paid_inputs_equal=prefix_equal,
        actions_equal=g['controller']['actions']==s['controller']['actions'],
        first_decision_divergence=differing,first_executed_action_divergence=None if differing is None else differing+1,
        same_prefix_candidate_scores_changed_steps=candidates_changed,same_prefix_selected_action_changed_steps=rank_changed,
        witnesses=witnesses,G_J5=gm['main_joint'],S_J5=sm['main_joint'],delta_J5=sm['main_joint']-gm['main_joint'],
        delta_C_map=sm['C_map']-gm['C_map'],delta_F1_5cm=sm['05cm']['f1']-gm['05cm']['f1'])


def main():
    started=time.monotonic();require(not OUTPUT.exists(),'independent review output already exists')
    from env.information_pixel_v34 import sensor_counts_v34
    before=dict(sensor_counts_v34());protocol,freeze=verify_batch()
    inputs={str((BATCH/name).relative_to(ROOT)):sha(BATCH/name) for name in ('source_freeze.json','source_archive.zip','protocol.json','preparation_seal.json')}
    prep=read(BATCH/'preparation_seal.json')
    for layout,files in prep.items():
        for name,pin in files.items():
            path=BATCH/'public_templates'/layout/name;require(sha(path)==pin,'template seal mismatch');inputs[str(path.relative_to(ROOT))]=pin
    cases=[];saved={}
    for case in protocol['cases']:
        report,data=review_case(case,inputs);cases.append(report);saved[case['id']]=data
    pairs=[];prefix_checks=[]
    for item in protocol['layouts']:
        rows=[r for r in cases if r['case']['scene']==item['id']]
        values=[payload([dict(pose=t['pose'],nonsemantic=t['nonsemantic']) for t in saved[r['case']['id']]['trace'][:19]]) for r in rows]
        require(len(rows)==4 and len(set(values))==1,'four-arm actual nonsemantic prefix mismatch')
        prefix_checks.append(dict(scene=item['id'],arms=4,actual_prefix_frames=19,nonsemantic_prefix_exact_equal=True))
        for h in (0,1):
            arms={r['case']['mode']:r for r in rows if r['case']['hypothesis']==h};g,s=arms['G'],arms['S']
            pairs.append(compare_pair(saved[g['case']['id']],saved[s['case']['id']],g,s))
    average_g=sum(p['G_J5'] for p in pairs)/4;average_s=sum(p['S_J5'] for p in pairs)/4
    after=dict(sensor_counts_v34());require(before==after,'review caused sensor calls')
    report=dict(schema='final.local_validation_independent_review.v1',status='passed',
        declared_tasks=8,completed_tasks=8,qualified_tasks=sum(c['qualified'] for c in cases),new_layout_units=2,
        final_goal_online_tasks_used=8,unused_reserved_tasks=8,reviewed_frames=sum(c['checked_frames'] for c in cases),
        cases=cases,pairs=pairs,prefix_checks=prefix_checks,mean_G_J5=average_g,mean_S_J5=average_s,
        mean_delta_J5=average_s-average_g,relative_mean_improvement_percent=100*(average_s-average_g)/average_g,
        wins=sum(p['delta_J5']>1e-12 for p in pairs),ties=sum(abs(p['delta_J5'])<=1e-12 for p in pairs),losses=sum(p['delta_J5']< -1e-12 for p in pairs),
        source_closure_count=len(freeze['source_sha256']),legacy_protected_source_count=len(freeze['protected_20260923_sources']),
        new_worlds=0,new_sensor_queries=0,new_controller_or_planner_runs=0,new_TSDF_runs=0,new_surface_evaluations=0,
        review_scope='all raw packets/contracts/hashes; independent geometry residual/prior/posterior arithmetic; saved candidate argmax and safety; all maps/mesh validity and metric arithmetic; no rerun of DP/TSDF or reference distances',
        inputs=inputs,reviewer_source_sha256=sha(Path(__file__)),helper_source_sha256=sha(ROOT/'scripts/replay_final_local_saved_case_20260929.py'),
        elapsed_seconds=time.monotonic()-started)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True);write(OUTPUT,report)
    print(json.dumps({k:report[k] for k in ('status','completed_tasks','qualified_tasks','reviewed_frames','wins','ties','losses','mean_delta_J5','relative_mean_improvement_percent')},indent=2))


if __name__=='__main__':main()
