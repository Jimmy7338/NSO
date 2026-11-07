#!/usr/bin/env python3
"""Read current C5/S4 logs, actual receipts and original aperture only. No policy/mapper/detector run."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
B=Path('/home/ouc/nso_robot/analyses/joint_coverage_pair_20261107');O=Path(__file__).resolve().parent
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge');sys.path[:0]=[str(W/'joint_vendor'),str(W)]
from env.virtual3d import camera_pose
from env.virtual3d_v2 import VirtualConfigV2
from nso.cpu_four_modules_v10 import aperture_support

def gates(a,p,c):
    back=float((p[:2,3]-a['rear_boundary_xy'])@a['back_axis'])
    if back<=.15:return back,[0,0,0]
    u,v=np.meshgrid(np.linspace(-.5,.5,5)*a['measured_width_m'],np.linspace(a['observed_low'][2],a['observed_high'][2],5))
    pts=np.column_stack([a['rear_boundary_xy'][0]+u.ravel()*a['side_axis'][0],a['rear_boundary_xy'][1]+u.ravel()*a['side_axis'][1],v.ravel()])
    local=(pts-p[:3,3])@p[:3,:3];z=local[:,2];t=np.tan(np.deg2rad(c.fov_deg/2))
    d=(z>.15)&(z<=c.max_depth_m);h=d&(np.abs(local[:,0])<z*t);v=h&(np.abs(local[:,1])<z*t*c.height_px/c.width_px)
    return back,[int(d.sum()),int(h.sum()),int(v.sum())]

rows=[];states=[];frontend={};checks=0
for arm in ['C','S']:
    P=B/'publication'/arm;cfg=json.loads((P/'source/joint_configuration.json').read_text())['configuration'];sensor=dict(cfg['sensor_config']);sensor.update(resolution_m=.2,robot_radius_m=cfg['robot_radius_m']);config=VirtualConfigV2(**sensor)
    receipts=[json.loads(p.read_text()) for p in (P/'semantic_receipts').glob('*.json')];byframe={str(r['frame_id']):r for r in receipts}
    frontend[arm]=dict(receipts=len(receipts),model_executions=sum(r['model_execution_performed'] for r in receipts),frames_with_canonical_category_pixels=sum(r['canonical_labelled_pixels']>0 for r in receipts),class2_assigned_pixel_sum=sum(d['assigned_pixels'] for r in receipts for d in r['detections'] if d.get('v10_class_id')==2),class3_assigned_pixel_sum=sum(d['assigned_pixels'] for r in receipts for d in r['detections'] if d.get('v10_class_id')==3))
    recorded={x['path']:x['sha256'] for x in json.loads((P/'source/source_manifest.json').read_text())['files']}
    for rel in ['nso_bridge/joint_vendor/env/virtual3d.py','nso_bridge/joint_vendor/nso/cpu_four_modules_v10.py','nso_bridge/joint_vendor/nso/competition_candidates_v8.py']:
        assert hashlib.sha256((W.parent/rel).read_bytes()).hexdigest()==recorded[rel],rel
    seen=set();last=None;paid=0
    for line in (P/'decisions.jsonl').read_text().splitlines():
        x=json.loads(line)
        if x['request']['method']=='observe':last=x['response']['result'];paid=x['request']['params']['action_id']
        if x['request']['method']!='next_action':continue
        q=x['response']['result'].get('selection')
        if not q or q['plan_number'] in seen:continue
        seen.add(q['plan_number']);d=last['summary']['semantic_diagnostics'];rc=d['receipt'];assert byframe[str(rc['frame_id'])]==rc
        annotation={a['group']:a for a in d['measured_assets']};geometry=q['candidate_audit'].get('measured_assets',[])
        marked=[(i,a) for i,a in enumerate(geometry) if annotation[a['group']]['marked_points']>0];detail=[]
        for ai,raw in marked:
            a={k:np.asarray(v) if isinstance(v,list) else v for k,v in raw.items()};cases=[]
            for route,score in zip(q['candidates'],q['score_audit']):
                factor=0.;maximum=np.zeros(3,int);maxback=-float('inf');backcount=0
                for st in route['outbound_states'][1:]:
                    p=camera_pose(tuple(st[:2]),st[2],config,cfg['shape'][0]);back,counts=gates(a,p,config);maxback=max(maxback,back);backcount+=back>.15;maximum=np.maximum(maximum,counts)
                    sup=aperture_support(a,p,config);delta=p[:2,3]-a['aabb_center'][:2];sector=int(np.floor((np.arctan2(delta[1],delta[0])+np.pi)/(2*np.pi)*8))%8
                    history=score['measured_direction_support'];novelty=1.-history[ai][sector] if ai<len(history) else 1.;floor=score['measured_novelty_floor'];factor=max(factor,float(sup.mean())*(floor+(1-floor)*novelty));checks+=1
                logged=score['incremental_aperture'][ai];assert math.isclose(factor,logged,abs_tol=1e-12,rel_tol=0)
                zero='rear_side_depth' if not backcount else 'optical_depth_range' if maximum[0]==0 else 'horizontal_FOV' if maximum[1]==0 else 'vertical_FOV' if maximum[2]==0 else 'nonzero_support'
                cases.append(dict(candidate_id=route['candidate_id'],max_back_depth_m=maxback,states_passing_rear=int(backcount),max_points_after_depth=int(maximum[0]),max_points_after_horizontal=int(maximum[1]),max_points_after_vertical=int(maximum[2]),first_zero=zero,logged_factor=logged,original_function_factor=factor))
            detail.append(dict(asset_index=ai,group=a['group'],marked_points=annotation[a['group']]['marked_points'],class_vote=annotation[a['group']]['class_vote'],original_target_audit=[t for t in q['candidate_audit'].get('target_audit',[]) if t['asset_index']==ai],candidates=cases))
        cat=[z for z in rc['detections'] if z.get('v10_class_id') in [2,3]];labels=sum(d['quality_labels'].values())
        earliest=('detected_category_has_no_valid_depth' if cat else 'no_category_depth_support') if labels==0 else 'quality_category_points_do_not_enter_qualified_assets' if not marked else 'qualified_marked_assets_have_zero_candidate_aperture'
        row=dict(arm=arm,plan=q['plan_number'],paid=paid,original_remaining_budget=q['candidate_audit']['remaining_budget'],candidates=len(q['candidates']),current_canonical_category_pixels=rc['canonical_labelled_pixels'],history_quality_class2=d['quality_labels']['2'],history_quality_class3=d['quality_labels']['3'],qualified_assets=len(d['measured_assets']),qualified_marked_assets=len(marked),earliest_semantic_loss=earliest,all_logged_G_equal_S=q['scores']['G']==q['scores']['S'],candidate_status=q['candidate_audit']['status'])
        rows.append(row);states.append(dict(**row,frontend_category_detections=[{k:z[k] for k in ['source_label','v10_class_id','valid_depth_pixels','assigned_pixels']} for z in cat],qualified_asset_annotations=d['measured_assets'],marked_asset_candidate_gate_checks=detail,scores={m:q['scores'][m] for m in ['C','G','S','X']},empty_call_is_not_a_choice=not bool(q['candidates'])))
result=dict(schema='nso.current_pair_semantic_chain.v1',frontend=frontend,table=rows,states=states,original_camera_asset_state_checks=checks,original_function_factors_match=True,limits=['Logged quality-to-asset loss localizes geometry qualification/grouping; exact rejected group predicates were not reconstructed.','Current receipt may have no class pixels while accumulated category quality persists.','S final empty call is the original circular current-footprint failure, not a semantic comparison.','Predicted categories and hypothetical aperture are not verified identities, physical visibility or independent accuracy.','No mapper, detector, planner, motion, budget/weight or runtime change.'])
result['native_function_sources_match_both_actual_episode_manifests']=True
result['conclusion']='No category-score cancellation: all qualified marked votes are+1, but their original candidate factors are0. Category prior is never reached with nonzero marked support in the actual recorded choices. Current evidence does not prove global physical-scene impossibility or semantic efficacy.'
result['next_bounded_diagnostic']={'status': 'recommended_not_executed', 'known_scene_constraint': 'Only the actual large display platform is established; no open shelf/small cabinet/box is assumed to exist. Original circle and unknown-exploration objective retained.', 'logged_route_answer': 'No reliable new feasible state or minimum budget is determined by these logs. At C paid21/22/23 original marked entry is absent at remaining19/18/17, deep lacks entry and quota is not reached. Desired rear coordinates are hypotheses, not reachable waypoints; same group number is not persistent object identity.', 'minimum_informative_next_test': ['One bounded mapper-only reconstruction through actual C action21/observation22, with original cached semantics and existing3m measured-extent adapter; no policy/detector/ROS replay.', 'On that frozen actual state use the original circular known-safe orientation graph, directed Dijkstra and native aperture/wanted heading to compute the minimum full current-view-original-anchor cost for its one qualified marked group.', 'Report either a recorded reachable state/path and exact cost or no support in the observed connected component. Do not scan budgets/weights, infer unknown free space or turn desired entry coordinates into commanded goals.'], 'physical_experiment_interpretation': 'This is experimental-condition selection, not a new runtime gate/filter. If the original measured graph has no support, increased budget cannot alone fix that graph. Do not repeat known-equal G/S trials under the same condition; retain current large-platform applicability boundary. Any changed real start/layout remains unknown-space exploration and uses actual scene geometry, with preparation costs and initial-information differences recorded.', 'semantic_claim': 'No semantic benefit; no fabricated object, class2 structure, score amplification, regrouping or smaller research circle.'}
result['source_sha256']={str(O/'audit_chain.py'):hashlib.sha256((O/'audit_chain.py').read_bytes()).hexdigest()}
result['original_function_source_sha256']={str(W/x):hashlib.sha256((W/x).read_bytes()).hexdigest() for x in ['joint_vendor/env/virtual3d.py','joint_vendor/nso/cpu_four_modules_v10.py','joint_vendor/nso/competition_candidates_v8.py']}
(O/'semantic_chain.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (O/'semantic_chain.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(dict(frontend=frontend,table=rows,checks=checks,marked_details=[dict(arm=z['arm'],paid=z['paid'],assets=z['marked_asset_candidate_gate_checks']) for z in states if z['marked_asset_candidate_gate_checks']]),ensure_ascii=False))
