#!/usr/bin/env python3
"""Score one actual prepared view through the original joint worker; no ROS or goal execution."""
import argparse, hashlib, json, math, os, pathlib, sys

WORK=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029')
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def choices(selection):
    if selection is None: return {mode:None for mode in ('C','G','S','X')}
    candidates=selection['candidates']; scores=selection['scores']; result={}
    for mode in ('C','G','S','X'):
        if len(candidates)!=len(scores[mode]): raise ValueError('score/candidate length mismatch')
        indices=sorted(range(len(candidates)),key=lambda i:(-scores[mode][i],candidates[i]['cost'],candidates[i]['candidate_id']))
        if not indices or scores[mode][indices[0]]<=0: result[mode]=None; continue
        i=indices[0];c=candidates[i];result[mode]={'candidate_id':c['candidate_id'],'pose':c['pose'],'score':scores[mode][i]}
    return result

def validate(settings,source):
    c=settings['configuration'];cal=settings['calibration'];s=c['sensor_config']
    checks={'score_S':c['score_mode']=='S','revision':c['planner_revision']=='v10_3_1','policy':c['policy_variant']=='exploration_joint_v1','budget40':c['total_budget']==40,'circle':math.isclose(c['robot_radius_m'],.7212185868375829,rel_tol=0,abs_tol=1e-12),'range':c['sensor_range_profile']=='sensor_range_matched_v1','extent':c['measured_extent_profile']=='measured_extent_3m_v1','no_instance_adapter':c.get('asset_grouping_profile') is None,'camera_Nov06':cal['camera_extrinsic_policy']=='confirmed_Nov06_composed_TF_actual_capture_match','camera_TF_receipt_match':cal['camera_TF_max_abs_error']<=1e-9,'ground_profile':cal['floor_reference_profile']=='depth_ground_reference_v1','ground_training_scalar':math.isclose(cal['ground_reference_receipt']['floor_relative_to_training_base_m'],-.12447258789516671,rel_tol=0,abs_tol=1e-12),'camera_nominal':math.isclose(s['camera_height_m'],.536,abs_tol=1e-12),'laser_nominal':math.isclose(s['laser_height_m'],.684,abs_tol=1e-12),'frontend':bool(settings.get('semantic_configuration')),'no_motion_capture':source.get('no_motion') is True}
    if not all(checks.values()):raise ValueError('configuration contract failed: '+str([k for k,v in checks.items() if not v]))
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--configuration',type=pathlib.Path,required=True);p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    settings=json.loads(a.configuration.read_text());source=json.loads(a.source.read_text());checks=validate(settings,source)
    packet=pathlib.Path(source['joint_input']);
    if not packet.is_file():raise FileNotFoundError(packet)
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(WORK/'nso_bridge'))
    from nso_bridge.joint_engine import JointSession
    params={'configuration':settings['configuration'],'calibration':settings['calibration'],'semantic_configuration':settings['semantic_configuration'],'execution_allowed':False,'semantic_output':str(a.output/'semantics')}
    with (a.output/'decisions.jsonl').open('w',buffering=1) as journal:
        def emit(id,method,params,result):
            row={'request':{'id':id,'method':method,'params':params},'response':{'id':id,'ok':True,'result':result}};journal.write(json.dumps(row,allow_nan=False)+'\n');return row
        session=JointSession(**params);emit(1,'init',params,session.identity())
        op={'path':str(packet),'action_id':0,'action':None};obs=session.observe(**op);emit(2,'observe',op,obs)
        if not obs['accepted'] or obs.get('duplicate'):raise ValueError('observation0 was not accepted once')
        result=session.next_action();emit(3,'next_action',{},result)
    selection=result.get('selection'); summary={'schema':'nso.semantic_development_view.first_state.v1','development_only':True,'scope':'Single observed development view; within-state shadow scores are not executed trajectories or physical counterfactuals.','configuration':str(a.configuration.resolve()),'configuration_sha256':sha(a.configuration),'source':str(a.source.resolve()),'source_sha256':sha(a.source),'joint_packet':str(packet.resolve()),'joint_packet_sha256':sha(packet),'contract_checks':checks,'first_state':obs['state'],'semantic_diagnostics':obs['summary']['semantic_diagnostics'],'semantic_diagnostics_stage':'after_observe0_before_planning','post_plan_semantic_diagnostics':result['summary']['semantic_diagnostics'],'post_plan_semantic_diagnostics_stage':'after_next_action','proposed_S_action':result['action'],'proposed_S_selected':result['selected'],'within_state_choices':choices(selection),'selection':selection,'runtime_events':result['runtime_events'],'decisions_sha256':sha(a.output/'decisions.jsonl')}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':str(a.output/'summary.json'),'choices':summary['within_state_choices']}))
if __name__=='__main__':main()
