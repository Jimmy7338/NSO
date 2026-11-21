#!/usr/bin/env python3
"""Descriptive actual two-episode comparison; no ROS/GT/independence claims."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wrap(x):return (float(x)+math.pi)%(2*math.pi)-math.pi

def observations(report):
    rows=[]
    with (report/'events.jsonl').open() as stream:
        for line in stream:
            d=json.loads(line)
            if d.get('method')=='observe' and d.get('response',{}).get('accepted'):
                rows.append((d['params']['path'],d['response']))
    return rows

def packet(path):
    # Selective NPZ access: no RGB/depth/cloud arrays are loaded or exported.
    with np.load(path,allow_pickle=False) as z:
        return dict(grid=np.array(z['joint_belief_grid']),pose=np.array(z['base_odom_pose']).ravel().tolist(),
            rgbd_stamp_s=float(z['rgbd_timestamp_s']))

def common_known(grid,source_planning_from_odom,target_planning_from_odom,target_shape,res=.2):
    """Cell-center projection into C's planning grid through the retained odom frame.

    This is an explicitly lossy rasterization when transforms differ, not GT.
    """
    rc=np.argwhere(grid!=-1);out=np.zeros(target_shape,bool)
    xyz=np.column_stack([(rc[:,1]+.5)*res,(grid.shape[0]-rc[:,0]-.5)*res,np.zeros(len(rc)),np.ones(len(rc))])
    target=(np.asarray(target_planning_from_odom)@np.linalg.inv(np.asarray(source_planning_from_odom))@xyz.T).T
    col=np.floor(target[:,0]/res).astype(int);row=target_shape[0]-1-np.floor(target[:,1]/res).astype(int)
    inside=(row>=0)&(col>=0)&(row<target_shape[0])&(col<target_shape[1])
    out[row[inside],col[inside]]=True
    return out,dict(source_known_cells=int(len(rc)),projected_unique_cells=int(out.sum()),
        out_of_common_domain_cells=int((~inside).sum()))

def arm(report):
    report=Path(report).resolve()
    if not (report/'result.json').exists():
        raise ValueError(f'episode not finalized: {report}; running observations are not final results')
    result=json.loads((report/'result.json').read_text());settings=json.loads((report/'joint_configuration.json').read_text());obs=observations(report)
    duration_limit=json.loads((report/'running.json').read_text()).get('duration_s') if (report/'running.json').exists() else None
    if not obs:
        c=settings['configuration']
        summary=dict(report=str(report),arm=c['score_mode'],status=result.get('status'),reason=result.get('reason'),
            budget=c['total_budget'],duration_limit_s=duration_limit,actual_paid_actions=None,accepted_observations=0,
            ROS_goals=result.get('goals'),verified_arrivals=len(result.get('successful_arrivals',[])),
            CAN_distance_m=result.get('actual_CAN_distance_m'),elapsed_s=result.get('elapsed_run_s'),
            method_failed=result.get('method_failed'),returned_to_discrete_anchor=None,
            observed_area_proxy_m2=dict(initial=None,final=None,delta=None),
            fixed_old_quality_proxy=dict(initial=None,final=None,delta=None,support_sha256=None),
            initial_actual_odom_pose=None,initial_rgbd_stamp_s=None,initial_packet_sha256=None,
            map_source_provenance=settings['calibration'].get('joint_coverage_pair_provenance'),
            limitations=['No accepted observation: failure retained; map/pose/surface proxies unavailable, not zero.'])
        return summary,settings,None,None
    first,last=obs[0][1],obs[-1][1]
    f=first['summary']['runtime']['modules']['feedback'];l=last['summary']['runtime']['modules']['feedback']
    initial,final=packet(obs[0][0]),packet(obs[-1][0]);c=settings['configuration'];accepted=result.get('successful_arrivals',[])
    terminal=last['state']['pose'];anchor=c['return_anchor'];last_arrival=accepted[-1] if accepted else {}
    summary=dict(report=str(report),arm=c['score_mode'],status=result.get('status'),reason=result.get('reason'),
        policy_variant=c['policy_variant'],planner_revision=c['planner_revision'],budget=c['total_budget'],
        duration_limit_s=duration_limit,actual_paid_actions=l['paid_actions'],accepted_observations=len(obs),
        ROS_goals=result.get('goals'),verified_arrivals=len(accepted),CAN_distance_m=result.get('actual_CAN_distance_m'),
        elapsed_s=result.get('elapsed_run_s'),method_failed=result.get('method_failed'),
        terminal_runtime_closed=result.get('joint_summary',{}).get('runtime',{}).get('closed'),
        last_observation_worker_closed=result.get('final_worker_summary',{}).get('closed'),
        physical_stop_independently_observed=result.get('physical_stop_independently_observed'),
        feedback_accounting_complete=result.get('feedback_accounting_complete'),
        return_anchor=anchor,terminal_grid_pose=terminal,returned_to_discrete_anchor=terminal==anchor,
        final_target_error_m=last_arrival.get('target_error_m'),final_target_yaw_error_rad=last_arrival.get('target_yaw_error_rad'),
        feedback_zero_after_close=result.get('feedback_zero_after_close'),adapter_shutdown_reason=result.get('adapter_shutdown_reason'),
        observed_area_proxy_m2=dict(initial=f['observed_2d_area_m2'],final=l['observed_2d_area_m2'],delta=l['observed_2d_area_m2']-f['observed_2d_area_m2']),
        fixed_old_quality_proxy=dict(initial=f['old_quality_proxy'],final=l['old_quality_proxy'],delta=l['old_quality_proxy']-f['old_quality_proxy'],support_count=l['fixed_old_support_count'],support_sha256=l['fixed_old_support_sha256'],initial_support_sha256=f['fixed_old_support_sha256'],
            support_held_fixed_within_arm=f['fixed_old_support_sha256']==l['fixed_old_support_sha256'],
            comparison_scope='within this arm bootstrap-fixed measured support only; different bootstrap supports do not yield matched-quality benefit from absolute between-arm values'),
        initial_actual_odom_pose=initial['pose'],initial_rgbd_stamp_s=initial['rgbd_stamp_s'],
        initial_packet_path=obs[0][0],initial_packet_sha256=sha(obs[0][0]),
        initial_grid_sha256=hashlib.sha256(initial['grid'].tobytes()).hexdigest(),
        final_grid_sha256=hashlib.sha256(final['grid'].tobytes()).hexdigest(),
        initial_planning_from_odom=c['planning_from_odom'],initial_frame=settings['calibration'].get('initial_odom_frame'),
        map_source_provenance=settings['calibration'].get('joint_coverage_pair_provenance'),
        actual_initialization_evidence=settings['calibration'].get('initialization_evidence'),
        navigation_map_source=settings['calibration'].get('navigation_map_source'),
        paired_map_epoch_receipt=settings['calibration'].get('paired_map_epoch_receipt'),
        common_grid_source=settings['calibration'].get('common_grid_source'),
        limitations=['grid return is worker-level acceptance, not independent pose accuracy','surface proxy is not ground-truth reconstruction quality'])
    return summary,settings,initial,final

def compare(c_report,s_report,out,c_summary=None,s_summary=None,protocol=None):
    if protocol is None: raise ValueError("predeclared protocol required")
    proto=json.loads(Path(protocol).read_text()); duration=proto["duration_limit_s"]; budget=proto["paid_action_budget_per_episode"]
    if proto.get("order") != ["C","S"]: raise ValueError("unexpected protocol order")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c,cs,ci,cf=arm(c_report);s,ss,si,sf=arm(s_report)
    if (c['arm'],s['arm'])!=('C','S'):raise ValueError('pass actual C and S reports in the declared order')
    cc,sc=cs['configuration'],ss['configuration']
    method_fields=['policy_variant','planner_revision','shape','robot_radius_m','sensor_config','sensor_range_profile','measured_extent_profile','asset_grouping_profile']
    settings_equal={k:cc.get(k)==sc.get(k) for k in method_fields}
    settings_equal['semantic_configuration']=cs.get('semantic_configuration')==ss.get('semantic_configuration')
    settings_equal['protocol_budget']=cc['total_budget']==sc['total_budget']==budget
    settings_equal['protocol_duration_limits']=c['duration_limit_s']==s['duration_limit_s']==duration
    settings_equal['return_anchor_grid']=cc['return_anchor']==sc['return_anchor']
    have_observations=ci is not None and si is not None
    pose_delta=None if not have_observations else np.asarray(s['initial_actual_odom_pose'])-np.asarray(c['initial_actual_odom_pose'])
    if pose_delta is not None:pose_delta[2]=wrap(pose_delta[2])
    common=None
    if have_observations:
        masks={};mapping={}
        for name,g,config in [('C_initial',ci['grid'],cc),('C_final',cf['grid'],cc),('S_initial',si['grid'],sc),('S_final',sf['grid'],sc)]:
            masks[name],mapping[name]=common_known(g,config['planning_from_odom'],cc['planning_from_odom'],tuple(cc['shape']))
        initdiff=np.count_nonzero(masks['C_initial']^masks['S_initial'])
        common=dict(source='actual packet joint_belief_grid navigation-map input, before native mapper RGBD/scan fusion; distinct from feedback observed-area proxy',frame='C planning grid through each run planning_from_odom and retained odom; shared odom epoch must be checked from actual capture/map receipts',
            resolution_m=.2,cell_area_m2=.04,rasterization='source known-cell centers; lossy when frames differ; no GT or hidden occupancy inferred',
            mapping=mapping,initial_three_state_disagreement_cells_on_identical_transform=int(np.count_nonzero(ci['grid']!=si['grid'])) if cc['planning_from_odom']==sc['planning_from_odom'] and ci['grid'].shape==si['grid'].shape else None,initial_known_mask_symmetric_difference_cells=int(initdiff),initial_known_mask_symmetric_difference_m2=float(initdiff*.04),
            C_initial_known_area_m2=float(masks['C_initial'].sum()*.04),S_initial_known_area_m2=float(masks['S_initial'].sum()*.04),
            C_new_known_area_vs_own_initial_m2=float((masks['C_final']&~masks['C_initial']).sum()*.04),
            S_new_known_area_vs_own_initial_m2=float((masks['S_final']&~masks['S_initial']).sum()*.04))
    cp=c.get('map_source_provenance') or {};sp=s.get('map_source_provenance') or {}
    pair_binding=bool(cp.get('pair_id') and cp.get('pair_id')==sp.get('pair_id') and cp.get('arm')=='C' and sp.get('arm')=='S')
    summary=dict(schema='nso.joint_coverage_pair.descriptive.v1',status='actual_reports_analyzed_no_motion',pair_protocol_provenance_bound=pair_binding,
        declared_order=['C','S'],observed_initial_source_stamp_order_C_before_S=None if not have_observations else c['initial_rgbd_stamp_s']<s['initial_rgbd_stamp_s'],
        experimental_units=dict(scenes=1,ordered_pairs=1 if pair_binding else None,episodes=2,pair_repetitions=1 if pair_binding else None,observations_are_independent_trials=False),
        arms={'C':c,'S':s},configuration_equalities=settings_equal,
        initial_inputs_byte_identical=None if not have_observations else c['initial_packet_sha256']==s['initial_packet_sha256'],
        initial_actual_odom_delta_S_minus_C=None if pose_delta is None else dict(x_m=float(pose_delta[0]),y_m=float(pose_delta[1]),yaw_rad=float(pose_delta[2]),translation_norm_m=float(np.linalg.norm(pose_delta[:2]))),
        planning_transforms_identical=cc['planning_from_odom']==sc['planning_from_odom'],common_coordinate_coverage_proxy=common,
        old_surface_support_identical=None if not have_observations else c['fixed_old_quality_proxy']['support_sha256']==s['fixed_old_quality_proxy']['support_sha256'],
        completion_comparison_scope=f'The common {budget}-action/{duration}s allowance is not equal completed exposure: report actual action counts, interruption and failure; shorter failed S runtime/travel is not an efficiency win and terminal proxy differences are not a paired performance ranking.',
        surface_comparison_scope='Each arm quality proxy uses its own bootstrap-fixed support. Report within-arm change and support hashes; do not interpret absolute C/S value differences as matched-quality benefit. Independent accuracy is not measured.',
        limits=['Two sequential episodes in one scene are not independent environments or a statistical superiority test.',
            'Fresh captures/maps are not assumed identical; initial pose/map/transform/source-epoch differences above remain confounds.',
            'Old S40 is excluded; no matched G/S semantic-effect claim follows from a C/S objective ablation.',
            'Surface support can differ between fresh starts; per-arm quality deltas are not an external accuracy ranking.',
            'Failures, interrupted returns and unequal actual action use remain in the record; never select budget/weights after outcome.'])
    if c_summary is not None and s_summary is not None:
        summaries={m:json.loads(Path(p).read_text()) for m,p in [('C',c_summary),('S',s_summary)]}
        summary['same_state_planning_analysis']={m:dict(source=str(Path(p).resolve()),new_planning_states=summaries[m]['planning']['new_planning_states'],G_S_equal_all_candidates=summaries[m]['planning']['G_S_equal_all_candidates_all_new_states'],states=summaries[m]['planning']['states'],original_runtime_rejections_and_close=summaries[m]['original_runtime_rejections_and_close']) for m,p in [('C',c_summary),('S',s_summary)]}
        summary['S_same_state_C_S_checks']=[dict(paid_actions_before=r['paid_actions_before'],candidate_count=r['candidate_count'],C=r['within_state_choices']['C'],S=r['within_state_choices']['S'],G_equals_S=r['G_equals_S_per_candidate']) for r in summaries['S']['planning']['states']]
        summary['route_attribution_limit']='Different actual sensor states/maps/candidate sets confound cross-arm routes. Use recorded same-state alternatives and original runtime rejection events above; route differences cannot be attributed to joint scoring alone.'
    (out/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    rows=[]
    for mode,a in [('C',c),('S',s)]:
        rows.append(dict(arm=mode,paid_actions=a['actual_paid_actions'],observations=a['accepted_observations'],CAN_m=a['CAN_distance_m'],elapsed_s=a['elapsed_s'],area_initial_m2=a['observed_area_proxy_m2']['initial'],area_final_m2=a['observed_area_proxy_m2']['final'],area_delta_m2=a['observed_area_proxy_m2']['delta'],within_arm_bootstrap_quality_initial=a['fixed_old_quality_proxy']['initial'],within_arm_bootstrap_quality_final=a['fixed_old_quality_proxy']['final'],within_arm_bootstrap_quality_delta=a['fixed_old_quality_proxy']['delta'],bootstrap_support_sha256=a['fixed_old_quality_proxy']['support_sha256'],surface_comparison_scope='within-arm bootstrap support; no matched-quality benefit from cross-arm absolute differences',returned_to_anchor=a['returned_to_discrete_anchor'],reason=a['reason'],method_failed=a['method_failed']))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print(json.dumps(dict(comparison=str(out/'comparison.json'),table=str(out/'comparison.csv')),ensure_ascii=False))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--c-report',required=True,type=Path);ap.add_argument('--s-report',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--protocol',required=True,type=Path);ap.add_argument('--c-summary',type=Path);ap.add_argument('--s-summary',type=Path);a=ap.parse_args();compare(a.c_report,a.s_report,a.output,a.c_summary,a.s_summary,a.protocol)
