#!/usr/bin/env python3
"""Reduce one terminal physical episode from records only; no ROS/model/replay."""
import argparse,collections,hashlib,json,pathlib

def _base_reduce(report):
    P=pathlib.Path(report).resolve();rootpath=P/'result.json'
    if not rootpath.is_file():raise ValueError('Terminal result.json required; no polling or inferred completion')
    load=lambda p:json.loads(p.read_text())
    root=load(rootpath);configpath=P/'joint_configuration.json';settings=load(configpath) if configpath.exists() else {};cfg=settings.get('configuration',{})
    rows=[];errors=[];rpc=P/'joint/decisions.jsonl'
    if rpc.exists():
        for line,text in enumerate(rpc.read_text().splitlines(),1):
            try:rows.append((line,json.loads(text)))
            except (ValueError,TypeError) as exc:errors.append(dict(line=line,error=str(exc)))
    obs=[];globals=[];seen=set();prefix=None;runtime_terminal=None
    for line,row in rows:
        req=row.get('request',{});resp=row.get('response',{});result=resp.get('result') or {}
        if not isinstance(result,dict):continue
        term=((result.get('summary') or {}).get('runtime') or {}).get('termination')
        if term is not None:runtime_terminal=term
        if req.get('method') in ('observe','observe_failed_action') and resp.get('ok') and result.get('accepted') and not result.get('duplicate'):
            prefix=req.get('params',{}).get('action_id');obs.append(dict(line=line,request_id=req.get('id'),params=req.get('params',{}),result=result))
        if req.get('method')!='next_action':continue
        s=result.get('selection')
        if not s or s.get('plan_number') in seen:continue
        seen.add(s.get('plan_number'));candidates=s.get('candidates',[]);modes={}
        for mode in ['C','G','S','X']:
            scores=s.get('scores',{}).get(mode)
            if scores is None or len(scores)!=len(candidates):
                modes[mode]=dict(error='missing/mismatched score vector');continue
            order=sorted(range(len(candidates)),key=lambda i:(-scores[i],candidates[i]['cost'],candidates[i]['candidate_id']))
            modes[mode]=dict(rank=[candidates[i]['candidate_id'] for i in order],scores=scores,admissible_top1=None if not order or scores[order[0]]<=0 else candidates[order[0]]['candidate_id'],top1_margin=None if len(order)<2 else scores[order[0]]-scores[order[1]])
        active=[dict(candidate_id=a['candidate_id'],factors=a.get('incremental_aperture'),prior_G=a.get('prior_potentials',{}).get('G'),prior_S=a.get('prior_potentials',{}).get('S'),prior_X=a.get('prior_potentials',{}).get('X')) for a in s.get('score_audit',[]) if any(v!=0 for v in a.get('incremental_aperture',[]))]
        globals.append(dict(line=line,plan_number=s.get('plan_number'),paid_prefix=prefix,actual_mode=s.get('mode'),actual_selected=None if s.get('selected') is None else s['selected'].get('candidate_id'),candidate_ids=[x['candidate_id'] for x in candidates],candidate_costs=[x['cost'] for x in candidates],shadow_modes=modes,active_aperture_candidates=active,belief_sha256=s.get('candidate_audit',{}).get('belief_sha256')))
    first=None if not obs else obs[0];last=None if not obs else obs[-1]
    summary={} if last is None else last['result'].get('summary',{});runtime=summary.get('runtime',{});feedback=runtime.get('modules',{}).get('feedback',{});initial={} if first is None else first['result'].get('summary',{}).get('runtime',{}).get('modules',{}).get('feedback',{})
    native_paid=summary.get('actual_paid_actions');native_arrived=runtime.get('arrived_count');final_state=None if last is None else last['result'].get('state');anchor=cfg.get('return_anchor',runtime.get('modules',{}).get('return_anchor'))
    native_actions=collections.Counter(x['params'].get('action') for x in obs if x['params'].get('action_id',0)>0);native_actions.pop(None,None)
    verified=[x for x in root.get('arrival_evidence',[]) if x.get('verified') is True]
    adapterpath=P/'adapter/result.json';adapter=load(adapterpath) if adapterpath.exists() else {};timingpath=P/'adapter/control_timing_fault.json';timing=load(timingpath) if timingpath.exists() else None
    Q=dict(initial_fixed_support=initial.get('fixed_old_support_count'),initial_support_sha256=initial.get('fixed_old_support_sha256'),final_fixed_support=feedback.get('fixed_old_support_count'),final_support_sha256=feedback.get('fixed_old_support_sha256'),initial_Q=initial.get('initial_old_quality_proxy'),final_Q=feedback.get('old_quality_proxy'),recorded_delta=feedback.get('old_quality_proxy_total_change'))
    Q['arithmetic_delta']=None if Q['initial_Q'] is None or Q['final_Q'] is None else Q['final_Q']-Q['initial_Q'];Q['support_identity_equal']=None if Q['initial_support_sha256'] is None else Q['initial_support_sha256']==Q['final_support_sha256']
    receipt=root.get('final_worker_observation_receipt',{})
    inputs=[p for p in [rootpath,configpath,rpc,adapterpath,timingpath] if p.exists()]
    return dict(schema='nso.matched_episode_record_reduction.v1',report=str(P),terminal=dict(status=root.get('status'),reason=root.get('reason'),method_failed=root.get('method_failed'),execution_interrupted=root.get('execution_interrupted'),core_termination=runtime_terminal),accounting=dict(accepted_observations=len(obs),native_paid_from_last_accepted_observe=native_paid,native_arrived_count=native_arrived,native_paid_action_counts=dict(native_actions),ROS_verified_arrival_records=len(verified),ROS_successful_arrivals_list_count=len(root.get('successful_arrivals',[])),ROS_goals=len(root.get('goals',[])),supervisor_joint_action_records=len(root.get('joint_actions',[])),final_receipt_worker_paid=receipt.get('worker_paid'),final_receipt_supervisor_paid=receipt.get('supervisor_paid'),receipt_error=receipt.get('error'),worker_minus_supervisor=(root.get('supervisor_worker_count_difference') or {}).get('worker_minus_supervisor')),return_audit=dict(declared_anchor=anchor,last_accepted_state=final_state,exact_discrete_pose_equals_anchor=None if final_state is None or anchor is None else final_state.get('pose')==anchor,recorded_returned_to_anchor=None if runtime_terminal is None else runtime_terminal.get('returned_to_anchor')),physical=dict(CAN_distance_m=root.get('actual_CAN_distance_m',adapter.get('can_distance_m')),CAN_absolute_yaw_rad=root.get('actual_CAN_absolute_yaw_rad',adapter.get('can_absolute_yaw_rad')),elapsed_run_s=root.get('elapsed_run_s'),feedback_accounting_complete=root.get('feedback_accounting_complete',adapter.get('feedback_accounting_complete')),zero_before=root.get('feedback_zero_before_close'),zero_after=root.get('feedback_zero_after_close'),adapter_reason=adapter.get('reason'),physical_stop_independently_observed=root.get('physical_stop_independently_observed')),fixed_support_quality=Q,global_states=globals,semantic_observations=[dict(action_id=x['params'].get('action_id'),quality_labels=x['result'].get('summary',{}).get('semantic_diagnostics',{}).get('quality_labels'),marked_assets=x['result'].get('summary',{}).get('semantic_diagnostics',{}).get('measured_assets')) for x in obs],timing=dict(sidecar_present=timing is not None,sidecar_reason=None if timing is None else timing.get('fault'),first_fault_evidence=adapter.get('fault_evidence'),filename_is_not_execution_fault_verdict=True),input_parse_errors=errors,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},limitations=['No model, mapper, policy replay or ROS; no new coverage estimator.','Native arrived_count increments only when actual discrete pose equals current option.pose and collision is false (runtime_integration.py:353-356); it counts option endpoint arrivals, not ROS primitive arrivals. Native paid and ROS verified primitive arrival counts remain separate; startup without accepted observation has unknown native counts.','Discrete exact anchor equality and recorded return flag are not independent physical position truth.','Shadow rankings are within-state comparisons, not executed counterfactual trajectories; no effectiveness inference.','Q is the native fixed-old-support proxy; negative deltas preserved, not independent reconstruction accuracy.','Fresh-map differences, setup pose deviations and failures must remain visible in matched comparison.'])

def recovery_accounting(report, root=None):
    """Journal snapshot may be running: never infer terminal completion."""
    P=pathlib.Path(report);events=[]
    event_path=P/'events.jsonl'
    if event_path.exists():
        for text in event_path.read_text().splitlines():
            try:events.append(json.loads(text))
            except ValueError:continue
    recoveries={};primary={};sent={}
    for event in events:
        name=event.get('event');goal=event.get('goal') or {};gid=goal.get('id')
        if name=='ROS_backoff_recovery_goal_sent' and gid:sent[gid]=event
        if name=='ROS_backoff_recovery_result' and gid:recoveries[gid]=event
        if name=='NSO_joint_action_result' and gid:primary[gid]=event
    root=root or {}
    for event in root.get('recovery_interventions',[]):
        gid=(event.get('goal') or {}).get('id')
        if gid:recoveries[gid]=event
    def delta(before,after,key):
        if before.get(key) is None or after.get(key) is None:return None
        return after[key]-before[key]
    recovery=[];ordinary=[]
    for gid,e in recoveries.items():
        recovery.append(dict(goal_id=gid,original_action_id=e.get('original_action_id'),status=(e.get('terminal') or {}).get('status'),CAN_distance_m=delta(e.get('CAN_before') or {},e.get('CAN_after') or {},'distance_m'),elapsed_s=e.get('elapsed_s'),actual_odom_translation_m=e.get('actual_odom_translation_m'),target_tolerance_verified=(e.get('target_error') or {}).get('verified'),NSO_arrival_credit=False,additional_NSO_primitive_charge=e.get('additional_NSO_primitive_charge'),CAN_accounting_complete=all((e.get(k) or {}).get('accounting_complete') is True for k in ('CAN_before','CAN_after'))))
    for gid,e in primary.items():
        a=e.get('arrival') or {};b=a.get('before') or {};c=a.get('after') or {}
        ordinary.append(dict(goal_id=gid,action_id=e.get('action_id'),status=(a.get('terminal') or {}).get('status'),ROS_arrival_verified=a.get('verified'),NSO_endpoint_verified=(e.get('endpoint_error') or {}).get('verified'),CAN_distance_m=delta(a.get('can_start') or {},a.get('can_end') or {},'distance_m'),pose_bracket_elapsed_s=delta(b,c,'receipt_monotonic')))
    def total(rows,key):
        values=[r.get(key) for r in rows]
        return None if any(v is None for v in values) else sum(values)
    ids=set(sent)|set(recoveries)
    return dict(source_scope='terminal result plus journal' if root else 'NONTERMINAL journal snapshot; counts may grow',recovery_goal_ids=sorted(ids),recovery_goals_sent=len(ids),recovery_goals_completed=len(recovery),recovery_goals_pending=sorted(ids-set(recoveries)),recovery_terminal_status_counts=dict(collections.Counter(str(x['status']) for x in recovery)),recovery_interventions=recovery,recovery_CAN_distance_m=total(recovery,'CAN_distance_m'),recovery_elapsed_s=total(recovery,'elapsed_s'),recovery_additional_NSO_charge=sum(x.get('additional_NSO_primitive_charge') or 0 for x in recovery),NSO_primitive_goal_records=ordinary,NSO_primitive_goal_status_counts=dict(collections.Counter(str(x['status']) for x in ordinary)),NSO_primitive_goal_CAN_distance_m=total(ordinary,'CAN_distance_m'),NSO_primitive_goal_pose_bracket_elapsed_s=total(ordinary,'pose_bracket_elapsed_s'),cost_scope='Recovery and primitive brackets do not include all capture/planning/waits. Primary efficiency must use full episode CAN distance and elapsed time including recoveries.')

def reduce_episode(report):
    """All accepted observations count, failed attempts never become arrivals."""
    P=pathlib.Path(report).resolve();result=_base_reduce(P)
    root=json.loads((P/'result.json').read_text());accepted=[];failed=[];normal=[];plans=[];observed=[];last=None;last_grid=None
    parse_errors=[];initial_observed_area=None
    import numpy as np
    for number,text in enumerate((P/'joint/decisions.jsonl').read_text().splitlines(),1):
        try:row=json.loads(text)
        except ValueError:continue
        req=row.get('request',{});response=row.get('response',{});value=response.get('result') or {};method=req.get('method');params=req.get('params',{})
        if not isinstance(value,dict):continue
        if method in ('observe','observe_failed_action') and response.get('ok') and value.get('accepted') and not value.get('duplicate'):
            state=value.get('state',{});aid=params.get('action_id');entry=dict(line=number,method=method,action_id=aid,action=params.get('action'),state=state)
            if method=='observe_failed_action':
                event=value.get('failed_execution_attempt') or {};entry['failure']=event;failed.append(entry)
                if event.get('arrival_credited') is not False or event.get('paid_attempt_charged') is not True:parse_errors.append(dict(line=number,error='failed attempt receipt does not explicitly reject arrival and charge attempt'))
            else:normal.append(entry)
            prior_unknown=None
            pose=state.get('pose')
            if last_grid is not None and pose is not None and 0<=pose[0]<last_grid.shape[0] and 0<=pose[1]<last_grid.shape[1]:prior_unknown=int(last_grid[tuple(pose[:2])])==-1
            observed.append(dict(action_id=aid,method=method,actual_pose=pose,actual_base_pose_planning=state.get('actual_base_pose_planning'),current_cell_unknown_in_previous_observed_grid=prior_unknown,arrival_credited=False if method=='observe_failed_action' else aid!=0))
            path=pathlib.Path(params.get('path',''))
            if not path.exists() and path.name:path=P/'joint_inputs'/path.name
            if path.is_file():
                try:
                    with np.load(path,allow_pickle=False) as z:last_grid=z['joint_belief_grid'].copy()
                except (KeyError,ValueError,OSError) as exc:parse_errors.append(dict(line=number,error='observed grid unavailable: '+str(exc)));last_grid=None
            else:last_grid=None
            if not accepted:
                initial_observed_area=value.get("summary",{}).get("runtime",{}).get("modules",{}).get("feedback",{}).get("observed_2d_area_m2")
            accepted.append(entry);last=value
        if method=='next_action' and response.get('ok'):
            sel=value.get('selection') or {};selected=sel.get('selected') or {};states=selected.get('outbound_states',[]);unknown=None
            if last_grid is not None:
                cells={tuple(p[:2]) for p in states if len(p)>=2 and 0<=p[0]<last_grid.shape[0] and 0<=p[1]<last_grid.shape[1]};unknown=sum(int(last_grid[c])==-1 for c in cells)
            plans.append(dict(line=number,plan_number=sel.get('plan_number'),pending_action=value.get('action'),planned_outbound_state_count=len(states),planned_unknown_cell_count=unknown,scope='planned only; not evidence of traversal'))
    summary={} if last is None else last.get('summary',{});runtime=summary.get('runtime',{});paid=[x for x in accepted if isinstance(x['action_id'],int) and x['action_id']>0];successful=[x for x in normal if isinstance(x['action_id'],int) and x['action_id']>0]
    result['schema']='nso.v2_episode_record_reduction.v2'
    result['accounting']=dict(accepted_total_observations=len(accepted),accepted_normal_observations=len(normal),accepted_failed_attempt_observations=len(failed),bootstrap_observations=sum(x['action_id']==0 for x in normal),accepted_paid_attempts=len(paid),last_paid_attempt_id=None if not paid else paid[-1]['action_id'],worker_actual_paid_attempts=summary.get('actual_paid_actions'),NSO_exact_primitive_success_observations=len(successful),NSO_failed_partial_attempts=len(failed),native_option_endpoint_arrivals=runtime.get('arrived_count'),ROS_verified_arrival_records=sum(x.get('verified') is True for x in root.get('arrival_evidence',[])),ROS_successful_arrivals_list_count=len(root.get('successful_arrivals',[])),ROS_goals=len(root.get('goals',[])),ROS_terminal_status_counts=dict(collections.Counter(str(x.get('status')) for x in root.get('goal_results',[]))),unobserved_attempts_not_charged_by_reducer=True)
    result['last_actual_state']=None if not accepted else accepted[-1]['state']
    recovery=recovery_accounting(P,root)
    result['recovery_accounting']=recovery
    recovery_ids=set(recovery['recovery_goal_ids'])
    verified=root.get('arrival_evidence',[])
    result['accounting']['ROS_verified_recovery_arrival_records']=sum(x.get('verified') is True and (x.get('goal') or {}).get('id') in recovery_ids for x in verified)
    result['accounting']['ROS_verified_nonrecovery_arrival_records']=sum(x.get('verified') is True and (x.get('goal') or {}).get('id') not in recovery_ids for x in verified)
    result['accounting']['ROS_verified_arrival_records_scope']='All ROS goals, including recovery; use split fields or NSO_exact_primitive_success_observations for primitive success.'
    ledger=runtime.get('modules',{}).get('feedback',{})
    area=ledger.get('observed_2d_area_m2');distance=result['physical'].get('CAN_distance_m');elapsed=result['physical'].get('elapsed_run_s')
    gain=None if area is None or initial_observed_area is None else area-initial_observed_area
    result['coverage_efficiency']=dict(initial_observed_2d_area_m2=initial_observed_area,observed_area_gain_m2=gain,gain_per_total_CAN_m=None if gain is None or not distance else gain/distance,gain_per_total_second=None if gain is None or not elapsed else gain/elapsed,observed_2d_area_m2=area,total_CAN_distance_m_including_recovery=distance,total_elapsed_s_including_recovery=elapsed,area_per_total_CAN_m=None if area is None or not distance else area/distance,area_per_total_second=None if area is None or not elapsed else area/elapsed,scope='Native observed area / full physical episode cost. Total area includes bootstrap/prior support; gain subtracts first accepted bootstrap ledger. Neither is independent ground truth.',paid_attempt_curves_role='secondary; recoveries charge no additional NSO primitive but physical costs remain included')

    result['accepted_attempt_outcomes']=paid
    result['route_evidence']=dict(planned=plans,observed_exposure_poses=observed,observed_cell_changes=sum(a.get('actual_pose',[])[:2]!=b.get('actual_pose',[])[:2] for a,b in zip(observed,observed[1:])),actual_new_cell_unknown_in_previous_observed_grid_count=sum(x['current_cell_unknown_in_previous_observed_grid'] is True for x in observed),scope='Exposure-pose cells and preceding supplied observed navigation grid only. This is not continuous traversal clearance, native fused belief, or full route coverage; a planned unknown route is not an executed route.')
    result['input_parse_errors'].extend(parse_errors)
    result['limitations'].append('Failed-action RPC observations charge attempts and fuse actual partial pose without arrival. Normal accepted paid observe is exact worker primitive success; native option arrivals and ROS verified records remain distinct.')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=pathlib.Path,required=True);parser.add_argument('--output',type=pathlib.Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    value=reduce_episode(args.report);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(output=str(args.output),terminal=value['terminal'],accounting=value['accounting'])))
if __name__=='__main__':main()
