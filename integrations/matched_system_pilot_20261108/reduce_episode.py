#!/usr/bin/env python3
"""Reduce one terminal physical episode from records only; no ROS/model/replay."""
import argparse,collections,hashlib,json,pathlib

def reduce_episode(report):
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
        if req.get('method')=='observe' and resp.get('ok') and result.get('accepted') and not result.get('duplicate'):
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
    return dict(schema='nso.matched_episode_record_reduction.v1',report=str(P),terminal=dict(status=root.get('status'),reason=root.get('reason'),method_failed=root.get('method_failed'),execution_interrupted=root.get('execution_interrupted'),core_termination=runtime_terminal),accounting=dict(accepted_observations=len(obs),native_paid_from_last_accepted_observe=native_paid,native_arrived_count=native_arrived,native_paid_action_counts=dict(native_actions),ROS_verified_arrival_records=len(verified),ROS_successful_arrivals_list_count=len(root.get('successful_arrivals',[])),ROS_goals=len(root.get('goals',[])),supervisor_joint_action_records=len(root.get('joint_actions',[])),final_receipt_worker_paid=receipt.get('worker_paid'),final_receipt_supervisor_paid=receipt.get('supervisor_paid'),receipt_error=receipt.get('error'),worker_minus_supervisor=root.get('supervisor_worker_count_difference',{}).get('worker_minus_supervisor')),return_audit=dict(declared_anchor=anchor,last_accepted_state=final_state,exact_discrete_pose_equals_anchor=None if final_state is None or anchor is None else final_state.get('pose')==anchor,recorded_returned_to_anchor=None if runtime_terminal is None else runtime_terminal.get('returned_to_anchor')),physical=dict(CAN_distance_m=root.get('actual_CAN_distance_m',adapter.get('can_distance_m')),CAN_absolute_yaw_rad=root.get('actual_CAN_absolute_yaw_rad',adapter.get('can_absolute_yaw_rad')),elapsed_run_s=root.get('elapsed_run_s'),feedback_accounting_complete=root.get('feedback_accounting_complete',adapter.get('feedback_accounting_complete')),zero_before=root.get('feedback_zero_before_close'),zero_after=root.get('feedback_zero_after_close'),adapter_reason=adapter.get('reason'),physical_stop_independently_observed=root.get('physical_stop_independently_observed')),fixed_support_quality=Q,global_states=globals,semantic_observations=[dict(action_id=x['params'].get('action_id'),quality_labels=x['result'].get('summary',{}).get('semantic_diagnostics',{}).get('quality_labels'),marked_assets=x['result'].get('summary',{}).get('semantic_diagnostics',{}).get('measured_assets')) for x in obs],timing=dict(sidecar_present=timing is not None,sidecar_reason=None if timing is None else timing.get('fault'),first_fault_evidence=adapter.get('fault_evidence'),filename_is_not_execution_fault_verdict=True),input_parse_errors=errors,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},limitations=['No model, mapper, policy replay or ROS; no new coverage estimator.','Native arrived_count increments only when actual discrete pose equals current option.pose and collision is false (runtime_integration.py:353-356); it counts option endpoint arrivals, not ROS primitive arrivals. Native paid and ROS verified primitive arrival counts remain separate; startup without accepted observation has unknown native counts.','Discrete exact anchor equality and recorded return flag are not independent physical position truth.','Shadow rankings are within-state comparisons, not executed counterfactual trajectories; no effectiveness inference.','Q is the native fixed-old-support proxy; negative deltas preserved, not independent reconstruction accuracy.','Fresh-map differences, setup pose deviations and failures must remain visible in matched comparison.'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=reduce_episode(a.report);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(output=str(a.output),terminal=result['terminal'],accounting=result['accounting'])))
if __name__=='__main__':main()
