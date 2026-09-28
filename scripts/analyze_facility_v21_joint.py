#!/usr/bin/env python3
"""Verify completed V21 G4 against equivalence-certified V20 N4, then plot.

Reads saved results only. No simulator, mapper, policy or new sensing is run.
All four pairs, failed endpoints and all three reference sensitivities remain.
"""
import argparse
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_facility_v20_probe as old
from scripts.analyze_facility_v20_probe import require, read, sha, digest, close, inside, verify_inventory, identity, write_csv

DEFAULT_G = ROOT / 'eval_results/facility_v21_joint_probe_20260915'
DEFAULT_N = ROOT / 'eval_results/facility_v20_coverage_probe_20260915'
DEFAULT_EQ = ROOT / 'audit_results/facility_v21_n_equivalence_20260915/result.json'
DEFAULT_OUT = ROOT / 'audit_results/facility_v21_joint_analysis_20260915'
NOTES = [
 'Four paired development configurations; replays and three reference samplings are not additional independent experiments.',
 'Primary endpoint: true evaluation C2D times six-facility equal-weight external F1@5cm, reference seed 2026. Missing facilities score zero.',
 'Eligibility requires C2D>=0.8, returned, zero collisions and no terminal failure. All cases and raw endpoint differences are retained; ineligible pairs do not establish qualified improvement.',
 'C_ROI counts observed known free and occupied cells over a fixed public ROI. It is not true reachable-floor coverage, a certified bound or motion permission. No posthoc coverage correction is applied.',
 'N is reused only after four saved-history V21-N equivalence certificates tied to the exact G frozen runtime/helper source hashes.',
 'Budget passes, positive reconstruction terms, noncoverage roles, selections, paid quality-intent actions, measured arrivals and explicit cancellations are distinct counts.',
 'Return actions can have coverage_intent=false. They are excluded from quality-intent paid action counts by authorization phase and option identity.',
 'A quality-role arrival is a measured target-pose event, not evidence that the view improved reconstruction. Checkpoint quality changes are not attributed to an individual option.',
 'Quality curves contain recorded 20-action checkpoints and the terminal point only; connecting lines are guides. No interpolation is used for scoring.',
 'This G/N probe does not test semantic information value, a learned semantic network, natural open vocabulary or real-robot efficacy.',
]


def atomic_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), 'Nonfinite numeric ' + label)
    return float(value)


def metric_check(metric):
    c = finite(metric['coverage_2d'], 'coverage'); require(0 <= c <= 1, 'Coverage out of range')
    instances = metric['instances']; require(len(instances) == 6 and sorted(x['id'] for x in instances) == list(range(6)), 'All six unique facilities required')
    require(metric['mission_asset_count'] == 6 and metric['missing_asset_count'] == sum(x['missing'] for x in instances), 'Missing-facility count differs')
    for threshold in ('02cm', '05cm', '10cm'):
        values = []
        for item in instances:
            row = item[threshold]; p, r, f = [finite(row[k], k) for k in ('precision', 'recall', 'f1')]
            require(all(0 <= v <= 1 for v in (p, r, f)), 'Surface score outside [0,1]')
            close(f, 2*p*r/(p+r) if p+r else 0., 'Instance F1')
            require(not item['missing'] or (p == r == f == 0), 'Missing facility must score zero')
            values.append(f)
        q = sum(values)/6; row = metric[threshold]
        close(row['external_macro_f1'], q, 'Six-facility macro F1')
        close(row['asset_macro_f1'], q, 'Legacy F1 alias')
        close(row['joint_external'], c*q, 'Primary joint formula')
        close(row['joint_asset'], c*q, 'Legacy joint alias')
    eligible = c >= .8 and metric['returned'] and metric['collisions'] == 0 and not metric['failed']
    require(metric['eligible'] == eligible, 'Eligibility contradicts coverage/return/collision/failure')


def verify_g_source(source):
    manifest = read(source / 'manifest.json'); require(manifest.get('status') == 'complete', 'G manifest must be complete before analysis')
    require(manifest.get('mode') == 'G' and manifest['version'] == 'facility-joint-online-budget-v21-probe-1', 'Expected completed V21 G batch')
    config = manifest['config']; expected = [(p,a) for p in ('D19-P00','D19-P01') for a in ('A_complex_B_simple','A_simple_B_complex')]
    require([(c['parent'],c['assignment']) for c in manifest['cases']] == expected, 'Exactly four G paired configurations required')
    require([c['index'] for c in manifest['cases']] == list(range(4)), 'Case indices must be 0..3')
    require(config['reference_seeds'] == [2026,2027,2028] and config['primary_reference_seed'] == 2026, 'Reference seeds changed')
    require(config['coverage_minimum'] == .8 and config['quality_curve_action_stride'] == 20 and config['coverage_trace_action_stride'] == 1, 'Frozen gate/curve sampling changed')
    require(config['physical_contract'] == {'max_depth_m':5.,'voxel_m':.04,'truncation_m':.12}, 'Physical contract changed')
    require(config['sensor_model'] == 'iid_025px' and config['noise_seed'] == 1901, 'Sensor condition changed')
    proxy = config['planning_coverage_proxy']
    require(proxy['kind'] == 'known_cells_over_fixed_public_roi' and proxy['target_fraction'] == .8 and proxy['target_is_heuristic'] is True and proxy['evaluation_truth_used'] is False and proxy['true_coverage_guaranteed'] is False, 'ROI proxy contract changed')
    require(config['replan_interval_actions'] == 5 and config['coverage_candidate_slots'] == 8 and config['task_asset_count'] == 6, 'Shared planning contract changed')
    require(not manifest['training_allowed'] and not manifest['semantic_information_test'] and not manifest['full_architecture_efficacy_proven'], 'Unexpected efficacy/training flag')
    require(config['initial_history_actions'] == 0 and not config['install_v19_declared_first_option'], 'Unexpected paid prefix or installed first option')
    count = verify_inventory(source)
    for case in manifest['cases']:
        require(case['mode'] == 'G' and case['budget'] == {'D19-P00':160,'D19-P01':184}[case['parent']], 'Case mode/budget differs')
        require(case['initial_history_actions'] == 0 and not case['declared_first_option_installed'], 'Case prefix/first-option differs')
        folder = source / f'case_{case["index"]:02d}'; verify_inventory(folder)
        receipt = read(folder / 'verification.json')
        require(receipt['status'] == 'passed' and receipt['independent_process'] is True and receipt['fresh_world_sensor_action_metric_replay'] is True and receipt['physical_process_id'] != receipt['replay_process_id'], 'Four fresh G process replays required')
    frozen = manifest['source_sha256']
    with zipfile.ZipFile(source / 'sources.zip') as archive:
        require(set(archive.namelist()) == set(frozen), 'Frozen G source archive differs')
        for name, wanted in frozen.items(): require(hashlib.sha256(archive.read(name)).hexdigest() == wanted, 'Frozen G source hash differs: '+name)
    reference = Path(manifest['reference_source_root'])
    require(sha(reference/'manifest.json') == manifest['reference_source_manifest_sha256'] and sha(reference/'artifact_hashes.json') == manifest['reference_source_inventory_sha256'], 'Reference source provenance changed')
    require(len(manifest['reference_sha256']) == 12, 'Twelve fixed references required')
    for name, wanted in manifest['reference_sha256'].items(): require(sha(inside(reference,name)) == wanted, 'Reference hash differs: '+name)
    return manifest, dict(root_artifacts_verified=count, frozen_source_files_verified=len(frozen), source_manifest_sha256=sha(source/'manifest.json'), source_inventory_sha256=sha(source/'artifact_hashes.json'), source_archive_is_authoritative=True)


def verify_equivalence(path, n_source, n_manifest, g_manifest):
    folder=path.parent; manifest=read(folder/'manifest.json'); result=read(path)
    require(path.name=='result.json' and manifest['status']=='complete' and result['status']=='passed','Complete four-case N equivalence proof required')
    count=verify_inventory(folder)
    require(Path(manifest['source_root']).resolve()==n_source and manifest['source_manifest_sha256']==sha(n_source/'manifest.json') and manifest['source_inventory_sha256']==sha(n_source/'artifact_hashes.json'),'Equivalence refers to different historical N evidence')
    require(result['baseline_reuse_supported_for_this_fixed_N4_protocol'] is True and result['all_four_final_meshes_exact'] is True and result['all_four_terminations_exact'] is True and result['independent_process'] is True,'N reuse/mesh/termination certificate failed')
    require(result['saved_history_reexecutions']==4 and result['new_physical_trajectories']==0 and not result['fresh_sensor_simulation'] and not result['future_packet_supplied_to_planner'],'Equivalence scope differs')
    require(not result['semantic_information_test'] and not result['full_architecture_efficacy_proven'],'Unexpected equivalence efficacy claim')
    frozen=manifest['source_sha256']; require(read(folder/'source_sha256.json')==frozen,'Equivalence source hash declarations differ')
    with zipfile.ZipFile(folder/'sources.zip') as archive:
        require(set(archive.namelist())==set(frozen),'Equivalence source archive differs')
        for name,wanted in frozen.items():require(hashlib.sha256(archive.read(name)).hexdigest()==wanted,'Equivalence frozen source differs: '+name)
    mandatory=('nso/facility_runtime_v21.py','nso/coverage_budget_v21.py','configs/virtual3d/facility_joint_budget_v21_probe.json')
    for name in mandatory:
        require(name in frozen and frozen[name]==g_manifest['source_sha256'].get(name),'Equivalence/G frozen runtime, helper or config differs: '+name)
    for name,wanted in frozen.items():
        if name.startswith(('env/','nso/','utils/')):require(g_manifest['source_sha256'].get(name)==wanted,'Equivalence/G shared source differs: '+name)
    require(manifest['configuration']==g_manifest['config'],'Equivalence/G protocol configuration differs')
    require(len(result['cases'])==4,'Equivalence omitted a case')
    cases=[]
    for declared,summary in zip(n_manifest['cases'],result['cases']):
        i=declared['index']; receipt=read(folder/f'case_{i:02d}.json'); original_folder=n_source/f'case_{i:02d}'
        original=read(original_folder/'result.json'); provenance=receipt['provenance']; replay=read(original_folder/'verification.json')
        require(summary['case']==receipt['case']==declared and receipt['status']=='passed' and receipt['mode']=='N','Equivalence case identity/status differs')
        require(receipt['runtime_decisions_on_saved_history_exact'] and receipt['independent_process'] and not receipt['future_packet_supplied_to_planner'] and not receipt['fresh_sensor_simulation'] and not receipt['evaluator_executed'],'Equivalence case scope differs')
        require(receipt['paid_actions']==original['paid_actions'] and receipt['raw_packets_consumed']==original['paid_actions']+1,'Equivalence saved packet count differs')
        require(receipt['trajectory_sha256']==receipt['original_trajectory_sha256']==digest(original['actions']),'Equivalence trajectory differs')
        require(receipt['final_mesh_sha256']==original['final_mesh_sha256'] and receipt['termination']==original['termination'],'Equivalence mesh or termination differs')
        require(Path(provenance['raw_source_root']).resolve()==original_folder and provenance['source_case_inventory_sha256']==sha(original_folder/'artifact_hashes.json') and provenance['source_case_artifact_sha256']==read(original_folder/'artifact_hashes.json'),'Equivalence case input inventory differs')
        require(provenance['original_replay']==replay and provenance['original_physical_process_id']==replay['physical_process_id'],'Equivalence historical process receipts differ')
        require(provenance['saved_history_verifier_process_id']==manifest['process_id'] and manifest['process_id'] not in (replay['physical_process_id'],replay['replay_process_id']),'Equivalence must execute in a separate process')
        require(provenance['hidden_reference_arrays_loaded'] is False and provenance['reference_member_read']=='metadata only','Equivalence loaded hidden reference arrays')
        require(sha(Path(provenance['reference_metadata_file']))==provenance['reference_file_sha256'],'Equivalence configuration reference differs')
        sequence=[dict(action_id=0,packet_sha256=declared['initial_packet_sha256'])]+[dict(action_id=j,packet_sha256=x['packet_sha256']) for j,x in enumerate(original['actions'],1)]
        require(receipt['raw_packet_sequence_sha256']==digest(sequence),'Equivalence raw sensor sequence differs')
        require(receipt['pre_and_post_belief_receipts_sha256']==digest(receipt['action_state_receipts']),'Equivalence belief receipts changed')
        for key,value in summary.items():require(receipt[key]==value,'Equivalence summary/case differs: '+key)
        cases.append(dict(index=i,receipt_sha256=sha(folder/f'case_{i:02d}.json'),paid_actions=receipt['paid_actions'],decisions_compared=receipt['decisions_compared'],route_refreshes_compared=receipt['route_refreshes_compared'],paid_authorizations_compared=receipt['paid_authorizations_compared']))
    for key,field in [('saved_paid_actions_compared','paid_actions'),('decisions_compared','decisions_compared'),('route_refreshes_compared','route_refreshes_compared'),('paid_authorizations_compared','paid_authorizations_compared')]:
        require(result[key]==sum(c[field] for c in cases),'Equivalence aggregate count differs: '+key)
    return dict(status='passed',result_sha256=sha(path),manifest_sha256=sha(folder/'manifest.json'),inventory_sha256=sha(folder/'artifact_hashes.json'),artifacts_verified=count,cases=cases,
        G_frozen_runtime_helper_and_shared_sources_exact=True,matched_runtime_helper_config_sha256={name:frozen[name] for name in mandatory},
        metrics_recomputed=False,metric_reuse_basis='Identical saved sensors, V21-N decisions, mapping/mesh and termination; unchanged V19 evaluator/world/reference sources.',baseline_reuse_supported_for_this_fixed_N4_protocol=True)


def roi_steps(calls, trace, n):
    boots = [c for c in calls if c['method'] == 'coverage_bootstrap_v21']
    events = [c for c in calls if c['method'] == 'observe_unique_coverage_v21']
    require(len(boots) == 1 and boots[0]['action_id'] == 0, 'One action-zero ROI bootstrap required')
    require([c['action_id'] for c in events] == list(range(1,n+1)), 'ROI events missing or duplicated')
    boot = boots[0]['outputs']; task = boot['task_cells']; known = boot['known_cells']; target = math.ceil(.8*task)
    require(task > 0 and task == math.prod(boots[0]['inputs']['task_shape']), 'Expected fixed full public task ROI')
    require(boot['denominator_fixed'] and boot['coverage_proxy'] == 'fixed_task_roi_known_fraction' and not boot['navigation_used_for_budget'] and not boot['true_coverage_guaranteed'], 'ROI/nav boundary flags differ')
    require(boot['rate_discount'] == boot['gain_discount'] == .5 and boot['prefix_actions'] == 5 and boot['history_prefixes'] == 6, 'ROI fixed discounts/history differ')
    success = failure = 1; history = deque(maxlen=6); pending = None; rows = []
    def state(i, event=None):
        prefixes = list(history)
        if pending is not None and pending['coverage_intent']: prefixes = (prefixes+[pending])[-6:]
        cost = sum(x['paid_actions'] for x in prefixes)
        net = sum(x['actual_new_known']-x['known_lost'] for x in prefixes)
        return dict(action_id=i, coverage_2d=trace[i]['coverage_2d'], C_ROI=known/task,
            known_cells=known, task_cells=task, deficit_cells=max(0,target-known),
            conservative_rate=.5*max(0,net)/cost if cost else None, union_yield=success/(success+failure),
            rate_paid_actions=cost, rate_net_known_gain=net,
            coverage_intent=None if event is None else event['coverage_intent'])
    rows.append(state(0)); close(boot['C_plan'],rows[0]['C_ROI'],'Initial ROI fraction')
    require(boot['deficit_cells']==rows[0]['deficit_cells'], 'Initial ROI deficit')
    for call in events:
        x = call['outputs']; i = call['action_id']; require(x['action_id'] == i, 'Event action ID differs')
        require(x['known_cells_before'] == known, 'Known-cell chain broken')
        added, lost = x['actual_new_known_cells'], x['known_lost_cells']
        require(isinstance(added,int) and isinstance(lost,int) and 0 <= added <= task-known and 0 <= lost <= known, 'Invalid new/lost known count')
        require(x['net_known_gain_cells'] == added-lost and x['coverage_intent'] == call['inputs']['coverage_intent'], 'Known gain/intention mismatch')
        known += added-lost; require(x['known_cells_after'] == known and 0 <= known <= task, 'Known-cell after count differs')
        require(x['actual_new_known_counts_shared_cell_once'] and not x['camera_seen_only_credit'], 'Duplicate/camera-only coverage credit')
        if x['predicted_union_cells'] is not None:
            hit, miss = x['realized_predicted_cells'], x['unrealized_predicted_cells']
            require(0 <= hit <= added and miss >= 0 and hit+miss == x['predicted_union_cells'] and x['unpredicted_realized_cells'] == added-hit, 'Predicted/realized union counts differ')
            success += hit; failure += miss
        intent = x['coverage_intent']
        if pending is not None and pending['coverage_intent'] != intent:
            if pending['coverage_intent']: history.append(pending)
            pending = None
        if pending is None: pending = dict(first_action_id=i,last_action_id=i,coverage_intent=intent,paid_actions=0,actual_new_known=0,known_lost=0)
        pending.update(last_action_id=i,paid_actions=pending['paid_actions']+1,actual_new_known=pending['actual_new_known']+added,known_lost=pending['known_lost']+lost)
        if pending['paid_actions'] == 5:
            require(all(x['closed_prefix'][key]==value for key,value in pending.items()), 'Closed ROI prefix differs')
            if intent: history.append(pending)
            pending = None
        row=state(i,x); row.update(actual_new_known_cells=added,known_lost_cells=lost,net_known_gain_cells=added-lost)
        rows.append(row)
    return rows


def budget_check(budget, route, step, remaining):
    prediction=budget['prediction']; count=prediction['predicted_union_cells']
    require(isinstance(count,int) and 0<=count<=step['task_cells']-step['known_cells'], 'Invalid unknown proposal count')
    close(prediction['union_yield_mean'],step['union_yield'],'Candidate union yield')
    require(prediction['gain_discount']==.5 and not prediction['navigation_used_for_prediction'] and not prediction['future_free_assumption_used'], 'Prediction units/navigation assumption changed')
    gain=count*step['union_yield']*.5; close(prediction['discounted_known_gain_cells'],gain,'Discounted known gain')
    close(budget['predicted_gain_cells'],gain,'Budget known gain'); close(budget['C_plan'],step['C_ROI'],'Candidate ROI')
    close(budget['conservative_rate'],step['conservative_rate'],'Candidate known rate')
    require(budget['deficit_cells']==step['deficit_cells'] and budget['remaining_budget']==remaining, 'Candidate deficit/budget differs')
    require(budget['outbound_cost']==route['outbound_cost'] and budget['return_cost']==route['return_cost'] and budget['reserve_actions']==5,'Candidate route costs differ')
    residual=max(0.,step['deficit_cells']-gain); rate=step['conservative_rate']
    reserve=0 if residual==0 else math.ceil(residual/rate) if rate is not None and rate>0 else None
    needed=route['outbound_cost']+route['return_cost']+5+reserve if reserve is not None else None
    reason=('route_and_return_exceed_budget' if route['outbound_cost']+route['return_cost']>remaining else 'coverage_rate_unavailable' if reserve is None else 'insufficient_predicted_coverage_reserve' if needed>remaining else 'model_budget_passed')
    require(budget['coverage_reserve_actions']==reserve and budget['total_required_actions']==needed and budget['reason']==reason and budget['allowed']==(reason=='model_budget_passed'),'Budget reserve formula differs')
    require(not budget['movement_authorized'] and not budget['true_coverage_guaranteed'] and not budget['evaluation_truth_used'] and not budget['class_used'],'Budget truth/permission claim')


def candidate_facts(output, action_id):
    routes=output['candidates']; audit=output['score_audit']; scores=output['scores']['G']
    require(len(routes)==len(audit)==len(scores),'Candidate/score cardinalities differ')
    require([r['candidate_id'] for r in routes]==[r['candidate_id'] for r in audit]==list(range(len(routes))), 'Candidate IDs differ')
    result=[]
    for i,(route,row,score) in enumerate(zip(routes,audit,scores)):
        require('v19_task_proxy' in row,'G candidate lacks reconstruction-score audit'); terms=row['v19_task_proxy']; quality=finite(terms.get('observed_direction_term',0.),'direction term')+finite(terms.get('observed_precision_term',0.),'precision term')
        budget=row['v21_coverage']['coverage_budget']; positive=finite(score,'G score')>0
        result.append(dict(action_id=action_id,candidate_id=i,group=route['group'],budget_allowed=budget['allowed'],
            positive_quality_term=quality>0,quality_term=quality,noncoverage_role=not route['group'].startswith('coverage_'),
            positive_total_G_score=positive,total_G_score=score,direction_term=terms.get('observed_direction_term',0.),precision_term=terms.get('observed_precision_term',0.),runtime_admitted=budget['allowed'] and positive))
    require(output['quality_budget_admitted_candidates']==[r['candidate_id'] for r in result if r['runtime_admitted']], 'Named admission list differs from actual budget+total-score rule')
    return result


def paid_kind(authorization, option):
    if authorization['phase']=='return': return 'return'
    if authorization['phase']!='outbound' or option is None: return 'unattributed'
    return 'coverage_intent' if option['v21_coverage_intent'] else 'quality_intent'


def analyze_g_case(folder, case, config):
    result=read(folder/'result.json'); calls=read(folder/'module_calls.json.gz'); runtime=read(folder/'runtime_audit.json.gz')
    replay=read(folder/'verification.json'); timing=read(folder/'timing.json'); base=identity(case); n=result['paid_actions']
    require(result['case']==case,'G result/declared case differs')
    require(digest([{k:v for k,v in c.items() if k!='elapsed_s'} for c in calls])==result['module_calls_sha256'] and digest(runtime)==result['runtime_audit_sha256'],'Module/runtime log digest differs')
    require([c['call_id'] for c in calls]==list(range(1,len(calls)+1)),'Module call IDs differ')
    for c in calls:
        require(digest(c['inputs'])==c['input_sha256'] and digest(c['outputs'])==c['output_sha256'],'Module I/O digest differs')
        require(not c['truth_used'] and not c['trained'],'Unexpected module truth/training')
    require({c['module'] for c in calls}=={'OV-SDF','STGHP','RPN-UQ','IGCR'},'Four module interfaces missing')
    require(len(result['actions'])==n==replay['actions']==result['termination']['final_action_id'],'G physical/replay action counts differ')
    require(replay['physical_process_id']==timing['process_id'],'G process receipt differs')
    require(result['primitive_budget_compliant']==(n<=case['budget']),'Primitive budget flag differs')
    require(sorted(p.name for p in (folder/'packets').glob('*.npz'))==[f'{i:04d}.npz' for i in range(n+1)],'Raw packet inventory is not contiguous')
    import numpy as np
    with np.load(folder/'final_mesh.npz',allow_pickle=False) as mesh:
        require(set(mesh.files)==set(result['final_mesh_sha256']),'Mesh fields differ')
        for name,wanted in result['final_mesh_sha256'].items():
            arr=np.ascontiguousarray(mesh[name]); require(hashlib.sha256(f'{arr.dtype.str}:{arr.shape}:'.encode()+arr.tobytes()).hexdigest()==wanted,'Final mesh array differs')
    trace=result['coverage_trace']; require([r['action_id'] for r in trace]==list(range(n+1)),'Every paid action needs true coverage')
    steps=roi_steps(calls,trace,n)
    options={}; selections=[]; candidates=[]; refreshes=[]; counts=Counter(); selection_by_call={}
    for call in calls:
        out=call['outputs']; action=call['action_id']; method=call['method']
        if method=='select_topo_target':
            require(out['mode']=='G','G selection mode differs'); state=out['coverage_state']; step=steps[action]
            for key in ('known_cells','task_cells','deficit_cells'): require(state[key]==step[key],'ROI snapshot count differs: '+key)
            close(state['C_plan'],step['C_ROI'],'ROI selection progress'); close(state['conservative_rate'],step['conservative_rate'],'ROI selection rate'); close(state['union_yield'],step['union_yield'],'ROI union yield')
            rows=candidate_facts(out,action); admitted=[r for r in rows if r['runtime_admitted']]
            for row,route,audit_row in zip(rows,out['candidates'],out['score_audit']):
                budget=audit_row['v21_coverage']['coverage_budget']; budget_check(budget,route,step,case['budget']-action)
                require(audit_row['v21_coverage']['unique_predicted_unknown_cells']==budget['prediction']['predicted_union_cells'],'ROI route mask count differs')
                row.update(**base,selection_call_id=call['call_id'],outbound_cost=route['outbound_cost'],return_cost=route['return_cost'],coverage_reserve_actions=budget['coverage_reserve_actions'],budget_reason=budget['reason'],budget_slack=None if budget['total_required_actions'] is None else budget['remaining_budget']-budget['total_required_actions'],deficit_cells=budget['deficit_cells'],conservative_rate=budget['conservative_rate'],predicted_unknown_cells=budget['prediction']['predicted_union_cells'],discounted_known_gain_cells=budget['predicted_gain_cells'],C_ROI=step['C_ROI'])
                candidates.append(row)
            counts['decisions']+=1; counts['candidate_occurrences']+=len(rows)
            for key,predicate in [('budget_allowed',lambda r:r['budget_allowed']),('positive_quality',lambda r:r['positive_quality_term']),('budget_allowed_positive_quality',lambda r:r['budget_allowed'] and r['positive_quality_term']),('budget_allowed_noncoverage',lambda r:r['budget_allowed'] and r['noncoverage_role']),('runtime_admitted',lambda r:r['runtime_admitted'])]:
                total=sum(predicate(row) for row in rows); counts[key+'_candidate_occurrences']+=total; counts[key+'_decisions']+=int(total>0)
            require(out['coverage_pressure']==(not admitted),'G pressure must follow the full admission rule')
            require(out['effective_objective']==('N' if not admitted else 'G'),'Effective objective differs')
            selected=out['selected']; chosen=None
            if selected is not None:
                chosen=rows[selected['candidate_id']]; oid=selected['option_id']; require(oid not in options,'Option ID reused')
                intent=not admitted or selected['group'].startswith('coverage_'); require(selected['v21_coverage_intent']==intent,'Selected option intent differs')
                require(selected['selection_call_id']==call['call_id'],'Selected attribution differs')
                options[oid]=dict(option=selected,selected_action=action,selected_positive_quality_term=chosen['positive_quality_term'],paid_action_ids=[],arrival_action_ids=[],refresh_cancel_action_ids=[],safety_denial_action_ids=[])
                counts['selected_noncoverage_decisions']+=int(chosen['noncoverage_role']); counts['selected_quality_intent_decisions']+=int(not intent); counts['selected_positive_quality_decisions']+=int(chosen['positive_quality_term'])
            selections.append({**base,'action_id':action,'selection_call_id':call['call_id'],'option_id':None if selected is None else selected['option_id'],'group':None if selected is None else selected['group'],'quality_intent':None if selected is None else not selected['v21_coverage_intent'],'positive_quality_term':None if chosen is None else chosen['positive_quality_term'],'selected_quality_term':None if chosen is None else chosen['quality_term'],'selected_total_G_score':None if chosen is None else chosen['total_G_score'],'C_ROI':step['C_ROI'],'coverage_2d':step['coverage_2d'],'candidate_count':len(rows),'budget_allowed_candidates':sum(r['budget_allowed'] for r in rows),'runtime_admitted_candidates':len(admitted),'effective_objective':out['effective_objective']})
            selection_by_call[call['call_id']]=selected
        elif method=='refresh_observed_route_v21':
            oid=out['option_id']; require(oid in options,'Route refresh has unknown option')
            require(out['old_target']==options[oid]['option']['pose'],'Refreshed target changed without new selection')
            require(out['target_retained']==(out['new_remaining_outbound'] is not None),'Refresh status/cost mismatch')
            if out['coverage_budget'] is not None:
                b=out['coverage_budget']; budget_check(b,dict(outbound_cost=b['outbound_cost'],return_cost=b['return_cost']),steps[action],case['budget']-action)
            if not out['target_retained']: options[oid]['refresh_cancel_action_ids'].append(action)
            refreshes.append({**base,**out,'quality_intent':not options[oid]['option']['v21_coverage_intent'],'causal_quality_gain':None})
        elif method=='assess_local_action' and not out['allowed']:
            oid=call.get('executed_option_id')
            if oid in options: options[oid]['safety_denial_action_ids'].append(action)
            else: counts['unattributed_or_return_safety_denials']+=1
    require([{k:v for k,v in x.items() if k!='event'} for x in runtime if x['event']=='v21_route_update']==[x['outputs'] for x in calls if x['method']=='refresh_observed_route_v21'],'Runtime/module refresh records differ')
    choices=[x['option'] for x in runtime if x['event']=='global_choice']; require(choices==list(selection_by_call.values()),'Installed global choices differ')
    auth=[x for x in runtime if x['event']=='paid_action_authorized']; obs=[x for x in runtime if x['event']=='observation']
    ledger_obs=[x for x in calls if x['method']=='observe_unique_coverage_v21']
    require([x['next_action_id'] for x in auth]==[x['action_id'] for x in obs]==list(range(1,n+1)),'Paid authorizations/observations missing')
    for i,(a,o,event,taken) in enumerate(zip(auth,obs,ledger_obs,result['actions']),1):
        check=a['assessment']; require(check['allowed'] and check['action']==taken['action'] and check['remaining_budget']==case['budget']-i+1,'Paid authorization differs')
        require(o['packet_sha256']==taken['packet_sha256'] and o['collision']==taken['collision'],'Raw observation receipt differs')
        if not taken['collision']: require(check['next_pose']==taken['pose'],'Measured/authorized pose differs')
        require(check['reserved_return_cost'] is not None and check['reserved_return_cost']<=case['budget']-i,'Paid action violates logged return reserve')
        oid=a['option_id']; item=options.get(oid); option=None if item is None else item['option']; kind=paid_kind(a,option)
        require(oid is None or item is not None,'Paid action refers to unknown selected option')
        require(event['executed_option_id']==oid,'Paid action/coverage-event option differs')
        expected_intent=bool(option and option['v21_coverage_intent']); require(event['outputs']['coverage_intent']==expected_intent,'Paid action/coverage intent differs')
        counts[kind+'_paid_actions']+=1
        if item is not None: item['paid_action_ids'].append(i)
        if o['arrived']:
            require(item is not None and a['phase']=='outbound' and not taken['collision'] and taken['pose']==option['pose'],'Measured target arrival inconsistent')
            item['arrival_action_ids'].append(i)
        steps[i].update(action=taken['action'],phase=a['phase'],paid_kind=kind,option_id=oid,arrived=o['arrived'],collision=taken['collision'])
    require(sum(int(a['collision']) for a in result['actions'])==result['collisions'],'Collision sum differs')
    outcomes=[]
    for oid,item in options.items():
        option=item['option']; arrived=bool(item['arrival_action_ids']); cancelled=bool(item['refresh_cancel_action_ids'] or item['safety_denial_action_ids']); quality_intent=not option['v21_coverage_intent']
        status='arrived' if arrived else 'explicit_cancel_or_safety_denial' if cancelled else 'unreached_without_unambiguous_cancel_reason'
        for prefix,condition in [('all',True),('quality_intent',quality_intent),('noncoverage_role',not option['group'].startswith('coverage_'))]:
            if condition:
                counts[prefix+'_selected_options']+=1; counts[prefix+'_arrived_options']+=int(arrived); counts[prefix+'_arrival_events']+=len(item['arrival_action_ids']); counts[prefix+'_refresh_cancelled_options']+=int(bool(item['refresh_cancel_action_ids'])); counts[prefix+'_safety_denied_options']+=int(bool(item['safety_denial_action_ids'])); counts[prefix+'_explicit_cancel_or_denial_options']+=int(cancelled); counts[prefix+'_unreached_unclassified_options']+=int(not arrived and not cancelled)
        outcomes.append({**base,'option_id':oid,'group':option['group'],'quality_intent':quality_intent,'selected_action':item['selected_action'],'status':status,'target_pose':option['pose'],'paid_actions':len(item['paid_action_ids']),**{k:item[k] for k in ('paid_action_ids','arrival_action_ids','refresh_cancel_action_ids','safety_denial_action_ids')},'causal_surface_quality_gain':None,'quality_attribution_note':'No single-option attribution from shared reconstruction checkpoints.'})
    for key in ('decisions','candidate_occurrences','selected_noncoverage_decisions','selected_quality_intent_decisions','selected_positive_quality_decisions','coverage_intent_paid_actions','quality_intent_paid_actions','return_paid_actions','unattributed_paid_actions','unattributed_or_return_safety_denials'):
        counts.setdefault(key,0)
    for prefix in ('all','quality_intent','noncoverage_role'):
        for suffix in ('selected_options','arrived_options','arrival_events','refresh_cancelled_options','safety_denied_options','explicit_cancel_or_denial_options','unreached_unclassified_options'):counts.setdefault(prefix+'_'+suffix,0)
    counts['route_refresh_events']=len(refreshes)
    counts['route_refresh_cancel_events']=sum(not x['target_retained'] for x in refreshes)
    counts['route_refresh_retained_events']=sum(x['target_retained'] for x in refreshes)
    require(sum(counts[k+'_paid_actions'] for k in ('coverage_intent','quality_intent','return','unattributed'))==n,'Paid phase attribution lost or duplicated an action')
    quality=[]; endpoints=[]
    require([x['action_id'] for x in result['quality_curve']]==sorted(set(range(0,n+1,20))|{n}),'Quality checkpoints differ')
    for x in result['quality_curve']:
        metric=x['metrics']; metric_check(metric); i=x['action_id']; close(metric['coverage_2d'],trace[i]['coverage_2d'],'Checkpoint true coverage')
        quality.append({**base,'action_id':i,'reference_seed':2026,'coverage_2d':metric['coverage_2d'],'external_macro_f1':metric['05cm']['external_macro_f1'],'joint_external':metric['05cm']['joint_external'],'eligible':metric['eligible'],'returned':metric['returned'],'failed':metric['failed'],'collisions':metric['collisions']})
    require(result['quality_curve'][0]['metrics']==result['before']['2026'] and result['quality_curve'][-1]['metrics']==result['after']['2026'],'Quality endpoint mismatch')
    require(set(result['before'])==set(result['after'])=={'2026','2027','2028'},'Three reference endpoints required')
    for seed in (2026,2027,2028):
        metric=result['after'][str(seed)]; metric_check(metric); metric_check(result['before'][str(seed)])
        close(metric['coverage_2d'],trace[-1]['coverage_2d'],'Reference endpoint coverage')
        require(metric['returned']==result['termination']['returned_to_anchor'] and metric['failed']==result['termination']['failed'] and metric['collisions']==result['collisions'],'Endpoint failure/return differs')
        endpoints.append({**base,'reference_seed':seed,'paid_actions':n,'coverage_2d':metric['coverage_2d'],'external_macro_f1':metric['05cm']['external_macro_f1'],'joint_external':metric['05cm']['joint_external'],'eligible':metric['eligible'],'returned':metric['returned'],'failed':metric['failed'],'collisions':metric['collisions'],'missing_asset_count':metric['missing_asset_count']})
    first=next((x['action_id'] for x in trace if x['coverage_2d']>=.8),None); require(first==result['first_evaluator_coverage_gate_action'],'First true coverage gate differs')
    summary={**endpoints[0],'C_ROI_final':steps[-1]['C_ROI'],'first_evaluator_coverage_gate_action':first,'termination':result['termination'],'primitive_budget_compliant':result['primitive_budget_compliant'],'behavior_counts':dict(counts),'runtime_safety_denials':sum(x['event']=='v14_route_denied' for x in runtime),'timing':timing,'independent_replay':replay,'raw_mask_reconstruction_performed':False,'ROI_progress_checked_from_logged_known_gains_and_losses':True,'individual_option_quality_improvement':None}
    for row in steps: row.update(base)
    return dict(summary=summary,steps=steps,quality=quality,endpoints=endpoints,selections=selections,candidates=candidates,refreshes=refreshes,option_outcomes=outcomes)


def paired_rows(n_data,g_data):
    result=[]
    for n,g in zip(n_data,g_data):
        require((n['summary']['parent'],n['summary']['assignment'])==(g['summary']['parent'],g['summary']['assignment']),'Pair identity differs')
        for a,b in zip(n['endpoints'],g['endpoints']):
            require(a['reference_seed']==b['reference_seed'],'Paired reference seed differs')
            row=dict(index=b['index'],parent=b['parent'],assignment=b['assignment'],reference_seed=b['reference_seed'],N_eligible=a['eligible'],G_eligible=b['eligible'],both_eligible=bool(a['eligible'] and b['eligible']),N_paid_actions=a['paid_actions'],G_paid_actions=b['paid_actions'])
            for key in ('coverage_2d','external_macro_f1','joint_external'):
                row.update({f'N_{key}':a[key],f'G_{key}':b[key],f'G_minus_N_{key}':b[key]-a[key],f'G_relative_to_N_{key}':(b[key]-a[key])/a[key] if a[key]>0 else None})
            row.update(N_returned=n['summary']['termination']['returned_to_anchor'],G_returned=g['summary']['termination']['returned_to_anchor'],N_failed=n['summary']['termination']['failed'],G_failed=g['summary']['termination']['failed'],N_collisions=n['summary']['collisions'],G_collisions=g['summary']['collisions'],qualified_improvement_comparison_available=row['both_eligible'])
            result.append(row)
    return result


def make_plots(output,n_data,g_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ('coverage','exterior_quality'):
        fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
        for ax,n,g in zip(axes.flat,n_data,g_data):
            for dataset,label,color in ((n,'N','#2769a6'),(g,'G','#be592d')):
                rows=dataset['steps' if kind=='coverage' else 'quality']
                key='coverage_2d' if kind=='coverage' else 'external_macro_f1'
                ax.plot([r['action_id'] for r in rows],[r[key] for r in rows],label=label+' '+('true C2D' if kind=='coverage' else 'Q external F1@5cm'),color=color,marker='o' if kind!='coverage' else None,markersize=3,linewidth=1.6)
            if kind=='coverage':
                ax.plot([r['action_id'] for r in g['steps']],[r['C_ROI'] for r in g['steps']],':',color='#a28b63',label='G C_ROI heuristic',linewidth=1.2)
                ax.axhline(.8,color='#777777',linestyle='--',linewidth=.7,label='True coverage gate 0.80')
            s=g['summary']; ax.set_title(s['parent']+' | '+('A complex / B simple' if s['assignment']=='A_complex_B_simple' else 'A simple / B complex'))
            ax.text(.02,.97,f"N eligible={n['summary']['eligible']}; G eligible={s['eligible']}",transform=ax.transAxes,va='top',fontsize=8)
            ax.axvline(s['budget'],color='#999999',linestyle=':',linewidth=.7); ax.set_xlim(0,188); ax.set_ylim(0,1); ax.grid(alpha=.2)
            ax.set_xlabel('Paid primitive actions'); ax.set_ylabel('Coverage fraction' if kind=='coverage' else 'External F1 fraction')
        handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',ncol=len(labels),frameon=False,bbox_to_anchor=(.5,.97))
        fig.suptitle('V21 G versus equivalence-certified N: '+kind.replace('_',' '))
        fig.text(.5,.015,'All four pairs retained. '+('C_ROI is not true C2D or a certified bound.' if kind=='coverage' else 'Recorded checkpoints only; connecting lines are guides; primary seed 2026.'),ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.90)); fig.savefig(output/(kind+'_curves.png'),dpi=150,bbox_inches='tight'); plt.close(fig)


def self_test():
    # Synthetic behavior checks only; no evidence tree or report is created.
    require(paid_kind({'phase':'return'},None)=='return','Return must not be called quality')
    require(paid_kind({'phase':'outbound'},{'v21_coverage_intent':False})=='quality_intent','Quality intent attribution')
    output=dict(candidates=[dict(candidate_id=i,group=g) for i,g in enumerate(['coverage_0','asset_0_corner','asset_1_corner'])],
        score_audit=[dict(candidate_id=i,v19_task_proxy=dict(observed_direction_term=q,observed_precision_term=0.),v21_coverage=dict(coverage_budget=dict(allowed=True))) for i,q in enumerate([0.,.2,.2])],scores={'G':[.1,0.,.2]},quality_budget_admitted_candidates=[0,2])
    facts=candidate_facts(output,0)
    require(sum(x['budget_allowed'] for x in facts)==3 and sum(x['runtime_admitted'] for x in facts)==2 and sum(x['runtime_admitted'] and x['positive_quality_term'] for x in facts)==1,'Budget admission and quality terms conflated')
    a=dict(index=0,parent='P',assignment='A',reference_seed=2026,eligible=True,paid_actions=5,coverage_2d=.8,external_macro_f1=.5,joint_external=.4)
    b={**a,'eligible':False,'coverage_2d':.7,'joint_external':.35}
    def pack(x):return dict(summary=dict(parent='P',assignment='A',termination=dict(returned_to_anchor=True,failed=False),collisions=0),endpoints=[x])
    pair=paired_rows([pack(a)],[pack(b)])[0]
    require(not pair['both_eligible'] and pair['G_minus_N_joint_external']<0,'Failed/negative pair must remain')
    metric=dict(coverage_2d=.9,returned=True,collisions=0,failed=False,eligible=True,mission_asset_count=6,missing_asset_count=6,
        instances=[dict(id=i,missing=True,**{t:dict(precision=0.,recall=0.,f1=0.) for t in ('02cm','05cm','10cm')}) for i in range(6)],
        **{t:dict(external_macro_f1=0.,asset_macro_f1=0.,joint_external=0.,joint_asset=0.) for t in ('02cm','05cm','10cm')})
    metric_check(metric)
    bad=deepcopy(metric);bad['failed']=True
    try:metric_check(bad)
    except ValueError:pass
    else:raise AssertionError('Failed endpoint with eligible=true must be rejected')
    boot=dict(task_cells=16,known_cells=4,C_plan=.25,deficit_cells=9,denominator_fixed=True,coverage_proxy='fixed_task_roi_known_fraction',navigation_used_for_budget=False,true_coverage_guaranteed=False,rate_discount=.5,gain_discount=.5,prefix_actions=5,history_prefixes=6)
    events=[dict(action_id=1,known_cells_before=4,known_cells_after=6,actual_new_known_cells=3,known_lost_cells=1,net_known_gain_cells=2,coverage_intent=True,actual_new_known_counts_shared_cell_once=True,camera_seen_only_credit=False,predicted_union_cells=4,realized_predicted_cells=3,unrealized_predicted_cells=1,unpredicted_realized_cells=0),dict(action_id=2,known_cells_before=6,known_cells_after=4,actual_new_known_cells=0,known_lost_cells=2,net_known_gain_cells=-2,coverage_intent=True,actual_new_known_counts_shared_cell_once=True,camera_seen_only_credit=False,predicted_union_cells=None)]
    calls=[dict(method='coverage_bootstrap_v21',action_id=0,inputs=dict(task_shape=[4,4]),outputs=boot)]+[dict(method='observe_unique_coverage_v21',action_id=x['action_id'],inputs=dict(coverage_intent=True),outputs=x) for x in events]
    trace=[dict(action_id=i,coverage_2d=v) for i,v in enumerate((.2,.3,.2))]
    rows=roi_steps(calls,trace,2)
    require(rows[-1]['known_cells']==4 and rows[-1]['conservative_rate']==0. and rows[-1]['C_ROI']==.25,'ROI lost knowledge must cancel earlier gain')
    bad=deepcopy(calls);bad[-1]['outputs']['known_cells_after']=5
    try:roi_steps(bad,trace,2)
    except ValueError:pass
    else:raise AssertionError('Broken known-cell chain must be rejected')
    print('Synthetic behavior checks passed; no experiment or artifacts created.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g-source',type=Path,default=DEFAULT_G)
    parser.add_argument('--n-source',type=Path,default=DEFAULT_N)
    parser.add_argument('--n-equivalence',type=Path,default=DEFAULT_EQ)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--self-test',action='store_true',help='Small synthetic behavior checks only; no evidence reads/writes')
    args=parser.parse_args()
    if args.self_test:self_test();return
    g_source=args.g_source.resolve();n_source=args.n_source.resolve();eq=args.n_equivalence.resolve();output=args.output.resolve()
    for source in (g_source,n_source,eq.parent):require(not output.is_relative_to(source) and not source.is_relative_to(output),'Analysis output must be separate from evidence')
    source_paths=[Path(__file__).resolve(),Path(old.__file__).resolve()];analysis_sources={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    gm,g_checks=verify_g_source(g_source);nm,n_checks=old.verify_source(n_source)
    equivalence=verify_equivalence(eq,n_source,nm,gm)
    require(gm['reference_sha256']==nm['reference_sha256'],'G/N reference caches differ')
    for key in ('parents','assignments','budgets','task_asset_count','sensor_model','noise_seed','replan_interval_actions','coverage_candidate_slots','coverage_minimum','reference_seeds','primary_reference_seed'):
        require(gm['config'][key]==nm['config'][key],'Shared G/N contract differs: '+key)
    for name,wanted in nm['source_sha256'].items():
        if name.startswith(('env/','nso/','utils/')):require(gm['source_sha256'].get(name)==wanted,'Shared source changed without a new N baseline: '+name)
    n_data=[old.analyze_case(n_source/f'case_{c["index"]:02d}',c,nm['config']) for c in nm['cases']]
    g_data=[analyze_g_case(g_source/f'case_{c["index"]:02d}',c,gm['config']) for c in gm['cases']]
    for c,n,g in zip(nm['cases'],n_data,g_data):
        require(c['initial_packet_sha256']==gm['cases'][c['index']]['initial_packet_sha256'],'G/N measured starting packet differs')
        raw=read(n_source/f'case_{c["index"]:02d}/result.json')
        for metrics in list(raw['before'].values())+list(raw['after'].values())+[x['metrics'] for x in raw['quality_curve']]:metric_check(metrics)
    for root,analyzed in ((n_source,n_data),(g_source,g_data)):
        agg=read(root/'result.json'); require(agg['status']=='complete' and agg['unique_physical_trajectories']==4 and agg['independent_process_replays']==4,'Four complete aggregate cases required')
        require(len(agg['cases'])==4,'Aggregate omitted a case')
        for recorded,case in zip(agg['cases'],analyzed):
            for key in ('index','parent','assignment','mode','budget','paid_actions','coverage_2d','external_macro_f1','joint_external','eligible','collisions','primitive_budget_compliant','first_evaluator_coverage_gate_action'):
                require(recorded[key]==case['summary'][key],'Aggregate/case differs: '+key)
    pairs=paired_rows(n_data,g_data);primary=[p for p in pairs if p['reference_seed']==2026]
    summary=dict(status='complete',G_root=str(g_source),N_root=str(n_source),N_equivalence=equivalence,
        checks=dict(G=g_checks,N=n_checks),paired_configurations=4,reference_sensitivities_per_pair=3,
        new_G_physical_trajectories=4,new_G_independent_replays=4,reused_N_physical_trajectories=4,reused_N_independent_replays=4,
        G_eligible_count=sum(g['summary']['eligible'] for g in g_data),N_eligible_count=sum(n['summary']['eligible'] for n in n_data),
        all_cases_retained=True,primary_paired_endpoints=primary,G_cases=[g['summary'] for g in g_data],N_cases=[n['summary'] for n in n_data],
        semantic_information_test=False,semantic_efficacy_proven=False,full_architecture_efficacy_proven=False,
        joint_planning_efficacy_proven=False,reference_sampling_repeats_are_independent_experiments=False,
        physical_experiments_started_by_analyzer=False,interpretation=NOTES)
    require(all(sha(ROOT/name)==wanted for name,wanted in analysis_sources.items()),'Analysis dependency changed while running')
    compact={k:summary[k] for k in ('status','paired_configurations','G_eligible_count','N_eligible_count')}
    if args.verify_only:print(json.dumps({**compact,'artifacts_written':False},indent=2));return
    output.mkdir(parents=True,exist_ok=False)
    try:
        write_csv(output/'paired_endpoints.csv',pairs)
        write_csv(output/'coverage_curves.csv',[{**row,'C_ROI':row.get('C_ROI'),'old_navigation_C_plan':row.get('C_plan') if row['mode']=='N' else None} for data in (n_data,g_data) for c in data for row in c['steps']])
        write_csv(output/'quality_checkpoints.csv',[row for data in (n_data,g_data) for c in data for row in c['quality']])
        for filename,key in [('G_decisions','selections'),('G_candidate_admissions','candidates'),('G_route_updates','refreshes'),('G_option_outcomes','option_outcomes')]:write_csv(output/(filename+'.csv'),[row for c in g_data for row in c[key]])
        make_plots(output,n_data,g_data)
        with zipfile.ZipFile(output/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for path in source_paths:archive.write(path,str(path.relative_to(ROOT)))
        atomic_json(output/'source_sha256.json',analysis_sources);atomic_json(output/'result.json',summary)
        atomic_json(output/'input_provenance.json',dict(G_manifest_sha256=sha(g_source/'manifest.json'),G_inventory_sha256=sha(g_source/'artifact_hashes.json'),N_manifest_sha256=sha(n_source/'manifest.json'),N_inventory_sha256=sha(n_source/'artifact_hashes.json'),N_equivalence_result_sha256=sha(eq),equivalence=equivalence))
        atomic_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
        verify_inventory(output)
    except Exception as error:
        atomic_json(output/'analysis_failure.json',dict(status='failed',error=repr(error)));raise
    print(json.dumps({**compact,'output':str(output)},indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError) as error:
        print('Analysis refused: '+str(error),file=sys.stderr);raise SystemExit(2)
