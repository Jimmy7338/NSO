#!/usr/bin/env python3
"""Independent whole-floor evaluation of sealed native runs and common action clocks."""

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import math
import os
from pathlib import Path
import statistics

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from regression import PRIOR as WEIGHTED, import_file, read, sha


ROOT = Path(__file__).resolve().parent
WEIGHTED_METRIC = WEIGHTED/'measurement/evaluate.py'
REGRESSION = ROOT/'regression/verified.json'
ARCHIVED_FREEZE = WEIGHTED/'holdout_freeze.json'
METRICS = ('C_free','F1_scene_05cm','J')
MOTION = ('A95_paid','repeat_distance_m','first_new_room_paid','path_length_m','target_arrival_rate',
          'elapsed_s','decision_wall_p50_s','decision_wall_p95_s','dispatched_plan_wall_p50_s',
          'dispatched_plan_wall_p95_s','returned','collisions')
FACTORIAL = {
    'joint_vs_original': {'region_S':1,'original_G':-1},
    'joint_vs_frontier': {'region_S':1,'F_strict':-1},
    'global_given_G': {'region_G':1,'original_G':-1},
    'global_given_S': {'region_S':1,'original_S':-1},
    'local_given_original': {'original_S':1,'original_G':-1},
    'local_given_regional': {'region_S':1,'region_G':-1},
    'global_main': {'region_G':.5,'original_G':-.5,'region_S':.5,'original_S':-.5},
    'local_main': {'region_S':.5,'region_G':-.5,'original_S':.5,'original_G':-.5},
    'interaction': {'region_S':1,'region_G':-1,'original_S':-1,'original_G':1},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pose_at(trace, start, paid):
    return start if paid==0 else [*trace[paid-1]['position'],trace[paid-1]['heading']]


def frame_paid(row):
    value = row.get('paid_before',row.get('paid',row.get('frame_id')))
    if isinstance(value,str):
        value = value.removeprefix('frame-')
    return int(value) if value is not None else None


def percentile(values, value):
    return float(np.percentile(values,value)) if values else None


def room_map(world, layout, shape, resolution):
    spec=world.SPECS[layout]
    x=((np.arange(shape[1])+.5)*resolution)[None,:]
    y=((shape[0]-.5-np.arange(shape[0]))*resolution)[:,None]
    labels=np.full(shape,'',dtype='<U30')
    if 'rooms' in spec:
        for room, (x0,y0,x1,y1) in spec['rooms'].items():
            labels[(x>=x0)&(x<x1)&(y>=y0)&(y<y1)]=room
    else:
        east,north=x>=spec['split_x'],y>=spec['split_y']
        for room,region in [('southwest',~east&~north),('northwest',~east&north),
                            ('southeast',east&~north),('northeast',east&north)]:
            labels[region]=room
    require(np.all(labels!=''),f'Truth room labels incomplete: {layout}')
    return labels


def executed_motion(trace, initial, truth_rooms, resolution):
    prior=tuple(initial['anchor'][:2]); visited={str(truth_rooms[prior])}; repeated, repeated_return, length=0,0,0
    crossed=[]; edges=set()
    for step in trace:
        current=tuple(step['position'])
        if current==prior:
            continue
        distance=sum(abs(a-b) for a,b in zip(prior,current,strict=True))
        require(distance==1 and step['action']=='forward' and not step['collision'],
                f'Trace jump or conflicting motion at {step["step"]}')
        edge=tuple(sorted((prior,current)))
        if edge in edges:
            repeated+=1
            repeated_return+=step['phase_before_action']=='return'
        edges.add(edge); length+=1
        last_room,new_room=str(truth_rooms[prior]),str(truth_rooms[current])
        if last_room!=new_room:
            crossed.append({'paid':step['step'],'from_room':last_room,'to_room':new_room,'new_room':new_room not in visited,
                            'first_visit':new_room not in visited,'post_pose':[*current,step['heading']],
                            'active_option_id':step['active_option_id'],'phase':step['phase_before_action']})
            visited.add(new_room)
        prior=current
    return {'actual_translated_edges':length,'repeat_edge_count':repeated,'repeat_distance_m':repeated*resolution,
            'repeat_return_distance_m':repeated_return*resolution,'unique_translated_edges':len(edges),
            'first_new_room_paid':next((event['paid'] for event in crossed if event['new_room']),None),
            'rooms_visited':len(visited),'truth_room_crossings':len(crossed)},crossed


def coverage_curve(directory, reference, trace):
    mandatory=directory/'observed_belief_trace.npz'
    with np.load(mandatory,allow_pickle=False) as value:
        paid,beliefs=value['paid'],value['belief']
    reachable=reference['reachable']
    require(paid.tolist()==list(range(len(trace)+1)) and beliefs.shape==(len(trace)+1,*reachable.shape),
            f'No contiguous original occupied/free estimate at every paid clock: {directory}')
    denominator=int(reachable.sum())
    require(denominator>0,f'Invalid reachable denominator for {directory}')
    counts=np.count_nonzero((beliefs==0)&reachable[None,:,:],axis=(1,2))
    records=[{'paid':n,'C_free':int(correct)/denominator,'correct_free_cells':int(correct),'reachable_cells':denominator}
             for n,correct in enumerate(counts)]
    first=next((r['paid'] for r in records if 100*r['correct_free_cells']>=95*denominator),None)
    return records,first


def choice_rows(directory, manifest, trace, initial):
    selections=read(directory/'global_choices.json')
    recorded=directory/'regional_choices.json'
    region={row['selection_call_id']:row for row in read(recorded)} if recorded.is_file() else {}
    recorded=directory/'regional_guards.json'
    regional_guards=read(recorded) if recorded.is_file() else []
    guard_rows=defaultdict(list)
    for row in [*read(directory/'audit.json'),*regional_guards]:
        if row.get('event')!='exploration_joint_replan_after_rejection' and 'motion' not in row:
            continue
        key=row.get('rejected_option_id',row.get('option_id'))
        entry=(frame_paid(row),row['reason'])
        if key and entry not in guard_rows[key]:
            guard_rows[key].append(entry)
    by_option=defaultdict(list)
    for step in trace:
        if step['active_option_id'] is not None:
            by_option[step['active_option_id']].append(step)
    results=[]; seen=set()
    for event in selections:
        selected=event['outputs']['selected']; call=event['call_id']; paid=event['action_id']
        extra=region.get(call,{})
        ranking=extra.get('ranking',{}); entrances=ranking.get('entrances',[])
        chosen=ranking.get('chosen',{})
        target=selected['pose'] if selected is not None else None
        option=selected['option_id'] if selected is not None else None
        actions=by_option.get(option,[])
        if option:
            require(option not in seen,f'Repeated selected option id {option}')
            seen.add(option)
            require(all((step['selection_call_id'],step['selected_target_pose'])==(call,target) for step in actions),
                    f'Actual dispatched macro cannot be bound to its selected target: {directory}/{option}')
        if extra:
            require(extra['actual_target_pose']==target and extra['paid_before']==paid and not ranking.get('uses_object_semantics'),
                    f'Regional sidecar differs from actual native target or used an excluded input: {directory}/{call}')
        reached=next((step for step in actions if [*step['position'],step['heading']]==target and not step['collision']),None)
        guards=sorted(guard_rows.get(option,[]),key=lambda entry: (entry[0] is None,entry[0],entry[1]))
        status=('no_option' if not option else 'arrived' if reached else 'guard_rejected' if guards else
                'dispatched_without_reach' if actions else 'selected_without_dispatch')
        comparisons=extra.get('native_shadow_pose') if extra else target
        results.append({'selection_call_id':call,'selected_paid_before':paid,'initial_pose':pose_at(trace,initial['anchor'],paid),
                        'candidate_count':len(event['outputs']['candidates']),'option_id':option,'target_pose':target,
                        'native_shadow_pose':comparisons,'changed_from_own_native_shadow':target!=comparisons,
                        'selection_origin':extra.get('decision_origin','original_native'),
                        'regional_reason':ranking.get('reason'),'proposal_count':extra.get('raw_proposal_count',0),
                        'new_certified_count':extra.get('new_native_certified_count',0),
                        'region_count':ranking.get('region_count'),'entrances_seen':len(entrances) if extra else None,
                        'credible_entrances':sum(bool(e.get('credible')) for e in entrances) if extra else None,
                        'crossed_observed_entrances':sum(bool(e.get('crossed_paid_steps')) for e in entrances) if extra else None,
                        'blocked_observed_entrances':sum(bool(e.get('blocked')) for e in entrances) if extra else None,
                        'visible_unvisited_region_ids':ranking.get('visible_unvisited_region_ids'),
                        'remaining_frontier_cells':chosen.get('remaining_frontier_cells'),
                        'chosen_observed_entrance_id':chosen.get('entrance_id'),
                        'planned_go':selected['outbound_cost'] if selected else None,
                        'planned_return':selected['return_cost'] if selected else None,
                        'first_dispatched_paid':actions[0]['step'] if actions else None,
                        'distinct_paid_actions_under_option':len(actions),'arrived_paid':reached['step'] if reached else None,
                        'guard_rejections':guards,'first_guard_paid_before':guards[0][0] if guards else None,
                        'first_guard_reason':guards[0][1] if guards else None,'actual_outcome':status})
    require(set(by_option)<=seen,f'Some activated native options lack a selected macro receipt: {directory}')
    if manifest.get('study_prior') is None:
        return frontier_choice_rows(directory,trace)
    return results,True


def frontier_choice_rows(directory, trace):
    """Consume paid-action observations from the optional formal-only logging hook."""
    events=read(directory/'frontier_events.json')
    dispatches=[row for row in events if row['event']=='instrumented_frontier_dispatch']
    if not dispatches:
        return [],False
    results=[]; observed=[row for row in events if row['event']=='instrumented_frontier_observation']
    guarded=[row for row in events if row['event']=='guard_rejection' and row['phase']=='outbound']
    active={}
    for row in dispatches:
        goal=row.get('goal_pose',row.get('target_pose'))
        if goal is None:
            continue
        paid=frame_paid(row)
        require(paid is not None and 0<=paid<len(trace),f'Invalid frontier dispatch clock: {directory}')
        expected=row.get('action')
        require(expected is None or trace[paid]['action']==expected,f'Frontier planned action was not the charged action: {directory}')
        identity=row.get('goal_count',row.get('option_id',tuple(goal)))
        if identity not in active:
            active[identity]={'selection_call_id':None,'selected_paid_before':paid,'option_id':str(identity),
                              'target_pose':goal,'first_dispatched_paid':paid+1,'last_dispatched_paid':paid+1,
                              'selection_origin':'strict_frontier','distinct_paid_actions_under_option':0}
        receipt=active[identity]
        require(receipt['target_pose']==goal,f'The same frontier receipt changed pose: {directory}/{identity}')
        receipt['last_dispatched_paid']=paid+1
        receipt['distinct_paid_actions_under_option']+=1
    for identity,row in active.items():
        arrivals=[q.get('paid',q.get('arrived_paid',q.get('paid_after'))) for q in observed
                  if q.get('goal_count',q.get('option_id'))==identity and q.get('goal_reached',q.get('arrived',False))]
        require(len(arrivals)<=1 and all(row['first_dispatched_paid']<=p<=row['last_dispatched_paid']
                and [*trace[p-1]['position'],trace[p-1]['heading']]==row['target_pose'] and not trace[p-1]['collision']
                for p in arrivals),f'Frontier arrival cannot be independently matched: {directory}/{identity}')
        reached=next(iter(arrivals),None)
        credited=[(frame_paid(q),q['assessment']['reason']) for q in guarded
                  if frame_paid(q)==row['last_dispatched_paid'] and reached is None]
        results.append(row|{'arrived_paid':reached,'actual_outcome':'arrived' if reached is not None else
                           'guard_rejected' if credited else 'dispatched_without_reach','guard_rejections':credited,
                           'first_guard_paid_before':credited[0][0] if credited else None,
                           'first_guard_reason':credited[0][1] if credited else None})
    return results,True


def validate_inputs(phase, partial):
    freeze_path=ROOT/f'{phase.upper()}_FREEZE.json'
    frozen=read(freeze_path); digest=sha(freeze_path)
    protocol=ROOT/frozen['protocol_relative']; spec=read(protocol)
    require(frozen['phase']==phase and sha(protocol)==frozen['protocol_sha256'], 'Study protocol/freeze mismatch')
    require(all(frozen[k]==spec[k] for k in ('budget','capture_paid','local_route_invariants','local_auxiliary_weights')),
            'Frozen comparison fields differ from protocol')
    jobs={(job['layout'],job['arm']):job for job in frozen['jobs']}
    require(set(jobs)=={(l,a) for l in spec['layouts'] for a in spec['arms']},
            'Frozen study does not contain the complete declared matrix')
    for name,expected in (frozen['actor_files']|frozen['evaluator_files']|frozen['actor_initial_files']).items():
        require(sha(name)==expected,f'Frozen file drift: {name}')
    execution=ROOT/'execution'/phase
    receipt_path=execution/'phase_receipt.json'
    if receipt_path.is_file():
        phase_receipt=read(receipt_path)
        require((phase_receipt['freeze_sha256'],phase_receipt['phase'])==(digest,phase),'Wrong phase receipt')
        rows=phase_receipt['episodes']
        require(len(rows)==len(jobs) and {(r['layout'],r['arm']) for r in rows}==set(jobs),
                'Phase receipt missing frozen arms')
    else:
        require(partial,'A complete phase receipt is required. Use --partial for interim results.')
        rows=[]
        for layout,arm in jobs:
            exits=sorted((execution/layout/arm).glob('attempt_*/exit.json'))
            if exits:
                rows.append(read(exits[-1]))
    receipts={}
    for row in rows:
        key=(row['layout'],row['arm']); job=jobs[key]
        require(row['freeze_sha256']==digest and all(row.get(k)==v for k,v in job.items()),
                f'Worker receipt differs from frozen job: {key}')
        receipts[key]=row
    return frozen,spec,digest,jobs,receipts,receipt_path if receipt_path.is_file() else None


def measure_episode(ev, directory, job, worker, frozen, digest, scene_root, world, metric):
    layout,arm=job['layout'],job['arm']
    id_fields={'layout':layout,'arm':arm}
    row=id_fields|{'status':'pending','prior':job['prior'],'regional_enabled':job['regional_enabled']}
    if worker is None:
        return row,{},{'raw':None,'start':None}
    row|={'run_directory':str(directory),'exit_code':worker['exit_code'],'literal':job['literal']}
    result_path=Path(worker['result_path']) if worker['result_path'] else None
    current_digest=sha(result_path) if result_path and result_path.is_file() else None
    require(current_digest==worker['result_sha256'],f'Process result receipt drift: {directory}')
    initial_path=scene_root/'prepared'/layout/'initial_state.json'
    if worker['exit_code']!=0:
        return failed_episode(ev,directory,row,worker,initial_path,metric)
    manifest_path=directory/'manifest.json'
    require(Path(worker['result_path'])==manifest_path,f'Completed attempt has no sealed manifest: {directory}')
    manifest,manifest_sha=ev.engine.verify_manifest(directory,'manifest.json')
    initial=read(initial_path)
    attempt=read(directory/'attempt.json')
    if job['literal']:
        expected_arm,expected_freeze,protocol=arm,digest,frozen['protocol_sha256']
        require(manifest.get('regional_enabled')==job['regional_enabled'],f'Regional switch drift: {directory}')
    else:
        expected_arm,expected_freeze,protocol='G_perf',sha(ARCHIVED_FREEZE),sha(WEIGHTED/'PROTOCOL.json')
        proof=read(REGRESSION)
        require(proof['status']=='exact_native_regression_passed' and proof['new_manifest_sha256']==manifest_sha
                and worker['result_sha256']==job['reused_manifest_sha256'],f'Archived neutral control differs: {directory}')
    require((manifest['status'],manifest['stage'],manifest['layout'],manifest['arm'])==
            ('completed_execution','holdout',layout,expected_arm) and manifest['actual_closed_loop']
            and not manifest['forced_route_or_goal'] and not manifest['evaluator_inputs_read_by_actor'],
            f'Closed-loop treatment drift: {directory}')
    require(manifest['native_budget']==frozen['budget'] and manifest['study_prior']==job['prior']
            and all(manifest[k]==v for k,v in frozen['local_auxiliary_weights'].items())
            and manifest['runner_freeze_sha256']==attempt['runner_freeze_sha256']==expected_freeze
            and manifest['source_protocol_sha256']==attempt['source_protocol_sha256']==protocol
            and manifest['actual_scene_seed']==initial['seed'] and manifest['frozen_cold_input_sha256']==sha(initial_path)
            and attempt['initial_owner_sha256']==initial['initial_owner_sha256']
            and attempt['initial_pose']==initial['anchor'],f'Runtime deviates from frozen configuration: {directory}')
    ref=ev.load_reference(layout,initial)
    raw=read(directory/'initial_raw_rgbd_scan_sha256.json')
    require(raw['step']==0 and raw['packet_sha256']==initial['initial_packet_sha256']
            and all(raw.get(field) for field in ev.SENSORS),f'Initial sensor fingerprint missing: {directory}')
    trace=read(directory/'steps.json')
    chain=ev.native.verify_trace(trace,initial,frozen['budget'],manifest)
    ev.native.verify_action_guard(trace,frozen['budget'])
    require(all(all(step.get(k) for k in ev.SENSORS) for step in trace),f'Missing actual sensor hashes: {directory}')
    if job['prior'] is None:
        require(all(read(directory/name)==[] for name in ('global_choices.json','local_choices.json')),
                f'Strict frontier used the NSO selection layer: {directory}')
        counts={}
    else:
        counts=ev.verify_pool(directory,manifest,initial,trace)
    expected=['start',*(f'paid_{clock}' for clock in frozen['capture_paid'] if clock),'terminal']
    require(set(manifest['checkpoints'])==set(expected),f'Checkpoint schedule drift: {directory}')
    clouds=[]; curve,a95=coverage_curve(directory,ref,trace)
    for event in expected:
        observation,masks=ev.native.evaluate_snapshot(directory,event,manifest,initial,trace,ref,metric)
        if event=='start':
            baseline,baseline_masks=observation,masks
        if event=='terminal':
            terminal=observation
            require(terminal['packet_chain_sha256']==chain,f'Unbound terminal map: {directory}')
        if event.startswith('paid_'):
            require(observation['status']==('captured' if int(event[5:])<=len(trace) else 'unavailable'),
                    f'Checkpoint absent or padded: {directory}/{event}')
        if observation['status']=='captured':
            require(abs(curve[observation['actual_paid']]['C_free']-observation['C_free'])<1e-12,
                    f'Per-action correct-free trace and checked checkpoint differ: {directory}/{event}')
            observation=ev.engine.augment_measurement(observation,masks,baseline,baseline_masks,ref)
        clouds.append(id_fields|{'checkpoint':event}|observation)
    motion,room_crossings=executed_motion(trace,initial,
        room_map(world,layout,initial['shape'],initial['grid_resolution_m']),initial['grid_resolution_m'])
    closed=ev.native.close_record(terminal,read(directory/'runtime_summary.json'),trace,initial)
    require(closed['returned']==manifest['returned'] and closed['terminal_pose']==manifest['final_pose']
            and closed['termination']==manifest['observed_native_termination'] and closed['collisions']==manifest['collisions']
            and math.isclose(closed['path_length_m'],motion['actual_translated_edges']*initial['grid_resolution_m'],abs_tol=1e-7),
            f'Independent native closure and actual trajectory disagree: {directory}')
    decisions,logged=choice_rows(directory,manifest,trace,initial)
    dispatched=[s for s in decisions if s.get('first_dispatched_paid') is not None]
    arrival_count=sum(s['actual_outcome']=='arrived' for s in dispatched)
    clocks=[step['decision_wall_s'] for step in trace]
    replans=[clocks[s['first_dispatched_paid']-1] for s in dispatched]
    elapsed=manifest['elapsed_s']
    require(all(math.isfinite(t) and t>=0 for t in [*clocks,elapsed]),f'Invalid clock: {directory}')
    row|=closed|counts|motion|{k:terminal[k] for k in (*METRICS,'p_scene_05cm','r_scene_05cm',
          'correct_free_cells','reachable_cells')}
    row|={'status':'completed_execution','A95_paid':a95,'elapsed_s':elapsed,
          'bootstrap_and_task_wall_s':manifest.get('bootstrap_and_task_wall_s'),
          'decision_wall_p50_s':percentile(clocks,50),'decision_wall_p95_s':percentile(clocks,95),
          'dispatched_plan_wall_p50_s':percentile(replans,50),'dispatched_plan_wall_p95_s':percentile(replans,95),
          'selected_macros':sum(s.get('option_id') is not None for s in decisions) if logged else None,
          'dispatched_macros':len(dispatched) if logged else None,'arrived_macros':arrival_count if logged else None,
          'target_arrival_rate':arrival_count/len(dispatched) if dispatched else None,
          'guarded_macros':sum(s['actual_outcome']=='guard_rejected' for s in decisions) if logged else None,
          'changed_own_shadow_dispatched':sum(bool(s.get('changed_from_own_native_shadow')) for s in dispatched),
          'source_manifest_sha256':manifest_sha,'source_reference_receipt_sha256':ref['source_reference_receipt_sha256'],
          'initial_packet_sha256':initial['initial_packet_sha256'],'initial_owner_sha256':initial['initial_owner_sha256']}
    tables={'checkpoints':clouds,'decisions':decisions,'coverage_curve':curve,'room_crossings':room_crossings,'trajectory':trace}
    binding={'start':baseline,'raw':raw,'trace':trace,'receipt':terminal,'decisions':decisions}
    return row,{k:[id_fields|record for record in records] for k,records in tables.items()},binding


def failed_episode(ev,directory,row,worker,initial_path,metric):
    row|={'status':'technical_failure','stop_reason':read(worker['result_path']).get('error')
          if worker['result_sha256'] else None,'failure_sha256':worker['result_sha256']}
    # Incomplete attempts may have atomically published checked checkpoint payloads before process loss.
    trace_path=directory/'steps.json'
    heartbeat=directory/'heartbeat.json'
    row['last_reported_paid']=read(heartbeat).get('actual_paid') if heartbeat.is_file() else None
    initial=read(initial_path)
    reference=ev.load_reference(row['layout'],initial)
    trace=read(trace_path) if trace_path.is_file() else None
    evidence=[]
    for file in sorted(directory.glob('*/checkpoint.json')):
        recorded=read(file)
        if recorded['status']!='captured':
            continue
        verified=trace is not None and recorded['actual_paid']<=len(trace)
        if verified:
            ev.engine.verify_checkpoint_against_trace(recorded,recorded['actual_paid'],trace,initial)
        paths={name:file.parent/name for name in ('native_mesh.ply','belief.npz')}
        require(all(sha(path)==recorded['files_sha256'][name] for name,path in paths.items()),
                f'Invalid partial checkpoint: {file}')
        measured,_=ev.engine.evaluate_verified_snapshot(paths['native_mesh.ply'],paths['belief.npz'],recorded,reference,metric)
        evidence.append({'layout':row['layout'],'arm':row['arm'],'checkpoint':file.parent.name,
                         'source_checkpoint_sha256':sha(file),'partial_action_trace_verified':verified,**recorded,**measured})
    if evidence:
        inherited=max(evidence,key=lambda e:e['actual_paid'])
        row|={'last_valid_paid':inherited['actual_paid'],
              'last_valid_partial_action_trace_verified':inherited['partial_action_trace_verified'],
              **{f'last_valid_{k}':inherited[k] for k in METRICS}}
    return row,{'checkpoints':evidence},None


def pair_divergence(ev,layout,names,bindings):
    left,right=(bindings[(layout,name)] for name in names)
    summary=ev.pair_trace(layout,names[0],left,names[1],right)
    if summary['status']!='measured':
        return summary
    dispatch=[]
    for binding in (left,right):
        by_step={e['first_dispatched_paid']:e for e in binding['decisions'] if e.get('first_dispatched_paid') is not None}
        dispatch.append(by_step)
    for paid in sorted(dispatch[0].keys()|dispatch[1].keys()):
        entries=[d.get(paid) for d in dispatch]
        targets=[e['target_pose'] if e else None for e in entries]
        if targets[0]==targets[1]:
            continue
        summary|={'first_dispatch_difference_paid':paid,'same_paid_prefix_at_dispatch':ev.native.same_paid_prefix(
                  left['trace'],right['trace'],paid-1)}
        for label,entry in zip(('a','b'),entries,strict=True):
            summary|={f'first_dispatch_{label}_{key}':entry.get(key) if entry else None for key in (
                'target_pose','option_id','arrived_paid','actual_outcome','first_guard_paid_before','first_guard_reason')}
        break
    else:
        summary['first_dispatch_difference_paid']=None
    return summary


def contrast_summary(spec,episodes,phase):
    definitions=FACTORIAL if phase=='formal' else {'development_global_given_G':FACTORIAL['global_given_G']}
    rows=[]; aggregates={}
    def observed(v):
        return v if isinstance(v,(int,float)) else None
    for name,terms in definitions.items():
        group=[]
        for layout in spec['layouts']:
            arms={arm:episodes[(layout,arm)] for arm in terms}
            finished=all(r['status']=='completed_execution' for r in arms.values())
            row={'layout':layout,'contrast':name,'status':'measured' if finished else 'incomplete_pair'}
            if finished:
                for key in (*METRICS,*MOTION):
                    values=[observed(r.get(key)) for r in arms.values()]
                    if all(v is not None for v in values):
                        row[f'delta_{key}']=sum(w*v for w,v in zip(terms.values(),values,strict=True))
                        if key in METRICS:
                            row[f'delta_{key}_pp']=100*row[f'delta_{key}']
                row['A95_reached_arms']=[a for a,r in arms.items() if r.get('A95_paid') is not None]
                row['first_new_room_arms']=[a for a,r in arms.items() if r.get('first_new_room_paid') is not None]
                if 'delta_first_new_room_paid' in row:
                    row['scale_first_new_room']=abs(row['delta_first_new_room_paid'])>=4
                if 'delta_repeat_distance_m' in row:
                    row['scale_repeat_distance']=abs(row['delta_repeat_distance_m'])>=.4-1e-10
            group.append(row)
        rows.extend(group)
        d=[r['delta_J'] for r in group if 'delta_J' in r]
        aggregates[name]={'terms':terms,'buildings_compared':len(d),'buildings_frozen':len(group),
            'map_effect_threshold_met':len(d)==4 and sum(v>.005 for v in d)>=3 and min(d)>=-.005
                if phase=='formal' and len(d)==4 else None,'metrics':{}}
        for key in (*METRICS,*MOTION):
            values=[r[f'delta_{key}'] for r in group if f'delta_{key}' in r]
            if values:
                aggregates[name]['metrics'][key]={'n':len(values),'mean':statistics.fmean(values),
                                                  'minimum':min(values),'maximum':max(values)}
    return rows,aggregates


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('development','formal'))
    parser.add_argument('--partial','--completed-only',action='store_true',dest='partial')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    frozen,spec,digest,jobs,receipts,phase_receipt=validate_inputs(args.phase,args.partial)
    scene_root=Path(frozen['scene_root'])
    ev=import_file('regional_independent_weighted_metrics',WEIGHTED_METRIC)
    ev.SCENES=scene_root
    world=import_file('regional_independent_floor_geometry',scene_root/'scene.py')
    metric=ev.files.load_metric(read(ev.files.SCENES/'freeze.json'),read(ev.files.OLD_EVAL/'freeze.json'))
    episodes,bindings,tables={},{},defaultdict(list)
    for key,job in jobs.items():
        worker=receipts.get(key)
        directory=Path(worker['run_path']) if worker else None
        value,parts,binding=measure_episode(ev,directory,job,worker,frozen,digest,scene_root,world,metric)
        episodes[key]=value
        bindings[key]=binding if value['status']=='completed_execution' else None
        for name,rows in parts.items():
            tables[name].extend(rows)
        print(f'{key[0]} {key[1]} {value["status"]} {value.get("J","")}',flush=True)
    tables['episodes']=list(episodes.values())
    tables['first_divergence']=[pair_divergence(ev,layout,names,bindings) for layout in spec['layouts']
       for names in combinations(spec['arms'],2)]
    tables['contrasts'],aggregates=contrast_summary(spec,episodes,args.phase)
    output=args.out or ROOT/'results'/args.phase/('interim' if args.partial else '')
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        write_csv(output/f'{name}.csv',rows)
    summary={'phase':args.phase,'status':'interim' if args.partial else 'evaluated',
             'run_freeze_sha256':digest,'protocol_sha256':frozen['protocol_sha256'],'evaluator_sha256':sha(__file__),
             'phase_receipt_sha256':sha(phase_receipt) if phase_receipt else None,
             'independent_reporting_unit':'building','frozen_trials':len(jobs),
             'complete_endpoints':sum(r['status']=='completed_execution' for r in episodes.values()),
             'technical_failures':sum(r['status']=='technical_failure' for r in episodes.values()),
             'pending_trials':sum(r['status']=='pending' for r in episodes.values()),
             'episodes':tables['episodes'],'pairs':tables['first_divergence'],'contrasts':tables['contrasts'],
             'contrast_aggregates':aggregates}
    temporary=output/'summary.json.tmp'
    temporary.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    temporary.replace(output/'summary.json')


if __name__=='__main__':
    main()
