#!/usr/bin/env python3
"""One native independently fused local mechanism execution, or read-only calibration."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import traceback

from direct_inspection import rank_direct


HERE = Path(__file__).resolve().parent
ANALYSES = HERE.parents[1]
WEIGHTED = ANALYSES/'weighted_geometric_control_20261122'
COLD = ANALYSES/'same_goal_cold_holdout_20261121'
SAME = ANALYSES/'same_goal_route_choice_20261121'
SCENES = HERE/'scenes'
PROTOCOL = HERE/'PROTOCOL.json'
CALIBRATION = HERE/'calibration.json'
FREEZE = HERE/'FROZEN_INPUTS.json'
ARMS = ('G','D','S','X')

sys.path[:0] = [str(WEIGHTED/'runner'),str(COLD/'runner')]
from cold_run import full_local_route_adapter, native, raw_sensor_hashes
from execute import missing
from weighted_score import Weights, choose_route, rank_admitted


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def protocol(layouts, protocol):
    source = read(SAME/'PROTOCOL.json')
    source['heldout'].update(layouts=list(layouts), paid_action_budget=protocol['native_budget'])
    for key,value in protocol['frozen_online_fields'].items():
        if source['local_route'][key] != value:
            raise ValueError(f'Native rule changed: {key}')
    source['diagnostic_checkpoints'] = protocol['capture_paid'][1:]
    return source


def calibrate():
    layout = 'paired_cycle_20'
    prior = read(PROTOCOL)
    spec = protocol((layout,),prior)
    native.HOLDOUT = WEIGHTED/'scenes'
    native.ARM_MODES = {'calibration': ('G','G')}
    old, (world, _, packet, components, runtime, _, _), _, _, _ = native.setup_runtime('holdout',layout,'calibration',spec)
    import runtime_binding as binding
    weights = Weights(**prior['weights'])
    archive = WEIGHTED/'runner/runs/holdout'/layout/'S'
    saved = next(e for e in read(archive/'local_choices.json') if e['selection_call_id']==5)['choice_audit']
    legacy = native.load_source('nso_direct_calibration_local',native.METHOD)
    getter = components.get_selected_option
    cfg = dict(spec['local_route'],max_alternates=4*components._cpu_backend.scenes[0]['mapper'].belief.size+1)
    calibration = {}

    def inspect(scene):
        option = getter(scene)
        if option is None or calibration:
            return option
        state = components._cpu_backend.scenes[scene]
        selected, audit = choose_route(legacy,legacy.choose_route,state,option,'G',weights,old.method.route_union_area,cfg,
                                       binding.revision.reweight)
        assert world.step_count==0 and packet.action_id==0
        assert audit['candidate_pool_sha256']==saved['candidate_pool_sha256']
        trials = [rank_direct(state,audit['candidates'],weights,scale,cfg['score_equality_tolerance'],rank_admitted)[1]
                  for scale in prior['direct']['scale_grid']]
        choice = next((x['scale'] for x in trials if x['selected_route_id']!=audit['selected_route_id']),1.)
        calibration.update(layout=layout,paid=0,actor_action_issued=False,ranked_pool_sha256=audit['candidate_pool_sha256'],
                           G_perf_route_id=audit['selected_route_id'],fixed_G_perf_weights=prior['weights'],
                           inspected_scales=trials,chosen_scale=choice,calibration_passes=1,
                           source_manifest_sha256=sha(archive/'manifest.json'))
        return selected

    components.get_selected_option = inspect
    runtime.next_local_action(0)
    assert calibration and world.step_count==0
    with CALIBRATION.open('x') as handle:
        json.dump(calibration,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print('One calibration on exact archived pool',calibration['G_perf_route_id'],calibration['chosen_scale'],flush=True)


def branch_event(steps, trigger, terminal, audit):
    if trigger is None:
        return {'status':'not_triggered','actual_paid':None}
    held = trigger['option_id']
    count, prior = 0, trigger['paid_before']
    path = next((row['path_length_m'] for row in steps if row['step']==prior),0.)
    current_path = path
    for row in steps[prior:]:
        if row['active_option_id'] != held:
            faults = [e['reason'] for e in audit if e.get('event')=='exploration_joint_replan_after_rejection'
                       and e.get('rejected_option_id')==held]
            return dict(status='abandoned_or_replanned_before_arrival',actual_paid=prior,actions=count,
                        goal=trigger['target_pose'],path_length_m=current_path-path,
                        reason=faults[0] if faults else 'native_active_option_replaced')
        count += 1
        prior = row['step']
        current_path = row['path_length_m']
        if row['collision'] or row.get('matched_target_pose_after_step'):
            return dict(status='collision' if row['collision'] else 'arrived',actual_paid=prior,actions=count,
                        goal=trigger['target_pose'],path_length_m=row['path_length_m']-path,reason=row['guard_reason'])
    return dict(status='terminal_before_arrival',actual_paid=prior,actions=count,goal=trigger['target_pose'],path_length_m=current_path-path,
                reason=terminal.get('reason'))


def execute(layout, arm, out):
    freeze = read(FREEZE)
    assert {'layout':layout,'arm':arm} in freeze['schedule']
    for path,digest in freeze['actor_files_sha256'].items():
        if sha(path)!=digest:
            raise ValueError(f'Frozen actor input changed: {path}')
    declared = read(PROTOCOL)
    spec = protocol(declared['placements'],declared)
    scale = read(CALIBRATION)['chosen_scale']
    weights = Weights(**declared['weights'])
    raw, tested = [], []
    fork = trigger = writer = None
    original_setup, original_load = native.setup_runtime,native.load_source
    original_bind, original_read = native.local_route_adapter,native.read
    native.HERE,native.HOLDOUT,native.PROTOCOL = HERE,SCENES,PROTOCOL
    native.ARM_MODES = {name:('G','G') for name in ARMS}
    initial = read(SCENES/'prepared'/layout/'initial_state.json')

    def read_native(path):
        return {'world_files_sha256':freeze['actor_files_sha256']} if Path(path)==HERE/'holdout_freeze.json' else original_read(path)

    def setup(*args):
        nonlocal writer
        result = original_setup(*args)
        old, (world, _, first, _, _, _, _), _, _, _ = result
        assert world.step_count==0 and first.sha256()==initial['initial_packet_sha256']
        raw.append(raw_sensor_hashes(first,native.array_sha))
        original_packet = old.packet

        def capture_packet(*a,**kw):
            packet = original_packet(*a,**kw)
            assert packet.action_id==len(raw)
            raw.append(raw_sensor_hashes(packet,native.array_sha))
            return packet

        old.packet=capture_packet
        import runtime_binding as binding
        writer=binding.write

        def defer(path,value):
            return writer(out/'.native_manifest.json' if Path(path)==out/'manifest.json' else path,value)

        binding.write=defer
        return result

    def load(name,path):
        module=original_load(name,path)
        if Path(path)!=native.METHOD:
            return module
        original=module.choose_route

        def choose(state,option,mode,area,config):
            nonlocal fork,trigger
            import runtime_binding as binding

            selected,g = choose_route(module,original,state,option,'G',weights,area,config,binding.revision.reweight)
            paid = state['packet'].action_id
            if trigger is not None or paid>64 or g['admitted_candidate_count']<2:
                return selected,g
            direct,d = rank_direct(state,g['candidates'],weights,scale,config['score_equality_tolerance'],rank_admitted)
            separation = max(d['values'])-min(d['values'])
            tested.append(dict(paid=paid,selection_call_id=option['selection_call_id'],direct_area_range_m2=separation,
                               candidates=g['admitted_candidate_count'],pool_sha256=g['candidate_pool_sha256']))
            if separation<=1e-9:
                return selected,g

            choices={'G':(selected,g),'D':(module._same_goal_option(option,direct),g)}
            for prior in ('S','X'):
                choices[prior]=choose_route(module,original,state,option,prior,weights,area,config,binding.revision.reweight)
                assert choices[prior][1]['candidate_pool_sha256']==g['candidate_pool_sha256']
            route, audit = choices[arm]
            winner = d['selected_route_id'] if arm=='D' else audit['selected_route_id']
            if arm=='D':
                audit=dict(audit,selected_mode='D',selected_route_id=winner,
                           chosen_outbound_sha256=direct['outbound_sha256'])
            audit['single_direct_inspection_mechanism_arm']=arm
            audit['direct_inspection']=d
            chain=native.chain_hash('0'*64,raw[0]['packet_sha256'])
            for row in raw[1:paid+1]:
                chain=native.chain_hash(chain,row['packet_sha256'])
            fork=native.capture(out,'fork',state['mapper'],state['packet'],chain,paid,'common_observed_local_mechanism_fork')
            trigger=dict(status='triggered',arm=arm,paid_before=paid,selection_call_id=option['selection_call_id'],
                         option_id=option['option_id'],target_pose=option['pose'],actual_route_id=winner,
                         actual_outbound_sha256=audit['chosen_outbound_sha256'],candidate_pool_sha256=g['candidate_pool_sha256'],
                         actual_nominal_go_actions=len(route['outbound_actions']),native_reserved_return_actions=len(route['return_actions']),
                         decision_table={key:(d['selected_route_id'] if key=='D' else rec['selected_route_id'])
                                         for key,(_,rec) in choices.items()},
                         inspection_area_range_m2=separation, observed_support_voxels=d['observed_support_voxels'])
            writer(out/'mechanism_trigger.json',trigger)
            return route,audit

        module.choose_route=choose
        return module

    native.read=read_native
    native.setup_runtime,native.load_source = setup,load
    native.local_route_adapter = lambda *a: full_local_route_adapter(*a,original_bind)
    try:
        native.run('holdout',layout,arm,spec,out)
    finally:
        if writer is not None:
            import runtime_binding as binding
            binding.write=writer
    manifest=read(out/'.native_manifest.json')
    (out/'.native_manifest.json').unlink()
    steps=read(out/'steps.json')
    assert len(steps)==len(raw)-1==manifest['paid_actions']<=declared['native_budget']
    for action,packet in zip(steps,raw[1:],strict=True):
        assert action['step']==packet['step'] and action['packet_sha256']==packet['packet_sha256']
        action.update({key:val for key,val in packet.items() if key.endswith('values_sha256') or key=='scan_ranges_sha256'})
    writer(out/'steps.json',steps)
    writer(out/'initial_raw_rgbd_scan_sha256.json',raw[0])
    writer(out/'trigger_probes.json',tested)
    if trigger is None:
        writer(out/'mechanism_trigger.json',{'status':'not_triggered','arm':arm})
    checkpoints=manifest['checkpoints']
    if fork:
        checkpoints['fork']=fork
    for paid in declared['capture_paid'][1:]:
        key=f'paid_{paid}'
        if key not in checkpoints:
            checkpoints[key]=missing(out,key,paid,manifest['paid_actions'],'native_stop_before_checkpoint')
    manifest.update(run_kind='same_goal_one_autonomous_local_mechanism',frozen_actor_files_sha256=sha(FREEZE),
                    experimental_local_arm=arm,applied_local_policy=trigger,continuation_policy='G_perf',
                    route_event=branch_event(steps,trigger,manifest['observed_native_termination'],read(out/'audit.json')),
                    files_sha256={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*'))
                                  if path.is_file() and path.name!='manifest.json'})
    writer(out/'manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--layout')
    parser.add_argument('--arm',choices=ARMS)
    args=parser.parse_args()
    if args.calibrate:
        calibrate()
        return
    if not args.arm or not args.layout:
        parser.error('Specify layout and arm')
    out=HERE/'runs'/args.layout/args.arm
    out.mkdir(parents=True,exist_ok=False)
    try:
        execute(args.layout,args.arm,out)
    except BaseException as error:
        write(out/'failure.json',{'status':'technical_failure','error':repr(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
