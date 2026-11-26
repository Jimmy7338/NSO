#!/usr/bin/env python3
"""Check the actual full-pool adapter against the two archived development witnesses."""

import argparse
from copy import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from cold_run import ANALYSES, HERE, PROTOCOL, SOURCE, full_local_route_adapter, native, raw_sensor_hashes, read, sha

PRIOR = ANALYSES / 'same_goal_two_witness_20261121'
WITNESSES = ('gallery_left:56:230', 'warehouse_right:142:578')
MODES = ('G', 'S')
RAW_KEYS = ('depth_values_sha256','rgb_values_sha256','scan_ranges_sha256')


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def one(witness, mode, out):
    protocol = read(PROTOCOL)
    contract = protocol['engineering_regression_before_holdout'][witness]
    layout, paid, call_id = witness.split(':')
    paid, call_id = int(paid),int(call_id)
    stop_paid = contract['paid_first_divergence']
    old_protocol = read(SOURCE / 'PROTOCOL.json')
    archived = PRIOR / 'runner' / 'runs' / layout / mode
    expected = read(archived / 'steps.json')
    old_manifest = read(archived / 'manifest.json')
    expected_witness = read(archived / 'intervention.json')
    assert old_manifest['status']=='completed_execution'
    for name in ('steps.json','intervention.json'):
        assert sha(archived/name)==old_manifest['files_sha256'][name]
    old, values, initial, global_mode, local_mode = native.setup_runtime('development',layout,'newG',old_protocol)
    world, transform, packet, components, runtime, _, owner = values
    mapper = runtime.states[0]['mapper']
    assert global_mode == local_mode == 'G' and packet.sha256()==old_manifest['initial_packet_sha256']
    config = old_protocol['local_route']
    assert all(config[key]==value for key,value in protocol['frozen_online_fields'].items())
    alternative_components = copy(components)
    events = full_local_route_adapter(alternative_components,old,mode,config,native.local_route_adapter)
    native.local_route_adapter(components,old,'G',config)
    normal = components.get_selected_option
    backend = components._cpu_backend
    guard_order = []

    def choose(scene_index):
        step = backend.calls[-1]
        if step['action_id'] != paid:
            return normal(scene_index)
        assert step['call_id']==call_id and step['method']=='select_topo_target'
        assert step['outputs']['mode']=='G' and step['outputs']['selected']['pose']==expected_witness['native_target']
        if 'original_guard_rejection_call' in contract:
            guard = backend.calls[contract['original_guard_rejection_call']-1]
            assert guard['call_id']==contract['original_guard_rejection_call']
            assert guard['call_id']<step['call_id']==contract['native_reselection_call']
            assert guard['method']=='assess_local_action' and guard['action_id']==paid
            assert not guard['outputs']['allowed'] and guard['outputs']['reason']=='next_footprint_not_known_safe'
            guard_order.extend([dict(call_id=guard['call_id'],method=guard['method'],outputs=guard['outputs']),
                                dict(call_id=step['call_id'],method=step['method'],target=step['outputs']['selected']['pose'])])
        selected = alternative_components.get_selected_option(scene_index)
        result = events[-1]
        assert result['active_local_route_id']==expected_witness['chosen_route_id']==contract['winners'][mode]
        assert result['choice_audit']['candidate_pool_sha256']==expected_witness['expanded_candidate_pool_sha256']
        assert result['choice_audit']['score_table_sha256']==expected_witness['expanded_score_table_sha256']
        return selected

    components.get_selected_option = choose
    chain = native.chain_hash('0'*64, packet.sha256())
    observed = []
    for row in expected[:stop_paid]:
        shadow = runtime.next_local_action(0)
        assert shadow==row['action'] and runtime.states[0]['pending']['allowed']
        if row['step']==paid+1 and 'authorization_call' in contract:
            auth = backend.calls[-1]
            assert auth['call_id']==contract['authorization_call'] and auth['method']=='assess_local_action'
            assert auth['outputs']['allowed'] and auth['outputs']['action']==shadow
            guard_order.append(dict(call_id=auth['call_id'],method=auth['method'],outputs=auth['outputs']))
        action = row['action'] if row['step']<=paid else shadow
        raw, collision, done = world.step(action)
        packet = old.packet(world,transform,raw,layout,action,collision,done)
        owner = old.front.associate(mapper,raw,packet.frame_id)
        old.method.observe(mapper,packet.frame)
        runtime.observe(0,world.step_count,None,None,None,sensor_packet=packet)
        chain = native.chain_hash(chain,packet.sha256())
        assert packet.sha256()==row['packet_sha256'] and chain==row['packet_chain_sha256']
        assert [*packet.position,packet.heading]==[*row['position'],row['heading']]
        assert owner['owner_sha256']==row['owner_sha256'] and int((mapper.belief==0).sum())==row['observed_free_cells']
        sensors = raw_sensor_hashes(packet,native.array_sha)
        assert all(sensors[key]==row[key] for key in RAW_KEYS)
        observed.append({**sensors,'action':action,'position':list(packet.position),'heading':packet.heading})
    assert len(events)==1 and world.step_count==stop_paid and not world.collisions
    event=events[0]
    write_once(out,dict(status='exact_short_development_regression',layout=layout,witness=witness,mode=mode,
        actor_full_pool_sha256=sha(HERE/'cold_run.py'),regression_code_sha256=sha(Path(__file__)),
        compared_archive_manifest_sha256=sha(archived/'manifest.json'),compared_steps=stop_paid,
        matched_before_intervention=paid,local_call_id=call_id,local_route_id=event['active_local_route_id'],
        full_pool=event['full_pool'],last_action=observed[-1]['action'],last_sensor_packet_sha256=packet.sha256(),
        restored_native_guard_and_observations=True,guard_sequence=guard_order,
        measured_action_pose_and_sensor_receipts=observed,holdout_opened=False,complete_exploration_trial=False,
        no_collision=not world.collisions))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--witness',choices=WITNESSES)
    parser.add_argument('--mode',choices=MODES)
    args=parser.parse_args()
    work=HERE/'regression'
    protocol=read(PROTOCOL)
    assert protocol['stage'] and protocol['native_budget']==512
    if args.mode:
        assert args.witness is not None
        one(args.witness,args.mode,work/f'{args.witness.split(":",1)[0]}_{args.mode}.json')
        return 0
    work.mkdir(exist_ok=False)
    indexed=read(PRIOR/'publication/receipt.json')
    assert indexed['status']=='verified_published' and indexed['head']==protocol['source_commit']
    write_once(work/'start.json',dict(status='regression_frozen_before_action',source_commit=indexed['head'],
        started_unix_s=time.time(),protocol_sha256=sha(PROTOCOL),actor_code_sha256=sha(HERE/'cold_run.py'),
        regression_code_sha256=sha(Path(__file__)),registered_witnesses=list(WITNESSES)))
    children=[]
    for witness in WITNESSES:
        for mode in MODES:
            with (work/f'{witness.split(":",1)[0]}_{mode}.log').open('x') as stream:
                task=subprocess.Popen([sys.executable,str(Path(__file__)),'--witness',witness,'--mode',mode],
                                      stdout=stream,stderr=subprocess.STDOUT)
            children.append(task)
    exits=[task.wait() for task in children]
    assert all(code==0 for code in exits)
    results={}
    for witness in WITNESSES:
        contract=protocol['engineering_regression_before_holdout'][witness]
        layout=witness.split(':',1)[0]
        modes={mode:read(work/f'{layout}_{mode}.json') for mode in MODES}
        a,b=(modes[mode]['measured_action_pose_and_sensor_receipts'] for mode in MODES)
        delta=contract['paid_first_divergence']
        assert len(a)==len(b)==delta and a[:-1]==b[:-1]
        assert all(a[-1][key]!=b[-1][key] for key in ('action',*RAW_KEYS))
        assert {mode:file['last_action'] for mode,file in modes.items()}==contract['actions']
        results[witness]=dict(modes=modes,first_actual_split_paid=delta,same_acquired_prefix_through=delta-1)
    write_once(work/'receipt.json',dict(status='passed',source_commit=indexed['head'],witnesses=results,
        actor_code_sha256=sha(HERE/'cold_run.py'),regression_code_sha256=sha(Path(__file__)),
        protocol_sha256=sha(PROTOCOL),files_sha256={str(path):sha(path) for path in sorted(work.iterdir())}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
