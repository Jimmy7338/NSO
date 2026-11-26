#!/usr/bin/env python3
"""First-frame NSO episode with separate global and local switches."""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time
import traceback

from regression import import_file, read, sha, write


ROOT=Path(__file__).resolve().parent
WEIGHTED=ROOT.parent/'weighted_geometric_control_20261122'
BASE_PROTOCOL=ROOT.parent/'same_goal_route_choice_20261121'/'PROTOCOL.json'
ARM_MODES={'original_G':('G',False),'region_G':('G',True),
           'original_S':('S',False),'region_S':('S',True),'F_strict':(None,False)}
INITIAL_KEYS=('layout','scene_id','anchor','initial_packet_sha256','initial_owner_sha256','stop_budget')


def run(args, frozen):
    sys.path.insert(0,str(WEIGHTED/'runner'))
    inherited=import_file('original_weighted_actor',WEIGHTED/'runner/weighted_run.py')
    native=inherited.native
    original_read,original_sha=native.read,native.sha
    original_setup,original_loader=native.setup_runtime,native.load_source
    original_adapter=native.local_route_adapter
    job=next(row for row in frozen['jobs'] if row['layout']==args.layout and row['arm']==args.arm)
    layout,arm=args.layout,args.arm
    prior,region=ARM_MODES[arm]
    assert job['prior']==prior and job['regional_enabled']==region
    scene_root=Path(frozen['scene_root'])
    prepared=scene_root/'prepared'/layout/'initial_state.json'
    cold=read(prepared)
    budget=frozen['budget']
    assert cold['paid_actions']==cold['policy_select_calls']==0 and cold['stop_budget']==budget
    assert cold['actual_config']['max_steps']==budget and cold['scene_id']==f'same-goal-holdout-{layout}'
    initial={key:cold[key] for key in INITIAL_KEYS}
    assert sha(prepared)==frozen['actor_files'][str(prepared)]
    protocol=read(BASE_PROTOCOL)
    protocol['heldout'].update(layouts=list(dict.fromkeys(r['layout'] for r in frozen['jobs'])),paid_action_budget=budget)
    protocol['diagnostic_checkpoints']=frozen['capture_paid'][1:]
    for key,value in frozen['local_route_invariants'].items():
        assert protocol['local_route'][key]==value
    weights=inherited.Weights(**frozen['local_auxiliary_weights'])
    raw=[]
    hooks=[]
    writer=None
    existing_writer=None
    out=args.out
    source_manifest=out/'.native_manifest.json'
    pseudo_seal=ROOT/'holdout_freeze.json'
    started=time.monotonic()

    def actor_read(path):
        if Path(path)==prepared:
            return dict(initial)
        if Path(path)==pseudo_seal:
            return {'world_files_sha256':frozen['actor_initial_files']}
        return original_read(path)

    def actor_sha(path):
        return original_sha(args.freeze if Path(path)==pseudo_seal else path)

    def actor_setup(stage,name,policy,settings):
        nonlocal writer,existing_writer
        result=original_setup(stage,name,policy,settings)
        old,(world,_,first,components,runtime,_,_),initial_read,global_mode,local_mode=result
        assert (stage,name,policy,global_mode,local_mode)==('holdout',layout,arm,'G',prior)
        assert initial_read==initial and world.step_count==0 and world.seed==cold['seed']
        raw.append(inherited.raw_sensor_hashes(first,native.array_sha))
        original_packet=old.packet

        def capture_packet(*packet_args,**packet_kwargs):
            packet=original_packet(*packet_args,**packet_kwargs)
            assert packet.action_id==len(raw)
            raw.append(inherited.raw_sensor_hashes(packet,native.array_sha))
            return packet

        old.packet=capture_packet
        if region:
            from policy.observed_regions import RegionalPlanner
            from regional_adapter import install
            planner=RegionalPlanner()
            hooks.extend(install(components,runtime,planner))
        import runtime_binding
        writer=runtime_binding.write
        existing_writer=runtime_binding
        runtime_binding.write=lambda path,data:writer(source_manifest if Path(path)==out/'manifest.json' else path,data)
        return result

    def actor_loader(name,path):
        module=original_loader(name,path)
        if Path(path)==native.METHOD:
            legacy=module.choose_route

            def choose(state,native_option,mode,route_union_area,config):
                assert mode==prior and mode in ('G','S')
                return inherited.choose_route(module,legacy,state,native_option,mode,weights,
                                              route_union_area,config,existing_writer.revision.reweight)

            module.choose_route=choose
        return module

    native.ARM_MODES={arm:('G',prior)}
    native.HERE,native.HOLDOUT,native.PROTOCOL=ROOT,scene_root,ROOT/frozen['protocol_relative']
    native.read,native.sha=actor_read,actor_sha
    native.load_source,native.setup_runtime=actor_loader,actor_setup
    native.local_route_adapter=lambda *a:inherited.full_local_route_adapter(*a,original_adapter)
    try:
        native.run('holdout',layout,arm,protocol,out)
    finally:
        if existing_writer is not None:
            existing_writer.write=writer
    result=read(source_manifest)
    source_manifest.unlink()
    rows=read(out/'steps.json')
    assert len(rows)==result['paid_actions']==len(raw)-1<=budget
    assert raw[0]['packet_sha256']==result['initial_packet_sha256']
    for row,sensor in zip(rows,raw[1:],strict=True):
        assert row['step']==sensor['step'] and row['packet_sha256']==sensor['packet_sha256']
        row.update({key:value for key,value in sensor.items() if key.endswith('values_sha256') or key=='scan_ranges_sha256'})
    writer(out/'steps.json',rows)
    writer(out/'initial_raw_rgbd_scan_sha256.json',raw[0])
    for paid in frozen['capture_paid'][1:]:
        key=f'paid_{paid}'
        if key not in result['checkpoints']:
            assert paid>result['paid_actions']
            result['checkpoints'][key]=inherited.missing(out,key,paid,result['paid_actions'],'native_stop_before_checkpoint')
    choices=read(out/'local_choices.json')
    if prior:
        assert all(e['full_pool']['enabled'] and not e['full_pool']['unassessed_after_cap'] for e in choices)
    else:
        assert not choices
    region_log,region_guards=hooks if hooks else ([],[])
    writer(out/'regional_choices.json',region_log)
    writer(out/'regional_guards.json',region_guards)
    result.update(regional_enabled=region,study_prior=prior,**asdict(weights),native_budget=budget,
                  source_protocol_sha256=sha(ROOT/frozen['protocol_relative']),runner_freeze_sha256=sha(args.freeze),
                  cold_started_at_paid_zero=True,actual_scene_seed=cold['seed'],
                  initial_owner_sha256=cold['initial_owner_sha256'],initial_grid_resolution_m=cold['grid_resolution_m'],
                  frozen_cold_input_sha256=sha(prepared),closed_loop_full_pool=prior is not None,
                  all_distinct_local_candidates_assessed=prior is not None,
                  charged_decision_wall_s=sum(row['decision_wall_s'] for row in rows),
                  bootstrap_and_task_wall_s=time.monotonic()-started,
                  files_sha256={path.relative_to(out).as_posix():sha(path) for path in sorted(out.rglob('*'))
                                if path.is_file() and path.name!='manifest.json'})
    writer(out/'manifest.json',result)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('layout')
    cli.add_argument('arm',choices=tuple(ARM_MODES))
    cli.add_argument('--freeze',required=True,type=Path)
    cli.add_argument('--out',required=True,type=Path)
    args=cli.parse_args()
    args.out,args.freeze=args.out.resolve(),args.freeze.resolve()
    frozen=read(args.freeze)
    scope=frozen['actor_files']
    assert all(sha(path)==digest for path,digest in scope.items())
    assert not any('evaluator_references' in k or 'evaluator_truth' in k for k in scope)
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        run(args,frozen)
    except BaseException as error:
        write(args.out/'failure.json',dict(layout=args.layout,arm=args.arm,status='technical_failure',
              error=repr(error),traceback=traceback.format_exc(),runner_freeze_sha256=sha(args.freeze)))
        raise


if __name__=='__main__':
    main()
