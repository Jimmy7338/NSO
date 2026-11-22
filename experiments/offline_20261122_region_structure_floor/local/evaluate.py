#!/usr/bin/env python3
"""Verify native executions and measure the frozen common-clock shelf contrasts."""

import argparse
import csv
import importlib.util
import math
import os
from pathlib import Path

for var in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[var]='1'

import numpy as np

from run import ANALYSES, ARMS, FREEZE, HERE, PROTOCOL, SCENES, read, sha, write


PRIOR = ANALYSES/'route_intervention_20261122/evaluation/evaluate.py'
binding = importlib.util.spec_from_file_location('nso_controlled_local_measurement_binding',PRIOR)
checks=importlib.util.module_from_spec(binding)
binding.loader.exec_module(checks)
FIXTURES = {'open31':'shelf','closed118':'cabinet'}


def episode(layout,arm,initial, weighted):
    directory = HERE/'runs'/layout/arm
    if (directory/'failure.json').exists():
        failure=directory/'failure.json'
        return dict(status='technical_failure',error=read(failure)['error'],receipt_sha256=sha(failure))
    receipt = directory/'manifest.json'
    if not receipt.exists():
        return {'status':'pending'}
    manifest, digest=weighted.engine.verify_manifest(directory,'manifest.json')
    assert manifest['frozen_actor_files_sha256']==sha(FREEZE)==read(directory/'attempt.json')['runner_freeze_sha256']
    assert manifest['layout']==layout and manifest['experimental_local_arm']==arm and not manifest['evaluator_inputs_read_by_actor']
    assert manifest['native_budget']==192 and manifest['actual_closed_loop'] and not manifest['forced_route_or_goal']
    trace=read(directory/'steps.json')
    weighted.native.verify_trace(trace,initial,192,manifest)
    weighted.native.verify_action_guard(trace,192)
    assert len(trace)==manifest['paid_actions']
    with np.load(directory/'observed_belief_trace.npz',allow_pickle=False) as recorded:
        assert np.array_equal(recorded['paid'],np.arange(len(trace)+1))
        belief=recorded['belief']
    trigger=read(directory/'mechanism_trigger.json')
    choices=read(directory/'local_choices.json')
    applied=[choice for choice in choices if 'single_direct_inspection_mechanism_arm' in choice['choice_audit']]
    if trigger['status']=='triggered':
        assert len(applied)==1
        choice,=applied
        audit=choice['choice_audit']
        assert trigger['paid_before']==choice['paid_before'] and trigger['paid_before']<=64
        assert trigger['selection_call_id']==choice['selection_call_id'] and trigger['target_pose']==choice['same_global_target']
        assert audit['single_direct_inspection_mechanism_arm']==arm and audit['selected_route_id']==trigger['actual_route_id']
        assert trigger['actual_route_id']==trigger['decision_table'][arm]
        assert trigger['candidate_pool_sha256']==audit['candidate_pool_sha256']
        assert audit['admitted_candidate_count']>=2 and trigger['inspection_area_range_m2']>1e-9
        assert choice['native_return_actions']==choice['rebound_return_actions']
        probes=read(directory/'trigger_probes.json')
        assert all(other['direct_area_range_m2']<=1e-9 for other in probes[:-1])
        assert probes[-1]['selection_call_id']==trigger['selection_call_id']
    else:
        assert not applied and 'fork' not in manifest['checkpoints']
    assert all(choice['full_pool']['unassessed_after_cap']==0 for choice in choices)
    assert all(choice['choice_audit']['weighted_ranking']['prior']=='G' for choice in choices if choice not in applied)
    return dict(status='verified_native_episode',manifest=manifest,receipt_sha256=digest,directory=directory,trace=trace,
                initial=initial,belief=belief,trigger=trigger,raw_initial=read(directory/'initial_raw_rgbd_scan_sha256.json'))


def first_diff(a,b):
    for left,right in zip(a,b):
        assert left['step']==right['step']
        if any(left[field]!=right[field] for field in ('action','position','heading','packet_sha256',*checks.RAW_FIELDS)):
            return dict(paid=left['step'],actions=[left['action'],right['action']],
                        poses=[[le['position'],le['heading']] for le in (left,right)])
    return None


def available(run,name):
    rec=run['manifest']['checkpoints'].get(name,{})
    return rec.get('status')=='captured'


def layout_result(layout, plan, weighted, metric):
    initial=read(SCENES/'prepared'/layout/'initial_state.json')
    reference=weighted.load_reference(layout,initial)
    records={arm:episode(layout,arm,initial,weighted) for arm in ARMS}
    raw=[episode for episode in records.values() if episode['status']=='verified_native_episode']
    if not raw:
        return {key:val for key,val in records.items()}, {}, {}
    assert all(item['raw_initial']==raw[0]['raw_initial'] for item in raw)
    stages={}
    b0={}
    common_paid=plan['paired_paid']
    for arm,run in records.items():
        if run['status']!='verified_native_episode':
            b0[arm]=run
            continue
        manifest,trigger=run['manifest'],run['trigger']
        measurements={}
        for name,paid in (('start',0),('fork',trigger.get('paid_before')),('common',common_paid),('terminal',manifest['paid_actions'])):
            checkpoint=f'paid_{paid}' if name=='common' else name
            if paid is None or not available(run,checkpoint):
                continue
            native_metrics,masks=checks.verify_measurement(weighted,metric,reference,run,checkpoint,paid)
            measurements[name]=(native_metrics,masks)
        stages[arm]=measurements
        b0[arm]=dict(status='verified_native_episode',trigger=trigger,steps_path=str(run['directory']/'steps.json'),
                     manifest_sha256=run['receipt_sha256'],paid=manifest['paid_actions'],returned=manifest['returned'],
                     collisions=manifest['collisions'],route_event=manifest['route_event'],path_length_m=manifest['path_length_m'],
                     actual_terminal=measurements['terminal'][0]['receipt']['actual_paid'],
                     whole_scene_clock={name:{key:row.get(key) for key in ('C_free','F1_scene_05cm','J')}
                                        for name,(row,_) in measurements.items()},
                     meshes={name:{'path':row['mesh_path'],'sha256':row['mesh_sha256']} for name,(row,_) in measurements.items()})
    paired_ready = bool(raw) and all(arm['trigger']['status']=='triggered' for arm in raw)
    if raw and len(raw)==4 and not paired_ready:
        assert all(arm['trigger']['status']=='not_triggered' for arm in raw)
    export={'reference_shelf_xyz_m':reference['samples']['shelf'],
            'reference_shelf_area_per_sample_m2':np.array(reference['weights']['shelf'])}
    comparisons={}
    if not paired_ready:
        return b0, comparisons, export
    chosen=list(stages)
    first=raw[0]
    clock=first['trigger']['paid_before']
    mf,base=stages[chosen[0]]['fork']
    for arm in chosen:
        other=records[arm]
        assert clock==other['trigger']['paid_before']
        assert all(first['trigger'][key]==other['trigger'][key]
                   for key in ('candidate_pool_sha256','decision_table','target_pose','selection_call_id'))
        assert first_diff(first['trace'][:clock],other['trace'][:clock]) is None
        assert np.array_equal(first['belief'][:clock+1],other['belief'][:clock+1])
        rm,rb=stages[arm]['fork']
        assert mf['mesh_sha256']==rm['mesh_sha256'] and np.array_equal(base['shelf'],rb['shelf'])
        for phase,(measured,masks) in stages[arm].items():
            export[f'{arm}_{phase}_shelf']=masks['shelf']
            if phase in ('common','terminal'):
                b0[arm][phase]={key:checks.facility_metrics(base[val],masks[val],reference['weights'][val])
                                for key,val in FIXTURES.items()}

    for left,right in (('S','D'),('S','G'),('S','X'),('D','G')):
        if left not in stages or right not in stages:
            continue
        a,b=records[left],records[right]
        metrics=dict(left=left,right=right,first_actual_difference=first_diff(a['trace'],b['trace']),
                     same_chosen_route=a['trigger']['actual_outbound_sha256']==b['trigger']['actual_outbound_sha256'],
                     fork_paid=clock,common_paid=common_paid)
        for phase in ('common','terminal'):
            if phase not in stages[left] or phase not in stages[right]:
                metrics[phase]={'status':'missing_actual_checkpoint'}
                continue
            _,la=stages[left][phase]
            _,ra=stages[right][phase]
            groups={}
            for key,rkey in FIXTURES.items():
                lnew=la[rkey]&~base[rkey]
                rnew=ra[rkey]&~base[rkey]
                weight=reference['weights'][rkey]
                row=dict(common_new_m2=int((lnew&rnew).sum())*weight,
                         left_only_new_m2=int((lnew&~rnew).sum())*weight,
                         right_only_new_m2=int((rnew&~lnew).sum())*weight,
                         gross_delta_m2=(int(lnew.sum())-int(rnew.sum()))*weight,
                         net_delta_m2=b0[left][phase][key]['net_m2']-b0[right][phase][key]['net_m2'])
                if phase=='common':
                    lfinal=stages[left]['terminal'][1][rkey]
                    rfinal=stages[right]['terminal'][1][rkey]
                    row['retention_m2']={}
                    for cohort,selected in (('left_only',lnew&~rnew),('right_only',rnew&~lnew)):
                        mutually_exclusive={'left':lfinal&~rfinal,'right':rfinal&~lfinal,'both':lfinal&rfinal,'neither':~(lfinal|rfinal)}
                        row['retention_m2'][cohort]={label:int((selected&mask).sum())*weight
                                                      for label,mask in mutually_exclusive.items()}
                        assert math.isclose(sum(row['retention_m2'][cohort].values()),int(selected.sum())*weight,abs_tol=1e-12)
                groups[key]=row
            metrics[phase]=groups
        comparisons[f'{left}_minus_{right}']=metrics
    return b0,comparisons,export


def output_csv(path, rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'results')
    parser.add_argument('--layout')
    args=parser.parse_args()
    plan=read(PROTOCOL)
    freeze=read(FREEZE)
    if args.layout and args.layout not in plan['placements']:
        parser.error('Only frozen layouts may be evaluated')
    for path,digest in freeze['evaluator_files_sha256'].items():
        assert sha(path)==digest
    weighted,metric=checks.load_evaluator(read(checks.DEFAULT_PROTOCOL))
    weighted.SCENES=SCENES
    records,branches,contrasts={},{},{}
    args.out.mkdir(parents=True,exist_ok=True)
    arm_rows,pair_rows=[],[]
    for layout in plan['placements'] if args.layout is None else (args.layout,):
        arms, comparisons, masks=layout_result(layout,plan,weighted,metric)
        branches[layout]=arms
        contrasts[layout]=comparisons
        if masks:
            path=args.out/f'{layout}_masks.npz'
            np.savez_compressed(path,**masks)
            records[layout]={'path':str(path),'sha256':sha(path)}
        for name,arm in arms.items():
            row=dict(layout=layout,arm=name,status=arm['status'])
            for key in ('paid','returned','collisions','path_length_m','manifest_sha256'):
                row[key]=arm.get(key)
            if ('trigger') in arm:
                row.update(fork_paid=arm['trigger'].get('paid_before'),route=arm['trigger'].get('actual_route_id'),
                           trigger=arm['trigger']['status'],route_event=arm['route_event']['status'],
                           route_event_paid=arm['route_event'].get('actual_paid'))
            for phase in ('common','terminal'):
                stats=arm.get(phase,{}).get('open31',{})
                for key in ('gross_new_m2','lost_m2','net_m2'):
                    row[f'{phase}_shelf_{key}']=stats.get(key)
                row[f'{phase}_J']=arm.get('whole_scene_clock',{}).get(phase,{}).get('J')
            arm_rows.append(row)
        for name,comp in comparisons.items():
            row=dict(layout=layout,pair=name,fork_paid=comp['fork_paid'],common_paid=comp['common_paid'],
                     same_chosen_route=comp['same_chosen_route'],first_actual_difference_paid=(comp['first_actual_difference'] or {}).get('paid'))
            for phase in ('common','terminal'):
                for field,item in comp[phase].get('open31',{}).items():
                    row[f'{phase}_shelf_{field}']=item
            pair_rows.append(row)
    output_csv(args.out/'arms.csv',arm_rows)
    output_csv(args.out/'paired.csv',pair_rows)
    summary=dict(status='offline_frozen_measured_run_audit',freeze_sha256=sha(FREEZE),metric_reference='same independent native O3D 5cm matcher',
                 budget=plan['native_budget'],common_paid=plan['paired_paid'],arms=branches,comparisons=contrasts,surface_masks=records)
    write(args.out/'measured.json',summary)
    print('Measured',sum(arm['status']=='verified_native_episode' for arm in arm_rows),'native episodes;',len(pair_rows),'paired contrasts.',flush=True)


if __name__=='__main__':
    main()
