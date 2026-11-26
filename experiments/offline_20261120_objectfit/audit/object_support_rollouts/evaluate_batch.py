#!/usr/bin/env python3
"""Audit eight object-support policy rollouts using unchanged native metrics."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
ROOT=STUDY/'rollouts/object_support'
BASE=STUDY.parent
B=BASE/'dual_template_exploration_20261119'
B2=BASE/'dual_template_repair_20261119'
V1=BASE/'dual_template_geometry_20261120'
EVALUATOR=B/'audit/evaluate_run.py'
ARMS=('G','S','S_no_fit','S_no_measured_consumption')
LAYOUTS=('shelf_left','shelf_right')
spec=importlib.util.spec_from_file_location('frozen_sampling_evaluation_helpers',V1/'audit/sampling_rollouts/evaluate_batch.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
helpers.STUDY=STUDY;helpers.ROOT=ROOT
verify_freeze=helpers.verify_freeze
compare=helpers.compare
read=helpers.read
sha=helpers.sha
digest=helpers.digest
write=helpers.write

def audit_run(layout,arm,output,protocol_sha):
    attempt=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
    r=read(attempt/'result.json');e=read(output/'result.json')
    assert r['status']=='completed_execution' and e['status']=='completed_independent_surface_evaluation'
    assert e['source_artifact_manifest_sha256']==sha(attempt/'artifact_manifest.json')
    for name,h in read(attempt/'artifact_manifest.json')['files_sha256'].items():assert sha(attempt/name)==h,name
    steps=read(attempt/'steps.json');choices=read(attempt/'choices.json')
    assert digest([(x['action'],x['position'],x['heading'])for x in steps])==r['trajectory_sha256']
    ar=read(attempt/'attempt.json')
    assert ar['protocol_sha256']==protocol_sha and ar['runner_sha256']==sha(B/'run.py')
    assert ar['prepared_freeze_sha256']==sha(ROOT/'layouts'/layout/'prepared/freeze.json')
    verify_freeze(layout)
    calls=read(attempt/'template_audit.json')
    frames=read(attempt/'template_frame_receipts.json');final=read(attempt/'final_template_receipt.json')
    assert final['mapper_frames']==final['registered_frames']==len(frames)==r['paid_actions']+1
    for call in calls:
        assert call['gain_implementation']=='B2_visibility_with_N20_object_support_partial_fit'
        assert call['selected_stage']=='object_support' and call['shared_observed_instance_support']
        assert call['staged_model_source_sha256']==sha(STUDY/'method/model.py')
        supports=[]
        for model in ('G','S'):
            a=call['model_audits'][model]
            assert a['applied_global_point_cap']is None and a['output_points']==a['raw_template_points']
            assert a['history_rejection'] and a['history_as_zero_weight'] and a['shared_full_quadrature']
            assert a['object_support'] and not a['full_observed_support']
            assert not any(a[k]for k in ('geometry_fit','free_space_likelihood','applicability','hidden_truth_input'))
            fit=not(model=='S'and arm=='S_no_fit')
            consumption=not(model=='S'and arm=='S_no_measured_consumption')
            assert a['fit_update']==fit and a['measured_surface_consumption']==consumption
            if not consumption:assert a['removed_as_measured_or_depth_tolerance']==0
            objects=a.get('objects',[])
            assert all(o['support_track']>0 and o['object_probability']==1 for o in objects)
            assert all(o['free_space_log_likelihood_ratio']==0 for o in objects)
            if not fit:assert all(o['consumed_geometry_log_likelihood_ratio']==0 for o in objects)
            supports.append([{key:o[key]for key in ('support_track','observed_points','fit_input_points_sha256','fit_input_normals_sha256','fit_input_labels_sha256')}for o in objects])
        assert supports[0]==supports[1]
        assert call['reused_original_gain_function']is False and set(call['source_gain_new_calls'])=={'G','S'}
    fields=['paid_actions','paid_action_budget','initial_coverage_fraction','final_coverage_fraction','free_correct_fraction',
        'false_occupied_fraction','path_length_m','turn_actions','collisions','final_pose_equals_initial_anchor','termination',
        'trajectory_sha256','global_planning_calls','first_coverage_action','first_free_correct_action','zero_gain_actions',
        'longest_zero_gain_streak','target_pose_matches']
    row={k:r[k]for k in fields}
    row.update(status='completed_independently_evaluated',surface_metrics=e['metrics'],reference_sha256=e['reference_sha256'],
        native_TSDF_mesh_sha256=e['native_TSDF_mesh_sha256'],manifest_sha256=sha(attempt/'artifact_manifest.json'),
        surface_result_sha256=sha(output/'result.json'),actual_frames=len(frames),template_score_calls=len(calls),
        plan_states_with_nonzero_S_minus_G=sum(any(abs(s-g)>1e-12 for s,g in zip(c['outputs']['scores']['S'],c['outputs']['scores']['G']))for c in choices),
        actual_template_point_cap_activations=0,max_raw_template_points=max([c['model_audits']['S']['raw_template_points']for c in calls],default=0),
        S_free_rejection_events=sum(c['model_audits']['S']['removed_by_free_space']for c in calls),
        S_measured_consumption_events=sum(c['model_audits']['S']['removed_as_measured_or_depth_tolerance']for c in calls),
        S_posterior_fit_consumed_nonzero_objects=sum(abs(o['consumed_geometry_log_likelihood_ratio'])>0 for c in calls for o in c['model_audits']['S'].get('objects',[])),
        excluded_untracked_template_groups_events=sum(len(c['model_audits']['S']['excluded_untracked_template_groups'])for c in calls),
        events_not_unique_surface_counts=True,shared_observed_support_verified=True,clean_single_switch_ablations_verified=True,
        history_zero_weight_full_basis_verified=True,measured_frame_fusion_once_verified=True,
        geometry_fit_and_free_likelihood_inactive=True,wrapper_runner_sha256=sha(ROOT/'run.py'),model_sha256=sha(STUDY/'method/model.py'))
    return row

def summarize(rows,receipts,protocol_sha,pending):
    sampling=read(V1/'audit/sampling_rollouts/summary.json');b2=read(B2/'audit/summary.json')
    paired={};historical={};table=[]
    for layout,items in rows.items():
        paired[layout]={};historical[layout]={}
        for a,b in [('G','S'),('S','S_no_fit'),('S','S_no_measured_consumption')]:
            if a in items and b in items:paired[layout][a+'_vs_'+b]=dict(difference_direction=b+' minus '+a,**compare(items[a],items[b]))
        for arm,row in items.items():
            source=sampling if arm in sampling['rows'][layout]else b2
            historical[layout][arm]=dict(reference_study='sampling'if source is sampling else'B2',
                reference_arm=arm,controls_reused=True,**compare(source['rows'][layout][arm],row))
            m=row['surface_metrics'];table.append(dict(layout=layout,arm=arm,origin='new_O_only_closed_loop',
                coverage_percent=100*row['final_coverage_fraction'],correct_free_percent=100*row['free_correct_fraction'],
                coverage90_action=row['first_coverage_action'].get('0.9'),coverage95_action=row['first_coverage_action'].get('0.95'),
                coverage99_action=row['first_coverage_action'].get('0.99'),coverage100_action=row['first_coverage_action'].get('1.0'),
                paid_actions=row['paid_actions'],path_length_m=row['path_length_m'],collisions=row['collisions'],
                F1_5cm_percent=100*m['f1_05cm'],F1_10cm_percent=100*m['f1_10cm'],
                open31_recall_5cm_percent=100*m['target_surface_metrics']['open31']['recall_05cm'],
                closed118_recall_5cm_percent=100*m['target_surface_metrics']['closed118']['recall_05cm'],template_cap_activations=0))
    failures=[r for r in receipts if r['status']!='evaluated']
    result=dict(status='partial_object_support_policy_evaluation'if pending else'completed_with_failures'if failures else'completed_independent_eight_run_evaluation',
        protocol_sha256=protocol_sha,rows=rows,paired_differences=paired,historical_same_arm_comparison=historical,
        historical_C_J_controls={layout:{arm:b2['rows'][layout][arm]for arm in ('C','J')}for layout in LAYOUTS},
        failures=failures,pending=pending,historical_summary_sha256={str(p):sha(p)for p in [V1/'audit/sampling_rollouts/summary.json',B2/'audit/summary.json']},
        evaluator_source_sha256={str(p):sha(p)for p in [Path(__file__),EVALUATOR,B/'audit/surface_metrics.py',V1/'audit/sampling_rollouts/evaluate_batch.py']},
        design=dict(new_runs_planned=8,new_runs_completed=sum(map(len,rows.values())),independent_floorplan_families=1,furniture_permutations=2,
            stochastic_repeats=0,reference_rebuilt_per_arm=False,actor_executed_by_evaluator=False),
        boundaries=['Both G and S receive the same acquired ideal object tracks; only class prior differs. G is not a no-semantic-frontend baseline.',
            'S_no_fit disables measured geometric likelihood only; S_no_measured_consumption disables measured-surface reward consumption only. Historical valid-free rejection and B2 visibility remain.',
            'O is a declared model-domain repair. OF partial geometry fitting, full-height support and free-applicability revisions are inactive.',
            'Historical controls are reused, not new independent trials. Fixed-budget reconstruction quality and time-to-coverage are separate outcomes.',
            'Ideal visible semantics, exact simulated pose and deterministic sensors do not establish natural detector, SLAM or robot performance.'])
    write(HERE/'summary.json',result)
    if table:
        with (HERE/'results.csv').open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

def main():
    assert not(HERE/'evaluation_batch_receipt.json').exists()
    protocol=read(ROOT/'protocol.json');protocol_sha=sha(ROOT/'protocol.json')
    assert tuple(protocol['arms'])==ARMS and tuple(protocol['layouts'])==LAYOUTS
    before=read(B2/'protocol.json')
    for key in ('layouts','environment'):assert protocol[key]==before[key],key
    for key in ('args','paid_action_budget','return_anchor_required','completion_area_coefficient'):assert protocol['policy'][key]==before['policy'][key],key
    for layout in LAYOUTS:verify_freeze(layout)
    pending=[(layout,arm)for layout in LAYOUTS for arm in ARMS];rows={};receipts=[]
    while pending:
        for layout,arm in pending.copy():
            attempt=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
            if(attempt/'artifact_manifest.json').exists():
                output=HERE/'results'/layout/arm/'attempt_001'
                if(output/'result.json').exists():receipt=dict(layout=layout,arm=arm,status='evaluated',returncode=0,reused_completed_independent_evaluation=True)
                else:
                    done=subprocess.run([sys.executable,str(EVALUATOR),'--attempt',str(attempt),'--reference',str(B/'audit/references'/layout),'--output',str(output)],text=True,capture_output=True)
                    receipt=dict(layout=layout,arm=arm,status='evaluated'if done.returncode==0 else'evaluation_failed',returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
                if receipt['status']=='evaluated':rows.setdefault(layout,{})[arm]=audit_run(layout,arm,output,protocol_sha)
                receipts.append(receipt);pending.remove((layout,arm));summarize(rows,receipts,protocol_sha,pending)
                print(json.dumps(dict(layout=layout,arm=arm,status=receipt['status'])),flush=True)
            elif(attempt/'failure.json').exists():
                receipts.append(dict(layout=layout,arm=arm,status='actor_failed_or_interrupted',actor_failure=read(attempt/'failure.json')))
                pending.remove((layout,arm));summarize(rows,receipts,protocol_sha,pending)
        write(HERE/'evaluation_progress.json',dict(receipts=receipts,pending=pending,actor_launched=False))
        if pending:time.sleep(5)
    assert sha(ROOT/'protocol.json')==protocol_sha
    for layout in LAYOUTS:verify_freeze(layout)
    write(HERE/'evaluation_batch_receipt.json',receipts)
    print(json.dumps(dict(status='finished',summary=str(HERE/'summary.json'))),flush=True)

if __name__=='__main__':main()
