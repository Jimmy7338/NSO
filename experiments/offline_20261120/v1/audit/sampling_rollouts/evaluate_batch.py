#!/usr/bin/env python3
"""Independently evaluate six clean-consumption rollouts; never launch actors."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
ROOT=STUDY/'rollouts/sampling'
B=STUDY.parent/'dual_template_exploration_20261119'
B2=STUDY.parent/'dual_template_repair_20261119'
EVALUATOR=B/'audit/evaluate_run.py'
ARMS=('G','S','S_no_measured_consumption')
LAYOUTS=('shelf_left','shelf_right')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def verify_freeze(layout):
    prepared=ROOT/'layouts'/layout/'prepared'
    frozen=read(prepared/'freeze.json')
    for path,h in frozen['code_sha256'].items():assert sha(path)==h,path
    parent=B/'layouts'/layout/'prepared'
    assert sha(parent/'freeze.json')==frozen['parent_freeze_sha256']
    assert sha(prepared/'initial_state.json')==frozen['initial_state_sha256']
    assert read(prepared/'initial_state.json')==read(parent/'initial_state.json')
    assert sha(B/'audit/references'/layout/'RECEIPT.json')==frozen['reference_receipt_sha256']
    old=read(parent/'freeze.json')
    for key in ('code_sha256','geometry_cache_sha256'):
        for path,h in old[key].items():assert sha(path)==h,path
    source=STUDY.parent/'reproduction_20261104/simulation_publication_stage/experiments/unknown_exploration_20261104/source'
    assets=STUDY.parent/'semantic_scene_redesign_20261119/assets/replicacad'
    for path,h in old['frozen_scientific_source_sha256'].items():assert sha(source/path)==h,path
    for path,h in old['source_asset_sha256'].items():assert sha(assets/path)==h,path
    for path,h in old['prepared_files_sha256'].items():assert sha(parent/path)==h,path

def audit_run(layout,arm,output,protocol_sha):
    attempt=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
    r=read(attempt/'result.json');e=read(output/'result.json')
    assert r['status']=='completed_execution'
    assert e['status']=='completed_independent_surface_evaluation'
    assert e['source_artifact_manifest_sha256']==sha(attempt/'artifact_manifest.json')
    for name,h in read(attempt/'artifact_manifest.json')['files_sha256'].items():assert sha(attempt/name)==h,name
    steps=read(attempt/'steps.json');choices=read(attempt/'choices.json')
    assert digest([(x['action'],x['position'],x['heading'])for x in steps])==r['trajectory_sha256']
    ar=read(attempt/'attempt.json')
    assert ar['protocol_sha256']==protocol_sha
    assert ar['runner_sha256']==sha(B/'run.py')
    assert ar['prepared_freeze_sha256']==sha(ROOT/'layouts'/layout/'prepared/freeze.json')
    verify_freeze(layout)
    calls=read(attempt/'template_audit.json')
    frames=read(attempt/'template_frame_receipts.json');final=read(attempt/'final_template_receipt.json')
    assert final['mapper_frames']==final['registered_frames']==len(frames)==r['paid_actions']+1
    for call in calls:
        assert call['gain_implementation']=='B2_visibility_with_N20_staged_geometry'
        assert call['selected_stage']=='sampling'
        assert call['staged_model_source_sha256']==sha(STUDY/'method/model.py')
        for model in ('G','S'):
            a=call['model_audits'][model]
            assert a['applied_global_point_cap'] is None
            assert a['output_points']==a['raw_template_points']
            assert a['history_rejection'] and a['history_as_zero_weight'] and a['shared_full_quadrature']
            assert not a['geometry_fit'] and not a['free_space_likelihood'] and not a['hidden_truth_input']
            expected=not(model=='S'and arm=='S_no_measured_consumption')
            assert a['measured_surface_consumption']==expected
            if not expected:assert a['removed_as_measured_or_depth_tolerance']==0
        # The staged wrapper records gain-call counts and source binding,
        # but does not export the old B2 aggregate occlusion event counters.
        assert call['reused_original_gain_function'] is False
        assert set(call['source_gain_new_calls'])=={'G','S'}
    fields=['paid_actions','paid_action_budget','initial_coverage_fraction','final_coverage_fraction','free_correct_fraction',
        'false_occupied_fraction','path_length_m','turn_actions','collisions','final_pose_equals_initial_anchor','termination',
        'trajectory_sha256','global_planning_calls','first_coverage_action','first_free_correct_action','zero_gain_actions',
        'longest_zero_gain_streak','target_pose_matches']
    row={k:r[k]for k in fields}
    row.update(status='completed_independently_evaluated',surface_metrics=e['metrics'],reference_sha256=e['reference_sha256'],
        native_TSDF_mesh_sha256=e['native_TSDF_mesh_sha256'],manifest_sha256=sha(attempt/'artifact_manifest.json'),
        surface_result_sha256=sha(output/'result.json'),actual_frames=len(frames),template_score_calls=len(calls),
        plan_states_with_nonzero_S_minus_G=sum(any(abs(s-g)>1e-12 for s,g in zip(c['outputs']['scores']['S'],c['outputs']['scores']['G']))for c in choices),
        actual_template_point_cap_activations=0,
        S_calls_exceeding_former_cap=sum(c['model_audits']['S']['points_after_history']>3500 for c in calls),
        max_raw_template_points=max([c['model_audits']['S']['raw_template_points']for c in calls],default=0),
        S_free_rejection_events=sum(c['model_audits']['S']['removed_by_free_space']for c in calls),
        S_measured_consumption_events=sum(c['model_audits']['S']['removed_as_measured_or_depth_tolerance']for c in calls),
        events_not_unique_surface_counts=True,history_zero_weight_full_basis_verified=True,measured_frame_fusion_once_verified=True,
        no_geometry_or_free_likelihood_revision=True,method_ablation_receipts_verified=True,
        visibility_binding_verified_by_frozen_model=True,per_call_occlusion_event_counts_exported=False,
        wrapper_runner_sha256=sha(STUDY/'run.py'),model_sha256=sha(STUDY/'method/model.py'))
    return row

def compare(x,y):
    assert x['reference_sha256']==y['reference_sha256']
    return dict(coverage_delta_pp=100*(y['final_coverage_fraction']-x['final_coverage_fraction']),
        F1_5cm_delta_pp=100*(y['surface_metrics']['f1_05cm']-x['surface_metrics']['f1_05cm']),
        open31_recall_5cm_delta_pp=100*(y['surface_metrics']['target_surface_metrics']['open31']['recall_05cm']-x['surface_metrics']['target_surface_metrics']['open31']['recall_05cm']),
        closed118_recall_5cm_delta_pp=100*(y['surface_metrics']['target_surface_metrics']['closed118']['recall_05cm']-x['surface_metrics']['target_surface_metrics']['closed118']['recall_05cm']),
        paid_action_delta=y['paid_actions']-x['paid_actions'],path_length_delta_m=y['path_length_m']-x['path_length_m'],
        trajectory_identical=x['trajectory_sha256']==y['trajectory_sha256'],native_TSDF_mesh_identical=x['native_TSDF_mesh_sha256']==y['native_TSDF_mesh_sha256'])

def summarize(rows,receipts,protocol_sha,pending):
    prior=read(B2/'audit/summary.json');paired={};parent={};legacy={};table=[]
    for layout,items in rows.items():
        paired[layout]={};parent[layout]={};legacy[layout]={}
        for a,b in [('G','S'),('S','S_no_measured_consumption')]:
            if a in items and b in items:paired[layout][a+'_vs_'+b]=dict(difference_direction=b+' minus '+a,**compare(items[a],items[b]))
        for arm,row in items.items():
            old_arm=arm if arm in ('G','S')else 'S'
            parent[layout][arm]=dict(reference_arm=old_arm,**compare(prior['rows'][layout][old_arm],row))
            if arm=='S_no_measured_consumption':
                legacy[layout][arm]=dict(comparison='descriptive only: old S_no_history simultaneously disabled free rejection and changed point thinning',
                    **compare(prior['rows'][layout]['S_no_history'],row))
            m=row['surface_metrics']
            table.append(dict(layout=layout,arm=arm,origin='new_clean_consumption_rollout',coverage_percent=100*row['final_coverage_fraction'],
                coverage100_action=row['first_coverage_action'].get('1.0'),paid_actions=row['paid_actions'],path_length_m=row['path_length_m'],
                collisions=row['collisions'],F1_5cm_percent=100*m['f1_05cm'],F1_10cm_percent=100*m['f1_10cm'],
                open31_recall_5cm_percent=100*m['target_surface_metrics']['open31']['recall_05cm'],
                closed118_recall_5cm_percent=100*m['target_surface_metrics']['closed118']['recall_05cm'],
                template_cap_activations=0,former_cap_exceeding_calls=row['S_calls_exceeding_former_cap']))
    historical={layout:{arm:prior['rows'][layout][arm]for arm in ('C','J')}for layout in LAYOUTS}
    failures=[r for r in receipts if r['status']!='evaluated']
    result=dict(status=('partial_clean_consumption_evaluation'if pending else 'completed_with_failures'if failures else 'completed_independent_six_run_evaluation'),
        protocol_sha256=protocol_sha,rows=rows,paired_differences=paired,B2_same_arm_or_S_comparison=parent,
        legacy_no_history_descriptive_comparison=legacy,historical_C_J_controls=historical,failures=failures,pending=pending,
        prior_summary_sha256=sha(B2/'audit/summary.json'),evaluator_source_sha256={str(p):sha(p)for p in [Path(__file__),EVALUATOR,B/'audit/surface_metrics.py']},
        design=dict(new_runs_planned=6,new_runs_completed=sum(map(len,rows.values())),independent_floorplan_families=1,furniture_permutations=2,
            stochastic_repeats=0,old_C_J_runs_reused=4,reference_rebuilt_per_arm=False,actor_executed_by_evaluator=False),
        boundaries=['S_no_measured_consumption removes measured-surface reward consumption only. Strict valid-free-space rejection, B2 visibility, and full quadrature remain.',
            'Historical C/J are reused controls, not newly executed repetitions. No between-run uncertainty or wall-time comparison is inferred.',
            'Controlled ideal visible semantic labels and exact poses do not test natural detector, real SLAM, trained reachability or robot performance.'])
    write(HERE/'summary.json',result)
    if table:
        with (HERE/'results.csv').open('w',newline='')as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

def main():
    assert not(HERE/'evaluation_batch_receipt.json').exists(),'completed batch cannot be overwritten'
    protocol=read(ROOT/'protocol.json');protocol_sha=sha(ROOT/'protocol.json')
    assert tuple(protocol['arms'])==ARMS and tuple(protocol['layouts'])==LAYOUTS
    before=read(B2/'protocol.json')
    for key in ('layouts','environment','perception'):assert protocol[key]==before[key],key
    for key in ('args','paid_action_budget','return_anchor_required','completion_area_coefficient'):assert protocol['policy'][key]==before['policy'][key],key
    for layout in LAYOUTS:verify_freeze(layout)
    pending=[(layout,arm)for layout in LAYOUTS for arm in ARMS];rows={};receipts=[]
    while pending:
        for layout,arm in pending.copy():
            attempt=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
            if (attempt/'artifact_manifest.json').exists():
                output=HERE/'results'/layout/arm/'attempt_001'
                if(output/'result.json').exists():
                    receipt=dict(layout=layout,arm=arm,status='evaluated',returncode=0,reused_completed_independent_evaluation=True)
                else:
                    done=subprocess.run([sys.executable,str(EVALUATOR),'--attempt',str(attempt),'--reference',str(B/'audit/references'/layout),'--output',str(output)],text=True,capture_output=True)
                    receipt=dict(layout=layout,arm=arm,status='evaluated'if done.returncode==0 else'evaluation_failed',returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
                if receipt['status']=='evaluated':rows.setdefault(layout,{})[arm]=audit_run(layout,arm,output,protocol_sha)
                receipts.append(receipt);pending.remove((layout,arm))
                summarize(rows,receipts,protocol_sha,pending)
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
