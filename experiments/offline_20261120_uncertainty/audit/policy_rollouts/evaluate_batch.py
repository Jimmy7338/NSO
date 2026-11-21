#!/usr/bin/env python3
"""Independently evaluate all ten B8 policy runs; never launch an actor."""
import csv,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
HERE=Path(__file__).resolve().parent;STUDY=HERE.parents[1];ROOT=STUDY/'rollouts/joint_depth'
BASE=STUDY.parent;B=BASE/'dual_template_exploration_20261119';B5=BASE/'dual_template_objectfit_20261120';V1=BASE/'dual_template_geometry_20261120'
EVALUATOR=B/'audit/evaluate_run.py'
ARMS=('G_fixed','S_fixed','G_uncertainty','S_uncertainty','S_mean_collapse');LAYOUTS=('shelf_left','shelf_right')
spec=importlib.util.spec_from_file_location('B8_frozen_native_rollout_helpers',V1/'audit/sampling_rollouts/evaluate_batch.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);h.STUDY=STUDY;h.ROOT=ROOT
verify_freeze,compare,read,sha,digest,write=h.verify_freeze,h.compare,h.read,h.sha,h.digest,h.write

def expected_mode(arm):return 'mean_collapse'if arm=='S_mean_collapse'else'fixed'if arm.endswith('_fixed')else'uncertainty'

def audit_call(call,arm):
    mode=expected_mode(arm)
    assert call['gain_implementation']=='B8_conditional_joint_particles_with_B2_visibility_and_active_only_exact_gain'
    assert call['model_source_sha256']==sha(STUDY/'method/model.py')
    assert call['selected_scale_sha256']==sha(STUDY/'audit/calibration_attempt_001/selected_scale.json')
    assert call['sigma_m']==.01 and call['shared_observed_instance_support'] and call['shared_particle_bank']
    assert call['shared_G_S_point_basis']==(mode!='mean_collapse') and not call['mean_collapse_likelihood_recomputed']
    identities=[]
    for name in ('G','S'):
        model=call['model_audits'][name]
        assert model['schema']=='B8_policy_joint_particles_v1' and model['extent_mode']==mode and model['sigma_m']==.01
        assert model['fit_update'] and model['history_rejection'] and model['measured_surface_consumption']
        assert model['applied_global_point_cap']is None and not model['old_surface_likelihood_reapplied']
        assert not model['source_geometry_consumed'] and not model['candidate_weights_stops_or_navigation_changed']
        own=[]
        for obj in model.get('objects',[]):
            assert obj['support_track']>0
            w=np.asarray(obj['joint_probability_mass']);assert w.shape==(len(obj['particles']),2)
            assert np.isfinite(w).all() and np.all(w>=0) and np.isclose(w.sum(),1,atol=1e-12,rtol=0)
            np.testing.assert_allclose(w.sum(axis=0),obj['shape_probability_mass'],atol=1e-12,rtol=0)
            assert obj['likelihood_applied'] and obj['sigma_m']==.01
            if name=='G':assert obj['shape_prior']==[.5,.5]
            if mode=='mean_collapse':assert all(not x['likelihood_recomputed']for x in obj['mean_collapse'])
            own.append({k:obj[k]for k in ('support_track','fit_input_points_sha256','fit_input_normals_sha256','fit_input_labels_sha256','particles','prior_mass','energy_m2','surface_energy_m2','free_energy_m2','conditional_geometry')})
        identities.append(own)
    assert identities[0]==identities[1],'G/S observed support, particles or likelihood bank differs'
    assert not call['reused_original_gain_function'] and set(call['source_gain_new_calls'])=={'G','S'}

def audit_run(layout,arm,output,protocol_sha):
    attempt=ROOT/'layouts'/layout/'runs'/arm/'attempt_001';r=read(attempt/'result.json');e=read(output/'result.json')
    assert r['status']=='completed_execution' and e['status']=='completed_independent_surface_evaluation'
    assert e['source_artifact_manifest_sha256']==sha(attempt/'artifact_manifest.json')
    for name,value in read(attempt/'artifact_manifest.json')['files_sha256'].items():assert sha(attempt/name)==value,name
    steps=read(attempt/'steps.json');choices=read(attempt/'choices.json')
    assert digest([(x['action'],x['position'],x['heading'])for x in steps])==r['trajectory_sha256']
    ar=read(attempt/'attempt.json');assert ar['protocol_sha256']==protocol_sha and ar['runner_sha256']==sha(B/'run.py')
    assert ar['prepared_freeze_sha256']==sha(ROOT/'layouts'/layout/'prepared/freeze.json');verify_freeze(layout)
    calls=read(attempt/'template_audit.json');frames=read(attempt/'template_frame_receipts.json');final=read(attempt/'final_template_receipt.json')
    assert final['mapper_frames']==final['registered_frames']==len(frames)==r['paid_actions']+1
    for call in calls:audit_call(call,arm)
    fields=['paid_actions','paid_action_budget','initial_coverage_fraction','final_coverage_fraction','free_correct_fraction','false_occupied_fraction','path_length_m','turn_actions','collisions','final_pose_equals_initial_anchor','termination','trajectory_sha256','global_planning_calls','first_coverage_action','first_free_correct_action','zero_gain_actions','longest_zero_gain_streak','target_pose_matches']
    row={k:r[k]for k in fields}
    row.update(status='completed_independently_evaluated',extent_mode=expected_mode(arm),surface_metrics=e['metrics'],reference_sha256=e['reference_sha256'],
        native_TSDF_mesh_sha256=e['native_TSDF_mesh_sha256'],manifest_sha256=sha(attempt/'artifact_manifest.json'),surface_result_sha256=sha(output/'result.json'),
        actual_frames=len(frames),template_score_calls=len(calls),
        plan_states_with_nonzero_S_minus_G=sum(any(abs(s-g)>1e-12 for s,g in zip(c['outputs']['scores']['S'],c['outputs']['scores']['G']))for c in choices),
        max_raw_template_points=max([c['model_audits']['S']['raw_template_points']for c in calls],default=0),
        actual_template_point_cap_activations=0,shared_observed_support_and_bank_verified=True,likelihood_applied_once_verified=True,measured_frame_fusion_once_verified=True,
        wrapper_runner_sha256=sha(ROOT/'run.py'),model_sha256=sha(STUDY/'method/model.py'),sigma_m=.01)
    return row

def paired(x,y):
    out=compare(x,y)
    out['first_coverage_action_delta']={k:(y['first_coverage_action'].get(k)-x['first_coverage_action'].get(k)if x['first_coverage_action'].get(k)is not None and y['first_coverage_action'].get(k)is not None else None)for k in ('0.9','0.95','0.99','1.0')}
    out['collisions_delta']=y['collisions']-x['collisions'];return out

def summarize(rows,receipts,protocol_sha,pending):
    old=read(B5/'audit/object_support_rollouts/summary.json');pairs={};history={};table=[]
    for layout,items in rows.items():
        pairs[layout]={};history[layout]={}
        for a,b in [('G_fixed','S_fixed'),('G_uncertainty','S_uncertainty'),('G_fixed','G_uncertainty'),('S_fixed','S_uncertainty'),('S_mean_collapse','S_uncertainty')]:
            if a in items and b in items:pairs[layout][b+'_minus_'+a]=dict(difference_direction=b+' minus '+a,**paired(items[a],items[b]))
        for arm,row in items.items():
            name=arm[0];history[layout][arm]=dict(old_arm=name,role='Historical O differs in geometry and likelihood; not a matched ablation or new repeat',**paired(old['rows'][layout][name],row))
            m=row['surface_metrics'];table.append(dict(layout=layout,arm=arm,coverage_percent=100*row['final_coverage_fraction'],correct_free_percent=100*row['free_correct_fraction'],
                coverage90_action=row['first_coverage_action'].get('0.9'),coverage95_action=row['first_coverage_action'].get('0.95'),coverage99_action=row['first_coverage_action'].get('0.99'),coverage100_action=row['first_coverage_action'].get('1.0'),
                paid_actions=row['paid_actions'],path_length_m=row['path_length_m'],collisions=row['collisions'],returned=row['final_pose_equals_initial_anchor'],F1_5cm_percent=100*m['f1_05cm'],F1_10cm_percent=100*m['f1_10cm'],
                open31_recall_5cm_percent=100*m['target_surface_metrics']['open31']['recall_05cm'],closed118_recall_5cm_percent=100*m['target_surface_metrics']['closed118']['recall_05cm']))
    failures=[r for r in receipts if r['status']!='evaluated']
    result=dict(status='partial_five_arm_policy_evaluation'if pending else'completed_with_failures'if failures else'completed_independent_ten_run_evaluation',
        protocol_sha256=protocol_sha,rows=rows,paired_differences=pairs,historical_O_reference=history,failures=failures,pending=pending,
        old_O_summary_sha256=sha(B5/'audit/object_support_rollouts/summary.json'),
        evaluator_source_sha256={str(p):sha(p)for p in [Path(__file__),EVALUATOR,B/'audit/surface_metrics.py',V1/'audit/sampling_rollouts/evaluate_batch.py']},
        design=dict(new_runs_planned=10,new_runs_completed=sum(map(len,rows.values())),independent_floorplan_families=1,furniture_permutations=2,stochastic_repeats=0,actor_executed_by_evaluator=False),
        boundaries=['Five matched B8 arms use the same sensor frontend, candidates, common scores, budget and native mapper. Only class prior and declared extent representation vary.',
            'G has neutral structure prior and shares ideal acquired instance support; it is not a semantics-free perception baseline.',
            'Fixed retains acquired-span and position conditions. Mean-collapse preserves S posterior structure masses without refitting likelihood.',
            'All geometry/source truth is confined to evaluation; no current result changes actor settings or selected sigma.',
            'Time-to-coverage and fixed-budget reconstruction are separate outcomes. All negative changes, failures and exact trajectory/mesh ties are retained.',
            'Two previously inspected furniture permutations remain dependent development layouts, not independent held-out floorplans; no robot or detector accuracy claim.'])
    write(HERE/'summary.json',result)
    if table:
        with(HERE/'results.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)

def main():
    assert not(HERE/'evaluation_batch_receipt.json').exists(),'Finalized evaluation cannot be overwritten'
    protocol=read(ROOT/'protocol.json');protocol_sha=sha(ROOT/'protocol.json');source=read(B/'protocol.json')
    assert tuple(protocol['arms'])==ARMS and tuple(protocol['layouts'])==LAYOUTS
    assert protocol['environment']==source['environment']
    for k in ('args','paid_action_budget','return_anchor_required','completion_area_coefficient'):assert protocol['policy'][k]==source['policy'][k],k
    for layout in LAYOUTS:verify_freeze(layout)
    freeze=read(HERE/'freeze.json')
    for p,s in freeze['files_sha256'].items():assert sha(p)==s,p
    pending=[(l,a)for l in LAYOUTS for a in ARMS];rows={};receipts=[]
    while pending:
        for layout,arm in pending.copy():
            attempt=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
            if(attempt/'artifact_manifest.json').exists():
                output=HERE/'results'/layout/arm/'attempt_001'
                try:
                    if(output/'result.json').exists():done=None
                    else:done=subprocess.run([sys.executable,str(EVALUATOR),'--attempt',str(attempt),'--reference',str(B/'audit/references'/layout),'--output',str(output)],text=True,capture_output=True)
                    if done is not None and done.returncode:raise RuntimeError(done.stderr)
                    rows.setdefault(layout,{})[arm]=audit_run(layout,arm,output,protocol_sha)
                    receipt=dict(layout=layout,arm=arm,status='evaluated')
                except Exception as exc:receipt=dict(layout=layout,arm=arm,status='evaluation_failed',error=repr(exc),traceback=traceback.format_exc())
                receipts.append(receipt);pending.remove((layout,arm));summarize(rows,receipts,protocol_sha,pending);print(json.dumps(receipt),flush=True)
            elif(attempt/'failure.json').exists():
                receipts.append(dict(layout=layout,arm=arm,status='actor_failed_or_interrupted',actor_failure=read(attempt/'failure.json')));pending.remove((layout,arm));summarize(rows,receipts,protocol_sha,pending)
        write(HERE/'evaluation_progress.json',dict(receipts=receipts,pending=pending,actor_launched=False))
        if pending:time.sleep(10)
    assert sha(ROOT/'protocol.json')==protocol_sha
    for layout in LAYOUTS:verify_freeze(layout)
    for p,s in freeze['files_sha256'].items():assert sha(p)==s,p
    write(HERE/'evaluation_batch_receipt.json',receipts);print(json.dumps(dict(status='finished',summary=str(HERE/'summary.json'))),flush=True)
if __name__=='__main__':main()
