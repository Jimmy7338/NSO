#!/usr/bin/env python3
"""Wait for the 12 declared artifacts and evaluate each once; no actor launch."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()

def main():
    protocol=read(ROOT/'protocol.json');protocol_sha=sha(ROOT/'protocol.json')
    pending=[(layout,arm)for layout in protocol['layouts']for arm in protocol['arms']]
    receipts=[];start=time.monotonic()
    # Only one declared attempt is expected. Failed/interrupted attempts remain
    # failures; this observer neither retries nor creates replacement actors.
    while pending:
        progressed=False
        for layout,arm in pending.copy():
            attempt=ROOT/'layouts'/layout/'runs'/arm/'attempt_001'
            if (attempt/'artifact_manifest.json').exists():
                output=HERE/'results'/layout/arm/'attempt_001'
                command=[sys.executable,str(HERE/'evaluate_run.py'),'--attempt',str(attempt),
                         '--reference',str(HERE/'references'/layout),'--output',str(output)]
                done=subprocess.run(command,text=True,capture_output=True,check=False)
                receipt={'layout':layout,'arm':arm,'status':'evaluated'if done.returncode==0 else'evaluation_failed',
                    'returncode':done.returncode,'attempt':str(attempt),'output':str(output),
                    'stdout':done.stdout,'stderr':done.stderr}
                receipts.append(receipt);pending.remove((layout,arm));progressed=True
                print(json.dumps({'layout':layout,'arm':arm,'status':receipt['status']}),flush=True)
            elif (attempt/'failure.json').exists():
                receipts.append({'layout':layout,'arm':arm,'status':'actor_failed_or_interrupted',
                    'attempt':str(attempt),'actor_failure':read(attempt/'failure.json')})
                pending.remove((layout,arm));progressed=True
                print(json.dumps({'layout':layout,'arm':arm,'status':'actor_failed_or_interrupted'}),flush=True)
        write(HERE/'evaluation_progress.json',{'completed_slots':len(receipts),'pending':pending,'receipts':receipts,
            'elapsed_s':time.monotonic()-start,'actor_launched':False})
        if pending:time.sleep(30)
    assert sha(ROOT/'protocol.json')==protocol_sha
    rows={};failures=[]
    for receipt in receipts:
        layout,arm=receipt['layout'],receipt['arm'];rows.setdefault(layout,{})
        if receipt['status']!='evaluated':
            rows[layout][arm]={'status':receipt['status']};failures.append(receipt);continue
        attempt=Path(receipt['attempt']);r=read(attempt/'result.json');e=read(Path(receipt['output'])/'result.json')
        steps=read(attempt/'steps.json');choices=read(attempt/'choices.json')
        trace=[(s['action'],s['position'],s['heading'])for s in steps]
        assert digest(trace)==r['trajectory_sha256']
        frozen=read(ROOT/'layouts'/layout/'prepared/freeze.json')
        ar=read(attempt/'attempt.json')
        assert ar['prepared_freeze_sha256']==sha(ROOT/'layouts'/layout/'prepared/freeze.json')
        assert ar['protocol_sha256']==protocol_sha
        assert ar['runner_sha256']==sha(ROOT/'run.py')
        template=read(attempt/'template_audit.json');frames=read(attempt/'template_frame_receipts.json')
        final=read(attempt/'final_template_receipt.json')
        assert final['mapper_frames']==final['registered_frames']==r['paid_actions']+1==len(frames)
        meaningful_calls=[c for c in template if c['model_audits']['S']['raw_template_points']>0]
        if arm=='S_no_fit':
            assert all(obj['consumed_geometry_log_likelihood_ratio']==0 for c in meaningful_calls for obj in c['model_audits']['S']['objects'])
        if arm=='S_no_history':
            assert all(c['model_audits']['S']['removed_by_free_space']==0 and c['model_audits']['S']['removed_as_measured_or_depth_tolerance']==0 for c in meaningful_calls)
        nonzero=sum(any(abs(s-g)>1e-12 for s,g in zip(c['outputs']['scores']['S'],c['outputs']['scores']['G']))for c in choices)
        row={k:r[k]for k in ['paid_actions','paid_action_budget','initial_coverage_fraction','final_coverage_fraction',
            'free_correct_fraction','false_occupied_fraction','path_length_m','collisions','final_pose_equals_initial_anchor',
            'termination','trajectory_sha256','global_planning_calls','first_coverage_action','first_free_correct_action']}
        row.update(status='completed_independently_evaluated',surface_metrics=e['metrics'],
            native_TSDF_mesh_sha256=e['native_TSDF_mesh_sha256'],reference_sha256=e['reference_sha256'],
            manifest_sha256=sha(attempt/'artifact_manifest.json'),surface_result_sha256=sha(Path(receipt['output'])/'result.json'),
            actual_frames=len(frames),template_score_calls=len(template),calls_with_template_hypotheses=len(meaningful_calls),
            plan_states_with_nonzero_S_minus_G=nonzero,
            template_point_cap_activations=sum(c['model_audits']['S']['points_after_history']>3500 for c in template),
            measured_frame_fusion_once_verified=True,mechanism_ablation_receipts_verified=True)
        rows[layout][arm]=row
    comparisons={}
    for layout,items in rows.items():
        pairs={}
        for a,b in [('G','S'),('J','G'),('J','S'),('C','S'),('S','S_no_fit'),('S','S_no_history')]:
            if 'surface_metrics'not in items[a]or'surface_metrics'not in items[b]:continue
            x,y=items[a],items[b]
            pairs[a+'_vs_'+b]={'difference_direction':b+' minus '+a,
                'coverage_delta_percentage_points':100*(y['final_coverage_fraction']-x['final_coverage_fraction']),
                'F1_5cm_delta_percentage_points':100*(y['surface_metrics']['f1_05cm']-x['surface_metrics']['f1_05cm']),
                'open31_completeness_5cm_delta_percentage_points':100*(y['surface_metrics']['target_surface_metrics']['open31']['recall_05cm']-x['surface_metrics']['target_surface_metrics']['open31']['recall_05cm']),
                'closed118_completeness_5cm_delta_percentage_points':100*(y['surface_metrics']['target_surface_metrics']['closed118']['recall_05cm']-x['surface_metrics']['target_surface_metrics']['closed118']['recall_05cm']),
                'paid_action_delta':y['paid_actions']-x['paid_actions'],'path_length_delta_m':y['path_length_m']-x['path_length_m'],
                'trajectory_identical':x['trajectory_sha256']==y['trajectory_sha256'],
                'native_TSDF_mesh_identical':x['native_TSDF_mesh_sha256']==y['native_TSDF_mesh_sha256']}
        comparisons[layout]=pairs
    summary={'status':'completed_with_failures'if failures else'completed_independent_12_slot_evaluation',
        'protocol_sha256':protocol_sha,'design':{'controlled_floorplan_families':1,'furniture_permutations':2,
            'arms_per_layout':6,'declared_runs':12,'independent_natural_scenes':0,'stochastic_repeats':0},
        'rows':rows,'paired_differences':comparisons,'failures':failures,
        'all_references_frozen_before_run_result_reads':True,'reference_rebuilt_per_arm':False,
        'actor_reexecution_or_replacement_by_evaluator':False,'registration_performed':False,
        'evaluator_source_sha256':{str(p):sha(p)for p in [Path(__file__),HERE/'evaluate_run.py',HERE/'surface_metrics.py',HERE/'prepare_surface_reference.py']},
        'limits':['Two paired permutations of one authored three-branch layout are not independent natural environments.',
            'Ideal current visible categories and common measured instance support do not establish detector performance.',
            'The original V3 model integrated with V10 coverage/candidates is a disclosed new variant, not unchanged original policy.',
            'Posterior is an uncalibrated model-discrepancy pseudo-posterior; template dimension mismatch is preserved.',
            'S_no_history jointly removes already-observed-surface consumption and free-space rejection.',
            'No real-robot motion, SLAM, trained reachability or natural-noise robustness is tested.'],
        'elapsed_s':time.monotonic()-start}
    write(HERE/'evaluation_batch_receipt.json',receipts);write(HERE/'summary.json',summary)
    print(json.dumps({'summary':str(HERE/'summary.json'),'failed_slots':len(failures),'elapsed_s':summary['elapsed_s']}),flush=True)

if __name__=='__main__':main()
