"""Read-only terminal inventory; independent reconstruction metrics live elsewhere."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
B11=HERE.parents[1]/'paired_unseen_gain_20261120'
LAYOUTS=('chain_a','cycle_a','chain_b','cycle_b')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resource(p):return dict(path=str(p.resolve()),sha256=sha(p))
def main():
    output=HERE/'runtime_completion.json';assert not output.exists()
    batch=read(HERE/'batches/attempt_001/status.json')
    assert batch['status']=='finished' and not batch['running'] and not batch['pending']
    frozen=read(HERE/'freeze.json')
    for p,h in frozen['files_sha256'].items():assert sha(Path(p))==h,p
    rows=[];all_attempts=[]
    for layout in LAYOUTS:
        folders=sorted((HERE/'layouts'/layout/'runs/G_visibility').glob('attempt_*'))
        for folder in folders:
            failure=read(folder/'failure.json')if(folder/'failure.json').exists()else None
            all_attempts.append(dict(layout=layout,attempt=str(folder),failure=failure,
                finalized=(folder/'B11_execution_binding.json').exists()))
        # No implicit replacement of a partial/failed first attempt.
        folder=HERE/'layouts'/layout/'runs/G_visibility/attempt_001'
        if (folder/'failure.json').exists():
            rows.append(dict(layout=layout,status='failed',failure=read(folder/'failure.json')));continue
        r=read(folder/'result.json');manifest=read(folder/'artifact_manifest.json')
        assert r['status']==manifest['status']=='completed_execution'
        assert 'B11_execution_binding.json'in manifest['files_sha256']
        for n,h in manifest['files_sha256'].items():assert sha(folder/n)==h,n
        binding=read(folder/'B11_execution_binding.json')
        assert binding['extension_freeze']==resource(HERE/'freeze.json')
        assert binding['selected_score_mode']=='G' and binding['predictor']=='visibility'
        assert r['source_and_adapter_hashes_unchanged']
        assert r['effective_args']['cpu_score_mode']=='G'
        calls=read(folder/'template_audit.json');steps=read(folder/'steps.json')
        assert len(steps)==r['paid_actions']
        assert all(c['arm']=='G_visibility'and c['B11_predictor']=='visibility'and c['original_candidate_provider']and not c['full_policy_cost_band_applied']for c in calls)
        choices=read(folder/'choices.json')
        assert all(c['outputs']['mode']=='G'for c in choices)
        baseline=B11/'execution/layouts'/layout/'runs/S_visibility/attempt_001'
        s=read(baseline/'result.json');old_steps=read(baseline/'steps.json')
        first_diff=next((i+1 for i,(a,b)in enumerate(zip(steps,old_steps))if(a['action'],a['position'],a['heading'])!=(b['action'],b['position'],b['heading'])),None)
        rows.append(dict(layout=layout,status=r['status'],arm=r['arm'],paid_actions=r['paid_actions'],
            budget=r['paid_action_budget'],correct_free_fraction=r['free_correct_fraction'],
            first_correct_free_action=r['first_free_correct_action'],path_length_m=r['path_length_m'],
            collisions=r['collisions'],returned=r['final_pose_equals_initial_anchor'],
            elapsed_s=r['elapsed_s'],decision_wall_s_total=r['decision_wall_s_total'],
            max_exported_action_decision_s=max((x['decision_wall_s']for x in steps),default=0),
            global_planning_calls=r['global_planning_calls'],all_selected_score_modes_G=True,
            all_original_candidates_retained=True,acquisition_engine_used=False,
            same_trajectory_as_S=r['trajectory_sha256']==s['trajectory_sha256'],
            first_trajectory_difference_action_vs_S=first_diff,
            same_native_mesh_as_S=sha(folder/'final_tsdf_mesh.ply')==sha(baseline/'final_tsdf_mesh.ply'),
            baseline_S=dict(correct_free_fraction=s['free_correct_fraction'],first_correct_free_action=s['first_free_correct_action'],path_length_m=s['path_length_m']),
            manifest=resource(folder/'artifact_manifest.json'),baseline_result=resource(baseline/'result.json')))
    receipt=dict(status='completed_four_run_runtime_inventory',rows=rows,all_attempts=all_attempts,
        extension_freeze=resource(HERE/'freeze.json'),frozen_resources_verified=len(frozen['files_sha256']),
        batch=resource(HERE/'batches/attempt_001/status.json'),batch_elapsed_s=batch['elapsed_s'],
        successful_runs=sum(r['status']=='completed_execution'for r in rows),
        failure_count=sum(x['failure']is not None for x in all_attempts),
        source=resource(Path(__file__)),timing_scope='Shared host with up to three G actors plus separately coordinated frontier actors; numerical-library thread limits one, not isolated real-time evidence.',
        native_metric_evaluation='Delegated to independent root evaluator; this file inventories execution and hashes only.')
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),sha256=sha(output),successful_runs=receipt['successful_runs'],failures=receipt['failure_count'])))
if __name__=='__main__':main()
