#!/usr/bin/env python3
"""Summarize a successful sealed replay without rerunning any actor."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'attempt_001'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def main():
    result=read(OUT/'result.json')
    assert result['status']=='completed_fixed_action_and_target_replay'
    points=read(OUT/'checkpoints.json');scores=read(OUT/'score_checks.json')
    score_by_action={r['paid_action']:r for r in scores}
    rows=[];csvrows=[]
    for previous,current in zip(points[:-1],points[1:]):
        start=previous['paid_action'];end=current['paid_action'];plan=score_by_action[start]
        score=plan['selected_score_audit'];planned=len(plan['selected_outbound_actions'])
        row={'start_action':start,'end_action':end,'acquired_actions':end-start,
            'first_selected_candidate':plan['selected_candidate_id'],
            'first_selected_outbound_actions':planned,
            'interval_exactly_one_selected_outbound_option':planned==end-start,
            'old_predicted_template_reward_m2':score['template_expected_visible_area_m2'],
            'old_common_total_proxy':score['common_total_proxy'],
            'old_coverage_only_proxy':score['coverage_only_proxy'],
            'old_measured_quality_total_proxy':score['quality_total_proxy'],'actual_reference_changes':{}}
        for key in ('all','open31','closed118'):
            old=previous['groups'][key];new=current['groups'][key]
            row['actual_reference_changes'][key]={
                'fixed_reference_area_estimate_m2':new['fixed_reference_visible_area_estimate_m2'],
                'new_actual_pose_continuous_FOV_visible_area_estimate_m2':new['new_continuous_FOV_visible_area_estimate_m2'],
                '05cm':new['05cm']['since_previous_checkpoint'],
                '10cm':new['10cm']['since_previous_checkpoint'],
                'recall_05cm_before':old['05cm']['recall'],'recall_05cm_after':new['05cm']['recall']}
            delta=new['05cm']['since_previous_checkpoint']
            csvrows.append({'start_action':start,'end_action':end,'reference_scope':key,
                'original_predicted_template_reward_S_m2':score['template_expected_visible_area_m2']['S'],
                'prediction_scope_same_as_reference_scope':False,
                'exact_outbound_interval':planned==end-start,
                'gross_new_correct_area_estimate_5cm_m2':delta['new_correct_area_estimate_m2'],
                'lost_correct_area_estimate_5cm_m2':delta['lost_correct_area_estimate_m2'],
                'net_correct_area_estimate_5cm_m2':delta['net_correct_area_estimate_m2'],
                'recall_delta_5cm_percentage_points':delta['recall_delta_percentage_points'],
                'new_actual_pose_continuous_FOV_visible_area_estimate_m2':new['new_continuous_FOV_visible_area_estimate_m2']})
        rows.append(row)
    with (HERE/'intervals.csv').open('w')as f:
        writer=csv.DictWriter(f,fieldnames=list(csvrows[0]));writer.writeheader();writer.writerows(csvrows)
    summary={'status':'completed_fixed_replay_and_independent_increment_evaluation',
        'source_replay_result':str(OUT/'result.json'),'replay_receipt':result,
        'intervals':rows,'all_input_checks_passed':True,
        'numeric_definitions':{
            'correct_reference_sample':'Distance from fixed source reference point to native measured TSDF triangles <= tolerance.',
            'new_correct':'False at interval start, true at end; includes distance improvement into tolerance, not only newly allocated TSDF triangles.',
            'lost_correct':'True at start, false at end; retained rather than forcing monotonicity.',
            'net':'End minus start; equals gross new minus lost.',
            'area':'Uniform area sample count times source sampling area/drawn count from the pre-existing reference receipt; Monte Carlo estimate, not exact surface integration.',
            'visibility':'Whole-source first-hit continuous FOV visibility at actual acquired poses; not a pixel-resolution observation or a TSDF surface metric.',
            'prediction_comparison':'Template reward covers all inferred measured groups; full-scene and target reference increments have different support. These side-by-side values are not a calibrated prediction residual or module-level causal effect.',
            'last_interval':'208→256 contains additional original replanning; the 208 prediction applies to only its two-action selected option, not the full final interval.'},
        'repair_hook_status':{
            'role':'Provisional counterfactual only; no repair was installed into the replay actor.',
            'original_gain_mirror':'Elementwise equality checked against unchanged frozen source at every traced viewpoint.',
            'repair_source_version':'Pre-boundary-revision implementation imported after action-0 checkpoint; exact full source SHA was not captured.',
            'known_loaded_near_surface_test':'abs(z - measured) <= tolerance',
            'subsequent_disk_revision':'Inclusive interval z >= measured - tolerance and z <= measured + tolerance; also diagnostics/install guards added later.',
            'do_not_attribute_to_current_disk_repair_sha':True,
            'effect_on_original_replay_or_TSDF':False,
            'provisional_outputs':str(OUT/'gain_hook_checks.json')},
        'artifacts_sha256':{str(p):sha(p) for p in [HERE/'protocol.json',HERE/'run_fixed_replay.py',OUT/'result.json',OUT/'checkpoints.json',OUT/'score_checks.json',OUT/'gain_hook_checks.json',HERE/'intervals.csv']},
        'native_mesh_retention':'Eight local checkpoint PLY files; not included in paper publication bundle.',
        'same_domain_prediction_validation':{'available':False,
            'missing':'Predicted point coordinates and per-point route union-gain vectors were not retained by either frozen hook.',
            'consequence':'Cannot match rewarded hypothesis samples to independently valid source surfaces that become newly correct in TSDF; no calibration residual or causal surface attribution is reported.',
            'no_third_replay':True},
        'hardware':False,'new_policy_episodes':0}
    versioned=HERE/'attempt_002/result.json'
    if versioned.exists():
        v=read(versioned);assert v['status']=='completed_versioned_fixed_state_hook_replay'
        h=read(HERE/'attempt_002/gain_hook_checks.json');freeze=read(HERE/'attempt_002/freeze.json')
        for path,digest in freeze['snapshot_sha256'].items():assert sha(path)==digest
        assert v['repair_source_sha256']=='57c5cd1bb58a2e816d81184014cbfbad2a3d5ddeb872c43c7b5d9895378bb2fb'
        summary['authoritative_current_repair_diagnostic']={
            'separate_execution_from_actual_geometry_replay':True,'result':v,
            'freeze_sha256':sha(HERE/'attempt_002/freeze.json'),
            'hooks_sha256':sha(HERE/'attempt_002/gain_hook_checks.json'),
            'states':[{'paid_action':x['paid_action'],
                'models':{name:{k:model[k] for k in ['original_route_area_m2','repaired_route_area_m2','recovered_union_area_m2','original_gain_elementwise_identical','unique_outbound_views']}
                          for name,model in x['models'].items()}}for x in h],
            'interpretation':'Counterfactual scoring of the same recorded selected routes at fixed measured states; no repaired route was executed and no performance advantage follows.',
            'old_provisional_hook_numbers_used_as_current_version_evidence':False}
        summary['repair_hook_status']['superseded_for_current_repair_quantitative_claims_by']=str(versioned)
        summary['artifacts_sha256'][str(versioned)]=sha(versioned)
    matched=HERE/'attempt_003/prediction_matching.json'
    if matched.exists():
        matching=read(matched);capture=read(HERE/'attempt_003/result.json')
        assert matching['status']=='completed_same_domain_weighted_prediction_matching'
        assert capture['status']=='completed_same_domain_point_capture_replay'
        summary['same_domain_prediction_validation_before_new_capture']=summary['same_domain_prediction_validation']
        summary['same_domain_prediction_validation']={
            'available':True,'independent_evaluation_result':str(matched),'sha256':sha(matched),
            'new_capture_result':str(HERE/'attempt_003/result.json'),
            'capture_actions_replayed':109,'capture_fused_frames':110,'fixed_states':[54,109],
            'threshold_m':.05,'source_truth_to_actor':False,
            'same_prediction_points_used_for_old_repair_delta':True,
            'weights_are_template_hypothesis_area_not_independent_real_area':True,
            'source_geometry_and_before_after_TSDF_point_match':True,
            'nearest_actual_source_point_secondary_check_retained':True,
            'separate_capture_execution_and_geometry_replay_disclosed':True,
            'repair_trajectory_performance_claim':False}
        summary['artifacts_sha256'][str(matched)]=sha(matched)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for row in rows:
        if row['start_action']in(54,109):
            print(json.dumps({'interval':[row['start_action'],row['end_action']],
                'predicted':row['old_predicted_template_reward_m2'],
                'actual':{k:v['05cm'] for k,v in row['actual_reference_changes'].items()}},indent=2))

if __name__=='__main__':main()
