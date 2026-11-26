"""Read-only source/result checks and evidence-level review receipt; no experiments."""
import csv,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
R=Path('/home/ouc/nso_robot/analyses/dual_template_followup_20261120')
P=Path('/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO/docs/thesis/submission_focus_20261108')
B8=R.parent/'dual_template_uncertainty_20261120'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b):assert abs(float(a)-float(b))<1e-10,(a,b)
old=read(B8/'audit/policy_rollouts/summary.json');new=read(R/'structure/policy_audit/summary.json')
rows=list(csv.DictReader((R/'paper_update/temperature_policy_table.csv').open()))
assert len(rows)==8 and sum(x['new_run']=='True'for x in rows)==4
for row in rows:
 t=int(row['temperature']);m=row['policy'];s=(new if t==16 else old)['rows'][row['layout']][m+('_temperature'if t==16 else '_uncertainty')]
 assert int(row['paid_actions'])==s['paid_actions']==256
 assert int(row['actions_to_100'])==s['first_free_correct_action']['1.0']
 for a,b in [(row['path_m'],s['path_length_m']),(row['coverage_percent'],100*s['free_correct_fraction']),(row['F1_percent'],100*s['surface_metrics']['f1_05cm']),(row['shelf_recall_percent'],100*s['surface_metrics']['target_surface_metrics']['open31']['recall_05cm']),(row['cabinet_recall_percent'],100*s['surface_metrics']['target_surface_metrics']['closed118']['recall_05cm'])]:near(a,b)
 assert row['trajectory_sha256']==s['trajectory_sha256'] and row['native_mesh_sha256']==s['native_TSDF_mesh_sha256']
 assert int(row['collisions'])==s['collisions']==0 and row['returned']=='True'
classes=read(R/'structure/audit/test_attempt_001/test_results.json')['summary']
crows=list(csv.DictReader((R/'paper_update/structure_temperature.csv').open()))
testrows=[x for x in crows if x['split']=='test'];assert len(testrows)==20
for row in testrows:
 s=classes[row['variant']][row['mode']]
 s=s['all_conditions']if row['group']=='all_conditions'else s['by_condition'][row['group']]
 for key in ['nll','binary_brier','accuracy','high_confidence_wrong_cases','marginal_depth_crps_m','marginal_depth_median_MAE_m','fallback_cases']:near(row[key],s[key])
interval=read(R/'uncertainty/audit/test_attempt_001/summary.json')
a=interval['interval_summary']['all_conditions'];assert (a['base']['covered_cases'],a['calibrated']['covered_cases'])==(61,93)
assert (a['base']['simultaneous_covered_objects'],a['calibrated']['simultaneous_covered_objects'])==(5,21)
assert interval['fallback_cases']==8 and all('got 2' in x['error']for x in interval['failures'])
mech=read(R/'semantic/mechanism_v2_meshfix/audit_attempt_001/summary.json')
mr=list(csv.DictReader((R/'paper_update/mechanism_conditions.csv').open()));assert len(mr)==12
for row in mr:
 for key in ['object_net_m2','global_net_m2','correct_free_gain','actions','path_m']:
  diff=float(row['S_'+key])-float(row['G_'+key])
  expected=row[{'object_net_m2':'delta_object_net_m2','global_net_m2':'delta_global_net_m2','correct_free_gain':'delta_correct_free_pp','actions':'delta_actions','path_m':'delta_path_m'}[key]]
  near(100*diff if key=='correct_free_gain'else diff,expected)
claims=read(R/'semantic/mechanism_v2_meshfix/audit_attempt_001/claim_evidence_assessment.json')
assert (claims['actual_branches'],claims['actual_actions'],claims['endpoints_reached'],claims['collisions'],claims['conditions'])==(8,114,8,0,12)
# Check table copies are those generated from the audited records, before any declared formatting-only change.
tables=['temperature_policy_table.tex','structure_temperature.tex','structure_temperature_conditions.tex','mechanism_conditions.tex','interval_results.tex','interval_original_pmf.tex']
for name in tables:assert (P/name).read_bytes()==(R/'paper_update'/name).read_bytes(),name
sources=['main.tex','supplement.tex','main_intro_20261120.tex','main_problem_20261120.tex','main_joint_method_20261120.tex','main_results_20261120.tex','main_discussion_20261120.tex','structural_temperature_20261120.tex','interval_diagnostic_20261120.tex','controlled_semantic_mechanism_20261120.tex','archived_supplied_graph_20261120.tex','archived_exploration_history_20261120.tex','archived_physical_history_20261120.tex']+tables
method=(P/'main_joint_method_20261120.tex').read_text();assert 'Any positive' in method and 'selected $T=16$' in method
main=(P/'main.tex').read_text();assert 'Joint Unknown-Space Exploration and Reconstruction' in main and 'nine actions later' in main
results=(P/'main_results_20261120.tex').read_text();assert '21/24' in results and '93/96' in results and '24.1218' in results and 'S$^*$' in results
paths=[B8/'audit/policy_rollouts/summary.json',R/'structure/policy_audit/summary.json',R/'structure/policy_audit/timing_and_first_divergence.json',R/'structure/audit/test_attempt_001/test_results.json',R/'structure/audit/calibration_attempt_001/selected_temperature.json',R/'structure/replay_audit/summary.json',R/'uncertainty/audit/test_attempt_001/summary.json',R/'semantic/mechanism_v2_meshfix/audit_attempt_001/summary.json',R/'semantic/mechanism_v2_meshfix/audit_attempt_001/semantic_transfer_audit.json',R/'semantic/mechanism_v2_meshfix/audit_attempt_001/independent_binding_review.json']
review=dict(status='passed_scientific_evidence_consistency_with_explicit_limits',created_utc=datetime.now(timezone.utc).isoformat(),reviewer='next_method_policy',scope='Read-only scientific-source and reported-number audit. Reviewer contributed the problem-formulation proposal; this is not a blind review or independent experimental replication.',checks={'policy_table_all_8_rows_exact':True,'new_policy_runs':4,'old_control_rows_reused':4,'new_independent_policy_layouts':0,'class_table_all_20_test_rows_exact':True,'structural_T_train_only_selected':16,'S_correct_category_diagnostic_not_detector':True,'neutral_accuracy_unchanged_and_around_NLL_worse_retained':True,'interval_condition_and_object_denominators_distinct':True,'interval_width_and_score_cost_retained':True,'interval_not_promoted_or_applied_to_T16':True,'mechanism_8_branches_114_actions_reused_for_12_conditions':True,'mechanism_object_gain_and_global_loss_both_retained':True,'within_layout_exact_but_cross_layout_not_exact_counterfactual':True,'old_66_branches_reused_not_new_evidence':True,'main_unknown_exploration_problem':True,'supplied_graph_study_supporting_only':True,'learned_reachability_inactive_disclosed':True,'virtual_budget_and_physical_contract_distinct':True,'T1_control_distribution_and_posterior_mean_ties_retained':True,'timing_not_claimed_as_real_time':True},corrected_during_review=['Positive T preserves class ordering; only selected T=16 is described as softening confidence.','Duplicate archived correction-factorial label was namespaced; numerical values unchanged.'],conclusions={'coverage':'All four new and four matched reused entries reach100% correct-free coverage,0collisions,return,256paid actions.','T16_policy':'Left S is9actions later,+1.2m,-0.3773059173F1 percentage points,object recall tradeoff;right ties.','class':'Neutral NLL5.0803913363→0.6791130338 without accuracy improvement; S*correct cue not detection.','interval':'93/96 condition and21/24 object coverage;mean width2.3159186253m and worse interval score2.7038219956m.','mechanism':'T16right objectnet+1.3564704m2,globalnet−24.12175647m2;notuniformorfulltaskbenefit.'},remaining_scientific_limits=['Only two inspected furniture permutations of one floorplan for complete policy comparison.','Ideal partial visible instance/category labels,exact poses,noiseless depth;not detector/physical accuracy.','No general semantic exploration-efficiency or reconstruction-superiority claim supported.','No independent benefit of retaining full distribution spread over posterior mean shown.','No guarantee that physical execution is safe or accurate follows from these virtual outcomes.'],manuscript_sha256={str(P/name):sha(P/name)for name in sources},evidence_sha256={str(x):sha(x)for x in paths},formatting_changes_after_review='Any later float-placement-only edits must be separately recorded and final hashes retained; scientific content changes require re-review.')
out=R/'paper_update/independent_manuscript_review.json';out.write_text(json.dumps(review,indent=2,ensure_ascii=False)+'\n');print(out);print(sha(out))
