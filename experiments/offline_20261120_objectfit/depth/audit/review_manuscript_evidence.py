#!/usr/bin/env python3
"""Final numeric-table and claim evidence audit, without modifying manuscripts."""
import json,hashlib,re
from pathlib import Path
BASE=Path('/home/ouc/nso_robot/analyses')
PAPER=Path('/home/ouc/nso_robot/workspaces/nso_research_20261029/NSO/docs/thesis/submission_focus_20261108')
OUT=BASE/'dual_template_depth_20261120/audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def numeric_rows(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if '&' not in line or not line.rstrip().endswith('\\\\'):continue
        cells=[s.strip() for s in line.rstrip()[:-2].split('&')]
        try:values=[float(s) for s in cells[2:]]
        except ValueError:continue
        rows.append(values)
    return rows
b5p=BASE/'dual_template_objectfit_20261120/audit/summary.json';b6p=BASE/'dual_template_parts_20261120/audit/summary.json';o8p=BASE/'dual_template_objectfit_20261120/audit/object_support_rollouts/summary.json';b7p=BASE/'dual_template_depth_20261120/audit/summary.json'
b5=read(b5p);b6=read(b6p);o8=read(o8p);b7=read(b7p)
expected=[]
for source,variants in [(b5,['object_support','object_full_support','object_fit']),(b6,['outer_orientation','observed_layers','outer_and_layers'])]:
    for variant in variants:
        for arm in ['G','S']:
            a=source['equal_state_weight_aggregate'][variant]['models'][arm];m=a['mean_candidate'];s=a['selected_means']
            vals=[m['true_object_supported_weight_m2'],m['realized_true_object_weight_m2'],a['template_rank_rho_vs_object_net'],s['actual_object_net_new_m2'],s['actual_global_net_new_m2'],s['completed_actions']]
            expected.append([float(f'{x:.3f}') for x in vals[:-1]]+[float(f'{vals[-1]:.1f}')])
actual=numeric_rows(PAPER/'object_parts_diagnostic_table_20261120.tex');assert actual==expected,(actual,expected)
expected=[]
for layout in ['shelf_left','shelf_right']:
    for arm in ['G','S','S_no_fit','S_no_measured_consumption']:
        r=o8['rows'][layout][arm];s=r['surface_metrics'];t=s['target_surface_metrics']
        vals=[r['first_free_correct_action']['1.0'],r['path_length_m'],100*s['f1_05cm'],100*s['f1_10cm'],100*t['open31']['recall_05cm'],100*t['closed118']['recall_05cm']]
        expected.append([float(vals[0]),float(f'{vals[1]:.1f}')]+[float(f'{x:.2f}') for x in vals[2:]])
        assert r['paid_actions']==256 and r['collisions']==0 and r['final_pose_equals_initial_anchor'] and r['free_correct_fraction']==1
actual=numeric_rows(PAPER/'object_support_policy_table_20261120.tex');assert actual==expected,(actual,expected)
for layout in ['shelf_left','shelf_right']:
    for key in ['trajectory_sha256','native_TSDF_mesh_sha256']:
        assert o8['rows'][layout]['S'][key]==o8['rows'][layout]['S_no_fit'][key]
        if layout=='shelf_right':assert o8['rows'][layout]['S'][key]==o8['rows'][layout]['G'][key]
assert b6['equal_state_weight_aggregate']['outer_and_layers']['models']['S']['template_rank_rho_vs_object_net']<b5['equal_state_weight_aggregate']['object_fit']['models']['S']['template_rank_rho_vs_object_net']
assert b7['completed_states']==6 and b7['object_states']==12 and b7['profiled_object_states']==4 and b7['proposed_depth_updates']==0
files=['main.tex','supplement.tex','observed_object_parts_20261120.tex','object_parts_diagnostic_table_20261120.tex','object_support_policy_table_20261120.tex']
record={
 'schema':'NSO_final_B5_B6_B7_manuscript_evidence_review_v1',
 'status':'pass_no_blocking_numeric_or_claim_error_in_reviewed_additions',
 'scope':'Primary main abstract and updated experiments/conclusion, supplement input and new observed-object section, 12-row diagnostic table and 8-row O policy table. Older unrelated studies and references not re-audited in this pass; PDF layout verification owned by publication agent.',
 'reviewed_files_sha256':{f:sha(PAPER/f) for f in files},
 'evidence_sha256':{str(p):sha(p) for p in [b5p,b6p,o8p,b7p,BASE/'dual_template_objectfit_20261120/audit/object_support_rollouts/left_first_divergence.json']},
 'checks':[
  'All 72 numeric cells in 12-row O/O_full/OF/A/B/AB diagnostic table exactly match declared display rounding of independent equal-state summaries.',
  'All 48 numeric cells in 8-row O-policy table exactly match declared display rounding; 256 actions, full correct-free coverage, zero simulated collisions and return verified for every run.',
  'Left S: full coverage 97 vs G 88, path +1.2m, F1 -0.377pp, shelf +1.294pp and cabinet -2.834pp correctly retained; right G/S exact trajectory/native-mesh tie retained.',
  'No-fit wording limited to no observed policy-level effect; actual S/no-fit trajectory/native-mesh hashes equal in both layouts. No general ineffectiveness claim.',
  'Measured-consumption ablation described as mixed, with free rejection retained, and not conflated with the older cap/history intervention.',
  'AB realized-weight improvement .0489 to .0729 is separated from rank decline .800 to .791; both remain below O and no B6 closed-loop promotion is claimed.',
  'B7 four curves prefer lower depth but all retain .65m under fixed heuristic band; eight object states lack unique anchors. No identified true size or no-information theorem claimed.',
  'Same six prefixes/66 reused branches explicitly remain dependent development diagnostics; O eight runs are four policies on two furniture permutations, not eight independent environments.',
  'G/S share ideal acquired instance support; G correctly described as neutral structural-class control, not a detector-free or wholly semantics-free baseline.',
  'Hypothetical weighted areas distinguished from unique measured physical area; reference geometry remains evaluator-only and no natural-detector/physical accuracy benefit claimed.'
 ],
 'required_corrections':[],
 'nonblocking_scope_caution':'Current positive primary-paper controlled-observation findings and null/adverse unknown-exploration findings remain distinct implementations and evidence levels. Do not present this review as submission readiness or general NSO efficiency validation.'
}
p=OUT/'final_manuscript_evidence_review.json';p.write_text(json.dumps(record,indent=2)+'\n');print(str(p));print(sha(p))
