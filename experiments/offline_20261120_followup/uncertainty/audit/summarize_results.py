#!/usr/bin/env python3
"""Post-evaluation concise evidence index; no fitting or new truth access."""
import hashlib,json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def resource(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
a=ROOT/'audit';s=read(a/'test_attempt_001/summary.json');c=read(a/'calibration_attempt_001/calibrated_config.json');cr=read(a/'calibration_attempt_001/calibration_rows.json');receipt=read(a/'calibration_attempt_001/calibration_receipt.json')
assert s['status']=='completed_once_frozen_test_evaluation'
assert len(s['opened_truth_paths'])==24 and len(receipt['opened_truth_paths'])==24
assert not set(s['opened_truth_paths'])&set(receipt['opened_truth_paths'])
assert s['blocked_paths']==[] and receipt['blocked_paths']==[]
assert read(a/'counterfactual_check.json')['status']=='passed'
summary=dict(status='completed_independent_report_only_interval_diagnostic',source=resource(a/'test_attempt_001/summary.json'),
 calibration=resource(a/'calibration_attempt_001/calibration_receipt.json'),calibrated_config=resource(a/'calibration_attempt_001/calibrated_config.json'),
 generator=resource(ROOT/'generator/dataset_manifest.json'),inference_index=resource(ROOT/'inference/inference_index.json'),
 evaluator_freeze=resource(a/'freeze.json'),counterfactual=resource(a/'counterfactual_check.json'),
 units=dict(new_independent_objects=48,calibration_objects=24,test_objects=24,dependent_conditions_per_object=4,primary_cases=192,primary_frames=576,extra_counterfactual_independent_units=0),
 sigma_m=.01,s0_m=c['s0_m'],q=c['q'],order_statistic_rank=c['order_statistic_rank'],
 calibration_fallback_cases=len(cr['failures']),test_fallback_cases=s['fallback_cases'],
 calibration_failure_reasons=dict(Counter(r['error']for r in cr['failures'])),test_failure_reasons=dict(Counter(r['error']for r in s['failures'])),
 interval_test=s['interval_summary'],original_distribution_test={k:{'all_conditions':v['all_conditions'],'by_condition':v['by_condition'],'axis_diagnostics':v['axis_diagnostics'],'fallback_cases':v['fallback_cases'],'truth_outside_original_depth_prior_cases':v['truth_outside_original_depth_prior_cases']}for k,v in s['original_distribution_summary'].items()},
 maximum_calibration_score_object=max(c['per_object'],key=lambda o:o['score']),
 new_test_used_for_fit_or_selection=False,original_PMF_changed=False,policy_or_navigation_changed=False,
 limitations=['Clean first-hit rendering with exact poses and ideal visible instance/category masks; external validity to noisy real scenes is untested.',
 'Targets are actual extents along the inferred axis, with the public world-y fallback; neither axis nor track is selected against truth.',
 'The four conditions of one object are dependent; coverage guarantee concerns the object maximum under the same exchangeable frozen geometry mixture.',
 'Observed test object coverage 21/24 is below the nominal 95% target; the finite-sample marginal statement does not guarantee a particular realized test fraction.',
 'The interval correction widens sets substantially and worsens mean interval score. It is not a new PMF or evidence of better geometry estimates, planning, or robot performance.'],
 summarizer=resource(__file__))
write(a/'summary.json',summary)
write(a/'claim_evidence_assessment.json',dict(status='complete_mixed_interval_result_not_policy_gain',
 supported=['Original neutral posterior test CRPS is better than prior-only and fixed-0.65 on this independent frozen object mixture.',
 'Report-only object-group correction increases observed per-condition and all-four-condition coverage.',
 'All input failures remain in the denominator and original PMFs are unchanged.'],
 unsupported=['Tight or accurately calibrated 95% uncertainty in each condition/class.',
 'A realized 95% all-four-condition test coverage rate.',
 'Improved point estimates, posterior CRPS, semantic planning, or navigation from expanding the interval.'],
 adverse=['Mean width rises from 0.2780m to 2.3159m.','Mean interval score worsens from 2.1397m to 2.7038m.','Three test objects still miss the corrected interval in front-only observations.'],
 source=resource(a/'summary.json')))
print(json.dumps(resource(a/'summary.json')))
