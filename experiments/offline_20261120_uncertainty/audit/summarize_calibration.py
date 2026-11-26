#!/usr/bin/env python3
"""Compact reporting only; never changes selected scale or acceptance criteria."""
import argparse,json
from pathlib import Path
from evaluate_calibration import resource,read,write,load_resource
ROOT=Path(__file__).resolve().parents[1]

def compact(summary):
    rows={}
    for name,value in summary.items():
        score=value['informative'];interval=score['intervals']['0.95']
        rows[name]=dict(informative_crps_m=score['crps_m'],informative_median_mae_m=score['median_absolute_error_m'],
            informative_interval95_covered=interval['covered_cases'],informative_interval95_denominator=interval['case_denominator'],
            informative_interval95_width_m=interval['equal_object_mean_width_m'],
            all_condition_crps_m=value['all_conditions']['crps_m'],fallback_cases=value['fallback_cases'],
            valid_only_informative_crps_m=(value['valid_only_informative']['crps_m'] if value['valid_only_informative'] else None),
            swapped_axis_cases=value['axis_diagnostics']['swapped_axis_cases'],mean_depth_axis_error_deg=value['axis_diagnostics']['mean_error_to_true_depth_axis_deg'],
            truth_outside_original_prior_cases=value['truth_outside_original_depth_prior_cases'])
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--counterfactual',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    selected=read(a.calibration/'selected_scale.json');training=load_resource(selected['training_scores']);assessment=read(a.evaluation/'assessment.json');counter=read(a.counterfactual)
    assert assessment['selected_scale']==resource(a.calibration/'selected_scale.json')
    chosen=next(s for s in training['scales'] if s['sigma_m']==selected['sigma_m'])
    splits={}
    for split,r in assessment['split_results'].items():
        result=load_resource(r)
        splits[split]=dict(rows=compact(result['summary']),fallback_cases=result['fallback_case_count'],valid_inferred_cases=result['valid_inferred_cases'],
            informative_crps_improves_both_baselines=result['informative_crps_improves_both_baselines'],
            failure_reasons=[f['error'] for f in result['failures']],source=r,
            by_condition={m:v['by_condition'] for m,v in result['summary'].items()})
    integrity=read(ROOT/'method/validation.json')['tests']['passed']==17 and read(ROOT/'audit/validation.json')['tests_passed']==14
    ready=assessment['distribution_metric_criterion_passes'] and counter['status']=='passed' and integrity
    result=dict(schema='B8_independent_calibration_compact_summary_v1',sigma_m=selected['sigma_m'],
        selected_status=selected['status'],training_scale_scores=[dict(sigma_m=s['sigma_m'],training_equal_object_crps_m=s['training_equal_object_crps_m'])for s in training['scales']],
        training_rows=compact(chosen['summary']),training_fallback_cases=training['fallback_case_count'],splits=splits,
        counterfactual_passed=counter['status']=='passed',numerical_method_tests_passed=integrity,
        geometry_progression_criteria_pass=bool(ready),policy_has_run=False,
        next_step=('Separately frozen complete fixed-prefix and matched policy evaluation may be considered; no semantic gain demonstrated.' if ready else 'Retain all cases and failures; only old-six-prefix development diagnostics next, no new closed-loop promotion.'),
        provenance=dict(selected_scale=resource(a.calibration/'selected_scale.json'),evaluation=resource(a.evaluation/'assessment.json'),counterfactual=resource(a.counterfactual),summary_script=resource(__file__)),
        limitations=['24 procedural objects, 12/6/6 split; 96 repeated observation-condition cases, not 96 independent objects.',
          'Noiseless synthetic depth, exact poses and ideal masks. No natural detector or physical sensor accuracy claim.',
          'Targets are extents projected onto inferred axis; reported axis mismatch and fallback world-y cases remain distinct.',
          'A provisional scalar with tracking/fit fallback does not mean the complete tracking and fitting pipeline is calibrated.'])
    write(a.output,result);print(json.dumps({k:v for k,v in result.items() if k not in ('splits','provenance','training_rows')},indent=2));print(resource(a.output))
if __name__=='__main__':main()
