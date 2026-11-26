"""Derive B9 scale and undercoverage audit from frozen B8 TRAINING only."""
from pathlib import Path
import hashlib
import json
import numpy as np
from conformal import fit_training_scale

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/ouc/nso_robot/analyses/dual_template_uncertainty_20261120/audit/calibration_attempt_001/training_scores.json')
EXPECTED = '348fa29021ffec2cac5fa10fa169db6d26953252ea55e2f975fbd8f9560acd2f'


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    data = json.loads(SOURCE.read_text())
    assert data['split'] == 'train'
    selected = next(row for row in data['scales'] if row['sigma_m'] == .01)
    inputs, evidence = [], []
    for row in selected['rows']['posterior']:
        metrics = row['metrics']; interval = metrics['intervals']['0.95']
        inputs.append(dict(object_id=row['object_id'], condition=row['condition'],
                           target_extent_m=row['target']['projected_true_extent_m'],
                           median_m=metrics['median_m'], lower_m=interval['lower_m'],
                           upper_m=interval['upper_m'], fallback=row['evaluator_prior_fallback']))
        evidence.append(dict(object_id=row['object_id'], case_id=row['case_id'],
                             condition=row['condition'], status=row['inference_status']['status'],
                             evaluator_prior_fallback=row['evaluator_prior_fallback'],
                             base_95_interval=interval, median_absolute_error_m=metrics['median_absolute_error_m'],
                             best_horizontal_axis_error_deg=row['target']['best_horizontal_axis_error_deg'],
                             target_outside_particle_support=metrics['truth_outside_particle_support']))
    fitted = fit_training_scale(inputs)
    assert fitted['objects'] == 12 and fitted['cases'] == 48
    fitted.update(source=dict(path=str(SOURCE), sha256=EXPECTED),
                  raw_training_scalar_inputs=inputs, opened_truth_splits=['B8_train'])
    grouped = {}
    for status in sorted({r['status'] for r in evidence}):
        rows = [r for r in evidence if r['status'] == status]
        grouped[status] = dict(cases=len(rows), zero_width_95_cases=sum(r['base_95_interval']['width_m'] == 0 for r in rows),
                               covered_95_cases=sum(r['base_95_interval']['covered'] for r in rows),
                               mean_median_absolute_error_m=float(np.mean([r['median_absolute_error_m'] for r in rows])),
                               mean_best_horizontal_axis_error_deg=float(np.mean([r['best_horizontal_axis_error_deg'] for r in rows])))
    diagnostic = dict(status='completed_training_only_undercoverage_diagnostic',
                      source=dict(path=str(SOURCE), sha256=EXPECTED), sigma_m=.01,
                      objects=12, condition_cases=48, by_inference_status=grouped,
                      cases=evidence, opened_truth_splits=['B8_train'],
                      interpretation='Two credible faces collapse depth to a point despite nonzero finite-resolution/model residual. Two additional grid posteriors also have zero-width 95% intervals. This audit does not calibrate or alter planning probabilities.',
                      old_validation_or_heldout_used_for_parameter_selection=False)
    for name, value in [('training_fit.json', fitted), ('training_undercoverage_diagnostic.json', diagnostic)]:
        path = HERE / name
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(objects=12, cases=48, s0_m=fitted['s0_m'], raw_median_scale_m=fitted['raw_median_scale_m'], training_only=True)))


if __name__ == '__main__':
    main()
