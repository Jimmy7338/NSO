#!/usr/bin/env python3
"""Close V37 CPU execution with saved-record checks; never load a detector."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from nso.research_evidence_v31 import (
    freeze, read, seal, sha, verify_inventory, verify_sources, write, write_bytes)
from scripts.seal_online_completion_v35 import (
    MUTABLE, check_pack, check_input_map, manuscript_checks, require)

OUTPUT = ROOT / 'audit_results/v37_cpu_completion_20260920'
SMOKE = ROOT / 'audit_results/v37_cpu_detector_smoke_20260920'
RECOVERY = ROOT / 'audit_results/v37_lfs_cache_recovery_20260920'
PREVIOUS = ROOT / 'audit_results/v37_completion_handoff_20260920'
REPORT = 'docs/research/V37_CPU_FRONTEND_EXECUTION_RESULT_20260920.md'
PRIOR = 'docs/research/V37_REAL_EQUIPMENT_PRIOR_FEASIBILITY_20260920.md'


def records_check():
    result = read(SMOKE / 'result.json')
    require(result['status'] == 'passed_software_smoke', 'smoke did not pass')
    def rows(record):
        return sorted(record['detections'] + record['rejected_detections'], key=lambda r: r['source_index'])
    adapter_records = [read(SMOKE / 'warmup.json'), read(SMOKE / 'blank_control.json')]
    checked_priors = 0
    summary = []
    for expected in result['image_summaries']:
        data = read(SMOKE / (Path(expected['input_file']).stem + '_predictions.json'))
        repetitions = data['repetitions']
        require(len(repetitions) == 5, 'repeat count differs')
        first = repetitions[0]
        for current in repetitions:
            require(rows(first) == rows(current), 'same-input detections differ')
            require(first['configuration_priors'] == current['configuration_priors'], 'priors differ')
            checked_priors += len(current['configuration_priors'])
        reference = data['PIL_reference']
        current_rows = rows(first)
        np.testing.assert_array_equal([r['source_class_id'] for r in current_rows], reference['classes'])
        boxes = np.array([r['box_xyxy'] for r in current_rows]).reshape(-1, 4)
        scores = np.array([r['detector_score'] for r in current_rows])
        box_error = float(np.max(np.abs(boxes - np.array(reference['boxes']).reshape(-1, 4)), initial=0))
        score_error = float(np.max(np.abs(scores - np.array(reference['scores'])), initial=0))
        require(max(box_error, score_error) <= 1e-5, 'PIL comparison differs')
        milliseconds = [r['runtime']['prediction_wall_seconds'] * 1000 for r in repetitions]
        require(statistics.median(milliseconds) == expected['median_prediction_wall_ms'], 'timing median differs')
        adapter_records.extend(repetitions)
        summary.append(dict(input_file=expected['input_file'], max_box_error=box_error,
                            max_score_error=score_error, median_prediction_wall_ms=statistics.median(milliseconds)))
    for record in adapter_records:
        require(record['model_execution_performed'] is True, 'mock record inside real smoke')
        runtime = record['runtime']
        require(runtime['device'] == 'cpu' and runtime['dtype'] == 'float32'
                and runtime['torch_num_threads'] == runtime['torch_num_interop_threads'] == 1, 'runtime differs')
        for prior in record['configuration_priors']:
            require(prior['probabilities'] == [.5, .5] and prior['reason'] == 'no_prior_registry'
                    and prior['geometry_fallback'], 'unsupported semantic prior')
    require(len(adapter_records) + 2 == result['counts']['explicit_predict_calls'] == 14, 'API counts differ')
    require(not result['network_attempts'] and result['qualified_equipment_instances'] == 0, 'scope differs')
    require(result['natural_equipment_accuracy'] is None and result['natural_semantic_planning_gain'] is None,
            'unmeasured efficacy was assigned a score')
    tests = (SMOKE / 'tests.txt').read_text()
    require('Ran 5 tests' in tests and tests.rstrip().endswith('OK'), 'boundary test transcript not complete')
    return dict(saved_adapter_records=len(adapter_records), saved_PIL_references=2,
                timed_detection_priors_checked=checked_priors, per_image=summary,
                new_detector_loads=0, new_predict_calls=0, new_world_TSDF_quality_calls=0)


def run():
    require(not OUTPUT.exists(), 'exclusive-create completion; no overwrite')
    for folder in (SMOKE, RECOVERY, PREVIOUS):
        check_pack(folder, None)
    validation = records_check()
    state = read(ROOT / MUTABLE[0])
    require(state['latest_phase']['total_main_attempts'] == 35
            and state['latest_phase']['actual_cpu_model_inference_executed'], 'state differs')
    require(not state['complete_four_module_system_proven']
            and not state['graduation_delivery_plan']['active_experiment_sessions'], 'unscoped system claim')
    inputs = {str(p.relative_to(ROOT)): sha(p) for folder in (SMOKE, RECOVERY, PREVIOUS)
              for p in folder.rglob('*') if p.is_file()}
    checks = manuscript_checks()
    OUTPUT.mkdir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in MUTABLE:
            archive.write(ROOT / relative, relative)
    write_bytes(OUTPUT, OUTPUT / 'mutable_document_snapshots.zip', buffer.getvalue())
    freeze(OUTPUT, [Path(__file__), ROOT / REPORT, ROOT / PRIOR,
                    ROOT / 'env/requirements-cpu-frontend-v37.txt',
                    ROOT / 'configs/virtual3d/cpu_frontend_smoke_v37_20260920.json'],
           input_sha256=inputs, mutable_live_files_not_permanently_frozen=MUTABLE,
           mutable_snapshot_sha256={p: sha(ROOT / p) for p in MUTABLE},
           mutable_archive_sha256=sha(OUTPUT / 'mutable_document_snapshots.zip'),
           scope='Close bounded V37 software and document-prior feasibility; natural-device efficacy remains unverified')
    write(OUTPUT, OUTPUT / 'validation.json', dict(saved_record_review=validation,
        independent_readonly_review='passed; separate reviewer used saved arrays and hashes only',
        manuscript_static_checks=checks, prior_report_is_documentary_not_instance_validation=True))
    write(OUTPUT, OUTPUT / 'result.json', dict(status='completed_bounded_CPU_feasibility_phase',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), report=REPORT,
        original_virtual_V35_V36_experiments_complete=True, actual_CPU_smoke_passed=True,
        actual_predict_calls_in_smoke=14, new_boundary_tests_passed=5,
        qualified_equipment_instances=0, natural_device_efficacy_gate_passed=False,
        actual_ROS_or_robot_validation=False, current_experiment_sessions=[], full_graduation_goal_complete=False,
        new_main_tasks=0, main_tasks_used=35, main_tasks_cap=36,
        new_model_or_physics_calls_in_this_seal=0, free_bytes_before_seal=shutil.disk_usage(ROOT).free))
    check_input_map(inputs)
    verify_sources(OUTPUT)
    seal(OUTPUT)
    verify_inventory(OUTPUT)
    print(json.dumps(dict(status='completed', result_sha256=sha(OUTPUT / 'result.json'),
                          inventory_sha256=sha(OUTPUT / 'artifact_hashes.json'),
                          saved_record_review=validation, free_mib=shutil.disk_usage(ROOT).free / 1024**2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run:
        run()
    else:
        parser.print_help()
