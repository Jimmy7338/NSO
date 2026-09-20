#!/usr/bin/env python3
"""Seal the completed V37 offline preparation and mutable thesis snapshots."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.research_evidence_v31 import (
    freeze, read, seal, sha, verify_inventory, verify_sources, write, write_bytes)
from scripts.seal_online_completion_v35 import (
    MUTABLE, check_pack, check_input_map, manuscript_checks, require)

OUTPUT = ROOT / 'audit_results/v37_completion_handoff_20260920'
REPORT = 'docs/research/V37_FRONTEND_AND_ROS1_READINESS_RESULT_20260920.md'
PLAN = 'docs/research/V37_SEMANTIC_FRONTEND_AND_ROS1_PLAN_20260918.md'
REVIEW = 'docs/research/V37_SEMANTIC_PRIOR_READINESS_REVIEW_20260918.md'
ASSETS = ROOT / 'audit_results/v37_semantic_assets_20260918'
TESTS = ROOT / 'audit_results/v37_interface_contract_tests_20260920'
PREVIOUS = ROOT / 'audit_results/v36_completion_handoff_20260918'


def run():
    require(not OUTPUT.exists(), 'exclusive-create completion; never overwrite')
    for folder in (ASSETS, TESTS, PREVIOUS):
        # These three packages use inventories, not a collector final_seal.
        require(not (folder / 'final_seal.json').exists(), 'unexpected package kind')
        check_pack(folder, None)
    assets, tests = read(ASSETS / 'result.json'), read(TESTS / 'result.json')
    require(tests['status'] == 'passed' and tests['tests_run'] == 14
            and tests['failures'] == tests['errors'] == tests['skipped'] == 0, 'interface tests failed')
    require(assets['qualified_natural_device_instances'] == 0 and not assets['raw_bag_files'],
            'asset qualification requires a revised report')
    require(assets['checkpoint']['cache_matches_pointer'] is True, 'model cache inventory differs')
    for data in (assets, tests):
        require(data['new_main_tasks'] == data['new_world_TSDF_quality_calls'] == 0, 'unexpected new experiment')
        require(data['natural_frontend_inference_executed'] is False, 'unexpected model execution')
    example = read(TESTS / 'unverified_example_preflight.json')
    require(example['ready_to_print_capture_commands'] is False and 'commands' not in example,
            'unverified example exported commands')
    diagnostics = read(TESTS / 'legacy_diagnostics.json')
    require(diagnostics['mixed_category_index_error']['reproduced']
            and diagnostics['all_category_channel_update']['per_channel_sums'] == [4, 4, 4],
            'legacy diagnostic differs')
    reviewed = dict(re.findall(r'\| `([^`]+)` \| `([a-f0-9]{64})` \|', (ROOT / REVIEW).read_text()))
    require(len(reviewed) == 11, 'expected 11 reviewed source hashes')
    check_input_map(reviewed)
    state = read(ROOT / MUTABLE[0])
    require(state['latest_phase']['total_main_attempts'] == 35
            and state['latest_phase']['offline_interface_tests_passed'] == 14, 'state disagrees')
    require(not state['complete_four_module_system_proven']
            and not state['graduation_delivery_plan']['active_experiment_sessions'], 'unscoped claim or active capture')
    inputs = dict(reviewed)
    for folder in (ASSETS, TESTS, PREVIOUS):
        inputs.update({str(p.relative_to(ROOT)): sha(p) for p in folder.rglob('*') if p.is_file()})
    checks = manuscript_checks()
    OUTPUT.mkdir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in MUTABLE:
            archive.write(ROOT / relative, relative)
    write_bytes(OUTPUT, OUTPUT / 'mutable_document_snapshots.zip', buffer.getvalue())
    freeze(OUTPUT, [Path(__file__), ROOT / REPORT, ROOT / PLAN, ROOT / REVIEW,
                    ROOT / 'docs/ROS1_FIELD_CAPTURE_V37_20260918.md',
                    ROOT / 'configs/virtual3d/ros1_field_capture_v37.example.json'],
           input_sha256=inputs, mutable_live_files_not_permanently_frozen=MUTABLE,
           mutable_snapshot_sha256={p: sha(ROOT / p) for p in MUTABLE},
           mutable_archive_sha256=sha(OUTPUT / 'mutable_document_snapshots.zip'),
           scope='V37 offline asset and interface preparation, not natural semantic or robot efficacy',
           main_tasks_used=35, new_main_tasks=0)
    write(OUTPUT, OUTPUT / 'validation.json', dict(
        checked_packages=[str(p.relative_to(ROOT)) for p in (ASSETS, TESTS, PREVIOUS)],
        checked_legacy_source_hashes=reviewed, manuscript_static_checks=checks,
        independent_readonly_code_review='no blocking findings for the offline contract scope',
        review_limits=['old runtime map unchanged', 'no instance tracking or cross-frame dedup',
                       'registry evidence declarations not scientifically authenticated',
                       'no image-detection alignment or model execution',
                       'no ROS TF/calibration/clock/physical verification'],
        unverified_example_refused_commands=True))
    result = dict(status='completed_bounded_offline_preparation',
                  completed_at_utc=datetime.now(timezone.utc).isoformat(), report=REPORT, next_plan=PLAN,
                  interface_tests_passed=14, legacy_defects_reproduced=2,
                  natural_frontend_inference_executed=False, natural_semantic_innovation_proven=False,
                  ros_live_communication_executed=False, robot_interface_ready=False,
                  full_goal_complete=False, new_main_tasks=0, main_tasks_used=35, main_tasks_cap=36,
                  new_world_controller_TSDF_quality_calls=0, active_sessions=[],
                  free_bytes_before_seal=shutil.disk_usage(ROOT).free)
    write(OUTPUT, OUTPUT / 'result.json', result)
    check_input_map(inputs)
    verify_sources(OUTPUT)
    seal(OUTPUT)
    verify_inventory(OUTPUT)
    print(json.dumps(dict(status=result['status'], result_sha256=sha(OUTPUT / 'result.json'),
                          inventory_sha256=sha(OUTPUT / 'artifact_hashes.json'),
                          bytes=sum(p.stat().st_size for p in OUTPUT.iterdir()),
                          free_mib=shutil.disk_usage(ROOT).free / 1024**2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run:
        run()
    else:
        parser.print_help()
