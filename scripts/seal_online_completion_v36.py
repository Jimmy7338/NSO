#!/usr/bin/env python3
"""Verify and seal the bounded V36 confirmation and editable-document snapshots."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.research_evidence_v31 import (
    read, sha, freeze, seal, verify_sources, verify_inventory, write, write_bytes)
from scripts.seal_online_completion_v35 import (
    MUTABLE, check_pack, check_input_map, manuscript_checks, require)

OUTPUT = ROOT / 'audit_results/v36_completion_handoff_20260918'
REPORT = 'docs/research/V36_ONLINE_CONFIRMATION_RESULT_20260918.md'
PLAN = 'docs/research/V37_SEMANTIC_FRONTEND_AND_ROS1_PLAN_20260918.md'
IMMUTABLE = [REPORT, PLAN,
    'docs/research/V36_ONLINE_CONFIRMATION_INDEPENDENT_REVIEW_20260918.md',
    'docs/research/V36_ADAPTER_STATIC_REVIEW_20260918.md',
    'docs/research/V36_ONLINE_CONFIRMATION_PROTOCOL_20260918.md']


def archived_failed_test(folder):
    """The first test version remains in its archive, not as today's live test."""
    verify_inventory(folder)
    manifest = read(folder / 'manifest.json')
    check_input_map(manifest['input_sha256'])
    require(sha(folder / 'sources.zip') == manifest['source_archive_sha256'], 'initial test archive changed')
    differences = []
    with zipfile.ZipFile(folder / 'sources.zip') as archive:
        require(set(archive.namelist()) == set(manifest['source_sha256']), 'initial source set changed')
        for rel, expected in manifest['source_sha256'].items():
            require(hashlib.sha256(archive.read(rel)).hexdigest() == expected, 'initial source bytes changed')
            if sha(ROOT / rel) != expected:
                differences.append(rel)
    require(set(differences) <= {
        'tests/virtual3d/test_information_pixel_v36.py',
        'audit_results/v36_seed_adapter_tests_20260918/reproduce.py'}, 'unexplained historical source change')
    require(read(folder / 'result.json')['status'] == 'failed', 'initial failure was not retained')
    return differences


def run():
    require(not OUTPUT.exists(), 'exclusive-create handoff; no overwrite')
    spec = importlib.util.spec_from_file_location('collector_v36', ROOT / 'scripts/run_online_routes_v36.py')
    collector = importlib.util.module_from_spec(spec); spec.loader.exec_module(collector)
    collector.frozen()
    roots = sorted(p for p in (ROOT / 'audit_results').glob('v36_*_20260918') if p.is_dir())
    roots += [ROOT / ('audit_results/' + name) for name in (
        'v35_completion_handoff_20260918', 'v35_online_development_20260918',
        'v35_semantic_chain_review_20260918')]
    archived_differences = []
    for folder in roots:
        if folder.name == 'v36_seed_adapter_tests_attempt01_20260918':
            archived_differences = archived_failed_test(folder)
        else:
            check_pack(folder, collector)
    batch = read(collector.OUTPUT / 'result.json')
    audit = read(ROOT / 'audit_results/v36_online_confirmation_review_20260918/result.json')
    require(batch['total_main_used'] == 35 and batch['qualification_and_replay_gate_passed'], 'batch qualification failed')
    require(audit['complete_confirmation_gate_passed'] is True, 'independent confirmation failed')
    state = read(ROOT / MUTABLE[0])
    require(state['latest_phase']['total_main_attempts'] == 35 and state['next_validation_plan'] == PLAN, 'state disagrees')
    require(not state['complete_four_module_system_proven'], 'unscoped whole-system claim')
    require(not state['graduation_delivery_plan']['active_experiment_sessions'], 'state has active acquisition')
    checks = manuscript_checks()
    inputs = {str(p.relative_to(ROOT)): sha(p) for folder in roots for p in folder.rglob('*') if p.is_file()}
    for name in ('v36_confirmation_figures_20260918', 'v36_confirmation_figures_layout_20260918'):
        figures = read(ROOT / 'audit_results' / name / 'result.json')['figures']
        check_input_map(figures); inputs.update(figures)
    OUTPUT.mkdir()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in MUTABLE: archive.write(ROOT / rel, rel)
    write_bytes(OUTPUT, OUTPUT / 'mutable_document_snapshots.zip', buffer.getvalue())
    freeze(OUTPUT, [Path(__file__)] + [ROOT / p for p in IMMUTABLE],
        input_sha256=inputs, mutable_live_files_not_permanently_frozen=MUTABLE,
        mutable_snapshot_sha256={p: sha(ROOT / p) for p in MUTABLE},
        mutable_archive_sha256=sha(OUTPUT / 'mutable_document_snapshots.zip'),
        scope='V36 bounded confirmation completion; no new physical, control, TSDF or Q calls',
        new_main_tasks=0, main_tasks_used=35)
    write(OUTPUT, OUTPUT / 'validation.json', dict(
        verified_evidence_roots=[str(p.relative_to(ROOT)) for p in roots],
        historical_test_differences_verified_in_original_archive=archived_differences,
        original_model_cache_recovery_receipt_verified=True, manuscript_static_checks=checks,
        independent_report_and_next_plan_readonly_review='passed',
        input_files=len(inputs)))
    write(OUTPUT, OUTPUT / 'result.json', dict(status='completed_bounded_phase', report=REPORT,
        next_plan=PLAN, full_goal_complete=False, main_tasks_used=35, main_tasks_cap=36,
        active_sessions=[], controlled_online_CPU_frozen_policy_confirmation_passed=True,
        overall_gate_source='audit_results/v36_online_confirmation_review_20260918/result.json',
        natural_semantic_innovation_proven=False, original_neural_ANS_general_advantage_proven=False,
        feedback_efficacy_reconfirmed=False, next_physical_matrix_started=False,
        remaining_one_not_automatic_retry=True, new_physical_controller_or_metric_calls=0,
        frozen_input_files=len(inputs), free_bytes_before_handoff_completion=shutil.disk_usage(ROOT).free))
    check_input_map(inputs); verify_sources(OUTPUT); seal(OUTPUT); verify_inventory(OUTPUT)
    print(json.dumps(dict(status='complete', result_sha256=sha(OUTPUT / 'result.json'),
        inventory_sha256=sha(OUTPUT / 'artifact_hashes.json'), inputs=len(inputs),
        bytes=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()),
        free_mib=shutil.disk_usage(ROOT).free / 1024**2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    if parser.parse_args().run: run()
    else: parser.print_help()
