#!/usr/bin/env python3
"""Seal V35 reports and mutable-document snapshots; no experiment execution."""
import argparse
import hashlib
import importlib.util
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
    read, sha, freeze, seal, verify_sources, verify_inventory, write, write_bytes)

OUTPUT = ROOT / 'audit_results/v35_completion_handoff_20260918'
REPORT = 'docs/research/V35_ONLINE_SEMANTIC_RESULT_20260918.md'
PLAN = 'docs/research/V36_FROZEN_CONFIRMATION_PLAN_20260918.md'
IMMUTABLE = [REPORT, PLAN,
    'docs/research/V35_ONLINE_SEMANTIC_INDEPENDENT_REVIEW_20260918.md',
    'docs/research/V35_SAVED_OBSERVATION_INDEPENDENT_REVIEW_20260918.md']
MUTABLE = ['docs/research/CURRENT_RESEARCH_STATE.json', 'docs/research/GOAL_PROMPT.md',
    'docs/research/RESEARCH_GOAL_AND_STORY.md',
    'docs/research/THESIS_CLAIM_EVIDENCE_LEDGER.md',
    'Semantic_Enhanced_Active_SLAM_Paper.tex']


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_input_map(mapping):
    for relative, expected in mapping.items():
        require(sha(ROOT / relative) == expected, 'frozen input changed: ' + relative)


def check_pack(folder, collector):
    if (folder / 'manifest.json').exists():
        manifest = verify_sources(folder)
        check_input_map(manifest.get('input_sha256', {}))
        if 'mutable_snapshot_sha256' in manifest:
            archive = folder / 'mutable_document_snapshots.zip'
            require(sha(archive) == manifest['mutable_archive_sha256'], 'old snapshot archive changed')
            with zipfile.ZipFile(archive) as zipped:
                require(set(zipped.namelist()) == set(manifest['mutable_snapshot_sha256']), 'snapshot set differs')
                for rel, expected in manifest['mutable_snapshot_sha256'].items():
                    require(hashlib.sha256(zipped.read(rel)).hexdigest() == expected, 'snapshot differs')
            # Historical mutable documents are checked in their archive, not against today's live files.
    else:
        check_input_map(read(folder / 'result.json')['source_sha256'])
    if (folder / 'final_seal.json').exists():
        collector.verify_seal(folder, 'final_seal.json')
    else:
        verify_inventory(folder)


def manuscript_checks():
    manuscript = (ROOT / MUTABLE[-1]).read_text()
    uncommented = re.sub(r'(?<!\\)%[^\n]*', '', manuscript)
    depth = 0
    for match in re.finditer(r'(?<!\\)[{}]', uncommented):
        depth += 1 if match[0] == '{' else -1
        require(depth >= 0, 'unbalanced manuscript braces')
    require(depth == 0, 'unbalanced manuscript braces')
    stack = []
    for kind, env in re.findall(r'\\(begin|end)\{([^}]+)\}', uncommented):
        if kind == 'begin':
            stack.append(env)
        else:
            require(stack and stack.pop() == env, 'unbalanced manuscript environments')
    require(not stack, 'unclosed manuscript environment')
    labels = re.findall(r'\\label\{([^}]+)\}', uncommented)
    require(len(labels) == len(set(labels)), 'duplicate manuscript label')
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', uncommented)
    require(all((ROOT / f).is_file() for f in figures), 'missing manuscript figure')
    return dict(braces=True, environments=True, unique_labels=len(labels),
        existing_figures=len(figures), PDF_compilation_performed=False,
        available_TeX_compilers=[x for x in ('xelatex', 'lualatex', 'pdflatex', 'latexmk') if shutil.which(x)])


def run():
    require(not OUTPUT.exists(), 'completion package is exclusive-create; never overwrite')
    spec = importlib.util.spec_from_file_location('collector_v35', ROOT / 'scripts/run_online_routes_v35.py')
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.frozen()
    roots = sorted(p for p in (ROOT / 'audit_results').glob('v35_*_20260918') if p.is_dir())
    roots += [ROOT / ('audit_results/' + name) for name in (
        'v34_completion_handoff_20260918', 'v34_fixed_measurement_20260918',
        'v34_pixel_information_20260918', 'v33_direction_information_r1_20260917')]
    for folder in roots:
        check_pack(folder, collector)
    batch = read(collector.OUTPUT / 'result.json')
    independent = read(ROOT / 'audit_results/v35_semantic_chain_review_20260918/result.json')
    require(batch['total_main_used'] == 27 and batch['qualification_and_replay_gate_passed'], 'batch not qualified')
    require(independent['complete_development_gate_passed'] is True, 'independent gate not passed')
    state = read(ROOT / MUTABLE[0])
    require(state['latest_phase']['total_main_attempts'] == 27, 'state count mismatch')
    require(state['next_validation_plan'] == PLAN, 'state next plan mismatch')
    require(not state['graduation_delivery_plan']['active_experiment_sessions'], 'state contains active sessions')
    require(not state['complete_four_module_system_proven'], 'unscoped overall claim')
    checks = manuscript_checks()
    inputs = {str(p.relative_to(ROOT)): sha(p) for root in roots for p in root.rglob('*') if p.is_file()}
    figures = read(ROOT / 'audit_results/v35_online_semantic_figures_20260918/result.json')['figures']
    check_input_map(figures)
    inputs.update(figures)
    OUTPUT.mkdir()
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in MUTABLE:
            archive.write(ROOT / rel, rel)
    write_bytes(OUTPUT, OUTPUT / 'mutable_document_snapshots.zip', payload.getvalue())
    freeze(OUTPUT, [Path(__file__)] + [ROOT / p for p in IMMUTABLE],
        input_sha256=inputs, mutable_live_files_not_permanently_frozen=MUTABLE,
        mutable_snapshot_sha256={p: sha(ROOT / p) for p in MUTABLE},
        mutable_archive_sha256=sha(OUTPUT / 'mutable_document_snapshots.zip'),
        scope='bounded V35 completion; mutable document snapshots only; no World, sensor, controller, TSDF or Q',
        new_main_tasks=0, main_tasks_used=27)
    write(OUTPUT, OUTPUT / 'validation.json', dict(
        all_frozen_packs_verified=[str(p.relative_to(ROOT)) for p in roots],
        manuscript_static_checks=checks, input_files=len(inputs),
        independent_report_review='V35 arithmetic, scope and V36 plan reviewed; two wording corrections applied before sealing',
        negative_results_retained=['h0 S-G zero', 'h1 feedback 2cm -0.0021593573062449467']))
    write(OUTPUT, OUTPUT / 'result.json', dict(status='completed_bounded_phase',
        report=REPORT, next_plan=PLAN, full_goal_complete=False, main_tasks_used=27,
        main_tasks_cap=36, active_sessions=[], controlled_online_CPU_four_module_development_passed=True,
        overall_gate_source='audit_results/v35_semantic_chain_review_20260918/result.json',
        full_architecture_advantage_proven=False, natural_semantic_innovation_proven=False,
        next_confirmation_started=False, next_confirmation_maximum_main_tasks=8,
        next_confirmation_seed=350918, frozen_input_files=len(inputs),
        new_physical_controller_or_metric_calls_during_handoff=0,
        free_bytes_before_handoff_completion=shutil.disk_usage(ROOT).free))
    check_input_map(inputs)
    verify_sources(OUTPUT)
    seal(OUTPUT)
    verify_inventory(OUTPUT)
    print(json.dumps(dict(status='complete', result_sha256=sha(OUTPUT / 'result.json'),
        inventory_sha256=sha(OUTPUT / 'artifact_hashes.json'), input_files=len(inputs),
        total_bytes=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file()),
        free_mib=shutil.disk_usage(ROOT).free / 1024**2), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    if parser.parse_args().run:
        run()
    else:
        parser.print_help()
