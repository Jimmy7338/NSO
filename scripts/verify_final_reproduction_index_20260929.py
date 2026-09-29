#!/usr/bin/env python3
"""Bounded publication check: hashes/receipts only; no scientific imports or replay.

Run from any directory. Only the explicit metadata and output lists in the index
are read. This is not an independent re-evaluation of surface quality, an archive
roundtrip, or a replacement for the sealed per-episode reviewers.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / 'audit_results/final_delivery_20260929/reproduction_index.json'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inside(relative, base=ROOT):
    path = (base / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'indexed path outside repository: {relative}')
    return path


def verify(index_path):
    began = time.monotonic()
    index = json.loads(index_path.read_text())
    checked, failures, cached, bytes_read = [], [], {}, 0

    def require(ok, label):
        checked.append(label)
        if not ok:
            failures.append(label)

    def file_pin(path, pin):
        nonlocal bytes_read
        expected = pin if isinstance(pin, str) else pin['sha256']
        if not path.is_file():
            require(False, f'missing: {path.relative_to(ROOT)}')
            return
        if path not in cached:
            cached[path] = digest(path)
            bytes_read += path.stat().st_size
        require(cached[path] == expected, f'SHA256: {path.relative_to(ROOT)}')
        if isinstance(pin, dict) and 'bytes' in pin:
            require(path.stat().st_size == pin['bytes'], f'bytes: {path.relative_to(ROOT)}')

    require(index['schema'] == 'final.method_reproduction_index.v1', 'index schema')
    for name, pin in index['verification']['pinned_files'].items():
        file_pin(inside(name), pin)

    syntax_trees = {}
    for entry in index['method_map']:
        for reference in entry['symbols']:
            name, symbol = reference.split('::', 1)
            if name not in syntax_trees:
                syntax_trees[name] = ast.parse(inside(name).read_text())
            node = syntax_trees[name]
            for component in symbol.split('.'):
                node = next((child for child in getattr(node, 'body', [])
                             if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                             and child.name == component), None)
                if node is None:
                    break
            require(node is not None, f'method symbol exists: {reference}')

    freeze_path = inside(index['verification']['source_freeze'])
    freeze = json.loads(freeze_path.read_text())
    current = freeze['source_sha256']
    protected = freeze['protected_20260923_sources']
    require(len(current) == 45, '45 execution sources')
    require(len(protected) == 59, '59 protected historical sources')
    for collection in (current, protected):
        for name, pin in collection.items():
            file_pin(inside(name), pin)
    archive = freeze_path.parent / 'source_archive.zip'
    file_pin(archive, freeze['source_archive_sha256'])
    with zipfile.ZipFile(archive) as source_zip:
        require(set(source_zip.namelist()) == set(current), 'source ZIP exact 45 member inventory')
        for name, pin in current.items():
            require(hashlib.sha256(source_zip.read(name)).hexdigest() == pin,
                    f'archived execution source: {name}')

    output_count = 0
    for name in index['verification']['output_manifests']:
        manifest_path = inside(name)
        manifest = json.loads(manifest_path.read_text())
        table = manifest.get('files', manifest.get('outputs', manifest.get('output_sha256')))
        require(isinstance(table, dict) and bool(table), f'nonempty output list: {name}')
        for filename, pin in (table or {}).items():
            # Historic provenance uses repository-relative keys; newer bundles use
            # manifest-relative keys. This choice never probes alternate files.
            base = ROOT if filename.startswith(('docs/', 'audit_results/', 'scripts/')) else manifest_path.parent
            file_pin(inside(filename, base), pin)
            output_count += 1

    delivery = ROOT / 'audit_results/final_delivery_20260929'
    review = json.loads((delivery / 'validation_review.json').read_text())
    require(review['status'] == 'passed' and review['qualified_tasks'] == 8
            and review['reviewed_frames'] == 344, 'new8 independent review: 8 qualified / 344 frames')
    require((review['wins'], review['ties'], review['losses']) == (1, 3, 0), 'all four new8 pairs retained')
    require(review['final_goal_online_tasks_used'] == 8 and review['unused_reserved_tasks'] == 8,
            'new final tasks: eight used, eight unused')
    replay = json.loads((delivery / 'saved_replay_case0/result.json').read_text())
    require(replay['status'] == 'passed' and replay['saved_frames_replayed'] == 43
            and replay['paid_actions'] == 42, 'saved case0 replay: passed / 43 frames / 42 actions')
    for field in ('exact_full_controller_plans_calls_actions', 'exact_per_frame_map_and_controller',
                  'exact_raw_and_roi_meshes'):
        require(replay[field] is True, f'replay receipt: {field}')
    for field in ('new_online_tasks', 'new_quality_evaluations', 'new_sensor_queries', 'new_worlds'):
        require(replay[field] == 0, f'replay receipt: {field}=0')
    browser = json.loads((delivery / 'replay_browser/result.json').read_text())
    require(browser['status'] == 'passed' and browser['checks_passed'] == 10
            and len(browser['checks']) == 10, 'browser: 10 passed checks')
    require(not browser['browser_errors'] and not browser['external_requests'], 'browser: no errors / external requests')
    file_pin(ROOT / 'docs/thesis/demos/v40_replay/index.html', browser['html_sha256'])
    for name, pin in json.loads((delivery / 'replay_browser/artifact_sha256.json').read_text()).items():
        path = ROOT / 'scripts/verify_replay_browser_v40.cjs' if name == 'verifier_source' else delivery / 'replay_browser' / name
        file_pin(path, pin)

    expansion = json.loads(inside(index['cohorts']['expansion128']['summary']).read_text())
    require((expansion['declared_cases'], expansion['qualified'], expansion['failed']) == (128, 96, 32),
            '128 cohort retains 96 qualified and 32 failed')
    scene = json.loads(inside(index['cohorts']['scene36']['summary']).read_text())
    require(scene['declared_attempts'] == 36 and scene['original_qualified'] ==
            {'OriginalV1': 11, 'GroundV2': 12, 'ExposureV3': 12}, '36 original statuses retained')
    checkpoints = json.loads(inside(index['cohorts']['checkpoints40']['summary']).read_text())
    require(checkpoints['checkpoint_count'] == 40 and checkpoints['cases'] == 8
            and checkpoints['new_independent_samples'] == 0, '40 checkpoints / 8 existing routes / no new samples')
    return dict(schema='final.reproduction_index_check.v1', status='passed' if not failures else 'failed',
                index_sha256=digest(index_path), verifier_source_sha256=digest(Path(__file__)),
                checks_passed=len(checked)-len(failures), checks_total=len(checked), failures=failures,
                unique_files_hashed=len(cached), file_bytes_hashed=bytes_read,
                listed_bundle_outputs_checked=output_count, execution_source_count=45,
                protected_historical_source_count=59, elapsed_seconds=time.monotonic()-began,
                scope='Explicit metadata/output pins and recorded PASS receipts; no raw-frame rescanning, quality remeasurement, archive roundtrip or replay.',
                new_worlds=0, new_sensor_queries=0, new_policy_runs=0, new_tsdf_integrations=0,
                new_surface_evaluations=0, browser_launches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()
    try:
        result = verify(args.index.resolve())
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        result = dict(status='failed', error=f'{type(exc).__name__}: {exc}')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
