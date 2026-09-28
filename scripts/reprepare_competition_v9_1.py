#!/usr/bin/env python3
"""Bind the unchanged V9 prefix-only candidate inventory to variable-pool rules.

No world, mapper, scorer or evaluator is imported. Original failed six-role
evidence remains immutable; all raw data, routes and predictions are reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def require(value, message):
    if not value: raise ValueError(message)


def run(base, review, output):
    base, review, output = base.resolve(), review.resolve(), output.resolve()
    require(not output.exists(), 'new output required')
    started = time.monotonic()
    manifest = read(base / 'artifact_hashes.json'); original = read(base / 'metadata.json')
    for name, digest in manifest.items(): require(sha(base / name) == digest, 'old preparation changed')
    require(original['status'] == 'halted_structural_failure' and original['new_candidate_branches'] == 0, 'original failure required')
    checked = read(review)
    require(checked['status'] == 'passed_independent_release_review' and checked['release_allowed'] is False
            and checked['prepared_manifest_sha256'] == sha(base / 'artifact_hashes.json')
            and checked['raw_prefix_frames_and_scans_checked'] == 1208, 'complete original failed release review required')
    protocol_path = 'configs/virtual3d/competition_v9_1_validation_protocol.json'
    protocol = read(ROOT / protocol_path)
    require(protocol['schema_version'] == 'competition_v9_1_validation_protocol/1', 'new protocol required')
    for name, digest in original['source_sha256'].items(): require(sha(ROOT / name) == digest, 'original source changed')
    output.mkdir(parents=True)
    names = sorted(set(original['source_sha256']) | {protocol_path, 'scripts/reprepare_competition_v9_1.py'})
    sources = {name: sha(ROOT / name) for name in names}
    with zipfile.ZipFile(output / 'sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in names: archive.write(ROOT / name, name)
    structures = []; prefixes = []; old_summary = read(base / 'structure_summary.json')
    for parent in protocol['parent_order']:
        for arrangement in protocol['arrangement_order']:
            source = base / parent / arrangement; target = output / parent / arrangement
            target.mkdir(parents=True)
            for p in source.rglob('*'):
                if p.is_file() and p.name != 'structural_audit.json':
                    destination = target / p.relative_to(source); destination.parent.mkdir(parents=True, exist_ok=True)
                    os.link(p, destination)
            routes = read(source / 'candidates.json')
            require([r['candidate_id'] for r in routes] == protocol['candidate_ids_by_parent'][parent], 'predeclared route inventory changed')
            require([r['group'] for r in routes] == protocol['candidate_roles_by_parent'][parent], 'predeclared role inventory changed')
            structure = read(source / 'structural_audit.json')
            checks = dict(structure['checks']); checks['candidate_roles'] = True
            structure.update(checks=checks, passed=all(checks.values()),
                original_structure_sha256=sha(source / 'structural_audit.json'),
                original_fixed_six_role_gate_passed=read(source / 'structural_audit.json')['checks']['candidate_roles'],
                missing_optional_roles=read(source / 'candidate_audit.json')['missing_roles'],
                raw_candidates_scores_reference_changed=False)
            structures.append(structure); write(target / 'structural_audit.json', structure)
            prefixes.append(next(h for h in old_summary['prefix_summaries'] if h['context'] == parent and h['arrangement'] == arrangement))
    passed = all(h['passed'] for h in structures) and all(v['union_area_equal'] for v in old_summary['pairs'].values())
    write(output / 'pre_evaluator_choices_seal.json', {str(p.relative_to(output)): sha(p)
          for parent in protocol['parent_order'] for p in (output / parent).rglob('*') if p.is_file()})
    write(output / 'structure_summary.json', {'passed': passed, 'histories': structures, 'prefix_summaries': prefixes,
        'pairs': old_summary['pairs'], 'new_candidate_branches': 0, 'new_sensor_captures': 0,
        'original_six_role_version_passed': False, 'rule_change_based_only_on_prefix_feasibility': True})
    write(output / 'metadata.json', {'status': 'complete_structural_pass' if passed else 'halted_structural_failure',
        'structural_gate_passed': passed, 'source_sha256': sources, 'protocol_path': protocol_path,
        'base_preparation': str(base), 'base_manifest_sha256': sha(base / 'artifact_hashes.json'),
        'base_release_review': str(review), 'base_release_review_sha256': sha(review),
        'new_sensor_captures': 0, 'new_candidate_branches': 0, 'elapsed_s': time.monotonic() - started})
    for name, digest in manifest.items(): require(sha(base / name) == digest, 'original input changed during rebind')
    for name, digest in sources.items(): require(sha(ROOT / name) == digest, 'source changed during rebind')
    write(output / 'artifact_hashes.json', {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps({'status': 'complete_structural_pass' if passed else 'halted_structural_failure',
                       'new_physical_actions': 0, 'reused_histories': 8, 'available_routes': sum(h['candidate_count'] for h in prefixes)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.base, args.base_review, args.output)
