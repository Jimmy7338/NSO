#!/usr/bin/env python3
"""Review the byte-preserving V9.1 cardinality revision, without future sensing.

Reuses the completed eight-history V9 raw replay only after checking every
source and asset binding. Independently checks all44 paths, the new finite
inventory and the Q3 missing-deep certificate with a direct free-cell stencil.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
def require(condition, message):
    if not condition: raise AssertionError(message)


def checked_manifest(directory):
    values = read(directory / 'artifact_hashes.json')
    for name, digest in values.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), 'manifest path escaped directory')
        require(sha(path) == digest, 'changed manifest input: ' + str(path))
    return values


def successor(state, action):
    r, c, h = state
    if action == 'left': return r, c, (h - 1) % 4
    if action == 'right': return r, c, (h + 1) % 4
    require(action == 'forward', 'unknown paid action')
    dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[h]
    return r + dr, c + dc, h


def run(args):
    args.run = args.run.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); own_sha = sha(__file__)
    try:
        current_manifest = checked_manifest(args.run)
        metadata = read(args.run / 'metadata.json')
        require(metadata['status'] == 'complete_structural_pass', 'new preparation not structurally complete')
        require(metadata['new_sensor_captures'] == metadata['new_candidate_branches'] == 0, 'unexpected new physical data')
        with zipfile.ZipFile(args.run / 'sources.zip') as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)) and set(names) == set(metadata['source_sha256']), 'archive inventory')
            for name, digest in metadata['source_sha256'].items():
                require(hashlib.sha256(archive.read(name)).hexdigest() == digest, 'source archive changed: ' + name)
            protocol = json.loads(archive.read(metadata['protocol_path']))
        require(protocol['schema_version'] == 'competition_v9_1_validation_protocol/1', 'incorrect revision protocol')
        provenance = protocol['revision_provenance']
        for name, digest in provenance['input_sha256'].items():
            require(sha(ROOT / name) == digest, 'original revision evidence changed: ' + name)
        original_protocol = read(ROOT / provenance['base_protocol'])
        base = (ROOT / provenance['base_preparation']).resolve()
        require(base == Path(metadata['base_preparation']).resolve(), 'base preparation path')
        original_manifest = checked_manifest(base)
        require(sha(base / 'artifact_hashes.json') == metadata['base_manifest_sha256'], 'base manifest binding')
        original_meta = read(base / 'metadata.json')
        require(original_meta['status'] == 'halted_structural_failure' and original_meta['new_candidate_branches'] == 0,
                'original V9 failure or zero-future status changed')
        for name, digest in original_meta['source_sha256'].items():
            require(metadata['source_sha256'].get(name) == digest, 'inherited source changed: ' + name)
        review_path = ROOT / provenance['base_independent_review']
        checked_manifest(review_path.parent)
        original_review = read(review_path)
        require(original_review['status'] == 'passed_independent_release_review'
                and original_review['release_allowed'] is False, 'original review did not retain scientific failure')
        require(original_review['prepared_manifest_sha256'] == sha(base / 'artifact_hashes.json'), 'original review preparation binding')
        require(original_review['raw_prefix_frames_and_scans_checked'] == 1208
                and original_review['actual_semantic_intervention_mapper_frames'] == 2416
                and original_review['rate_and_total_score_arrays_verified'] == 96
                and original_review['additional_slot_and_ground_footprint_gates_passed'], 'incomplete inherited independent review')
        require(sha(review_path.parent / 'verifier_source.py') == original_review['verifier_sha256'], 'original reviewer source binding')
        require(sha(review_path) == metadata['base_release_review_sha256'], 'new metadata review binding')
        for key in ('metrics', 'prediction_release', 'reference', 'sensors', 'prospective_world_contract',
                    'parent_order', 'arrangement_order', 'excluded_outer_seeds', 'architecture', 'aggregation'):
            require(protocol[key] == original_protocol[key], 'unapproved scientific rule change: ' + key)
        changed_procedural = {'all_48_branches_collision_free',
                              'all_48_branches_reach_original_target_and_restore_original_pose_heading_within48',
                              'failure_action'}
        for key, value in original_protocol['progression_necessary_conditions'].items():
            if key not in changed_procedural:
                require(protocol['progression_necessary_conditions'][key] == value, 'effect gate changed: ' + key)
        require(protocol['progression_necessary_conditions']['all_44_branches_collision_free']
                and protocol['progression_necessary_conditions']['all_44_branches_reach_original_target_and_restore_original_pose_heading_within48'],
                'all-route safety criterion relaxed')
        for key, value in original_protocol['pre_outcome_structural_gates'].items():
            if key not in ('candidate_roles_in_order', 'failure_action'):
                require(protocol['pre_outcome_structural_gates'][key] == value, 'non-cardinality structure rule changed: ' + key)
        require(protocol['physical_branch_limit'] == 44 and protocol['branch_actions'] == 48
                and protocol['prefix_paid_actions'] == 150, 'budget or finite count changed')
        required_roles = ['coverage_anchor', 'old_surface_rotation', 'west_entry', 'east_entry']
        require(protocol['pre_outcome_structural_gates']['mandatory_candidate_roles'] == required_roles, 'mandatory roles changed')
        seal = read(args.run / 'pre_evaluator_choices_seal.json')
        original_seal = read(base / 'pre_evaluator_choices_seal.json')
        for name, digest in original_seal.items():
            require(seal.get(name) == digest == current_manifest.get(name), 'original decision asset changed: ' + name)
        rows, branch_count, inherited_files = [], 0, 0
        for parent in protocol['parent_order']:
            pair_routes = []
            expected_ids = list(range(4 if parent == 'Q3' else 6))
            expected_roles = required_roles + ([] if parent == 'Q3' else ['west_deep', 'east_deep'])
            require(protocol['candidate_ids_by_parent'][parent] == expected_ids
                    and protocol['candidate_roles_by_parent'][parent] == expected_roles, 'declared feasible pool changed')
            for arrangement in protocol['arrangement_order']:
                folder = args.run / parent / arrangement; old_folder = base / parent / arrangement
                paths = [old_folder / name for name in ('fixture.json', 'prefix_map.npz', 'reference.npz',
                                                       'candidates.json', 'candidate_audit.json', 'predictions.json')]
                paths += [p for p in (old_folder / 'prefix').rglob('*') if p.is_file()]
                for path in paths:
                    target = folder / path.relative_to(old_folder)
                    require(sha(path) == sha(target), 'inherited asset changed: ' + str(target))
                    inherited_files += 1
                routes = read(folder / 'candidates.json'); audit = read(folder / 'candidate_audit.json')
                structural = read(folder / 'structural_audit.json'); old_structural = read(old_folder / 'structural_audit.json')
                require([r['candidate_id'] for r in routes] == expected_ids
                        and [r['group'] for r in routes] == expected_roles, 'actual finite candidate pool differs')
                require(structural['original_structure_sha256'] == sha(old_folder / 'structural_audit.json')
                        and structural['original_fixed_six_role_gate_passed'] == old_structural['passed'], 'original gate status changed')
                missing = [name for name in ('west_deep', 'east_deep') if name not in expected_roles]
                require(structural['missing_optional_roles'] == audit['missing_roles'] == missing, 'missing optional role hidden')
                checks = dict(old_structural['checks']); checks['candidate_roles'] = True
                require(structural['checks'] == checks and all(checks.values()) and structural['passed'], 'new gate mismatch')
                for key, value in old_structural.items():
                    if key not in ('checks', 'passed'):
                        require(structural[key] == value, 'structural measurement changed: ' + key)
                with np.load(folder / 'prefix_map.npz', allow_pickle=False) as saved:
                    free = saved['belief'] == 0
                fixture = read(folder / 'fixture.json'); config = fixture['config']
                require(config['resolution_m'] == config['robot_radius_m'] == .2, 'stencil assumes original robot/grid')
                padded = np.pad(free, 1, constant_values=False); safe = np.ones_like(free)
                for dr in range(3):
                    for dc in range(3): safe &= padded[dr:dr+free.shape[0], dc:dc+free.shape[1]]
                require(hashlib.sha256(safe.tobytes()).hexdigest() == audit['safe_sha256'], 'independent complete footprint differs')
                last = read(folder / 'prefix/records.json')[-1]; anchor = (*last['position'], last['heading'])
                for route in routes:
                    states = [tuple(v) for v in route['states']]
                    require(1 <= route['cost'] <= 48 and len(route['actions']) == route['cost'] == len(states)-1,
                            'unpaid or over-budget route')
                    require(states[0] == states[-1] == anchor and states[route['arrival_action']] == tuple(route['pose']),
                            'route misses target or original pose/heading')
                    for state in states:
                        require(0 <= state[0] < safe.shape[0] and 0 <= state[1] < safe.shape[1]
                                and 0 <= state[2] < 4 and safe[state[:2]], 'unsafe route state')
                    for before, action, after in zip(states, route['actions'], states[1:]):
                        require(successor(before, action) == after, 'route transition differs')
                absent_certificates = []
                if missing:
                    cells = np.argwhere(safe)
                    xy = np.column_stack([(cells[:, 1]+.5)*.2, (safe.shape[0]-cells[:, 0]-.5)*.2])
                    for slot in (0, 1):
                        asset = audit['measured_assets'][slot]
                        rear, axis = np.asarray(asset['rear_boundary_xy']), np.asarray(asset['back_axis'])
                        entry = next(r for r in routes if r['asset_index'] == slot and r['group'].endswith('entry'))
                        er, ec, _ = entry['pose']; entry_xy = np.array([(ec+.5)*.2, (safe.shape[0]-er-.5)*.2])
                        maximum = float(np.max((xy-rear) @ axis)); needed = float((entry_xy-rear) @ axis) + .2
                        require(maximum < needed-1e-12, 'optional deep omitted without the declared impossibility certificate')
                        absent_certificates.append({'slot': slot, 'max_known_safe_offset_m': maximum,
                                                    'required_offset_m': needed, 'impossible_on_current_safe_map': True})
                rows.append({'context': parent, 'arrangement': arrangement, 'candidate_ids': expected_ids,
                             'roles': expected_roles, 'missing_optional_roles': missing,
                             'missing_deep_certificates': absent_certificates,
                             'all_routes_actions_and_complete_footprints_valid': True,
                             'inherited_raw_scores_references_byte_identical': True,
                             'original_V9_gate_passed': old_structural['passed']})
                branch_count += len(routes); pair_routes.append(routes)
            require(pair_routes[0] == pair_routes[1], 'paired action set changed')
        require(branch_count == 44 and len(rows) == 8, 'finite pool incomplete')
        require(read(args.run / 'structure_summary.json')['passed'], 'new summary not passed')
        require(checked_manifest(args.run) == current_manifest and checked_manifest(base) == original_manifest,
                'inputs changed during review')
        require(sha(__file__) == own_sha, 'reviewer changed during review')
        summary = {'status': 'passed_independent_release_review', 'release_allowed': True, 'release_permitted': True,
                   'run': str(args.run), 'protocol_path': metadata['protocol_path'],
                   'protocol_sha256': metadata['source_sha256'][metadata['protocol_path']],
                   'prepared_manifest_sha256': sha(args.run / 'artifact_hashes.json'),
                   'source_archive_sha256': sha(args.run / 'sources.zip'),
                   'choices_seal_sha256': sha(args.run / 'pre_evaluator_choices_seal.json'),
                   'base_preparation_manifest_sha256': sha(base / 'artifact_hashes.json'),
                   'base_independent_review_sha256': sha(review_path),
                   'original_V9_release_allowed': False, 'original_V9_failure_preserved': True,
                   'declared_physical_branches': 44, 'histories_checked': 8,
                   'inherited_asset_files_byte_checked': inherited_files,
                   'base_raw_prefix_rgbd_scan_pairs_verified': 1208,
                   'base_actual_X_M_mapper_frames_verified': 2416,
                   'base_score_arrays_and_choices_verified': 96,
                   'raw_prefix_replayed_in_this_revision_review': 0,
                   'future_branches_generated': 0, 'world_or_production_planner_called': False,
                   'scientific_effect_thresholds_changed': False,
                   'scope': 'prefix-informed byte-preserving revision release only; semantic utility untested',
                   'histories': rows, 'verifier_sha256': own_sha, 'elapsed_s': time.monotonic()-started}
        write(args.output / 'summary.json', summary)
        (args.output / 'verifier_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({k:v for k,v in summary.items() if k != 'histories'}, ensure_ascii=False), flush=True)
    except Exception:
        write(args.output / 'failure.json', {'status': 'failed_independent_review', 'release_allowed': False,
                                            'traceback': traceback.format_exc()})
        raise
    finally:
        write(args.output / 'artifact_hashes.json', {p.name: sha(p) for p in args.output.iterdir()
                                                    if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
