#!/usr/bin/env python3
"""Compare saved V20 paired nonsemantic observations; no world or mapper.

Exact byte identity is the declared pairing test. Numerical diagnostics never
relax that rule. Old evidence is read only; each receipt is newly created.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = ('frame__depth_m', 'scan__ranges_m', 'frame__intrinsic',
          'frame__world_from_camera', 'scan__world_from_laser',
          'frame__timestamp_s', 'scan__timestamp_s', 'scan__angle_min_rad',
          'scan__angle_increment_rad', 'scan__range_max_m')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = f'{value.dtype.str}:{value.shape}:'.encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compare_arrays(a, b, *, depth=False):
    same_layout = a.shape == b.shape and a.dtype == b.dtype
    row = dict(shape_a=list(a.shape), shape_b=list(b.shape), dtype_a=str(a.dtype), dtype_b=str(b.dtype),
        sha256_a=array_hash(a), sha256_b=array_hash(b), same_layout=same_layout)
    row['strict_equal'] = row['sha256_a'] == row['sha256_b']
    if a.shape != b.shape:
        return row
    finite_a, finite_b = np.isfinite(a), np.isfinite(b)
    finite = finite_a & finite_b
    numerical_equal = (a == b) | (np.isnan(a) & np.isnan(b))
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    row.update(numeric_equal=bool(numerical_equal.all()),
        numeric_different_elements=int(np.count_nonzero(~numerical_equal)),
        finite_mask_different_elements=int(np.count_nonzero(finite_a != finite_b)),
        max_abs_difference_common_finite=float(difference[finite].max()) if finite.any() else None)
    if depth:
        valid_a, valid_b = finite_a & (a > 0), finite_b & (b > 0)
        common = valid_a & valid_b
        row.update(valid_depth_definition='finite and greater than zero',
            valid_pixels_a=int(valid_a.sum()), valid_pixels_b=int(valid_b.sum()),
            valid_mask_different_pixels=int(np.count_nonzero(valid_a != valid_b)),
            changed_common_valid_pixels=int(np.count_nonzero(common & ~numerical_equal)),
            max_abs_difference_common_valid_m=float(difference[common].max()) if common.any() else None)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        raise ValueError('New audit must be outside historical evidence')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('result.json', 'frames.json', 'provenance.json', 'artifact_hashes.json'):
        if (output / name).exists():
            raise FileExistsError('Refusing to overwrite audit: ' + name)
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['mode'] != 'N':
        raise ValueError('Completed V20 pure N evidence required')
    inventory = json.loads((source / 'artifact_hashes.json').read_text())
    used = {}

    def checked(name):
        path = source / name
        value = sha(path)
        if name not in inventory or value != inventory[name]:
            raise ValueError('Historical artifact missing or changed: ' + name)
        used[name] = value
        return path

    checked('manifest.json')
    groups = {}
    for case in manifest['cases']:
        groups.setdefault(case['parent'], []).append(case)
    summaries, all_frames = [], {}
    for parent, cases in groups.items():
        cases = sorted(cases, key=lambda case: case['assignment'])
        if len(cases) != 2 or [c['assignment'] for c in cases] != ['A_complex_B_simple', 'A_simple_B_complex']:
            raise ValueError('Exactly two declared assignments per parent required')
        results, selection_actions = [], []
        for case in cases:
            folder = f'case_{case["index"]:02d}'
            result = json.loads(checked(folder + '/result.json').read_text())
            verification = json.loads(checked(folder + '/verification.json').read_text())
            if verification['status'] != 'passed':
                raise ValueError('Original independent replay must have passed')
            results.append(result)
            calls = json.loads(gzip.decompress(checked(folder + '/module_calls.json.gz').read_bytes()))
            selection_actions.append([row['action_id'] for row in calls if row['method'] == 'select_topo_target'])
        if results[0]['paid_actions'] != results[1]['paid_actions']:
            raise ValueError('Paired recordings have different lengths')
        frames = []
        prefix_motion_equal = True
        for action_id in range(results[0]['paid_actions'] + 1):
            arrays, metas = [], []
            for case, result in zip(cases, results):
                name = f'case_{case["index"]:02d}/packets/{action_id:04d}.npz'
                with np.load(checked(name), allow_pickle=False) as data:
                    meta = json.loads(str(data['metadata'].item()))
                    values = {name: data[name].copy() for name in FIELDS}
                if meta['action_id'] != action_id:
                    raise ValueError('Recorded action id mismatch')
                if action_id:
                    expected = result['actions'][action_id - 1]
                    if meta['action'] != expected['action'] or [*meta['position'], meta['heading']] != expected['pose']:
                        raise ValueError('Recorded raw action/pose differs from result')
                arrays.append(values); metas.append(meta)
            action_equal = metas[0]['action'] == metas[1]['action']
            pose_equal = (metas[0]['position'], metas[0]['heading']) == (metas[1]['position'], metas[1]['heading'])
            prefix_motion_equal = prefix_motion_equal and action_equal and pose_equal
            fields = {name: compare_arrays(arrays[0][name], arrays[1][name], depth=name == 'frame__depth_m') for name in FIELDS}
            frames.append(dict(action_id=action_id, action_a=metas[0]['action'], action_b=metas[1]['action'],
                pose_a=[*metas[0]['position'], metas[0]['heading']], pose_b=[*metas[1]['position'], metas[1]['heading']],
                current_action_equal=action_equal, current_pose_equal=pose_equal,
                whole_prefix_action_pose_equal=prefix_motion_equal,
                strict_nonsemantic_equal=all(row['strict_equal'] for row in fields.values()),
                combined_nonsemantic_sha256_a=digest({k: v['sha256_a'] for k, v in fields.items()}),
                combined_nonsemantic_sha256_b=digest({k: v['sha256_b'] for k, v in fields.items()}),
                differing_fields=[k for k, v in fields.items() if not v['strict_equal']], fields=fields))
        all_frames[parent] = frames
        first_motion = next((r['action_id'] for r in frames if not r['whole_prefix_action_pose_equal']), None)
        first_nonsemantic = next((r['action_id'] for r in frames if r['whole_prefix_action_pose_equal'] and not r['strict_nonsemantic_equal']), None)
        strict_prefix_end = -1
        for row in frames:
            if not row['whole_prefix_action_pose_equal'] or not row['strict_nonsemantic_equal']:
                break
            strict_prefix_end = row['action_id']
        checkpoints = {0, 5, 10}
        first_globals = [next((step for step in steps if step > 0), None) for steps in selection_actions]
        checkpoints.update(step for step in first_globals if step is not None)
        if first_nonsemantic is not None:
            checkpoints.add(first_nonsemantic)
        def compact(row):
            return {k: row[k] for k in ('action_id', 'current_action_equal', 'current_pose_equal',
                'whole_prefix_action_pose_equal', 'strict_nonsemantic_equal', 'differing_fields', 'fields')}
        summaries.append(dict(parent=parent, cases=cases, frame_pairs=len(frames),
            first_action_or_pose_difference=first_motion,
            first_nonsemantic_difference_with_identical_motion_prefix=first_nonsemantic,
            entire_history_strictly_paired_through_action=strict_prefix_end,
            first_nonsemantic_difference_detail=None if first_nonsemantic is None else compact(frames[first_nonsemantic]),
            global_selection_actions_a=selection_actions[0], global_selection_actions_b=selection_actions[1],
            first_post_initial_global_selection_actions=first_globals,
            checkpoints=[compact(frames[step]) for step in sorted(checkpoints)],
            exact_nonsemantic_pairs_with_entire_motion_prefix_equal=sum(r['whole_prefix_action_pose_equal'] and r['strict_nonsemantic_equal'] for r in frames)))
    result = dict(status='complete', scope='saved-packet nonsemantic pairing diagnosis only',
        source_root=str(source), pairing_rule='Exact dtype, shape and bytes; no tolerance substitution',
        compared_fields=list(FIELDS), excluded_fields=['frame__color_rgb', 'frame__semantic'],
        numeric_diagnostics_do_not_relax_pairing=True, pairs=summaries,
        no_world_or_mapper_imported=True, new_physical_actions=0, new_sensor_frames=0,
        semantic_information_gain_proven=False,
        interpretation='Identical complete nonsemantic prefixes are necessary for strict information pairing, not sufficient evidence of undisclosed valuable structure or feasible quality actions.')
    write_new(output / 'frames.json', all_frames)
    write_new(output / 'result.json', result)
    write_new(output / 'provenance.json', dict(source_root=str(source),
        source_inventory_sha256=sha(source / 'artifact_hashes.json'), inputs_sha256=used,
        source_evidence_read_only=True, input_files_verified=len(used),
        script_sha256=sha(Path(__file__).resolve())))
    write_new(output / 'artifact_hashes.json', {p.name: sha(p) for p in sorted(output.iterdir())
        if p.is_file() and p.name != 'artifact_hashes.json'})
    for pair in summaries:
        print(json.dumps({k: pair[k] for k in ('parent', 'frame_pairs',
            'first_action_or_pose_difference', 'first_nonsemantic_difference_with_identical_motion_prefix',
            'entire_history_strictly_paired_through_action', 'first_post_initial_global_selection_actions')}, sort_keys=True))


if __name__ == '__main__':
    main()
