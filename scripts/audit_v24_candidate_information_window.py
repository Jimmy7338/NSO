#!/usr/bin/env python3
"""Static clean-ray disclosure audit of already saved observed candidates.

This evaluator-only diagnostic creates no SensorPacket, motion, map or quality
result. It tests the actual N/G first choices and the existing aperture-view
role for each marked observed asset, selected before reading future rays.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import numpy as np
from env.facility_choice_v24_1 import FacilityChoiceWorldV24_1
from env.canonical_rgbd_v15 import render_axial_depth
from env.virtual3d import camera_pose

PREFIX = ROOT/'audit_results/facility_choice_v24_paid_prefix_20260915'
READY = ROOT/'audit_results/facility_runtime_v24_readiness_20260915'
OUTPUT = ROOT/'audit_results/facility_choice_v24_candidate_window_20260915'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    if len(data) > 1024**2 or shutil.disk_usage(OUTPUT).free < 32*1024**2 + len(data) + 65536:
        raise OSError('Static diagnostic output capacity exceeded')
    with path.open('xb') as stream:
        stream.write(data)


class StaticWorld(FacilityChoiceWorldV24_1):
    def step(self, *args, **kwargs):
        raise RuntimeError('Static audit forbids world.step')

    def sense(self, *args, **kwargs):
        raise RuntimeError('Static audit forbids world.sense')

    def scan(self, *args, **kwargs):
        raise RuntimeError('Static audit forbids world.scan')


def main():
    if OUTPUT.exists():
        raise ValueError('Existing evidence root must not be reused')
    prefix = read(PREFIX/'manifest.json')
    ready = read(READY/'result.json')
    assert prefix['status'] == ready['status'] == 'complete'
    assert all(p['passed'] for p in ready['pair_checks'])
    frozen = prefix['source_sha256']
    for name, expected in frozen.items():
        assert sha(ROOT/name) == expected, name
    input_hashes = {}
    for root in (PREFIX, READY):
        inventory = read(root/'artifact_hashes.json')
        for name, value in inventory.items():
            expected = value['sha256'] if isinstance(value, dict) else value
            assert sha(root/name) == expected, name
        input_hashes[str(root.relative_to(ROOT))] = sha(root/'artifact_hashes.json')
    own_hash = sha(Path(__file__).resolve())
    protocol = dict(
        chosen_route_rule='For each parent use saved N first choice, G first choice, and existing asset_<observed_index>_aperture_view for each marked asset; never choose by future rays or quality.',
        history='Use the first already paired arrangement and require each chosen candidate to be identical in the other arrangement.',
        disclosure='Count unequal clean axial-depth pixels at each stored round-trip pose; preserve first differing local and prefix-plus-local action index.',
        no_new_sensor_packets=True, no_motion=True, no_noise_or_tsdf=True,
        no_quality_or_semantic_outcome=True, no_online_privileged_inputs=True,
        zero_disclosure_scope='Only the sampled poses and ideal clean depth; does not prove all cheap views uninformative.',
        some_disclosure_scope='Geometric information is available, not proof that a policy uses it or that quality improves.',
        output_limit_per_file_bytes=1024**2, reserve_bytes=32*1024**2)
    OUTPUT.mkdir()
    write(OUTPUT/'protocol.json', protocol)
    write(OUTPUT/'source_sha256.json', {'script': own_hash, 'prefix_sources': frozen, 'input_inventories': input_hashes})
    shutil.copyfile(Path(__file__).resolve(), OUTPUT/'audit_script.py')
    try:
        results = []
        for parent, index in [('D24-P00', 0), ('D24-P01', 2)]:
            n = read(READY/f'case_{index:02d}_N.json')
            g = read(READY/f'case_{index:02d}_G.json')
            other = read(READY/f'case_{index+1:02d}_G.json')
            assert g['selection']['candidates'] == other['selection']['candidates']
            chosen = [('N_first_choice', n['selected']), ('G_first_choice', g['selected'])]
            for asset in g['marked_asset_indices']:
                found = [c for c in g['selection']['candidates'] if c['group'] == f'asset_{asset}_aperture_view']
                assert len(found) == 1
                chosen.append((f'observed_asset_{asset}_aperture', found[0]))
            worlds = [StaticWorld(parent=parent, assignment=a) for a in ('A_complex_B_simple', 'A_simple_B_complex')]
            boxes = [np.asarray(w._solid_primitives, float)[:, :6] for w in worlds]
            cache = {}
            route_rows = []
            for role, route in chosen:
                changes = []
                for local_id, state in enumerate(route['states']):
                    key = tuple(state)
                    if key not in cache:
                        images = []
                        for world, primitives in zip(worlds, boxes):
                            pose = camera_pose(state[:2], state[2], world.config, world.shape[0])
                            depth, _ = render_axial_depth(primitives, world.intrinsic, pose,
                                world.config.height_px, world.config.width_px, world.config.max_depth_m)
                            images.append(depth)
                        cache[key] = int(np.count_nonzero(images[0] != images[1]))
                    if cache[key]:
                        changes.append(dict(local_action_id=local_id,
                            total_action_id=g['paid_prefix_actions']+local_id, pose=state,
                            differing_depth_pixels=cache[key], on_outbound=local_id <= route['outbound_cost']))
                route_rows.append(dict(role=role, candidate_id=route['candidate_id'], group=route['group'],
                    asset_index=route['asset_index'], proposed_round_trip_cost=route['cost'],
                    proposed_outbound_cost=route['outbound_cost'], checked_poses=len(route['states']),
                    geometry_disclosed=bool(changes), first_difference=changes[0] if changes else None,
                    differing_poses=changes, actually_executed=False))
            assert all(w.step_count == 0 for w in worlds)
            results.append(dict(parent=parent, unique_static_poses=len(cache), routes=route_rows))
            print(parent, [(r['role'], r['first_difference']) for r in route_rows], flush=True)
        assert sha(Path(__file__).resolve()) == own_hash
        assert all(sha(ROOT/name) == expected for name, expected in frozen.items())
        assert all(sha(ROOT/name/'artifact_hashes.json') == h for name, h in input_hashes.items())
        write(OUTPUT/'result.json', dict(status='complete', parents=results, protocol=protocol,
            new_physical_actions=0, new_sensor_packets=0, new_tsdf_fusions=0))
    except BaseException:
        write(OUTPUT/'failure.json', dict(status='failed', traceback=traceback.format_exc()))
        raise
    finally:
        write(OUTPUT/'artifact_hashes.json', {str(p.relative_to(OUTPUT)):sha(p) for p in sorted(OUTPUT.iterdir())
            if p.is_file() and p.name != 'artifact_hashes.json'})


if __name__ == '__main__':
    main()
