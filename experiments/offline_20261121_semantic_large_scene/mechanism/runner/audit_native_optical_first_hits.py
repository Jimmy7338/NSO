"""Post-hoc optical-only check of the already recorded cold-start native paths."""

import argparse
import hashlib
import json

from collections import defaultdict
import numpy as np

import run

SAMPLE_BIN_M = .05


def first_hit_bins(world, position, heading):
    from env.virtual3d import camera_pose

    config = world.config
    pose = camera_pose(tuple(position), heading, config, world.shape[0])
    k = world.intrinsic
    result = world._renderer.render(k, pose, config.width_px, config.height_px,
                                    max_depth_m=config.max_depth_m, min_depth_m=.15)
    depth = result['depth_m']
    ids = result['reference_geometry_id']
    response = {}
    for gid in np.unique(ids[depth > 0]):
        name = world._node_for_gid[int(gid)]
        if not name.startswith(('left/', 'right/')):
            continue
        r, c = np.nonzero((ids == gid) & (depth > 0))
        local = np.column_stack(((c-k[0, 2])/k[0, 0], (r-k[1, 2])/k[1, 1], np.ones(len(r))))
        xyz = (local * depth[r, c, None]) @ pose[:3, :3].T + pose[:3, 3]
        integer_bins = np.rint(xyz / SAMPLE_BIN_M).astype(np.int32)
        response[name] = {tuple(map(int, point)) for point in integer_bins}
    return response


def audit(layout):
    choices = run.HERE / 'layouts' / layout / 'runs/G/attempt_001/choices.json'
    plan = run.read(choices)[0]
    if 'X' not in plan['outputs']['scores']:
        return dict(layout=layout, status='first_choice_did_not_trigger')
    first = run.read(run.prepared(layout) / 'initial_state.json')
    old, _ = run.install()
    world, _, packet, _, _, _, _ = old.bootstrap(layout, 'G')
    assert world.step_count == 0 and packet.sha256() == first['initial_packet_sha256']
    geometry = run.read(run.SCENES / 'prepared' / layout / 'receipt.json')
    camera_cache = {}

    def render(state):
        key = tuple(map(int, state))
        if key not in camera_cache:
            camera_cache[key] = first_hit_bins(world, key[:2], key[2])
        return camera_cache[key]

    before = render(first['anchor'])
    assert not any('/interior_' in part for part in before)
    relevant = plan['outputs']['candidates']
    results = []
    for route in relevant:
        assert route['outbound_states'][0] == first['anchor']
        new = defaultdict(set)
        first_new = {}
        for i, state in enumerate(route['outbound_states'][1:], 1):
            for part, bins in render(state).items():
                candidate_new = bins - before.get(part, set())
                if candidate_new:
                    new[part] |= candidate_new
                    first_new.setdefault(part, i)
        interior = {part: len(x) for part, x in sorted(new.items()) if '/interior_' in part}
        results.append(dict(candidate_id=route['candidate_id'], role=route['group'],
            outbound_actions=len(route['outbound_actions']), reserved_return_actions=len(route['return_actions']),
            new_interior_part_sample_bins=interior, total_open_interior_part_bins=sum(interior.values()),
            all_new_facility_part_sample_bins={part: len(x) for part, x in sorted(new.items())},
            first_new_interior_hit_by_part={part: first_new[part] for part in interior}))
    assert world.step_count == 0
    return dict(status='audited_pure_first_hit_visibility_in_all_native_outbound_paths', layout=layout,
        script_sha256=hashlib.sha256(run.Path(__file__).read_bytes()).hexdigest(),
        expected_informative_slot=geometry['route_evaluation']['expected_diagnostic_side'],
        initial_paid_actions=0, additional_actor_actions=0, complete_outbound_path=True,
        part_sample_bin_size_m=SAMPLE_BIN_M, distinct_camera_poses_rendered=len(camera_cache),
        start_visible_part_sample_bins={part: len(x) for part, x in sorted(before.items())},
        candidates=results, interpretation='Parts and occupied surface sample bins after backing out the start first-hit samples. Not a mesh accuracy or recovered surface area measure.',
        truth_used_by_controlled_actor=False, supplies_data_to_controlled_actor=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=run.LAYOUTS)
    args = parser.parse_args()
    print(json.dumps(audit(args.layout), indent=2, allow_nan=False))
