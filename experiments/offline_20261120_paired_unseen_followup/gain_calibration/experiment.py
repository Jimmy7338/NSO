"""Freeze, run and bind the observed-data side of the gain calibration."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
B11 = BASE / 'paired_unseen_gain_20261120'
WORKSPACE = BASE.parent / 'workspaces/nso_research_20261029/NSO'
sys.path[:0] = [str(B11 / 'execution'), str(WORKSPACE)]

import runtime_binding as binding
from full_posterior_forecast import FullPosteriorForecast
from short_branches import branch, mesh_resource, verify as verify_b11
from world import LAYOUTS, PROTOCOL, make_world


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    verify_b11()
    out = HERE / 'freeze.json'
    if out.exists():
        raise FileExistsError(out)
    sources = [HERE / name for name in ('world.py', 'experiment.py', 'evaluate.py', 'protocol.json', 'run_split.py')]
    sources += [B11 / 'execution/freeze.json', B11 / 'execution/full_posterior_forecast.py',
                B11 / 'scene/unseen_world.py', BASE / 'dual_template_exploration_20261119/audit/surface_metrics.py',
                BASE / 'semantic_scene_redesign_20261119/exploration_evaluation/surface_reference.py']
    binding.write(out, dict(status='frozen_before_calibration_and_heldout_acquisition',
                             source_sha256={str(path): sha(path) for path in sources},
                             created_unix_s=time.time()))


def verify():
    manifest = json.loads((HERE / 'freeze.json').read_text())
    for path, expected in manifest['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Frozen source mismatch: {path}')


def install():
    # This is a private CLI process. Only the environment and fixed prefix
    # hooks are replaced. Gate checks and predictor logic come from B11.
    frozen_protocol = binding.protocol()
    for name in LAYOUTS:
        frozen_protocol['layouts'][name] = {'scene_id': f'gain-calibration-{name}'}
    binding.protocol = lambda: frozen_protocol

    def one_packet(layout):
        from nso.cpu_sensor_contract_v10 import GridTransform
        world = make_world(layout)
        transform = GridTransform(tuple(world.shape), world.config.resolution_m)
        return world, transform, binding.old.packet(world, transform, world.sense(), layout)

    binding.world_and_packet = one_packet
    binding.scripted_prefix = lambda layout, _prefix: (['forward'] * LAYOUTS[layout]['prefix'], None)
    binding.install()


def route_states(start, actions):
    from utils.grid_geometry import DIRECTIONS
    row, col, heading = start
    states = [list(start)]
    for action in actions:
        if action == 'forward':
            dr, dc = DIRECTIONS[heading]
            row, col = row + dr, col + dc
        else:
            heading = (heading + (1 if action == 'right' else -1)) % 4
        states.append([int(row), int(col), int(heading)])
    return states


def support_features(engine):
    source = list(engine.mapper.quality.items())
    keys = np.asarray([key for key, _ in source], dtype=np.int64).reshape(-1, 3)
    tracks = engine.mapper.instance_support_index.memberships_for_keys(keys)
    points = np.asarray([row['point'] for _, row in source]).reshape(-1, 3)
    support = {}
    counts = {}
    for oi, item in enumerate(engine.bank.objects):
        track = item['support_track']
        cloud = points[tracks == track]
        if len(cloud) == 0:
            raise AssertionError('Fitted object missing acquired support')
        support[oi] = cKDTree(cloud)
        counts[oi] = dict(track_id=int(track), observed_quality_points=int(len(cloud)))
    return support, counts


def split_acquisition_gain(engine, result, support):
    near_total = far_total = 0.0
    tolerance = PROTOCOL['evaluation']['distance_tolerance_m']
    support_radius = PROTOCOL['method']['support_neighborhood_m']
    fitted = engine.data['points']
    terms = {entry['hypothesis_id']: entry for entry in result['terms']}
    weights = engine.models['S'].base_weights
    for hid, (oi, _, _, lo, hi) in enumerate(engine.data['slices']):
        if hid not in terms:
            continue
        entry = terms[hid]
        expected = engine.after_cache[entry['future_sequence_sha256']]
        d_after = expected['distance'][lo:hi]
        new = (engine.before_distance[lo:hi] > np.float32(tolerance))
        new &= (d_after <= np.float32(tolerance)) & ~engine.data['history']['strict_free'][lo:hi]
        if not np.any(new):
            continue
        distance, _ = support[oi].query(fitted[lo:hi][new], workers=1)
        near = distance <= support_radius
        area = weights[lo:hi][new]
        near_total += float(area[near].sum())
        far_total += float(area[~near].sum())
    np.testing.assert_allclose(near_total + far_total, result['gross_m2']['S'], rtol=1.e-10, atol=1.e-10)
    return near_total, far_total


def run(layout, dest):
    verify()
    split = LAYOUTS[layout]['split']
    calibrated = HERE / 'calibrated_coefficients.json'
    if split == 'heldout':
        fit = json.loads(calibrated.read_text())
        if fit['status'] != 'calibration_coefficients_locked_before_heldout':
            raise ValueError('Held-out run requires locked calibration')
        if fit['freeze_sha256'] != sha(HERE / 'freeze.json'):
            raise ValueError('Coefficient sources do not match current calibration freeze')
        for path, expected in fit['calibration_sources_sha256'].items():
            if sha(path) != expected:
                raise ValueError(f'Calibration input changed: {path}')
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        verify_b11()
        install()
        ctx = binding.bootstrap_scripted_prefix(layout, 'frontal')
        world = ctx['world']
        start = (*world.position, world.heading)
        route_cfg = PROTOCOL['movement']
        routes = [dict(name=name, candidate_id=i, outbound_actions=route_cfg['routes'][name],
                       outbound_states=route_states(start, route_cfg['routes'][name]))
                  for i, name in enumerate(route_cfg['route_order'])]
        assert len({len(r['outbound_actions']) for r in routes}) == 1
        mapper = ctx['mapper']
        before = mesh_resource(mapper, dest / 'before_native.ply')
        cache = binding.old.method._cache(mapper, 'S_visibility')
        engine = FullPosteriorForecast(mapper, cache['models'], mapper._b11_acquired_frames, after_cache_limit=1024)
        support, counts = support_features(engine)
        predictions = []
        for route in routes:
            future = route['outbound_states'][1:]
            visibility = engine.visibility(future)
            result = engine.acquisition(future)
            near, far = split_acquisition_gain(engine, result, support)
            predictions.append(dict(id=route['candidate_id'], route=route['name'],
                                    raw_visibility_m2=visibility['S'], raw_acquisition_m2=result['gross_m2']['S'],
                                    raw_acquisition_lost_m2=result['lost_m2']['S'],
                                    support_near_m2=near, support_far_m2=far,
                                    acquisition_wall_s=result['elapsed_s'],
                                    all_positive_hypothesis_route_terms=result['counts']['hypothesis_route_terms']))
        # Future packets and private evaluation references are never provided to the forecast.
        binding.write(dest / 'predictions.json', dict(layout=layout, split=split, recorded_before_actual_branches=True,
                      initial_history_packet_sha256=[packet.sha256() for packet in ctx['packets']],
                      prefix_pose=start, modeled_support_tracks=counts,
                      correction_coefficients_sha256=sha(calibrated) if split == 'heldout' else None,
                      engine_audit=engine.assert_unchanged(), routes=routes, predictions=predictions))
        outcomes = [branch(ctx, route, dest / f'route_{route["candidate_id"]}', before) for route in routes]
        binding.write(dest / 'run_receipt.json', dict(status='all_fixed_routes_attempted', layout=layout, split=split,
                      predictions=binding.resource(dest / 'predictions.json'), before_mesh=before, branches=outcomes,
                      freeze_sha256=sha(HERE / 'freeze.json'), elapsed_s=time.monotonic()-started))
        verify()
        print(json.dumps(dict(status='fixed_routes_attempted', layout=layout, templates=len(counts),
                              elapsed_s=round(time.monotonic()-started, 2))), flush=True)
    except BaseException as exc:
        binding.write(dest / 'failure.json', dict(layout=layout, split=split, error=repr(exc),
                      traceback=traceback.format_exc(), elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run'))
    parser.add_argument('--layout', choices=tuple(LAYOUTS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    else:
        run(args.layout, args.output)
