#!/usr/bin/env python3
"""Static DEV_A geometry/evaluator integration; synthetic meshes, no reconstruction."""
import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nso.surface_evaluation_v40 import CandidateViewV40, freeze_reference_v40, evaluate_surface_v40


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def intersects_rectangle(a, b, rectangle):
    """Closed segment versus closed inflated XY rectangle, including thin walls."""
    low, high = 0., 1.
    for axis, (minimum, maximum) in enumerate(((rectangle[0], rectangle[1]),
                                               (rectangle[2], rectangle[3]))):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-12:
            if a[axis] < minimum or a[axis] > maximum:
                return False
        else:
            t0, t1 = (minimum - a[axis]) / delta, (maximum - a[axis]) / delta
            low, high = max(low, min(t0, t1)), min(high, max(t0, t1))
            if low > high:
                return False
    return True


def reachable_candidates(metadata, protocol):
    defaults = protocol['public_defaults']
    workspace = metadata['public_workspace']
    bounds = np.asarray(workspace['room_inner_bounds_xy_m'], float)
    # This start and lattice are declared before any prediction or metric is read.
    declared_start = np.asarray(workspace['start_position_world_m'], float)
    if declared_start.shape != (3,) or not np.isfinite(declared_start).all():
        raise ValueError('finite declared start position required')
    if declared_start[2] != defaults['motion']['camera_height_m']:
        raise ValueError('start camera height does not match protocol')
    start = declared_start[:2].copy()
    radius, grid_spacing = .2, 1.5
    rectangles = []
    for instance in metadata['private_instances']:
        lo, hi = instance['world_aabb_m']
        rectangles.append([lo[0]-radius, hi[0]+radius, lo[1]-radius, hi[1]+radius])
    for box in metadata['background_boxes']:
        if box[5] <= .05 or box[4] > .9:
            continue
        rectangles.append([box[0]-radius, box[1]+radius, box[2]-radius, box[3]+radius])
    axes = []
    for axis in range(2):
        first = math.ceil((bounds[0, axis] + radius - start[axis]) / grid_spacing)
        last = math.floor((bounds[1, axis] - radius - start[axis]) / grid_spacing)
        axes.append(list(range(first, last + 1)))
    cells = {(i, j): start + grid_spacing * np.array([i, j]) for i in axes[0] for j in axes[1]}
    free = {key: point for key, point in cells.items()
            if not any(intersects_rectangle(point, point, r) for r in rectangles)}
    if (0, 0) not in free:
        raise ValueError('the predeclared start is blocked; do not silently change it')
    visited, queue = {(0, 0)}, deque([(0, 0)])
    while queue:
        key = queue.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (key[0] + di, key[1] + dj)
            if neighbor in visited or neighbor not in free:
                continue
            if any(intersects_rectangle(free[key], free[neighbor], r) for r in rectangles):
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    sensor = defaults['sensor']
    views, serialized = [], []
    for index, cell in enumerate(sorted(visited)):
        position = [*free[cell], defaults['motion']['camera_height_m']]
        for yaw_deg in (0, 90, 180, 270):
            yaw = math.radians(yaw_deg)
            transform = np.eye(4)
            transform[:3, :3] = np.array([[math.sin(yaw), 0, math.cos(yaw)],
                                          [-math.cos(yaw), 0, math.sin(yaw)],
                                          [0, -1, 0]])
            transform[:3, 3] = position
            view_id = f'grid_{index:03d}_yaw_{yaw_deg:03d}'
            row = dict(intrinsic=sensor['intrinsic'], world_from_camera=transform.tolist(),
                       width=sensor['width'], height=sensor['height'],
                       near_m=sensor['depth_min_m'], far_m=sensor['depth_max_m'], view_id=view_id)
            views.append(CandidateViewV40(**row))
            serialized.append(row)
    return views, {'scope': 'private development evaluator lattice; not an implemented public navigation graph',
                   'start_xy_m': start.tolist(), 'robot_radius_m': radius,
                   'grid_spacing_m': grid_spacing, 'yaw_degrees': [0, 90, 180, 270],
                   'grid_cells': len(cells), 'collision_free_cells': len(free),
                   'reachable_cells': len(visited), 'candidate_views': serialized,
                   'collision_model': 'conservative target AABBs and background solid XY rectangles; segment-checked four-neighbor BFS',
                   'selection_uses_prediction_or_planner_trajectory': False}


def run(source, output):
    mesh_file = source / 'DEV_A_00/renderer_private/geometry.npz'
    metadata_file = source / 'DEV_A_00/evaluation_private/instances.json'
    protocol_file = ROOT / 'configs/virtual3d/v40_scene_protocol_20260920.json'
    geometry_manifest_file = source / 'manifest.json'
    geometry_manifest = json.loads(geometry_manifest_file.read_text())
    for path in (mesh_file, metadata_file):
        if sha(path) != geometry_manifest['artifact_sha256'][str(path.relative_to(source))]:
            raise ValueError('geometry / collision metadata does not match source manifest')
    metadata = json.loads(metadata_file.read_text())
    if metadata['parent_id'] != 'DEV_A_00':
        raise ValueError('wrong development parent')
    protocol = json.loads(protocol_file.read_text())
    with np.load(mesh_file, allow_pickle=False) as data:
        vertices, triangles = data['vertices'].copy(), data['triangles'].copy()
        owners = data['triangle_instance_id'].copy()
    if {item['instance_id'] for item in metadata['private_instances']} != set(owners.tolist()) - {-1}:
        raise ValueError('mesh and metadata instance inventories differ')
    for item in metadata['private_instances']:
        points = vertices[triangles[owners == item['instance_id']]].reshape(-1, 3)
        if not np.allclose(item['world_aabb_m'], [points.min(axis=0), points.max(axis=0)], rtol=0, atol=1e-10):
            raise ValueError('collision envelope does not enclose the corresponding mesh')
    output.mkdir(parents=True, exist_ok=False)
    views, candidate_record = reachable_candidates(metadata, protocol)
    write(output / 'candidate_views.json', candidate_record)
    reference = freeze_reference_v40(vertices, triangles, owners, views,
                                    sample_spacing_m=.3, seed=4001, max_samples=50000)
    reference_record = reference.manifest()
    write(output / 'reference_manifest.json', reference_record)
    np.savez_compressed(output / 'reference_surface.npz', points=reference.points,
                        instance_id=reference.point_instance_id, area_weights=reference.area_weights)
    missing_id = int(min(set(owners.tolist()) - {-1}))
    extra_vertices = np.array([[vertices[:, 0].max()+3, 1, .2],
                               [vertices[:, 0].max()+3, 3, .2],
                               [vertices[:, 0].max()+3, 3, 2.2],
                               [vertices[:, 0].max()+3, 1, 2.2]])
    extra_triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int) + len(vertices)
    fixtures = {
        'exact_complete_mesh': (vertices, triangles),
        'one_facility_deleted': (vertices, triangles[owners != missing_id]),
        'unmatched_extra_surface': (np.vstack([vertices, extra_vertices]), np.vstack([triangles, extra_triangles]))}
    rows = {}
    for label, (prediction_vertices, prediction_triangles) in fixtures.items():
        rows[label] = evaluate_surface_v40(reference, prediction_vertices, prediction_triangles,
                                          C_map=1., threshold_m=.05, sample_spacing_m=.3,
                                          seed=4002, max_samples=50000)
        write(output / (label + '.json'), rows[label])
    # These invariants test the evaluator; C_map=1 is a synthetic input, not mapping coverage.
    complete = rows['exact_complete_mesh']
    missing = rows['one_facility_deleted']
    extra = rows['unmatched_extra_surface']
    count = len(set(owners.tolist()) - {-1})
    assert abs(complete['macro_f1'] - 1.) < 1e-8, 'exact mesh must score one'
    assert abs(complete['full_mesh_global_precision'] - 1.) < 1e-8
    assert abs(missing['macro_f1'] - (count - 1) / count) < 1e-8, 'missing instance must remain in macro denominator'
    assert len(missing['per_instance']) == count
    assert next(row['f1'] for row in missing['per_instance'] if row['instance_id'] == missing_id) == 0.
    assert extra['macro_f1'] < complete['macro_f1']
    assert extra['full_mesh_global_precision'] < complete['full_mesh_global_precision']
    assert abs(extra['extra_false_positive_area_m2'] - 4.) < 1e-8, 'unmatched outside surface must be counted'
    assert all(row['reference_fingerprint'] == complete['reference_fingerprint'] for row in rows.values())
    result = {'status': 'passed',
              'scope': 'static development geometry and synthetic prediction sanity checks',
              'reference': reference_record, 'synthetically_deleted_instance': missing_id,
              'fixtures': rows, 'C_map_is_synthetic_constant': True,
              'new_worlds': 0, 'new_sensor_trajectories': 0, 'new_planner_calls': 0,
              'new_TSDF_integrations': 0, 'analytic_evaluation_calls': 3,
              'semantic_performance_claim': False, 'formal_reference_frozen': False,
              'invariants': ['exact mesh scores one', 'deleted facility retained in macro denominator',
                             'outside incorrect geometry incurs false positives', 'identical fixed reference for all fixtures'],
              'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                               (mesh_file, metadata_file, geometry_manifest_file, protocol_file, Path(__file__),
                                ROOT/'nso/surface_evaluation_v40.py')}}
    write(output / 'result.json', result)
    write(output / 'artifact_sha256.json', {p.name: sha(p) for p in sorted(output.iterdir())})
    print(json.dumps({'status': result['status'], 'candidate_views': len(views),
                      'reference': reference_record, 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'audit_results/v40_p1_development_geometry_20260920')
    parser.add_argument('--output', type=Path, default=ROOT / 'audit_results/v40_p1_surface_integration_20260920')
    args = parser.parse_args()
    run(args.source.resolve(), args.output.resolve())
