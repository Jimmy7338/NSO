"""Read-only saved-episode evaluation against a presealed development reference.

No World, sensor, mapper or controller is constructed here.  C_nav measures
correctly observed free area on a fixed conservative navigable-floor domain;
it is neither SLAM pose accuracy nor generic occupancy-map accuracy.
"""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from nso.surface_evaluation_v40 import (CandidateViewV40, ReferenceSurfaceV40,
    evaluate_surface_v40, freeze_reference_v40)


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = 'audit_results/v40_p1_development_geometry_20260920'
P1_SEAL = 'audit_results/v40_p1_static_release_20260920/manifest.json'
P0_SEAL = 'audit_results/v40_p0_20260920/manifest.json'
SCENE_PROTOCOL = 'configs/virtual3d/v40_scene_protocol_20260920.json'
REFERENCE_ARRAYS = ('vertices', 'triangles', 'triangle_instance_id',
    'triangle_normals', 'points', 'point_instance_id', 'area_weights')
REFERENCE_FILES = {'reference.json', 'surface.npz', 'candidate_views.json',
                   'coverage_domain.npz'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(f'{value.dtype.str}:{value.shape}:'.encode()+value.tobytes()).hexdigest()


def _json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def _write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _frozen_inputs(asset_id, source_root):
    if not isinstance(asset_id, str) or not re.fullmatch(r'DEV_[A-F]_00', asset_id):
        raise ValueError('development asset ID required; held-out assets are not accepted')
    source_root = Path(source_root).resolve()
    seals = [_json(source_root/name)['files'] for name in (P0_SEAL, P1_SEAL)]
    names = [f'{GEOMETRY_ROOT}/{asset_id}/{suffix}' for suffix in
        ('renderer_private/geometry.npz', 'evaluation_private/instances.json', 'public_workspace.json')]
    names += [SCENE_PROTOCOL, 'nso/surface_evaluation_v40.py',
              'scripts/verify_development_surface_pipeline_v40.py']
    hashes = {}
    for name in names:
        rows = [seal[name] for seal in seals if name in seal]
        path = source_root/name
        if not rows or not path.is_file() or path.is_symlink():
            raise ValueError('missing frozen reference input: '+name)
        actual = sha256(path)
        if any(row['sha256'] != actual or row['bytes'] != path.stat().st_size for row in rows):
            raise ValueError('frozen input differs from release seal: '+name)
        hashes[name] = actual
    return source_root, hashes


def conservative_floor_domain_v44(metadata, *, resolution_m=.1):
    """Fixed start-connected grid of safe robot centers, independent of a route.

    Each retained center represents one resolution**2 area quadrature weight.
    Inflated AABBs are conservative; cell-center area quadrature is approximate.
    Continuous edge checks prevent connectivity through sub-cell thin walls.
    """
    if resolution_m != .1 or isinstance(resolution_m, bool):
        raise ValueError('V44 development coverage resolution is fixed at .1 m')
    from scripts.verify_development_surface_pipeline_v40 import intersects_rectangle
    workspace = metadata['public_workspace']
    bounds = np.asarray(workspace['bounds_xy_m'], dtype=float)
    inner = np.asarray(workspace['room_inner_bounds_xy_m'], dtype=float)
    start = np.asarray(workspace['start_position_world_m'], dtype=float)
    if (bounds.shape != (2, 2) or inner.shape != (2, 2) or start.shape != (3,)
            or not np.isfinite(bounds).all() or not np.isfinite(inner).all()
            or not np.isfinite(start).all() or np.any(bounds[1] <= bounds[0])
            or np.any(inner[0] < bounds[0]) or np.any(inner[1] > bounds[1])
            or np.any(inner[1] <= inner[0])):
        raise ValueError('ordered workspace bounds and finite start required')
    radius = .2
    rectangles = []
    for row in metadata['private_instances']:
        box = np.asarray(row['world_aabb_m'], dtype=float)
        if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] <= box[0]):
            raise ValueError('finite ordered facility AABB required')
        rectangles.append([box[0, 0]-radius, box[1, 0]+radius,
                           box[0, 1]-radius, box[1, 1]+radius])
    for box in metadata['background_boxes']:
        if len(box) != 6 or not np.isfinite(box).all() or any(box[i] >= box[i+1] for i in (0, 2, 4)):
            raise ValueError('finite ordered background box required')
        if box[5] > .05 and box[4] <= .9:
            rectangles.append([box[0]-radius, box[1]+radius, box[2]-radius, box[3]+radius])
    columns, rows = np.ceil((bounds[1]-bounds[0])/resolution_m).astype(int)
    if rows*columns > 4_000_000:
        raise ValueError('coverage grid exceeds mapper bound')
    ys, xs = np.indices((rows, columns))
    centers = np.stack((bounds[0, 0]+(xs+.5)*resolution_m,
                        bounds[0, 1]+(ys+.5)*resolution_m), axis=-1)
    free = np.all((centers > inner[0]+radius) & (centers < inner[1]-radius), axis=-1)
    for x0, x1, y0, y1 in rectangles:
        free &= ~((centers[..., 0] >= x0) & (centers[..., 0] <= x1)
                  & (centers[..., 1] >= y0) & (centers[..., 1] <= y1))
    col, row = np.floor((start[:2]-bounds[0])/resolution_m).astype(int)
    if not (0 <= row < rows and 0 <= col < columns and free[row, col]):
        raise ValueError('predeclared start cell is not free; no replacement start allowed')
    if any(intersects_rectangle(start[:2], centers[row, col], box) for box in rectangles):
        raise ValueError('declared start cannot reach its coverage lattice center')
    domain = np.zeros(free.shape, dtype=bool)
    domain[row, col] = True
    queue = deque([(row, col)])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = row+dr, col+dc
            if not (0 <= nr < rows and 0 <= nc < columns) or not free[nr, nc] or domain[nr, nc]:
                continue
            if any(intersects_rectangle(centers[row, col], centers[nr, nc], box) for box in rectangles):
                continue
            domain[nr, nc] = True
            queue.append((nr, nc))
    return domain, dict(schema='v44.conservative_navigable_floor.v1',
        shape=[int(rows), int(columns)], resolution_m=resolution_m,
        origin_xy_m=bounds[0].tolist(), robot_radius_m=radius,
        denominator_cells=int(domain.sum()), denominator_area_m2=float(domain.sum()*resolution_m**2),
        denominator_mask_sha256=_array_sha(domain),
        definition='start-connected conservative traversable robot-center cells; all fixed task instances excluded as obstacles',
        observable_definition='each free domain center is reachable in the static conservative 4-neighbor grid; sensor coverage still requires saved measured-free belief',
        approximation='cell-center area quadrature on inflated facility AABBs and background boxes; not exact free-floor polygons or a sensor visibility union',
        uses_planner_trajectory=False, uses_detected_instances=False,
        disconnected_free_cells=int(free.sum()-domain.sum()),
        grid_convention='row increases with world y; column increases with world x; origin is lower-left boundary')


def measure_navigation_coverage_v44(belief, domain, descriptor):
    belief, domain = np.asarray(belief), np.asarray(domain)
    if (belief.dtype.kind not in 'iu' or domain.dtype != np.bool_ or belief.shape != domain.shape
            or list(belief.shape) != descriptor['shape'] or not np.isin(belief, [-1, 0, 1]).all()
            or _array_sha(domain) != descriptor['denominator_mask_sha256']
            or int(domain.sum()) != descriptor['denominator_cells'] or not domain.any()):
        raise ValueError('bound occupancy and fixed nonempty floor domain required')
    denominator = int(domain.sum())
    correct = int(np.count_nonzero(domain & (belief == 0)))
    touched = int(np.count_nonzero(domain & (belief >= 0)))
    incorrect_occupied = int(np.count_nonzero(domain & (belief == 1)))
    return dict(metric='C_nav', C_nav=correct/denominator,
        correctly_measured_free_cells=correct, sensor_touched_domain_cells=touched,
        domain_false_occupied_cells=incorrect_occupied,
        touched_domain_fraction=touched/denominator,
        correctly_measured_free_area_m2=correct*descriptor['resolution_m']**2,
        denominator=descriptor, last_frame_observed_mask_used=False,
        interpretation='correct measured-free coverage of conservative navigable floor, not generic map accuracy')


def prepare_reference_v44(asset_id, output, *, source_root=ROOT):
    """Create one route-independent reference BEFORE episode results exist."""
    source_root, input_sha = _frozen_inputs(asset_id, source_root)
    from scripts.verify_development_surface_pipeline_v40 import reachable_candidates
    base = source_root/GEOMETRY_ROOT/asset_id
    metadata = _json(base/'evaluation_private/instances.json')
    workspace = _json(base/'public_workspace.json')
    if metadata.get('parent_id') != asset_id or metadata['public_workspace'] != workspace:
        raise ValueError('reference metadata/workspace identity mismatch')
    protocol = _json(source_root/SCENE_PROTOCOL)
    with np.load(base/'renderer_private/geometry.npz', allow_pickle=False) as data:
        vertices, triangles = data['vertices'].copy(), data['triangles'].copy()
        owners = data['triangle_instance_id'].copy()
    target_ids = sorted(row['instance_id'] for row in metadata['private_instances'])
    if target_ids != sorted(set(owners.tolist())-{-1}) or len(set(target_ids)) != len(target_ids):
        raise ValueError('all task facilities must occur exactly once in fixed target inventory')
    views, candidates = reachable_candidates(metadata, protocol)
    reference = freeze_reference_v40(vertices, triangles, owners, views,
        sample_spacing_m=.3, seed=4001, max_samples=50000)
    reference_manifest = reference.manifest()
    if reference_manifest['target_instances'] != target_ids:
        raise ValueError('an unobserved facility must never be dropped from macro denominator')
    if asset_id == 'DEV_A_00':
        legacy_path = source_root/'audit_results/v40_p1_surface_integration_20260920/reference_manifest.json'
        expected = _json(source_root/P1_SEAL)['files'][str(legacy_path.relative_to(source_root))]
        if sha256(legacy_path) != expected['sha256'] or _json(legacy_path)['fingerprint'] != reference.fingerprint:
            raise ValueError('DEV_A reference differs from previously sealed static integration reference')
    domain, coverage = conservative_floor_domain_v44(metadata)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'surface.npz', **{key: getattr(reference, key) for key in REFERENCE_ARRAYS})
    np.savez_compressed(output/'coverage_domain.npz', domain=domain)
    _write_json(output/'candidate_views.json', candidates)
    record = dict(schema='v44.offline_development_reference.v1', asset_id=asset_id,
        surface=reference_manifest, coverage=coverage, input_sha256=input_sha,
        generator_source_sha256=sha256(Path(__file__)),
        surface_evaluator_sha256=sha256(source_root/'nso/surface_evaluation_v40.py'),
        target_instance_inventory=target_ids, development_only=True,
        formal_reference_frozen=False, new_worlds=0, new_sensor_packets=0,
        new_trajectories=0, new_tsdf_integrations=0,
        prediction_selection_or_trajectory_used=False,
        predeclaration_requirement='seal this manifest SHA256 before any corresponding episode, then supply that SHA to evaluation',
        surface_sampling='fixed V40 1.5 m reachable lattice, four yaw views, 0.3 m area quadrature, seed 4001; sparse visibility approximation')
    _write_json(output/'reference.json', record)
    _write_json(output/'manifest.json', dict(schema='v44.offline_reference_manifest.v1',
        asset_id=asset_id, files={name: dict(sha256=sha256(output/name), bytes=(output/name).stat().st_size)
                                for name in sorted(REFERENCE_FILES)}))
    return dict(asset_id=asset_id, output=str(output), manifest_sha256=sha256(output/'manifest.json'),
                target_instances=len(target_ids), observable_samples=len(reference.points), coverage=coverage,
                new_worlds=0, new_sensor_packets=0, new_tsdf_integrations=0)


def load_reference_v44(root, *, manifest_sha256, asset_id, source_root=ROOT):
    root = Path(root).resolve()
    if not isinstance(manifest_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', manifest_sha256):
        raise ValueError('explicit presealed reference manifest SHA256 required')
    manifest_path = root/'manifest.json'
    if manifest_path.is_symlink() or sha256(manifest_path) != manifest_sha256:
        raise ValueError('reference manifest does not match predeclared SHA256')
    manifest = _json(manifest_path)
    if (manifest.get('schema') != 'v44.offline_reference_manifest.v1'
            or manifest.get('asset_id') != asset_id or set(manifest.get('files', {})) != REFERENCE_FILES):
        raise ValueError('reference manifest schema, identity or inventory mismatch')
    for name, row in manifest['files'].items():
        path = root/name
        if path.is_symlink() or not path.is_file() or sha256(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('reference artifact hash mismatch: '+name)
    source_root, inputs = _frozen_inputs(asset_id, source_root)
    record = _json(root/'reference.json')
    if (record.get('schema') != 'v44.offline_development_reference.v1'
            or record.get('asset_id') != asset_id or record.get('input_sha256') != inputs
            or record.get('generator_source_sha256') != sha256(Path(__file__))):
        raise ValueError('reference source/asset binding mismatch')
    candidates = _json(root/'candidate_views.json')
    views = tuple(CandidateViewV40(**row) for row in candidates['candidate_views'])
    with np.load(root/'surface.npz', allow_pickle=False) as data:
        if set(data.files) != set(REFERENCE_ARRAYS):
            raise ValueError('complete reference surface arrays required')
        arrays = {key: data[key].copy() for key in REFERENCE_ARRAYS}
    surface = record['surface']
    reference = ReferenceSurfaceV40(**arrays, candidate_views=views,
        sample_spacing_m=surface['sample_spacing_m'], seed=surface['seed'],
        full_target_sample_count=surface['full_target_sample_count'], fingerprint=surface['fingerprint'])
    if reference.manifest() != surface or surface['target_instances'] != record['target_instance_inventory']:
        raise ValueError('reference content or full facility denominator mismatch')
    with np.load(root/'coverage_domain.npz', allow_pickle=False) as data:
        if data.files != ['domain']:
            raise ValueError('coverage domain inventory mismatch')
        domain = data['domain'].copy()
    measure_navigation_coverage_v44(np.full(domain.shape, -1, dtype=np.int8), domain, record['coverage'])
    return reference, domain, record


def evaluate_saved_episode_v44(episode_root, reference_root, *, reference_manifest_sha256,
                               expected_episode_manifest_sha256, source_root=ROOT):
    """Scores saved whole prediction only; neither C_nav nor an ROI is an input."""
    from nso.saved_replay_v44 import load_saved_episode_v44
    if (not isinstance(expected_episode_manifest_sha256, str)
            or not re.fullmatch(r'[0-9a-f]{64}', expected_episode_manifest_sha256)
            or sha256(Path(episode_root)/'artifact_manifest.json') != expected_episode_manifest_sha256):
        raise ValueError('saved episode manifest does not match externally pinned SHA256')
    episode = load_saved_episode_v44(episode_root, source_root=source_root,
                                   expected_manifest_sha256=expected_episode_manifest_sha256)
    if episode.started.get('finite_fixture') is True:
        raise ValueError('finite fixture cannot be scored as a development episode')
    slot = episode.started['slot']
    asset_id = slot['asset_id']
    reference, domain, record = load_reference_v44(reference_root,
        manifest_sha256=reference_manifest_sha256, asset_id=asset_id, source_root=source_root)
    coverage_descriptor = record['coverage']
    if episode.public_workspace != _json(Path(source_root)/GEOMETRY_ROOT/asset_id/'public_workspace.json'):
        raise ValueError('saved episode and fixed reference workspaces differ')
    mapper = _json(episode.root/'prediction/mapper.json')
    if (mapper.get('backend_poisoned') is not False or mapper.get('shape') != coverage_descriptor['shape']
            or mapper.get('resolution_m') != coverage_descriptor['resolution_m']
            or mapper.get('origin_xy_m') != coverage_descriptor['origin_xy_m']
            or mapper.get('grid_convention') != coverage_descriptor['grid_convention']
            or mapper.get('frames') != len(episode.frames)):
        raise ValueError('prediction map coordinates/history do not match fixed floor domain')
    with np.load(episode.root/'prediction/occupancy.npz', allow_pickle=False) as data:
        if set(data.files) != {'belief', 'observed'}:
            raise ValueError('complete saved occupancy arrays required')
        belief, observed = data['belief'].copy(), data['observed'].copy()
    if (_array_sha(belief) != mapper['occupancy_sha256'] or observed.dtype != np.bool_
            or observed.shape != belief.shape or np.any(observed & (belief < 0))):
        raise ValueError('occupancy content disagrees with saved mapper')
    coverage = measure_navigation_coverage_v44(belief, domain, coverage_descriptor)
    with np.load(episode.root/'prediction/mesh.npz', allow_pickle=False) as data:
        if set(data.files) != {'vertices', 'triangles', 'vertex_colors'}:
            raise ValueError('complete original mapper mesh required; ROI/owner files are not accepted')
        vertices, triangles, colors = (data[key].copy() for key in ('vertices', 'triangles', 'vertex_colors'))
    if colors.shape != vertices.shape or not np.isfinite(colors).all():
        raise ValueError('original full mesh colors do not align with vertices')
    metrics = evaluate_surface_v40(reference, vertices, triangles, C_map=coverage['C_nav'],
        threshold_m=.05, sample_spacing_m=.3, seed=4002, max_samples=50000)
    # The frozen V40 API calls the scalar C_map/J. Rename in this new protocol
    # so conservative floor coverage cannot be mistaken for generic map accuracy.
    metrics.pop('C_map'); metrics.pop('J')
    metrics['C_nav'] = coverage['C_nav']
    metrics['J_nav'] = coverage['C_nav']*metrics['Q']
    success = bool(episode.task_success)
    autonomous = episode.started.get('task_kind') == 'autonomous_controller'
    actual_development = (episode.runtime.get('world_created') is True
                          and episode.started.get('finite_fixture') is not True)
    ledger_binding = None
    if actual_development:
        relative = Path(episode.protocol['ledger_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('bounded repository-relative start ledger required')
        ledger_path = Path(source_root)/relative
        ledger = _json(ledger_path)
        rows = [row for row in ledger.get('entries', []) if row.get('run_id') == episode.started['run_id']]
        if len(rows) != 1:
            raise ValueError('actual development episode must bind its unique durable start reservation')
        row = rows[0]
        if (row.get('world_created') is not True or row.get('status') != episode.status
                or row.get('result_sha256') != sha256(episode.root/'result.json')
                or row.get('metadata', {}).get('slot') != slot
                or row['metadata'].get('source_sha256') != episode.manifest['source_sha256']):
            raise ValueError('episode and durable start ledger disagree')
        ledger_binding = dict(path=str(relative), sha256=sha256(ledger_path),
                              run_id=episode.started['run_id'], reservation_verified=True)
    return dict(schema='v44.saved_prediction_offline_evaluation.v1',
        status='development_endpoint_scored' if success else 'diagnostic_non_success_scored',
        run_id=episode.started['run_id'], asset_id=asset_id, mode=slot['mode'],
        original_episode_status=episode.status, task_success=success,
        eligible_successful_development_endpoint=success and autonomous and actual_development,
        formal_performance_evidence=False, semantic_performance_claim=False,
        coverage=coverage, metrics=metrics,
        episode_manifest_sha256=sha256(episode.root/'artifact_manifest.json'),
        externally_pinned_episode_manifest=True,
        development_start_ledger_binding=ledger_binding,
        reference_manifest_sha256=reference_manifest_sha256,
        input_prediction_sha256={name: sha256(episode.root/'prediction'/name)
                                for name in ('mesh.npz', 'occupancy.npz', 'mapper.json')},
        evaluation_source_sha256=sha256(Path(__file__)),
        all_task_instances_in_macro_denominator=True, prediction_roi_cropped=False,
        prediction_reintegrated=False, independent_replay_performed_here=False,
        new_worlds=0, new_sensor_packets=0, new_tsdf_integrations=0,
        protocol_scope='development endpoint diagnostic; fixed sparse surface quadrature and conservative C_nav; not a formal benchmark')
