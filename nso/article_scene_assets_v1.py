"""Small article-stage industrial assets and isolated public/private loaders.

Builds static box-union meshes and a supplied navigation prior. No simulator,
sensor, controller, TSDF, policy result or metric evaluator is imported. Public
runtime loading reads only manifest and four public JSON files. Scene/design
schemas and identities are new; nested geometry/navigation protocols deliberately
reuse their published formats without pretending to be old scene registrations.
"""
from collections import Counter, deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from nso.development_geometry_v40 import (
    DevelopmentGeometryV40, PALETTE_V40, STRUCTURES, load_mesh_arrays,
    mesh_audit, mesh_npz_bytes, structure_boxes, union_box_mesh,
)
from nso.primitive_navigation_v41 import PrimitiveStateV41, PublicPrimitiveGraphV41
from nso.public_navigation_v43 import (
    _ASSUMPTIONS, compile_blueprint_graph_v43, segment_intersects_rectangle_v43,
)
from nso.scene_contract_v40 import canonical_json_bytes, content_sha256, validate_public_spec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / 'configs/virtual3d/article_scene_design_20260928.json'
BASE_PROTOCOL = ROOT / 'configs/virtual3d/v40_scene_protocol_20260920.json'
FAMILIES = ('AISLE', 'CELL', 'LOOP')
SCENE_IDS = tuple(f'ART1_{family}_{split}' for family in FAMILIES for split in ('DEV', 'T0', 'T1'))
MAX_BUNDLE_BYTES = 30 * 1024 ** 2
PUBLIC_FILES = ('graph.json', 'certificate.json', 'public_workspace.json', 'public_planner_spec.json')
PRIVATE_FILES = ('renderer_private/geometry.npz', 'renderer_private/markers.json',
                 'evaluation_private/instances.json', 'evaluation_private/mesh_audit.json',
                 'evaluation_private/static_geometry_audit.json')
ASSET_FILES = PUBLIC_FILES + PRIVATE_FILES
_WORKSPACE_KEYS = {'schema_version', 'bounds_xy_m', 'room_inner_bounds_xy_m',
    'start_position_world_m', 'start_yaw_deg', 'marker_palette', 'navigation_graph_status',
    'instance_truth_included', 'source'}
_DESIGN_KEYS = {'schema_version', 'design_id', 'status', 'condition', 'layout_units',
    'development_layouts', 'test_layouts', 'selection_rule', 'generalization_scope',
    'start_position_world_m', 'start_yaw_deg', 'facility_dimensions_m',
    'nominal_probability_by_category', 'structure_assignment_rule', 'sensor_noise_rule',
    'maximum_bundle_bytes', 'families', 'layouts'}
_LAYOUT_KEYS = {'scene_id', 'family', 'split', 'room_size_m', 'facility_xy_yaw_deg',
    'occluder_boxes_m', 'true_classes', 'structure_quantiles'}
_MANIFEST_KEYS = {'schema', 'design_id', 'design_sha256', 'assets', 'artifact_sha256',
    'source_sha256', 'layout_units', 'development_layouts', 'test_layouts', 'condition',
    'worlds_created', 'sensor_frames_created', 'policy_trajectories_created',
    'quality_evaluations', 'old_assets_modified', 'primary_matrix_started'}
_META_KEYS = {'schema_version', 'scene_id', 'asset_id', 'parent_id', 'family', 'split',
    'condition', 'private_instances', 'background_boxes', 'public_workspace',
    'actual_geometry_gt', 'observable_evaluation_target_set_status',
    'ground_contact_bottom_faces_need_visibility_filtering'}


def _strict(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + ' field whitelist violation')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _decode(data):
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def _plain_bytes(path, maximum_bytes=MAX_BUNDLE_BYTES):
    path = Path(path)
    if (path.is_symlink() or any(p.is_symlink() for p in path.parents)
            or not path.is_file() or path.stat().st_size > maximum_bytes):
        raise ValueError('bounded plain file required: ' + str(path))
    return path.read_bytes()


def parse_scene_id(scene_id):
    if scene_id not in SCENE_IDS:
        raise ValueError('new article scene identity required; historical aliases rejected')
    _, family, split = scene_id.split('_')
    return family, 'development' if split == 'DEV' else 'test'


def load_design(path=DEFAULT_DESIGN):
    design = _decode(_plain_bytes(path, 256 * 1024))
    _strict(design, _DESIGN_KEYS, 'article design')
    if (design['schema_version'] != 'article.scene_design.v1'
            or design['status'] != 'static_design_before_new_policy_results'
            or design['condition'] != 'nominal_relationship'
            or (design['layout_units'], design['development_layouts'], design['test_layouts']) != (9, 3, 6)
            or design['start_position_world_m'] != [.75, .75, .9] or design['start_yaw_deg'] != 0.
            or design['maximum_bundle_bytes'] != MAX_BUNDLE_BYTES
            or set(design['families']) != set(FAMILIES)
            or [x.get('scene_id') for x in design['layouts']] != list(SCENE_IDS)):
        raise ValueError('complete new nine-layout article design required')
    base = _decode(_plain_bytes(BASE_PROTOCOL))['public_defaults']
    if design['nominal_probability_by_category'] != base['structure_prior']['probability_by_category']:
        raise ValueError('retain published nominal prior; no post-result prior tuning')
    dimensions = np.asarray(design['facility_dimensions_m'], float)
    if dimensions.shape != (3,) or not np.array_equal(dimensions, [1.2, .8, 1.6]):
        raise ValueError('common published nominal facility dimensions required')
    signatures = set()
    for row in design['layouts']:
        _strict(row, _LAYOUT_KEYS, 'article layout')
        if (row['family'], row['split']) != parse_scene_id(row['scene_id']):
            raise ValueError('scene identity/family/split mismatch')
        size = np.asarray(row['room_size_m'], float)
        poses = np.asarray(row['facility_xy_yaw_deg'], float)
        boxes = np.asarray(row['occluder_boxes_m'], float).reshape(-1, 6)
        if (size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 6) or np.any(size > 8)
                or poses.shape != (4, 3) or not np.isfinite(poses).all()
                or not np.isfinite(boxes).all() or len(boxes) > 8
                or np.any(boxes[:, [0, 2, 4]] >= boxes[:, [1, 3, 5]])):
            raise ValueError('finite compact four-facility geometry required')
        if (len(row['true_classes']) != 4 or sorted(Counter(row['true_classes']).values()) != [2, 2]
                or not set(row['true_classes']) <= set(PALETTE_V40)
                or len(row['structure_quantiles']) != 4):
            raise ValueError('two repeated public classes and four nominal quantiles required')
        for label in set(row['true_classes']):
            if sorted(q for c, q in zip(row['true_classes'], row['structure_quantiles']) if c == label) != [.25, .75]:
                raise ValueError('fixed within-class stratified quantiles required')
        signature = canonical_json_bytes([row['room_size_m'], row['facility_xy_yaw_deg'], row['occluder_boxes_m']])
        if signature in signatures:
            raise ValueError('new layout IDs cannot merely rename identical geometry')
        signatures.add(signature)
    return design


def _workspace(size):
    width, depth = map(float, size)
    return dict(schema_version='v40.development_workspace.v1',
        bounds_xy_m=[[0., 0.], [width, depth]],
        room_inner_bounds_xy_m=[[.12, .12], [width-.12, depth-.12]],
        start_position_world_m=[.75, .75, .9], start_yaw_deg=0.,
        marker_palette={k: list(v) for k, v in PALETTE_V40.items()},
        navigation_graph_status='materialized', instance_truth_included=False,
        source='declared_task_workspace_not_facility_roi')


def _validate_workspace(value):
    _strict(value, _WORKSPACE_KEYS, 'public article workspace')
    if (value['schema_version'] != 'v40.development_workspace.v1'
            or value['start_position_world_m'] != [.75, .75, .9] or value['start_yaw_deg'] != 0.
            or value['instance_truth_included'] is not False
            or value['navigation_graph_status'] != 'materialized'
            or value['source'] != 'declared_task_workspace_not_facility_roi'
            or value['marker_palette'] != {k: list(v) for k, v in PALETTE_V40.items()}):
        raise ValueError('public workspace identity/start/palette mismatch')
    bounds = np.asarray(value['bounds_xy_m'], float)
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or np.any(bounds[1] < 6.) or np.any(bounds[1] > 8.)):
        raise ValueError('finite compact public room bounds required')
    size = bounds[1].tolist()
    if value != _workspace(size):
        raise ValueError('finite rectangular public workspace required')


def navigation_blueprint(layout, dimensions=(1.2, .8, 1.6)):
    """Navigation envelopes use all four structures; never read class/quantile."""
    width, depth = map(float, layout['room_size_m'])
    obstacles = np.asarray(layout['occluder_boxes_m'], float).reshape(-1, 6)
    if any(b[0] < .12 or b[1] > width-.12 or b[2] < .12 or b[3] > depth-.12
           or b[4] < 0 or b[5] > 2.4 for b in obstacles):
        raise ValueError('occluders must stay within the physical room')
    walls = [[0., .12, 0., depth, 0., 2.4], [width-.12, width, 0., depth, 0., 2.4],
             [.12, width-.12, 0., .12, 0., 2.4], [.12, width-.12, depth-.12, depth, 0., 2.4]]
    envelopes = []
    for x, y, yaw in layout['facility_xy_yaw_deg']:
        a = math.radians(yaw)
        rotation = np.array([[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
        points = []
        for structure in STRUCTURES:
            boxes, _ = structure_boxes(structure, dimensions)
            points.extend([np.asarray([i, j, k]) @ rotation.T + [x, y, 0.]
                           for box in np.asarray(boxes).reshape(-1, 3, 2)
                           for i in box[0] for j in box[1] for k in box[2]])
        lower, upper = np.min(points, axis=0), np.max(points, axis=0)
        if np.any(lower[:2] <= .12) or np.any(upper[:2] >= [width-.12, depth-.12]):
            raise ValueError('facility envelope intersects room wall')
        for lo, hi in envelopes + [(b.reshape(3, 2)[:, 0], b.reshape(3, 2)[:, 1]) for b in obstacles]:
            if np.all(np.minimum(upper, hi) - np.maximum(lower, lo) > 1e-9):
                raise ValueError('facility envelopes/occluders overlap')
        envelopes.append((lower, upper))
    background = [[0., width, 0., depth, -.1, 0.]] + walls + obstacles.tolist()
    rectangles = [[lo[0], hi[0], lo[1], hi[1]] for lo, hi in envelopes]
    rectangles += [[b[0], b[1], b[2], b[3]] for b in background if b[5] > 0]
    return envelopes, background, np.asarray(rectangles, float).tolist()


def _build_geometry(layout, design, envelopes, background):
    dims = design['facility_dimensions_m']
    vertices, triangles = union_box_mesh(background)
    vs, ts = [vertices], [triangles]
    owners, colors = [np.full(len(triangles), -1, np.int32)], [np.full((len(triangles), 3), 175, np.uint8)]
    instances, markers, offset = [], [], len(vertices)
    prior = design['nominal_probability_by_category']
    for i, ((x, y, yaw), label, quantile) in enumerate(zip(
            layout['facility_xy_yaw_deg'], layout['true_classes'], layout['structure_quantiles'])):
        structure = STRUCTURES[int(np.searchsorted(np.cumsum(prior[label]), quantile, side='right'))]
        a = math.radians(yaw)
        rotation = np.array([[math.cos(a), -math.sin(a), 0.], [math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
        position = np.array([x, y, 0.])
        boxes, marker = structure_boxes(structure, dims)
        v, t = union_box_mesh(boxes)
        world = v @ rotation.T + position
        vs.append(world); ts.append(t + offset); offset += len(v)
        owners.append(np.full(len(t), i, np.int32)); colors.append(np.full((len(t), 3), 165, np.uint8))
        lo, hi = envelopes[i]
        instances.append(dict(instance_id=i, category=label, observed_category=label, structure=structure,
            dimensions_m=list(dims), position_world_m=position.tolist(), yaw_deg=float(yaw),
            world_from_local_rotation=rotation.tolist(), local_solid_boxes=boxes,
            world_aabb_m=[world.min(axis=0).tolist(), world.max(axis=0).tolist()],
            navigation_envelope_aabb_m=[lo.tolist(), hi.tolist()],
            center_world_m=((world.min(axis=0)+world.max(axis=0))*.5).tolist(),
            conservative_footprint_xy_m=[[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
            declared_class_structure_probability=prior[label][STRUCTURES.index(structure)],
            structure_assignment_rule='predeclared_within_class_stratified_quantile',
            structure_quantile=quantile, relationship_shift=False, recognition_corruption=False))
        markers.append(dict(instance_id=i, category=label,
            center_world_m=(rotation @ marker['center_local_m'] + position).tolist(),
            normal_world=(rotation @ marker['normal_local']).tolist(),
            u_world=(rotation @ marker['u_local']).tolist(), v_world=(rotation @ marker['v_local']).tolist(),
            width_m=marker['width_m'], height_m=marker['height_m'], rgb=list(PALETTE_V40[label]),
            decal_on_existing_solid_face=True))
    arrays = [np.concatenate(values).astype(dtype) for values, dtype in
              ((vs, np.float64), (ts, np.int32), (owners, np.int32), (colors, np.uint8))]
    for array in arrays:
        array.flags.writeable = False
    audit = mesh_audit(*arrays[:3])
    audit.update(scene_id=layout['scene_id'], independent_facilities=4,
        facility_envelopes_disjoint=True, navigation_envelope_independent_of_class_and_structure=True,
        selection_uses_method_results=False, world_or_sensor_instantiated=False)
    return DevelopmentGeometryV40(*arrays, tuple(instances), tuple(markers), tuple(background),
                                  _workspace(layout['room_size_m']), audit)


def _all_distances(graph, home):
    distance, reverse = {home: 0}, {}
    queue = deque([home])
    while queue:
        state = queue.popleft()
        for action in ('left', 'right', 'forward'):
            try:
                following = graph.successor(state, action)
            except ValueError:
                continue
            reverse.setdefault(following, []).append(state)
            if following not in distance:
                distance[following] = distance[state] + 1
                queue.append(following)
    inbound, queue = {home: 0}, deque([home])
    while queue:
        state = queue.popleft()
        for previous in reverse.get(state, []):
            if previous not in inbound:
                inbound[previous] = inbound[state] + 1
                queue.append(previous)
    return distance, inbound


def _static_views(geometry, graph_spec, rectangles):
    """Private offline marker geometry witnesses, not sensor/quality outcomes."""
    graph = PublicPrimitiveGraphV41(graph_spec)
    home = PrimitiveStateV41('home', 0)
    outbound, inbound = _all_distances(graph, home)
    per_instance = []
    for i, marker in enumerate(geometry.marker_patches):
        anchor = np.asarray(marker['center_world_m'])
        normal = np.asarray(marker['normal_world'])
        blockers = rectangles[:i] + rectangles[i+1:]
        witnesses = []
        for node in sorted(graph.original_nodes):
            center = np.r_[graph.positions[node], .9]
            toward = center - anchor
            range_xy = float(np.linalg.norm(toward[:2]))
            if not .35 <= range_xy <= 1.30 or normal @ toward < .5 * range_xy:
                continue
            # Check all actual marker corners, not just its center ray.
            corners = [anchor + su * marker['width_m']*.5*np.asarray(marker['u_world'])
                       + sv * marker['height_m']*.5*np.asarray(marker['v_world'])
                       for su in (-1, 1) for sv in (-1, 1)]
            if any(segment_intersects_rectangle_v43(center[:2], p[:2], r)
                   for p in corners for r in blockers):
                continue
            heading = int(round(math.atan2(anchor[1]-center[1], anchor[0]-center[0])/(math.pi/6.))) % 12
            state = PrimitiveStateV41(node, heading)
            a = heading*math.pi/6.
            rotation = np.array([[math.sin(a), 0., math.cos(a)], [-math.cos(a), 0., math.sin(a)], [0., -1., 0.]])
            camera = (np.array(corners)-center) @ rotation
            if not np.all((camera[:, 2] >= .1) & (camera[:, 2] <= 4.)):
                continue
            uv = camera[:, :2] / camera[:, 2, None] * 48. + [47.5, 35.5]
            if not (np.all(uv >= 0.) and np.all(uv < [96., 72.])):
                continue
            if state not in outbound or state not in inbound:
                continue
            cost = outbound[state] + 1 + inbound[state]
            witnesses.append(dict(node=node, heading=heading, range_to_marker_xy_m=range_xy,
                marker_fully_in_nominal_fov=True, unoccluded_static_marker_corner_segments=True,
                outbound_paid_actions=outbound[state], full_pose_return_paid_actions=inbound[state],
                one_view_return_paid_actions=cost))
        witnesses.sort(key=lambda row: (row['one_view_return_paid_actions'], row['range_to_marker_xy_m'], row['node']))
        if not witnesses or witnesses[0]['one_view_return_paid_actions'] > 160:
            raise ValueError(f"{geometry.audit['scene_id']} instance {i}: no close affordable static frontal pose")
        per_instance.append(dict(instance_slot=i, close_frontal_coarse_positions=len(witnesses),
            minimum_individual_view_return_actions=witnesses[0]['one_view_return_paid_actions'],
            witnesses=witnesses))
    return dict(schema='article.static_geometry_audit.v1', scene_id=geometry.audit['scene_id'],
        per_facility=per_instance, camera_size=[96, 72], camera_focal_lengths=[48., 48.],
        maximum_front_range_m=1.30, minimum_front_cosine=.5,
        nominal_marker_geometry_used_only_offline=True, exported_to_policy=False,
        sensor_pixels_or_depth_fit_verified=False, quality_evaluated=False, policy_executed=False,
        note='Existential geometry and primitive action costs only; no supplied policy/witness, no success probability.')


def build_article_scene_assets(output_root, *, design_path=DEFAULT_DESIGN):
    """Create all nine assets once; reject existing output instead of overwriting."""
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError('new article assets cannot overwrite existing evidence')
    design_raw = _plain_bytes(design_path, 256 * 1024)
    design = load_design(design_path)
    base = _decode(_plain_bytes(BASE_PROTOCOL))['public_defaults']
    validate_public_spec(base)
    payloads, entries = {'design_private.json': design_raw}, {}
    for layout in design['layouts']:
        identity = layout['scene_id']
        envelopes, background, rectangles = navigation_blueprint(layout, design['facility_dimensions_m'])
        graph_spec, certificate = compile_blueprint_graph_v43(_workspace(layout['room_size_m']), rectangles)
        if certificate['safe_lattice_nodes'] != certificate['reachable_nodes']:
            raise ValueError(identity + ': disconnected safe coarse navigation nodes')
        geometry = _build_geometry(layout, design, envelopes, background)
        static_audit = _static_views(geometry, graph_spec, rectangles)
        graph_bytes = canonical_json_bytes(graph_spec)
        spec = deepcopy(base)
        spec['navigation'].update(status='materialized',
            graph_asset=dict(status='materialized', relative_path='graph.json', sha256=_sha(graph_bytes)))
        validate_public_spec(spec)
        certificate.update(schema_version='article.navigation_certificate.v1', scene_id=identity,
            collision_envelopes_are_all_structure_union=True, class_and_structure_independent=True)
        metadata = dict(schema_version='article.private_instances.v1', scene_id=identity,
            asset_id=identity, parent_id=identity, family=layout['family'], split=layout['split'],
            condition='nominal_relationship', private_instances=geometry.private_instances,
            background_boxes=geometry.background_boxes, public_workspace=geometry.public_workspace,
            actual_geometry_gt=True, observable_evaluation_target_set_status='pending',
            ground_contact_bottom_faces_need_visibility_filtering=True)
        files = {'graph.json': graph_bytes, 'certificate.json': canonical_json_bytes(certificate),
            'public_workspace.json': canonical_json_bytes(geometry.public_workspace),
            'public_planner_spec.json': canonical_json_bytes(spec),
            'renderer_private/geometry.npz': mesh_npz_bytes(geometry),
            'renderer_private/markers.json': canonical_json_bytes(dict(
                schema_version='article.marker_planes.v1', scene_id=identity,
                marker_patches=geometry.marker_patches, whole_instance_color_semantics=False,
                application='color only actual first-hit points on existing solid face patches')),
            'evaluation_private/instances.json': canonical_json_bytes(metadata),
            'evaluation_private/mesh_audit.json': canonical_json_bytes(geometry.audit),
            'evaluation_private/static_geometry_audit.json': canonical_json_bytes(static_audit)}
        payloads.update({identity + '/' + name: value for name, value in files.items()})
        entries[identity] = dict(family=layout['family'], split=layout['split'], files=list(ASSET_FILES),
            graph_sha256=_sha(graph_bytes), mesh_logical_sha256=geometry.logical_sha256())
    source_names = ('nso/article_scene_assets_v1.py', 'nso/development_geometry_v40.py',
                    'nso/public_navigation_v43.py', 'nso/primitive_navigation_v41.py',
                    'nso/scene_contract_v40.py', str(BASE_PROTOCOL.relative_to(ROOT)))
    manifest = dict(schema='article.scene_assets.v1', design_id=design['design_id'],
        design_sha256=_sha(design_raw), assets=entries,
        artifact_sha256={name: _sha(data) for name, data in payloads.items()},
        source_sha256={name: _sha(_plain_bytes(ROOT/name)) for name in source_names},
        layout_units=9, development_layouts=3, test_layouts=6, condition='nominal_relationship',
        worlds_created=0, sensor_frames_created=0, policy_trajectories_created=0,
        quality_evaluations=0, old_assets_modified=False, primary_matrix_started=False)
    payloads['manifest.json'] = canonical_json_bytes(manifest)
    total = sum(map(len, payloads.values()))
    if total > MAX_BUNDLE_BYTES:
        raise ValueError('article asset bundle exceeds 30 MiB cap')
    for name, data in payloads.items():
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(data)
    return dict(output_root=str(output.resolve()), assets=9, bytes_written=total,
        manifest_sha256=_sha(payloads['manifest.json']), worlds_created=0,
        development_scene_ids=[x for x in SCENE_IDS if x.endswith('_DEV')],
        test_scene_ids=[x for x in SCENE_IDS if not x.endswith('_DEV')])


def _checked_manifest(asset_dir, expected_manifest_sha256):
    directory = Path(asset_dir)
    parse_scene_id(directory.name)
    if (not isinstance(expected_manifest_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', expected_manifest_sha256) is None
            or directory.is_symlink()):
        raise ValueError('plain article scene directory and external manifest pin required')
    raw = _plain_bytes(directory.parent/'manifest.json', 256*1024)
    if _sha(raw) != expected_manifest_sha256:
        raise ValueError('article asset manifest differs from external pin')
    value = _decode(raw)
    _strict(value, _MANIFEST_KEYS, 'article manifest')
    if (value['schema'] != 'article.scene_assets.v1' or set(value['assets']) != set(SCENE_IDS)
            or value['condition'] != 'nominal_relationship'
            or (value['layout_units'], value['development_layouts'], value['test_layouts']) != (9, 3, 6)):
        raise ValueError('complete new article manifest required')
    for identity, entry in value['assets'].items():
        _strict(entry, {'family', 'split', 'files', 'graph_sha256', 'mesh_logical_sha256'}, 'manifest scene')
        if (entry['family'], entry['split']) != parse_scene_id(identity) or entry['files'] != list(ASSET_FILES):
            raise ValueError('manifest scene identity or files mismatch')
    expected_paths = {i+'/'+p for i in SCENE_IDS for p in ASSET_FILES} | {'design_private.json'}
    if set(value['artifact_sha256']) != expected_paths or any(
            not isinstance(h, str) or re.fullmatch('[0-9a-f]{64}', h) is None
            for h in value['artifact_sha256'].values()):
        raise ValueError('complete bounded artifact hashes required')
    return directory, value


def _read_asset(directory, name, manifest):
    if name not in ASSET_FILES:
        raise ValueError('undeclared article artifact')
    data = _plain_bytes(directory/name, 256*1024 if name in PUBLIC_FILES else MAX_BUNDLE_BYTES)
    if _sha(data) != manifest['artifact_sha256'][directory.name+'/'+name]:
        raise ValueError('article artifact hash mismatch: ' + name)
    return data


def load_public_article_scene(asset_dir, *, expected_manifest_sha256):
    """Read manifest and only PUBLIC_FILES; never design, mesh or private metadata."""
    directory, manifest = _checked_manifest(asset_dir, expected_manifest_sha256)
    loaded = {name: _decode(_read_asset(directory, name, manifest)) for name in PUBLIC_FILES}
    workspace, spec = loaded['public_workspace.json'], loaded['public_planner_spec.json']
    _validate_workspace(workspace); validate_public_spec(spec)
    graph_spec, certificate = loaded['graph.json'], loaded['certificate.json']
    _strict(graph_spec, {'schema_version', 'source_kind', 'nodes', 'edges'}, 'public graph')
    graph = PublicPrimitiveGraphV41(graph_spec, camera_height_m=spec['motion']['camera_height_m'])
    certificate_keys = {'schema_version', 'graph_content_sha256', 'grid_spacing_m', 'robot_radius_m',
        'collision_model', 'start_component_only', 'all_nodes_and_edges_certified', 'declared_lattice_nodes',
        'safe_lattice_nodes', 'reachable_nodes', 'coarse_edges', 'expanded_primitive_nodes',
        'blueprint_content_sha256', 'source_kind', 'method_results_or_semantics_used', 'world_or_sensor_calls',
        'collision_envelopes_exported', 'assumptions', 'scene_id',
        'collision_envelopes_are_all_structure_union', 'class_and_structure_independent'}
    _strict(certificate, certificate_keys, 'article navigation certificate')
    graph_sha = manifest['artifact_sha256'][directory.name+'/graph.json']
    if (spec['task']['max_actions'] != 160 or spec['task']['forced_prefix_actions'] != []
            or graph.positions.get('home') != (.75, .75)
            or spec['navigation']['graph_asset'] != dict(status='materialized', relative_path='graph.json', sha256=graph_sha)
            or certificate['schema_version'] != 'article.navigation_certificate.v1'
            or certificate['scene_id'] != directory.name
            or certificate['graph_content_sha256'] != content_sha256(graph_spec)
            or certificate['reachable_nodes'] != len(graph.original_nodes)
            or certificate['safe_lattice_nodes'] != len(graph.original_nodes)
            or certificate['expanded_primitive_nodes'] != len(graph.positions)
            or certificate['coarse_edges'] != len(graph_spec['edges'])
            or certificate['grid_spacing_m'] != 1. or certificate['robot_radius_m'] != .2
            or certificate['all_nodes_and_edges_certified'] is not True
            or certificate['collision_envelopes_exported'] is not False
            or certificate['assumptions'] != _ASSUMPTIONS
            or certificate['world_or_sensor_calls'] != 0
            or certificate['method_results_or_semantics_used'] is not False
            or certificate['class_and_structure_independent'] is not True
            or certificate['collision_envelopes_are_all_structure_union'] is not True):
        raise ValueError('public graph/spec/certificate binding mismatch')
    return dict(scene_id=directory.name, asset_id=directory.name, graph=graph, graph_spec=graph_spec,
        public_spec=spec, workspace=workspace, home_state=PrimitiveStateV41('home', 0),
        certificate=certificate, asset_manifest_sha256=expected_manifest_sha256)


def load_private_article_scene(asset_dir, *, expected_manifest_sha256):
    """Renderer/evaluator only; return value must never be passed into a policy."""
    public = load_public_article_scene(asset_dir, expected_manifest_sha256=expected_manifest_sha256)
    directory, manifest = _checked_manifest(asset_dir, expected_manifest_sha256)
    for name in PRIVATE_FILES:
        _read_asset(directory, name, manifest)
    arrays = load_mesh_arrays(directory/'renderer_private/geometry.npz')
    metadata = _decode(_read_asset(directory, 'evaluation_private/instances.json', manifest))
    markers = _decode(_read_asset(directory, 'renderer_private/markers.json', manifest))
    _strict(metadata, _META_KEYS, 'article private metadata')
    _strict(markers, {'schema_version', 'scene_id', 'marker_patches',
        'whole_instance_color_semantics', 'application'}, 'article markers')
    if (metadata['schema_version'] != 'article.private_instances.v1'
            or any(metadata[k] != directory.name for k in ('scene_id', 'asset_id', 'parent_id'))
            or (metadata['family'], metadata['split']) != parse_scene_id(directory.name)
            or metadata['public_workspace'] != public['workspace']
            or metadata['condition'] != 'nominal_relationship'
            or metadata['actual_geometry_gt'] is not True
            or markers['schema_version'] != 'article.marker_planes.v1'
            or markers['scene_id'] != directory.name
            or markers['whole_instance_color_semantics'] is not False
            or len(markers['marker_patches']) != 4 or len(metadata['private_instances']) != 4):
        raise ValueError('private/public article identity mismatch')
    return dict(scene_id=directory.name, asset_id=directory.name, arrays=arrays,
        metadata=metadata, markers=markers['marker_patches'],
        public_workspace=public['workspace'], public_spec=public['public_spec'],
        manifest_sha256=expected_manifest_sha256)
