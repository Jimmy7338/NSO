"""New static semantic-mechanism assets; no sensor, World, or policy calls.

The old V40 builders and their frozen identities remain unchanged. This module
reuses their geometry primitives while binding the new explicit scene draft.
Only the renderer/evaluator should call the private asset loader below.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from nso.development_geometry_v40 import (
    DevelopmentGeometryV40, PALETTE_V40, STRUCTURES, load_mesh_arrays,
    mesh_audit, mesh_npz_bytes, structure_boxes, union_box_mesh,
)
from nso.scene_contract_v40 import canonical_json_bytes, validate_public_spec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFT = ROOT / 'configs/virtual3d/semantic_scene_matrix_draft_20260923.json'
BASE_PROTOCOL = ROOT / 'configs/virtual3d/v40_scene_protocol_20260920.json'
CONDITIONS = ('nominal_relationship', 'structure_relationship_shift',
              'persistent_recognition_error', 'uninformative_relationship')
PARENTS = tuple(f'SEM_P{i:02d}' for i in range(6))
ASSET_FILES = ('renderer_private/geometry.npz', 'renderer_private/markers.json',
               'evaluation_private/instances.json', 'evaluation_private/mesh_audit.json',
               'public_workspace.json', 'public_planner_spec.json')
MAX_BUNDLE_BYTES = 16 * 1024 ** 2
_SHA = re.compile(r'[0-9a-f]{64}')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, *, maximum_bytes=1024 ** 2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_bytes:
        raise ValueError('bounded plain JSON file required')
    return json.loads(path.read_text(), parse_constant=lambda value:
                     (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def asset_id(parent_id, condition):
    if parent_id not in PARENTS or condition not in CONDITIONS:
        raise ValueError('declared new parent and condition required')
    return parent_id + '__' + condition


def parse_asset_id(identity):
    if not isinstance(identity, str) or identity.count('__') != 1:
        raise ValueError('explicit new asset identity required')
    parent, condition = identity.split('__')
    asset_id(parent, condition)
    return parent, condition


def load_design(path=DEFAULT_DRAFT):
    design = read_json(path)
    if (design.get('schema_version') != 'semantic_scene_matrix_draft.v1'
            or design.get('status') != 'unfrozen'
            or design.get('primary_matrix_started') is not False
            or design.get('old_v40_assets_or_protocol_modified') is not False):
        raise ValueError('explicit unfrozen new-phase static design required')
    parents = design.get('parents')
    if not isinstance(parents, list) or [p.get('parent_id') for p in parents] != list(PARENTS):
        raise ValueError('all six new independent layout identities required')
    prior = design['semantic_generation']['public_nominal_probability_by_category']
    for parent in parents:
        labels = parent['true_classes']
        if (len(labels) != 4 or sorted(Counter(labels).values()) != [2, 2]
                or not set(labels) <= set(PALETTE_V40)
                or len(parent['facility_xy_yaw_deg']) != 4):
            raise ValueError('four instances with two explicit repeated classes required')
        if set(parent['fixed_assignments']) != set(CONDITIONS):
            raise ValueError('all four separately declared relationship conditions required')
        for condition, assignment in parent['fixed_assignments'].items():
            if (assignment['true_classes'] != labels or len(assignment['observed_classes']) != 4
                    or len(assignment['true_structures']) != 4
                    or not set(assignment['observed_classes']) <= set(PALETTE_V40)
                    or not set(assignment['true_structures']) <= set(STRUCTURES)):
                raise ValueError('separate valid class, visible cue and structure arrays required')
            if condition != 'persistent_recognition_error' and assignment['observed_classes'] != labels:
                raise ValueError('relationship blocks cannot silently corrupt recognition')
        nominal = parent['fixed_assignments']['nominal_relationship']['true_structures']
        uniforms = parent['structure_uniforms']
        if len(uniforms) != 4 or not all(type(u) in (float, int) and 0 < u < 1 for u in uniforms):
            raise ValueError('four declared independent structure quantiles required')
        expected = [STRUCTURES[min(int(np.searchsorted(np.cumsum(prior[c]), u, side='right')), 3)]
                    for c, u in zip(labels, uniforms)]
        if nominal != expected:
            raise ValueError('nominal structures differ from fixed class-conditional quantiles')
        shifted = [design['semantic_generation']['structure_shift_permutation'][h] for h in nominal]
        if parent['fixed_assignments']['structure_relationship_shift']['true_structures'] != shifted:
            raise ValueError('structure shift differs from declared permutation')
        wrong = parent['fixed_assignments']['persistent_recognition_error']
        if (wrong['true_structures'] != nominal or wrong['observed_classes'] !=
                [design['semantic_generation']['persistent_wrong_label_permutation'][c] for c in labels]):
            raise ValueError('persistent wrong cues must preserve nominal physical structures')
        neutral = parent['fixed_assignments']['uninformative_relationship']['true_structures']
        multisets = [Counter(h for c, h in zip(labels, neutral) if c == label) for label in sorted(set(labels))]
        if multisets[0] != multisets[1]:
            raise ValueError('uninformative condition must balance structure within each class')
    return design


def _readonly(value, dtype):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def public_spec(design):
    spec = read_json(BASE_PROTOCOL)['public_defaults']
    spec['structure_prior']['probability_by_category'] = deepcopy(
        design['semantic_generation']['public_nominal_probability_by_category'])
    validate_public_spec(spec)
    return spec


def build_scene_geometry(parent, condition, *, nominal_prior):
    """Build closed meshes from explicit new parameters, not old DEV aliases."""
    identity = asset_id(parent['parent_id'], condition)
    assignment = parent['fixed_assignments'][condition]
    size = np.asarray(parent['room_size_m'], dtype=float)
    poses = np.asarray(parent['facility_xy_yaw_deg'], dtype=float)
    if (size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 2)
            or poses.shape != (4, 3) or not np.isfinite(poses).all()
            or parent['spatial_jitter_m'] != 0 or parent['dimension_jitter_fraction'] != 0):
        raise ValueError('fixed finite layout without undeclared jitter required')
    width, depth = map(float, size)
    occluders = np.asarray(parent['occluder_boxes_m'], dtype=float).reshape(-1, 6)
    if not np.isfinite(occluders).all():
        raise ValueError('finite obstacle boxes required')
    background = [[0., width, 0., depth, -.1, 0.], [0., .12, 0., depth, 0., 2.4],
                  [width-.12, width, 0., depth, 0., 2.4], [.12, width-.12, 0., .12, 0., 2.4],
                  [.12, width-.12, depth-.12, depth, 0., 2.4]] + occluders.tolist()
    vertices, triangles = union_box_mesh(background)
    all_vertices, all_triangles = [vertices], [triangles]
    owners, colors = [np.full(len(triangles), -1, np.int32)], [np.full((len(triangles), 3), 175, np.uint8)]
    offset, instances, markers, envelopes = len(vertices), [], [], []
    dimensions = np.asarray(parent['facility_dimensions_m'], dtype=float)
    for index, ((x, y, yaw), category, observed, structure) in enumerate(zip(
            poses, assignment['true_classes'], assignment['observed_classes'], assignment['true_structures'])):
        angle = np.deg2rad(yaw)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        center = np.array([x, y, 0.])
        boxes, marker = structure_boxes(structure, dimensions)
        local_vertices, local_triangles = union_box_mesh(boxes)
        world_vertices = local_vertices @ rotation.T + center
        all_vertices.append(world_vertices); all_triangles.append(local_triangles + offset)
        owners.append(np.full(len(local_triangles), index, np.int32))
        colors.append(np.full((len(local_triangles), 3), 165, np.uint8))
        offset += len(local_vertices)
        # The collision/navigation envelope is fixed across hidden structures.
        alternatives = []
        for alternative in STRUCTURES:
            candidate_boxes, _ = structure_boxes(alternative, dimensions)
            bounds = np.asarray(candidate_boxes).reshape(-1, 3, 2)
            corners = np.array([[a, b, c] for box in bounds
                                for a in box[0] for b in box[1] for c in box[2]])
            alternatives.append(corners @ rotation.T + center)
        envelope = np.concatenate(alternatives)
        lower, upper = envelope.min(axis=0), envelope.max(axis=0)
        if (np.any(lower[:2] <= .12) or upper[0] >= width-.12 or upper[1] >= depth-.12):
            raise ValueError('facility envelope intersects boundary walls')
        for other_lower, other_upper in envelopes:
            if np.all(np.minimum(upper, other_upper)-np.maximum(lower, other_lower) > 1e-9):
                raise ValueError('facility envelopes overlap')
        for obstacle in occluders:
            bounds = obstacle.reshape(3, 2).T
            if np.all(np.minimum(upper, bounds[1])-np.maximum(lower, bounds[0]) > 1e-9):
                raise ValueError('facility envelope intersects occluder')
        envelopes.append((lower, upper))
        footprint = [[float(lower[0]), float(lower[1])], [float(upper[0]), float(lower[1])],
                     [float(upper[0]), float(upper[1])], [float(lower[0]), float(upper[1])]]
        instances.append(dict(instance_id=index, category=category, observed_category=observed,
            structure=structure, dimensions_m=dimensions.tolist(), position_world_m=center.tolist(),
            yaw_deg=float(yaw), world_from_local_rotation=rotation.tolist(), local_solid_boxes=boxes,
            world_aabb_m=[world_vertices.min(axis=0).tolist(), world_vertices.max(axis=0).tolist()],
            navigation_envelope_aabb_m=[lower.tolist(), upper.tolist()],
            center_world_m=((world_vertices.min(axis=0)+world_vertices.max(axis=0))*.5).tolist(),
            conservative_footprint_xy_m=footprint,
            declared_class_structure_probability=nominal_prior[category][STRUCTURES.index(structure)],
            structure_assignment_rule='separate_declared_structure_array_in_new_draft',
            relationship_shift=condition == 'structure_relationship_shift',
            recognition_corruption=condition == 'persistent_recognition_error'))
        markers.append(dict(instance_id=index, category=observed,
            center_world_m=(rotation @ marker['center_local_m'] + center).tolist(),
            normal_world=(rotation @ marker['normal_local']).tolist(),
            u_world=(rotation @ marker['u_local']).tolist(), v_world=(rotation @ marker['v_local']).tolist(),
            width_m=marker['width_m'], height_m=marker['height_m'], rgb=list(PALETTE_V40[observed]),
            decal_on_existing_solid_face=True))
    arrays = [_readonly(np.concatenate(values), dtype) for values, dtype in
              ((all_vertices, np.float64), (all_triangles, np.int32), (owners, np.int32), (colors, np.uint8))]
    audit = mesh_audit(*arrays[:3])
    audit.update(asset_id=identity, independent_facilities=4, facility_envelopes_disjoint=True,
        navigation_envelope_independent_of_class_and_structure=True,
        selection_uses_method_results=False, world_or_sensor_instantiated=False)
    workspace = dict(schema_version='v40.development_workspace.v1', bounds_xy_m=[[0., 0.], [width, depth]],
        room_inner_bounds_xy_m=[[.12, .12], [width-.12, depth-.12]],
        start_position_world_m=[.75, .75, .9], start_yaw_deg=0.,
        marker_palette={key: list(value) for key, value in PALETTE_V40.items()},
        navigation_graph_status='pending', instance_truth_included=False,
        source='declared_task_workspace_not_facility_roi')
    return DevelopmentGeometryV40(*arrays, tuple(instances), tuple(markers), tuple(background), workspace, audit)


def build_assets(output_root, *, draft_path=DEFAULT_DRAFT):
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError('new semantic assets cannot overwrite existing evidence')
    design = load_design(draft_path)
    spec = public_spec(design)
    payloads = {'design.json': Path(draft_path).read_bytes()}
    entries = {}
    for parent in design['parents']:
        for condition in CONDITIONS:
            identity = asset_id(parent['parent_id'], condition)
            geometry = build_scene_geometry(parent, condition,
                nominal_prior=design['semantic_generation']['public_nominal_probability_by_category'])
            metadata = dict(schema_version='v40.development_private_instances.v1',
                asset_schema='semantic_scene_private.v1', asset_id=identity, parent_id=parent['parent_id'],
                family=parent['topology'], condition=condition,
                private_instances=geometry.private_instances, background_boxes=geometry.background_boxes,
                public_workspace=geometry.public_workspace, actual_geometry_gt=True,
                observable_evaluation_target_set_status='pending',
                ground_contact_bottom_faces_need_visibility_filtering=True)
            files = {
                ASSET_FILES[0]: mesh_npz_bytes(geometry),
                ASSET_FILES[1]: canonical_json_bytes(dict(schema_version='v40.development_marker_planes.v1',
                    marker_patches=geometry.marker_patches,
                    application='color only actual first-hit points on existing solid face patches',
                    whole_instance_color_semantics=False)),
                ASSET_FILES[2]: canonical_json_bytes(metadata),
                ASSET_FILES[3]: canonical_json_bytes(geometry.audit),
                ASSET_FILES[4]: canonical_json_bytes(geometry.public_workspace),
                ASSET_FILES[5]: canonical_json_bytes(spec),
            }
            payloads.update({identity + '/' + name: value for name, value in files.items()})
            entries[identity] = dict(parent_id=parent['parent_id'], condition=condition, instances=4,
                mesh_logical_sha256=geometry.logical_sha256(), files=list(ASSET_FILES))
    manifest = dict(schema='semantic_scene_assets.v1', status='static_development_assets_unfrozen_matrix',
        assets=entries, source_sha256={str(path.relative_to(ROOT)): file_sha256(path) for path in
            (Path(__file__), ROOT/'nso/development_geometry_v40.py', ROOT/'nso/scene_contract_v40.py', BASE_PROTOCOL)},
        design_sha256=file_sha256(draft_path),
        artifact_sha256={name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        worlds_created=0, sensor_frames_created=0, policy_trajectories_created=0, quality_evaluations=0,
        old_assets_modified=False, primary_matrix_started=False)
    payloads['manifest.json'] = canonical_json_bytes(manifest)
    total = sum(len(value) for value in payloads.values())
    if total > MAX_BUNDLE_BYTES:
        raise ValueError('static semantic asset bundle exceeds 16 MiB cap')
    for name, value in payloads.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(value)
    return dict(output_root=str(output), assets=len(entries), bytes_written=total,
                manifest_sha256=file_sha256(output/'manifest.json'), worlds_created=0,
                primary_matrix_started=False)


def checked_manifest(root, expected_manifest_sha256):
    if not isinstance(expected_manifest_sha256, str) or not _SHA.fullmatch(expected_manifest_sha256):
        raise ValueError('externally pinned manifest SHA256 required')
    root = Path(root).resolve()
    manifest = read_json(root/'manifest.json')
    if file_sha256(root/'manifest.json') != expected_manifest_sha256:
        raise ValueError('semantic asset manifest differs from external pin')
    if (manifest.get('schema') != 'semantic_scene_assets.v1'
            or set(manifest.get('assets', {})) != {asset_id(p, c) for p in PARENTS for c in CONDITIONS}):
        raise ValueError('new complete 24-asset manifest required')
    return manifest


def checked_asset_file(asset_dir, relative, manifest):
    directory = Path(asset_dir)
    parse_asset_id(directory.name)
    if relative not in ASSET_FILES or directory.is_symlink():
        raise ValueError('declared plain asset path required')
    root = directory.parent.resolve()
    path = directory / relative
    if (path.is_symlink() or not path.resolve().is_relative_to(root)
            or not path.is_file() or path.stat().st_size > MAX_BUNDLE_BYTES):
        raise ValueError('bounded plain asset file inside bundle required')
    if file_sha256(path) != manifest['artifact_sha256'].get(directory.name + '/' + relative):
        raise ValueError('semantic asset hash mismatch: ' + relative)
    return path


def load_semantic_scene_asset(asset_dir, *, expected_manifest_sha256):
    """Private renderer/evaluator loader. Never pass its return value to policy."""
    directory = Path(asset_dir)
    parent, condition = parse_asset_id(directory.name)
    manifest = checked_manifest(directory.parent, expected_manifest_sha256)
    paths = {name: checked_asset_file(directory, name, manifest) for name in ASSET_FILES}
    arrays = load_mesh_arrays(paths[ASSET_FILES[0]])
    markers = read_json(paths[ASSET_FILES[1]])['marker_patches']
    metadata = read_json(paths[ASSET_FILES[2]])
    workspace = read_json(paths['public_workspace.json'])
    spec = read_json(paths['public_planner_spec.json'])
    validate_public_spec(spec)
    if (metadata.get('asset_id') != directory.name or metadata.get('parent_id') != parent
            or metadata.get('condition') != condition or metadata.get('public_workspace') != workspace):
        raise ValueError('private/public asset identity or workspace mismatch')
    if len(markers) != 4 or len(metadata['private_instances']) != 4:
        raise ValueError('four declared facilities required')
    return dict(arrays=arrays, markers=markers, metadata=metadata,
        public_workspace=workspace, public_spec=spec, asset_id=directory.name,
        manifest_sha256=expected_manifest_sha256)
