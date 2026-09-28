"""Shared supplied navigation prior and bounded public candidate selection.

Only ``compile_blueprint_graph_v43`` accepts offline 2-D collision rectangles.
The runtime loader reads public JSON assets only; it never opens scene meshes,
instance metadata, labels, or simulator objects. A provided topology reveals
coarse traversability and is not an unknown-environment exploration setting.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41
from nso.scene_contract_v40 import content_sha256, load_json_strict, validate_public_spec


FROZEN_PROTOCOL_SHA256_V43 = 'abb23ef5eb45f46006dddc736ff4e6e9203cc499d7d936657e63f1ba450dda98'
DEFAULT_PROTOCOL_PATH_V43 = Path(__file__).resolve().parents[1]/'configs/virtual3d/v40_scene_protocol_20260920.json'
_PARENTS = {f'DEV_{family}_00' for family in 'ABCDEF'}
_PUBLIC_FILES = ('graph.json', 'sidecar.json', 'certificate.json')
_SIDECAR_KEYS = {'schema_version', 'parent_id', 'frozen_inputs_sha256',
                 'frozen_public_spec_content_sha256', 'frozen_protocol_content_sha256',
                 'graph_content_sha256', 'start_state', 'graph_asset', 'assumptions'}
_ASSUMPTIONS = dict(navigation_prior='provided coarse traversability compiled offline from conservative 2D blueprint',
    shared_with_all_methods=True, unknown_environment_exploration=False,
    reveals='reachable 1m lattice positions and collision-certified corridors; coarse obstacle constraints can be inferred',
    hidden_surface_mesh_or_class_structure_exposed=False)
_WORKSPACE_KEYS = {'bounds_xy_m', 'instance_truth_included', 'marker_palette', 'navigation_graph_status',
    'room_inner_bounds_xy_m', 'schema_version', 'source', 'start_position_world_m', 'start_yaw_deg'}
_COLLISION_MODEL = 'closed continuous segment against XY envelope inflated by axis-aligned radius square'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strict(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label+' field whitelist violation')


def _hashes(value, keys, label):
    _strict(value, keys, label)
    if any(type(v) is not str or re.fullmatch('[0-9a-f]{64}', v) is None for v in value.values()):
        raise ValueError(label+' requires SHA256 values')


def _public_workspace(workspace, protocol):
    _strict(workspace, _WORKSPACE_KEYS, 'frozen public workspace')
    if (workspace['schema_version'] != 'v40.development_workspace.v1'
            or workspace['instance_truth_included'] is not False
            or workspace['navigation_graph_status'] != 'pending'
            or workspace['source'] != 'declared_task_workspace_not_facility_roi'
            or workspace['start_position_world_m'] != [.75, .75, .9]
            or workspace['start_yaw_deg'] != 0.):
        raise ValueError('frozen workspace public contract mismatch')
    bounds, inner = (np.asarray(workspace[key], float) for key in ('bounds_xy_m', 'room_inner_bounds_xy_m'))
    if (bounds.shape != (2, 2) or inner.shape != (2, 2)
            or not np.isfinite(bounds).all() or not np.isfinite(inner).all()
            or np.any(bounds[0] >= bounds[1]) or np.any(inner[0] >= inner[1])
            or np.any(inner[0] < bounds[0]) or np.any(inner[1] > bounds[1])):
        raise ValueError('finite ordered nested public workspace bounds required')
    palette = workspace['marker_palette']
    _strict(palette, protocol['public_defaults']['structure_prior']['categories'], 'public marker palette')
    for color in palette.values():
        if type(color) is not list or len(color) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in color):
            raise ValueError('public marker colors require RGB integer triplets')
    if len({tuple(color) for color in palette.values()}) != len(palette):
        raise ValueError('public marker colors must be distinct')


def segment_intersects_rectangle_v43(start, end, rectangle):
    """Closed segment versus closed axis-aligned XY rectangle, no sampling."""
    a, b, r = np.asarray(start, float), np.asarray(end, float), np.asarray(rectangle, float)
    if (a.shape != (2,) or b.shape != (2,) or r.shape != (4,)
            or not all(np.isfinite(x).all() for x in (a, b, r))
            or r[0] > r[1] or r[2] > r[3]):
        raise ValueError('finite XY segment and ordered rectangle required')
    lo, hi = 0., 1.
    for axis, (minimum, maximum) in enumerate(((r[0], r[1]), (r[2], r[3]))):
        delta = b[axis]-a[axis]
        if abs(delta) < 1e-14:
            if a[axis] < minimum or a[axis] > maximum:
                return False
        else:
            first, last = sorted(((minimum-a[axis])/delta, (maximum-a[axis])/delta))
            lo, hi = max(lo, first), min(hi, last)
            if lo > hi:
                return False
    return True


def compile_blueprint_graph_v43(public_workspace, obstacle_rectangles_xy_m):
    """OFFLINE compile at 1 m spacing, with a conservative 0.2 m swept robot.

    Rectangles are navigation-only collision envelopes, not surface targets.
    Inflation uses an axis-aligned square containing the circular robot, hence
    it can remove valid narrow routes. Only the start-connected graph leaves
    the compiler; rectangle coordinates are absent from both returned values.
    """
    if type(public_workspace) is not dict:
        raise ValueError('public workspace mapping required')
    if public_workspace.get('schema_version') != 'v40.development_workspace.v1':
        raise ValueError('registered V40 development workspace required')
    bounds = np.asarray(public_workspace['room_inner_bounds_xy_m'], float)
    start = np.asarray(public_workspace['start_position_world_m'], float)
    if (bounds.shape != (2, 2) or start.shape != (3,) or not np.isfinite(bounds).all()
            or not np.isfinite(start).all() or np.any(bounds[1] <= bounds[0])
            or not np.array_equal(start, [.75, .75, .9])
            or public_workspace['start_yaw_deg'] != 0.):
        raise ValueError('finite ordered room bounds and fixed [.75,.75,.9]/yaw0 start required')
    rectangles = np.asarray(obstacle_rectangles_xy_m, float)
    if rectangles.size == 0:
        rectangles = np.empty((0, 4), float)
    if (rectangles.ndim != 2 or rectangles.shape[1] != 4 or len(rectangles) > 1024
            or not np.isfinite(rectangles).all()
            or np.any(rectangles[:, 0] >= rectangles[:, 1]) or np.any(rectangles[:, 2] >= rectangles[:, 3])):
        raise ValueError('bounded finite positive XY collision rectangles required')
    radius, spacing = .2, 1.
    inflated = rectangles + np.asarray([-radius, radius, -radius, radius])
    axes = []
    for axis in range(2):
        first = math.ceil((bounds[0, axis]+radius-start[axis])/spacing)
        last = math.floor((bounds[1, axis]-radius-start[axis])/spacing)
        axes.append(range(first, last+1))
    if len(axes[0])*len(axes[1]) > 256:
        raise ValueError('declared 1m lattice exceeds 256 coarse nodes; do not silently change its resolution')
    all_nodes = {(i, j): start[:2]+spacing*np.asarray([i, j]) for i in axes[0] for j in axes[1]}
    free = {key: p for key, p in all_nodes.items()
            if np.all(p > bounds[0]+radius) and np.all(p < bounds[1]-radius)
            and not any(segment_intersects_rectangle_v43(p, p, r) for r in inflated)}
    if (0, 0) not in free:
        raise ValueError('predeclared start blocked by supplied blueprint; relocation forbidden')
    reached, queue, edges = {(0, 0)}, deque([(0, 0)]), set()
    while queue:
        key = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = key[0]+dx, key[1]+dy
            if neighbor not in free or any(segment_intersects_rectangle_v43(free[key], free[neighbor], r) for r in inflated):
                continue
            edges.add(tuple(sorted((key, neighbor))))
            if neighbor not in reached:
                reached.add(neighbor); queue.append(neighbor)
    names = {key: ('home' if key == (0, 0) else f'n_{key[0]:03d}_{key[1]:03d}') for key in sorted(reached)}
    graph = dict(schema_version='v41.public_navigation.v1', source_kind='provided_navigation_prior',
        nodes={names[key]: free[key].tolist() for key in sorted(reached)},
        edges=[[names[a], names[b]] for a, b in sorted(edges)])
    expanded = PublicPrimitiveGraphV41(graph)
    certificate = dict(schema_version='v43.static_navigation_certificate.v1',
        graph_content_sha256=content_sha256(graph), grid_spacing_m=spacing, robot_radius_m=radius,
        collision_model=_COLLISION_MODEL,
        start_component_only=True, all_nodes_and_edges_certified=True,
        declared_lattice_nodes=len(all_nodes), safe_lattice_nodes=len(free), reachable_nodes=len(reached),
        coarse_edges=len(edges), expanded_primitive_nodes=len(expanded.positions),
        blueprint_content_sha256=content_sha256(rectangles.tolist()),
        source_kind='offline conservative 2D collision blueprint',
        method_results_or_semantics_used=False, world_or_sensor_calls=0,
        collision_envelopes_exported=False, assumptions=deepcopy(_ASSUMPTIONS))
    return graph, certificate


def bind_public_navigation_v43(frozen_public_spec, sidecar, graph_spec, protocol):
    """Resolve only the pending graph fields into a new validated spec copy."""
    validate_public_spec(frozen_public_spec)
    if frozen_public_spec != protocol.get('public_defaults'):
        raise ValueError('every frozen public parameter must match the protocol defaults')
    _strict(sidecar, _SIDECAR_KEYS, 'navigation sidecar')
    if sidecar['schema_version'] != 'v43.navigation_sidecar.v1' or sidecar['parent_id'] not in _PARENTS:
        raise ValueError('registered development navigation sidecar required')
    if sidecar['assumptions'] != _ASSUMPTIONS:
        raise ValueError('provided-navigation assumptions must be preserved')
    _hashes(sidecar['frozen_inputs_sha256'], ('public_planner_spec.json', 'public_workspace.json', 'protocol.json'), 'frozen inputs')
    if sidecar['frozen_public_spec_content_sha256'] != content_sha256(frozen_public_spec):
        raise ValueError('frozen public spec content mismatch')
    if sidecar['frozen_protocol_content_sha256'] != content_sha256(protocol):
        raise ValueError('frozen protocol content mismatch')
    if sidecar['graph_content_sha256'] != content_sha256(graph_spec):
        raise ValueError('public graph content mismatch')
    expected_pending = dict(status='pending', relative_path=None, sha256=None)
    if (frozen_public_spec['navigation']['graph_asset'] != expected_pending
            or frozen_public_spec['navigation']['status'] != 'pending'):
        raise ValueError('V43 sidecar only resolves the frozen pending graph')
    _strict(sidecar['graph_asset'], ('status', 'relative_path', 'sha256'), 'graph asset')
    if sidecar['graph_asset']['status'] != 'materialized' or sidecar['graph_asset']['relative_path'] != 'graph.json':
        raise ValueError('graph asset must be the local materialized graph.json')
    _hashes({'graph': sidecar['graph_asset']['sha256']}, ('graph',), 'graph hash')
    if sidecar['start_state'] != dict(node='home', heading=0):
        raise ValueError('unchanged home/yaw0 start required')
    graph = PublicPrimitiveGraphV41(graph_spec, camera_height_m=frozen_public_spec['motion']['camera_height_m'])
    if graph.positions.get('home') != (.75, .75):
        raise ValueError('graph home differs from frozen start')
    resolved = deepcopy(frozen_public_spec)
    resolved['navigation'].update(status='materialized', graph_asset=deepcopy(sidecar['graph_asset']))
    validate_public_spec(resolved)
    return resolved


def make_navigation_sidecar_v43(parent_id, frozen_public_spec, workspace, protocol, graph_spec,
                                *, frozen_inputs_sha256, graph_file_sha256):
    """Offline sidecar constructor; no private scene fields are accepted."""
    sidecar = dict(schema_version='v43.navigation_sidecar.v1', parent_id=parent_id,
        frozen_inputs_sha256=deepcopy(frozen_inputs_sha256),
        frozen_public_spec_content_sha256=content_sha256(frozen_public_spec),
        frozen_protocol_content_sha256=content_sha256(protocol), graph_content_sha256=content_sha256(graph_spec),
        start_state=dict(node='home', heading=0),
        graph_asset=dict(status='materialized', relative_path='graph.json', sha256=graph_file_sha256),
        assumptions=deepcopy(_ASSUMPTIONS))
    _public_workspace(workspace, protocol)
    bind_public_navigation_v43(frozen_public_spec, sidecar, graph_spec, protocol)
    return sidecar


def load_public_navigation_bundle_v43(bundle_dir, frozen_asset_dir, protocol_path=DEFAULT_PROTOCOL_PATH_V43):
    """Read only public bundle/spec/workspace/protocol files, with byte hashes.

    ``bundle_dir`` is one DEV parent's directory, not the six-scene root.
    No private metadata is required even to reject malformed public inputs.
    The old workspace's pending flag is returned unchanged as legacy context;
    the resolved navigation status is in ``public_spec`` and ``sidecar``.
    """
    directory, frozen, protocol_path = Path(bundle_dir), Path(frozen_asset_dir), Path(protocol_path)
    if directory.name != frozen.name or directory.name not in _PARENTS:
        raise ValueError('bundle and frozen public assets must name the same registered DEV parent')
    if _sha(protocol_path) != FROZEN_PROTOCOL_SHA256_V43:
        raise ValueError('protocol bytes differ from the frozen V40 protocol')
    manifest = load_json_strict(directory/'manifest.json')
    _strict(manifest, ('schema_version', 'parent_id', 'artifact_sha256'), 'bundle manifest')
    if manifest['schema_version'] != 'v43.public_navigation_bundle.v1' or manifest['parent_id'] != directory.name:
        raise ValueError('bundle manifest identity mismatch')
    _hashes(manifest['artifact_sha256'], _PUBLIC_FILES, 'bundle files')
    for name, expected in manifest['artifact_sha256'].items():
        if (directory/name).stat().st_size > 256*1024 or _sha(directory/name) != expected:
            raise ValueError('public navigation asset bytes/hash mismatch: '+name)
    graph_spec, sidecar, certificate = (load_json_strict(directory/name) for name in _PUBLIC_FILES)
    if sidecar.get('parent_id') != directory.name or sidecar.get('graph_asset', {}).get('sha256') != _sha(directory/'graph.json'):
        raise ValueError('sidecar graph byte hash or parent mismatch')
    sources = {'public_planner_spec.json': frozen/'public_planner_spec.json',
               'public_workspace.json': frozen/'public_workspace.json', 'protocol.json': protocol_path}
    if sidecar.get('frozen_inputs_sha256') != {name: _sha(path) for name, path in sources.items()}:
        raise ValueError('frozen public input hash mismatch')
    base, workspace, protocol = (load_json_strict(sources[name]) for name in sources)
    _public_workspace(workspace, protocol)
    spec = bind_public_navigation_v43(base, sidecar, graph_spec, protocol)
    graph = PublicPrimitiveGraphV41(graph_spec, camera_height_m=spec['motion']['camera_height_m'])
    expected_keys = {'schema_version', 'graph_content_sha256', 'grid_spacing_m', 'robot_radius_m',
        'collision_model', 'start_component_only', 'all_nodes_and_edges_certified', 'declared_lattice_nodes',
        'safe_lattice_nodes', 'reachable_nodes', 'coarse_edges', 'expanded_primitive_nodes',
        'blueprint_content_sha256', 'source_kind', 'method_results_or_semantics_used',
        'world_or_sensor_calls', 'collision_envelopes_exported', 'assumptions'}
    _strict(certificate, expected_keys, 'static public certificate')
    if (certificate['schema_version'] != 'v43.static_navigation_certificate.v1'
            or certificate['graph_content_sha256'] != content_sha256(graph_spec)
            or certificate['reachable_nodes'] != len(graph.original_nodes)
            or certificate['coarse_edges'] != len(graph_spec['edges'])
            or certificate['expanded_primitive_nodes'] != len(graph.positions)
            or certificate['grid_spacing_m'] != 1. or certificate['robot_radius_m'] != .2
            or certificate['collision_model'] != _COLLISION_MODEL
            or certificate['source_kind'] != 'offline conservative 2D collision blueprint'
            or certificate['start_component_only'] is not True
            or certificate['all_nodes_and_edges_certified'] is not True
            or certificate['method_results_or_semantics_used'] is not False
            or certificate['collision_envelopes_exported'] is not False
            or certificate['world_or_sensor_calls'] != 0 or certificate['assumptions'] != _ASSUMPTIONS):
        raise ValueError('public graph certificate mismatch')
    if (any(type(certificate[key]) is not int or certificate[key] < 1 for key in
            ('declared_lattice_nodes', 'safe_lattice_nodes', 'reachable_nodes', 'expanded_primitive_nodes'))
            or not certificate['reachable_nodes'] <= certificate['safe_lattice_nodes'] <= certificate['declared_lattice_nodes'] <= 256):
        raise ValueError('certificate node counts invalid')
    _hashes({'blueprint': certificate['blueprint_content_sha256']}, ('blueprint',), 'blueprint binding')
    return dict(graph=graph, graph_spec=graph_spec, public_spec=spec, workspace=workspace,
        certificate=certificate, sidecar=sidecar, public_bundle_sha256=_sha(directory/'manifest.json'),
        home_state=PrimitiveStateV41('home', 0))


def select_candidate_states_v43(graph, current, paid_camera_states=(), limit=32):
    """Shared public pool with position diversity and no semantic ranking.

    A unit-cost BFS covers the supplied primitive graph, respecting removed
    edges. At most four unpaid headings per position and at least the closest
    eight eligible positions (when available) prevent a pool of only nearby
    turns. Original coarse positions and the current intermediate node qualify.
    Controller goal persistence and return feasibility are separate contracts.
    """
    if type(graph) is not PublicPrimitiveGraphV41:
        raise TypeError('strict public primitive graph required')
    graph.validate_state(current)
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError('candidate limit must be in 1..32')
    paid = set(paid_camera_states)
    for state in paid:
        graph.validate_state(state)
    distance, queue = {current: 0}, deque([current])
    while queue:
        state = queue.popleft()
        for action in ('left', 'right', 'forward'):
            try:
                successor = graph.successor(state, action)
            except ValueError:
                continue
            if successor not in distance:
                distance[successor] = distance[state]+1; queue.append(successor)
    positions = graph.original_nodes | {current.node}
    grouped = {}
    for state, cost in distance.items():
        if state.node in positions and state not in paid:
            grouped.setdefault(state.node, []).append((cost, state.heading, state))
    for rows in grouped.values():
        rows.sort(key=lambda x: (x[0], x[1]))
    ordered_nodes = sorted(grouped, key=lambda node: (grouped[node][0][0], node))
    selected_nodes = ordered_nodes[:min(8, limit)]
    index = len(selected_nodes)
    while sum(min(4, len(grouped[node])) for node in selected_nodes) < limit and index < len(ordered_nodes):
        selected_nodes.append(ordered_nodes[index]); index += 1
    chosen = [grouped[node][0][2] for node in selected_nodes[:limit]]
    quota = {node: 1 for node in selected_nodes[:limit]}
    remaining = sorted((cost, node, heading, state) for node in selected_nodes
                       for cost, heading, state in grouped[node][1:])
    for _, node, _, state in remaining:
        if len(chosen) >= limit:
            break
        if quota.get(node, 0) < 4:
            chosen.append(state); quota[node] = quota.get(node, 0)+1
    receipt = dict(schema_version='v43.public_candidate_pool.v1', limit=limit, selected=len(chosen),
        graph_input_sha256=graph.input_sha256, current_state=current.__dict__,
        paid_camera_states_excluded=len(paid), reachable_primitive_states=len(distance),
        original_coarse_positions_only_except_current=True, maximum_headings_per_node=4,
        preferred_minimum_positions=8, selection_uses_semantics=False,
        selection_uses_observed_surface_or_private_geometry=False,
        selection='public primitive BFS; seed up to eight closest unpaid positions then nearest headings with per-position quota',
        candidates=[dict(node=state.node, heading=state.heading, outbound_primitive_cost=distance[state]) for state in chosen])
    return chosen, receipt
