"""Static new-scene navigation and an exclusively public runtime loader.

Compilation alone may read private collision envelopes. Runtime loading reads
only its separate public bundle; no old V40/V43 identifier or pin is bypassed.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import re

from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41
from nso.public_navigation_v43 import compile_blueprint_graph_v43
from nso.scene_contract_v40 import canonical_json_bytes, content_sha256, validate_public_spec
from nso.semantic_scene_assets import (
    ASSET_FILES, CONDITIONS, PARENTS, asset_id, checked_asset_file, checked_manifest,
    file_sha256, parse_asset_id, read_json,
)


PUBLIC_FILES = ('graph.json', 'certificate.json', 'public_workspace.json', 'public_planner_spec.json')
MAX_NAVIGATION_BYTES = 2 * 1024 ** 2
_SHA = re.compile(r'[0-9a-f]{64}')


def _workspace(workspace):
    expected = {'schema_version', 'bounds_xy_m', 'room_inner_bounds_xy_m',
                'start_position_world_m', 'start_yaw_deg', 'marker_palette',
                'navigation_graph_status', 'instance_truth_included', 'source'}
    if (not isinstance(workspace, dict) or set(workspace) != expected
            or workspace['schema_version'] != 'v40.development_workspace.v1'
            or workspace['instance_truth_included'] is not False
            or workspace['start_position_world_m'] != [.75, .75, .9]
            or workspace['start_yaw_deg'] != 0.
            or workspace['source'] != 'declared_task_workspace_not_facility_roi'):
        raise ValueError('public workspace whitelist or start contract mismatch')


def build_public_navigation(asset_root, output_root, *, expected_asset_manifest_sha256):
    """Offline only: certify class/structure-independent collision envelopes."""
    source, output = Path(asset_root).resolve(), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError('new navigation cannot overwrite an existing bundle')
    manifest = checked_manifest(source, expected_asset_manifest_sha256)
    payloads, entries, parent_graphs = {}, {}, {}
    for identity in sorted(manifest['assets']):
        parent, condition = parse_asset_id(identity)
        directory = source / identity
        private = read_json(checked_asset_file(directory, ASSET_FILES[2], manifest))
        workspace = read_json(checked_asset_file(directory, 'public_workspace.json', manifest))
        spec = read_json(checked_asset_file(directory, 'public_planner_spec.json', manifest))
        _workspace(workspace); validate_public_spec(spec)
        if (private['public_workspace'] != workspace or private['parent_id'] != parent
                or private['asset_id'] != identity or private['condition'] != condition):
            raise ValueError('private blueprint and public asset identity mismatch')
        rectangles = []
        for item in private['private_instances']:
            lower, upper = item['navigation_envelope_aabb_m']
            rectangles.append([lower[0], upper[0], lower[1], upper[1]])
        rectangles.extend([box[0], box[1], box[2], box[3]] for box in private['background_boxes'] if box[5] > 0)
        graph, certificate = compile_blueprint_graph_v43(workspace, rectangles)
        if certificate['safe_lattice_nodes'] != certificate['reachable_nodes']:
            raise ValueError('new parent has an unreachable intended safe coarse component')
        graph_bytes = canonical_json_bytes(graph)
        graph_sha = hashlib.sha256(graph_bytes).hexdigest()
        if parent in parent_graphs and parent_graphs[parent] != graph_sha:
            raise ValueError('hidden structure or cue condition changed the public navigation graph')
        parent_graphs[parent] = graph_sha
        resolved = deepcopy(spec)
        resolved['navigation'].update(status='materialized',
            graph_asset=dict(status='materialized', relative_path='graph.json', sha256=graph_sha))
        validate_public_spec(resolved)
        certificate.update(schema_version='semantic_scene_navigation_certificate.v1',
            asset_manifest_sha256=expected_asset_manifest_sha256,
            common_graph_across_relationship_conditions=True,
            collision_envelopes_are_all_structure_union=True)
        payloads[identity+'/graph.json'] = graph_bytes
        payloads[identity+'/certificate.json'] = canonical_json_bytes(certificate)
        payloads[identity+'/public_workspace.json'] = canonical_json_bytes(workspace)
        payloads[identity+'/public_planner_spec.json'] = canonical_json_bytes(resolved)
        entries[identity] = dict(files=list(PUBLIC_FILES), graph_sha256=graph_sha,
            coarse_nodes=certificate['reachable_nodes'], expanded_nodes=certificate['expanded_primitive_nodes'])
    result = dict(schema='semantic_scene_public_navigation.v1', assets=entries,
        source_asset_manifest_sha256=expected_asset_manifest_sha256,
        source_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
            for path in (Path(__file__), Path(__file__).with_name('public_navigation_v43.py'),
                         Path(__file__).with_name('primitive_navigation_v41.py'))},
        artifact_sha256={name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
        worlds_created=0, policy_trajectories_created=0, quality_evaluations=0,
        known_navigation_prior=True, independent_parent_layouts=6,
        shared_facility_prototype_family=True, old_assets_modified=False)
    payloads['manifest.json'] = canonical_json_bytes(result)
    total = sum(len(value) for value in payloads.values())
    if total > MAX_NAVIGATION_BYTES:
        raise ValueError('new public navigation bundle exceeds 2 MiB cap')
    for name, value in payloads.items():
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(value)
    return dict(output_root=str(output), assets=len(entries), bytes_written=total,
        manifest_sha256=file_sha256(output/'manifest.json'),
        source_asset_manifest_sha256=expected_asset_manifest_sha256, worlds_created=0)


def load_semantic_navigation_bundle(bundle_dir, *, expected_manifest_sha256):
    """Open only the public bundle; neither private assets nor design are read."""
    directory = Path(bundle_dir)
    parse_asset_id(directory.name)
    if directory.is_symlink() or not isinstance(expected_manifest_sha256, str) or not _SHA.fullmatch(expected_manifest_sha256):
        raise ValueError('plain public bundle and externally pinned manifest required')
    root = directory.parent.resolve()
    manifest = read_json(root/'manifest.json')
    if file_sha256(root/'manifest.json') != expected_manifest_sha256:
        raise ValueError('public navigation manifest differs from external pin')
    if (manifest.get('schema') != 'semantic_scene_public_navigation.v1'
            or set(manifest.get('assets', {})) != {asset_id(p, c) for p in PARENTS for c in CONDITIONS}
            or manifest['assets'][directory.name].get('files') != list(PUBLIC_FILES)):
        raise ValueError('complete new public navigation manifest required')
    loaded = {}
    for name in PUBLIC_FILES:
        path = directory/name
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or file_sha256(path) != manifest['artifact_sha256'].get(directory.name+'/'+name)):
            raise ValueError('public navigation asset hash mismatch: ' + name)
        loaded[name] = read_json(path, maximum_bytes=256*1024)
    graph_spec, spec, workspace, certificate = (loaded[name] for name in
        ('graph.json', 'public_planner_spec.json', 'public_workspace.json', 'certificate.json'))
    _workspace(workspace); validate_public_spec(spec)
    graph = PublicPrimitiveGraphV41(graph_spec, camera_height_m=spec['motion']['camera_height_m'])
    graph_sha = file_sha256(directory/'graph.json')
    if (graph.positions.get('home') != (.75, .75)
            or spec['navigation']['graph_asset'] != dict(status='materialized', relative_path='graph.json', sha256=graph_sha)
            or certificate.get('schema_version') != 'semantic_scene_navigation_certificate.v1'
            or certificate.get('graph_content_sha256') != content_sha256(graph_spec)
            or certificate.get('asset_manifest_sha256') != manifest['source_asset_manifest_sha256']
            or certificate.get('reachable_nodes') != len(graph.original_nodes)
            or certificate.get('safe_lattice_nodes') != len(graph.original_nodes)
            or certificate.get('expanded_primitive_nodes') != len(graph.positions)
            or certificate.get('world_or_sensor_calls') != 0
            or certificate.get('method_results_or_semantics_used') is not False
            or certificate.get('collision_envelopes_are_all_structure_union') is not True):
        raise ValueError('public graph/spec/certificate binding mismatch')
    return dict(graph=graph, graph_spec=graph_spec, public_spec=spec, workspace=workspace,
        certificate=certificate, home_state=PrimitiveStateV41('home', 0),
        public_bundle_sha256=expected_manifest_sha256,
        asset_manifest_sha256=manifest['source_asset_manifest_sha256'])
