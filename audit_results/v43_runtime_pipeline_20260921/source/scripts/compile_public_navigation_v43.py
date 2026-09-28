#!/usr/bin/env python3
"""Compile six static DEV navigation blueprints; no World or sensor calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nso.public_navigation_v43 import (compile_blueprint_graph_v43, make_navigation_sidecar_v43,
    load_public_navigation_bundle_v43, DEFAULT_PROTOCOL_PATH_V43, FROZEN_PROTOCOL_SHA256_V43)
from nso.scene_contract_v40 import canonical_json_bytes, load_json_strict


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_bytes(canonical_json_bytes(value))


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if sha(DEFAULT_PROTOCOL_PATH_V43) != FROZEN_PROTOCOL_SHA256_V43:
        raise ValueError('frozen protocol bytes changed')
    protocol = load_json_strict(DEFAULT_PROTOCOL_PATH_V43)
    source_manifest = load_json_strict(source/'manifest.json')
    # Lock the six predeclared development parents. No held-out scene or seed
    # path is accepted by this compiler, even if present beside these assets.
    inputs, prepared = {}, []
    for family in 'ABCDEF':
        parent = f'DEV_{family}_00'; base = source/parent
        names = ('public_planner_spec.json', 'public_workspace.json', 'evaluation_private/instances.json')
        for name in names:
            path = base/name; relative = f'{parent}/{name}'
            if sha(path) != source_manifest['artifact_sha256'].get(relative):
                raise ValueError('frozen input asset hash mismatch: '+relative)
            inputs[str(path.relative_to(ROOT))] = sha(path)
        spec, workspace, metadata = (load_json_strict(base/name) for name in names)
        if metadata['parent_id'] != parent or metadata['public_workspace'] != workspace:
            raise ValueError('private compiler blueprint must match the declared public workspace')
        rectangles = []
        for item in metadata['private_instances']:
            lower, upper = item['world_aabb_m']
            rectangles.append([lower[0], upper[0], lower[1], upper[1]])
        for box in metadata['background_boxes']:
            if box[5] > 0.:  # Same conservative collision policy as V41; floor excluded.
                rectangles.append([box[0], box[1], box[2], box[3]])
        graph, certificate = compile_blueprint_graph_v43(workspace, rectangles)
        prepared.append((parent, spec, workspace, graph, certificate))
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for parent, spec, workspace, graph, certificate in prepared:
        folder = output/parent; folder.mkdir()
        write(folder/'graph.json', graph)
        sidecar = make_navigation_sidecar_v43(parent, spec, workspace, protocol, graph,
            frozen_inputs_sha256={'public_planner_spec.json': sha(source/parent/'public_planner_spec.json'),
                'public_workspace.json': sha(source/parent/'public_workspace.json'), 'protocol.json': sha(DEFAULT_PROTOCOL_PATH_V43)},
            graph_file_sha256=sha(folder/'graph.json'))
        write(folder/'sidecar.json', sidecar); write(folder/'certificate.json', certificate)
        write(folder/'manifest.json', dict(schema_version='v43.public_navigation_bundle.v1', parent_id=parent,
            artifact_sha256={name: sha(folder/name) for name in ('graph.json', 'sidecar.json', 'certificate.json')}))
        loaded = load_public_navigation_bundle_v43(folder, source/parent)
        summaries.append(dict(parent_id=parent, coarse_nodes=len(loaded['graph'].original_nodes),
            coarse_edges=len(graph['edges']), primitive_nodes=len(loaded['graph'].positions),
            graph_sha256=sha(folder/'graph.json'), bundle_sha256=sha(folder/'manifest.json')))
    result = dict(schema_version='v43.static_public_navigation_release.v1', status='passed',
        scope='six development known-navigation priors; static geometry certification only', parents=summaries,
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in
            (Path(__file__), ROOT/'nso/public_navigation_v43.py', ROOT/'nso/primitive_navigation_v41.py',
             DEFAULT_PROTOCOL_PATH_V43, source/'manifest.json')},
        offline_compiler_input_sha256=inputs, development_worlds_created=0, sensor_frames_created=0,
        planner_rollouts_executed=0, held_out_seed_or_scene_read=False, semantic_performance_claim=False,
        blueprint_detail_exposed_online='only coarse reachable node/edge constraints; no obstacle coordinates or class/structure metadata',
        shared_with_all_methods=True, unknown_environment_exploration=False, old_assets_modified=False)
    write(output/'result.json', result)
    write(output/'artifact_sha256.json', {str(path.relative_to(output)): sha(path)
        for path in sorted(output.rglob('*')) if path.is_file()})
    total = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    if total > 1024**2:
        raise ValueError('static public navigation package exceeded declared 1 MiB cap')
    print(json.dumps(dict(status='passed', parents=summaries, output_bytes=total), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'audit_results/v40_p1_development_geometry_20260920')
    parser.add_argument('--output', type=Path, default=ROOT/'audit_results/v43_public_navigation_r1_20260921')
    args = parser.parse_args()
    run(args.source, args.output)
