#!/usr/bin/env python3
"""Evaluate acquired E1 branch meshes on the historical fixed source reference."""

import csv
import hashlib
import json
from pathlib import Path
import sys

from run_e1 import HERE, OLD, ORIGINAL, PREFIX, PREPARED, REFERENCE, SEED, lineage, load_matrix, read, sha, verify_freeze


def checked_manifest(attempt):
    manifest = read(attempt / 'artifact_manifest.json')
    assert manifest['status'] == 'completed_execution'
    for rel, expected in manifest['files_sha256'].items():
        assert sha(attempt / rel) == expected, rel


def mesh_report(evaluator, path, coverage):
    import numpy as np
    import open3d as o3d
    from utils.reconstruction_metrics import ray_scene

    mesh = o3d.io.read_triangle_mesh(str(path))
    assert len(mesh.triangles) and len(mesh.vertices)
    values = evaluator.evaluate(mesh, coverage, thresholds=(.05,))
    d = ray_scene(mesh).compute_distance(o3d.core.Tensor(evaluator.reference), nthreads=1).numpy()
    covered = d <= .05
    assert np.isclose(covered.mean(), values['recall_05cm'], atol=1e-14, rtol=0)
    return values, covered


def increments(before, after, area_per_point):
    gross = int((~before & after).sum())
    lost = int((before & ~after).sum())
    return dict(gross_points=gross, lost_points=lost, net_points=gross-lost,
                gross_m2=gross*area_per_point, lost_m2=lost*area_per_point,
                net_m2=(gross-lost)*area_per_point)


def later_observation(rows, boundary, indicator):
    return next((dict(action=x['step'], additional_paid_actions=x['step']-boundary['paid_at_route_end'],
                      additional_path_m=x['path_length_m']-boundary['path_length_m'],
                      position=x['position'], option_group=x['option_group'])
                 for x in rows if x['step'] > boundary['paid_at_route_end'] and indicator(x)), None)


def report(attempt, evaluator, reachable, per_point, expected_terminal_5):
    import numpy as np

    checked_manifest(attempt)
    result = read(attempt / 'result.json')
    rows = [json.loads(line) for line in (attempt / 'steps.jsonl').read_text().splitlines()]
    assert len(rows) == result['paid_actions'] <= 512
    final_row, prefix_row = rows[-1], rows[PREFIX-1]
    with np.load(attempt / 'final_belief.npz', allow_pickle=False) as z:
        coverage = float((reachable & (z['belief'] == 0)).sum() / reachable.sum())
        final_pose = [*z['position'].tolist(), int(z['heading'])]
    with np.load(attempt / 'prefix_belief.npz', allow_pickle=False) as z:
        prefix_coverage = float((reachable & (z['belief'] == 0)).sum() / reachable.sum())
    assert final_pose == result['final_pose'] == [*final_row['position'], final_row['heading']]
    original_start = read(PREPARED / 'initial_state.json')['anchor']
    assert result['returned_to_anchor'] == (final_pose == original_start)
    assert result['collisions'] == sum(row['collision'] for row in rows)
    assert coverage == final_row['free_correct_fraction'] == result['final_evaluation']['free_correct_fraction']
    assert prefix_coverage == prefix_row['free_correct_fraction']
    final_metrics, final_mask = mesh_report(evaluator, attempt / 'final_tsdf_mesh.ply', coverage)
    prefix_metrics, prefix_mask = mesh_report(evaluator, attempt / 'prefix_mesh.ply', prefix_coverage)
    boundary = result['route']
    route_end_metrics = route_area = None
    frontier = coverage_arrival = None
    if boundary and boundary['arrived']:
        route_end_metrics, route_mask = mesh_report(evaluator, attempt / 'route_end_mesh.ply',
                                                   boundary['observed_free_fraction'])
        route_area = increments(prefix_mask, route_mask, per_point)
        frontier = later_observation(rows, boundary, lambda x: x['observed_frontier_at_robot'])
        coverage_arrival = later_observation(rows, boundary, lambda x:
            x['option_arrived'] and x['option_group'].startswith('coverage') and
            x['option_id'] != rows[boundary['paid_at_route_end']-1]['option_id'])
    if result['candidate_id'] == 5:
        assert final_metrics['f1_05cm'] == expected_terminal_5
        assert result['historical_tail_exactly_verified']
    return dict(arm=result['candidate_id'], layout='E1', actions=len(rows), C_free=coverage,
                F1_scene_05cm=final_metrics['f1_05cm'], J=coverage*final_metrics['f1_05cm'],
                returned=result['returned_to_anchor'], collisions=result['collisions'],
                stopping_reason=result['termination']['reason'], route=boundary,
                prefix_metrics=prefix_metrics, route_end_metrics=route_end_metrics,
                terminal_metrics=final_metrics, route_surfaces=route_area,
                terminal_surfaces=increments(prefix_mask, final_mask, per_point),
                first_observed_frontier_contact=frontier, first_selected_coverage_arrival=coverage_arrival,
                evidence_manifest=str(attempt / 'artifact_manifest.json'),
                evidence_manifest_sha256=sha(attempt / 'artifact_manifest.json'))


def main():
    matrix = load_matrix()
    sys.path.insert(0, str(OLD / 'surface_evaluation'))
    import numpy as np
    from exterior_reference import build_reference

    verify_freeze()
    lineage(matrix)
    world, _, _ = matrix.world_and_packet(SEED)
    evaluator, _, _, info = build_reference(world, count=100000, seed=20261119)
    historical = read(REFERENCE / 'result.json')
    with np.load(REFERENCE / 'fixed_reference.npz', allow_pickle=False) as fixed:
        np.testing.assert_array_equal(fixed['points'], evaluator.reference)
        np.testing.assert_array_equal(fixed['classes'], evaluator.classes)
    reference_digest = hashlib.sha256(evaluator.reference.tobytes() + evaluator.classes.tobytes()).hexdigest()
    assert reference_digest == historical['reference_points_and_classes_sha256']
    with np.load(PREPARED / 'evaluator_truth.npz', allow_pickle=False) as prepared:
        reachable = prepared['reachable']
    np.testing.assert_array_equal(reachable, world.reachable)
    per_point = info['exterior_mesh_area_m2'] / info['area_samples_drawn']
    paths = [HERE / f'candidate_{arm}' / 'attempt_001' for arm in (3, 5)]
    a, b = [read(p / 'intervention.json') for p in paths]
    for key in ('prefix_packet_sequence_sha256', 'prefix_belief_sha256', 'options', 'original_scores'):
        assert a[key] == b[key]
    assert sha(paths[0] / 'prefix_mesh.ply') == sha(paths[1] / 'prefix_mesh.ply')
    rows = [report(p, evaluator, reachable, per_point, historical['metrics']['f1_05cm']) for p in paths]
    keys = ('arm', 'layout', 'actions', 'C_free', 'F1_scene_05cm', 'J', 'returned', 'collisions', 'stopping_reason')
    cell_deltas = {key: rows[1][key] - rows[0][key] for key in ('C_free', 'F1_scene_05cm', 'J', 'actions')}
    with (HERE / 'e1_comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    with (HERE / 'e1_report.json').open('x') as stream:
        json.dump(dict(status='complete_independent_evaluation', layout_count=1, rows=rows,
                       candidate_5_minus_candidate_3=cell_deltas, reference_point_count=len(evaluator.reference),
                       reference_points_classes_sha256=reference_digest, area_per_point_m2=per_point,
                       historical_candidate5_mesh_F1_exact=True, freeze_sha256=sha(HERE / 'freeze.json')),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete_independent_evaluation', deltas=cell_deltas)))


if __name__ == '__main__':
    main()
