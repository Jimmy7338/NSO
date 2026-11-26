#!/usr/bin/env python3
"""Aggregate the fixed spatial masks over the original room partition coordinates."""

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SCENE = ROOT.parent / 'same_goal_route_choice_20261121/holdout/scene.py'
LABELS = ('neither', 'G_only', 'S_only', 'both')
ROOMS = ('southwest', 'southeast', 'northwest', 'northeast')


def room_specifications(source):
    tree = ast.parse(source.read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'ORDINARY' for target in statement.targets
        ):
            return {ast.literal_eval(key): {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords}
                    for key, value in zip(statement.value.keys, statement.value.values, strict=True)}
    raise ValueError('Frozen room specifications not found')


def aggregate(reference, state, weights, select):
    counts = np.bincount(state[select], minlength=4)
    areas = np.bincount(state[select], weights=weights[select], minlength=4)
    bounds = {}
    for bit in (1, 2):
        spatial = reference[select & (state == bit)]
        bounds[LABELS[bit]] = [spatial.min(0).tolist(), spatial.max(0).tolist()] if len(spatial) else None
    return {'counts': dict(zip(LABELS, counts.tolist(), strict=True)),
            'area_m2': dict(zip(LABELS, areas.tolist(), strict=True)),
            'net_S_minus_G_m2': float(areas[2] - areas[1]),
            'exclusive_bounds_world_xyz_m': bounds}


def regional_audit(pairs, registered_rows, specs):
    output = {}
    for panel in pairs['panels'].tolist():
        identity = registered_rows[panel]
        spec = specs[identity['layout']]
        points = pairs[f'{panel}__reference_xyz']
        state = pairs[f'{panel}__reference_status']
        weights = pairs[f'{panel}__reference_weight_m2']
        source = pairs[f'{panel}__reference_source_id']
        if not (len(points) == len(state) == len(weights) == len(source)):
            raise ValueError(f'Fixed point columns are misaligned for {panel}')
        east = points[:, 0] >= spec['split_x']
        north = points[:, 1] >= spec['split_y']
        code = east.astype(np.uint8) + 2 * north.astype(np.uint8)
        zero = points[:, 2] == 0
        all_regions = {}
        for idx, room in enumerate(ROOMS):
            region = code == idx
            all_regions[room] = {
                'whole_region': aggregate(points, state, weights, region),
                'other_source_surface': aggregate(points, state, weights, region & (source == 0)),
                'fixture_source_surface': aggregate(points, state, weights, region & (source != 0)),
                'exact_z_zero': aggregate(points, state, weights, region & zero),
                'nonzero_z': aggregate(points, state, weights, region & ~zero),
            }
        net = sum(value['whole_region']['net_S_minus_G_m2'] for value in all_regions.values())
        if not np.isclose(net, float(identity['scene_net_S_minus_G_m2']), rtol=0, atol=1e-9):
            raise ValueError(f'Regional area does not recover the sealed point comparison for {panel}')
        output[panel] = {'layout': identity['layout'], 'comparison_scope': identity['comparison_scope'],
                         'actual_paid': int(identity['actual_paid']), 'original_split_xy_m': [spec['split_x'], spec['split_y']],
                         'regions': all_regions}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-scene', type=Path, default=SCENE)
    parser.add_argument('--points', type=Path, default=ROOT / 'spatial_pairs.npz')
    parser.add_argument('--pairs', type=Path, default=ROOT / 'spatial_pairs.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'regional_surface_audit.json')
    options = parser.parse_args()
    with options.pairs.open(newline='') as stream:
        records = {row['panel']: row for row in csv.DictReader(stream)}
    with np.load(options.points, allow_pickle=False) as points:
        outputs = regional_audit(points, records, room_specifications(options.source_scene))
    options.output.write_text(json.dumps({
        'coordinate_rule': 'East/North include the samples on the original room split planes, if any.',
        'height_rule': 'z=0 is a slice in the source coordinate frame, used without adding surface categories.',
        'area_rule': 'Use the exact frozen global point weight. No reweighting or new ground truth.',
        'panels': outputs,
    }, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
