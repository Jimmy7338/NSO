#!/usr/bin/env python3
"""Read-only diagnosis of all recorded T collision branches, without rerouting."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from env.virtual3d_response_v7 import create_response_world
from scripts.eval_counterfactual_views import remap, update_collision
from scripts.compact_response_v7_meshes import validate_run_assets
from utils.grid_geometry import inflated_obstacles
from utils.rgbd_contract import RGBDFrame, PlanarScan


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = []; hashes = {}
    for context in (f'T{i}' for i in range(8)):
        run = args.runs / context
        meta = read(run / 'metadata.json')
        validate_run_assets(run)
        for name, digest in meta['source_sha256'].items():
            assert sha(ROOT / name) == digest, f'current source differs from archived run: {name}'
        hashes[str(run / 'artifact_hashes.json')] = sha(run / 'artifact_hashes.json')
        for family in ('storage_shelves', 'ventilation_baffles'):
            folder = run / family
            routes = read(folder / 'candidates.json')
            for route in routes:
                branch = folder / f"candidate_{route['candidate_id']:03d}"
                outcome = read(branch / 'outcome.json')
                if outcome['failure'] is None:
                    continue
                # GT below is used only to diagnose an already recorded failure.
                world = create_response_world(context, family)
                c = world.config
                prefix = folder / 'prefix'; records = read(prefix / 'records.json')
                frames = [RGBDFrame.load(prefix / 'frames' / f'{j:04d}.npz') for j in range(len(records))]
                scans = [PlanarScan.load(prefix / 'scans' / f'{j:04d}.npz') for j in range(len(records))]
                mapper = remap(frames, scans, records, c, world.shape)
                initial_safe = ~inflated_obstacles(mapper.belief != 0, c.robot_radius_m / c.resolution_m)
                actions = read(branch / 'actions.json'); trace = []
                for row in actions:
                    index = row['action_index']; target = tuple(route['states'][index][:2])
                    safe = ~inflated_obstacles(mapper.belief != 0, c.robot_radius_m / c.resolution_m)
                    xy = np.array([(target[1] + .5) * c.resolution_m,
                                   (world.shape[0] - target[0] - .5) * c.resolution_m])
                    distances = []
                    for x, y, sx, sy in world.boxes:
                        nearest = np.maximum([x, y], np.minimum(xy, [x + sx, y + sy]))
                        distances.append(float(np.linalg.norm(xy - nearest)))
                    trace.append({'action_index': index, 'action': row['action'], 'target': list(target),
                        'prefix_map_target_safe': bool(initial_safe[target]),
                        'latest_map_target_safe_before_action': bool(safe[target]),
                        'collision_recorded': row['collision'],
                        'GT_conservative_grid_blocks_target': bool(world._blocked[target]),
                        'GT_target_center_to_physical_boxes_m': min(distances),
                        'robot_radius_m': c.robot_radius_m})
                    frame = RGBDFrame.load(branch / 'frames' / f'{index:04d}.npz')
                    scan = PlanarScan.load(branch / 'scans' / f'{index:04d}.npz')
                    mapper.update(frame, scan)
                    update_collision(mapper, tuple(row['position']), row['heading'], row['collision'])
                unsafe = [row['action_index'] for row in trace if not row['latest_map_target_safe_before_action']]
                results.append({'context': context, 'family': family, 'candidate_id': route['candidate_id'],
                    'failure': outcome['failure'], 'planned_actions': outcome['planned_actions'],
                    'paid_actions': outcome['paid_actions'], 'first_latest_map_unsafe_action': min(unsafe) if unsafe else None,
                    'last_action': trace[-1], 'trace': trace})
        validate_run_assets(run)
    report = {'status': 'complete', 'scope': 'all recorded T failures; diagnostic only, no new routes or candidate changes',
              'new_physical_branches': 0, 'failed_branches': len(results), 'rows': results,
              'input_manifest_sha256': hashes, 'script_sha256': sha(Path(__file__)),
              'GT_used_only_in_posthoc_diagnosis': True}
    (args.output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'trace'} for r in results], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
