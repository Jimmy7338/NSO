#!/usr/bin/env python3
"""Freeze an unexecuted cold start and its separate source-only reference geometry."""

import argparse
import importlib.util
from pathlib import Path

from scene import ANALYSES, DEVELOPMENT, HERE, LAYOUTS


PRIOR = ANALYSES / 'same_goal_route_choice_20261121/holdout/prepare.py'


def load_reference_preparer():
    spec = importlib.util.spec_from_file_location('weighted_reused_source_preparer', PRIOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=LAYOUTS)
    args = parser.parse_args()
    inherited = load_reference_preparer()
    reference_dir = HERE / 'evaluator_references' / args.layout
    actor_dir = HERE / 'prepared' / args.layout
    if reference_dir.exists() or actor_dir.exists():
        raise FileExistsError(f'Cold start or source reference already exists: {args.layout}')
    initial_dir, reachable, sensor_config, _, unknown = inherited.initial(args.layout)
    initial_path = initial_dir / 'initial_state.json'
    state = inherited.read(initial_path)
    state['scene_role'] = 'known_development' if args.layout in DEVELOPMENT else 'fresh_holdout'
    initial_path.unlink()
    inherited.write(initial_path, state)
    inherited.reference(args.layout, initial_dir, reachable, sensor_config, state)
    print(inherited.json.dumps({'layout': args.layout, 'bounds_grid_rc': state['shape'],
                               'role': state['scene_role'], 'actor_select_calls': state['policy_select_calls'],
                               'reachable_cells': int(reachable.sum()), 'unknown_reachable_at_start': unknown}),
          flush=True)


if __name__ == '__main__':
    main()
