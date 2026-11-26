#!/usr/bin/env python3
"""Prepare only frozen cold starts and independent evaluator-side references."""

import argparse
import json
from pathlib import Path

from scene import ANALYSES, ARCHIVED_SCENES, DEVELOPMENT, HERE, HOLDOUT, LAYOUTS, SPECS, load_module


PRIOR = ANALYSES / 'same_goal_route_choice_20261121/holdout/prepare.py'


def link_development(layout):
    for subdir in ('prepared', 'evaluator_references'):
        target = HERE/subdir/layout
        source = ARCHIVED_SCENES/subdir/layout
        target.parent.mkdir(exist_ok=True)
        if not source.is_dir():
            raise FileNotFoundError(source)
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        target.symlink_to(source, target_is_directory=True)


def prepare(layout):
    if layout in DEVELOPMENT:
        link_development(layout)
        print(json.dumps({'layout':layout, 'role':'known_development', 'archives_reused':True}), flush=True)
        return
    if any((HERE/folder/layout).exists() for folder in ('prepared','evaluator_references')):
        raise FileExistsError(f'An attempt already exists for {layout}')
    inherited = load_module('regional_floor_reference_preparer', PRIOR)
    inherited.SEED = SPECS[layout]['seed']
    directory, reachable, config, metadata, unknown = inherited.initial(layout)
    initial_path = directory/'initial_state.json'
    metadata['scene_role'] = 'fresh_holdout'
    temporary = initial_path.with_suffix('.tmp')
    inherited.write(temporary, metadata)
    temporary.replace(initial_path)
    inherited.reference(layout, directory, reachable, config, metadata)
    print(json.dumps({'layout':layout, 'grid_shape':list(reachable.shape),
                      'reachable_footprint_centers':int(reachable.sum()), 'initial_unknown':unknown,
                      'actor_calls':0, 'seed':SPECS[layout]['seed']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=LAYOUTS)
    prepare(parser.parse_args().layout)
