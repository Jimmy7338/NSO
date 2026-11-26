#!/usr/bin/env python3
"""Generate original-runtime bootstrap packets and independent surface references."""

import argparse
import json

from scene import ANALYSES, HERE, LAYOUTS, SPECS, load_module


PREPARER = ANALYSES / 'same_goal_route_choice_20261121' / 'holdout' / 'prepare.py'


def prepare(layout):
    if any((HERE/folder/layout).exists() for folder in ('prepared', 'evaluator_references')):
        raise FileExistsError(f'Preparation already attempted for {layout}')
    native = load_module('regional_revision_native_preparer', PREPARER)
    native.SEED = SPECS[layout]['seed']
    directory, reachable, config, metadata, unknown = native.initial(layout)
    native.reference(layout, directory, reachable, config, metadata)
    print(json.dumps({
        'layout': layout,
        'shape': list(reachable.shape),
        'seed': SPECS[layout]['seed'],
        'initial_unknown_reachable': unknown,
        'actor_select_calls': 0,
    }), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=LAYOUTS)
    prepare(parser.parse_args().layout)
