#!/usr/bin/env python3
"""Prepare individually frozen initial observations and independent references."""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/'scenes'))
from scene import ANALYSES, HERE, LAYOUTS, RAW, SPECS, source


ORIGINAL = ANALYSES/'same_goal_route_choice_20261121/holdout/prepare.py'
spec = importlib.util.spec_from_file_location('nso_local_original_reference_prepare', ORIGINAL)
reference_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_api)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', choices=LAYOUTS)
    args = parser.parse_args()
    name = args.layout
    folder, reachable, config, metadata, initially_unknown = reference_api.initial(name)
    reference_api.reference(name, folder, reachable, config, metadata)

    # This nominated side view is for evaluator preflight only. The runtime is
    # never told either its location or whether a screen is present.
    from env.virtual3d import camera_pose
    q = RAW[name]
    measurement = HERE/'evaluator_references'/name
    rc = source.xy_cell(*q['lateral_xy'], q['height_m'])
    mesh, labels, _ = reference_api.native_reference.source_mesh(name)
    shelf = reference_api.subset_mesh(mesh, labels == 31)
    samples, _ = reference_api.surface_samples(shelf, 10000, 202611227119)
    pose = np.asarray([camera_pose(rc,q['lateral_heading'],config,SPECS[name]['shape'][0])])
    visible, _, _ = reference_api.visible_reference(samples, reference_api.ray_scene(mesh), pose, config,
                                                    tolerance_m=.025)
    value = dict(layout=name, side_window_condition='screened' if q['screened'] else 'exposed',
                 predeclared_lateral_xy=q['lateral_xy'], nominated_lateral_pose=[*rc,q['lateral_heading']],
                 lateral_cell_reachable=bool(reachable[rc]), shelf_samples=int(len(samples)),
                 visible_shelf_samples=int(visible.sum()), visible_shelf_area_m2=float(shelf.get_surface_area()*visible.mean()),
                 actor_outcome_seen=False, source_actor_select_calls=0, initially_unknown=initially_unknown)
    reference_api.write(measurement/'optical_preflight.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    main()
