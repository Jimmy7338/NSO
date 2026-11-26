#!/usr/bin/env python3
"""One west-screened G simulator execution with isolated endpoint quantization."""

import argparse
import faulthandler
import sys

from precision_patch import bind
from west_replay import HERE, LAYOUT, RUN, native, read, sha, write
from run import FREEZE, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dual', action='store_true')
    args = parser.parse_args()
    if args.dual:
        shadow_root = HERE / 'west_screened_dual_endpoint_shadow'
        passive = [shadow_root / f'{mode}.json' for mode in ('archived','rgb_only')]
        good = [read(path) for path in passive]
        for verified in good:
            assert verified['baseline_all_packets_and_beliefs_exact']
            assert verified['newly_admitted_physical_colliding_centers_max'] == 0
            assert verified['old_occupied_real_centers_not_occupied_after_dual_max'] == 0
            assert verified['history'][-1]['dual_return_available']
            assert verified['history'][-1]['dual_return_source_swept_valid_all']
        assert good[0]['binding'] == good[1]['binding']
        binding_reference = good[0]['binding']
        folder = HERE / 'west_screened_G_dual_endpoints_native'
    else:
        passive = [HERE / 'west_screened_precision_shadow' / 'receipt.json']
        verified = read(passive[0])
        assert verified['baseline_all_62_observation_belief_states_exact']
        assert verified['original_invalid_to_new_true_physical_footprint_max'] == 0
        assert verified['original_true_occupied_centers_no_longer_occupied_max'] == 0
        assert verified['history'][61]['normalized_guard_return']['available']
        binding_reference = verified['binding']
        folder = HERE / 'west_screened_G_precision_native'
    folder.mkdir(exist_ok=False)
    out = folder / 'native'
    out.mkdir()
    sys.path.insert(0, str(native.B11))
    import runtime_binding
    ordinary_install = runtime_binding.install
    binding = {}

    def install():
        context = ordinary_install()
        binding.update(bind(include_laser=args.dual))
        assert binding == binding_reference
        from nso.mapping3d import SensorMapper
        init_original = SensorMapper.__init__

        def init_marked(mapper, *args, **kwargs):
            init_original(mapper, *args, **kwargs)
            mapper._simulator_camera_boundary_shadow = True
            mapper._simulator_lidar_boundary_shadow = args.dual

        SensorMapper.__init__ = init_marked
        return context

    with (folder / 'faulthandler.log').open('x') as fault_log:
        faulthandler.enable(file=fault_log)
        runtime_binding.install = install
        execute(LAYOUT, 'G', out)
        faulthandler.disable()

    old, new = (read(p / 'steps.json') for p in (RUN,out))
    assert read(RUN / 'initial_raw_rgbd_scan_sha256.json') == read(out / 'initial_raw_rgbd_scan_sha256.json')
    motion = ('action', 'position', 'heading', 'packet_sha256')
    shared = [all(a[k]==b[k] for k in motion) for a,b in zip(old,new)]
    original_manifest, new_manifest = (read(p / 'manifest.json') for p in (RUN,out))
    original_summary, new_summary = (read(p / 'runtime_summary.json') for p in (RUN,out))
    write(folder / 'receipt.json', {
        'scope': 'Single simulator debugging replay with an independent autonomous actor from the same cold observations. Source geometry supplied only to the renderer.',
        'binding': binding,
        'archived_same_layout_arm': 'west_screened/G',
        'historical_manifest_sha256': sha(RUN / 'manifest.json'),
        'normalized_manifest_sha256': sha(out / 'manifest.json'),
        'shared_frozen_config_sha256': sha(FREEZE),
        'passive_evidence_sha256': {str(path.relative_to(HERE)):sha(path) for path in passive},
        'same_initial_observations': True,
        'first_executed_action_or_packet_difference_paid': next((i for i,equal in enumerate(shared,1) if not equal),None),
        'historical': {name: original_manifest[name] for name in ('paid_actions','returned','collisions','route_event')}
                      | {'termination': original_summary['termination']},
        'normalized': {name:new_manifest[name] for name in ('paid_actions','returned','collisions','route_event')}
                      | {'termination':new_summary['termination']},
    })


if __name__ == '__main__':
    main()
