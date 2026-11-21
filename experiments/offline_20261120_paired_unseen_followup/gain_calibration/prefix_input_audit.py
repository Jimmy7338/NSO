"""Check the observed prefix labels and all saved prediction/endpoint bindings."""
import argparse
import json
import numpy as np

from experiment import HERE, LAYOUTS, binding, install, verify
from seal_evidence import read, verify_resource


def check(split):
    verify()
    install()
    inputs = []
    for layout, entry in LAYOUTS.items():
        if entry['split'] != split:
            continue
        run = HERE/'runs'/split/layout/'run_receipt.json'
        if not run.exists():
            inputs.append({'layout': layout, 'status': 'missing_or_failed'})
            continue
        receipt = read(run)
        sealed = read(verify_resource(receipt['predictions']))
        world, transform, first = binding.world_and_packet(layout)
        packets = [first]
        for _ in range(entry['prefix']):
            frame, collision, done = world.step('forward')
            assert not collision and not done
            packets.append(binding.old.packet(world, transform, frame, layout, 'forward', collision, done))
        assert [p.sha256() for p in packets] == sealed['initial_history_packet_sha256']
        category = 2 if entry['family'] == 'open' else 3
        counts = [int(np.count_nonzero(p.frame.semantic == category)) for p in packets]
        assert all(not np.any(p.frame.semantic[p.frame.depth_m <= 0]) for p in packets)
        opposite = 5 - category
        assert all(not np.any(p.frame.semantic == opposite) for p in packets)
        predictions = sealed['predictions']
        inputs.append(dict(layout=layout, status='acquired_prefix_packets_match', expected_category=category,
                           visible_category_pixels_by_frame=counts, with_any_category=sum(c > 0 for c in counts),
                           fitted_tracks=sealed['modeled_support_tracks'],
                           positive_A=sum(p['raw_acquisition_m2'] > 0. for p in predictions),
                           positive_V=sum(p['raw_visibility_m2'] > 0. for p in predictions),
                           forecast_route_count=len(predictions),
                           route_collisions=[read(verify_resource(b['execution']))['simulated_collisions']
                                             for b in receipt['branches']],
                           completed_endpoints=sum(read(verify_resource(b['execution']))['endpoint_reached']
                                                   for b in receipt['branches'])))
    with (HERE/f'{split}_prefix_audit.json').open('x') as out:
        json.dump(dict(split=split, layouts=inputs, actor_policy_runs=0, future_fitting_updates=0,
                       scope='Observed categories and fitted-track activity; does not isolate category benefit.'),
                  out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=('calibration','heldout'))
    check(parser.parse_args().split)
