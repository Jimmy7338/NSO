#!/usr/bin/env python3
"""Run one independent cold start using the published explorer and full local pool."""

import argparse
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ANALYSES = HERE.parents[1]
PROTOCOL = STUDY / 'PROTOCOL.json'
SOURCE = ANALYSES / 'same_goal_route_choice_20261121'
HOLDOUT = SOURCE / 'holdout'
SOURCE_RUNNER = SOURCE / 'runner'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SOURCE_RUNNER))
import full_run as native
from execute import missing

SCENES = ('ordinary_chain_13', 'ordinary_cycle_14', 'ordinary_chain_15', 'ordinary_cycle_16')
ARM_MAP = {'fullG': 'G', 'fullS': 'S', 'originalS': None, 'F_strict': None}
CAP_REASON = 'remaining_candidates_after_admitted_pool_cap'


def full_local_route_adapter(components, old, mode, config, original_adapter):
    """Retain the frozen chooser, changing only its maximum accepted row count."""
    backend = components._cpu_backend
    cells = backend.scenes[0]['mapper'].belief.size
    # The published proposer emits at most one trace for each oriented waypoint
    # state. Its distinct alternate count therefore cannot reach this bound.
    exhaustive_config = dict(config, max_alternates=4 * int(cells) + 1)
    events = original_adapter(components, old, mode, exhaustive_config)
    choose = components.get_selected_option

    def exhaustive_option(scene_idx):
        previous = len(events)
        selected = choose(scene_idx)
        if selected is None:
            assert len(events) == previous
            return None
        assert len(events) == previous + 1
        event = events[-1]
        audit = event['choice_audit']
        distinct_traces = audit['proposal_diagnostics']['distinct_alternate_traces']
        skipped = audit['proposal_diagnostics']['exclusions'].get(CAP_REASON, 0)
        assert not skipped and distinct_traces < exhaustive_config['max_alternates']
        assert len(audit['candidates']) == audit['assessed_candidate_count'] == distinct_traces + 1
        assert audit['admitted_candidate_count'] == sum(row['admitted'] for row in audit['candidates'])
        event['full_pool'] = dict(enabled=True, original_small_pool_cap=config['max_alternates'],
            exhaustive_cap_upper_bound=exhaustive_config['max_alternates'],
            original_distinct_alternates=distinct_traces, unassessed_after_cap=skipped,
            full_candidate_pool_sha256=audit['candidate_pool_sha256'], full_score_sha256=audit['score_table_sha256'])
        return selected

    components.get_selected_option = exhaustive_option
    return events


def raw_sensor_hashes(packet, array_sha):
    return dict(step=packet.action_id, packet_sha256=packet.sha256(),
                depth_values_sha256=array_sha(packet.frame.depth_m),
                rgb_values_sha256=array_sha(packet.frame.color_rgb),
                scan_ranges_sha256=array_sha(packet.scan.ranges_m))


def run(layout, arm, protocol, frozen, out):
    # This imported module is private to one child interpreter process.
    # The stock runtime and frozen source files remain untouched.
    native.HERE = HERE
    native.PROTOCOL = PROTOCOL
    native.ARM_MODES = {**native.ARM_MODES, 'fullG': ('G', 'G'), 'fullS': ('G', 'S')}
    previous_setup = native.setup_runtime
    previous_adapter = native.local_route_adapter
    previous_read = native.read
    raw = []
    source_manifest = out / '.native_manifest.json'
    writer = None
    actor_receipt = read(HERE / 'cold_inputs' / f'{layout}.json')
    initial_filter = HOLDOUT / 'prepared' / layout / 'initial_state.json'

    def actor_read(path):
        if Path(path) == initial_filter:
            return dict(actor_receipt['initial'])
        return previous_read(path)

    def actor_setup(stage, name, policy, settings):
        nonlocal writer
        result = previous_setup(stage, name, policy, settings)
        old, (_, _, first, _, _, _, _), initial, _, _ = result
        world = result[1][0]
        assert policy == arm and stage == 'holdout' and name == layout
        assert world.step_count == 0 and int(world.seed) == actor_receipt['seed']
        assert initial == actor_receipt['initial']
        raw.append(raw_sensor_hashes(first, native.array_sha))
        original_packet = old.packet

        def capture_packet(*args, **kwargs):
            packet = original_packet(*args, **kwargs)
            assert packet.action_id == len(raw)
            raw.append(raw_sensor_hashes(packet, native.array_sha))
            return packet

        old.packet = capture_packet
        import runtime_binding
        writer = runtime_binding.write

        def defer_manifest(path, data):
            if Path(path) == out / 'manifest.json':
                return writer(source_manifest, data)
            return writer(path, data)

        runtime_binding.write = defer_manifest
        return result

    native.read = actor_read
    native.setup_runtime = actor_setup
    if arm in ('fullG', 'fullS'):
        native.local_route_adapter = lambda *args: full_local_route_adapter(*args, previous_adapter)
    source_protocol = read(SOURCE / 'PROTOCOL.json')
    assert source_protocol['heldout']['paid_action_budget'] == protocol['native_budget']
    assert all(source_protocol['local_route'][key] == value for key, value in protocol['frozen_online_fields'].items())
    source_protocol['diagnostic_checkpoints'] = frozen['capture_paid']
    native.run('holdout', layout, arm, source_protocol, out)
    import runtime_binding

    runtime_binding.write = write = writer
    manifest = read(source_manifest)
    source_manifest.unlink()
    rows = read(out / 'steps.json')
    assert len(rows) == manifest['paid_actions'] == len(raw) - 1
    assert raw[0]['packet_sha256'] == manifest['initial_packet_sha256']
    for row, contents in zip(rows, raw[1:], strict=True):
        assert row['step'] == contents['step'] and row['packet_sha256'] == contents['packet_sha256']
        row.update({key:value for key,value in contents.items() if key.endswith('values_sha256') or key=='scan_ranges_sha256'})
    write(out / 'steps.json', rows)
    write(out / 'initial_raw_rgbd_scan_sha256.json', raw[0])
    for paid in frozen['capture_paid']:
        name = f'paid_{paid}'
        if name not in manifest['checkpoints']:
            assert paid > manifest['paid_actions']
            manifest['checkpoints'][name] = missing(out,name,paid,manifest['paid_actions'],'native_stop_before_checkpoint')
    choices = read(out / 'local_choices.json')
    if arm in ('fullG', 'fullS'):
        assert all(choice['full_pool']['enabled'] and not choice['full_pool']['unassessed_after_cap'] for choice in choices)
    else:
        assert not choices
    manifest.update(source_commit=frozen['source_commit'], source_protocol_sha256=sha(PROTOCOL),
        runner_freeze_sha256=sha(HERE / 'holdout_freeze.json'), cold_started_at_paid_zero=True,
        published_original_goal_selector_unswitched=True, seeded_sensor_policy='original_scene_seed_and_action_clock',
        actual_scene_seed=actor_receipt['seed'], frozen_cold_input_sha256=sha(HERE / 'cold_inputs' / f'{layout}.json'),
        closed_loop_full_pool=arm in ('fullG','fullS'), all_distinct_local_candidates_assessed=arm in ('fullG','fullS'),
        files_sha256={str(path.relative_to(out)):sha(path) for path in sorted(out.rglob('*'))
                      if path.is_file() and path.name != 'manifest.json'})
    write(out / 'manifest.json',manifest)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return native.sha(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout',choices=SCENES)
    parser.add_argument('arm',choices=tuple(ARM_MAP))
    args = parser.parse_args()
    frozen = read(HERE / 'holdout_freeze.json')
    assert {'layout':args.layout,'arm':args.arm} in frozen['schedule']
    assert all(sha(Path(path))==digest for path,digest in frozen['files_sha256'].items())
    out = HERE / 'runs' / args.layout / args.arm
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(args.layout,args.arm,read(PROTOCOL),frozen,out)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps(dict(status='failed_cold_holdout',layout=args.layout,arm=args.arm,
           error=repr(error),traceback=traceback.format_exc()),indent=2)+'\n')
        raise


if __name__=='__main__':
    main()
