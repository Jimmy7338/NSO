#!/usr/bin/env python3
"""Frozen offline B2 binding; reuse the complete previous runner and source.

Only declared historical-free template self-occlusion changes. No robot access.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'dual_template_exploration_20261119'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load('dual_template_B2_parent_runner', PARENT / 'run.py')
write, sha = old.write, old.sha


def protocol():
    return json.loads((HERE / 'protocol.json').read_text())


def prepared(layout):
    return HERE / 'layouts' / layout / 'prepared'


def verify_parent(layout):
    d = PARENT / 'layouts' / layout / 'prepared'
    frozen = json.loads((d / 'freeze.json').read_text())
    for field in ('code_sha256', 'geometry_cache_sha256'):
        for name, expected in frozen[field].items():
            assert sha(Path(name)) == expected, (field, name)
    assert old.base.source_hashes() == frozen['frozen_scientific_source_sha256']
    assert old.base.asset_hashes() == frozen['source_asset_sha256']
    assert old.base.library_versions() == frozen['library_versions']
    for name, expected in frozen['prepared_files_sha256'].items():
        assert sha(d / name) == expected, name
    return json.loads((d / 'initial_state.json').read_text())


def code_hashes():
    paths = [Path(__file__), HERE / 'protocol.json',
             HERE / 'method/occlusion_repair.py']
    return {str(p): sha(p) for p in paths}


def prepare(layout):
    assert protocol()['status'] == 'frozen_before_twelve_closed_loop_runs'
    first = verify_parent(layout)
    out = prepared(layout)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'initial_state.json', first)
    write(out / 'freeze.json', dict(schema='B2_frozen_repair_v1',
        code_sha256=code_hashes(), parent_freeze_sha256=sha(PARENT / 'layouts' / layout / 'prepared/freeze.json'),
        initial_state_sha256=sha(out / 'initial_state.json'),
        fixed_reference_receipt_sha256=sha(PARENT / 'audit/references' / layout / 'RECEIPT.json'),
        frozen_unix_s=time.time(), actors_run_before_freeze=0))
    print(json.dumps(dict(layout=layout, status='prepared_shared_parent_inputs')))


def verify(layout):
    first = verify_parent(layout)
    f = json.loads((prepared(layout) / 'freeze.json').read_text())
    assert code_hashes() == f['code_sha256'], 'B2 method/protocol/runner changed after freeze'
    assert sha(PARENT / 'layouts' / layout / 'prepared/freeze.json') == f['parent_freeze_sha256']
    assert sha(prepared(layout) / 'initial_state.json') == f['initial_state_sha256']
    assert sha(PARENT / 'audit/references' / layout / 'RECEIPT.json') == f['fixed_reference_receipt_sha256']
    return first


def aggregate(events):
    integer_fields = ('original_self_occluded_rays', 'original_self_occluded_free_first_hits',
        'candidate_intersections', 'strict_free_intersections', 'measured_surface_intersections',
        'unresolved_intersections', 'history_conflicts_vetoed', 'recovered_points')
    result = dict(gain_calls=len(events), count_unit='point-view or intersection events, not unique surface',
                  **{k: sum(e[k] for e in events) for k in integer_fields})
    result['recovered_area_view_sum_m2'] = sum(e['recovered_expected_area_m2'] for e in events)
    result['stages'] = {name: dict(point_view_events=sum(e['stages'][name]['points'] for e in events),
        expected_area_view_sum_m2=sum(e['stages'][name]['expected_area_m2'] for e in events))
        for name in ('all', 'range', 'fov', 'measured_TSDF', 'original_self_occlusion', 'final')}
    return result


def dependencies():
    old.dependencies()
    repair = load('B2_occlusion_repair_runtime', HERE / 'method/occlusion_repair.py')
    binding = repair.install(old.method.FullTemplateModel, enabled=True)
    cls = old.method.ExplorationJointCPUFourModules
    original_score = cls._scores_v10_1

    def score(self, state, routes, attempted):
        values, rows = original_score(self, state, routes, attempted)
        cache = state['mapper']._dual_template_state
        latest = cache['latest_call']
        if latest is not None:
            latest['reused_original_gain_function'] = False
            latest['gain_implementation'] = 'B2_history_consistent_template_self_occlusion'
            latest['repair_binding'] = binding
            latest['occlusion_diagnostics'] = {key: repair.diagnostics(model)
                                              for key, model in cache['cache']['models'].items()}
        for row in rows:
            row['gain_implementation'] = 'B2_history_consistent_template_self_occlusion'
        return values, rows

    cls._scores_v10_1 = score
    old.HERE = HERE
    old.protocol = protocol
    old.prepared = prepared
    old.verify = verify
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'selfcheck', 'run'))
    parser.add_argument('--layout', required=True, choices=('shelf_left', 'shelf_right'))
    parser.add_argument('--arm', choices=('C', 'J', 'G', 'S', 'S_no_fit', 'S_no_history'))
    args = parser.parse_args()
    binding = dependencies()
    if args.command == 'prepare':
        prepare(args.layout)
    elif args.command == 'verify':
        print(json.dumps(dict(initial=verify(args.layout), binding=binding)))
    elif args.command == 'selfcheck':
        old.selfcheck(args.layout)
    elif args.arm:
        old.run(args.layout, args.arm)
    else:
        parser.error('run requires --arm')


if __name__ == '__main__':
    main()
