#!/usr/bin/env python3
"""Explicit observed-object / partial-shape binding for conditional offline NSO runs.

No changes to the frozen parent source, candidate/scoring common terms,
sensor acquisition, mapping, execution, budget or return semantics.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parent
B = ROOT.parent / 'dual_template_exploration_20261119'
B2 = ROOT.parent / 'dual_template_repair_20261119'
STAGES = {'object_support': dict(object_support=True, full_observed_support=False, geometry_fit=False),
          'object_full_support': dict(object_support=True, full_observed_support=True, geometry_fit=False),
          'object_fit': dict(object_support=True, full_observed_support=True, geometry_fit=True)}
HERE = None


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load('N20_objectfit_parent_runner', B / 'run.py')
write, sha = old.write, old.sha


def protocol():
    return json.loads((HERE / 'protocol.json').read_text())


def prepared(layout):
    return HERE / 'layouts' / layout / 'prepared'


def verify_parent(layout):
    d = B / 'layouts' / layout / 'prepared'
    f = json.loads((d / 'freeze.json').read_text())
    for field in ('code_sha256', 'geometry_cache_sha256'):
        for path, expected in f[field].items():
            assert sha(Path(path)) == expected, (field, path)
    assert old.base.source_hashes() == f['frozen_scientific_source_sha256']
    assert old.base.asset_hashes() == f['source_asset_sha256']
    assert old.base.library_versions() == f['library_versions']
    for path, expected in f['prepared_files_sha256'].items():
        assert sha(d / path) == expected, path
    return json.loads((d / 'initial_state.json').read_text())


def code_hashes():
    files = [Path(__file__), HERE / 'protocol.json', B2 / 'method/occlusion_repair.py']
    files += sorted((ROOT / 'method').glob('*.py'))
    for name in ('dual_template_geometry_20261120', 'dual_template_instance_20261120'):
        files += sorted((ROOT.parent / name / 'method').glob('*.py'))
    return {str(p): sha(p) for p in files}


def prepare(layout):
    assert protocol()['status'] == 'frozen_before_policy_runs'
    first = verify_parent(layout)
    out = prepared(layout)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'initial_state.json', first)
    write(out / 'freeze.json', dict(schema='N20_frozen_objectfit_stage_v1',
        code_sha256=code_hashes(), parent_freeze_sha256=sha(B / 'layouts' / layout / 'prepared/freeze.json'),
        initial_state_sha256=sha(out / 'initial_state.json'),
        reference_receipt_sha256=sha(B / 'audit/references' / layout / 'RECEIPT.json'),
        frozen_unix_s=time.time(), policy_runs_before_freeze=0))
    print(json.dumps(dict(status='prepared', stage=protocol()['selected_stage'], layout=layout)), flush=True)


def verify(layout):
    first = verify_parent(layout)
    f = json.loads((prepared(layout) / 'freeze.json').read_text())
    assert code_hashes() == f['code_sha256'], 'N20 frozen stage code/protocol changed'
    assert sha(B / 'layouts' / layout / 'prepared/freeze.json') == f['parent_freeze_sha256']
    assert sha(prepared(layout) / 'initial_state.json') == f['initial_state_sha256']
    assert sha(B / 'audit/references' / layout / 'RECEIPT.json') == f['reference_receipt_sha256']
    return first


def dependencies(stage):
    import numpy as np
    old.dependencies()
    module = load('N20_object_support_partial_fit_model', ROOT / 'method/model.py')
    model_class = module.FullTemplateModel
    from utils.reconstruction_metrics import ray_scene
    flags = STAGES[stage]
    arms = protocol()['arms']
    old.method.ARMS = tuple(arms)
    old.method.MODE = {k: v['score_mode'] for k, v in arms.items()}
    old.method.FullTemplateModel = model_class

    def cache(mapper, arm):
        s = mapper._dual_template_state
        assert mapper.frames == len(s['frames'])
        key = (int(mapper.frames), arm, stage)
        if s['cache'] is not None and s['cache']['key'] == key:
            return s['cache']
        common = dict(flags)
        G = model_class(mapper, semantic=False, **common)
        variant = dict(common)
        if arm == 'S_no_geometry':
            variant['geometry_fit'] = False
        S = model_class(mapper, semantic=True,
                        history_rejection=True,
                        measured_surface_consumption=arm not in ('S_no_history', 'S_no_measured_consumption'), **variant)
        if arm != 'S_no_geometry':
            np.testing.assert_array_equal(G.points, S.points)
            np.testing.assert_array_equal(G.hypothesis_ids, S.hypothesis_ids)
        mesh = mapper.mesh()
        ray = ray_scene(mesh) if len(mesh.triangles) else None
        value = dict(key=key, models={'G': G, 'S': S}, ray=ray,
            gains={'G': {}, 'S': {}}, gain_calls={'G': 0, 'S': 0},
            state_sha256=hashlib.sha256((''.join(x['sha256'] for x in s['frames']) + arm + stage).encode()).hexdigest())
        s['cache'] = value
        return value

    old.method._cache = cache
    cls = old.method.ExplorationJointCPUFourModules
    score_original = cls._scores_v10_1

    def score(self, state, routes, attempted):
        values, rows = score_original(self, state, routes, attempted)
        saved = state['mapper']._dual_template_state
        saved['latest_call'].update(reused_original_gain_function=False,
            gain_implementation='B2_visibility_with_N20_object_support_partial_fit',
            selected_stage=stage, staged_model_source_sha256=sha(ROOT / 'method/model.py'),
            stage_arms=arms, shared_observed_instance_support=True,
            G_definition='neutral class prior with same acquired instance support as S')
        for row in rows:
            row['selected_stage'] = stage
        return values, rows

    cls._scores_v10_1 = score
    old.HERE = HERE
    old.protocol = protocol
    old.prepared = prepared
    old.verify = verify


def selfcheck(layout):
    from nso.cpu_sensor_contract_v10 import digest
    first = verify(layout)
    shared = old.world_and_packet(layout)
    rows = {}
    for arm in protocol()['arms']:
        world, tr, p, comp, runtime, effective, owner = old.bootstrap(layout, arm, shared)
        assert p.sha256() == first['initial_packet_sha256']
        comp._cpu_backend.select_target(0)
        state = comp._cpu_backend.scenes[0]
        plan = state['last_selection']
        rows[arm] = dict(candidate_sha256=digest(plan['candidates']),
            belief_sha256=digest(state['mapper'].belief), first_plan=plan,
            paid_actions=world.step_count)
        assert world.step_count == 0
    assert len({r['candidate_sha256'] for r in rows.values()}) == 1
    assert len({r['belief_sha256'] for r in rows.values()}) == 1
    verify(layout)
    write(HERE / 'layouts' / layout / 'selfcheck.json',
          dict(status='passed', shared_initial_sensor_geometry_candidates=True, arms=rows))
    print(json.dumps(dict(status='passed', layout=layout, arms=list(rows))), flush=True)


def main():
    global HERE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'verify', 'selfcheck', 'run'))
    p.add_argument('--stage', required=True, choices=tuple(STAGES))
    p.add_argument('--layout', required=True, choices=('shelf_left', 'shelf_right'))
    p.add_argument('--arm')
    args = p.parse_args()
    HERE = ROOT / 'rollouts' / args.stage
    dependencies(args.stage)
    if args.command == 'prepare':
        prepare(args.layout)
    elif args.command == 'verify':
        print(json.dumps(verify(args.layout)))
    elif args.command == 'selfcheck':
        selfcheck(args.layout)
    elif args.arm in protocol()['arms']:
        old.run(args.layout, args.arm)
    else:
        p.error('run requires a declared arm')


if __name__ == '__main__':
    main()
