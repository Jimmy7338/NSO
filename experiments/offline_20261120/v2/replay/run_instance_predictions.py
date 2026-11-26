#!/usr/bin/env python3
"""Reconstruct six sealed B prefixes; score instance variants on reused branches.

No new candidate is executed. Each original B prefix must reproduce its packet,
actor-feedback and native TSDF hashes. The 66 measured branch outcomes are the
already sealed v1 acquisition diagnostic, not additional independent trials.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import ModuleType

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
V1 = STUDY.parent / 'dual_template_geometry_20261120'
STATES = {'shelf_left': (24, 54, 109), 'shelf_right': (24, 53, 110)}
VARIANTS = {
    'pooling': dict(geometry_fit=False, free_space_likelihood=False, instance_pooling=True, applicability=False),
    'pooling_geometry': dict(geometry_fit=True, free_space_likelihood=False, instance_pooling=True, applicability=False),
    'pooling_applicability': dict(geometry_fit=False, free_space_likelihood=True, instance_pooling=True, applicability=True),
    'pooling_geometry_applicability': dict(geometry_fit=True, free_space_likelihood=True, instance_pooling=True, applicability=True),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def resource(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_output(layout, paid):
    return V1 / 'replay' / f'{layout}_{paid}_attempt_001'


def frozen_paths():
    paths = [Path(__file__), STUDY/'protocol.json',
             V1/'replay/run_staged_branches.py', V1/'replay/freeze.json']
    # Preserve the complete v1 code/input provenance, including its evaluator.
    prior = json.loads((V1/'replay/freeze.json').read_text())
    paths += [Path(path) for path in prior['files_sha256']]
    paths += sorted((STUDY/'method').glob('*.py'))
    for layout, states in STATES.items():
        for paid in states:
            paths += [old_output(layout, paid)/name for name in
                ('result.json', 'manifest.json', 'fixed_candidates.json', 'prefix_execution.json')]
    return sorted(set(paths))


def freeze():
    protocol = json.loads((STUDY/'protocol.json').read_text())
    assert protocol['stages'] == VARIANTS
    assert protocol['fixed_replay']['states'] == {k:list(v) for k,v in STATES.items()}
    path = HERE/'freeze.json'
    assert not path.exists(), 'A frozen version cannot be overwritten.'
    old = load('B4_frozen_prefix_runner', V1/'replay/run_staged_branches.py')
    old.verify()
    write(path, dict(schema='B4_instance_prediction_reuse_v1', created_unix_s=time.time(),
        files_sha256={str(p):sha(p) for p in frozen_paths()}, states=STATES,
        variants=VARIANTS, reused_branches=66, reused_candidate_actions=1569,
        new_candidate_actions=0, hardware=False))
    return path


def verify():
    frozen = json.loads((HERE/'freeze.json').read_text())
    assert {str(p):sha(p) for p in frozen_paths()} == frozen['files_sha256']
    return frozen


def dependencies():
    old = load('B4_frozen_prefix_runner', V1/'replay/run_staged_branches.py')
    old.verify()
    b, repair, _ = old.dependencies()
    package = ModuleType('B4_instance_model_package')
    package.__path__ = [str(STUDY/'method')]
    sys.modules[package.__name__] = package
    model = load(package.__name__+'.model', STUDY/'method/model.py')
    return old, b, repair, model


def reused_branches(layout, paid):
    """Verify each source receipt and native mesh once, without actor access."""
    root = old_output(layout, paid)
    result = json.loads((root/'result.json').read_text())
    assert result['status'] == 'completed_staged_fixed_prefix_all_candidate_branches'
    assert result['layout'] == layout and result['state'] == paid
    assert resource(root/'manifest.json') == result['manifest']
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['all_candidates_complete']
    rows = {}
    checked = set()
    for row in manifest['stages']:
        assert row['layout'] == layout and row['state_action'] == paid
        if row['variant'] != 'B2':
            continue
        cid = row['candidate_id']
        assert cid not in rows
        for key in ('before_mesh', 'after_mesh', 'execution_receipt'):
            ref = row[key]
            if ref['path'] not in checked:
                assert resource(ref['path']) == ref
                checked.add(ref['path'])
        execution = json.loads(Path(row['execution_receipt']['path']).read_text())
        assert execution['candidate_id'] == cid and execution['initial_paid_action'] == paid
        assert execution['prefix_mesh_sha256'] == row['before_mesh']['sha256']
        assert execution['prefix_mesh_byte_identical']
        rows[cid] = row
    assert len(rows) == result['candidates']
    return rows


def capture_models(repair, model_module, context, out):
    import numpy as np
    from nso.cpu_sensor_contract_v10 import json_value, digest
    from utils.reconstruction_metrics import ray_scene
    mapper, routes = context['mapper'], context['routes']
    mesh = mapper.mesh()
    measured_ray = ray_scene(mesh) if len(mesh.triangles) else None
    before = (mapper.frames, digest(mapper.belief), len(mapper.quality))
    captures = {}
    for variant, flags in VARIANTS.items():
        started = time.monotonic()
        cls = model_module.configure(**flags)
        models = {name:cls(mapper, semantic=name == 'S') for name in ('G','S')}
        np.testing.assert_array_equal(models['G'].points, models['S'].points)
        np.testing.assert_array_equal(models['G'].hypothesis_ids, models['S'].hypothesis_ids)
        arrays, audit, route_rows = {}, {}, []
        caches = {name:{} for name in models}
        for name, model in models.items():
            arrays[name+'_points'] = model.points
            arrays[name+'_weights'] = model.weights
            arrays[name+'_hypothesis_ids'] = model.hypothesis_ids
            audit[name] = model.audit
        for route, common in zip(routes, context['common_rows']):
            cid, values = int(route['candidate_id']), {}
            for name, model in models.items():
                union = np.zeros(len(model.points))
                for raw in route['outbound_states'][1:]:
                    state = tuple(map(int, raw))
                    cache = caches[name]
                    if state not in cache:
                        cache[state], _ = repair.trace_gain(model, state[:2], state[2], measured_ray, repair_enabled=True)
                    np.maximum(union, cache[state], out=union)
                arrays[name+f'_gain_{cid}'] = union
                values[name] = float(union.sum())
            route_rows.append(dict(candidate_id=cid, template_area=values,
                common_total_proxy=common['common_total_proxy'], coverage_only_proxy=common['coverage_only_proxy'],
                scores={name:common['common_total_proxy']+value for name,value in values.items()},
                outbound_actions=len(route['outbound_actions']), return_actions_excluded=len(route['return_actions'])))
        path = out/(variant+'_predictions.npz')
        np.savez_compressed(path, **arrays)
        receipt = dict(variant=variant, flags=flags, model_audits=audit, routes=route_rows,
            G_S_same_points_and_hypothesis_ids=True, source_geometry_loaded_by_model=False,
            candidate_acquisition_reused=True, elapsed_s=time.monotonic()-started)
        write(out/(variant+'_predictions.json'), json_value(receipt))
        captures[variant] = dict(predictions=resource(path), receipt=resource(out/(variant+'_predictions.json')))
        print(json.dumps(dict(variant=variant, candidates=len(routes), seconds=receipt['elapsed_s'])), flush=True)
    assert (mapper.frames, digest(mapper.belief), len(mapper.quality)) == before
    return captures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--layout', choices=tuple(STATES))
    parser.add_argument('--state', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.freeze:
        print(freeze()); return
    if args.layout is None or args.state not in STATES[args.layout] or args.output is None:
        parser.error('Specify a declared --layout --state and a fresh --output')
    verify()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        old, b, repair, model_module = dependencies()
        context = old.sealed_prefix(b, args.layout, args.state)
        previous = old_output(args.layout, args.state)
        assert context['prefix_receipt'] == json.loads((previous/'prefix_execution.json').read_text())
        assert context['routes'] == json.loads((previous/'fixed_candidates.json').read_text())
        # Only this locally acquired mesh enters model construction. Reused
        # after-mesh resources never enter context or any model call.
        rebuilt_path = out/'prefix_reconstructed_check.ply'
        rebuilt = old.save_mesh(context['mapper'], rebuilt_path)
        original_prefix = resource(previous/'prefix_native_tsdf_mesh.ply')
        assert rebuilt['sha256'] == original_prefix['sha256']
        rebuilt_path.unlink()
        write(out/'prefix_execution.json', dict(**context['prefix_receipt'],
            native_mesh_rebuilt_sha256=rebuilt['sha256'], previous_native_mesh=original_prefix,
            native_prefix_byte_identical_to_v1=True, newly_executed_candidate_actions=0))
        write(out/'fixed_candidates.json', context['routes'])
        captures = capture_models(repair, model_module, context, out)
        reused = reused_branches(args.layout, args.state)
        assert set(reused) == {int(route['candidate_id']) for route in context['routes']}
        stages = []
        for route in context['routes']:
            cid = int(route['candidate_id'])
            source = reused[cid]
            for variant in VARIANTS:
                stages.append(dict(stage_id=f'{args.layout}_{args.state}_{variant}_{cid}',
                    layout=args.layout, variant=variant, state_action=args.state, candidate_id=cid,
                    before_mesh=source['before_mesh'], after_mesh=source['after_mesh'],
                    predictions=captures[variant]['predictions'],
                    prediction_groups=[dict(name=name, points_key=name+'_points', weights_key=name+f'_gain_{cid}') for name in ('G','S')],
                    execution_receipt=source['execution_receipt'], execution_reused=True))
        write(out/'manifest.json', dict(layout=args.layout, mode='candidate_branch', stages=stages,
            all_candidates_complete=True, candidate_acquisition_reused=True,
            acquisition_manifest=resource(previous/'manifest.json'), newly_executed_candidate_actions=0))
        verify(); old.verify(); b.verify(args.layout)
        write(out/'result.json', dict(status='completed_instance_fixed_prefix_reused_branches',
            layout=args.layout, state=args.state, candidates=len(reused), variants=list(VARIANTS),
            fixed_prefix_exact=True, native_prefix_byte_identical_to_v1=True,
            every_recorded_candidate_acquired=True, candidate_acquisition_reused=True,
            newly_executed_candidate_actions=0, prefix_reconstruction_actions=args.state,
            manifest=resource(out/'manifest.json'), freeze=resource(HERE/'freeze.json'),
            elapsed_s=time.monotonic()-started, hardware=False))
    except BaseException as exc:
        write(out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            elapsed_s=time.monotonic()-started, complete_experiment=False,
            newly_executed_candidate_actions=0))
        raise


if __name__ == '__main__':
    main()
