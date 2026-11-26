#!/usr/bin/env python3
"""Fixed original-B prefixes and all frozen outbound candidate branches.

Models only see the acquired mapper. The source world is used solely to emit
sensor packets; independent source meshes are loaded only by audit/evaluate_stage.
Branches are fixed-route acquisition diagnostics, not closed-loop policy trials.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType, ModuleType

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
B = STUDY.parent / 'dual_template_exploration_20261119'
B2 = STUDY.parent / 'dual_template_repair_20261119'
STATES = {'shelf_left': (24, 54, 109), 'shelf_right': (24, 53, 110)}
VARIANTS = ('B2', 'sampling', 'geometry', 'geometry_free')


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


def frozen_paths():
    paths = [Path(__file__), STUDY / 'protocol.json',
             B2 / 'method/occlusion_repair.py', STUDY / 'audit/evaluate_stage.py',
             B / 'audit/surface_metrics.py',
             STUDY.parent / 'semantic_scene_redesign_20261119/exploration_evaluation/surface_reference.py']
    paths += sorted((STUDY / 'method').glob('*.py'))
    for layout in STATES:
        old = B / 'layouts' / layout / 'runs/S/attempt_001'
        paths += [old / x for x in ('choices.json', 'steps.json', 'artifact_manifest.json')]
    return paths


def freeze():
    assert (STUDY / 'method/model.py').exists()
    out = HERE / 'freeze.json'
    assert not out.exists(), 'Use a new explicit replay version after a freeze.'
    write(out, dict(schema='B3_staged_replay_v1', created_unix_s=time.time(),
        files_sha256={str(p): sha(p) for p in frozen_paths()},
        states=STATES, variants=VARIANTS, every_recorded_candidate=True,
        branches_reused_across_variants=True, hardware=False))
    return out


def verify():
    frozen = json.loads((HERE / 'freeze.json').read_text())
    assert {str(p): sha(p) for p in frozen_paths()} == frozen['files_sha256']
    return frozen


def dependencies():
    b = load('B3_sealed_prefix_original_runner', B / 'run.py')
    b.dependencies()
    repair = load('B3_fixed_branch_visibility', B2 / 'method/occlusion_repair.py')
    package = ModuleType('B3_staged_model_package')
    package.__path__ = [str(STUDY / 'method')]
    sys.modules[package.__name__] = package
    model = load(package.__name__ + '.model', STUDY / 'method/model.py')
    return b, repair, model


def sealed_prefix(b, layout, paid):
    """Reproduce original actor feedback as well as sensor fusion and actions."""
    import numpy as np
    from nso.cpu_sensor_contract_v10 import digest
    old = B / 'layouts' / layout / 'runs/S/attempt_001'
    choices = json.loads((old / 'choices.json').read_text())
    steps = json.loads((old / 'steps.json').read_text())
    first_expected = b.verify(layout)
    world, tr, first, comp, runtime, effective, owner = b.bootstrap(layout, 'S')
    assert first.sha256() == first_expected['initial_packet_sha256']
    mapper = runtime.states[0]['mapper']
    backend = comp._cpu_backend
    state = backend.scenes[0]
    acquired = [(first.frame, first.scan)]
    cursor, checks, packets = [0], [], []

    def fixed(self, scene_idx):
        s = self.scenes[scene_idx]
        record = copy.deepcopy(choices[cursor[0]])
        expected = record['outputs']
        inputs = dict(semantic_version=s['semantic_version'], graph_version=s['graph_version'],
            planning_feedback_version=s['feedback']['planning_feedback_version'],
            actual_feedback_version=s['feedback']['feedback_version'],
            planning_camera_poses_sha256=digest(s['ledger'].planning_camera_poses()),
            remaining_budget=s['ledger'].remaining_budget)
        assert inputs == record['inputs']
        attempted = s['gain'].attempted_camera_mask(mapper, s['ledger'].planning_camera_poses())
        scores, rows = self._scores_v10_1(s, expected['candidates'], attempted)
        errors = {}
        for name, value in scores.items():
            np.testing.assert_allclose(value, expected['scores'][name], rtol=0, atol=1e-10)
            errors[name] = float(np.max(np.abs(np.asarray(value)-expected['scores'][name]))) if len(value) else 0.
        assert s['gain'].snapshot() == expected['gain_calibration']
        s['plans'] += 1
        assert s['plans'] == expected['plan_number']
        s['selected'] = copy.deepcopy(expected['selected'])
        s['last_selection'] = expected
        self._record(scene_idx, 'STGHP', 'select_topo_target', inputs, expected, time.perf_counter())
        checks.append(dict(paid_action=int(s['packet'].action_id), max_absolute_score_error=errors,
                           candidate_count=len(expected['candidates'])))
        cursor[0] += 1
        return None if s['selected'] is None else np.asarray(s['selected']['pose'][:2], int)

    backend.select_target = MethodType(fixed, backend)
    for expected in steps[:paid]:
        action = runtime.next_local_action(0)
        assert action == expected['action']
        frame, collision, done = world.step(action)
        packet = b.packet(world, tr, frame, layout, action, collision, done)
        assert packet.sha256() == expected['packet_sha256']
        assert list(world.position) == expected['position'] and world.heading == expected['heading']
        assert collision == expected['collision']
        owner = b.front.associate(mapper, frame, packet.frame_id)
        assert owner['owner_sha256'] == expected['owner_sha256']
        b.method.observe(mapper, frame)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=packet)
        assert mapper.frames == world.step_count + 1
        assert len(mapper.quality) == expected['measured_patch_count']
        assert len(state['assets']) == expected['measured_asset_count']
        assert state['gain'].snapshot()['posterior_mean'] == expected['gain_posterior']
        acquired.append((packet.frame, packet.scan))
        packets.append(packet.sha256())
    assert runtime.states[0]['pending'] is None
    fixed(backend, 0)  # score and install sealed selection; execute no new action
    assert checks[-1]['paid_action'] == paid
    assert mapper.frames == paid + 1
    record = choices[cursor[0]-1]
    return dict(world=world, transform=tr, mapper=mapper, state=state,
        routes=record['outputs']['candidates'], common_rows=record['outputs']['score_audit'],
        acquired=acquired, prefix_receipt=dict(paid_actions=paid, fused_frames=mapper.frames,
            original_action_packet_pose_owner_quality_asset_feedback_match=True,
            old_score_checks=checks, packet_hashes=packets, old_trace=str(old),
            choices_sha256=sha(old/'choices.json'), steps_sha256=sha(old/'steps.json'),
            fixed_candidate_sha256=digest(record['outputs']['candidates']),
            source_map_belief_sha256=digest(mapper.belief)))


def save_mesh(mapper, path):
    import open3d as o3d
    mesh = mapper.mesh()
    assert len(mesh.triangles)
    assert o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    return resource(path)


def capture_models(b, repair, model_module, context, out):
    import numpy as np
    from nso.cpu_sensor_contract_v10 import json_value, digest
    from utils.reconstruction_metrics import ray_scene
    mapper, routes = context['mapper'], context['routes']
    mesh = mapper.mesh()
    measured_ray = ray_scene(mesh) if len(mesh.triangles) else None
    captures = {}
    before = (mapper.frames, digest(mapper.belief), len(mapper.quality))
    for variant in VARIANTS:
        started = time.monotonic()
        cls = b.method.FullTemplateModel if variant == 'B2' else model_module.configure(
            geometry_fit=variant in ('geometry', 'geometry_free'),
            free_space_likelihood=variant == 'geometry_free')
        models = {name: cls(mapper, semantic=(name == 'S')) for name in ('G', 'S')}
        np.testing.assert_array_equal(models['G'].points, models['S'].points)
        np.testing.assert_array_equal(models['G'].hypothesis_ids, models['S'].hypothesis_ids)
        arrays, audit, route_rows = {}, {}, []
        model_cache = {name: {} for name in models}
        for name, model in models.items():
            arrays[name+'_points'] = model.points
            arrays[name+'_weights'] = model.weights
            arrays[name+'_hypothesis_ids'] = model.hypothesis_ids
            audit[name] = model.audit
        for route, common in zip(routes, context['common_rows']):
            values = {}
            cid = int(route['candidate_id'])
            for name, model in models.items():
                union = np.zeros(len(model.points))
                for raw_state in route['outbound_states'][1:]:
                    st = tuple(map(int, raw_state))
                    cache = model_cache[name]
                    if st not in cache:
                        cache[st], _ = repair.trace_gain(model, st[:2], st[2], measured_ray, repair_enabled=True)
                    np.maximum(union, cache[st], out=union)
                arrays[name+f'_gain_{cid}'] = union
                values[name] = float(union.sum())
            route_rows.append(dict(candidate_id=cid, template_area=values,
                common_total_proxy=common['common_total_proxy'], coverage_only_proxy=common['coverage_only_proxy'],
                scores={name: common['common_total_proxy']+value for name, value in values.items()},
                outbound_actions=len(route['outbound_actions']), return_actions_excluded=len(route['return_actions'])))
        path = out / (variant+'_predictions.npz')
        np.savez_compressed(path, **arrays)
        receipt = dict(variant=variant, model_audits=audit, routes=route_rows,
            G_S_same_points_and_hypothesis_ids=True, source_geometry_loaded_by_model=False,
            elapsed_s=time.monotonic()-started)
        write(out/(variant+'_predictions.json'), json_value(receipt))
        captures[variant] = dict(predictions=resource(path), receipt=resource(out/(variant+'_predictions.json')))
        print(json.dumps(dict(variant=variant, candidates=len(routes), seconds=receipt['elapsed_s'])), flush=True)
    assert (mapper.frames, digest(mapper.belief), len(mapper.quality)) == before
    return captures


def branch(context, b, layout, route, out, before_mesh):
    import numpy as np
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    from nso.cpu_sensor_contract_v10 import digest
    started = time.monotonic()
    original = context['world']
    original_mutable_state = (tuple(original.position), int(original.heading),
        int(original.step_count), int(original.collisions), int(original.moves))
    # Immutable source renderer/cache is shared; mutable pose/counters are isolated.
    world = copy.copy(original)
    mapper = ObservedRuntimeMapperV10(world.shape, world.config)
    for frame, scan in context['acquired']:
        mapper.update(frame, scan)
    assert mapper.frames == context['mapper'].frames
    np.testing.assert_array_equal(mapper.belief, context['mapper'].belief)
    clone_path = out/'prefix_clone_mesh.ply'
    clone = save_mesh(mapper, clone_path)
    assert clone['sha256'] == before_mesh['sha256'], 'Branch prefix native TSDF must be byte-identical'
    clone_path.unlink()  # duplicate bytes; digest is retained in execution receipt
    states, actions = route['outbound_states'], route['outbound_actions']
    assert len(states) == len(actions)+1
    assert list(states[0]) == [*world.position, world.heading]
    rows = []
    initial_coverage = b.base.evaluate(mapper.belief, world.reachable)
    reason = 'completed_fixed_outbound'
    for action, expected_pose in zip(actions, states[1:]):
        frame, collision, done = world.step(action)
        p = b.packet(world, context['transform'], frame, layout, action, collision, done)
        mapper.update(frame, p.scan)
        pose = [*world.position, world.heading]
        rows.append(dict(action=action, pose=pose, packet_sha256=p.sha256(),
            collision=bool(collision), pose_matches_expected=pose==list(expected_pose)))
        if collision or pose != list(expected_pose):
            reason = 'fixed_outbound_execution_failure'
            break
    after = save_mesh(mapper, out/'native_tsdf_mesh.ply')
    assert (tuple(original.position), int(original.heading), int(original.step_count),
            int(original.collisions), int(original.moves)) == original_mutable_state
    result = dict(status=reason, diagnostic_only=True, closed_loop_policy_trial=False,
        completed_actions=len(rows), planned_actions=len(actions), candidate_id=int(route['candidate_id']),
        initial_paid_action=original.step_count, fused_frames=mapper.frames,
        prefix_mesh_sha256=clone['sha256'], prefix_mesh_byte_identical=True,
        original_world_pose_counters_unchanged=True,
        branch_actions=rows, endpoint_pose=[*world.position, world.heading],
        expected_endpoint_pose=list(states[-1]), endpoint_reached=[*world.position, world.heading]==list(states[-1]),
        simulated_collisions=sum(x['collision'] for x in rows),
        initial_coverage=initial_coverage, final_coverage=b.base.evaluate(mapper.belief, world.reachable),
        path_length_m=(world.moves-original.moves)*world.config.resolution_m,
        branch_belief_sha256=digest(mapper.belief), elapsed_s=time.monotonic()-started,
        no_predicted_geometry_fused=True, hidden_reference_loaded_by_actor=False)
    write(out/'execution.json', result)
    return after, resource(out/'execution.json')


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
        parser.error('Specify --layout, its predeclared --state and fresh --output')
    verify()
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed = []
    try:
        b, repair, model_module = dependencies()
        context = sealed_prefix(b, args.layout, args.state)
        write(out/'prefix_execution.json', context['prefix_receipt'])
        write(out/'fixed_candidates.json', context['routes'])
        before = save_mesh(context['mapper'], out/'prefix_native_tsdf_mesh.ply')
        captures = capture_models(b, repair, model_module, context, out)
        stages = []
        for route in context['routes']:
            cid = int(route['candidate_id'])
            branch_out = out/f'candidate_{cid:03d}'; branch_out.mkdir()
            after, execution = branch(context, b, args.layout, route, branch_out, before)
            for variant in VARIANTS:
                stages.append(dict(stage_id=f'{args.layout}_{args.state}_{variant}_{cid}',
                    layout=args.layout, variant=variant, state_action=args.state, candidate_id=cid,
                    before_mesh=before, after_mesh=after, predictions=captures[variant]['predictions'],
                    prediction_groups=[dict(name=name, points_key=name+'_points', weights_key=name+f'_gain_{cid}') for name in ('G','S')],
                    execution_receipt=execution))
            completed.append(cid)
            write(out/'manifest.json', dict(layout=args.layout, mode='candidate_branch', stages=stages,
                all_candidates_complete=len(completed)==len(context['routes'])))
            print(json.dumps(dict(layout=args.layout, state=args.state, completed_candidate=cid,
                                  completed_branches=len(completed), total_candidates=len(context['routes']))), flush=True)
        verify(); b.verify(args.layout)
        write(out/'result.json', dict(status='completed_staged_fixed_prefix_all_candidate_branches',
            layout=args.layout, state=args.state, candidates=len(completed), variants=VARIANTS,
            fixed_prefix_exact=True, every_recorded_candidate_acquired=True,
            manifest=resource(out/'manifest.json'), freeze=resource(HERE/'freeze.json'),
            elapsed_s=time.monotonic()-started, hardware=False))
    except BaseException as exc:
        write(out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            completed_candidates=completed, elapsed_s=time.monotonic()-started,
            complete_experiment=False))
        raise


if __name__ == '__main__':
    main()
