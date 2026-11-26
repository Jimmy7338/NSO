#!/usr/bin/env python3
"""Explicit paired offline dual-template runs. No ROS or robot access."""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'semantic_scene_redesign_20261119'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


front = load('dual_template_shared_frontend', PREVIOUS / 'instance_frontend_study/run.py')
base = front.r
write, sha = base.write, base.sha
_installed = False


def protocol():
    return json.loads((HERE / 'protocol.json').read_text())


def dependencies():
    global _installed, method
    if _installed:
        return
    front.install()
    sys.path.insert(0, str(HERE / 'scene'))
    sys.path.insert(0, str(HERE))
    method = importlib.import_module('method.adapter')
    assert Path(method.__file__).resolve() == HERE / 'method/adapter.py'
    method.install()
    _installed = True


def packet(world, transform, frame, layout, action=None, collision=False, done=False):
    import numpy as np
    from nso.cpu_sensor_contract_v10 import SensorPacket
    assert np.isin(frame.semantic, (0, 2, 3)).all()
    assert not np.any(frame.semantic[frame.depth_m == 0])
    return SensorPacket(protocol()['layouts'][layout]['scene_id'], 'single_observation_cold_start',
        f'frame-{world.step_count}', int(world.step_count), frame, world.scan(),
        tuple(map(int, world.position)), int(world.heading),
        'current_first_hit_RGBD_scan_visible_ideal_structure_labels',
        'exact_discrete_simulator_odometry', action, bool(collision), bool(done)
    ).validate(transform, world.config)


def world_and_packet(layout):
    from controlled_world import ControlledConfig, ControlledFunctionalWorld
    from nso.cpu_sensor_contract_v10 import GridTransform
    cfg = protocol()['environment']
    world = ControlledFunctionalWorld(ControlledConfig(**cfg['config']),
                                     seed=cfg['noise_seed'], layout=layout)
    assert list(world.position) == cfg['initial_position']
    assert world.heading == cfg['initial_heading']
    transform = GridTransform(tuple(world.shape), world.config.resolution_m)
    return world, transform, packet(world, transform, world.sense(), layout)


def code_hashes():
    paths = [Path(__file__), HERE / 'protocol.json', PREVIOUS / 'instance_frontend_study/run.py',
             PREVIOUS / 'exploration_study/run.py', base.OLD_RUNNER]
    paths += list((HERE / 'method').glob('*.py')) + list((HERE / 'scene').glob('*.py'))
    paths += [PREVIOUS / 'scene_adapter/replicacad_world.py',
              PREVIOUS / 'assets/textured_raycast.py',
              PREVIOUS / 'assets/replicacad/assemble_apt0.py',
              PREVIOUS / 'assets/replicacad/download_receipt.json',
              HERE / 'scene/PROTOCOL.json']
    paths += front.MODULES
    return {str(p): sha(p) for p in sorted(set(paths))}


def prepared(layout):
    return HERE / 'layouts' / layout / 'prepared'


def prepare(layout):
    import numpy as np
    from nso.observed_runtime_mapper_v10 import ObservedRuntimeMapperV10
    out = prepared(layout)
    out.mkdir(parents=True, exist_ok=False)
    try:
        world, tr, p = world_and_packet(layout)
        mapper = ObservedRuntimeMapperV10(world.shape, world.config)
        assert np.all(mapper.belief == -1)
        mapper.update(p.frame, p.scan)
        p.frame.save(out / 'initial_frame.npz')
        p.scan.save(out / 'initial_scan.npz')
        np.savez_compressed(out / 'initial_sensor.npz', depth_m=p.frame.depth_m,
            rgb=p.frame.color_rgb, visible_labels=p.frame.semantic,
            intrinsic=p.frame.intrinsic, world_from_camera=p.frame.world_from_camera,
            ranges_m=p.scan.ranges_m, world_from_laser=p.scan.world_from_laser)
        np.savez_compressed(out / 'evaluator_truth.npz', **world.evaluator_truth())
        first = dict(layout=layout, scene_id=protocol()['layouts'][layout]['scene_id'],
            scene_role='paired_controlled_development_layout', shape=world.shape,
            anchor=[*world.position, world.heading], actual_config=dataclasses.asdict(world.config),
            initial_packet_sha256=p.sha256(), initial_nonsemantic_packet_sha256=base.nonsemantic_packet_sha(p),
            initial_owner_sha256=hashlib.sha256(front.owner(p.frame).tobytes()).hexdigest(),
            evaluation=base.evaluate(mapper.belief, world.reachable),
            current_footprint_conflict=bool(mapper.current_footprint_conflict),
            partial_ideal_semantics=True, natural_detector=False,
            geometry_receipt=world.geometry_receipt, simulator_actions=0)
        assert not first['current_footprint_conflict']
        assert first['evaluation']['coverage_fraction'] <= .5
        write(out / 'initial_state.json', first)
        cache_base = HERE / 'scene/precomputed_assets' / world.geometry_receipt['topology_cache_key']
        cache_hashes = {str(cache_base.with_suffix(ext)): sha(cache_base.with_suffix(ext))
                        for ext in ('.npz', '.json')}
        write(out / 'freeze.json', dict(code_sha256=code_hashes(),
            frozen_scientific_source_sha256=base.source_hashes(),
            source_asset_sha256=base.asset_hashes(), library_versions=base.library_versions(),
            geometry_cache_sha256=cache_hashes,
            prepared_files_sha256={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
            created_unix_s=time.time(), no_actor_policy_executed=True))
        print(json.dumps(dict(layout=layout, status='prepared', initial=first['evaluation'])), flush=True)
    except BaseException as exc:
        write(out / 'prepare_failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise


def verify(layout):
    out = prepared(layout)
    frozen = json.loads((out / 'freeze.json').read_text())
    assert code_hashes() == frozen['code_sha256'], 'runner/adapter/protocol changed'
    assert base.source_hashes() == frozen['frozen_scientific_source_sha256'], 'original source changed'
    assert base.asset_hashes() == frozen['source_asset_sha256'], 'source assets changed'
    assert base.library_versions() == frozen['library_versions']
    for path, digest in frozen['geometry_cache_sha256'].items():
        assert sha(path) == digest, 'geometry cache changed'
    for name, digest in frozen['prepared_files_sha256'].items():
        assert sha(out / name) == digest, name
    return json.loads((out / 'initial_state.json').read_text())


def bootstrap(layout, arm, shared=None):
    import numpy as np
    from nso.components import NSO_Components
    from nso_bridge.exploration_joint_policy import ExplorationJointCPUFourModules, ExplorationJointRuntimeIntegration
    cfg = protocol()
    world, tr, p = shared if shared is not None else world_and_packet(layout)
    assert world.step_count == 0
    effective = dict(cfg['policy']['args'])
    effective.update(cpu_score_mode=cfg['arms'][arm]['score_mode'], dual_template_arm=arm,
                     run_id=f'dual-template-{layout}-{arm}')
    args = SimpleNamespace(**effective)
    comp = NSO_Components(args)
    comp.initialize('cpu', 1, *world.shape, *world.shape)
    comp._cpu_backend = ExplorationJointCPUFourModules(args, 1, world.shape)
    comp.capabilities = dict(comp._cpu_backend.capabilities)
    runtime = ExplorationJointRuntimeIntegration(comp, 1, world.shape)
    runtime.start_sensor_episode(0, config=world.config, transform=tr, packets=[p],
        total_budget=cfg['policy']['paid_action_budget'],
        return_anchor=(*p.position, p.heading), paid_prefix_actions=0)
    mapper = runtime.states[0]['mapper']
    state = comp._cpu_backend.scenes[0]
    before = (mapper.frames, state['ledger'].remaining_budget,
              copy.deepcopy(state['gain'].snapshot()), mapper.belief.copy())
    mapper.instance_support_index = front.isp.InstanceSupportIndex()
    mapper.asset_grouping_profile = 'measured_instance_support_v1'
    first_owner = front.associate(mapper, p.frame, p.frame_id)
    method.initialize(mapper, p.frame)
    state['planning_direction_support'] = None
    comp._cpu_backend.update_semantic(0)
    assert (mapper.frames, state['ledger'].remaining_budget) == before[:2]
    assert state['gain'].snapshot() == before[2] and np.array_equal(mapper.belief, before[3])
    assert state['ledger'].feedback_enabled and state['gain'].feedback_enabled
    assert not state['gain'].consume_predicted_attempts
    audit = dict(arm=arm, layout=layout, effective_args=effective,
        initial_packet_sha256=p.sha256(), policy_variant=cfg['policy']['version'],
        original_method_equivalence=False, hidden_reference_to_actor=False,
        partial_ideal_semantics=True, common_instance_frontend=True,
        trained_reachability_network_active=False, natural_detector=False)
    return world, tr, p, comp, runtime, audit, first_owner


def selfcheck(layout):
    import numpy as np
    from nso.cpu_sensor_contract_v10 import digest
    first = verify(layout)
    shared = world_and_packet(layout)
    rows, signatures = {}, []
    for arm in protocol()['arms']:
        world, tr, p, comp, runtime, effective, owner = bootstrap(layout, arm, shared)
        assert p.sha256() == first['initial_packet_sha256']
        comp._cpu_backend.select_target(0)
        state = comp._cpu_backend.scenes[0]
        plan = state['last_selection']
        signature = (digest(plan['candidates']), digest(state['mapper'].belief))
        signatures.append(signature)
        assert world.step_count == 0
        rows[arm] = dict(effective=effective, owner_receipt=owner,
            candidate_sha256=signature[0], belief_sha256=signature[1], first_plan=plan,
            paid_actions=0)
    assert len(set(signatures)) == 1, 'initial candidates/geometry must be shared'
    common = rows['G']['first_plan']['score_audit']
    np.testing.assert_allclose(rows['J']['first_plan']['scores']['N'],
                               [r['common_total_proxy'] for r in common])
    np.testing.assert_allclose(rows['C']['first_plan']['scores']['C'],
                               [r['coverage_only_proxy'] for r in common])
    verify(layout)
    write(HERE / 'layouts' / layout / 'selfcheck.json',
          dict(status='passed', shared_first_state=True, simulator_actions=0, arms=rows))
    print(json.dumps(dict(layout=layout, status='passed', arms=list(rows),
        initial_candidates=len(rows['G']['first_plan']['candidates']))), flush=True)


def new_attempt(layout, arm):
    parent = HERE / 'layouts' / layout / 'runs' / arm
    parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, 10000):
        out = parent / f'attempt_{i:03d}'
        try:
            out.mkdir()
            return out
        except FileExistsError:
            continue
    raise RuntimeError('attempt directory exhausted')


def run(layout, arm):
    import numpy as np
    first = verify(layout)
    cfg = protocol()
    out = new_attempt(layout, arm)
    start = time.monotonic()
    steps, owners, timings = [], [], []
    mapper = world = runtime = None
    write(out / 'attempt.json', dict(status='started', arm=arm, layout=layout,
        started_unix_s=time.time(), prepared_freeze_sha256=sha(prepared(layout) / 'freeze.json'),
        protocol_sha256=sha(HERE / 'protocol.json'), runner_sha256=sha(__file__),
        paid_action_budget=cfg['policy']['paid_action_budget']))
    def interrupted(signum, frame):
        raise InterruptedError(f'external signal {signum}, not algorithmic completion')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        world, tr, p, comp, runtime, effective, owner = bootstrap(layout, arm)
        assert p.sha256() == first['initial_packet_sha256']
        owners.append(owner)
        mapper = runtime.states[0]['mapper']
        state = comp._cpu_backend.scenes[0]
        initial_eval = base.evaluate(mapper.belief, world.reachable)
        write(out / 'effective_configuration.json', effective)
        write(out / 'initial_evaluation.json', initial_eval)
        streak = 0
        choices = []
        while not runtime.states[0]['closed']:
            t = time.monotonic()
            action = runtime.next_local_action(0)
            decision_s = time.monotonic() - t
            timings.append(decision_s)
            choices = [dict(inputs=x['inputs'], outputs=x['outputs'])
                       for x in comp._cpu_backend.calls if x['method'] == 'select_topo_target']
            write(out / 'choices.json', choices)
            write(out / 'runtime_audit.json', runtime.audit)
            if action is None:
                break
            active = runtime.states[0]['option']
            phase = runtime.states[0]['phase']
            target = None if active is None else list(active['pose'])
            option = None if active is None else active['option_id']
            known_before = (mapper.belief != -1).copy()
            t = time.monotonic()
            raw, collision, done = world.step(action)
            p = packet(world, tr, raw, layout, action, collision, done)
            sensor_s = time.monotonic() - t
            t = time.monotonic()
            owner = front.associate(mapper, raw, p.frame_id)
            owners.append(owner)
            method.observe(mapper, p.frame)
            runtime.observe(0, world.step_count, None, None, None, sensor_packet=p)
            observe_s = time.monotonic() - t
            measured = base.evaluate(mapper.belief, world.reachable)
            newly_known = int(np.count_nonzero((mapper.belief != -1) & ~known_before & world.reachable))
            streak = streak + 1 if newly_known == 0 else 0
            steps.append(dict(step=world.step_count, action=action, position=list(world.position),
                heading=world.heading, **measured, new_reference_known_cells=newly_known,
                zero_coverage_gain_streak=streak, path_length_m=world.moves * world.config.resolution_m,
                collision=bool(collision), done=bool(done), packet_sha256=p.sha256(),
                nonsemantic_packet_sha256=base.nonsemantic_packet_sha(p),
                owner_sha256=owner['owner_sha256'], active_option_id=option,
                selected_target_pose=target, phase_before_action=phase,
                matched_target_pose_after_step=target == [*world.position, world.heading] if target else None,
                decision_wall_s=decision_s, sensor_wall_s=sensor_s, observe_wall_s=observe_s,
                partial_semantic_pixels={str(c): int((p.frame.semantic == c).sum()) for c in (0, 2, 3)},
                measured_patch_count=len(mapper.quality), measured_asset_count=len(state['assets']),
                gain_posterior=state['gain'].snapshot()['posterior_mean']))
            write(out / 'steps.json', steps)
            write(out / 'owner_receipts.json', owners)
            write(out / 'heartbeat.json', dict(step=world.step_count,
                elapsed_s=time.monotonic() - start, coverage_fraction=measured['coverage_fraction']))
        summary = runtime.sensor_episode_summary(0)
        write(out / 'episode_summary.json', summary)
        write(out / 'runtime_audit.json', runtime.audit)
        geometry = base.base.export_geometry(out, mapper, world)
        write(out / 'final_instance_index.json', mapper.instance_support_index.summary())
        write(out / 'final_template_receipt.json', method.receipt(mapper))
        write(out / 'template_audit.json', method.audit_log(mapper))
        write(out / 'template_frame_receipts.json', method.frame_receipts(mapper))
        write(out / 'final_gain_calibration.json', state['gain'].snapshot())
        final = base.evaluate(mapper.belief, world.reachable)
        def hits(field):
            return {str(x): 0 if initial_eval[field] >= x else
                    next((r['step'] for r in steps if r[field] >= x), None)
                    for x in cfg['evaluation']['thresholds']}
        verify(layout)
        trace = json.dumps([(r['action'], r['position'], r['heading']) for r in steps], sort_keys=True).encode()
        result = dict(status='completed_execution', arm=arm, layout=layout,
            scene_id=first['scene_id'], scene_role=first['scene_role'],
            policy_variant=cfg['policy']['version'], original_method_equivalence=False,
            paid_actions=len(steps), paid_action_budget=cfg['policy']['paid_action_budget'],
            initial_coverage_fraction=initial_eval['coverage_fraction'],
            final_coverage_fraction=final['coverage_fraction'], final_evaluation=final,
            free_correct_fraction=final['free_correct_fraction'], false_occupied_fraction=final['false_occupied_fraction'],
            reference_free_cells=final['reference_free_cells'], first_coverage_action=hits('coverage_fraction'),
            first_free_correct_action=hits('free_correct_fraction'),
            path_length_m=world.moves * world.config.resolution_m,
            turn_actions=sum(r['action'] in ('left', 'right') for r in steps),
            collisions=world.collisions, termination=summary['termination'],
            final_pose=[*world.position, world.heading],
            final_pose_equals_initial_anchor=[*world.position, world.heading] == first['anchor'],
            zero_gain_actions=sum(r['new_reference_known_cells'] == 0 for r in steps),
            longest_zero_gain_streak=max((r['zero_coverage_gain_streak'] for r in steps), default=0),
            target_pose_matches=sum(r['matched_target_pose_after_step'] is True for r in steps),
            initial_packet_sha256=p.sha256() if not steps else first['initial_packet_sha256'],
            trajectory_sha256=hashlib.sha256(trace).hexdigest(), global_planning_calls=len(choices),
            decision_wall_s_total=sum(timings), elapsed_s=time.monotonic() - start,
            quality_export=geometry, reconstruction_accuracy_evaluated=False,
            partial_ideal_semantics=True, natural_detector=False,
            effective_args=effective['effective_args'], source_and_adapter_hashes_unchanged=True)
        write(out / 'result.json', result)
        write(out / 'artifact_manifest.json', dict(status='completed_execution',
            files_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
        print(json.dumps(dict(output=str(out), arm=arm, layout=layout, actions=len(steps),
            coverage=final['coverage_fraction'], elapsed_s=result['elapsed_s'])), flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(status='interrupted' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
            error=repr(exc), traceback=traceback.format_exc(), paid_actions=len(steps),
            elapsed_s=time.monotonic() - start, completed_episode=False))
        if runtime is not None:
            write(out / 'runtime_audit.json', runtime.audit)
        if mapper is not None:
            if hasattr(mapper, '_dual_template_state'):
                write(out / 'partial_template_audit.json', method.audit_log(mapper))
                write(out / 'partial_template_frame_receipts.json', method.frame_receipts(mapper))
            partial = out / 'partial_geometry'
            partial.mkdir()
            try:
                base.base.export_geometry(partial, mapper, world)
                write(partial / 'PARTIAL_ONLY.json', dict(completed_episode=False, paid_actions=len(steps)))
            except Exception as save_error:
                write(partial / 'export_failure.json', dict(error=repr(save_error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'selfcheck', 'run'))
    parser.add_argument('--layout', required=True, choices=tuple(protocol()['layouts']))
    parser.add_argument('--arm', choices=tuple(protocol()['arms']))
    args = parser.parse_args()
    dependencies()
    if args.command == 'prepare':
        prepare(args.layout)
    elif args.command == 'verify':
        print(json.dumps(verify(args.layout)))
    elif args.command == 'selfcheck':
        selfcheck(args.layout)
    elif args.arm is None:
        parser.error('run requires explicit --arm')
    else:
        run(args.layout, args.arm)


if __name__ == '__main__':
    main()
