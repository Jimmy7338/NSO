"""Four new matched-world nearest-frontier episodes; simulation only."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse, dataclasses, hashlib, importlib.util, json, signal, sys, time, traceback
from types import SimpleNamespace
import numpy as np
from controller import FrontierDriver

HERE = Path(__file__).resolve().parent
B11 = HERE.parents[1] / 'paired_unseen_gain_20261120'
sys.path.insert(0, str(B11 / 'execution'))
import runtime_binding as R
LAYOUTS = ('chain_a', 'cycle_a', 'chain_b', 'cycle_b')
ARM = 'F_strict'
write, sha = R.write, R.sha


def read(p):
    return json.loads(Path(p).read_text())


def verify():
    f = read(HERE / 'freeze.json')
    assert sha(B11 / 'execution/freeze.json') == f['B11_execution_freeze_sha256']
    for p, h in f['files_sha256'].items():
        assert sha(p) == h, p
    for p, h in read(B11 / 'execution/freeze.json')['files_sha256'].items():
        assert sha(p) == h, p
    return f


def initialize(layout):
    old = R.install()
    values = old.bootstrap(layout, 'S_visibility')
    world, tr, p, comp, runtime, inherited, owner = values
    initial = read(B11 / 'execution/layouts' / layout / 'prepared/initial_state.json')
    assert p.sha256() == initial['initial_packet_sha256']
    assert [*p.position, p.heading] == initial['anchor']
    from strict_policy import FrontierPolicy
    from env.grid_exploration import GridObservation, readonly_copy
    from nso.cpu_sensor_contract_v10 import digest
    from nso.frontier_policy import __file__ as frontier_file
    assert sha(frontier_file) == read(HERE / 'protocol.json')['frontier_source_sha256']
    mapper = runtime.states[0]['mapper']
    expected = read(B11 / 'execution/layouts' / layout / 'selfcheck.json')['arms']['S_visibility']
    assert digest(mapper.belief) == expected['belief_sha256']
    assert owner['owner_sha256'] == expected['owner_sha256']

    class AuditedFrontier(FrontierPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.selections = []

        def _candidate_score(self, obs, cell, heading, gain, distance, parent, base_score):
            self._scored.append(dict(cell=list(cell), heading=int(heading),
                gain_cells=int(gain), distance_cells=int(distance), gain_per_cost=float(base_score)))
            return super()._candidate_score(obs, cell, heading, gain, distance, parent, base_score)

        def _select(self, obs, traversable):
            self._scored = []
            chosen = super()._select(obs, traversable)
            self.selections.append(dict(paid_actions=int(obs.step), position=list(obs.position),
                heading=int(obs.heading), observed_belief_sha256=digest(obs.belief),
                reachable_candidate_scored=self._scored,
                selected_cell=None if not chosen else list(self.goal),
                selected_heading=None if not chosen else int(self.goal_heading),
                chosen=chosen, distance_priority='distance_cells,-gain_cells,cell',
                truth_or_reference_input=False))
            return chosen

    policy = AuditedFrontier(SimpleNamespace(resolution_m=world.config.resolution_m,
        robot_radius_m=world.config.robot_radius_m, sensor_range_m=world.config.max_depth_m,
        sensor_fov_deg=world.config.fov_deg), method='nearest_frontier', max_candidates=24)
    def observation():
        current = runtime.states[0]['packet']
        return GridObservation(readonly_copy(mapper.belief), readonly_copy(mapper.visible),
            current.position, current.heading, current.action_id, current.collision)
    driver = FrontierDriver(comp, runtime, policy, observation)
    return values, initial, mapper, policy, driver


def selfcheck(layout):
    verify()
    values, first, mapper, policy, driver = initialize(layout)
    world, tr, p, comp, runtime, inherited, owner = values
    action = driver.next_action()
    assert world.step_count == 0 and mapper.frames == 1
    assert comp._cpu_backend.scenes[0]['ledger'].remaining_budget == 256
    assert not any(x['method'] == 'select_topo_target' for x in comp._cpu_backend.calls)
    out = HERE / 'layouts' / layout
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'selfcheck.json', dict(status='passed_zero_action_same_packet_mapper_guard',
        initial_packet_sha256=p.sha256(), initial_anchor=first['anchor'],
        initial_evaluation=R.old.base.evaluate(mapper.belief, world.reachable),
        B11_selfcheck_sha256=sha(B11 / 'execution/layouts' / layout / 'selfcheck.json'),
        first_frontier_decisions=policy.selections, initial_authorized_action=action,
        initial_guard_assessment=runtime.states[0]['pending'], simulator_actions=0,
        template_scoring_calls=0, total_budget=256))
    print(json.dumps(dict(layout=layout, status='selfcheck_passed', paid_actions=0)), flush=True)


def export_geometry(out, mapper, world):
    import open3d as o3d
    np.savez_compressed(out / 'final_belief.npz', belief=mapper.belief,
        camera_seen=mapper.camera_seen, position=np.asarray(world.position), heading=np.asarray(world.heading))
    mesh = mapper.mesh()
    assert len(mesh.vertices) and len(mesh.triangles)
    assert o3d.io.write_triangle_mesh(str(out / 'final_tsdf_mesh.ply'), mesh, write_ascii=False)
    return dict(mesh_vertices=len(mesh.vertices), mesh_triangles=len(mesh.triangles),
                tsdf_voxel_m=mapper.config.voxel_m, measured_only=True)


def run(layout):
    verify()
    check = read(HERE / 'layouts' / layout / 'selfcheck.json')
    assert check['status'] == 'passed_zero_action_same_packet_mapper_guard'
    out = HERE / 'layouts' / layout / 'runs' / ARM / 'attempt_001'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    steps, owners, terminal_times = [], [], []
    mapper = runtime = world = None
    write(out / 'attempt.json', dict(status='started', arm=ARM, layout=layout,
        started_unix_s=time.time(), protocol_sha256=sha(HERE / 'protocol.json'),
        frontier_freeze_sha256=sha(HERE / 'freeze.json'), paid_action_budget=256))
    def interrupted(signum, frame):
        raise InterruptedError(f'external signal {signum}; incomplete episode')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        values, initial, mapper, policy, driver = initialize(layout)
        world, tr, p, comp, runtime, inherited, owner = values
        owners.append(owner)
        initial_eval = R.old.base.evaluate(mapper.belief, world.reachable)
        write(out / 'initial_evaluation.json', initial_eval)
        write(out / 'effective_configuration.json', dict(arm=ARM, baseline=read(HERE/'protocol.json'),
            inherited_sensor_runtime=inherited, NSO_target_selection_called=False,
            mapper_and_guard_same_as_B11=True))
        streak = 0
        while not runtime.states[0]['closed']:
            t = time.monotonic()
            action = driver.next_action()
            decision_s = time.monotonic() - t
            write(out / 'choices.json', policy.selections)
            write(out / 'interventions.json', driver.events)
            if action is None:
                terminal_times.append(decision_s)
                break
            state = runtime.states[0]
            phase = state['phase']
            target = None if policy.goal is None or phase == 'return' else [*policy.goal, policy.goal_heading]
            known_before = (mapper.belief != -1).copy()
            t = time.monotonic()
            raw, collision, done = world.step(action)
            p = R.old.packet(world, tr, raw, layout, action, collision, done)
            sensor_s = time.monotonic() - t
            t = time.monotonic()
            owner = R.old.front.associate(mapper, raw, p.frame_id)
            owners.append(owner)
            R.old.method.observe(mapper, raw)
            runtime.observe(0, world.step_count, None, None, None, sensor_packet=p)
            observe_s = time.monotonic() - t
            # Evaluation-only fields are never passed to FrontierPolicy/driver.
            measured = R.old.base.evaluate(mapper.belief, world.reachable)
            newly_known = int(np.count_nonzero((mapper.belief != -1) & ~known_before & world.reachable))
            streak = streak + 1 if newly_known == 0 else 0
            steps.append(dict(step=world.step_count, action=action, position=list(world.position),
                heading=world.heading, collision=bool(collision), done=bool(done), **measured,
                new_reference_known_cells=newly_known, zero_coverage_gain_streak=streak,
                path_length_m=world.moves * world.config.resolution_m,
                packet_sha256=p.sha256(), nonsemantic_packet_sha256=R.old.base.nonsemantic_packet_sha(p),
                owner_sha256=owner['owner_sha256'], selected_target_pose=target,
                matched_target_pose_after_step=target == [*world.position, world.heading] if target else None,
                phase_before_action=phase, decision_wall_s=decision_s,
                sensor_wall_s=sensor_s, observe_wall_s=observe_s,
                frontier_decision=driver.last_decision))
            write(out / 'steps.json', steps)
            write(out / 'heartbeat.json', dict(paid_actions=len(steps), elapsed_s=time.monotonic()-started,
                free_correct_fraction=measured['free_correct_fraction'], phase=phase))
        summary = runtime.sensor_episode_summary(0)
        write(out / 'episode_summary.json', summary)
        write(out / 'runtime_audit.json', runtime.audit)
        write(out / 'owner_receipts.json', owners)
        write(out / 'choices.json', policy.selections)
        write(out / 'interventions.json', driver.events)
        write(out / 'frame_receipts.json', R.old.method.frame_receipts(mapper))
        write(out / 'final_frame_receipt.json', R.old.method.receipt(mapper))
        geometry = export_geometry(out, mapper, world)
        final = R.old.base.evaluate(mapper.belief, world.reachable)
        def hits(field):
            return {str(t): 0 if initial_eval[field] >= t else next((s['step'] for s in steps
                    if s[field] >= t), None) for t in (.9, .95, .99, 1.)}
        assert len(steps) == world.step_count <= 256
        assert mapper.frames == len(steps) + 1
        assert not any(x['method'] == 'select_topo_target' for x in comp._cpu_backend.calls)
        verify()
        result = dict(status='completed_execution', arm=ARM, layout=layout, anchor=initial['anchor'],
            initial_packet_sha256=initial['initial_packet_sha256'], paid_actions=len(steps),
            paid_action_budget=256, initial_coverage_fraction=initial_eval['coverage_fraction'],
            final_coverage_fraction=final['coverage_fraction'], final_evaluation=final,
            free_correct_fraction=final['free_correct_fraction'], false_occupied_fraction=final['false_occupied_fraction'],
            reference_free_cells=final['reference_free_cells'],
            first_coverage_action=hits('coverage_fraction'), first_free_correct_action=hits('free_correct_fraction'),
            path_length_m=world.moves * world.config.resolution_m, collisions=world.collisions,
            turn_actions=sum(s['action'] in ('left','right') for s in steps),
            final_pose=[*world.position, world.heading],
            final_pose_equals_initial_anchor=[*world.position, world.heading] == initial['anchor'],
            termination=summary['termination'], elapsed_s=time.monotonic()-started,
            decision_wall_s_total=sum(s['decision_wall_s'] for s in steps),
            terminal_decision_wall_s=sum(terminal_times), global_planning_calls=len(policy.selections),
            trajectory_sha256=hashlib.sha256(json.dumps([(s['action'],s['position'],s['heading'])
                for s in steps],sort_keys=True).encode()).hexdigest(),
            zero_gain_actions=sum(s['new_reference_known_cells']==0 for s in steps),
            longest_zero_gain_streak=max((s['zero_coverage_gain_streak'] for s in steps),default=0),
            target_pose_matches=sum(s['matched_target_pose_after_step'] is True for s in steps),
            source_and_adapter_hashes_unchanged=True, geometry_export=geometry,
            independent_frontier_candidates=True, same_candidates_as_NSO=False,
            reconstruction_accuracy_evaluated=False, template_scoring_calls=0,
            no_reference_used_by_policy=True, no_truth_based_stopping=True)
        write(out / 'baseline_binding.json', dict(frontier_freeze=R.resource(HERE/'freeze.json'),
            B11_execution_freeze=R.resource(B11/'execution/freeze.json'),
            B11_initial_state=R.resource(B11/'execution/layouts'/layout/'prepared/initial_state.json'),
            protocol=R.resource(HERE/'protocol.json'), same_observation_update_and_guard=True,
            same_candidates=False, source_review_not_OS_access_trace=True))
        write(out / 'result.json', result)
        write(out / 'artifact_manifest.json', dict(status='completed_execution',
            files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_manifest.json'}))
        print(json.dumps(dict(output=str(out), layout=layout, paid_actions=len(steps),
            returned=result['final_pose_equals_initial_anchor'], free_correct=final['free_correct_fraction'])),flush=True)
    except BaseException as exc:
        write(out / 'failure.json', dict(status='interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt))else'failed',
            error=repr(exc), traceback=traceback.format_exc(), paid_actions=len(steps),
            completed_episode=False, elapsed_s=time.monotonic()-started))
        if mapper is not None:
            partial=out/'partial_geometry';partial.mkdir()
            try:export_geometry(partial,mapper,world)
            except Exception as err:write(partial/'export_failure.json',dict(error=repr(err)))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['selfcheck','run','verify'])
    parser.add_argument('--layout',choices=LAYOUTS);args=parser.parse_args()
    if args.command=='verify':verify()
    elif args.layout is None:parser.error('layout required')
    elif args.command=='selfcheck':selfcheck(args.layout)
    else:run(args.layout)
