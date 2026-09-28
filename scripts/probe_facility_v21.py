#!/usr/bin/env python3
"""Bounded V21 G/N joint-planning probe with frozen sources and fresh replays.

The default batch contains exactly four G trajectories. A separate N batch
requires explicit preparation in another output directory. The public-ROI
known-cell proxy target is a planning heuristic; evaluation coverage never
enters the runtime. Uses unchanged V19 world, sensing, TSDF and read-only
reference caches. No first-option intervention, training or semantic test.

The physical/save/replay protocol is derived from probe_facility_v20.py;
only the explicitly imported historical I/O helpers are runtime dependencies.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'configs/virtual3d/facility_joint_budget_v21_probe.json'

import numpy as np
from env.facility_documentation_v19 import FacilityWorldV19
from nso.cpu_sensor_contract_v10 import GridTransform, digest, json_value
from nso.decision_replay_v13 import save_packet, load_packet, array_hash
from utils.facility_metrics_v19 import FacilityEvaluatorV19
from utils.counterfactual_surface_visibility import reference_visible
from scripts.collect_facility_v19 import read, gzwrite, seal, check_sources
from scripts.collect_semantic_gain_v13_history import packet, sha, write


def require_space(config, path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    required = int(config['minimum_free_space_mib']) * 1024 * 1024
    if shutil.disk_usage(path).free < required:
        raise OSError(f'Insufficient free space: require at least {required} bytes')


def validate_config(config):
    if config['parents'] != ['D19-P00', 'D19-P01']:
        raise ValueError('The bounded probe requires the two existing V19 parents')
    if config['assignments'] != ['A_complex_B_simple', 'A_simple_B_complex']:
        raise ValueError('The bounded probe requires both paired assignments')
    if config['budgets'] != {'D19-P00': 160, 'D19-P01': 184} or config['coverage_minimum'] != .8:
        raise ValueError('V19 budgets and coverage gate must remain unchanged')
    if config['reference_seeds'] != [2026, 2027, 2028] or config['primary_reference_seed'] != 2026:
        raise ValueError('V19 reference sampling contract must remain unchanged')
    if config['global_reference_count'] != 12000:
        raise ValueError('Legacy visibility area uses the original 12000-sample denominator')
    if config['initial_history_actions'] != 0 or config['install_v19_declared_first_option']:
        raise ValueError('Unknown start with no paid prefix or first-option intervention required')
    if config['training_allowed'] or config['automatic_mode_expansion']:
        raise ValueError('No training or automatic mode expansion is allowed')
    if config['version'] != 'facility-joint-online-budget-v21-probe-1':
        raise ValueError('Expected the declared V21 collector/config version')
    if config['default_mode'] != 'G' or config['allowed_modes'] != ['G', 'N']:
        raise ValueError('V21 default is G; N requires separate explicit preparation')
    if config['sensor_model'] != 'iid_025px' or config['noise_seed'] != 1901:
        raise ValueError('V19 stereo model and paired noise seed must remain unchanged')
    if config['task_asset_count'] != 6 or config['replan_interval_actions'] != 5 or config['coverage_candidate_slots'] != 8:
        raise ValueError('Six assets, five-action replanning and eight coverage slots are fixed')
    if config['coverage_trace_action_stride'] != 1 or config['quality_curve_action_stride'] != 20:
        raise ValueError('Every-action coverage and 20-action quality checkpoints are required')
    if config['physical_contract'] != {'max_depth_m': 5., 'voxel_m': .04, 'truncation_m': .12}:
        raise ValueError('The V19 physical range and reconstruction resolution are fixed')
    proxy = config['planning_coverage_proxy']
    if (proxy['kind'] != 'known_cells_over_fixed_public_roi' or proxy['target_fraction'] != .8
            or proxy['target_is_heuristic'] is not True or proxy['evaluation_truth_used'] is not False
            or proxy['true_coverage_guaranteed'] is not False):
        raise ValueError('V21 requires a fixed-ROI observed proxy, not a true coverage guarantee')
    if config['semantic_information_test'] or config['full_architecture_efficacy_proven']:
        raise ValueError('This bounded G/N probe cannot declare semantic/full-architecture efficacy')


def bootstrap(case, config):
    # Lazy import permits syntax/CLI inspection while runtime development proceeds.
    from nso.facility_runtime_v21 import facility_components_v21, FacilityRuntimeV21
    world = FacilityWorldV19(case['parent'], case['assignment'],
        sensor_model=case['sensor_model'], noise_seed=case['noise_seed'])
    for name, expected in config['physical_contract'].items():
        if getattr(world.config, name) != expected:
            raise ValueError('Frozen physical world contract changed: ' + name)
    args = SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode=case['mode'], cpu_disable_feedback=False,
        cpu_max_candidates=5, cpu_coverage_slots=4, cpu_planner_revision='v10_3_1',
        cpu_semantic_source_schema='inspection_v4', cpu_measured_novelty_floor=.25,
        cpu_task_asset_count=config['task_asset_count'],
        cpu_v20_replan_interval=config['replan_interval_actions'],
        cpu_v20_coverage_slots=config['coverage_candidate_slots'])
    components = facility_components_v21(args, world.shape)
    runtime = FacilityRuntimeV21(components, 1, world.shape)
    first = packet(world, {'parent': case['parent']}, None)
    runtime.start_sensor_episode(0, config=world.config,
        transform=GridTransform(world.shape, world.config.resolution_m), packets=[first],
        total_budget=case['budget'], return_anchor=(*first.position, first.heading))
    return world, runtime, first


def reference_name(case, seed):
    return f'references/{case["parent"]}_{case["assignment"]}_{seed}.npz'


def check_references(manifest):
    source = Path(manifest['reference_source_root'])
    if sha(source / 'manifest.json') != manifest['reference_source_manifest_sha256']:
        raise ValueError('Historical V19 manifest changed')
    if sha(source / 'artifact_hashes.json') != manifest['reference_source_inventory_sha256']:
        raise ValueError('Historical V19 artifact inventory changed')
    references = manifest['reference_sha256']
    if len(references) != 12:
        raise ValueError('Exactly twelve pre-existing V19 reference caches required')
    for name, wanted in references.items():
        path = source / name
        if not path.is_file() or sha(path) != wanted:
            raise ValueError('Read-only V19 reference missing or changed: ' + name)


def evaluators_for(world, case, manifest):
    evaluators = {}
    for seed in manifest['config']['reference_seeds']:
        name = reference_name(case, seed)
        path = Path(manifest['reference_source_root']) / name
        # The evaluator can create a missing cache: fail before its constructor.
        if name not in manifest['reference_sha256'] or not path.is_file():
            raise ValueError('Refusing to create a missing V19 reference: ' + name)
        if sha(path) != manifest['reference_sha256'][name]:
            raise ValueError('Read-only V19 reference changed: ' + name)
        evaluators[seed] = FacilityEvaluatorV19(world, seed=seed, reference_cache=path)
    return evaluators


def prepare(root, reference_source, mode=None):
    config = read(CONFIG)
    validate_config(config)
    mode = mode or config['default_mode']
    if mode not in config['allowed_modes']:
        raise ValueError('Only N or G can be explicitly prepared')
    reference_source = reference_source.resolve()
    historical = read(reference_source / 'manifest.json')
    if historical.get('status') != 'complete':
        raise ValueError('Completed V19 physical/replay evidence required as cache source')
    required = {reference_name(dict(parent=p, assignment=a), seed)
        for p in config['parents'] for a in config['assignments'] for seed in config['reference_seeds']}
    references = historical.get('reference_sha256', {})
    if set(references) != required:
        raise ValueError('Historical V19 reference inventory does not match the four worlds')
    if reference_source == root.resolve() or reference_source in root.resolve().parents:
        raise ValueError('Probe output cannot modify the historical V19 evidence tree')
    sources = {str(path.relative_to(ROOT)): sha(path)
        for directory in ('env', 'nso', 'utils')
        for path in sorted((ROOT / directory).glob('*.py'))}
    for path in (Path(__file__).resolve(), CONFIG, ROOT / 'scripts/collect_facility_v19.py',
                 ROOT / 'scripts/collect_semantic_gain_v13_history.py'):
        sources[str(path.relative_to(ROOT))] = sha(path)
    manifest = dict(status='preparing', version=config['version'], mode=mode, config=config,
        source_sha256=sources, reference_source_root=str(reference_source),
        planning_coverage_proxy=config['planning_coverage_proxy'],
        reference_source_manifest_sha256=sha(reference_source / 'manifest.json'),
        reference_source_inventory_sha256=sha(reference_source / 'artifact_hashes.json'),
        reference_sha256=references, cases=[], training_allowed=False,
        reference_sampling_repeats_are_independent_experiments=False,
        semantic_information_test=False, full_architecture_efficacy_proven=False)
    check_references(manifest)
    require_space(config, root)
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'manifest.json', manifest)
    try:
        with zipfile.ZipFile(root / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in sources:
                archive.write(ROOT / name, name)
        paired_geometry = {}
        for parent in config['parents']:
            for assignment in config['assignments']:
                case = dict(index=len(manifest['cases']), parent=parent, assignment=assignment,
                    mode=mode, budget=config['budgets'][parent], sensor_model=config['sensor_model'],
                    noise_seed=config['noise_seed'], initial_history_actions=0,
                    declared_first_option_installed=False)
                world, runtime, first = bootstrap(case, config)
                evaluators_for(world, case, manifest)
                geometry = digest(dict(depth=first.frame.depth_m, scan=first.scan.ranges_m,
                    intrinsic=first.frame.intrinsic, world_from_camera=first.frame.world_from_camera,
                    world_from_laser=first.scan.world_from_laser))
                if parent in paired_geometry and paired_geometry[parent] != geometry:
                    raise ValueError('Paired initial measured geometry differs')
                paired_geometry[parent] = geometry
                audit_start = len(runtime.audit)
                first_action = runtime.next_local_action(0)
                denials = [x for x in runtime.audit[audit_start:]
                           if x.get('event') == 'v14_route_denied']
                case.update(initial_packet_sha256=first.sha256(), shared_geometry_sha256=geometry,
                    initial_guard=dict(first_authorized_action=first_action,
                        any_action_authorized=first_action is not None, denials=denials))
                if world.step_count != 0:
                    raise ValueError('Preparation must not execute paid actions')
                manifest['cases'].append(case)
                write(root / 'manifest.json', manifest)
        check_sources(manifest)
        check_references(manifest)
        manifest['status'] = 'prepared'
    except Exception as error:
        manifest.update(status='failed_preparation', error=repr(error))
        raise
    finally:
        write(root / 'manifest.json', manifest)
        seal(root)


def verify_case_inventory(folder):
    for name, wanted in read(folder / 'artifact_hashes.json').items():
        if sha(folder / name) != wanted:
            raise ValueError('Recorded physical artifact changed: ' + name)


def run_case(root, index, replay=False):
    manifest = read(root / 'manifest.json')
    config = manifest['config']
    validate_config(config)
    check_sources(manifest)
    check_references(manifest)
    if manifest['status'] not in ('prepared', 'running', 'failed', 'complete') or len(manifest['cases']) != 4:
        raise ValueError('A fully prepared four-case probe is required')
    if not 0 <= index < 4:
        raise ValueError('Case index must be in 0..3')
    require_space(config, root)
    case = manifest['cases'][index]
    folder = root / f'case_{index:02d}'
    expected = None
    physical_pid = None
    if replay:
        verify_case_inventory(folder)
        expected = read(folder / 'result.json')
        physical_pid = read(folder / 'timing.json')['process_id']
        if physical_pid == os.getpid():
            raise ValueError('Replay must execute in a fresh process')
        if (folder / 'verification.json').exists():
            raise FileExistsError('An existing verification must not be overwritten')
    else:
        folder.mkdir()
        (folder / 'packets').mkdir()
    world, runtime, first = bootstrap(case, config)
    backend = runtime.components._cpu_backend
    state = runtime.states[0]
    mapper = state['mapper']
    if first.sha256() != case['initial_packet_sha256']:
        raise ValueError('Initial measured packet changed')
    evaluators = evaluators_for(world, case, manifest)
    primary = config['primary_reference_seed']
    evaluator = evaluators[primary]
    visible = reference_visible(evaluator.global_evaluator.reference, first.frame, evaluator.truth,
        first.frame.world_from_camera, world.config.max_depth_m)
    initial_visible = visible.copy()
    anchor = (*first.position, first.heading)

    def true_coverage():
        # Evaluator-only truth: this scalar is never sent to the runtime.
        return float(np.count_nonzero((mapper.belief != -1) & world.reachable) / world.reachable.sum())

    def evaluate(seeds=None):
        seeds = config['reference_seeds'] if seeds is None else seeds
        coverage = true_coverage()
        returned = tuple((*world.position, world.heading)) == tuple(anchor)
        failed = bool(state.get('termination', {}).get('failed', False))
        mesh = mapper.mesh()
        return {str(seed): evaluators[seed].evaluate(mesh, coverage, returned=returned,
            collisions=world.collisions, failed=failed) for seed in seeds}

    before = evaluate()
    initial_path = folder / 'packets/0000.npz'
    if replay:
        if load_packet(initial_path).sha256() != first.sha256():
            raise ValueError('Fresh initial packet differs from recorded raw packet')
    else:
        save_packet(initial_path, first)
    actions = []
    coverage_trace = [dict(action_id=0, coverage_2d=true_coverage())]
    curve = [dict(action_id=0, metrics=before[str(primary)])]
    started = perf_counter()
    while not state['closed']:
        action = runtime.next_local_action(0)
        if action is None:
            break
        frame, collision, done = world.step(action)
        measured = packet(world, {'parent': case['parent']}, action, frame, collision, done)
        path = folder / f'packets/{world.step_count:04d}.npz'
        if replay:
            if measured.sha256() != load_packet(path).sha256():
                raise ValueError('Fresh world sensor packet differs')
        else:
            save_packet(path, measured)
        runtime.observe(0, world.step_count, None, None, None, sensor_packet=measured)
        visible |= reference_visible(evaluator.global_evaluator.reference, frame, evaluator.truth,
            frame.world_from_camera, world.config.max_depth_m)
        actions.append(dict(action=action, pose=[*world.position, world.heading],
            collision=collision, packet_sha256=measured.sha256()))
        coverage_trace.append(dict(action_id=world.step_count, coverage_2d=true_coverage()))
        if world.step_count % config['quality_curve_action_stride'] == 0:
            curve.append(dict(action_id=world.step_count, metrics=evaluate((primary,))[str(primary)]))
        if world.step_count % 50 == 0:
            print(f'case={index} mode={case["mode"]} replay={replay} actions={world.step_count}', flush=True)
    elapsed = perf_counter() - started
    if not state['closed']:
        runtime.close_sensor_episode(0, reason='joint_budget_probe_terminated')
    after = evaluate()
    mesh = mapper.mesh()
    if curve[-1]['action_id'] != world.step_count:
        curve.append(dict(action_id=world.step_count, metrics=after[str(primary)]))
    else:
        curve[-1]['metrics'] = after[str(primary)]
    calls = [{key: value for key, value in row.items() if key != 'elapsed_s'} for row in backend.calls]
    reached = [row['action_id'] for row in coverage_trace if row['coverage_2d'] >= config['coverage_minimum']]
    result = json_value(dict(case=case, actions=actions, before=before, after=after,
        coverage_trace=coverage_trace, quality_curve=curve,
        first_evaluator_coverage_gate_action=min(reached) if reached else None,
        coverage_gate_first_hit_is_not_endpoint_eligibility=True,
        termination=state['termination'], paid_actions=world.step_count,
        path_distance_m=world.moves * world.config.resolution_m, collisions=world.collisions,
        module_calls_sha256=digest(calls), runtime_audit_sha256=digest(runtime.audit),
        final_mesh_sha256={key: array_hash(np.asarray(getattr(mesh, key)))
            for key in ('vertices', 'triangles', 'vertex_colors')},
        new_visible_surface_m2=float(np.count_nonzero(visible & ~initial_visible))
            * float(world.mesh.get_surface_area()) / config['global_reference_count'],
        curve_action_stride=config['quality_curve_action_stride'], coverage_trace_action_stride=1,
        task='six_facility_joint_budget_v21_probe',
        planning_coverage_proxy=config['planning_coverage_proxy'],
        sensor_calibration_scope=world.sensor_metadata,
        primitive_budget_compliant=world.step_count <= case['budget'],
        initial_history_actions=0, declared_first_option_installed=False,
        semantic_information_test=False, trained=False))
    if replay:
        if result != expected:
            raise ValueError('Replay actions, coverage, metrics, mesh, or module decisions differ')
        write(folder / 'verification.json', dict(status='passed',
            fresh_world_sensor_action_metric_replay=True, independent_process=True,
            physical_process_id=physical_pid, replay_process_id=os.getpid(), actions=world.step_count))
    else:
        write(folder / 'result.json', result)
        gzwrite(folder / 'module_calls.json.gz', backend.calls)
        gzwrite(folder / 'runtime_audit.json.gz', runtime.audit)
        write(folder / 'timing.json', dict(process_id=os.getpid(), simulation_runtime_s=elapsed,
            includes_sensor_simulation_and_visibility_tracking=True,
            includes_interval_quality_evaluation=True,
            recorded_module_compute_s=sum(row['elapsed_s'] for row in backend.calls),
            reference_initialization_and_final_evaluation_excluded=True))
        np.savez_compressed(folder / 'final_mesh.npz',
            **{key: np.asarray(getattr(mesh, key)) for key in ('vertices', 'triangles', 'vertex_colors')})
    seal(folder)
    check_sources(manifest)
    check_references(manifest)
    final = result['after'][str(primary)]
    print(f'case={index} mode={case["mode"]} replay={replay} complete; '
          f'actions={world.step_count}; C={final["coverage_2d"]:.6f}; eligible={final["eligible"]}', flush=True)


def aggregate(root, manifest):
    rows = []
    for case in manifest['cases']:
        folder = root / f'case_{case["index"]:02d}'
        verify_case_inventory(folder)
        result = read(folder / 'result.json')
        verification = read(folder / 'verification.json')
        if (verification.get('status') != 'passed' or not verification.get('independent_process')
                or not verification.get('fresh_world_sensor_action_metric_replay')
                or verification.get('physical_process_id') == verification.get('replay_process_id')
                or verification.get('actions') != result['paid_actions']):
            raise ValueError('All four fresh process replays must pass before aggregation')
        if result['case'] != case:
            raise ValueError('Recorded case differs from the prepared V21 manifest')
        final = result['after'][str(manifest['config']['primary_reference_seed'])]
        rows.append(dict(index=case['index'], parent=case['parent'], assignment=case['assignment'],
            mode=case['mode'], budget=case['budget'], paid_actions=result['paid_actions'],
            coverage_2d=final['coverage_2d'], external_macro_f1=final['05cm']['external_macro_f1'],
            joint_external=final['05cm']['joint_external'], eligible=final['eligible'],
            returned_to_anchor=result['termination']['returned_to_anchor'],
            collisions=result['collisions'], failed=result['termination']['failed'],
            primitive_budget_compliant=result['primitive_budget_compliant'],
            first_evaluator_coverage_gate_action=result['first_evaluator_coverage_gate_action'],
            replay_status=verification['status'], replay_actions=verification['actions']))
    summary = dict(status='complete', mode=manifest['mode'], cases=rows,
        unique_physical_trajectories=len(rows), independent_process_replays=len(rows),
        total_physical_actions=sum(row['paid_actions'] for row in rows),
        total_replay_actions=sum(row['replay_actions'] for row in rows),
        eligible_count=sum(row['eligible'] for row in rows),
        common_online_coverage_passed=all(row['eligible'] and row['primitive_budget_compliant'] for row in rows),
        reference_sampling_repeats_are_independent_experiments=False,
        planning_coverage_proxy=manifest['config']['planning_coverage_proxy'],
        semantic_information_test=False, training_allowed=False, full_architecture_efficacy_proven=False,
        joint_planning_efficacy_proven=False)
    write(root / 'result.json', summary)
    return summary


def run_all(root):
    manifest = read(root / 'manifest.json')
    check_sources(manifest)
    check_references(manifest)
    if manifest['status'] != 'prepared':
        raise ValueError('Batch launch requires fresh prepared evidence; use explicit case for recovery')
    manifest['status'] = 'running'
    write(root / 'manifest.json', manifest)
    try:
        for case in manifest['cases']:
            command = [sys.executable, str(Path(__file__).resolve()), '--output', str(root),
                       '--case', str(case['index'])]
            subprocess.run(command, check=True)
            subprocess.run(command + ['--replay'], check=True)
        aggregate(root, manifest)
        check_sources(manifest)
        check_references(manifest)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write(root / 'manifest.json', manifest)
        seal(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--prepare', type=Path, metavar='COMPLETED_V19_ROOT')
    operation.add_argument('--case', type=int)
    operation.add_argument('--all', action='store_true', dest='run_all')
    parser.add_argument('--mode', choices=('N', 'G'), help='Preparation only; default G; N requires a separate output root')
    parser.add_argument('--replay', action='store_true', help='Fresh process replay of --case')
    args = parser.parse_args()
    if args.mode is not None and args.prepare is None:
        parser.error('--mode is only accepted during preparation; execution uses the frozen mode')
    if args.replay and args.case is None:
        parser.error('--replay requires --case')
    root = args.output.resolve()
    if args.prepare is not None:
        prepare(root, args.prepare, args.mode)
    elif args.case is not None:
        try:
            run_case(root, args.case, args.replay)
        except Exception as error:
            receipt = root / f'case_{args.case:02d}_{"replay" if args.replay else "physical"}_failure.json'
            if root.is_dir() and not receipt.exists():
                write(receipt, dict(status='failed', case=args.case, replay=args.replay, error=repr(error)))
            raise
    else:
        run_all(root)


if __name__ == '__main__':
    main()
