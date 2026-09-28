#!/usr/bin/env python3
"""Early coverage N/A/B information screen with common N continuation.

Twelve declarations deduplicate into ten physical trajectories only when the
actual initial complete paths are equal within a world assignment. Each unique
case is freshly executed and independently replayed. Aliases are not extra
experiments. No semantic policy is trained/deployed; no strong-G comparison.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
from time import perf_counter
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG = ROOT / 'configs/virtual3d/facility_early_coverage_v22_probe.json'
PROTOCOL = ROOT / 'docs/research/V22_EARLY_COVERAGE_PROTOCOL_20260915.md'
FIXED_SEMANTIC_RULE = dict(
    input='actual initial v22_first_choice.observed_targets canonical class_vote',
    aligned='choose the unique A/B role with negative class_vote (complex prior)',
    flipped='choose the unique A/B role with positive class_vote',
    forbidden_inputs=['assignment', 'hidden geometry', 'future observations', 'evaluation rewards'],
    fitted=False, deployed_online=False, oracle_is_separate_upper_bound=True)

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
    expected = dict(version='facility-early-coverage-v22-information-screen-1',
        parents=['D19-P00', 'D19-P01'], assignments=['A_complex_B_simple', 'A_simple_B_complex'],
        budgets={'D19-P00': 160, 'D19-P01': 184}, coverage_minimum=.8,
        reference_seeds=[2026, 2027, 2028], primary_reference_seed=2026, global_reference_count=12000,
        sensor_model='iid_025px', noise_seed=1901, task_asset_count=6,
        replan_interval_actions=5, coverage_candidate_slots=8, quality_curve_action_stride=20,
        coverage_trace_action_stride=1, default_mode='N', allowed_modes=['N'], continuation_mode='N',
        first_options=['N', 'A', 'B'], first_choice_action_id=0, initial_history_actions=0,
        expected_declared_cases=12, expected_unique_cases=10,
        expected_unique_cases_per_parent={'D19-P00': 4, 'D19-P01': 6},
        minimum_free_space_mib=48, estimated_total_output_mib=100,
        parallel_case_workers=1, fixed_semantic_rule=FIXED_SEMANTIC_RULE,
        physical_contract={'max_depth_m': 5., 'voxel_m': .04, 'truncation_m': .12},
        training_allowed=False, semantic_policy_deployed=False, strong_G_comparison_performed=False,
        old_N_metric_reuse_allowed=False, aliases_are_independent_experiments=False,
        semantic_information_test=True, semantic_information_screen=True,
        semantic_information_gain_proven=False, full_architecture_efficacy_proven=False,
        install_v19_declared_first_option=False, automatic_mode_expansion=False)
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError('Fixed V22 declaration changed: ' + name)
    proxy = config['planning_coverage_proxy']
    if (proxy['kind'] != 'known_cells_over_fixed_public_roi' or proxy['target_fraction'] != .8
            or proxy['evaluation_truth_used'] or proxy['true_coverage_guaranteed']
            or not proxy['target_is_heuristic']):
        raise ValueError('Fixed public ROI heuristic declaration changed')


PATH_FIELDS = ('pose', 'states', 'actions', 'outbound_states', 'outbound_actions',
    'return_states', 'return_actions', 'outbound_cost', 'return_cost', 'cost',
    'arrival_action', 'return_anchor')


def physical_path(selected):
    if selected is None:
        raise ValueError('A declared initial coverage path must exist')
    path = {name: selected[name] for name in PATH_FIELDS}
    if (path['cost'] != path['outbound_cost'] + path['return_cost']
            or len(path['actions']) != path['cost']
            or len(path['outbound_actions']) != path['outbound_cost']
            or len(path['return_actions']) != path['return_cost']):
        raise ValueError('Initial path paid cost is inconsistent')
    return json_value(path)


def nonsemantic_identity(first):
    return digest(dict(depth=first.frame.depth_m, scan=first.scan.ranges_m,
        intrinsic=first.frame.intrinsic, world_from_camera=first.frame.world_from_camera,
        world_from_laser=first.scan.world_from_laser, camera_timestamp=first.frame.timestamp_s,
        laser_timestamp=first.scan.timestamp_s, angle_min=first.scan.angle_min_rad,
        angle_increment=first.scan.angle_increment_rad, range_max=first.scan.range_max_m))


def observed_votes(scene):
    marked = [a for a in scene['assets'] if a['marked_points'] > 0]
    marked.sort(key=lambda a: float((np.asarray(a['observed_low'])[0] + np.asarray(a['observed_high'])[0]) / 2.))
    if len(marked) != 2:
        raise ValueError('Initial A/B association requires exactly two observed marked assets')
    return {role: float(a['class_vote']) for role, a in zip(('A', 'B'), marked)}


def receipt_votes(choice, scene):
    rows = choice['observed_targets']
    votes = {row['role']: float(row['class_vote']) for row in rows}
    if len(rows) != 2 or set(votes) != {'A', 'B'} or votes != observed_votes(scene):
        raise ValueError('Actual initial receipt and observed A/B category votes differ')
    if sum(v < 0 for v in votes.values()) != 1 or sum(v > 0 for v in votes.values()) != 1:
        raise ValueError('Fixed semantic rule requires one negative and one positive observed vote')
    return votes


def validate_manifest(manifest):
    config = manifest['config']; validate_config(config)
    cases = manifest['cases']; declared = manifest['declared_cases']
    if len(cases) != 10 or len(declared) != 12 or [c['index'] for c in cases] != list(range(10)):
        raise ValueError('Expected ten unique cases and twelve declarations')
    if [d['declared_index'] for d in declared] != list(range(12)):
        raise ValueError('Declared indices are not complete')
    for d in declared:
        case = cases[d['unique_case_index']]
        if digest(physical_path(d['initial_selected'])) != d['initial_path_sha256']:
            raise ValueError('Declared physical path differs from its digest')
        if (d['parent'], d['assignment'], d['initial_path_sha256']) != (case['parent'], case['assignment'], case['initial_path_sha256']):
            raise ValueError('Alias crossed a world assignment or physical path')
        if manifest['declared_to_unique'].get(str(d['declared_index'])) != case['index']:
            raise ValueError('Explicit alias mapping differs')
    for parent in config['parents']:
        if sum(c['parent'] == parent for c in cases) != config['expected_unique_cases_per_parent'][parent]:
            raise ValueError('Unexpected unique-case count for ' + parent)
        for assignment in config['assignments']:
            ds = {d['first_option']: d for d in declared if d['parent'] == parent and d['assignment'] == assignment}
            if set(ds) != {'N', 'A', 'B'}:
                raise ValueError('Incomplete N/A/B declaration group')
            ids = {key: value['unique_case_index'] for key, value in ds.items()}
            if parent == 'D19-P00' and not (ids['N'] == ids['B'] and ids['A'] != ids['N']):
                raise ValueError('Observed P00 N=B path alias did not hold')
            if parent == 'D19-P01' and len(set(ids.values())) != 3:
                raise ValueError('Observed P01 requires three distinct paths')


def initial_option_outcome(case, audit, actions):
    oid = case['initial_selected']['option_id']
    authored = {row['next_action_id']: row for row in audit
        if row.get('event') == 'paid_action_authorized' and row.get('option_id') == oid
        and row.get('phase') == 'outbound'}
    arrivals = [row['action_id'] for row in audit if row.get('event') == 'observation'
        and row.get('arrived') and row['action_id'] in authored]
    cancellations = [row for row in audit if row.get('option_id') == oid
        and ((row.get('event') == 'v21_route_update' and not row.get('target_retained', True))
             or 'denied' in row.get('event', ''))]
    return dict(option_id=oid, declared_first_option=case['first_option'],
        initial_target=case['initial_selected']['pose'],
        paid_outbound_action_ids=sorted(authored), paid_outbound_actions=len(authored),
        arrival_action_ids=arrivals, measured_arrived=bool(arrivals),
        cancellation_or_denial_events=cancellations,
        explicit_cancel_or_denial=bool(cancellations),
        target_pose_ever_observed=any(a['pose'] == case['initial_selected']['pose'] for a in actions),
        target_pose_visit_is_not_attributed_quality_gain=True)


def bootstrap(case, config):
    # Lazy import permits syntax/CLI inspection while runtime development proceeds.
    from nso.facility_runtime_v22 import facility_components_v22, FacilityRuntimeV22
    world = FacilityWorldV19(case['parent'], case['assignment'],
        sensor_model=case['sensor_model'], noise_seed=case['noise_seed'])
    for name, expected in config['physical_contract'].items():
        if getattr(world.config, name) != expected:
            raise ValueError('Frozen physical world contract changed: ' + name)
    args = SimpleNamespace(nso_backend='cpu_v10', eval=True, train_global=False,
        use_open_vocab_semantic=True, use_topo_graph=True, use_rpn_uq=True, use_igcr=True,
        cpu_score_mode='N', cpu_v22_first_option=case['first_option'], cpu_disable_feedback=False,
        cpu_max_candidates=5, cpu_coverage_slots=4, cpu_planner_revision='v10_3_1',
        cpu_semantic_source_schema='inspection_v4', cpu_measured_novelty_floor=.25,
        cpu_task_asset_count=config['task_asset_count'],
        cpu_v20_replan_interval=config['replan_interval_actions'],
        cpu_v20_coverage_slots=config['coverage_candidate_slots'])
    components = facility_components_v22(args, world.shape)
    runtime = FacilityRuntimeV22(components, 1, world.shape)
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


def prepare(root, reference_source):
    config = read(CONFIG); validate_config(config)
    reference_source = reference_source.resolve()
    historical = read(reference_source / 'manifest.json')
    if historical.get('status') != 'complete':
        raise ValueError('Completed V19 reference evidence required')
    required = {reference_name(dict(parent=p, assignment=a), seed)
        for p in config['parents'] for a in config['assignments'] for seed in config['reference_seeds']}
    references = historical.get('reference_sha256', {})
    if set(references) != required:
        raise ValueError('Reference inventory differs from the four fixed worlds')
    if reference_source == root.resolve() or reference_source in root.resolve().parents:
        raise ValueError('Output must be outside historical reference evidence')
    require_space(config, root)
    space_path = root.resolve()
    while not space_path.exists(): space_path = space_path.parent
    estimated = (config['minimum_free_space_mib'] + config['estimated_total_output_mib']) * 1024 * 1024
    if shutil.disk_usage(space_path).free < estimated:
        raise OSError('Preparation requires 100 MiB estimated output plus 48 MiB free reserve')
    sources = {str(path.relative_to(ROOT)): sha(path) for directory in ('env', 'nso', 'utils')
        for path in sorted((ROOT / directory).glob('*.py'))}
    # These are the only imported script modules, including their transitive
    # script helper. The copied V21 collector is not imported at runtime.
    for path in (Path(__file__).resolve(), CONFIG, ROOT / 'scripts/collect_facility_v19.py',
                 ROOT / 'scripts/collect_semantic_gain_v13_history.py', PROTOCOL):
        sources[str(path.relative_to(ROOT))] = sha(path)
    if 'nso/facility_runtime_v22.py' not in sources:
        raise ValueError('V22 runtime must exist before source freezing')
    manifest = dict(status='preparing', version=config['version'], mode='N', continuation_mode='N', config=config,
        source_sha256=sources, reference_source_root=str(reference_source),
        reference_source_manifest_sha256=sha(reference_source / 'manifest.json'),
        reference_source_inventory_sha256=sha(reference_source / 'artifact_hashes.json'),
        reference_sha256=references, cases=[], declared_cases=[], declared_to_unique={},
        training_allowed=False, semantic_information_test=True, semantic_information_screen=True,
        semantic_information_gain_proven=False, semantic_policy_deployed=False,
        strong_G_comparison_performed=False, full_architecture_efficacy_proven=False,
        reference_sampling_repeats_are_independent_experiments=False,
        aliases_are_independent_experiments=False, old_N_metric_reuse_allowed=False)
    check_references(manifest)
    root.mkdir(parents=True, exist_ok=False); write(root / 'manifest.json', manifest)
    try:
        with zipfile.ZipFile(root / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
            for name in sources: archive.write(ROOT / name, name)
        paths, parent_geometry, parent_pools = {}, {}, {}
        parent_option_maps, parent_target_geometry, paired_votes = {}, {}, {}
        for parent in config['parents']:
            for assignment in config['assignments']:
                for first_option in config['first_options']:
                    declaration = dict(declared_index=len(manifest['declared_cases']), parent=parent,
                        assignment=assignment, mode='N', first_option=first_option,
                        budget=config['budgets'][parent], sensor_model=config['sensor_model'],
                        noise_seed=config['noise_seed'], initial_history_actions=0,
                        declared_first_option_installed=False, first_choice_selected_by_backend=True)
                    world, runtime, first = bootstrap(declaration, config)
                    evaluators_for(world, declaration, manifest)
                    audit_start = len(runtime.audit); action = runtime.next_local_action(0)
                    scene = runtime.components._cpu_backend.scenes[0]
                    selection = scene['last_selection']; selected = selection['selected']
                    choice = selection['v22_first_choice']
                    if action is None or selected is None or not selected['group'].startswith('coverage_'):
                        raise ValueError('Every declaration needs an authorized initial coverage choice')
                    if choice['requested_option'] != first_option:
                        raise ValueError('Backend did not apply the declared first option')
                    denials = [row for row in runtime.audit[audit_start:] if row.get('event') == 'v14_route_denied']
                    if denials: raise ValueError('Initial declaration was denied before any paid observation')
                    path = physical_path(selected); path_sha = digest(path)
                    geometry = nonsemantic_identity(first); pool = choice['common_pool_sha256']
                    if parent in parent_geometry and parent_geometry[parent] != geometry:
                        raise ValueError('Paired initial nonsemantic sensor bytes differ')
                    if parent in parent_pools and parent_pools[parent] != pool:
                        raise ValueError('Declared common initial coverage pool differs')
                    option_map = choice['option_map']
                    target_geometry = [{k: v for k, v in target.items() if k != 'class_vote'}
                        for target in choice['observed_targets']]
                    votes = receipt_votes(choice, scene)
                    if parent in parent_option_maps and parent_option_maps[parent] != option_map:
                        raise ValueError('Paired initial option_map differs')
                    if parent in parent_target_geometry and parent_target_geometry[parent] != target_geometry:
                        raise ValueError('Paired observed target geometry differs')
                    vote_key = (parent, assignment)
                    if vote_key in paired_votes and paired_votes[vote_key] != votes:
                        raise ValueError('Votes changed across declarations of the same initial world')
                    paired_votes[vote_key] = votes
                    other_assignment = next(a for a in config['assignments'] if a != assignment)
                    other_votes = paired_votes.get((parent, other_assignment))
                    if other_votes is not None and (votes['A'] != other_votes['B'] or votes['B'] != other_votes['A']):
                        raise ValueError('Canonical initial category votes did not exactly exchange')
                    parent_geometry[parent] = geometry; parent_pools[parent] = pool
                    parent_option_maps[parent] = option_map; parent_target_geometry[parent] = target_geometry
                    declaration.update(initial_packet_sha256=first.sha256(), initial_selected=selected,
                        initial_path=path, initial_path_sha256=path_sha,
                        initial_nonsemantic_sha256=geometry, shared_geometry_sha256=geometry,
                        initial_candidate_pool_sha256=pool, observed_category_votes=votes,
                        fixed_semantic_rule_choice=next(role for role, vote in votes.items() if vote < 0),
                        flipped_semantic_rule_choice=next(role for role, vote in votes.items() if vote > 0),
                        category_vote_association='observed left/right marked clusters, no hidden bounds',
                        v22_first_choice=choice, initial_guard=dict(first_authorized_action=action,
                            any_action_authorized=True, denials=[]))
                    key = (parent, assignment, path_sha)
                    if key not in paths:
                        index = len(manifest['cases']); paths[key] = index
                        case = dict(declaration, index=index)
                        case.pop('declared_index')
                        manifest['cases'].append(case)
                        alias = None
                    else:
                        index = paths[key]; alias = manifest['cases'][index]['first_option']
                        if manifest['cases'][index]['initial_path'] != path:
                            raise ValueError('Physical path digest collision')
                    declaration.update(unique_case_index=index, alias_of_first_option=alias)
                    manifest['declared_cases'].append(declaration)
                    manifest['declared_to_unique'][str(declaration['declared_index'])] = index
                    if world.step_count != 0: raise ValueError('Preparation executed a paid action')
                    write(root / 'manifest.json', manifest)
        validate_manifest(manifest); check_sources(manifest); check_references(manifest)
        manifest.update(status='prepared',
            initial_pair_checks=dict(nonsemantic_sensor_bytes_equal=True, common_candidate_pool_equal=True,
                option_map_equal=True, observed_target_geometry_equal=True,
                canonical_class_votes_exactly_exchanged=True, paid_actions_executed=0),
            unique_case_count=len(manifest['cases']),
            declared_case_count=len(manifest['declared_cases']))
    except Exception as error:
        manifest.update(status='failed_preparation', error=repr(error)); raise
    finally:
        write(root / 'manifest.json', manifest); seal(root)


def verify_case_inventory(folder):
    for name, wanted in read(folder / 'artifact_hashes.json').items():
        if sha(folder / name) != wanted:
            raise ValueError('Recorded physical artifact changed: ' + name)


def run_case(root, index, replay=False):
    manifest = read(root / 'manifest.json')
    config = manifest['config']
    validate_manifest(manifest)
    check_sources(manifest)
    check_references(manifest)
    if manifest['status'] not in ('prepared', 'running', 'failed', 'complete') or len(manifest['cases']) != config['expected_unique_cases']:
        raise ValueError('A fully prepared ten-unique/twelve-declared probe is required')
    if not 0 <= index < len(manifest['cases']):
        raise ValueError('Case index must identify a unique prepared trajectory')
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
        if world.step_count == 0:
            initial = backend.scenes[0]['last_selection']
            if (digest(physical_path(initial['selected'])) != case['initial_path_sha256']
                    or json_value(initial['selected']) != case['initial_selected']
                    or json_value(initial['v22_first_choice']) != case['v22_first_choice']
                    or action != case['initial_guard']['first_authorized_action']):
                raise ValueError('Prepared backend first choice changed before execution')
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
        runtime.close_sensor_episode(0, reason='early_coverage_screen_terminated')
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
        task='six_facility_early_coverage_v22_information_screen',
        first_option=case['first_option'], continuation_mode='N',
        initial_option_outcome=initial_option_outcome(case, runtime.audit, actions),
        planning_coverage_proxy=config['planning_coverage_proxy'],
        sensor_calibration_scope=world.sensor_metadata,
        primitive_budget_compliant=world.step_count <= case['budget'],
        initial_history_actions=0, declared_first_option_installed=False,
        semantic_information_test=True, semantic_information_screen=True,
        semantic_information_gain_proven=False, semantic_policy_deployed=False,
        strong_G_comparison_performed=False, trained=False))
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
            reference_initialization_and_final_evaluation_excluded=True,
            timing_is_not_robot_speed_comparison=True, parallel_case_workers=1))
        np.savez_compressed(folder / 'final_mesh.npz',
            **{key: np.asarray(getattr(mesh, key)) for key in ('vertices', 'triangles', 'vertex_colors')})
    seal(folder)
    check_sources(manifest)
    check_references(manifest)
    final = result['after'][str(primary)]
    print(f'case={index} mode={case["mode"]} replay={replay} complete; '
          f'actions={world.step_count}; C={final["coverage_2d"]:.6f}; eligible={final["eligible"]}', flush=True)


def aggregate(root, manifest):
    validate_manifest(manifest)
    rows = []
    for case in manifest['cases']:
        folder = root / f'case_{case["index"]:02d}'
        verify_case_inventory(folder)
        result = read(folder / 'result.json'); verification = read(folder / 'verification.json')
        if (verification.get('status') != 'passed' or not verification.get('independent_process')
                or not verification.get('fresh_world_sensor_action_metric_replay')
                or verification.get('physical_process_id') == verification.get('replay_process_id')
                or verification.get('actions') != result['paid_actions'] or result['case'] != case):
            raise ValueError('Unique case and fresh replay contract differ')
        final = result['after'][str(manifest['config']['primary_reference_seed'])]
        rows.append(dict(index=case['index'], parent=case['parent'], assignment=case['assignment'],
            first_option=case['first_option'], continuation_mode='N', budget=case['budget'],
            paid_actions=result['paid_actions'], coverage_2d=final['coverage_2d'],
            external_macro_f1=final['05cm']['external_macro_f1'], joint_external=final['05cm']['joint_external'],
            eligible=final['eligible'], returned_to_anchor=result['termination']['returned_to_anchor'],
            collisions=result['collisions'], failed=result['termination']['failed'],
            primitive_budget_compliant=result['primitive_budget_compliant'],
            initial_option_outcome=result['initial_option_outcome'],
            replay_status=verification['status'], replay_actions=verification['actions']))
    declared = [dict(row, endpoint=rows[row['unique_case_index']]) for row in manifest['declared_cases']]
    summary = dict(status='complete', cases=rows, declared_cases=declared,
        declared_to_unique=manifest['declared_to_unique'], continuation_mode='N',
        unique_physical_trajectories=len(rows), declared_option_cases=len(declared),
        independent_process_replays=len(rows),
        total_physical_actions=sum(row['paid_actions'] for row in rows),
        total_replay_actions=sum(row['replay_actions'] for row in rows),
        eligible_unique_count=sum(row['eligible'] for row in rows),
        eligible_declared_count=sum(row['endpoint']['eligible'] for row in declared),
        common_online_coverage_passed=all(row['eligible'] and row['primitive_budget_compliant'] for row in rows),
        reference_sampling_repeats_are_independent_experiments=False,
        aliases_are_independent_experiments=False, old_N_metric_reuse_allowed=False,
        semantic_information_test=True, semantic_information_screen=True,
        semantic_information_gain_proven=False, semantic_policy_deployed=False,
        strong_G_comparison_performed=False, training_allowed=False, full_architecture_efficacy_proven=False)
    write(root / 'result.json', summary)
    return summary


def run_all(root):
    manifest = read(root / 'manifest.json')
    validate_manifest(manifest)
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
    parser.add_argument('--replay', action='store_true', help='Fresh process replay of --case')
    args = parser.parse_args()
    if args.replay and args.case is None:
        parser.error('--replay requires --case')
    root = args.output.resolve()
    if args.prepare is not None:
        prepare(root, args.prepare)
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
