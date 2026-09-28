"""Two new-phase pilots, retaining the unmodified V43 planner and physical loop.

This entry point deliberately has its own protocol and artifact closure. It does
not pretend to be an old five-slot episode to pass the frozen V44 verifier.
"""
import gzip
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from env.development_sensor_v41 import (SensorStepV41, create_development_sensor,
    runtime_counts_v41, storage_report_v41)
from nso.episode_driver_v43 import (ACTION_TO_SENSOR_V43, DevelopmentStartLedgerV43,
    canonical_bytes, execute_episode_v43, file_sha256, validate_step_v43)
from nso.evidence_writer_v44 import CompressedStepWriterV44

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/virtual3d/mechanism_development_20260923.json'
DEFAULT_OUTPUT = ROOT/'audit_results/mechanism_development_20260923/episodes'
COMPLETE = {'controller_stop', 'controller_blocked', 'budget_exhausted',
            'stopped_without_confirmed_return'}


def _json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError('nonfinite JSON: '+value)))


def source_names():
    from scripts.run_development_v44 import source_names_v44
    return sorted(set(source_names_v44()) | {CONFIG, 'nso/mechanism_development.py',
        'scripts/run_mechanism_development.py', 'nso/saved_replay_v44.py',
        'nso/offline_evaluation_v44.py', 'scripts/verify_development_surface_pipeline_v40.py'})


def _bundle(protocol, slot):
    from nso.public_navigation_v43 import load_public_navigation_bundle_v43
    return load_public_navigation_bundle_v43(ROOT/protocol['navigation_root']/slot['asset_id'],
        ROOT/protocol['asset_root']/slot['asset_id'], protocol_path=ROOT/protocol['scene_protocol_path'])


def _controller_and_mapper(protocol, slot, bundle):
    from nso.controller_v43 import ANSControllerV43
    from nso.observed_mapper_v42 import ObservedMapperV42
    public, workspace = bundle['public_spec'], bundle['workspace']
    prior = public['structure_prior']
    controller = ANSControllerV43(bundle['graph'], home=bundle['home_state'],
        budget=public['task']['max_actions'], palette=workspace['marker_palette'],
        structure_names=prior['abstract_structures'],
        class_structure_prior=prior['probability_by_category'], mode=slot['mode'], **protocol['controller'])
    bounds = workspace['bounds_xy_m']; origin = bounds[0]
    resolution = protocol['mapper']['resolution_m']
    shape = [math.ceil((bounds[1][1]-origin[1])/resolution),
             math.ceil((bounds[1][0]-origin[0])/resolution)]
    return controller, ObservedMapperV42(shape=shape, origin_xy_m=origin, **protocol['mapper'])


def run_pilot(run_id, *, output_root=DEFAULT_OUTPUT, preflight_only=False):
    protocol = _json(ROOT/CONFIG)
    if run_id not in protocol['slots']:
        raise ValueError('only the two explicitly declared cost pilots are authorized')
    output = Path(output_root).resolve()/run_id
    resource = storage_report_v41(output, protocol['expected_batch_peak_bytes'])
    if not resource['passed']:
        return dict(status='blocked_before_world_creation', resource=resource, start_slot_reserved=False,
                    runtime=runtime_counts_v41())
    slot = protocol['slots'][run_id]
    bundle = _bundle(protocol, slot)
    if bundle['public_spec']['task']['max_actions'] != 160:
        raise ValueError('pilot retains the original 160-action task budget')
    # Pin the pre-existing evaluation reference before any sensor creation.
    from nso.offline_evaluation_v44 import load_reference_v44
    load_reference_v44(ROOT/protocol['reference_root'],
        manifest_sha256=protocol['reference_manifest_sha256'], asset_id=slot['asset_id'])
    import open3d as o3d
    if preflight_only:
        return dict(status='ready_without_world_creation', run_id=run_id, resource=resource,
            backend='Open3D '+o3d.__version__, start_slot_reserved=False,
            old_ledgers_used=False, runtime=runtime_counts_v41())
    if output.exists():
        raise FileExistsError('retained pilot output cannot be overwritten')
    sources = {name: file_sha256(ROOT/name) for name in source_names()}
    ledger = DevelopmentStartLedgerV43(ROOT/protocol['ledger_relative_path'],
                                     maximum_slots=protocol['maximum_new_world_slots'])
    metadata = dict(run_id=run_id, slot=slot, output=str(output),
        protocol_path=CONFIG, protocol_sha256=sources[CONFIG], source_sha256=sources,
        public_graph_sha256=bundle['graph'].input_sha256,
        new_phase_ledger=True, old_five_slot_episode=False)
    ledger.reserve(run_id, metadata)
    output.mkdir(parents=True, exist_ok=False)
    writer = CompressedStepWriterV44(output, maximum_bytes=protocol['maximum_episode_bytes'],
        maximum_file_bytes=protocol['maximum_file_bytes'],
        terminal_reserve_bytes=protocol['terminal_record_reserve_bytes'])
    sensor = None; world_created = False; before = runtime_counts_v41()
    try:
        writer.json('started.json', dict(metadata, resource=resource,
            task_kind='cost_observation_chain_pilot', exact_pose_model=True,
            known_coarse_navigation_prior=True, strong_geometry_control=False,
            new_candidate_mechanism_evaluated=False, semantic_performance_claim=False))
        for name, sha in sources.items():
            writer._write('source/'+name, (ROOT/name).read_bytes())
            if file_sha256(output/'source'/name) != sha:
                raise RuntimeError('source changed during snapshot')
        for name, key in (('public_graph.json', 'graph_spec'), ('public_spec.json', 'public_spec'),
                          ('public_workspace.json', 'workspace')):
            writer.json(name, bundle[key])
        controller, mapper = _controller_and_mapper(protocol, slot, bundle)
        sensor = create_development_sensor(ROOT/protocol['asset_root']/slot['asset_id'],
            bundle['public_spec'], episode_id=run_id, noise_seed=slot['noise_seed'],
            persistent_output_root=output, expected_batch_peak_bytes=protocol['expected_batch_peak_bytes'])
        world_created = True
        result = execute_episode_v43(sensor, controller, mapper, writer, budget=160,
                                    maximum_elapsed_s=protocol['maximum_elapsed_s'])
        changed = [name for name, sha in sources.items() if file_sha256(ROOT/name) != sha]
        if changed:
            raise RuntimeError('source changed during pilot: '+','.join(changed))
        writer.json('encoding.json', dict(schema='v44.step_encoding.v1', steps=writer.step_encoding), terminal=True)
        writer.json('runtime.json', dict(before=before, after=runtime_counts_v41(), world_created=True,
            source_unchanged=True, evaluation_executed=False, old_ledgers_used=False), terminal=True)
        writer.json('artifact_manifest.json', dict(schema='mechanism.pilot.artifacts.v1',
            files=dict(writer.files), source_sha256=sources, protocol_path=CONFIG), terminal=True)
        ledger.finish(run_id, status=result['status'], world_created=True,
                      result_sha256=file_sha256(output/'result.json'))
        return dict(status=result['status'], run_id=run_id, output=str(output),
            executed_paid_actions=result['executed_paid_actions'], elapsed_s=result['elapsed_s'],
            artifact_manifest_sha256=file_sha256(output/'artifact_manifest.json'),
            artifact_bytes=writer.bytes_written, world_created=True, primary_experiment_started=False)
    except Exception as exc:
        close_error = None
        if sensor is not None:
            try:
                sensor.close()
            except Exception as close_exc:
                close_error=str(close_exc)
        failure = dict(status='pilot_attempt_failed', run_id=run_id, world_created=world_created,
            type=type(exc).__name__, message=str(exc), close_error=close_error, automatic_retry=False)
        writer.json('attempt_failure.json', failure, terminal=True)
        ledger.finish(run_id, status=failure['status'], world_created=world_created,
                      result_sha256=file_sha256(output/'attempt_failure.json'))
        return failure


def inspect_pilot(root, expected_manifest_sha256):
    root = Path(root).resolve()
    if not re.fullmatch('[0-9a-f]{64}', expected_manifest_sha256 or ''):
        raise ValueError('externally pinned manifest SHA256 required')
    if file_sha256(root/'artifact_manifest.json') != expected_manifest_sha256:
        raise ValueError('episode manifest differs from external pin')
    manifest = _json(root/'artifact_manifest.json')
    if manifest.get('schema') != 'mechanism.pilot.artifacts.v1' or manifest.get('protocol_path') != CONFIG:
        raise ValueError('new pilot artifact schema required; old episode aliases are forbidden')
    files = manifest['files']
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if actual != set(files)|{'artifact_manifest.json'}:
        raise ValueError('episode inventory mismatch')
    for name, row in files.items():
        path = root/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('plain bounded episode files required')
        if path.stat().st_size != row['bytes'] or file_sha256(path) != row['sha256']:
            raise ValueError('episode artifact changed: '+name)
    sources = manifest['source_sha256']
    if set(sources) != set(source_names()):
        raise ValueError('incomplete new-phase source closure')
    for name, sha in sources.items():
        if file_sha256(ROOT/name) != sha or file_sha256(root/'source'/name) != sha:
            raise ValueError('source differs from executed archive: '+name)
    protocol = _json(root/'source'/CONFIG)
    started, result, runtime = (_json(root/name) for name in ('started.json', 'result.json', 'runtime.json'))
    slot = protocol['slots'].get(started['run_id'])
    if (slot != started['slot'] or started['source_sha256'] != sources
            or started['protocol_sha256'] != sources[CONFIG]
            or runtime.get('world_created') is not True or runtime.get('source_unchanged') is not True):
        raise ValueError('pilot source/protocol/runtime binding mismatch')
    ledger = _json(ROOT/protocol['ledger_relative_path'])
    entries = [row for row in ledger['entries'] if row['run_id'] == started['run_id']]
    if (len(entries) != 1 or entries[0]['status'] != result['status']
            or entries[0]['result_sha256'] != file_sha256(root/'result.json')
            or entries[0]['world_created'] is not True
            or entries[0]['metadata']['source_sha256'] != sources
            or entries[0]['metadata']['output'] != str(root)):
        raise ValueError('new-phase reservation does not bind this complete result')
    if result['status'] not in COMPLETE or result['error'] is not None or result['finalization_errors']:
        raise ValueError('failed or incomplete pilot retained; no successful endpoint scoring')
    expected_counts = dict(worlds_created=1, rgbd_frames=result['acquired_and_saved_packets'],
                           scans=result['acquired_and_saved_packets'],
                           paid_actions=result['executed_paid_actions'], blocked_before_world_creation=0)
    if any(runtime['after'][key]-runtime['before'][key] != value
           for key, value in expected_counts.items()):
        raise ValueError('live sensor counters disagree with complete saved trajectory')
    return protocol, slot, started, result, manifest


def _replay(root, protocol, slot, result, manifest):
    from nso.instance_belief_v40 import PaidRGBDObservationV40
    from nso.saved_replay_v44 import _arrays, _canonical_mesh, _first_difference
    from utils.rgbd_contract import PlanarScan
    bundle = _bundle(protocol, slot)
    for name, key in (('public_graph.json', 'graph_spec'), ('public_spec.json', 'public_spec'),
                      ('public_workspace.json', 'workspace')):
        if _json(root/name) != bundle[key]:
            raise ValueError('saved shared public bundle differs: '+name)
    count = result['acquired_and_saved_packets']
    if (type(count) is not int or not 1 <= count <= 161
            or any(result[key] != count-1 for key in ('executed_paid_actions', 'submitted_paid_actions'))
            or result['received_sensor_packets'] != count or result['mapper_frames'] != count):
        raise ValueError('incomplete paid packet accounting')
    controller, mapper = _controller_and_mapper(protocol, slot, bundle)
    encodings = _json(root/'encoding.json')['steps']
    if len(encodings) != count:
        raise ValueError('complete compressed step ledger required')
    previous_pose = None; previous_decision = None; frame_ids = set(); actions = []; submissions = []; collisions = 0
    for index in range(count):
        prefix = f'packets/{index:03d}'
        values = _arrays(root/(prefix+'_rgbd.npz'), PaidRGBDObservationV40.__dataclass_fields__)
        for name in ('frame_id', 'paid_step'):
            values[name] = values[name].item()
        rgbd = PaidRGBDObservationV40.from_mapping(values)
        scan_values = _arrays(root/(prefix+'_scan.npz'), PlanarScan.__dataclass_fields__)
        for name in set(scan_values)-{'ranges_m', 'world_from_laser'}:
            scan_values[name] = scan_values[name].item()
        scan = PlanarScan(**scan_values)
        receipt = _json(root/(prefix+'_receipt.json'))
        if (rgbd.frame_id in frame_ids or receipt['observation_sha256'] != rgbd.sha256()
                or receipt['rgbd_artifact'] != manifest['files'][prefix+'_rgbd.npz']
                or receipt['scan_artifact'] != manifest['files'][prefix+'_scan.npz']):
            raise ValueError('duplicate frame or saved packet identity mismatch')
        frame_ids.add(rgbd.frame_id)
        step_name = f'steps/{index:03d}.json.gz'; encoding = encodings[index]
        if (encoding['artifact'] != step_name or encoding['stored_bytes'] != manifest['files'][step_name]['bytes']
                or not 0 < encoding['uncompressed_bytes'] <= protocol['maximum_file_bytes']):
            raise ValueError('step compression ledger mismatch')
        with gzip.open(root/step_name, 'rb') as stream:
            content = stream.read(protocol['maximum_file_bytes']+1)
        if len(content) != encoding['uncompressed_bytes']:
            raise ValueError('step decompression length mismatch')
        record = json.loads(content)
        chosen = None if index == 0 else previous_decision['action']
        action = 'initial_observation' if index == 0 else ACTION_TO_SENSOR_V43[chosen]
        accounting = validate_step_v43(SensorStepV41(rgbd, scan, receipt['execution']), expected_step=index,
            expected_action=action, expected_previous_pose=previous_pose)
        mapped = mapper.update(rgbd, scan)
        evidence = controller.accept(rgbd, mapper,
            execution_outcome='collision' if accounting['collision'] else 'success')
        decision = controller.choose()
        for name, actual in (('accounting', accounting), ('mapper', mapped),
                             ('controller_evidence', evidence), ('decision', decision)):
            difference = _first_difference(record[name], actual, name)
            if difference:
                raise ValueError(f'saved observation replay diverged at step {index}: {difference}')
        if index:
            actions.append(dict(paid_step=index, controller_action=chosen, sensor_action=action,
                                observation_sha256=rgbd.sha256(), collision=accounting['collision']))
            submissions.append(dict(expected_paid_step=index, controller_action=chosen, sensor_action=action))
        collisions += int(accounting['collision'])
        previous_pose = receipt['execution']['pose_xyyaw_rad']; previous_decision = decision
    if actions != result['actions'] or submissions != result['submitted_action_records'] or collisions != result['collisions']:
        raise ValueError('terminal action ledger differs from saved packets')
    terminal = {k: previous_decision[k] for k in ('action', 'reason', 'target', 'paid_step') if k in previous_decision}
    if terminal != result['last_decision']:
        raise ValueError('terminal decision differs')
    if ((result['status'] == 'controller_stop' and (previous_decision['action'] != 'stop'
            or result['sensor_status'].get('returned_xy_and_yaw') is not True))
            or (result['status'] == 'controller_blocked' and previous_decision['action'] != 'blocked')):
        raise ValueError('terminal status differs from actual final decision')
    if _first_difference(_json(root/'prediction/mapper.json'), mapper.snapshot()) is not None:
        raise ValueError('final mapper differs from saved-observation reconstruction')
    occupancy = _arrays(root/'prediction/occupancy.npz', ('belief', 'observed'))
    if any(not np.array_equal(occupancy[key], value) for key, value in zip(('belief', 'observed'), mapper.occupancy_arrays())):
        raise ValueError('final occupancy differs')
    expected_mesh = _canonical_mesh(_arrays(root/'prediction/mesh.npz', ('vertices', 'triangles', 'vertex_colors')))
    actual_mesh = _canonical_mesh(mapper.mesh_arrays())
    if any(a.shape != b.shape or not np.allclose(a, b, atol=1e-9, rtol=0) for a, b in zip(expected_mesh, actual_mesh)):
        raise ValueError('final mesh differs')
    return dict(status='verified', verification_kind='saved_observation_policy_and_TSDF_replay',
        frames_verified=count, prediction_verified=True, occupancy_exact=True,
        mesh_order_invariant_tolerance_m=1e-9, new_worlds=0, physical_actions=0,
        counterfactual_trajectory=False)


def _evaluate(root, protocol, slot, result):
    from nso.offline_evaluation_v44 import load_reference_v44, measure_navigation_coverage_v44
    from nso.saved_replay_v44 import _arrays
    from nso.surface_evaluation_v40 import evaluate_surface_v40
    reference, domain, record = load_reference_v44(ROOT/protocol['reference_root'],
        manifest_sha256=protocol['reference_manifest_sha256'], asset_id=slot['asset_id'])
    mapper = _json(root/'prediction/mapper.json'); descriptor = record['coverage']
    for key in ('shape', 'resolution_m', 'origin_xy_m', 'grid_convention'):
        if mapper[key] != descriptor[key]:
            raise ValueError('fixed floor coverage coordinates differ from mapper')
    occupancy = _arrays(root/'prediction/occupancy.npz', ('belief', 'observed'))
    coverage = measure_navigation_coverage_v44(occupancy['belief'], domain, descriptor)
    mesh = _arrays(root/'prediction/mesh.npz', ('vertices', 'triangles', 'vertex_colors'))
    metrics = evaluate_surface_v40(reference, mesh['vertices'], mesh['triangles'], C_map=coverage['C_nav'],
        threshold_m=.05, sample_spacing_m=.3, seed=4002, max_samples=50000)
    metrics.pop('C_map'); metrics.pop('J')
    metrics.update(C_nav=coverage['C_nav'], J_nav=coverage['C_nav']*metrics['Q'])
    return dict(metrics=metrics, coverage=coverage, reference_manifest_sha256=protocol['reference_manifest_sha256'],
        task_success=result['status'] == 'controller_stop' and result['sensor_status'].get('returned_xy_and_yaw') is True,
        all_task_instances_in_macro_denominator=True, prediction_roi_cropped=False,
        original_episode_status=result['status'], new_worlds=0, semantic_performance_claim=False,
        scope='cost and observation-chain pilot, not new-candidate efficacy or a strong geometry comparison')


def review_pilot(episode, *, expected_manifest_sha256, output):
    root, output = Path(episode).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError('new review output outside immutable episode is required')
    started_s = time.monotonic()
    protocol, slot, started, result, manifest = inspect_pilot(root, expected_manifest_sha256)
    review = dict(schema='mechanism.pilot.review.v1', status='pilot_review_failed',
        run_id=started['run_id'], episode_manifest_sha256=expected_manifest_sha256,
        source_sha256=manifest['source_sha256'], primary_experiment_started=False,
        semantic_performance_claim=False, automatic_retry=False)
    runtime_before = runtime_counts_v41()
    try:
        review['replay'] = _replay(root, protocol, slot, result, manifest)
        review['evaluation'] = _evaluate(root, protocol, slot, result)
        inspect_pilot(root, expected_manifest_sha256)
        review['status'] = 'pilot_reviewed'
    except Exception as exc:
        review['error'] = dict(type=type(exc).__name__, message=str(exc))
    runtime_after = runtime_counts_v41()
    review['runtime'] = dict(before=runtime_before, after=runtime_after,
        no_new_world_or_sensor_action=runtime_before == runtime_after)
    if runtime_before != runtime_after:
        review['status'] = 'pilot_review_failed'
        review['runtime_error'] = 'saved-frame review unexpectedly queried a live sensor'
    review['elapsed_s'] = time.monotonic()-started_s
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(canonical_bytes(review))
    return review
