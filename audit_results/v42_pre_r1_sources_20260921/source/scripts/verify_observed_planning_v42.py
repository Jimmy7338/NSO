#!/usr/bin/env python3
"""Bounded analytic observation -> forecast -> routing and measured TSDF checks.

No development World, future actual sensor query, or held-out asset is used.
The two-packet prefix is a scripted analytic fixture, not an autonomous run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.development_sensor_v41 import camera_transform_xyyaw, runtime_counts_v41, storage_report_v41
from nso.analytic_fixture_v42 import ANALYTIC_MARKER_COLOR_V42, analytic_forward_sequence_v42
from nso.observed_instances_v41 import ObservedInstancesV41
from nso.observed_residual_v41 import ObservedResidualV41
from nso.observed_mapper_v42 import ObservedMapperV42
from nso.view_quality_v42 import ViewQualityPredictorV42
from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41, ReturnAwarePrimitiveRouterV41
from nso.surface_evaluation_v40 import CandidateViewV40


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def source_files():
    names = ['nso/instance_belief_v40.py', 'nso/observed_instances_v41.py',
             'nso/observed_residual_v41.py', 'nso/primitive_navigation_v41.py',
             'nso/development_geometry_v40.py', 'nso/surface_evaluation_v40.py',
             'env/development_sensor_v41.py', 'utils/rgbd_contract.py',
             'configs/virtual3d/v40_scene_protocol_20260920.json',
             'configs/virtual3d/v42_analytic_interface_protocol_20260921.json',
             'scripts/verify_observed_planning_v42.py']
    for folder, pattern in [('nso', '*v42.py'), ('tests', 'test_*v42.py')]:
        names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/folder).glob(pattern))]
    return sorted(set(names))


def old_seals():
    result = []
    for name in ['v40_p0_20260920', 'v40_p1_static_release_20260920', 'v41_interface_release_20260920']:
        manifest = ROOT/'audit_results'/name/'manifest.json'
        data = json.loads(manifest.read_text())
        rows = data['files']
        assert isinstance(rows, dict), 'unexpected old seal format'
        mismatches = [p for p, entry in rows.items() if sha(ROOT/p) != (entry['sha256'] if isinstance(entry, dict) else entry)]
        if mismatches:
            raise AssertionError(f'old sealed evidence changed: {mismatches}')
        result.append(dict(manifest=str(manifest.relative_to(ROOT)), manifest_sha256=sha(manifest),
                           files_checked=len(rows), mismatches=0))
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = ROOT/'configs/virtual3d/v42_analytic_interface_protocol_20260921.json'
    protocol = json.loads(protocol_path.read_text())
    sources = {p: sha(ROOT/p) for p in source_files()}
    write(output/'started.json', dict(scope=protocol['scope'], source_sha256=sources,
        protocol_sha256=sha(protocol_path), actual_development_worlds_authorized=0,
        held_out_scene_assets_authorized=0, maximum_output_bytes=protocol['maximum_output_bytes']))
    sealed_before = old_seals()
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*v42.py', '-v']
    start = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300,
        env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1'))
    log = completed.stdout+completed.stderr
    (output/'unittest.txt').write_text(log)
    counts = re.findall(r'Ran (\d+) tests?', log)
    tests = dict(exit_code=completed.returncode, tests_run=int(counts[-1]) if counts else None,
        elapsed_seconds=time.monotonic()-start, log_sha256=sha(output/'unittest.txt'), command=command)
    write(output/'test_result.json', tests)
    if completed.returncode:
        raise RuntimeError('analytic contract tests failed; retained output must not be overwritten')

    base = json.loads((ROOT/'configs/virtual3d/v40_scene_protocol_20260920.json').read_text())['public_defaults']
    structure_names = base['structure_prior']['abstract_structures']
    prior = base['structure_prior']['probability_by_category']['cabinet']
    priors = {'G': prior, 'S': prior, 'flat': protocol['flat_prior'],
              'permuted': [prior[i] for i in protocol['permutation']]}
    ledgers = {name: ObservedInstancesV41(palette={'cabinet': ANALYTIC_MARKER_COLOR_V42},
        structure_names=structure_names, class_structure_prior={'cabinet': class_prior},
        mode='G' if name == 'G' else 'S') for name, class_prior in priors.items()}
    predictor = ViewQualityPredictorV42()
    residual = ObservedResidualV41()
    mapper = ObservedMapperV42(**protocol['mapper'])
    observations = analytic_forward_sequence_v42()
    graph = PublicPrimitiveGraphV41(protocol['public_navigation'])
    states = [PrimitiveStateV41(**row) for row in protocol['candidate_states']]
    candidate_ids = [f'{state.node}/{state.heading}' for state in states]
    candidates = [CandidateViewV40(intrinsic=observations[0].intrinsic,
        world_from_camera=camera_transform_xyyaw((*graph.positions[state.node], state.heading*np.pi/6)),
        width=observations[0].depth_m.shape[1], height=observations[0].depth_m.shape[0], view_id=view_id)
        for state, view_id in zip(states, candidate_ids)]
    write(output/'candidate_views.json', [dict(view_id=view.view_id, state=state.__dict__,
        intrinsic=view.intrinsic.tolist(), world_from_camera=view.world_from_camera.tolist(),
        width=view.width, height=view.height, near_m=view.near_m, far_m=view.far_m)
        for view, state in zip(candidates, states)])
    traces = []
    for observation in observations:
        np.savez_compressed(output/f'analytic_packet_{observation.paid_step:03d}.npz',
            frame_id=observation.frame_id, paid_step=observation.paid_step, rgb=observation.rgb,
            depth_m=observation.depth_m, intrinsic=observation.intrinsic,
            world_from_camera=observation.world_from_camera)
        associations = {name: ledger.observe(observation) for name, ledger in ledgers.items()}
        for value in associations.values():
            # Public association receipts also contain the condition-specific
            # belief snapshot; compare the actual shared geometric inputs.
            for key in ('accepted', 'rejected', 'payload_sha256'):
                assert value[key] == associations['G'][key]
        evidence = residual.observe(observation, associations['G']['accepted'])
        prediction_ingest = predictor.observe(observation, associations['G']['accepted'])
        map_receipt = mapper.update(observation)
        conditions = {}
        for name, ledger in ledgers.items():
            feedback = []
            for item in evidence['results']:
                if item['accepted']:
                    applied = ledger.apply_geometry_feedback(item['instance_id'], frame_id=item['frame_id'],
                        observation_sha256=item['observation_sha256'], log_likelihoods=item['log_likelihoods'])
                    feedback.append({key: value for key, value in applied.items() if key != 'instance'})
            snapshot = ledger.snapshot()
            assert len(snapshot['instances']) == 1
            forecast = predictor.forecast(snapshot['instances'][0], candidates)
            conditions[name] = dict(feedback=feedback, snapshot=snapshot,
                geometry_sha256=digest(ledger.geometry_snapshot()), forecast=forecast)
        assert len({row['geometry_sha256'] for row in conditions.values()}) == 1
        traces.append(dict(paid_step=observation.paid_step, frame_id=observation.frame_id,
            observation_sha256=observation.sha256(), association=associations['G'], residual=evidence,
            predictor_ingest=prediction_ingest, mapper_receipt=map_receipt, conditions=conditions))
        write(output/f'observed_pipeline_{observation.paid_step:03d}.json', traces[-1])

    # All methods receive the same two saved packets. This prefix is an
    # explicitly scripted fixture, never counted as a method's chosen route.
    # The final submitted action does use the measured-data predictor output.
    routing = {}
    areas = {}
    for name, condition in traces[-1]['conditions'].items():
        rows = condition['forecast']['candidates']
        values = {row['view_id']: row['expected_new_surface_area_m2'] for row in rows}
        assert set(values) == set(candidate_ids)
        areas[name] = [values[view_id] for view_id in candidate_ids]
        router = ReturnAwarePrimitiveRouterV41(PublicPrimitiveGraphV41(protocol['public_navigation']),
            home=PrimitiveStateV41('home', 0), budget=protocol['router_budget'])
        router.accept(observations[0])
        priming = router.choose({PrimitiveStateV41('advance', 0): 1.})
        assert priming['action'] == 'forward'
        router.accept(observations[1])
        decision = router.choose({state: values[view_id] for state, view_id in zip(states, candidate_ids)})
        routing[name] = dict(prefix_source=protocol['routing_prefix_source'], prefix_decision=priming,
            forecast_sha256=digest(condition['forecast']), decision=decision,
            candidate_action_executed=False, policy_rollout=False)
    np.testing.assert_allclose(areas['G'], areas['flat'], rtol=0, atol=1e-12)
    first = traces[0]['conditions']
    for name in conditions:
        assert not first[name]['snapshot']['instances'][0]['semantic_conditioning_used']
    assert traces[1]['conditions']['S']['snapshot']['instances'][0]['semantic_conditioning_used']
    assert not traces[1]['conditions']['G']['snapshot']['instances'][0]['semantic_conditioning_used']
    mesh = mapper.mesh_arrays()
    np.savez_compressed(output/'analytic_tsdf_mesh.npz', **mesh)
    belief, visible = mapper.occupancy_arrays()
    np.savez_compressed(output/'analytic_occupancy.npz', belief=belief, visible=visible)
    write(output/'mapper_snapshot.json', mapper.snapshot())
    write(output/'routing.json', routing)
    write(output/'observed_planes.json', predictor.observed_plane_receipts())
    resource = storage_report_v41(output, 64*1024**2)
    runtime = runtime_counts_v41()
    assert runtime['worlds_created'] == 0 and runtime['paid_actions'] == 0
    write(output/'resource_gate.json', dict(report=resource, runtime=runtime,
        physical_world_factory_called=False, analytic_TSDF_not_a_development_rollout=True))
    assert sealed_before == old_seals()
    assert sources == {p: sha(ROOT/p) for p in sources}, 'source changed during verification'
    result = dict(status='passed', scope=protocol['scope'], contract_tests=tests['tests_run'],
        analytic_unique_paid_packets=len(observations), analytic_mapper_update_calls=len(observations),
        analytic_forecasts=len(observations)*len(ledgers), counterfactual_conditions=list(ledgers),
        s_g_shared_geometry_and_residual_equal=True, flat_prior_matches_geometry_utilities=True,
        semantic_conditioning_active_after_second_packet=True, candidate_view_ids=candidate_ids,
        final_posteriors={name: row['snapshot']['instances'][0]['structure_probabilities'] for name, row in conditions.items()},
        candidate_expected_new_surface_area_m2=areas,
        maximum_semantic_proxy_difference_m2=float(np.max(np.abs(np.array(areas['S'])-areas['G']))),
        routed_final_actions={name: row['decision']['action'] for name, row in routing.items()},
        routed_final_targets={name: row['decision'].get('target') for name, row in routing.items()},
        forecast_selected_actions_executed=0, analytic_mesh_vertices=len(mesh['vertices']),
        analytic_mesh_triangles=len(mesh['triangles']), actual_development_worlds=0,
        actual_development_sensor_trajectories=0, actual_development_planner_rollouts=0,
        actual_development_TSDF_integrations=0, semantic_performance_claim_added=False,
        full_four_module_rollout_complete=False, physical_batch_storage_ready=resource['passed'],
        no_scan_in_analytic_fixture=True, true_coverage_fraction_computed=False,
        prototype_surface_fused_into_map=False, old_seals=sealed_before,
        source_sha256=sources, source_unchanged_during_verification=True)
    write(output/'result.json', result)
    write(output/'artifact_sha256.json', {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(output.iterdir())})
    total = sum(p.stat().st_size for p in output.iterdir())
    if total > protocol['maximum_output_bytes']:
        raise RuntimeError('output exceeded declared cap; retained artifacts must be investigated')
    print(json.dumps({key: result[key] for key in ('status', 'contract_tests',
        'maximum_semantic_proxy_difference_m2', 'routed_final_actions', 'analytic_mesh_triangles',
        'actual_development_worlds', 'physical_batch_storage_ready')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'audit_results/v42_observed_planning_20260921')
    run(parser.parse_args().output.resolve())
