#!/usr/bin/env python3
"""Seal R0/R1 analytic interface checks; never start a development World."""
import argparse
import hashlib
import importlib.util
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
from nso.instance_belief_v40 import PaidRGBDObservationV40
from nso.observed_instances_v41 import ObservedInstancesV41
from nso.observed_residual_v41 import ObservedResidualV41
from nso.primitive_navigation_v41 import PublicPrimitiveGraphV41, PrimitiveStateV41, ReturnAwarePrimitiveRouterV41
from env.development_sensor_v41 import storage_report_v41, create_development_sensor, ResourceGateBlocked, runtime_counts_v41


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    names = ['nso/observed_instances_v41.py', 'nso/observed_residual_v41.py',
             'nso/primitive_navigation_v41.py', 'env/development_sensor_v41.py',
             'scripts/verify_runtime_interfaces_v41.py', 'nso/instance_belief_v40.py',
             'nso/development_geometry_v40.py', 'configs/virtual3d/v40_scene_protocol_20260920.json']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_*v41.py'))]
    before = {name: sha(ROOT/name) for name in names}
    write(output/'started.json', dict(scope='R0/R1 analytic fixtures only', source_sha256=before,
                                     new_worlds_authorized=0, new_development_trajectory_starts_authorized=0))
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*v41.py', '-v']
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=180)
    log = completed.stdout + completed.stderr
    (output/'unittest.txt').write_text(log)
    counts = re.findall(r'Ran (\d+) tests?', log)
    test_result = dict(exit_code=completed.returncode, tests_run=int(counts[-1]) if counts else None,
                       elapsed_seconds=time.monotonic()-started, log_sha256=sha(output/'unittest.txt'), command=command)
    write(output/'test_result.json', test_result)
    if completed.returncode:
        raise RuntimeError('contract tests failed; output is retained')

    # This fixture consists of independently specified measured planes. It is
    # not a DEV asset rollout, a reconstruction, or a planning efficacy trial.
    fixture_path = ROOT/'tests/test_observed_residual_v41.py'
    spec = importlib.util.spec_from_file_location('v41_analytic_fixture', fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    original = fixture.fixture()[0]
    transform = original.world_from_camera.copy()
    transform[2, 3] += .1  # Translate the entire analytic world to camera z=.9.
    paid = PaidRGBDObservationV40('analytic-interface-0', 0, original.rgb, original.depth_m,
                                  original.intrinsic, transform)
    np.savez_compressed(output/'analytic_observation.npz', rgb=paid.rgb, depth_m=paid.depth_m,
                        intrinsic=paid.intrinsic, world_from_camera=paid.world_from_camera)
    rows, geometries = {}, []
    for mode in ('G', 'S'):
        ledger = ObservedInstancesV41(palette={'cabinet': fixture.COLOR}, structure_names=fixture.NAMES,
            class_structure_prior={'cabinet': [.4, .3, .2, .1]}, mode=mode)
        associations = ledger.observe(paid)
        assert len(associations['accepted']) == 1
        residual = ObservedResidualV41().observe(paid, associations['accepted'])
        evidence = residual['results'][0]
        assert evidence['accepted'] and evidence['informative']
        feedback = ledger.apply_geometry_feedback(evidence['instance_id'], frame_id=evidence['frame_id'],
            observation_sha256=evidence['observation_sha256'], log_likelihoods=evidence['log_likelihoods'])
        assert feedback['applied']
        geometries.append(ledger.geometry_snapshot())
        rows[mode] = dict(association=associations, residual=residual, feedback=feedback, snapshot=ledger.snapshot())
        write(output/f'{mode}_observed_feedback.json', rows[mode])
    assert geometries[0] == geometries[1]
    assert rows['G']['residual']['results'][0]['log_likelihoods'] == rows['S']['residual']['results'][0]['log_likelihoods']

    x, y = transform[:2, 3]
    graph = PublicPrimitiveGraphV41({'schema_version': 'v41.public_navigation.v1',
        'source_kind': 'provided_navigation_prior', 'nodes': {'home': [float(x), float(y)],
        'goal': [float(x), float(y+1.)]}, 'edges': [['home', 'goal']]})
    router = ReturnAwarePrimitiveRouterV41(graph, home=PrimitiveStateV41('home', 3), budget=21)
    router.accept(paid)
    # The routing fixture supplies a utility. No semantic forecast is claimed.
    decision = router.choose({PrimitiveStateV41('goal', 3): 1.})
    assert decision['action'] == 'forward' and decision['fixed_prefix_actions'] == 0
    assert decision['candidates'][0]['total_with_observation_and_return'] == 21
    write(output/'analytic_action0_routing.json', dict(decision=decision,
        candidate_utility_is_synthetic=True, primitive_executed=False,
        semantic_view_quality_predictor_implemented=False))

    resource = storage_report_v41(output, 64*1024**2)
    counts_before = runtime_counts_v41()
    if not resource['passed']:
        # Confirm the real factory fails before reading the supplied nonexistent
        # asset path. A ready system is deliberately not instantiated in R1.
        try:
            create_development_sensor(ROOT/'not_a_development_asset', {}, episode_id='gate-probe',
                noise_seed=0, persistent_output_root=output, expected_batch_peak_bytes=64*1024**2)
        except ResourceGateBlocked as error:
            resource = error.report
        else:
            raise AssertionError('resource gate failed to block before asset access')
    counts_after = runtime_counts_v41()
    assert counts_after['worlds_created'] == counts_before['worlds_created'] == 0
    write(output/'resource_gate.json', dict(report=resource, runtime_before=counts_before, runtime_after=counts_after,
        actual_asset_read_by_gate_probe=False, actual_development_observation_requested=False))
    after = {name: sha(ROOT/name) for name in names}
    assert before == after, 'source changed during verification'
    result = dict(status='passed', phase='V41_R0_R1_interfaces', contract_tests=test_result['tests_run'],
        analytic_paid_observations_created=1, analytic_ledger_observe_calls=2,
        analytic_residual_calls=2, analytic_feedback_updates=2, analytic_router_decisions=1,
        s_g_geometry_and_residual_equal=True, actual_development_worlds=0, actual_sensor_trajectories=0,
        actual_planner_rollouts=0, actual_TSDF_integrations=0, semantic_performance_claim_added=False,
        semantic_view_quality_predictor_implemented=False, full_four_module_rollout_complete=False,
        physical_batch_storage_ready=resource['passed'], source_unchanged_during_verification=True,
        source_sha256=before, analytic_observation_sha256=paid.sha256(),
        structure_names=list(fixture.NAMES), actual_pixel_residual_losses_m=rows['G']['residual']['results'][0]['losses_m'],
        actual_pixel_log_likelihoods=rows['G']['residual']['results'][0]['log_likelihoods'])
    write(output/'result.json', result)
    write(output/'artifact_sha256.json', {p.name: sha(p) for p in sorted(output.iterdir())})
    print(json.dumps({key: result[key] for key in ('status', 'contract_tests', 's_g_geometry_and_residual_equal',
        'actual_development_worlds', 'physical_batch_storage_ready', 'actual_pixel_residual_losses_m')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'audit_results/v41_runtime_interfaces_20260920')
    run(parser.parse_args().output.resolve())
