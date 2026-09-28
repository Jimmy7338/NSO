#!/usr/bin/env python3
"""Bounded static/reference and saved-packet verification; no study rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(output):
    import numpy as np
    from env.development_sensor_v41 import runtime_counts_v41
    from scripts.run_development_v44 import source_names_v44, run_development_v44
    from nso.offline_evaluation_v44 import (prepare_reference_v44, load_reference_v44,
        measure_navigation_coverage_v44, evaluate_saved_episode_v44)
    from nso.surface_evaluation_v40 import evaluate_surface_v40
    from v44_episode_fixture import create_finite_episode_v44
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    names = set(source_names_v44())
    names.add('nso/analytic_fixture_v42.py')
    names.add('scripts/verify_development_surface_pipeline_v40.py')
    for pattern in ('nso/*v44.py', 'scripts/*v44.py', 'tests/*v44.py', 'tests/v44_episode_fixture.py'):
        names.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
    sources = {name: sha(ROOT/name) for name in sorted(names)}
    for name in sources:
        target = output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    save(output/'started.json', dict(scope='static references, finite saved packets, independent fresh-process recomputation',
        source_sha256=sources, actual_development_worlds_authorized=0, maximum_output_bytes=16*1024**2))
    old = []
    for name in ('v40_p0_20260920', 'v40_p1_static_release_20260920',
                 'v41_interface_release_20260920', 'v42_interface_release_20260921',
                 'v43_interface_release_20260921'):
        manifest_path = ROOT/'audit_results'/name/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for relative, entry in manifest['files'].items():
            assert sha(ROOT/relative) == entry['sha256'], relative
        old.append(dict(manifest=str(manifest_path.relative_to(ROOT)), sha256=sha(manifest_path),
                        entries_checked=len(manifest['files'])))
    start = time.monotonic()
    tested = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                             '-p', 'test_*v44.py', '-v'], cwd=ROOT, capture_output=True, text=True)
    (output/'unittest.txt').write_text(tested.stdout+tested.stderr)
    save(output/'test_result.json', dict(returncode=tested.returncode, elapsed_s=time.monotonic()-start,
        performance_benchmark=False))
    if tested.returncode:
        raise RuntimeError('new V44 tests failed; retained output must not be overwritten')
    references = []
    for asset in ('DEV_A_00', 'DEV_C_00'):
        folder = output/'references'/asset
        prepared = prepare_reference_v44(asset, folder)
        ref, domain, record = load_reference_v44(folder, manifest_sha256=prepared['manifest_sha256'], asset_id=asset)
        coverage = measure_navigation_coverage_v44(np.full(domain.shape, -1, dtype=np.int8), domain, record['coverage'])
        sanity = evaluate_surface_v40(ref, np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), C_map=coverage['C_nav'])
        assert coverage['C_nav'] == 0 and sanity['Q'] == 0 and sanity['J'] == 0
        prepared['empty_prediction_sanity'] = dict(C_nav=0., Q=0., J_nav=0.,
            all_task_instances_in_denominator=len(record['target_instance_inventory']), study_result=False)
        references.append(prepared)
    save(output/'reference_predeclaration.json', dict(references=references,
        sealed_before_any_corresponding_actual_development_episode=True,
        actual_development_episode_count=0, prediction_or_method_winner_used=False))
    fixture = create_finite_episode_v44(output/'finite_initial_packet_episode', real_controller=True, compressed=True)
    fixture_result = fixture['result']
    assert fixture_result['status'] == 'controller_stop'
    assert fixture_result['executed_paid_actions'] == fixture_result['submitted_paid_actions'] == 0
    assert fixture_result['mapper_frames'] == fixture_result['mapper_tsdf_integrations'] == 1
    episode = fixture['root']
    digest = sha(episode/'artifact_manifest.json')
    replay_process = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/replay_episode_v44.py'),
        str(episode), '--expected-manifest-sha256', digest], cwd=ROOT, capture_output=True, text=True)
    (output/'replay_stdout.txt').write_text(replay_process.stdout)
    (output/'replay_stderr.txt').write_text(replay_process.stderr)
    if replay_process.returncode:
        raise RuntimeError('fresh-process saved-packet recomputation failed')
    replay = json.loads(replay_process.stdout)
    assert replay['status'] == 'verified' and replay['frames_verified'] == 1
    save(output/'independent_saved_replay.json', replay)
    # The known analytic packet is not an observation of DEV_A. It must never
    # be paired with that scene's GT to manufacture a development endpoint.
    try:
        evaluate_saved_episode_v44(episode, output/'references/DEV_A_00',
            reference_manifest_sha256=references[0]['manifest_sha256'], expected_episode_manifest_sha256=digest)
    except ValueError as exc:
        assert str(exc) == 'finite fixture cannot be scored as a development episode', str(exc)
        rejection = dict(rejected=True, exception=type(exc).__name__, reason=str(exc),
                         fixture_not_scored_as_development=True)
    else:
        raise AssertionError('analytic fixture was incorrectly admitted to development evaluation')
    save(output/'fixture_evaluation_rejection.json', rejection)
    preflight = run_development_v44('R2_A_diagnostic', output_root=output/'not_started', preflight_only=True)
    save(output/'actual_entrypoint_preflight.json', preflight)
    assert not preflight['start_slot_reserved']
    for name, digest in sources.items():
        assert sha(ROOT/name) == sha(output/'source'/name) == digest, name
    counts = runtime_counts_v41()
    assert not any(counts.values()), counts
    total = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    assert total < 16*1024**2, total
    result = dict(status='passed', phase='V44_saved_evidence_pipeline',
        references=references, fresh_process_saved_replay=replay,
        fixture_evaluation_rejected=True, old_seals=old, source_sha256=sources,
        sources_unchanged=True, actual_development_worlds=0, actual_autonomous_trajectories=0,
        physical_actions_executed=0, main_fixture_TSDF_integrations=1,
        main_fresh_process_recomputation_TSDF_integrations=1,
        unit_fixture_integrations_counted_separately=True,
        semantic_performance_claim_added=False, full_autonomous_pipeline_validated=False,
        actual_entrypoint_preflight_status=preflight['status'], storage_ready=preflight['resource']['passed'],
        runtime_counts=counts, new_world_replay_slots_consumed=0, bytes_before_final_records=total)
    save(output/'result.json', result)
    save(output/'artifact_sha256.json', {str(p.relative_to(output)): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(output.rglob('*')) if p.is_file()})
    assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= 16*1024**2
    print(json.dumps(dict(status=result['status'], output=str(output), references=2,
        fresh_process_saved_frames_verified=1, actual_development_worlds=0, storage_ready=result['storage_ready']),
        ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
