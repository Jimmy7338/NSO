#!/usr/bin/env python3
"""Archive and validate batch interfaces; actual study launches are forbidden."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def run(output):
    from nso.development_batch_v45 import run_batch_v45, source_names_v45
    from nso.batch_report_v45 import report_batch_v45
    from env.development_sensor_v41 import runtime_counts_v41
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    names = set(source_names_v45())
    for pattern in ('nso/*v45.py', 'scripts/*v45.py', 'tests/*v45.py'):
        names.update(str(path.relative_to(ROOT)) for path in ROOT.glob(pattern))
    sources = {name: sha(ROOT/name) for name in sorted(names)}
    for name in sources:
        path = output/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    save(output/'started.json', dict(scope='finite batch/subprocess tests and real preflight only',
        source_sha256=sources, actual_worlds_authorized=0, maximum_output_bytes=8*1024**2))
    seals = []
    for name in ('v40_p0_20260920', 'v40_p1_static_release_20260920',
                 'v41_interface_release_20260920', 'v42_interface_release_20260921',
                 'v43_interface_release_20260921', 'v44_interface_release_20260921'):
        path = ROOT/'audit_results'/name/'manifest.json'; data = json.loads(path.read_text())
        for relative, entry in data['files'].items():
            assert sha(ROOT/relative) == entry['sha256'], relative
        seals.append(dict(manifest=str(path.relative_to(ROOT)), sha256=sha(path), entries=len(data['files'])))
    before = time.monotonic()
    test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests',
                           '-p', 'test_*v45.py', '-v'], cwd=ROOT, capture_output=True, text=True)
    log = test.stdout+test.stderr; (output/'unittest.txt').write_text(log)
    test_record = dict(returncode=test.returncode, elapsed_s=time.monotonic()-before,
        tests_reported=int(re.search(r'Ran (\d+) tests', log).group(1)), performance_benchmark=False)
    save(output/'test_result.json', test_record)
    if test.returncode:
        raise RuntimeError('tests failed; retain output directory and diagnose before a new revision')
    preflight = run_batch_v45(output/'not_started_episodes', preflight_only=True)
    assert preflight['status'] in ('blocked_before_world_creation', 'ready_without_world_creation')
    assert not (output/'not_started_episodes').exists()
    save(output/'batch_preflight.json', preflight)
    report = report_batch_v45(output/'batch_preflight.json')
    save(output/'batch_report.json', report)
    assert report['reserved_slots'] == report['recorded_launches'] == report['verified_autonomous_endpoints'] == 0
    assert report['counts']['not_started'] == 5
    assert all(row['delta_S_minus_G'] is None for row in report['paired_deltas'])
    legacy = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/verify_legacy_evidence_v45.py'),
        '--output', str(output/'legacy_evidence_recheck.json')], cwd=ROOT, capture_output=True, text=True)
    (output/'legacy_recheck_stdout.txt').write_text(legacy.stdout)
    (output/'legacy_recheck_stderr.txt').write_text(legacy.stderr)
    if legacy.returncode:
        raise RuntimeError('legacy source evidence recheck failed')
    for name, digest in sources.items():
        assert sha(ROOT/name) == sha(output/'source'/name) == digest, name
    counts = runtime_counts_v41(); assert not any(counts.values()), counts
    result = dict(status='passed', scope='V45 batch correctness and legacy arithmetic only',
        contract_tests=test_record['tests_reported'], source_sha256=sources,
        sources_unchanged=True, old_seals=seals, runtime_counts=counts,
        storage_gate_passed=preflight['resource']['passed'],
        real_preflight_status=preflight['status'], declared_slots=5, actual_started_slots=0,
        verified_autonomous_endpoints=0, actual_worlds=0, actual_sensor_packets=0,
        actual_TSDF_integrations=0, semantic_performance_evidence_added=False,
        legacy_main_records_rechecked=36, legacy_sources_rechecked=158,
        finite_subprocess_tests_are_not_study_runs=True)
    save(output/'result.json', result)
    save(output/'artifact_sha256.json', {str(path.relative_to(output)): dict(sha256=sha(path), bytes=path.stat().st_size)
        for path in sorted(output.rglob('*')) if path.is_file()})
    assert sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) <= 8*1024**2
    print(json.dumps(dict(status='passed', tests=result['contract_tests'], actual_worlds=0,
        storage_gate_passed=result['storage_gate_passed'], output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
