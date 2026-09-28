#!/usr/bin/env python3
"""V27 four-case autonomous effect batch, reusing the frozen V26 collector.

Only the runtime feedback implementation and declared S/S_no_feedback matrix
change. World, packet collection, representation, evaluator and replay logic
remain in the archived V26 collector and are hash-pinned in every batch.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.run_observed_autonomous_v26 as base
from nso.observed_runtime_v27 import ObservedANSRuntimeV27

DEFAULT = ROOT/'audit_results/observed_autonomous_v27_20260916'
OLD = ROOT/'audit_results/observed_autonomous_v26_20260916'
REFERENCE = ROOT/'audit_results/observed_v27_geometry_reference_20260916'
SHADOW = ROOT/'audit_results/observed_feedback_v27_shadow_recovered_20260916'
SHADOW_VERIFY = ROOT/'audit_results/observed_feedback_v27_shadow_verification_20260916'
PROTOCOL = ROOT/'docs/research/V27_AUTONOMOUS_EFFECT_PROTOCOL_20260916.md'
ANALYSIS = ROOT/'docs/research/V27_AUTONOMOUS_ANALYSIS_PROTOCOL_20260916.md'
CONFIG = dict(version='observed-autonomous-v27-effect-1', parent='D25-P00', budget=400,
    sensor_model='iid_025px', noise_seed=1901, replan_interval=5, candidate_limit=12,
    quota_start=8, quota_limit=36, scheduled_tasks=4,
    task_cap_bytes=40*1024**2, shared_cap_bytes=2*1024**2,
    free_reserve_bytes=64*1024**2, receipt_reserve_bytes=65536, packet_cap_bytes=65536,
    shape_representation='unchanged V24 measured-only; enclosure disabled',
    intervention='V27-A stable measured feedback support only',
    independent_replay='fresh process/world/runtime computes its own actions then verifies saved packets and decisions')
PROGRESS = dict(case=None, replay=False, phase='unstarted', last_attempted_action=None,
                last_saved_packet=None, owned=False, prepare_owned=False)


def configure_base():
    base.CONFIG = CONFIG
    base.DEFAULT = DEFAULT
    base.PROTOCOL = PROTOCOL
    base.ANALYSIS = ANALYSIS
    base.OLD = OLD
    base.PROGRESS = PROGRESS
    base.ObservedANSRuntimeV26 = ObservedANSRuntimeV27
    base.source_files = source_files
    base.check_frozen = check_frozen
    base.claim_case = claim_case
    base.aggregate = aggregate
    base.failure = failure


def source_files():
    paths = {Path(__file__).resolve(), Path(base.__file__).resolve(), PROTOCOL, ANALYSIS}
    for module in tuple(base.sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and not any(
                    part.startswith('.venv') for part in path.relative_to(ROOT).parts):
                paths.add(path)
    for name in ('test_observed_state_v26.py', 'test_observed_planner_v26.py',
                 'test_observed_runtime_v26.py', 'test_facility_measurement_v26.py',
                 'test_autonomous_collector_v26.py', 'test_observed_feedback_v27.py',
                 'test_autonomous_collector_v27.py'):
        path = ROOT/'tests/virtual3d'/name
        if path.exists():
            paths.add(path)
    analyzer = ROOT/'scripts/analyze_observed_autonomous_v27.py'
    if analyzer.exists():
        paths.add(analyzer)
    return paths


def prepare(root):
    if base.shutil.disk_usage(ROOT).free < 4*CONFIG['task_cap_bytes'] + CONFIG['free_reserve_bytes']:
        raise OSError('prepare requires conservative four-task capacity plus 64 MiB reserve')
    for evidence in (OLD, REFERENCE, SHADOW, SHADOW_VERIFY):
        base.verify_inventory(evidence)
    old = base.read(OLD/'manifest.json')
    reference = base.read(REFERENCE/'result.json')
    shadow = base.read(SHADOW/'result.json')
    verified = base.read(SHADOW_VERIFY/'result.json')
    if old['status'] != 'complete' or old['main_attempts_started'] != 4:
        raise ValueError('completed four-case V26 batch required')
    if reference['status'] != 'complete_v26_v27_G_policy_equivalence':
        raise ValueError('V27/V26 G policy equivalence audit required')
    if (shadow['status'] != 'complete_matched_fixed_feedback_shadow'
            or verified['status'] != 'passed_independent_readonly_shadow_verification'):
        raise ValueError('fixed-history V27 feedback repair verification required')
    base.verify_mapping(old['source_sha256'])
    sources = {**old['source_sha256'], **{str(path.relative_to(ROOT)): base.sha(path)
               for path in source_files()}}
    cases = [dict(index=i, parent=CONFIG['parent'], assignment=assignment, mode=mode,
                  physical_status='unstarted', replay_status='unstarted')
             for i, (assignment, mode) in enumerate((
                 ('A_complex_B_simple', 'S_no_feedback'), ('A_complex_B_simple', 'S'),
                 ('A_simple_B_complex', 'S_no_feedback'), ('A_simple_B_complex', 'S')))]
    root.mkdir(parents=True, exist_ok=False)
    PROGRESS.update(prepare_owned=True, phase='preparing_source_archive')
    stream = base.io.BytesIO()
    with base.zipfile.ZipFile(stream, 'w', base.zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.writestr(name, (ROOT/name).read_bytes())
    base.save_bytes(root, root/'sources.zip', stream.getvalue())
    inputs = [OLD/'manifest.json', OLD/'result.json', OLD/'artifact_hashes.json',
              REFERENCE/'result.json', REFERENCE/'artifact_hashes.json',
              SHADOW/'result.json', SHADOW/'artifact_hashes.json',
              SHADOW_VERIFY/'result.json', SHADOW_VERIFY/'artifact_hashes.json']
    manifest = dict(status='prepared', config=CONFIG, cases=cases, source_sha256=sources,
        input_sha256={str(path.relative_to(ROOT)): base.sha(path) for path in inputs},
        source_archive_sha256=base.sha(root/'sources.zip'), versions=base.versions(),
        world_version=base.VERSION_V25, main_attempts_started=0,
        quota_start=8, quota_limit=36, preparation_worlds=0,
        preparation_sensor_calls=0, preparation_actions=0,
        historical_geometry_reference_cases=[0, 2], reference_equivalence=reference)
    base.write(root, root/'manifest.json', manifest)
    check_frozen(root)
    PROGRESS['prepare_owned'] = False
    print('PREPARED: 4 V27 effect tasks; prior attempts 8/36; no world constructed', flush=True)


def check_frozen(root):
    manifest = base.read(root/'manifest.json')
    if manifest['config'] != CONFIG or manifest['quota_start'] != 8:
        raise ValueError('frozen V27 configuration/quota changed')
    base.verify_mapping(manifest['source_sha256'])
    base.verify_mapping(manifest['input_sha256'])
    if base.sha(root/'sources.zip') != manifest['source_archive_sha256'] or base.versions() != manifest['versions']:
        raise ValueError('source archive/dependency version changed')
    return manifest


def claim_case(root, index):
    manifest = check_frozen(root)
    if manifest['status'].startswith('stopped') or manifest['status'] == 'complete':
        raise ValueError('batch is stopped or complete')
    if index not in range(4) or manifest['cases'][index]['physical_status'] != 'unstarted':
        raise ValueError('case already claimed or invalid')
    if any(case['replay_status'] != 'complete' for case in manifest['cases'][:index]):
        raise ValueError('fixed order requires earlier independent replays first')
    owned = len(list(root.glob('case_*')))
    if owned != manifest['main_attempts_started'] or owned != index or CONFIG['quota_start'] + owned >= CONFIG['quota_limit']:
        raise ValueError('case ownership and main-attempt ledger disagree')
    if base.shutil.disk_usage(ROOT).free < CONFIG['task_cap_bytes'] + CONFIG['free_reserve_bytes']:
        raise OSError('unstarted task requires 40 MiB plus 64 MiB; no directory claimed')
    folder = root/f'case_{index:02d}'
    folder.mkdir(exist_ok=False)
    PROGRESS['owned'] = True
    manifest['main_attempts_started'] += 1
    manifest['status'] = 'running'
    case = manifest['cases'][index]
    case.update(physical_status='running', physical_process_id=base.os.getpid(),
                main_attempt_number=CONFIG['quota_start'] + manifest['main_attempts_started'])
    base.write(root, root/'manifest.json', manifest)
    for name in ('packets', 'audit', 'meshes'):
        (folder/name).mkdir()
    return manifest, folder


def aggregate(root):
    manifest = check_frozen(root)
    rows = []
    for case in manifest['cases']:
        folder = root/f"case_{case['index']:02d}"
        base.verify_inventory(folder)
        result = base.read(folder/'result.json')
        verification = base.read(folder/'verification.json')
        if verification['status'] != 'passed':
            raise ValueError('all four independent autonomous replays required')
        rows.append(dict(index=case['index'], assignment=case['assignment'], mode=case['mode'],
            paid_actions=result['paid_actions'], returned=result['returned_to_anchor'],
            collisions=result['collisions'], eligible=result['eligible'],
            C=result['final_coverage_2d'],
            Q=result['final_main_observed']['05cm']['outline_macro_quality'],
            J=result['final_main_observed']['05cm']['joint_outline'],
            missing=result['final_main_observed']['missing_asset_count'],
            feedback=result['summary'].get('feedback_table'),
            result_sha256=base.sha(folder/'result.json')))
    pairs = []
    for off, on in ((0, 1), (2, 3)):
        left, right = rows[off], rows[on]
        pairs.append(dict(assignment=left['assignment'], disabled_index=off, enabled_index=on,
            delta_C=right['C']-left['C'], delta_Q=right['Q']-left['Q'],
            delta_J=right['J']-left['J'], relative_delta_J=(right['J']-left['J'])/left['J']))
    manifest['status'] = 'complete'
    base.write(root, root/'manifest.json', manifest)
    base.write(root, root/'result.json', dict(status='complete_four_v27_effect_tasks_and_independent_replays',
        cases=rows, feedback_pairs=pairs, new_main_attempts=4, cumulative_main_attempts=12,
        quota_limit=36, main_paid_actions=sum(row['paid_actions'] for row in rows),
        independent_replay_paid_actions=sum(row['paid_actions'] for row in rows),
        full_semantic_efficacy_claimed=False))
    base.seal(root)


def failure(root, error):
    if PROGRESS['prepare_owned'] and not PROGRESS['owned']:
        base.write(root, root/'failure_prepare.json', dict(status='failed_before_any_main_attempt',
            error=repr(error), traceback=base.traceback.format_exc(), progress=PROGRESS,
            new_main_attempts=0, prior_main_attempts=8), True)
        return
    if not PROGRESS['owned']:
        print('UNSTARTED/UNOWNED ERROR', repr(error), flush=True)
        return
    folder = root/f"case_{PROGRESS['case']:02d}"
    receipt = dict(status='failed', error=repr(error), traceback=base.traceback.format_exc(),
        progress=PROGRESS, process_id=base.os.getpid(), quota_refunded=False)
    base.write(folder, folder/('replay_failure.json' if PROGRESS['replay'] else 'failure.json'), receipt, True)
    base.seal(folder, True)
    manifest = base.read(root/'manifest.json')
    manifest['main_attempts_started'] = len(list(root.glob('case_*')))
    manifest['status'] = 'stopped_input_source_io_or_interface_failure'
    manifest['cases'][PROGRESS['case']]['replay_status' if PROGRESS['replay'] else 'physical_status'] = 'failed'
    base.write(root, root/'manifest.json', manifest, True)


def main():
    configure_base()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'replay', 'aggregate'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--case', type=int, choices=range(4))
    args = parser.parse_args()
    with (ROOT/'audit_results/.observed_autonomous_v26.lock').open('a') as old_lock, \
         (ROOT/'audit_results/.observed_autonomous_v27.lock').open('a') as lock:
        fcntl.flock(old_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.command == 'prepare':
                prepare(args.output)
            elif args.command == 'aggregate':
                aggregate(args.output)
            else:
                if args.case is None:
                    raise ValueError('--case required')
                base.run(args.output, args.case, args.command == 'replay')
        except BaseException as error:
            failure(args.output, error)
            raise


if __name__ == '__main__':
    main()
