"""Bounded local publication preparation; no hardware/network operations."""
from pathlib import Path
import json
import hashlib
import shutil

ROOT = Path('/home/ouc/nso_robot')
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'analyses/restriction_cleanup_20261112'
RESEARCH = ROOT / 'analyses/ros_checked_exploration_v2_20261112'
WORK = ROOT / 'workspaces/nso_hybrid_20261029'
REPORT = ROOT / 'reports/nso-hybrid-20261112'


def main():
    stage = OUT / 'stage'
    stage.mkdir(exist_ok=True)
    selected = []
    for name in ['joint_navigation_params.before.yaml', 'joint_navigation_unknown_v2.before_live_local.yaml',
                 'ROS_configuration_change.json', 'ROS_active_parameters.json', 'ROS_live_local_v3.json',
                 'reduce_v2_episode.py', 'ros_checked_S40_v2_reduction.json', 'v2_reducer_validation.json',
                 'ROS_rejection/local_costmap.json', 'ROS_rejection/rotation_footprint_costs.json',
                 'runtime_audit/inventory.json', 'runtime_audit/candidate_profiles.json', 'runtime_audit/README.md',
                 'deployment/receipt.json', 'deployment/joint_ros_coordinator.py.before',
                 'main_S40/protocol.json', 'main_S40/configuration.json', 'main_S40/fresh/source.json',
                 'live_local_S40/protocol.json', 'live_local_S40/configuration.json', 'live_local_S40/fresh/source.json']:
        selected.append((SRC / name, name))
    for folder in ['runtime_audit/candidate', 'runtime_audit/coordinator_candidate']:
        for name in ['README.md', 'manifest.json', 'change.patch', 'test_results.txt']:
            selected.append((SRC / folder / name, folder + '/' + name))
    for name in ['runtime_audit/candidate/final_targeted_test_results.txt',
                 'runtime_audit/candidate/tests/test_recoverable_sensor_pause.py',
                 'runtime_audit/coordinator_candidate/tests/test_failed_attempt_coordinator.py']:
        selected.append((SRC / name, name))
    for name in ['PROFILE.md', 'implementation.json', 'tests.log', 'first_actual_plan.json',
                 'replay_C9.py', 'C9_summary.json', 'C9_next_action.json']:
        selected.append((RESEARCH / name, 'research_v2/' + name))
    for arm, report in [('original_v2', 'ros_checked_S40_20261112_01'),
                        ('live_local_v3', 'ros_live_local_S40_20261112_01')]:
        for name in ['source_manifest.json', 'runtime_parameters.json', 'reference-task-receipt.json',
                     'authorization.json', 'joint_navigation_contract.json']:
            selected.append((REPORT / report / name, arm + '_report/' + name))
    for name in ['ordinary_reverse/candidate/change.patch', 'ordinary_reverse/candidate/manifest.json',
                 'ordinary_reverse/candidate/README.md', 'ordinary_reverse/candidate/test_results.txt',
                 'ordinary_reverse/candidate/tests/test_ordinary_reverse.py',
                 'ordinary_reverse/deployment/receipt.json', 'ordinary_reverse/initial_backoff/goal.json',
                 'reverse_recovery_candidate/proposal.patch', 'reverse_recovery_candidate/receipt.json',
                 'reverse_recovery_candidate/test.log', 'reverse_recovery_candidate/tests/test_ROS_backoff_recovery.py',
                 'reverse_recovery_candidate/deployment/receipt.json',
                 'reverse_main_S40/configuration.json', 'reverse_main_S40/protocol.json',
                 'reverse_main_S40/fresh/source.json']:
        selected.append((SRC / name, name))
    for name in ['source_manifest.json', 'runtime_parameters.json', 'authorization.json', 'reference-task-receipt.json']:
        selected.append((REPORT / 'ros_reverse_exploration_S40_20261112_01' / name, 'reverse_main_report/' + name))
    for name in ['result.json', 'summary.json', 'navigation_reverse_contract.json', 'source_manifest.json']:
        selected.append((REPORT / 'ordinary_reverse_backoff_20261112_01' / name, 'standalone_ROS_preparation/' + name))
    extra_index = OUT / 'extra_index.json'
    if extra_index.exists():
        selected.extend((Path(row['source']), row['path']) for row in json.loads(extra_index.read_text()))
    frozen = OUT / 'frozen_source'
    frozen.mkdir(exist_ok=True)
    active_paths = ['runtime/mission.py', 'runtime/visual_velocity_adapter.py', 'runtime/joint_ros_coordinator.py',
                    'config/joint_navigation_params.yaml', 'nso_bridge/nso_bridge/joint_engine.py',
                    'nso_bridge/nso_bridge/exploration_joint_policy.py',
                    'nso_bridge/nso_bridge/ros_checked_exploration_v2.py',
                    'nso_bridge/tests/test_ros_checked_exploration_v2.py']
    manifest = json.loads((REPORT / 'ros_live_local_S40_20261112_01/source_manifest.json').read_text())
    expected = {r['path']: r['sha256'] for r in manifest['files']}
    bindings = []
    for relative in active_paths:
        target = frozen / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WORK / relative, target)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        authority = 'live_local_v3_report/source_manifest.json'
        wanted = expected.get(relative)
        if wanted is None:
            authority = 'research_v2/implementation.json'
            wanted = json.loads((RESEARCH / 'implementation.json').read_text())['changed_source_sha256'][str(WORK / relative)]
        assert sha == wanted, relative
        bindings.append(dict(path=relative, sha256=sha, source_manifest=authority))
        selected.append((target, 'active_v3_source/' + relative))
    reverse_manifest = json.loads((REPORT / 'ros_reverse_exploration_S40_20261112_01/source_manifest.json').read_text())
    reverse_expected = {r['path']: r['sha256'] for r in reverse_manifest['files']}
    for relative in ['runtime/mission.py', 'runtime/visual_velocity_adapter.py', 'runtime/joint_ros_coordinator.py']:
        target = OUT / 'frozen_reverse_source' / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(WORK / relative, target)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        assert sha == reverse_expected[relative], relative
        bindings.append(dict(path='reverse_source/' + relative, sha256=sha,
                             source_manifest='reverse_main_report/source_manifest.json'))
        selected.append((target, 'reverse_source/' + relative))
    rows = []
    for source, relative in selected:
        assert source.is_file() and source.stat().st_size < 600000, source
        data = source.read_bytes()
        data.decode('utf-8')
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(dict(path=relative, source=str(source), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    (stage / 'SOURCE_BINDINGS.json').write_text(json.dumps(bindings, indent=2) + '\n')
    (stage / 'README.md').write_text('''# Changed research admission and live local ROS profile

The user authorized unknown-space attempts and broad restriction cleanup. ros_checked_exploration_v2 changes research policy: known-free circular-footprint and certified return restrictions are removed, unknown-space attempts are admitted, failed executed primitives consume budget and retain actual observations without arrival credit. Original vendor source remaining unchanged does not establish method equivalence. Scorer/source identity and optimistic best-effort return must remain distinct from original NSO evidence.

The first modified-policy trial was operator-stopped after 28 attempts: 3 successful and 25 failed partial attempts, approximately 0.429 m and 418.94 s. Known-safe admission no longer blocked the trials; repeated ROS control rejection remained. Its corrected failed-attempt reducer and validation are retained. This is an interrupted integration failure, not successful exploration.

ros_live_local_v3 removes the local static map layer, retains live laser/depth obstacle and inflation layers, removes 4 cm footprint padding while retaining actual body vertices, and replans globally at 1 Hz. It enables one stock local obstacle-cache clearing recovery with reset_distance=0, correct obstacle_layer name and force_updating. Stock rotation recovery is NOT enabled. This changes execution/map-state behavior and protection margins; actual local collision checking and hardware protection remain. recoverable_sensor_pause_v1 is separately declared. Earlier measured_020_v1 feedback-stop profile and command cap remain disclosed in configuration.

The initial live_local_v3 trial was subsequently operator-stopped; its final failed-attempt reduction is pending. Its source bindings remain frozen independently of later runtime changes.

The user explicitly authorized ordinary low-speed ROS reverse. ordinary_ros_reverse_005_v1 changes the command bounds to [-0.05,0.05] m/s. ROS_backoff_040_v1 adds one automatic ROS backoff after eligible failed primitives, followed by actual observation and replanning. This is a declared research/execution revision; recovery movement/costs are separate from NSO-selected actions and no successful research arrival is fabricated. Candidate/deployment records and the separate 41-test and 31-test evidence are retained. The initial standalone ROS backoff preparation reached its ROS target and stopped (approximately 0.3109 m, 13.89 s); it was NOT an NSO-selected goal or an exploration episode.

REVERSE MAIN TERMINAL PENDING. The reverse_main_S40 protocol/configuration and source bindings are frozen; no completed main outcome is asserted. Full RGBD, raw maps/arrays, model weights, binaries and bulk logs are omitted. Historical isolated candidate manifests describe their preparation-time status; later deployment receipts record activation. Compact JSON diagnostics do not establish physical obstacle truth or clearance. No network publication is performed by this script.
''')
    terminal_pending = not (OUT / 'FINAL_STATUS.md').exists()
    if not terminal_pending:
        shutil.copyfile(OUT / 'FINAL_STATUS.md', stage / 'README.md')
    (stage / 'MANIFEST.json').write_text(json.dumps(dict(schema='nso.restriction_stage.v1',
        expected_parent='cea07d8d2f37cec11f689d3e627e80c034008260', terminal_pending=terminal_pending,
        research_policy_changed=True, original_method_equivalence=False, files=rows), indent=2) + '\n')
    payload = [dict(path=str(p.relative_to(stage)), mode='100644', type='blob', content=p.read_bytes().decode())
               for p in sorted(stage.rglob('*')) if p.is_file()]
    (OUT / 'payload.json').write_text(json.dumps(payload, ensure_ascii=False))
    validation = dict(files=len(payload), bytes=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()),
                      hashes_passed=all(hashlib.sha256((stage / r['path']).read_bytes()).hexdigest() == r['sha256'] for r in rows),
                      source_bindings_passed=len(bindings), terminal_pending=terminal_pending, network_writes=False)
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
