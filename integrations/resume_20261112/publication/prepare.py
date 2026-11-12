"""Prepare bounded text evidence locally; never starts hardware or publishes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path('/home/ouc/nso_robot')
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'analyses/resume_20261112'
WORK = ROOT / 'workspaces/nso_hybrid_20261029'
PARENT = 'f866801b352c00e9b3bc7155f6a3c59037707477'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal-index', type=Path,
                        help='Root-reviewed JSON: {files:[{source,path}], summary:text}')
    args = parser.parse_args()
    stage = OUT / 'stage'
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    selected = [(SRC / name, name) for name in (
        'restore_data.py', 'configuration_S96.json',
        'configuration_S96.before_actual_ready.json', 'protocol_S96.json',
        'processes.json', 'RC_discrepancy.json', 'pre_start_can.json',
        'sdk_profile.json', 'camera_startup_profile.json', 'camera_launch.before',
        'fresh/source.json', 'fresh_sdk50/source.json', 'fresh_ready/source.json',
        'run/process.json', 'run/RUN_NEXT.sh',
        'semantic_audit/actual_initial_summary.json',
        'semantic_audit/preview/run_preview.py', 'semantic_audit/preview/identity.json',
        'semantic_audit/preview/observe.json', 'semantic_audit/preview/summary.json')]
    frozen = OUT / 'frozen_S96_source'
    selected += [((frozen / 'hybrid_camera_neural.launch') if frozen.exists() else WORK / 'config/hybrid_camera_neural.launch', 'active_source/hybrid_camera_neural.launch'),
                 ((frozen / 'mission.py') if frozen.exists() else WORK / 'runtime/mission.py', 'active_source/mission.py'),
                 ((frozen / 'joint_engine.py') if frozen.exists() else WORK / 'nso_bridge/nso_bridge/joint_engine.py', 'active_source/joint_engine.py'),
                 (ROOT / 'staging/ans-live-20261110-snapshot-v1/manifest.json', 'active_source/native_stage_manifest.json')]
    # Historical diagnosis is distinct from the fresh actual Nov12 decision audit.
    historical = ROOT / 'analyses/resume_20261110/semantic_entry_review'
    selected += [(historical / name, 'historical_entry_diagnosis/' + name)
                 for name in ('README.md', 'diagnose_saved.py', 'restore_paid36.py',
                              'exact_diagnosis_summary.json', 'replay_result.json', 'result.json')]
    terminal = json.loads(args.terminal_index.read_text()) if args.terminal_index else None
    if terminal:
        assert terminal['summary'].strip() and terminal['files']
        selected += [(Path(row['source']), row['path']) for row in terminal['files']]
    rows = []
    for source, relative in selected:
        target = stage / relative
        assert target.resolve().is_relative_to(stage.resolve())
        assert source.is_file(), source
        assert source.stat().st_size < 600000, source
        data = source.read_bytes()
        data.decode('utf-8')  # Reject binary data, including raw arrays/images.
        assert source.suffix.lower() not in ('.npz', '.npy', '.jpg', '.png', '.pt', '.bag')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(dict(path=relative, source=str(source), bytes=len(data),
                         sha256=hashlib.sha256(data).hexdigest()))
    profile = json.loads((stage / 'camera_startup_profile.json').read_text())
    for path, key in [('camera_launch.before', 'before_sha256'),
                      ('active_source/hybrid_camera_neural.launch', 'after_sha256')]:
        assert hashlib.sha256((stage / path).read_bytes()).hexdigest() == profile[key]
    config = json.loads((stage / 'protocol_S96.json').read_text())
    assert hashlib.sha256((stage / 'configuration_S96.json').read_bytes()).hexdigest() == config['configuration_sha256']
    source_manifest = stage / 'S96_report/source_manifest.json'
    if source_manifest.exists():
        bindings = {r['path']: r['sha256'] for r in json.loads(source_manifest.read_text())['files']}
        for target, original in [('mission.py', 'runtime/mission.py'),
                                 ('joint_engine.py', 'nso_bridge/nso_bridge/joint_engine.py'),
                                 ('hybrid_camera_neural.launch', 'config/hybrid_camera_neural.launch')]:
            assert hashlib.sha256((stage / 'active_source' / target).read_bytes()).hexdigest() == bindings[original]
    deployment = stage / 'speed_fault/deployment/receipt.json'
    if deployment.exists():
        for row in json.loads(deployment.read_text())['files']:
            name = Path(row['path']).name
            assert hashlib.sha256((stage / 'speed_fault/candidate/runtime' / name).read_bytes()).hexdigest() == row['after_sha256'] == row['candidate_sha256']
            assert hashlib.sha256((stage / 'speed_fault/deployment' / (name + '.before')).read_bytes()).hexdigest() == row['before_sha256']
    readme = '''# Fresh Nov12 S96 continuation evidence

This is one fresh localization epoch in the same exhibition scene. No previous grid or observation ledger was imported. The declared objective remains unknown-space coverage and measured-surface mapping. The frozen budget is 96 actions and 1200 seconds.

The startup script restores data, mapping, navigation services and display processes; it does not submit actuator goals. The startup camera confidence default was changed from 80 to the last declared 50, retaining texture confidence 100. The launch before/after files and receipt preserve that change. The SDK receipt records a dormant white-balance backing value of 0 clamped to 28 while auto white balance stayed enabled: combined level 14|1 became 15 and bypassed the confidence case 1. Separate WB and confidence updates produced three stable 50/100 readbacks. This diagnosis does not establish a causal repair of historical depth artifacts.

The offline preview used the fresh_sdk50 stopped input; the actual decision audit uses recorded mission decisions after fresh_ready initialization. Forecasts are not executed endpoint outcomes. Historical entry diagnosis is retained under its own directory. Original scorer and candidate behavior are represented by the active joint_engine source; no Nov12 research-policy change is claimed. Prior native snapshot and physical sensor adaptations remain declared in configuration and source manifests. This evidence supports integration and within-scene mechanism observations, not matched method efficacy, independent scene replication or independently measured reconstruction accuracy.

Raw RGBD, scans, images, arrays, binaries, models and bulk logs remain local. Source hashes support provenance but omitted sensor inputs prevent fully reproducing physical observations from this compact package alone. Historical failures and negative semantic findings remain valid.
'''
    readme += '\n' + (terminal['summary'] if terminal else 'TERMINAL PENDING: the experiment was running when this preparation was requested. No completed outcome is asserted. Publication is pending root review of terminal evidence.') + '\n'
    (stage / 'README.md').write_text(readme)
    manifest = dict(schema='nso.resume_publication.v1', local_stage_only=True,
                    expected_parent=PARENT, terminal_pending=terminal is None,
                    files=rows, omitted='raw RGBD/scans/images/arrays/models/binaries/bulk logs')
    (stage / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    payload = [dict(path=str(p.relative_to(stage)), mode='100644', type='blob', content=p.read_text())
               for p in sorted(stage.rglob('*')) if p.is_file()]
    (OUT / 'payload.json').write_text(json.dumps(payload, ensure_ascii=False))
    (OUT / 'payload.sha256').write_text(hashlib.sha256((OUT / 'payload.json').read_bytes()).hexdigest() + '  payload.json\n')
    validation = dict(files=len(payload), bytes=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()),
                      hashes_passed=all(hashlib.sha256((stage / r['path']).read_bytes()).hexdigest() == r['sha256'] for r in rows),
                      launch_before_after_passed=True, frozen_configuration_passed=True,
                      terminal_pending=terminal is None, network_writes=False)
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
