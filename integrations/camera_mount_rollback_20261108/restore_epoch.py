"""User-requested rollback of camera mounting parameters and fresh mapping; no actuator."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen
import psutil

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/ouc/nso_robot')
OLD = BASE/'analyses/map_reset_20261108_01/processes.json'
DISPLAY = BASE/'experiments/live_dashboard_mount_rollback_20261108'
SEMANTIC = BASE/'experiments/live_dashboard_semantic_mount_rollback_20261108'

def save(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    assert not (ROOT/'processes.json').exists(), 'Reset already started; inspect existing processes.'
    assert not DISPLAY.exists() and not SEMANTIC.exists(), 'New output directories required.'
    old = json.loads(OLD.read_text())
    records = {x['name']: x for x in old['started']}
    records['camera'] = old['camera_process_retained']
    records['master'] = old['master_process_retained']
    pair = json.loads((BASE/'analyses/rapid_evidence_20261108/physical/epoch_S/start_receipt.json').read_text())
    records['navigation'] = dict(name='navigation', **pair['navigation'])
    records['pair_helper'] = dict(name='pair_helper', **pair['mapper_helper'])
    targets = ['dashboard_semantic', 'dashboard_overlay', 'dashboard',
               'navigation', 'sensor_bridge', 'pair_helper', 'sensors', 'csm_bridge', 'camera']
    for process in psutil.process_iter(['pid', 'cmdline']):
        argv = process.info['cmdline'] or []
        if any(Path(a).name in ('mission.py', 'visual_velocity_adapter.py',
                                'nso_onboard_supervisor', 'radius_native_sender') for a in argv):
            raise RuntimeError('Unexpected autonomous owner: '+str(process.info['pid']))
    snapshot = dict(reason='User explicitly requests restoring the prior camera configuration because yesterday 3D mapping looked better; only the six mount parameters are reverted.',
                    old_process_manifest=str(OLD), old_trial='fresh_calibrated_pair_20261108_01_S',
                    old_trial_valid_for_mapping_accuracy_or_coverage_benefit=False,
                    calibration_retained=False, policy_changed=False, actuator_started=False,
                    timestamp=time.time())
    for port in (8765, 8766):
        try:
            with urlopen('http://127.0.0.1:%d/api/state'%port, timeout=3) as response:
                save('previous_display_%d.json'%port, json.load(response))
        except Exception as exc:
            snapshot['display_%d_error'%port] = str(exc)
    save('request_and_previous_epoch.json', snapshot)
    stopped = []
    for name in targets:
        record = records[name]
        process = psutil.Process(record['pid'])
        actual = process.cmdline()
        # roslaunch may expose its Python interpreter before the configured executable.
        assert all(a in actual for a in record['argv'][1:]), (name, actual)
        children = process.children(recursive=True)
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=12)
        except psutil.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        for child in children:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                child.terminate()
        _, alive = psutil.wait_procs(children, timeout=5)
        assert not [p.pid for p in alive if p.status()!=psutil.STATUS_ZOMBIE], name
        stopped.append(dict(name=name,pid=record['pid'],children=[p.pid for p in children]))
        save('stopped.json', stopped)
    launch = BASE/'workspaces/nso_hybrid_20261029/config/hybrid_camera_neural.launch'
    backup = BASE/'analyses/calibrated_mount_20261108/hybrid_camera_neural.before.launch'
    (ROOT/'hybrid_camera_neural.calibrated.before.launch').write_bytes(launch.read_bytes())
    (ROOT/'hybrid_camera_neural.restored.launch').write_bytes(backup.read_bytes())
    launch.write_bytes(backup.read_bytes())
    env = os.environ.copy()
    env.update(old['env'])
    updated = dict(new_epoch=True, old_alignment_reused=False, no_actuator=True,
                   camera_restarted_with_legacy_mount=True, master_process_retained=records['master'],
                   env=old['env'], source_workspace=old['source_workspace'], started=[],
                   display_output=str(DISPLAY),semantic_output=str(SEMANTIC))
    for name in ['camera','csm_bridge','sensors','sensor_bridge','navigation','dashboard','dashboard_overlay','dashboard_semantic']:
        argv = [a.replace('live_dashboard_semantic_reset_20261108_01',SEMANTIC.name)
                 .replace('live_dashboard_reset_20261108_01',DISPLAY.name)
                for a in records[name]['argv']]
        if name == 'navigation':
            argv = [a for a in argv if not a.startswith('map_topic:=')]
        if name == 'dashboard_overlay' and '--camera-calibration-receipt' in argv:
            index = argv.index('--camera-calibration-receipt')
            del argv[index:index+2]
        with (ROOT/(name+'.log')).open('xb', buffering=0) as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        updated['started'].append(dict(name=name,pid=process.pid,argv=argv,started_unix_s=time.time()))
        save('processes.json', updated)
        time.sleep(1 if name != 'sensors' else 5)
        assert process.poll() is None, (name, process.returncode)
    print(json.dumps(dict(reset_started=True,manifest=str(ROOT/'processes.json'),no_actuator=True)),flush=True)

if __name__ == '__main__':
    main()
