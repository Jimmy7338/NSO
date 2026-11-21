"""User-requested fresh map epoch; no actuator or autonomous mission is started."""
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
OLD = BASE/'analyses/calibrated_mount_20261108/processes.json'
DISPLAY = BASE/'experiments/live_dashboard_reset_20261108_01'
SEMANTIC = BASE/'experiments/live_dashboard_semantic_reset_20261108_01'

def save(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def main():
    assert not (ROOT/'processes.json').exists(), 'Reset already started; inspect existing processes.'
    assert not DISPLAY.exists() and not SEMANTIC.exists(), 'New output directories required.'
    old = json.loads(OLD.read_text())
    records = {x['name']: x for x in old['started']}
    records['sensor_bridge']['pid'] = 496924
    targets = ['dashboard_semantic', 'dashboard_overlay', 'dashboard',
               'navigation', 'sensor_bridge', 'sensors', 'csm_bridge']
    for process in psutil.process_iter(['pid', 'cmdline']):
        argv = process.info['cmdline'] or []
        if any(Path(a).name in ('mission.py', 'visual_velocity_adapter.py',
                                'nso_onboard_supervisor', 'radius_native_sender') for a in argv):
            raise RuntimeError('Unexpected autonomous owner: '+str(process.info['pid']))
    snapshot = dict(reason='User reports drift while manually driving to experimental area; requests new 2D and 3D maps.',
                    old_process_manifest=str(OLD), old_trial='calibrated_S24_02',
                    old_trial_valid_for_mapping_accuracy_or_coverage_benefit=False,
                    calibration_retained=True, policy_changed=False, actuator_started=False,
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
    env = os.environ.copy()
    env.update(old['env'])
    updated = dict(new_epoch=True, old_alignment_reused=False, no_actuator=True,
                   camera_process_retained=records['camera'], master_process_retained=records['master'],
                   env=old['env'], source_workspace=old['source_workspace'], started=[],
                   display_output=str(DISPLAY),semantic_output=str(SEMANTIC))
    for name in ['csm_bridge','sensors','sensor_bridge','navigation','dashboard','dashboard_overlay','dashboard_semantic']:
        argv = [a.replace('live_dashboard_semantic_calibrated_20261108',SEMANTIC.name)
                 .replace('live_dashboard_calibrated_20261108',DISPLAY.name)
                for a in records[name]['argv']]
        with (ROOT/(name+'.log')).open('xb', buffering=0) as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        updated['started'].append(dict(name=name,pid=process.pid,argv=argv,started_unix_s=time.time()))
        save('processes.json', updated)
        time.sleep(1 if name != 'sensors' else 5)
        assert process.poll() is None, (name, process.returncode)
    print(json.dumps(dict(reset_started=True,manifest=str(ROOT/'processes.json'),no_actuator=True)),flush=True)

if __name__ == '__main__':
    main()
