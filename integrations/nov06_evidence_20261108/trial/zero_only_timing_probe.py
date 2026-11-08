#!/usr/bin/env python3
"""Observe the actual adapter loop without starting a native CAN actuator or arming."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xmlrpc.client

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--duration', type=float, default=90.)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    uri = 'http://127.0.0.1:11382'
    assert os.environ.get('ROS_MASTER_URI') == uri
    master = xmlrpc.client.ServerProxy(uri)
    system = master.getSystemState('/nso_passive_timing_probe')
    assert system[0] == 1
    nodes = {node for rows in system[2] for topic, names in rows for node in names}
    assert not nodes.intersection({'/nso_radius_guard', '/nso_scout_radius_driver_v1', '/nso_onboard_supervisor'}), nodes
    # No publisher/service is created by this harness; both subscriptions are receive-only.
    import rospy
    from geometry_msgs.msg import TwistStamped
    from std_msgs.msg import String
    rospy.init_node('nso_passive_timing_probe', disable_signals=True)
    data = dict(command_count=0, nonzero_command_count=0, ever_armed=False,
                first_state=None, last_state=None)
    def command(msg):
        data['command_count'] += 1
        if msg.twist.linear.x != 0. or msg.twist.angular.z != 0.:
            data['nonzero_command_count'] += 1
    def state(msg):
        row = json.loads(msg.data)
        if row.get('session_id') != cfg['session_id']:
            return
        data['ever_armed'] |= row.get('ever_armed', False)
        if data['first_state'] is None:
            data['first_state'] = row
        data['last_state'] = row
    b = Path('/home/ouc/nso_robot')
    cfg = json.loads((b/'reports/nso-hybrid-20261108/nov06_sameground_S40_20261108_01/adapter-config.json').read_text())
    cfg.update(session_id='zero_only_timing_'+str(time.time_ns()), output_directory=str(out/'adapter'))
    raw = (json.dumps(cfg, indent=2)+'\n').encode()
    (out/'config.json').write_bytes(raw)
    source = b/'workspaces/nso_hybrid_20261029/runtime/visual_velocity_adapter.py'
    subscribers = [rospy.Subscriber('/nso_radius/guarded_cmd', TwistStamped, command, queue_size=100),
                   rospy.Subscriber('/nso_visual/actuator_state', String, state, queue_size=10)]
    argv = ['/usr/bin/python3','-s','-B',str(source),'--config',str(out/'config.json'),
            '--config-sha256',hashlib.sha256(raw).hexdigest(),'--run']
    started = time.monotonic()
    with (out/'adapter.log').open('x') as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        (out/'start.json').write_text(json.dumps(dict(pid=proc.pid, argv=argv,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            no_native_driver_started=True, no_arm_requested=True, master_nodes_before=sorted(nodes),
            started_unix_s=time.time(), duration_s=args.duration),indent=2)+'\n')
        try:
            while proc.poll() is None and time.monotonic()-started < args.duration:
                time.sleep(.1)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10.)
    data.update(returncode=proc.returncode, elapsed_s=time.monotonic()-started,
                no_arm_requested=True, no_native_driver_started=True,
                scope='Zero-only actual adapter timing observation; no proof of armed/moving reliability.')
    if (out/'adapter/result.json').exists():
        data['adapter_result']=json.loads((out/'adapter/result.json').read_text())
    (out/'probe_result.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:data[k] for k in ['returncode','elapsed_s','command_count','nonzero_command_count','ever_armed']}),flush=True)
    rospy.signal_shutdown('passive probe complete')

if __name__ == '__main__':
    main()
