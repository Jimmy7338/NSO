#!/usr/bin/env python3
"""Run three fresh CSM scan histories using an isolated local ROS master."""
import argparse, os, signal, socket, subprocess, sys, time
from pathlib import Path

p = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--output', type=Path, required=True, help='separate output directory')
ap.add_argument('--roscore', default='/opt/ros/noetic/bin/roscore')
args = ap.parse_args()
out = args.output.resolve()
if out == p or p in out.parents:
    raise ValueError('output must be outside this evidence package')
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, ROS_MASTER_URI='http://127.0.0.1:11398', ROS_HOSTNAME='127.0.0.1', ROS_IP='127.0.0.1', REPLAY_OUTPUT_DIR=str(out))
for label, indices in [('audit_full150',range(150)),('audit_start050',range(50,150)),('audit_pair098099',[98,99])]:
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',11398)) == 0:
            raise RuntimeError('isolated ROS master port 11398 is already occupied')
    with (out/(label+'_master.log')).open('w') as f:
        master=subprocess.Popen([args.roscore,'-p','11398'],stdout=f,stderr=f,env=env)
        try:
            time.sleep(2)
            subprocess.run([sys.executable,str(p/'replay.py'),'worker',label,'false','1'],env=dict(env,REPLAY_SCAN_INDICES=','.join(map(str,indices))),check=True,timeout=35)
        finally:
            master.send_signal(signal.SIGINT)
            master.wait(timeout=10)
