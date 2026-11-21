import os,sys,subprocess,time,signal,socket
from pathlib import Path
p=Path(__file__).resolve().parent
out=Path(os.environ['STATIONARY_OUTPUT_DIR']).resolve()
if out==p or p in out.parents:raise ValueError('output must be separate')
out.mkdir(parents=True,exist_ok=True)
for profile in ['default','every_frame']:
 s=socket.socket();assert s.connect_ex(('127.0.0.1',11402))!=0;s.close()
 with (out/(profile+'_master.log')).open('w') as log:
  master=subprocess.Popen(['/opt/ros/noetic/bin/roscore','-p','11402'],env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:11402',ROS_HOSTNAME='127.0.0.1',ROS_IP='127.0.0.1'),stdout=log,stderr=log)
  try:
   time.sleep(2)
   subprocess.run([sys.executable,str(p/'replay.py'),profile],check=True,timeout=25)
  finally:master.send_signal(signal.SIGINT);master.wait(timeout=10)
