#!/usr/bin/env python3
"""Offline actual-state ICP association audit; starts only localhost master 11401."""
import os,sys,json,pathlib,socket,signal,subprocess,time
P=pathlib.Path(__file__).resolve().parent
F=P.parent/'instrumented_csm_candidate'
JOB_ROOT=pathlib.Path(os.environ['FIXTURE_OUTPUT_DIR']).resolve()
OUT=pathlib.Path(os.environ['AUDIT_OUTPUT_DIR']).resolve()
if OUT==P or P in OUT.parents:raise ValueError('output must be separate')
OUT.mkdir(parents=True,exist_ok=True)
os.environ.update(ROS_MASTER_URI='http://127.0.0.1:11401',ROS_HOSTNAME='127.0.0.1',ROS_IP='127.0.0.1')
if len(sys.argv)>1 and sys.argv[1]=='worker':
 import rospy
 rospy.init_node('correspondence_audit',disable_signals=True)
 params=json.loads((F/'state_fixture_results.json').read_text())['parameters'];rospy.set_param('/fixture/matcher',params)
 rows=(JOB_ROOT/'state_fixture_jobs.txt').read_text().splitlines();(OUT/'jobs.txt').write_text('\n'.join(x for x in rows if x.split()[0] in ('96','98'))+'\n')
 with (OUT/'results.jsonl').open('w') as out,(OUT/'correspondence.tmp').open('w') as err:
  subprocess.run([str(OUT/'correspondence_fixture'),str(OUT/'jobs.txt')],stdout=out,stderr=err,check=True,timeout=20)
 rospy.signal_shutdown('done')
else:
 with socket.socket() as s:
  if s.connect_ex(('127.0.0.1',11401))==0:raise RuntimeError('isolated port 11401 occupied')
 with (OUT/'master.log').open('w') as f:
  m=subprocess.Popen(['/opt/ros/noetic/bin/roscore','-p','11401'],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
  try:
   time.sleep(2);subprocess.run([sys.executable,__file__,'worker'],check=True,timeout=25,env=os.environ.copy())
  finally:
   m.send_signal(signal.SIGINT);m.wait(timeout=10)
