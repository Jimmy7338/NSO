#!/usr/bin/env python3
"""Own-process-only offline fixture for native explore and passive observer."""
import argparse, hashlib, json, os, pathlib, shlex, signal, socket, subprocess, sys, tempfile, time, uuid
P=pathlib.Path(__file__).resolve().parent; root=P.parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ros-setup', required=True, type=pathlib.Path)
parser.add_argument('--overlay-setup', type=pathlib.Path)
parser.add_argument('--native-binary', required=True, type=pathlib.Path)
args_cli=parser.parse_args()
ros_setup=args_cli.ros_setup.resolve(); overlay_setup=args_cli.overlay_setup.resolve() if args_cli.overlay_setup else None; native_binary=args_cli.native_binary.resolve()
for path in (ros_setup, overlay_setup, native_binary):
 if path is not None and not path.is_file(): parser.error(f'file not found: {path}')
if not os.access(native_binary, os.X_OK): parser.error(f'native binary is not executable: {native_binary}')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
source_files=[root/'package.xml',root/'CMakeLists.txt',root/'launch/reference.launch',root/'scripts/native_launcher.py',root/'scripts/event_recorder.py',P/'run_isolated.py',P/'fixture.py',P/'cancel_owner.py']
source_hashes={str(path.relative_to(root)):sha(path) for path in source_files}
out=P/'runs'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
with socket.socket() as sock:
 sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
assert port not in (11311,11382)
env=os.environ.copy();env.update(ROS_MASTER_URI=f'http://127.0.0.1:{port}',ROS_HOSTNAME='127.0.0.1',ROS_IP='127.0.0.1',ROS_HOME=str(out/'ros_home'),FIXTURE_LOG=str(out/'fixture.jsonl'))
setup='source '+shlex.quote(str(ros_setup))+' && '
if overlay_setup: setup+='source '+shlex.quote(str(overlay_setup))+' && '
setup+='export ROS_PACKAGE_PATH='+shlex.quote(str(root.parent))+':${ROS_PACKAGE_PATH:-} && '
owned=[]; files=[];receipt={'port':port,'uri':env['ROS_MASTER_URI'],'ros_home':env['ROS_HOME'],'checks':{},'fixture_only':True,'errors':[],'inputs':{'ros_setup':str(ros_setup),'ros_setup_sha256':sha(ros_setup),'overlay_setup':None if overlay_setup is None else str(overlay_setup),'overlay_setup_sha256':None if overlay_setup is None else sha(overlay_setup),'native_binary':str(native_binary),'native_binary_sha256':sha(native_binary),'package_root':str(root),'package_source_sha256':source_hashes}}
def start(name,cmd):
 f=open(out/(name+'.log'),'w');files.append(f);p=subprocess.Popen(['bash','-c',setup+'exec '+cmd],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);owned.append((name,p));return p
def run(cmd,timeout=10):
 p=subprocess.run(['bash','-c',setup+cmd],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout);return p.returncode,p.stdout
def rows(path):
 if not path.exists():return []
 return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def waitfor(pred,seconds):
 end=time.time()+seconds
 while time.time()<end:
  if pred():return True
  time.sleep(.2)
 return bool(pred())
def stop(p):
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGINT)
  try:p.wait(8)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(4)
try:
 master=start('roscore',f'roscore -p {port}')
 if not waitfor(lambda:run('rosparam list',3)[0]==0,15):raise RuntimeError('isolated master did not start')
 fixture=start('fixture',shlex.quote(sys.executable)+' '+shlex.quote(str(P/'fixture.py')))
 if not waitfor(lambda:any(x['event']=='ready' for x in rows(out/'fixture.jsonl')),15):raise RuntimeError('fixture did not start')
 args=' '.join(shlex.quote(x) for x in [str(root/'launch/reference.launch'),'navigation_namespace:=fixture_nav','costmap_topic:=/fixture/costmap','costmap_updates_topic:=/fixture/costmap_updates','action_namespace:=/fixture/move_base','base_frame:=fixture_base','event_log:='+str(out/'observer.jsonl'),'protection_topic:=/fixture/protection','native_binary:='+str(native_binary)])
 default=start('default',f'roslaunch {args}');time.sleep(4)
 r,nodes=run('rosnode list');(out/'default_nodes.txt').write_text(nodes)
 receipt['checks']['disabled_no_goals']=not any(x['event']=='goal' for x in rows(out/'fixture.jsonl'))
 receipt['checks']['disabled_no_explore_node']='/fixture_nav/explore' not in nodes
 stop(default)
 recorder_args=args.replace(shlex.quote('event_log:='+str(out/'observer.jsonl')),shlex.quote('event_log:='+str(out/'recorder_only.jsonl')))
 only=start('recorder_only',f'roslaunch {recorder_args} enable_exploration:=false enable_recorder:=true')
 receipt['checks']['recorder_only_started']=waitfor(lambda:any(x['event']=='recorder_start' for x in rows(out/'recorder_only.jsonl')),12)
 r,only_nodes=run('rosnode list');receipt['checks']['recorder_only_no_explore']='/fixture_nav/explore' not in only_nodes
 run("rostopic pub -1 /fixture/protection std_msgs/String 'data: standalone-test'",10);time.sleep(1);stop(only)
 receipt['checks']['recorder_only_protection']=any(x['event']=='protection_event' for x in rows(out/'recorder_only.jsonl'))
 receipt['checks']['recorder_only_stop']=any(x['event']=='recorder_stop' for x in rows(out/'recorder_only.jsonl'))
 session_only=next((x.get('session_id') for x in rows(out/'recorder_only.jsonl') if x['event']=='recorder_start'),None)
 try: uuid.UUID(session_only); receipt['checks']['empty_session_generates_uuid']=True
 except (ValueError,TypeError,AttributeError): receipt['checks']['empty_session_generates_uuid']=False
 launch=start('enabled',f'roslaunch {args} enable_exploration:=true session_id:=fixture_explicit')
 if not waitfor(lambda:any(x['event']=='goal' for x in rows(out/'fixture.jsonl')),20):raise RuntimeError('enabled native node emitted no goal')
 r,nodes=run('rosnode list');(out/'enabled_nodes.txt').write_text(nodes)
 r,topics=run('rostopic list');(out/'enabled_topics.txt').write_text(topics)
 receipt['checks']['native_explore_node']='/fixture_nav/explore' in nodes
 receipt['checks']['no_cmd_vel_topic']=not any('cmd_vel' in x for x in topics.splitlines())
 receipt['checks']['no_hardware_nodes']=not any(any(k in x.lower() for k in ['can','driver','sensor','camera','laser','scout','serial']) for x in nodes.splitlines())
 run("rostopic pub -1 /fixture/protection std_msgs/String 'data: synthetic-protection-test'",10)
 receipt['checks']['success_and_abort']=waitfor(lambda:{'succeeded','aborted'} <= {x['event'] for x in rows(out/'fixture.jsonl')},30)
 # The first fixture has one frontier; abort may blacklist it and stop exploration.
 # Relaunch the same native algorithm to obtain a separate deliberately held goal.
 run('rosnode kill /fixture_nav/explore',8);time.sleep(1);stop(launch)
 receipt['checks']['native_process_shutdown']= '/fixture_nav/explore' not in run('rosnode list')[1]
 cancel_args=args.replace(shlex.quote('event_log:='+str(out/'observer.jsonl')),shlex.quote('event_log:='+str(out/'cancel_observer.jsonl')))
 launch2=start('enabled_cancel',f'roslaunch {cancel_args} enable_exploration:=true session_id:=fixture_explicit')
 receipt['checks']['held_goal_accepted']=waitfor(lambda:len([x for x in rows(out/'fixture.jsonl') if x['event']=='goal'])>=3,20)
 run('rosnode kill /fixture_nav/explore',8);time.sleep(2)
 prior=rows(out/'cancel_observer.jsonl');before=rows(out/'fixture.jsonl')
 receipt['unsupported_auto_cancel']=not any(x['event']=='cancel' for x in prior) and not any(x['event']=='preempted' and x.get('index',0)>=3 for x in before)
 receipt['checks']['native_exit_observed_without_auto_cancel']=receipt['unsupported_auto_cancel']
 # Fixture-only explicit cancellation tests the action server and observer separately.
 run(shlex.quote(sys.executable)+' '+shlex.quote(str(P/'cancel_owner.py')),10)
 receipt['checks']['explicit_cancel_preempts']=waitfor(lambda:any(x['event']=='preempted' and x.get('index',0)>=3 for x in rows(out/'fixture.jsonl')),8)
 time.sleep(.5);stop(launch2)
 ev=rows(out/'observer.jsonl')+rows(out/'cancel_observer.jsonl');fx=rows(out/'fixture.jsonl');receipt['fixture_events']=fx;receipt['observer_events']=ev
 receipt['checks']['goal_callerid_and_session']=all(x.get('callerid')=='/fixture_nav/explore' and x.get('session_id')=='fixture_explicit' for x in ev if x['event']=='goal')
 transitions=[x for x in ev if x['event']=='status_transition'];last={};dedup=True
 for x in transitions:
  key=(x.get('goal_id'),x.get('session_id'));value=(x.get('status'),x.get('text'))
  if last.get(key)==value:dedup=False
  last[key]=value
 receipt['checks']['status_transitions_deduplicated']=dedup
 receipt['checks']['observer_explicit_owner_cancel']=any(x['event']=='cancel' and x.get('callerid')=='/fixture_supervisor' for x in ev)
 receipt['checks']['observer_preempt_result']=any(x['event']=='result' and x.get('status')==2 for x in ev)
 receipt['checks']['observer_goal']=any(x['event']=='goal' for x in ev)
 receipt['checks']['observer_success']=any(x['event']=='result' and x.get('status')==3 for x in ev)
 receipt['checks']['observer_abort']=any(x['event']=='result' and x.get('status')==4 for x in ev)
 receipt['checks']['observer_cancel']=any(x['event']=='cancel' for x in ev)
 receipt['checks']['observer_protection']=any(x['event']=='protection_event' for x in ev)
 receipt['checks']['observer_stop']=any(x['event']=='recorder_stop' for x in ev)
except Exception as exc:receipt['errors'].append(repr(exc))
finally:
 for name,p in reversed(owned):
  try:stop(p)
  except Exception as exc:receipt['errors'].append(f'{name} cleanup: {exc!r}')
 for f in files:f.close()
 receipt['owned_processes']=[{'name':n,'pid':p.pid,'exit_code':p.poll()} for n,p in owned]
 receipt['all_checks_pass']=bool(receipt['checks']) and all(receipt['checks'].values()) and not receipt['errors']
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'output':str(out),'checks':receipt['checks'],'errors':receipt['errors'],'all_checks_pass':receipt['all_checks_pass']},indent=2));sys.exit(0 if receipt['all_checks_pass'] else 1)
