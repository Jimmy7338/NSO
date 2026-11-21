#!/usr/bin/python3
"""Launch one explicitly requested episode; no restart loop or motion overrides."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

base=Path('/home/ouc/nso_robot')
episode_dir=base/'analyses/resume_20261115'/sys.argv[1]
configuration=json.loads((episode_dir/'configuration.json').read_text())
protocol=json.loads((episode_dir/'protocol.json').read_text())
alignment_profile=protocol.get('ordinary_reverse_alignment_profile','original_nose_v1')
local_controller_profile=protocol.get('local_controller_profile','legacy_dwa_v1')
assert local_controller_profile in ('legacy_dwa_v1','teb_slow_polygon_v1','teb_onsite_062x059_v1')
assert alignment_profile in ('original_nose_v1','centered_v1','teb_native_v1')
assert (alignment_profile=='teb_native_v1') == (local_controller_profile in ('teb_slow_polygon_v1','teb_onsite_062x059_v1'))
native_cpu=protocol.get('native_process_cpu')
native_sender_cpu=protocol.get('native_sender_cpu')
native_sender_rt_priority=protocol.get('native_sender_rt_priority')
adapter_cpu=protocol.get('adapter_cpu')
execution_controller=protocol.get('execution_controller','config')
assert execution_controller in ('config','primitive_feedback')
pilot_attempts=protocol.get('primitive_feedback_pilot_attempts')
assert pilot_attempts is None or (execution_controller=='primitive_feedback' and type(pilot_attempts) is int and pilot_attempts>0)
assert native_cpu is None or native_sender_cpu is None
assert native_sender_rt_priority is None or native_sender_cpu is not None
background_cpus=protocol.get('mission_and_other_workload_cpus')
cpu_prefix=[] if background_cpus is None else ['/usr/bin/taskset','-c',','.join(map(str,background_cpus))]
episode=configuration['configuration']['episode_id']
report=base/'reports/nso-hybrid-20261115'/episode
run=episode_dir/'run'
assert not report.exists() and not run.exists()
for d in Path('/proc').iterdir():
    if not d.name.isdigit():continue
    try:
        args=(d/'cmdline').read_bytes().split(b'\0')
        if any(a.endswith(b'/runtime/mission.py') for a in args):
            raise RuntimeError('another mission still exists: '+d.name)
    except FileNotFoundError:continue
run.mkdir()
workspace=base/'workspaces/nso_hybrid_20261029'
recordcmd=cpu_prefix+['/usr/bin/python3','-s','-B',str(base/'analyses/full_coverage_20261112/passive_coverage_recorder.py'),'--output',str(episode_dir/'coverage')]
with (run/'coverage.log').open('x') as log:
    recorder=subprocess.Popen(recordcmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'coverage_process.json').write_text(json.dumps(dict(pid=recorder.pid,argv=recordcmd,started_unix_s=time.time(),no_actuator=True,key_map_snapshots_only=True,recording_time_limit=None),indent=2)+'\n')
argv=['/bin/bash',str(workspace/'run_hybrid.sh'),'--mode','nso','--execution-controller',execution_controller,'--joint-configuration',str(episode_dir/'configuration.json'),'--duration-profile','exploration_until_stopped_v1','--ros-allow-reverse','--ros-reverse-alignment-profile',alignment_profile,'--odometry-jump-profile','source_owned_diagnostic_v1','--feedback-speed-profile','measured_020_v1','--sensor-loss-profile','recoverable_sensor_pause_v1','--execute','--report',str(report)]
if local_controller_profile in ('teb_slow_polygon_v1','teb_onsite_062x059_v1'):
    index=argv.index('--ros-reverse-alignment-profile')
    del argv[index:index+2]
    argv+=['--local-controller-profile',local_controller_profile]
if native_cpu is not None:argv+=['--native-cpu',str(native_cpu)]
if native_sender_cpu is not None:argv+=['--native-sender-cpu',str(native_sender_cpu)]
if native_sender_rt_priority is not None:argv+=['--native-sender-rt-priority',str(native_sender_rt_priority)]
if adapter_cpu is not None:argv+=['--adapter-cpu',str(adapter_cpu)]
if pilot_attempts is not None:argv+=['--primitive-feedback-pilot-attempts',str(pilot_attempts)]
argv=cpu_prefix+argv
started=time.time()
with (run/'mission.log').open('x') as log:
    proc=subprocess.Popen(argv,cwd=str(workspace),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=proc.pid,started_unix_s=started,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),report=str(report),argv=argv,time_limit=None,waypoint_limit=None,roundtrip_budget=None,automatic_return=False,reverse_alignment_profile=alignment_profile,protocol_sha256=hashlib.sha256((episode_dir/'protocol.json').read_bytes()).hexdigest())
(run/'process.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
priority_receipt=dict(workers=[],control_and_native_untouched=True)
deadline=time.monotonic()+12
while time.monotonic()<deadline and proc.poll() is None:
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            args=[a.decode() for a in (d/'cmdline').read_bytes().split(b'\0') if a]
            if 'nso_bridge.joint_engine' not in args:continue
            status=(d/'status').read_text()
            ppid=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
            if ppid!=proc.pid:continue
            pid=int(d.name)
            os.sched_setaffinity(pid,{4,5,6,7})
            os.setpriority(os.PRIO_PROCESS,pid,10)
            priority_receipt['workers'].append(dict(pid=pid,argv=args,nice=os.getpriority(os.PRIO_PROCESS,pid),affinity=sorted(os.sched_getaffinity(pid))))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    if priority_receipt['workers']:break
    time.sleep(.2)
(run/'science_resources.json').write_text(json.dumps(priority_receipt,indent=2)+'\n')
priorities=base/'PROJECT_PRIORITIES.json'
d=json.loads(priorities.read_text());d['updated_unix_s']=time.time()
d['current_active_robot_run'].update(status='running',report=str(report),process_receipt=str(run/'process.json'),recorder=str(run/'coverage_process.json'),actuator_running=True,reverse_alignment_profile=alignment_profile)
d['resume_20261115'].update(status='NSOv4_'+episode+'_running',report=str(report),protocol=str(episode_dir/'protocol.json'),motion_process=str(run/'process.json'),coverage_recorder=str(run/'coverage_process.json'),navigation_map_retained_from_episode01=bool(protocol.get('retained_navigation_map',False)),independent_matched_repeat=False,reverse_alignment_profile=alignment_profile,full_coverage_complete=False)
priorities.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(priority_receipt),flush=True)
