#!/usr/bin/python3
"""Reviewed root task driver: default read-only plan; --execute resets data stack.

Never sends robot commands or starts a mission/native/adapter. Existing DBs remain.
The caller must stop the prior experiment and confirm zero before --execute.
"""
import argparse,copy,hashlib,importlib.util,json,os,signal,subprocess,time,xmlrpc.client
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('fresh_spec_builder',HERE/'build_launch_spec.py')
B=importlib.util.module_from_spec(s);s.loader.exec_module(B)
A=B.A;W=B.W;REGISTRY=A/'processes.json'

def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(str(tmp),str(path))

def proc(pid):
    try:
        p=Path('/proc')/str(pid);parts=(p/'stat').read_text().rsplit(')',1)[1].split()
        argv=[v.decode(errors='replace') for v in (p/'cmdline').read_bytes().split(b'\0') if v]
        if not argv or parts[0]=='Z':return None
        return dict(pid=int(pid),uid=p.stat().st_uid,ppid=int(parts[1]),start_ticks=int(parts[19]),argv=argv)
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None

def assert_no_actuators():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        q=proc(p.name)
        if q is None:continue
        args=q['argv'];exe=Path(args[0]).name
        worker=any(a in (str(W/'runtime/mission.py'),str(W/'runtime/visual_velocity_adapter.py')) for a in args)
        native=exe in ('nso_scout_attended_node','scout_tether_attended_control_v4')
        if worker or native:found.append(q)
    if found:raise RuntimeError('Actuator/supervisor still present: '+json.dumps(found))

def same(q):
    actual=proc(q['pid'])
    return actual is not None and actual['uid']==q['uid'] and actual['start_ticks']==q['start_ticks'] and actual['argv']==q['argv']

def children(root):
    result=[];pending=[root['pid']];known={root['pid']}
    while pending:
        parent=pending.pop()
        for p in Path('/proc').iterdir():
            if not p.name.isdigit() or int(p.name) in known:continue
            q=proc(p.name)
            if q is not None and q['ppid']==parent:
                if q['uid']!=os.getuid():raise RuntimeError('Different-owner descendant: '+str(q))
                known.add(q['pid']);result.append(q);pending.append(q['pid'])
    return result

def owned_targets(registry):
    started=registry['started'];backend=registry['unified_mapping_backend']
    old_db=next(v.split(':=',1)[1] for v in backend['argv'] if v.startswith('_source_database_path:='))
    old_epoch=next(v.split(':=',1)[1] for v in backend['argv'] if v.startswith('_source_epoch:='))
    byname={v['name']:v for v in started}
    byname.update({v['name']:v for v in registry['unified_viewer']})
    byname['mapping_backend']=backend;byname['tf_hold']=registry['unified_tf_hold']
    targets=[]
    for name in ['navigation','unified_viewer','history_relay','assembler','rtabmap','mapping_backend','tf_hold']:
        entry=byname[name];q=proc(entry['pid'])
        if q is None:continue
        if q['uid']!=os.getuid():raise RuntimeError('Different-owner process: '+name)
        args=q['argv']
        anchors={
            'navigation':[str(W/'config/joint_navigation.launch')],
            'rtabmap':[str(W/'config/rtabmap_loop_shadow.launch'),'database_path:='+old_db],
            'mapping_backend':[str(W/'runtime/rtabmap_mapping_backend.py'),'_source_database_path:='+old_db,'_source_epoch:='+old_epoch],
            'tf_hold':[str(W/'runtime/rtabmap_tf_hold.py')],
            'unified_viewer':[str(A/'unified_rtab_viewer/candidate/unified_viewer.py'),'--epoch-id',old_epoch],
            'history_relay':[str(A/'unified_rtab_viewer/candidate/full_history_relay.py'),'--expected-database',old_db],
            'assembler':[str(A/'unified_rtab_viewer/candidate/unified_rtab.launch'),'expected_database:='+old_db]}
        if not all(v in args for v in anchors[name]):raise RuntimeError('Registry PID/cmd mismatch for '+name+': '+str(q))
        targets.append(dict(name=name,root=q,descendants=children(q)))
    return targets,old_db,old_epoch

def stop_target(target):
    # Each signal addresses a captured exact process, never a group/session.
    root=target['root'];all_processes=[root]+target['descendants']
    for q in all_processes:
        actual=proc(q['pid'])
        if actual is not None and not same(q):raise RuntimeError('PID identity changed before stop: '+str(q))
    if same(root):os.kill(root['pid'],signal.SIGINT)
    limit=time.monotonic()+15.
    while any(same(q) for q in all_processes) and time.monotonic()<limit:time.sleep(.1)
    alive=[q for q in all_processes if same(q)]
    for q in alive:os.kill(q['pid'],signal.SIGTERM)
    limit=time.monotonic()+8.
    while any(same(q) for q in all_processes) and time.monotonic()<limit:time.sleep(.1)
    alive=[q for q in all_processes if same(q)]
    if alive:raise RuntimeError('Owned processes did not close; no automatic SIGKILL: '+json.dumps(alive))
    return dict(name=target['name'],stopped_pids=[q['pid'] for q in all_processes],old_files_deleted=False)

def master_call(master,name,*args):
    result=getattr(master,name)('/nso_fresh_mapping_reset',*args)
    if result[0]!=1:raise RuntimeError('ROS '+name+': '+str(result))
    return result[2]

def snapshot_valid(value,spec):
    return (value.get('schema')=='rtabmap_mapping_backend/1'
        and value.get('source_epoch')==spec['epoch_id']
        and value.get('source_database_path')==spec['new_database_path']
        and value.get('map_frame')=='nso_loop_map' and value.get('alias_frame')=='nso_pair_map'
        and value.get('source_caller')=='/nso_loop/rtabmap'
        and int(value.get('map_stamp_ns',0))>=int(spec['epoch_start_s']*1e9)
        and value.get('map_stamp_ns')==value.get('graph_source_stamp_ns')==value.get('grid_source_stamp_ns')
        and value.get('initial_metadata_receipt') is None and value.get('recorded_epoch_graph') is None
        and any(n.get('odom_from_base') is not None for n in value.get('nodes',[])))

def reset(args):
    old=json.loads(REGISTRY.read_text())
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if old['boot_id']!=boot:raise RuntimeError('Registry boot differs; refresh identities before reset')
    assert_no_actuators()
    # Must freeze live patched navigation environment before its process exits.
    spec=B.build(args.epoch_id,time.time(),environment_from=args.environment_from)
    targets,old_db,old_epoch=owned_targets(old)
    summary=dict(prepared_only=not args.execute,epoch_id=spec['epoch_id'],new_database=spec['new_database_path'],old_database_retained=old_db,stop_names=[t['name'] for t in targets],no_robot_motion=True,no_Grid_RangeMax_override=True)
    if not args.execute:print(json.dumps(summary,indent=2));return
    run=Path(spec['run_directory']);run.mkdir(parents=True,exist_ok=False)
    save(run/'launch_spec.json',spec);save(run/'registry.before.json',old);save(run/'old_owned_processes.json',targets)
    env=dict(os.environ,**spec['environment']);env.update(ROS_DISTRO='noetic',ROS_VERSION='1',ROS_PYTHON_VERSION='3',ROS_ROOT='/opt/ros/noetic/share/ros')
    master=xmlrpc.client.ServerProxy(spec['environment']['ROS_MASTER_URI'])
    started=[];handles=[];stopped=[]
    receipt=dict(schema='root_fresh_RTAB_reset/1',status='starting',started_unix_s=time.time(),boot_id=boot,epoch_id=spec['epoch_id'],old_epoch=old_epoch,old_database_path=old_db,new_database_path=spec['new_database_path'],run_directory=str(run),robot_motion=False)
    def status(phase):
        receipt.update(phase=phase,started=started,stopped=stopped,updated_unix_s=time.time());save(run/'reset_receipt.json',receipt);print(json.dumps(dict(phase=phase,epoch_id=spec['epoch_id'])),flush=True)
    def launch(command):
        assert_no_actuators()
        with (run/(command['name']+'.log')).open('x') as log:
            handle=subprocess.Popen(command['argv'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        handles.append(handle);time.sleep(.15)
        if handle.poll() is not None:raise RuntimeError(command['name']+' exited; read its log')
        q=proc(handle.pid)
        entry=dict(name=command['name'],pid=handle.pid,argv=command['argv'],actual_identity=q,started_unix_s=time.time(),boot_id=boot,source_epoch=spec['epoch_id'],source_database_path=spec['new_database_path'])
        if command['name']=='navigation':entry['environment']=spec['environment']
        started.append(entry);save(run/(command['name']+'.process.json'),entry)
    def wait_for(predicate,label,seconds=45.):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            for h in handles:
                if h.poll() is not None:raise RuntimeError('New data child exited '+str(h.pid))
            try:
                answer=predicate()
                if answer:return answer
            except (FileNotFoundError,json.JSONDecodeError,xmlrpc.client.Fault,ConnectionRefusedError):pass
            time.sleep(.1)
        raise RuntimeError('Timed out: '+label)
    try:
        status('stopping_owned_data_nodes')
        for target in targets:
            assert_no_actuators();stopped.append(stop_target(target))
        status('clearing_only_owned_private_parameters')
        for command in spec['parameter_cleanup_after_stop']:
            namespace=command[-1]
            if master_call(master,'hasParam',namespace):master_call(master,'deleteParam',namespace)
        status('starting_readers_before_source')
        for command in spec['launch_commands']:
            if command['phase']==1:launch(command)
        def readers_registered():
            _,subscribers,_=master_call(master,'getSystemState');subscribers=dict(subscribers)
            required={'/nso_loop/grid_map':'/nso_rtabmap_mapping_backend','/nso_loop/mapGraph':'/nso_rtabmap_mapping_backend','/nso_loop/mapData':'/nso_rtabmap_mapping_backend','/nso_pair/epoch_correction':'/nso_rtabmap_tf_hold','/nso_csm/odom':'/nso_rtabmap_tf_hold'}
            return all(name in subscribers.get(topic,[]) for topic,name in required.items())
        wait_for(readers_registered,'primary reader registration',30.)
        status('starting_new_empty_RTAB_database')
        for command in spec['launch_commands']:
            if command['phase']==2:launch(command)
        def first_snapshot():
            p=Path(spec['new_snapshot_path'])
            if not p.is_file():return None
            value=json.loads(p.read_text());return value if snapshot_valid(value,spec) else None
        snapshot=wait_for(first_snapshot,'first exact new RTAB epoch',60.)
        actual_db=master_call(master,'getParam','/nso_loop/rtabmap/database_path')
        if str(Path(actual_db).resolve())!=spec['new_database_path']:raise RuntimeError('RTAB actual database parameter differs')
        save(run/'first_snapshot.json',snapshot)
        status('starting_current_epoch_history_and_navigation')
        for command in spec['launch_commands']:
            if command['phase']==3:launch(command)
        def history_ready():
            p=run/'full_history_load.json'
            if not p.is_file():return None
            h=json.loads(p.read_text());return h if h.get('database_path')==spec['new_database_path'] and h.get('epoch_id')==spec['epoch_id'] and h.get('source_stamp_ns',0)>=int(spec['epoch_start_s']*1e9) else None
        history=wait_for(history_ready,'new current-session history receipt',45.)
        wait_for(lambda:'/nso_visual/move_base' in dict(master_call(master,'getSystemState')[0]).get('/nso_visual/move_base/status',[]),'fresh move_base status publisher',45.)
        current=copy.deepcopy(old);names={r['name'] for r in started}
        current['started']=[r for r in current['started'] if r['name'] not in names]+started
        byname={r['name']:r for r in started}
        current['prior_RTAB_mapping_epoch']=dict(epoch_id=old_epoch,database_path=old_db,registry_backup=str(run/'registry.before.json'))
        current['localization_epoch_id']=old.get('localization_epoch_id',old.get('CSM_continuous_epoch_id',old.get('epoch_id')))
        current['CSM_continuous_epoch_id']=current['localization_epoch_id'];current['epoch_id']=spec['epoch_id'];current['mapping_epoch_id']=spec['epoch_id']
        current.update(new_localization_epoch=False,new_mapping_epoch=True,CSM_registration_epoch_unchanged_by_mapping_reset=True,old_grid_reused=False,current_mapping_has_prior_manual_history=False,old_map_imported=False,status='fresh_RTAB_data_ready_no_actuator',fresh_mapping_reset_receipt=str(run/'reset_receipt.json'),last_registry_update_unix_s=time.time())
        current['unified_mapping_backend']=byname['mapping_backend'];current['unified_tf_hold']=byname['tf_hold'];current['unified_viewer']=[byname[k] for k in ['assembler','history_relay','unified_viewer']]
        current['rtabmap']=dict(session=spec['epoch_id'],fresh_database=True,database_path=spec['new_database_path'],publish_tf=False,navigation_authority=True,old_database_reused=False)
        current['source_routing'].update(RTAB='filtered',primary_mapping_profile='fresh_RTAB_exact_epoch_no_old_bootstrap',pair_map='new RTAB grid exactgraph alias /nso_pair/map; no imported prior map',research='new RTAB map and measured relative CSM source geometry; prior research history forbidden')
        current['source_bindings'].update(spec['source_bindings']);current['initial_map']=dict(source_epoch=spec['epoch_id'],source_database_path=spec['new_database_path'],map_stamp_ns=snapshot['map_stamp_ns'],graph_revision_sha256=snapshot['graph_revision_sha256'],metadata_coverage=snapshot.get('metadata_coverage'),node_count=len(snapshot.get('nodes',[])),source='first actual fresh RTAB snapshot; not prior manual map')
        current['last_health']=None;current['fresh_map_samples']=[];save(REGISTRY,current)
        range_key='/nso_loop/rtabmap/Grid/RangeMax'
        receipt.update(status='ready',new_first_map_stamp_ns=snapshot['map_stamp_ns'],first_node_count=len(snapshot.get('nodes',[])),history_signature_count=history.get('signature_count'),initial_metadata_imported=False,old_graph_bootstrap_imported=False,CSM_continuous_epoch_id=current['localization_epoch_id'],range_max_changed=False,Grid_RangeMax=master_call(master,'getParam',range_key) if master_call(master,'hasParam',range_key) else 'not explicitly set; existing core default retained',registry=str(REGISTRY),actuator_started=False)
        status('ready_for_root_new_capture_and_primitive_pilot')
    except BaseException as exc:
        receipt.update(status='failed',error=type(exc).__name__+': '+str(exc));status('failed_no_automatic_rollback_or_trial')
        failed=copy.deepcopy(old);failed.update(status='fresh_mapping_reset_failed',fresh_mapping_reset_receipt=str(run/'reset_receipt.json'),pending_mapping_epoch=spec['epoch_id'],new_data_processes=started,last_registry_update_unix_s=time.time());save(REGISTRY,failed)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch-id',default='feedback_fresh_20261115_01');p.add_argument('--environment-from',type=Path);p.add_argument('--execute',action='store_true');reset(p.parse_args())
