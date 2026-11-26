#!/usr/bin/python3
"""Write a root-review launch specification only; never starts/stops ROS.

The existing RTAB wrapper itself creates the fresh DB session at execution.
This generator does not create that session, write ROS params or launch tools.
"""
import argparse,hashlib,json,os,re,time
from pathlib import Path
BASE=Path('/home/ouc/nso_robot')
W=BASE/'workspaces/nso_hybrid_20261029'
A=BASE/'analyses/resume_20261115'
SELF=Path(__file__).resolve().parent
CPU='0,1,4,5,6,7'

def build(epoch,epoch_start_s,host='192.168.31.216',environment_from=None):
    if not re.fullmatch('[A-Za-z0-9_-]+',epoch):raise ValueError('simple unique epoch required')
    if not isinstance(epoch_start_s,(int,float)) or epoch_start_s<=0:raise ValueError('positive source epoch lower bound required')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    db=BASE/'experiments/rtabmap_loop_shadow'/boot/epoch/'rtabmap.db'
    if db.parent.exists():raise ValueError('new mapping session path already exists: '+str(db.parent))
    run=SELF/'runs'/epoch
    snap=run/'latest_snapshot.json';history=run/'full_history_load.json'
    registry=json.loads((A/'processes.json').read_text())
    nav=next(e for e in registry['started'] if e['name']=='navigation')
    if environment_from is None:
        raw=(Path('/proc')/str(nav['pid'])/'environ').read_bytes()
        inherited=dict(x.split('=',1) for x in raw.decode().split('\0') if '=' in x)
        environment_source=dict(kind='actual_navigation_process',pid=nav['pid'])
    else:
        environment_file=Path(environment_from).resolve()
        inherited=json.loads(environment_file.read_text())['environment']
        environment_source=dict(kind='reviewed_saved_launch_spec',path=str(environment_file),sha256=hashlib.sha256(environment_file.read_bytes()).hexdigest())
    required=['ROS_PACKAGE_PATH','LD_LIBRARY_PATH','CMAKE_PREFIX_PATH','PYTHONPATH']
    if any(not inherited.get(k) for k in required):raise ValueError('live navigation environment unavailable')
    env={k:inherited[k] for k in required}
    env.update(ROS_MASTER_URI='http://127.0.0.1:11382',ROS_HOSTNAME='127.0.0.1',ROS_IP='127.0.0.1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    package_paths=[str(W/'loop_module'),'/home/ouc/hjd/catkin_ws/src']+env['ROS_PACKAGE_PATH'].split(':')
    env['ROS_PACKAGE_PATH']=':'.join(dict.fromkeys(package_paths))
    py=['/usr/bin/taskset','-c',CPU,'/usr/bin/python3','-s','-B']
    roslaunch=['/usr/bin/taskset','-c',CPU,'/opt/ros/noetic/bin/roslaunch']
    viewer=A/'unified_rtab_viewer/candidate'
    semantic=Path(registry['semantic_output'])/'semantic_state.json'
    commands=[
      dict(name='mapping_backend',phase=1,argv=py+[str(W/'runtime/rtabmap_mapping_backend.py'),
        '_latest_snapshot_path:='+str(snap),'_source_database_path:='+str(db),'_source_epoch:='+epoch,
        '_initial_metadata_path:=""','_initial_epoch_graph_path:=""','_initial_epoch_graph_manifest_path:=""','_publish_held_tf:=false'],
        requirement='Start before fresh RTAB; no old bootstrap or reused snapshot; explicit empty private params'),
      dict(name='tf_hold',phase=1,argv=py+[str(W/'runtime/rtabmap_tf_hold.py')],
        requirement='Old correction publisher/holder already exited; starts with no correction'),
      dict(name='assembler',phase=1,argv=roslaunch+[str(viewer/'unified_rtab.launch'),'expected_database:='+str(db),'epoch_id:='+epoch,'epoch_start_s:='+str(epoch_start_s),'receipt:='+str(history)],
        requirement='New process to discard earlier map/point caches; waits for history relay'),
      dict(name='unified_viewer',phase=1,argv=py+[str(viewer/'unified_viewer.py'),'--epoch-id',epoch,'--epoch-start-s',str(epoch_start_s),'--semantic-state',str(semantic),'--history-receipt',str(history),'--host',host,'--port','8766'],
        requirement='New process discards old graph/map/cloud; existing camera upstream retained, semantic packets before epoch rejected'),
      dict(name='rtabmap',phase=2,argv=['/usr/bin/taskset','-c',CPU,str(W/'runtime/run_rtabmap_loop_shadow.sh'),epoch,'--mode','mapping','--scan-topic','/nso_scan/self_filtered'],
        requirement='Run only after primary backend+TFholder subscribers registered. Wrapper creates new absent DB; mapping cannot import source DB'),
      dict(name='history_relay',phase=3,argv=py+[str(viewer/'full_history_relay.py'),'--expected-database',str(db),'--epoch-id',epoch,'--epoch-start-s',str(epoch_start_s),'--receipt',str(history)],
        requirement='After first live exact snapshot; otherwise empty GetMap header can make relay exit. Service pulls current DB only'),
      dict(name='navigation',phase=3,argv=roslaunch+[str(W/'config/joint_navigation.launch'),'latch_xy_goal_tolerance:=true','odom_topic:=/nso_csm/control_odom','map_topic:=/nso_pair/map','navigation_params:='+str(W/'config/joint_navigation_teb_rtabmap_062x059_v1.yaml')],
        requirement='New move_base clears costmaps/TFbuffer and namespace parameters via clear_params=true; retained current patchedTEB+privateGlobalPlanner environment'),
    ]
    source_paths=[W/'runtime/rtabmap_mapping_backend.py',W/'runtime/rtabmap_tf_hold.py',W/'runtime/rtabmap_loop_session.py',W/'runtime/run_rtabmap_loop_shadow.sh',W/'config/rtabmap_loop_shadow.launch',W/'config/joint_navigation.launch',W/'config/joint_navigation_teb_rtabmap_062x059_v1.yaml',viewer/'unified_rtab.launch',viewer/'full_history_relay.py',viewer/'unified_viewer.py']
    return dict(schema='fresh_matched_RTAB_launch_spec/1',generated_unix_s=time.time(),prepared_only=True,actuator_launched=False,boot_id=boot,epoch_id=epoch,epoch_start_s=epoch_start_s,new_database_path=str(db),database_session_preexisting=False,run_directory=str(run),new_snapshot_path=str(snap),old_bootstrap_imported=False,environment=env,
      kept=['camera/current extrinsics','LiDAR hardware/driver/self-filter','CSM registration + rawodom bridge/control feedback','hybrid sensor bridge','capture-only camera dashboard/upstream semantic display'],
      root_stop_before_start=['current mission+native+adapter (confirmed zero first)','move_base navigation process','unified_viewer+history_relay+assembler','RTAB wrapper+rtabmap+rgbd_sync+loopmonitor','mapping_backend+tf_hold'],
      parameter_cleanup_after_stop=[['rosparam','delete','/nso_loop'],['rosparam','delete','/nso_rtabmap_mapping_backend'],['rosparam','delete','/nso_rtabmap_tf_hold']],
      parameter_cleanup_note='Prepared command arguments only; root handles absent namespaces and enumerates actual process identities. No global param deletion; no commands executed by this generator.',
      launch_commands=commands,
      first_data_contract=dict(map_source='/nso_loop/grid_map',graph_source='/nso_loop/mapGraph or enclosing MapData header',map_alias='/nso_pair/map',map_frame='nso_loop_map',alias_frame='nso_pair_map',expected_RTAB_caller='/nso_loop/rtabmap',expected_correction_caller='/nso_rtabmap_mapping_backend',expected_database=str(db),expected_epoch=epoch,earliest_source_time_s=epoch_start_s,initial_metadata_path='',initial_epoch_graph_path='',initial_epoch_graph_manifest_path='',no_metadata_service_bootstrap_needed='All node raw poses are observed from first live MapData; do not start source first then recover from old latched graph'),
      root_before_motion=['Verify new snapshot source_database_path/source_epoch/map_stamp >= epoch lowerbound; currentrawnode metadata present','Verify one aliasTFowner /nso_rtabmap_tf_hold at realCSMstamps and fresh /nso_pair/map provenance','Create new RGBD/scan packet and new research session/P matrix in this alias frame; nevercopy prior planning transform/records/semantic_cache','Record actual initial RTAB map+graph+packet hashes and physicalstartplace/orientation; then execute declared G orS mode'],
      new_trial_configuration_bindings=dict(geometry_pose_backend='rtabmap_optimized_replay_v1',
        rtabmap_graph_snapshot_path=str(snap),restore_epoch_id=epoch,
        navigation_map_initialization='fresh own RTAB mapping DB; no prior manual/exploration map imported',
        current_input='New actual measured packet from this epoch, created after live source routing is established',
        planning_from_odom='Must derive from new actual alias-frame capture; do not copy prior P matrix',
        prior_observation_records_imported=False,prior_semantic_cache_imported=False,
        paired_map_epoch_receipt='Replace old registry/provenance bundle with actual fresh-source launch and first-snapshot receipts',
        stale_provenance_keys_to_remove=['warm_history_candidate','warm_history_paid_actions','warm_history_arrival_credit']),
      navigation_environment_source=environment_source,source_bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
      boundary='This resets mapping state only. It neither validates policy reachability nor creates G/S efficacy evidence. Root alone may execute reviewed commands.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--epoch-id',required=True);p.add_argument('--epoch-start-s',type=float,default=None);p.add_argument('--environment-from',type=Path,help='Reviewed saved launch spec; use if prior navigation process has already exited');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=build(a.epoch_id,time.time() if a.epoch_start_s is None else a.epoch_start_s,environment_from=a.environment_from)
    with a.output.open('x') as f:json.dump(spec,f,indent=2);f.write('\n')
    print(str(a.output))
if __name__=='__main__':main()
