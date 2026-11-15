import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,pathlib,json,tempfile,numpy as np,copy
w=pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(w/'nso_bridge'))
from nso_bridge.rtabmap_viewpoint_session import RtabmapViewpointSession
p=pathlib.Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_surroundings_20261115_07');cfg=json.loads((p/'joint_configuration.json').read_text());es=[json.loads(x)for x in(p/'events.jsonl').read_text().splitlines()];events=[e for e in es if e.get('event')=='NSO_viewpoint_actual_observation'][:2]
tmp=tempfile.TemporaryDirectory();td=pathlib.Path(tmp.name);source=[];nodes=[]
for i,e in enumerate(events):
 with np.load(e['path'],allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
 receipt=json.loads(str(arrays['receipt_json'].item()));T=receipt['historical_tf']['odom_from_base']['matrix'];nodes.append(dict(id=i+1,stamp_ns=receipt['headers']['depth']['stamp_ns'],odom_from_base=T,map_from_base=T));source.append(arrays)
snap=dict(graph_revision_sha256='actual-source-identity-test-only',graph_source_stamp_ns=nodes[-1]['stamp_ns'],nodes=nodes)
paths=[]
for i,arrays in enumerate(source):
 meta=json.loads(str(arrays['joint_metadata_json'].item()));meta.update(geometry_world_frame='nso_pair_map',rtabmap_graph_snapshot=snap);arrays['joint_metadata_json']=np.asarray(json.dumps(meta));f=td/f'source{i}.npz';np.savez_compressed(f,**arrays);paths.append(f)
s=RtabmapViewpointSession(cfg['configuration'],calibration=cfg['calibration'],semantic_configuration=None,execution_allowed=False,replay_directory=td/'replay')
r=s.observe_viewpoint_result(paths[0],0);s.inner.viewpoint_pending=dict(attempt_id=1,pose=[200,200,0],target_odom_pose=[0,0,0],selected=None)
r=s.observe_viewpoint_result(paths[1],1,4,'offline original actual failed observation smoke')
assert s.summary()['waypoint_attempts']==1 and s.summary()['waypoint_ROS_successes']==0
print(json.dumps(dict(passed=True,actual_source_observations=2,attempts=s.summary()['waypoint_attempts'],ROS_successes=s.summary()['waypoint_ROS_successes'],worlds_constructed=0,chooser_calls=0,scope='nonexecuting real original v4 mapper/feedback bootstrap and failed-attempt replay API smoke; no method/accuracy claim')))
