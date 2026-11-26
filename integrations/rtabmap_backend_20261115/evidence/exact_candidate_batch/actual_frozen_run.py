import sys,json,time,cProfile,pstats,io
from pathlib import Path
import numpy as np
W=Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029');sys.path.insert(0,str(W/'nso_bridge'))
from nso_bridge.rtabmap_viewpoint_session import RtabmapViewpointSession
O=Path('/home/ouc/nso_robot/analyses/resume_20261115/NSO_rtabmap_current0_20261115');D=Path('/home/ouc/nso_robot/analyses/resume_20261115/exact_candidate_batch/actual_frozen');D.mkdir(exist_ok=True)
import importlib.util
p=Path('/home/ouc/nso_robot/analyses/resume_20261115/exact_candidate_batch/candidate/nso_bridge/exact_candidate_batch_v1.py');spec=importlib.util.spec_from_file_location('candidate',p);candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate);candidate.install(candidate.PROFILE)
settings=json.loads((O/'configuration.json').read_text());settings['semantic_configuration']=None
session=RtabmapViewpointSession(**settings)
receipt=json.loads(next((O/'joint/semantics').glob('*.json')).read_text())
with np.load(next((O/'joint/semantics').glob('*.npz')),allow_pickle=False) as z:labels=z['semantic'].copy()
session.semantic_cache[str(receipt['frame_id'])]=dict(labels=labels,owner=None,receipt=receipt)
path=json.loads((O/'capture.json').read_text())['joint']
session.observe_viewpoint_result(path=path,attempt_id=0,ros_status=None,reason='bootstrap_actual_observation_readonly_preview')
started=time.monotonic();result=session.next_viewpoint(remaining_time_s=None,return_requested=False);elapsed=time.monotonic()-started
baseline=json.loads((O/'preview_results.json').read_text())['next_viewpoint']['response']
summary=dict(scope='same frozen current0 and original saved canonical labels; optional exact_candidate_batch_v1; noROS/inference',wall_s=elapsed,baseline_wall_s=118.53385903401067,target_exact_equal=result.get('target')==baseline.get('target'),selection_exact_equal=result.get('selection')==baseline.get('selection'),target=result.get('target',{}).get('target_odom_pose'),candidate_compute_profile=candidate.PROFILE)
(D/'receipt.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
if not summary['target_exact_equal'] or not summary['selection_exact_equal']:
 (D/'mismatch.json').write_text(json.dumps(dict(actual=result,baseline=baseline)))
