"""Software reconstruction of original S3 prediction epochs, not new samples."""
import json,copy,types
from pathlib import Path
import numpy as np
from nso_bridge.primitive_feedback_unbounded_v1 import FrozenEpochPrimitivePilot
REPORT=Path('/home/ouc/nso_robot/reports/nso-hybrid-20261115/NSO_primitive_feedback_fresh_S_20261115_01')
def build_original_actual_pilot(output):
 lines=[json.loads(x)for x in (REPORT/'joint/decisions.jsonl').read_text().splitlines()]
 settings=copy.deepcopy(lines[0]['request']['params']);semantic=copy.deepcopy(settings['semantic_configuration']);settings['semantic_configuration']=None;settings['semantic_output']=str(REPORT/'joint/semantics');settings['execution_allowed']=False;settings['feedback_output']=str(output)
 pilot=FrozenEpochPrimitivePilot(**settings);cache={}
 for path in (REPORT/'joint/semantics').glob('*.json'):
  receipt=json.loads(path.read_text())
  with np.load(receipt['mask_file'],allow_pickle=False)as z:labels=z['semantic'].copy();owner=z['instance_owner'].copy()if'instance_owner'in z else None
  cache[str(receipt['frame_id'])]=dict(labels=labels,owner=owner,receipt=receipt)
 def original_cached(this,z,meta,sha):
  item=cache[str(meta['frame_id'])];this._instance_owner_map=None if item['owner']is None else item['owner'].copy();this._attempt_semantic_receipt=copy.deepcopy(item['receipt']);return item['labels'].copy()
 pilot.inner._semantic_labels=types.MethodType(original_cached,pilot.inner)
 observations=[x for x in lines if x['request']['method']=='observe_primitive_result'and x['response']['ok']]
 choice=next(x['response']['result']for x in lines if x['request']['method']=='next_action')
 pilot.observe_primitive_result(**observations[0]['request']['params']);state=pilot.inner.runtime.states[0];scene=pilot.inner.backend.scenes[0]
 state['option']=copy.deepcopy(choice['selected']);state['active_actions']=copy.deepcopy(choice['selected']['outbound_actions']);state['phase']='outbound';scene['plans']=1;scene['selected']=copy.deepcopy(choice['selected']);scene['last_selection']=copy.deepcopy(choice['selection'])
 for row in observations[1:]:
  pilot.next_action();value=pilot.observe_primitive_result(**row['request']['params'])
  if value['primitive_outcome']['gain_snapshot']!=row['response']['result']['primitive_outcome']['gain_snapshot']:raise ValueError('reconstructed original prediction-epoch statistics differ from actual saved event')
 pilot.inner.primitive_outcomes=copy.deepcopy(observations[-1]['response']['result']['summary']['primitive_outcomes'])
 pilot.inner.semantic_configuration=semantic  # Cached real labels still required; no inference.
 return pilot,observations,cache,settings
