"""One remaining40/total60 frozen resource diagnostic; no planner side effects."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from nso_bridge.exploration_joint_policy import generate_options
from nso.cpu_sensor_contract_v10 import json_value


def _fingerprint(session):
    s=session.backend.scenes[0]
    p=s['packet']
    arrays={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in [
        ('belief',s['mapper'].belief),('camera_seen',s['mapper'].camera_seen)]}
    data=dict(arrays=arrays,frames=s['mapper'].frames,remaining=s['ledger'].remaining_budget,
        total=s['ledger'].total_budget,plans=s['plans'],calls=len(session.backend.calls),
        packet=[*p.position,p.heading,p.action_id,p.frame_id],
        graph_version=s['graph_version'],semantic_version=s['semantic_version'],
        selected=s.get('selected'),planning_direction_support=s.get('planning_direction_support'),
        gain=s['gain'].snapshot(),runtime_state=session.runtime.states[0],
        history=s['ledger'].planning_camera_poses())
    return hashlib.sha256(json.dumps(json_value(data),sort_keys=True,separators=(',',':'),
        default=lambda value: value.sha256() if callable(getattr(value,'sha256',None)) else repr(value)).encode()).hexdigest()


def audit(session, output_path):
    started=perf_counter();s=session.backend.scenes[0];backend=session.backend;p=s['packet']
    assert s['mapper'].frames==21 and s['ledger'].remaining_budget==20
    assert s['ledger'].total_budget==40 and tuple((*p.position,p.heading))==(200,218,0)
    assert s['semantic_version']==s['graph_version']==21
    assert backend.planner_revision=='v10_3_1'
    before=_fingerprint(session)
    history=s['ledger'].planning_camera_poses()
    attempted=s['gain'].attempted_camera_mask(s['mapper'],history)
    kwargs=dict(max_candidates=int(getattr(backend.args,'cpu_max_candidates',12)),
        coverage_strategy='total_diverse',coverage_slots=int(getattr(backend.args,'cpu_coverage_slots',4)),
        attempted_camera_mask=attempted)
    routes,candidate_audit=generate_options(s['mapper'],p.position,p.heading,40,s['return_anchor'],**kwargs)
    generated_ids=[r['candidate_id'] for r in routes]
    graph=s['graph']
    routes=[r for r in routes if graph['current_region']>0 and graph['current_region']==graph['anchor_region']
        ==int(graph['regions'][tuple(r['pose'][:2])])]
    scores,rows=backend._scores_v10_1(s,routes,attempted) if routes else ({m:[] for m in ['G','S','X']},[])
    choices={}
    for mode in ['G','S','X']:
        best=min(range(len(routes)),key=lambda i:(-scores[mode][i],routes[i]['cost'],routes[i]['candidate_id'])) if routes else None
        best=None if best is None or scores[mode][best]<=0 else best
        choices[mode]=None if best is None else dict(candidate_id=routes[best]['candidate_id'],
            group=routes[best]['group'],score=float(scores[mode][best]),cost=routes[best]['cost'],
            target=routes[best]['pose'],first_action=routes[best]['outbound_actions'][0],
            outbound_actions=routes[best]['outbound_actions'])
    after=_fingerprint(session)
    assert before==after,'candidate/scoring diagnostic mutated native state'
    here=Path(__file__).resolve().parent
    # Reuse only the original measured-cost extraction; do not invoke availability again.
    report=here.parent/'source'
    actual=json.loads((report/'result.json').read_text());running=json.loads((report/'running.json').read_text())
    a=actual['successful_arrivals'][19]
    result=dict(schema='nso.prefix20_remaining40_original_candidate_score.v1',
        scope='Resource counterfactual on actual21-observation prefix; not executed S60, a new observation, matched physical comparison or accuracy validation',
        observed_prefix_actions=20,observations=21,current_state=[*p.position,p.heading],
        actual_base_pose_planning=p.base_pose_planning,anchor=s['return_anchor'],
        original_remaining_budget=20,original_total_budget=40,hypothetical_remaining_budget=40,hypothetical_total_budget=60,
        prefix_acquisition=dict(paid_actions=20,CAN_distance_m=a['can_end']['distance_m']-actual['successful_arrivals'][0]['can_start']['distance_m'],
            elapsed_to_arrival20_pose_source_s=a['after']['source_stamp_s']-running['started_unix_s'],
            timing_limit='source timestamp minus running wall start, not exact monotonic elapsed or observation21 completion'),
        generate_kwargs={k:v for k,v in kwargs.items() if k!='attempted_camera_mask'},
        attempted_sha256=hashlib.sha256(np.ascontiguousarray(attempted).tobytes()).hexdigest(),
        history_count=len(history),gain_posterior=s['gain'].snapshot()['posterior_mean'],
        generated_candidate_ids=generated_ids,kept_after_original_region_filter=[r['candidate_id'] for r in routes],
        candidate_audit=candidate_audit,candidates=routes,
        scores={m:scores[m] for m in ['G','S','X']},score_audit=rows,choices=choices,
        selection_rule='min(-score, full outbound+return cost, candidate_id); no selection if best score<=0',
        state_before_sha256=before,state_after_sha256=after,read_only_state_unchanged=True,
        no_select_target_called=True,elapsed_s=perf_counter()-started,
        source_sha256={inspect.getfile(f):hashlib.sha256(Path(inspect.getfile(f)).read_bytes()).hexdigest()
            for f in [generate_options,backend._scores_v10_1]},
        limits=['Native entry/deep projection/quota is reported, not overridden.',
            'No ledger, actual pose, gain/history, candidate settings, circle, scoring or live runtime changes.'])
    output_path=Path(output_path)
    if output_path.suffix!='.json':output_path=output_path/'candidates_remaining40.json'
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(json_value(result),ensure_ascii=False,indent=2)+'\n')
    return result
