"""Bind observed regional proposals before the native local route selector."""

from copy import deepcopy
import inspect

import numpy as np
from scipy.sparse.csgraph import dijkstra


def certify(options, existing, state, source):
    """Materialize proposals using the native directed known-free, round-trip graph."""
    if not options:
        return [], []
    mapper, packet = state['mapper'], state['packet']
    current = (*packet.position, packet.heading)
    anchor = tuple(state['return_anchor'])
    budget = state['ledger'].remaining_budget
    scale = mapper.config.robot_radius_m / mapper.config.resolution_m
    safe = ~source.inflated_obstacles(mapper.belief != 0, scale)
    regions = state['graph']
    checked, bound = [], []
    if not safe[current[:2]] or not safe[anchor[:2]]:
        return [], [{'pose': row['pose'], 'status': 'unsafe_start_or_return_anchor'} for row in options]
    graph, cells, ids = source.orientation_graph(safe)
    start, home = (4 * int(ids[p[:2]]) + p[2] for p in (current, anchor))
    outbound, prev = dijkstra(graph, directed=True, indices=start, return_predecessors=True)
    returning, rev = dijkstra(graph.T.tocsr(), directed=True, indices=home, return_predecessors=True)
    old_poses = {tuple(row['pose']): row['candidate_id'] for row in existing}
    next_id = max((row['candidate_id'] for row in existing), default=-1) + 1
    q = mapper.quality_evidence(max_points=10000)
    points = np.empty((0, 3)) if q is None else np.asarray(q['point'])
    for proposal in options:
        target = tuple(proposal['pose'])
        audit = {'pose': list(target), 'proposal': proposal}
        if target in old_poses:
            checked.append(audit | {'status': 'existing_candidate', 'candidate_id': old_poses[target]})
            continue
        if not (len(target)==3 and 0<=target[0]<safe.shape[0] and 0<=target[1]<safe.shape[1] and 0<=target[2]<4):
            checked.append(audit | {'status': 'outside_observed_grid'})
            continue
        if not safe[target[:2]]:
            checked.append(audit | {'status': 'footprint_not_known_safe'})
            continue
        if not (regions['current_region']>0 and regions['current_region']==regions['anchor_region']
                ==int(regions['regions'][target[:2]])):
            checked.append(audit | {'status': 'native_observed_return_component_mismatch'})
            continue
        node = 4 * int(ids[target[:2]]) + target[2]
        a, b = outbound[node], returning[node]
        if not np.isfinite(a+b) or a<=0 or a+b>budget:
            checked.append(audit | {'status': 'directed_round_trip_not_feasible_or_over_budget',
                                    'outbound_actions': int(a) if np.isfinite(a) else None,
                                    'return_actions': int(b) if np.isfinite(b) else None,
                                    'remaining_paid': budget})
            continue
        go = source._path(prev,start,node,cells)
        back = source._path(rev,home,node,cells,reverse=True)
        go_cmd, back_cmd = source._actions(go), source._actions(back)
        assert go[0]==current and go[-1]==target and back[0]==target and back[-1]==anchor
        assert (len(go_cmd),len(back_cmd))==(int(a),int(b)) and len(go_cmd)+len(back_cmd)<=budget
        camera = source.camera_pose(target[:2],target[2],mapper.config,mapper.shape[0])
        row=dict(candidate_id=next_id, group='observed_regional_view',asset_index=None,pose=list(target),
                 states=[list(p) for p in go+back[1:]],actions=go_cmd+back_cmd,outbound_actions=go_cmd,
                 return_actions=back_cmd,return_states=[list(p) for p in back],outbound_states=[list(p) for p in go],
                 cost=len(go_cmd)+len(back_cmd),outbound_cost=len(go_cmd),return_cost=len(back_cmd),
                 arrival_action=len(go_cmd),return_anchor=list(anchor),is_coverage_anchor=False,
                 endpoint_measured_fov_support=source._fov_support(points,camera,mapper.config),
                 regional_proposal=proposal)
        checked.append(audit | {'status':'admitted', 'candidate_id':next_id,
                                'outbound_actions':len(go_cmd),'return_actions':len(back_cmd)})
        bound.append(row)
        old_poses[target]=next_id
        next_id+=1
    return bound,checked


def install(components,runtime,planner):
    from nso.cpu_sensor_contract_v10 import digest,json_value

    backend=components._cpu_backend
    original_selector=backend.select_target
    source=inspect.getmodule(original_selector.__func__)
    history=[]
    guard_events=[]
    dispatched=set()
    before_observe=runtime.observe
    before_guard=components.assess_local_action
    site=backend.scenes[0]

    def sensing_state(state):
        packet=state['packet']
        return state['mapper'].belief,(*packet.position,packet.heading),packet.action_id

    planner.observe(*sensing_state(site))

    def observe(scene_index,*args,**kwargs):
        receipt=before_observe(scene_index,*args,**kwargs)
        planner.observe(*sensing_state(backend.scenes[scene_index]))
        return receipt

    def guard(scene_index,action):
        result=before_guard(scene_index,action)
        executed=backend.scenes[scene_index].get('execution_option')
        if not result['allowed'] and executed is not None:
            paid=backend.scenes[scene_index]['packet'].action_id
            guard_events.append({'paid':paid,'option_id':executed['option_id'],'selection_call_id':executed['selection_call_id'],
                                 'reason':result['reason'],'motion':action})
            planner.note_attempt(executed,result['reason'],paid)
        if result['allowed'] and executed is not None and executed['option_id'] not in dispatched:
            dispatched.add(executed['option_id'])
            planner.note_attempt(executed,'dispatched',backend.scenes[scene_index]['packet'].action_id)
        return result

    def select(scene_index):
        vanilla_target=original_selector(scene_index)
        state=backend.scenes[scene_index]
        previous=backend.calls[-1]
        assert previous['method']=='select_topo_target'
        output=previous['outputs']
        vanilla=deepcopy(output['selected'])
        belief,pose,paid=sensing_state(state)
        config=state['mapper'].config
        proposals=planner.proposals(belief,pose,paid,float(config.resolution_m),
                                   max_depth_m=float(config.max_depth_m),fov_deg=float(config.fov_deg),
                                   robot_radius_m=float(config.robot_radius_m))
        new,certification=certify(proposals,output['candidates'],state,source)
        chosen,audit=planner.rank(belief,pose,paid,float(config.resolution_m),output['candidates'],new)
        row={'selection_call_id':previous['call_id'],'paid_before':paid,'start_pose':list(pose),
             'native_shadow_pose':None if vanilla is None else vanilla['pose'],
             'native_output_sha256':previous['output_sha256'],'raw_proposal_count':len(proposals),
             'new_native_certified_count':len(new),'certification':certification,'ranking':audit}
        if chosen is None:
            row['actual_target_pose']=None if vanilla is None else vanilla['pose']
            row['decision_origin']='native_fallback'
            history.append(json_value(row))
            return vanilla_target
        assert any(candidate['candidate_id']==chosen['candidate_id'] and candidate['pose']==chosen['pose']
                   for candidate in [*output['candidates'],*new])
        if new:
            attempted=state['gain'].attempted_camera_mask(state['mapper'],state['ledger'].planning_camera_poses())
            extra_scores, extra_rows=backend._scores_v10_1(state,new,attempted)
            all_scores={name:[*output['scores'].get(name,[]),*vals] for name,vals in extra_scores.items()}
            output.update(candidates=[*output['candidates'],*new],scores=all_scores,
                          score_audit=[*output['score_audit'],*extra_rows])
        option=deepcopy(chosen)
        option.update(option_id=f"{state['scene_id']}/{state['episode_id']}/plan-{state['plans']}",
                      selection_call_id=previous['call_id'],selected_map_version=state['mapper'].frames,
                      selected_feedback_version=state['feedback']['feedback_version'],
                      parent_region=state['graph']['current_region'])
        state['selected']=option
        row['actual_target_pose']=option['pose']
        row['selected_candidate_id']=option['candidate_id']
        row['decision_origin']='regional_generated' if option.get('regional_proposal') else 'regional_rerank'
        output['selected']=option
        output['regional_global_audit']=row
        state['last_selection']=json_value(output)
        previous['outputs']=state['last_selection']
        previous['output_sha256']=digest(output)
        history.append(json_value(row))
        return np.asarray(option['pose'][:2],dtype=int)

    runtime.observe=observe
    components.assess_local_action=guard
    backend.select_target=select
    return history,guard_events
