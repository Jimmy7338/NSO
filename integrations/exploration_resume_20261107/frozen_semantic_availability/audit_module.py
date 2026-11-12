"""Read-only availability on a replayed measured state; never calls a planner."""
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import binary_closing, label
from env.virtual3d import camera_pose
from nso.route_coverage_v2 import orientation_graph
from nso.competition_candidates_v8_1 import aperture_support, navigation_rear_boundary, XY_DIRECTIONS
from utils.grid_geometry import inflated_obstacles


def _digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _costs(safe, current, anchor):
    graph, cells, ids = orientation_graph(safe)
    for name, p in [('current', current), ('anchor', anchor)]:
        if not (0 <= p[0] < safe.shape[0] and 0 <= p[1] < safe.shape[1] and safe[p[:2]]):
            raise ValueError(name + '_footprint_not_known_safe')
    start = int(ids[current[:2]]) * 4 + current[2]
    home = int(ids[anchor[:2]]) * 4 + anchor[2]
    outward = dijkstra(graph, directed=True, indices=start)
    returning = dijkstra(graph.T.tocsr(), directed=True, indices=home)
    return cells, outward, returning


def _state(cells, n):
    return [int(cells[n // 4, 0]), int(cells[n // 4, 1]), int(n % 4)]


def _raw_geometry_groups(mapper, assets):
    """Mirror only the original pre-label geometry qualification, not policy."""
    from nso.response_features_v7 import ResponseFeatureConfig
    cfg=ResponseFeatureConfig()
    profile=getattr(mapper,'measured_extent_profile',None)
    if getattr(mapper,'asset_grouping_profile',None) == 'measured_instance_support_v1':
        raise ValueError('raw closing audit not applicable to instance-grouped profile')
    max_horizontal=3.0 if profile == 'measured_extent_3m_v1' else cfg.max_horizontal_extent_m
    q=mapper.quality_evidence(max_points=10000)
    if q is None: return dict(status='no_quality',groups=[],passed_group_ids=[])
    points=q['point']
    cells=np.column_stack([mapper.shape[0]-1-np.floor(points[:,1]/.2),np.floor(points[:,0]/.2)]).astype(int)
    inside=((cells>=0)&(cells<np.array(mapper.shape))).all(axis=1)
    occupied=np.zeros(mapper.shape,bool)
    occupied[tuple(cells[inside].T)]=True
    groups,_=label(binary_closing(occupied,structure=np.ones((3,3))))
    group_ids=np.zeros(len(points),int)
    group_ids[inside]=groups[tuple(cells[inside].T)]
    rows=[]
    for group in sorted(set(group_ids)):
        ids=np.flatnonzero(group_ids==group)
        lo,hi=np.quantile(points[ids],[.05,.95],axis=0)
        span=hi-lo
        labs=q['label'][ids]
        fails=dict(min_support=len(ids)<cfg.min_support_points,
                   max_horizontal_extent=bool(max(span[:2])>max_horizontal),
                   min_vertical_extent=bool(span[2]<cfg.min_vertical_extent_m))
        first='group0_excluded_before_geometry_loop' if group==0 else next((k for k,v in fails.items() if v),None)
        rows.append(dict(group=int(group),support_points=int(len(ids)),marked_class2=int(np.count_nonzero(labs==2)),
            marked_class3=int(np.count_nonzero(labs==3)),observed_low=lo.tolist(),observed_high=hi.tolist(),
            observed_spans_m=span.tolist(),failed_predicates=fails,first_failed=first,accepted=first is None))
    passed=sorted(x['group'] for x in rows if x['accepted'])
    native=sorted(int(a['group']) for a in assets)
    details={int(a['group']):a for a in assets}
    counts_match=all(int(details[x['group']]['support_points'])==x['support_points'] and
        int(details[x['group']]['marked_points'])==x['marked_class2']+x['marked_class3'] for x in rows if x['accepted'] and x['group'] in details)
    return dict(status='ok',profile=profile,algorithm='original quality_evidence(max_points=10000),0.2m XY cells,3x3 binary_closing,label,5/95quantiles; no changed thresholds',
        thresholds=dict(min_support_points=cfg.min_support_points,max_horizontal_extent_m=max_horizontal,min_vertical_extent_m=cfg.min_vertical_extent_m),
        quality_sample_points=int(len(points)),quality_sample_class2=int(np.count_nonzero(q['label']==2)),quality_sample_class3=int(np.count_nonzero(q['label']==3)),
        outside_grid_points=int(np.count_nonzero(~inside)),groups=rows,passed_group_ids=passed,native_asset_group_ids=native,
        passed_groups_equal_native=passed==native,passed_support_and_marked_counts_equal_native=counts_match,
        identity_limit='Connected geometry groups may mix surfaces; no physical object identity or reason for physical scene arrangement is established.')


def audit(session, output_path, *, expected_observations=41, expected_paid_actions=40, prefix_acquisition=None):
    """Called in replay runner's process after all 41 actual observations.

    Additional B40/B80 are resource counterfactuals, not original choices.
    Reachability is for the original circular known-safe graph only. No route
    shortlist, score, history, asset, guard, budget or pose is modified.
    """
    started = perf_counter()
    s = session.backend.scenes[0]
    mapper, packet = s['mapper'], s['packet']
    current = tuple(map(int, (*packet.position, packet.heading)))
    anchor = tuple(map(int, s['return_anchor']))
    config = mapper.config
    if int(s['semantic_version']) != int(mapper.frames):
        raise ValueError('stale native semantic/assets version; do not silently refresh replay state')
    belief = np.asarray(mapper.belief)
    remaining = int(s['ledger'].remaining_budget)
    paid = int(s['ledger'].total_budget) - remaining
    if int(mapper.frames) != expected_observations or paid != expected_paid_actions:
        raise ValueError('session does not match the explicitly declared measured-prefix scope')
    if prefix_acquisition is None:
        if (expected_observations,expected_paid_actions)!=(41,40):
            raise ValueError('non-final prefix requires its own measured acquisition provenance')
        prefix_acquisition=dict(paid_actions=40,CAN_distance_m=7.393505834221814,elapsed_s=542.4439440459973,
            source='exploration_resume_S40_01/result.json; not a free prefix')
    if int(prefix_acquisition['paid_actions']) != paid:
        raise ValueError('prefix acquisition count mismatch')
    before = (_digest(belief), mapper.frames, current, anchor, remaining,
              repr(s.get('planning_direction_support')))
    safe = ~inflated_obstacles(belief != 0, config.robot_radius_m / config.resolution_m)
    occupied_inflation = inflated_obstacles(belief == 1, config.robot_radius_m / config.resolution_m)
    unknown_inflation = inflated_obstacles(belief == -1, config.robot_radius_m / config.resolution_m)
    result = dict(schema='nso.frozen_semantic_availability.v1',
        observation_count=int(mapper.frames), semantic_version=int(s['semantic_version']),
        graph_version=int(s['graph_version']), current_state=list(current), return_anchor=list(anchor),
        actual_base_pose_planning=list(map(float, getattr(packet, 'base_pose_planning', ()))),
        current_state_source='unchanged final native packet discrete pose; actual measured continuous pose reported separately',
        current_equals_anchor=current == anchor, original_remaining_budget=remaining,
        fresh_additional_budgets=[40,80], hypothetical_remaining_budgets=[40,80],
        hypothetical_total_budgets=[paid+40,paid+80],
        hypothetical_budget_increments_over_original=[40-remaining,80-remaining],
        prefix_acquisition=prefix_acquisition,
        interpretation='Frozen accumulated measured prefix; hypothetical remaining40/80 (total paid-prefix+remaining), not original episode output, a free prefix, or physical efficacy.',
        grid_resolution_m=float(config.resolution_m), robot_radius_m=float(config.robot_radius_m),
        belief_sha256=before[0], safe_sha256=_digest(safe),
        cost_contract='native orientation_graph: paid forward/left/right; current to target plus target to original anchor including final heading; at least one outbound action',
        assets=[], limits=['Aperture is hypothetical measured-geometry support, not ray-traced visibility or physical reachability proof.',
            'No candidate shortlist, entry/deep projection, candidate-cap or G/S selection has been run.',
            'No unobserved map cells have been filled or traversed; no policy or guard changed.',
            'Blocked-cell attribution describes inflation masks; it does not establish unseen free space.'])
    result['raw_geometry_group_audit']=_raw_geometry_groups(mapper,s['assets'])
    result['native_asset_count']=len(s['assets'])
    result['marked_native_asset_count']=sum(int(a.get('marked_points',0))>0 for a in s['assets'])
    try:
        cells, outward, returning = _costs(safe, current, anchor)
    except ValueError as exc:
        result['status'] = str(exc)
    else:
        total = outward + returning
        connected = np.isfinite(total) & (outward > 0)
        result.update(status='ok', known_safe_cells=int(len(cells)),
            known_safe_orientation_states=int(len(outward)),
            reachable_nonzero_outbound_states=int(connected.sum()),
            minimum_current_return_cost=float(returning[np.flatnonzero(outward == 0)[0]]))
        poses = [camera_pose(tuple(cell), h, config, belief.shape[0]) for cell in cells for h in range(4)]
        for ai, asset in enumerate(s['assets']):
            if int(asset.get('marked_points', 0)) <= 0:
                continue
            support = np.array([float(aperture_support(asset, p, config).mean()) for p in poses])
            rear = np.array([float((p[:2,3]-asset['rear_boundary_xy']) @ asset['back_axis']) > .15 for p in poses])
            eligible = connected & (support > 0)
            nodes = np.flatnonzero(eligible)
            best = min(nodes, key=lambda n: (total[n], outward[n], int(n))) if len(nodes) else None
            row=dict(asset_index=ai, group=int(asset['group']), class_vote=float(asset['class_vote']),
                marked_points=int(asset['marked_points']),
                prior_geometry=float(asset['measured_width_m']*asset['measured_height_m']+3*asset['measured_width_m']*asset['measured_depth_m']),
                prior_semantic=float(asset['measured_width_m']*asset['measured_height_m']+3*(1-asset['class_vote'])*asset['measured_width_m']*asset['measured_depth_m']),
                known_safe_rear_states=int(rear.sum()), known_safe_aperture_states=int((support>0).sum()),
                reachable_rear_states=int((connected & rear).sum()), reachable_aperture_states=int(eligible.sum()),
                minimum_additional_budget=None if best is None else int(round(total[best])),
                best_state=None if best is None else _state(cells,best),
                best_outbound_cost=None if best is None else int(round(outward[best])),
                best_return_cost=None if best is None else int(round(returning[best])),
                best_aperture_fraction=None if best is None else float(support[best]))
            wanted_heading=int(np.argmax(XY_DIRECTIONS @ asset['front_axis']))
            wanted_nodes=np.flatnonzero(eligible & (np.arange(len(total)) % 4 == wanted_heading))
            best_wanted=min(wanted_nodes,key=lambda n:(total[n],outward[n],int(n))) if len(wanted_nodes) else None
            nav_rear=navigation_rear_boundary(mapper,asset)
            row.update(aabb_center=np.asarray(asset['aabb_center']).tolist(),
                back_axis=np.asarray(asset['back_axis']).tolist(),
                front_axis=np.asarray(asset['front_axis']).tolist(),
                measured_rear_boundary_xy=np.asarray(asset['rear_boundary_xy']).tolist(),
                navigation_rear_boundary_xy=np.asarray(nav_rear).tolist(),
                native_desired_entry_xy=(nav_rear+.4*asset['back_axis']).tolist(),
                native_desired_deep_xy=(nav_rear+.8*asset['back_axis']).tolist(),
                native_wanted_heading=wanted_heading,
                best_state_has_native_wanted_heading=None if best is None else int(best)%4==wanted_heading,
                wanted_heading_minimum_additional_budget=None if best_wanted is None else int(round(total[best_wanted])),
                wanted_heading_best_state=None if best_wanted is None else _state(cells,best_wanted),
                budget_interpretation='B* is arbitrary-heading aperture availability, not a selected native entry; projection<=.35, entry/deep sequencing and quota not evaluated')
            row['reason'] = ('available_in_known_safe_component' if best is not None else
                'aperture_exists_only_outside_current_anchor_connected_component' if (support>0).any() else
                'no_aperture_pose_in_known_safe_graph')
            row['original_remaining_budget_test']=dict(remaining_budget=remaining,
                available=bool(np.any(eligible & (total<=remaining))),states=int(np.count_nonzero(eligible & (total<=remaining))))
            row['additional_budget_tests']={str(b):dict(available=bool(np.any(eligible & (total <= b))),
                states=int(np.count_nonzero(eligible & (total <= b))),
                reason=('available' if np.any(eligible & (total<=b)) else 'budget_insufficient' if best is not None else row['reason'])) for b in [40,80]}
            # Attribute excluded measured-free rear/FOV poses to the actual masks;
            # these poses remain excluded and are never used for B*.
            blocked = dict(unknown_inflation_only=0, occupied_inflation_only=0, both=0)
            for cell in np.argwhere((belief == 0) & ~safe):
                pos = tuple(map(int,cell))
                if not any(aperture_support(asset,camera_pose(pos,h,config,belief.shape[0]),config).max()>0 for h in range(4)):
                    continue
                u,o=bool(unknown_inflation[pos]),bool(occupied_inflation[pos])
                blocked['both' if u and o else 'unknown_inflation_only' if u else 'occupied_inflation_only'] += 1
            row['excluded_measured_free_cells_with_aperture'] = blocked
            result['assets'].append(row)
    after=(_digest(mapper.belief),mapper.frames,tuple(map(int,(*s['packet'].position,s['packet'].heading))),
           tuple(map(int,s['return_anchor'])),int(s['ledger'].remaining_budget),repr(s.get('planning_direction_support')))
    assert after == before, 'read-only audit mutated session'
    result['read_only_state_fingerprint_unchanged']=True
    result['elapsed_s']=perf_counter()-started
    result['native_source']={inspect.getfile(f):hashlib.sha256(Path(inspect.getfile(f)).read_bytes()).hexdigest() for f in [orientation_graph,aperture_support,camera_pose,inflated_obstacles]}
    output_path=Path(output_path)
    if output_path.suffix != '.json': output_path=output_path/'availability.json'
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
