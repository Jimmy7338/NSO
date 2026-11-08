#!/usr/bin/env python3
"""Fixed synthetic counterexample, original entry logic; no ROS or rollout."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import hashlib, inspect, json, math, pathlib, sys
from types import SimpleNamespace
import numpy as np
from scipy.sparse.csgraph import dijkstra

OUT = pathlib.Path(__file__).resolve().parent
BRIDGE = pathlib.Path('/home/ouc/nso_robot/workspaces/nso_hybrid_20261029/nso_bridge')
sys.path[:0] = [str(BRIDGE), str(BRIDGE/'joint_vendor')]
from env.virtual3d import camera_pose
from env.virtual3d_v2 import VirtualConfigV2
from utils.grid_geometry import inflated_obstacles
from nso.route_coverage_v2 import orientation_graph
from nso.competition_candidates_v8_1 import measured_assets, navigation_rear_boundary, aperture_support, XY_DIRECTIONS
import nso.hierarchical_options_v10 as original
import nso_bridge.exploration_joint_policy as joint

R = .7212185868375829
RES = .2
BUDGET = 80  # fixed generous budget, identical in both radii; not a physical episode
SHAPE = (50, 50)
CURRENT = (14, 25, 2)  # x=5.1,y=7.1, facing south; also fixed return anchor

def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def fixture(radius):
    cfg = VirtualConfigV2(robot_radius_m=radius, resolution_m=RES)
    belief = np.zeros(SHAPE, np.int8)
    # Finite wide wall: centers x=2.1..7.9,y=5.1. Ends remain traversable.
    belief[24, 10:40] = 1
    xx, zz = np.meshgrid(np.linspace(4.1, 5.9, 19), np.linspace(.3, 1.3, 11))
    points = np.column_stack([xx.ravel(), np.full(xx.size, 5.1), zz.ravel()])
    q = dict(point=points, normal=np.tile([0., -1., 0.], (len(points), 1)),
             label=np.full(len(points), 3, np.uint8), bits=np.ones(len(points), np.uint8))
    # Synthetic retained first view controls the native front/back convention.
    frame = SimpleNamespace(world_from_camera=camera_pose((34, 25), 0, cfg, SHAPE[0]),
        intrinsic=np.array([[48.,0.,47.5],[0.,48.,35.5],[0.,0.,1.]]),
        depth_m=np.full((72,96), 2.))
    mapper = SimpleNamespace(config=cfg, shape=SHAPE, belief=belief,
        camera_seen=np.ones(SHAPE, bool), keyframes=[frame], quality_evidence=lambda **kwargs:q)
    return mapper

def detail(mapper, asset):
    cfg=mapper.config
    safe=~inflated_obstacles(mapper.belief!=0, cfg.robot_radius_m/RES)
    graph,cells,ids=orientation_graph(safe)
    start=int(ids[CURRENT[:2]])*4+CURRENT[2]
    a=dijkstra(graph,directed=True,indices=start)
    b=dijkstra(graph.T.tocsr(),directed=True,indices=start)
    nav=navigation_rear_boundary(mapper,asset)
    desired=nav+.4*asset['back_axis']
    wanted=int(np.argmax(XY_DIRECTIONS@asset['front_axis']))
    rows=[]
    for node in range(len(a)):
        if node%4!=wanted: continue
        cell=tuple(map(int,cells[node//4])); cam=camera_pose(cell,wanted,cfg,SHAPE[0])
        depth=float((cam[:2,3]-asset['rear_boundary_xy'])@asset['back_axis'])
        aperture=float(aperture_support(asset,cam,cfg).max())
        if depth<=.15 or aperture<=0:continue
        rows.append(dict(pose=[*cell,wanted],xy=cam[:2,3].tolist(),
            error_m=float(np.linalg.norm(cam[:2,3]-desired)),cost=float(a[node]+b[node]),
            outward=float(a[node]),aperture_max=aperture,
            edge_normal_clearance_m=float((cam[:2,3]-nav)@asset['back_axis'])))
    feasible=[r for r in rows if np.isfinite(r['cost']) and r['outward']>0 and r['cost']<=BUDGET]
    nearest=min(feasible,key=lambda r:(r['error_m'],r['cost'],r['pose']))
    passing=[r for r in feasible if r['error_m']<=.35]
    return dict(nav_rear_xy=nav.tolist(),measured_rear_xy=asset['rear_boundary_xy'].tolist(),
        desired_entry_xy=desired.tolist(),wanted_heading=wanted,
        safe_cells=int(safe.sum()),feasible_wanted_backside_positive_aperture=len(feasible),
        passing_entry_geometry_count=len(passing),nearest=nearest,
        safe_sha256=hashlib.sha256(safe.tobytes()).hexdigest())

def run():
    records=[]
    for radius in (.2,R):
        mapper=fixture(radius);assets=measured_assets(mapper)
        assert len(assets)==1, len(assets)
        asset=assets[0];d=detail(mapper,asset)
        native={}
        for name,module in [('restored_hierarchical_options_v10',original),('physical_exploration_joint_v1',joint)]:
            routes,audit=module.generate_options(mapper,CURRENT[:2],CURRENT[2],BUDGET,CURRENT,
                max_candidates=12,coverage_strategy='total_diverse',coverage_slots=4)
            for r in routes:
                assert r['states'][0]==list(CURRENT) and r['states'][-1]==list(CURRENT)
                assert r['cost']==len(r['outbound_actions'])+len(r['return_actions'])<=BUDGET
            native[name]=dict(target_audit=audit['target_audit'],available_roles=audit['available_roles'],
                safe_sha256=audit['safe_sha256'],entry_routes=[r for r in routes if r['group']=='asset_0_entry'])
            assert audit['safe_sha256']==d['safe_sha256']
            entry=next(x for x in audit['target_audit'] if x['role']=='asset_0_entry')
            assert entry['reason']==('selected' if radius==.2 else 'target_projection_error_exceeds_limit'),entry
        records.append(dict(radius_m=radius,geometry=d,native_generators=native))
    assert records[0]['geometry']['passing_entry_geometry_count']>0
    assert records[1]['geometry']['passing_entry_geometry_count']==0
    # Hash imported implementation and mathematical dependencies, not just filenames.
    funcs=[original.generate_options,joint.generate_options,navigation_rear_boundary,
           aperture_support,measured_assets,inflated_obstacles,orientation_graph,camera_pose]
    files=sorted(set(inspect.getsourcefile(f) for f in funcs))
    sources=[dict(path=p,sha256=sha(p)) for p in files]
    margin=RES*math.sqrt(2)/2
    bound=R+margin-RES/2
    discrete_center=RES*(math.floor((R+margin)/RES)+1)
    output=dict(schema='nso.physical_entry_scale_audit.v1',status='fixed_synthetic_configuration_counterexample_only',
        fixture=dict(shape=SHAPE,resolution_m=RES,budget=BUDGET,current_anchor=CURRENT,
          known_free_except_wall=True,camera_seen_all=True,wall_row=24,wall_cols_halfopen=[10,40],
          physical_acquisition_simulated=False,rendering_or_detector_used=False,score_or_rollout_run=False,
          common_inputs_except_radius=True,source_quality_points=209),
        algebra=dict(radius_m=R,obstacle_center_margin_m=margin,
          nav_boundary_edge_shift_from_center_m=RES/2,
          strict_flat_face_edge_clearance_lower_bound_m=bound,
          entry_offset_plus_tolerance_m=.75,continuous_necessary_gap_m=bound-.75,
          first_safe_axial_center_distance_m=discrete_center,
          first_safe_axial_edge_clearance_m=discrete_center-RES/2,
          axial_projection_error_lower_bound_m=discrete_center-RES/2-.4,
          continuous_radius_threshold_m=.75-margin+RES/2,
          fixed_tolerance_min_axial_offset_for_this_grid_m=discrete_center-RES/2-.35),
        cases=records,sources=sources,script_sha256=sha(__file__))
    (OUT/'result.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status=output['status'],algebra=output['algebra'],
        nearest=[dict(radius_m=r['radius_m'],**r['geometry']['nearest']) for r in records]),indent=2))

if __name__=='__main__':run()
