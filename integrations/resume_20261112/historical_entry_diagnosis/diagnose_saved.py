"""Read-only paid36 diagnostic. No replay, ROS, inference, or production writes.

Optional --belief accepts a saved native mapper belief (.npy); its hash MUST
match paid36_graph.json. Default performs only provable input-occupancy bounds.
"""
import argparse, ast, hashlib, json, os, pathlib, sys, types
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

BASE=pathlib.Path('/home/ouc/nso_robot')
OUT=pathlib.Path(__file__).parent
VENDOR=BASE/'workspaces/nso_hybrid_20261029/nso_bridge/joint_vendor'
sys.path.insert(0,str(VENDOR))
from utils.grid_geometry import inflated_obstacles, DIRECTIONS

# Extract the original pure functions without importing Open3D or model modules.
def original(path, names, namespace):
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

args=argparse.ArgumentParser()
args.add_argument('--belief',type=pathlib.Path)
args=args.parse_args()
g=json.loads((BASE/'analyses/budget_semantic_20261109/S96_posthoc/paid36_graph.json').read_text())
cfg=types.SimpleNamespace(resolution_m=.2,robot_radius_m=.7212185868375829,camera_height_m=.536,
                          fov_deg=101.34615631189132,max_depth_m=4.,width_px=320,height_px=180)
ns=original(VENDOR/'env/virtual3d.py',['camera_pose'],dict(np=np,DIRECTIONS=DIRECTIONS))
original(VENDOR/'nso/competition_candidates_v8_1.py',['aperture_support'],ns)
orientation_graph=original(VENDOR/'nso/route_coverage_v2.py',['orientation_graph'],dict(np=np,DIRECTIONS=DIRECTIONS,csr_matrix=csr_matrix))['orientation_graph']
asset=g['original_candidate_audit']['measured_assets'][2]
asset={k:np.asarray(v) if isinstance(v,list) else v for k,v in asset.items()}
target=np.array(g['assets'][2]['desired_entry_xy'])
source=BASE/'reports/nso-hybrid-20261109/coverage_S96_20261109_01/joint_inputs/joint-1794224384042323868.npz'
with np.load(source,allow_pickle=False) as z: nav=z['joint_belief_grid']
nav_block=inflated_obstacles(nav==1,cfg.robot_radius_m/cfg.resolution_m)
local=[]
for r in range(200,208):
    for c in range(232,240):
        pose=ns['camera_pose']((r,c),3,cfg,400)
        error=float(np.linalg.norm(pose[:2,3]-target))
        if error<=.35:
            local.append(dict(pose=[r,c,3],xy=pose[:2,3].tolist(),entry_error_m=error,
                              supplied_nav=int(nav[r,c]),blocked_by_supplied_occupied=bool(nav_block[r,c]),
                              aperture_positive=bool(ns['aperture_support'](asset,pose,cfg).max()>0)))
result=dict(status='native_belief_missing',local=local,
            native_belief_required_sha256=g['belief_sha256'],
            supplied_nav_sha256=hashlib.sha256(nav.tobytes()).hexdigest(),
            interpretation='Input occupied cells necessarily remain occupied under mapper np.maximum; unknown/free input does not determine native mapper occupancy. Local unblocked cells are not certified native-safe.')
if args.belief:
    belief=np.load(args.belief,allow_pickle=False)
    assert belief.shape==(400,400)
    assert hashlib.sha256(belief.tobytes()).hexdigest()==g['belief_sha256'], 'Not original paid36 native belief'
    safe=~inflated_obstacles(belief!=0,cfg.robot_radius_m/cfg.resolution_m)
    assert hashlib.sha256(safe.tobytes()).hexdigest()==g['safe_sha256']
    graph,cells,ids=orientation_graph(safe)
    current=g['current_pose']; anchor=g['return_anchor']
    start=int(ids[tuple(current[:2])])*4+current[2]
    home=int(ids[tuple(anchor[:2])])*4+anchor[2]
    d=dijkstra(graph,directed=True,indices=start)
    ret=dijkstra(graph.T.tocsr(),directed=True,indices=home)
    options=[]
    for ci,cell in enumerate(cells):
        node=ci*4+3
        if not np.isfinite(d[node]+ret[node]) or d[node]<=0: continue
        pose=ns['camera_pose'](cell,3,cfg,400)
        if ns['aperture_support'](asset,pose,cfg).max()<=0: continue
        options.append(dict(pose=[int(cell[0]),int(cell[1]),3],xy=pose[:2,3].tolist(),
                            entry_error_m=float(np.linalg.norm(pose[:2,3]-target)),
                            outward_cost=int(d[node]),return_cost=int(ret[node]),total_cost=int(d[node]+ret[node])))
    result['nearest_connected_wanted_heading_aperture']=min(options,key=lambda x:(x['entry_error_m'],x['total_cost'],x['pose'])) if options else None
    for row in local:
        r,c,_=row['pose']; row.update(native_belief=int(belief[r,c]),native_safe=bool(safe[r,c]))
    result['status']='exact_native_saved_belief_diagnosed_no_budget_filter'
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
