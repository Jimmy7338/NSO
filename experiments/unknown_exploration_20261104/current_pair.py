import os
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import sys,json,hashlib,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
R=Path(__file__).resolve().parent / 'source'; sys.path.insert(0,str(R))
from env.virtual3d_competition_v9 import create_competition_world,collect_competition_prefix
from scripts.prepare_response_v7 import records_from_prefix
from scripts.run_cpu_semantic_inspection_v10_2 import execute,write_json
from utils.reconstruction_metrics import ReconstructionEvaluator
from utils.counterfactual_surface_visibility import reference_visible
D=Path(os.environ['NSO_OUTPUT_ROOT']) / 'current_source_pair'; D.mkdir(exist_ok=False)
def hashfile(p): return hashlib.sha256(p.read_bytes()).hexdigest()
sources=sorted([p for d in ('env','nso','utils','scripts','configs/virtual3d') for p in (R/d).glob('*.py' if d!='configs/virtual3d' else '*.json')])
hashes={str(p.relative_to(R)):hashfile(p) for p in sources}
write_json(D/'manifest.json',dict(status='running',scope='new current-source Q0 shelf_west S/G controlled simulation; not historical recovery',source_sha256=hashes,script_sha256=hashfile(Path(__file__)),threads=1))
t=time.monotonic(); w=create_competition_world('Q0','shelf_west'); raw=collect_competition_prefix(w)
assert len(raw)==151 and np.allclose(raw[-1]['frame'].world_from_camera[:2,3],w.context.final_prefix_xy_m,rtol=0,atol=1e-12) and w.heading==w.context.final_prefix_heading
f=D/'prepared/Q0/shelf_west'; (f/'prefix/frames').mkdir(parents=True); (f/'prefix/scans').mkdir()
write_json(f/'fixture.json',dict(context=asdict(w.context),config=asdict(w.config),arrangement='shelf_west'))
records=records_from_prefix(raw,w.shape,w.config); write_json(f/'prefix/records.json',records)
for i,row in enumerate(raw): row['frame'].save(f/'prefix/frames'/f'{i:04d}.npz'); row['scan'].save(f/'prefix/scans'/f'{i:04d}.npz')
protocol=json.loads((R/'configs/virtual3d/competition_v9_validation_protocol.json').read_text()); count=protocol['reference']['samples_requested']; e=ReconstructionEvaluator(w,count=count,seed=2026); weight=float(w.mesh.get_surface_area())/count
seen=np.zeros(len(e.reference),bool)
for row in raw: seen |= reference_visible(e.reference,row['frame'],e.truth,row['frame'].world_from_camera,w.config.max_depth_m)
mid=w.config.width_m/2
np.savez_compressed(f/'reference.npz',points=e.reference,classes=e.classes,weights=np.full(len(e.reference),weight),prefix_seen=seen,west_slot=(e.classes!=1)&(e.reference[:,0]<mid),east_slot=(e.classes!=1)&(e.reference[:,0]>=mid),reachable=w.reachable,vertices=np.asarray(w.mesh.vertices),triangles=np.asarray(w.mesh.triangles))
write_json(D/'prepared_hashes.json',{str(p.relative_to(D/'prepared')):hashfile(p) for p in (D/'prepared').rglob('*') if p.is_file()}); print('prepared',round(time.monotonic()-t,1),flush=True)
rows=[]
for condition in ('G','S'):
 row=execute(D/'prepared','Q0','shelf_west',condition,revision='v10_3_1',novelty_floor=.25); rows.append(row); write_json(D/'partial.json',rows); print(condition,round(time.monotonic()-t,1),flush=True)
assert all(hashfile(R/k)==v for k,v in hashes.items())
write_json(D/'results.json',rows)
write_json(D/'summary.json',dict(scope='single paired current-source controlled simulation; no historical recovery or real robot claim',elapsed_s=time.monotonic()-t,by_condition={r['condition']:{k:r[k] for k in ('future_paid_actions','new_unique_surface_area_m2','final_f1_05cm','new_area_times_final_f1_05cm','coverage_2d_gain_m2','path_distance_m','failed','collisions','returned_to_anchor','modules_called','gain_calibration')} for r in rows},S_minus_G_joint=rows[1]['new_area_times_final_f1_05cm']-rows[0]['new_area_times_final_f1_05cm']))
m=json.loads((D/'manifest.json').read_text());m['status']='complete';write_json(D/'manifest.json',m)
