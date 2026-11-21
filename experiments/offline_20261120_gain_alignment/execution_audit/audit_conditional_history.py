#!/usr/bin/env python3
"""Diagnose declared conditional-mean surfaces from acquired history only.

No future rays or candidate scores are computed. The .22 m box quadrature is
an audit basis from B8, not the final C forecast integration resolution.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path('/home/ouc/nso_robot/analyses/dual_template_gain_alignment_20261120')
FOLDER=ROOT/'execution_audit/acquisition_prefix/shelf_left_54_attempt_001'
B8=ROOT.parent/'dual_template_uncertainty_20261120'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=load('C_history_frozen_dependencies',B8/'replay/run_particle_predictions.py');parent.verify();_,_,repair,_,model=parent.dependencies()
receipt=json.loads((FOLDER/'receipt.json').read_text());hypotheses=json.loads((FOLDER/'conditional_hypotheses.json').read_text())
with np.load(FOLDER/'acquired_frames.npz',allow_pickle=False)as z:
 frames=[SimpleNamespace(depth_m=z['depth_m'][i].copy(),intrinsic=z['intrinsic'][i].copy(),world_from_camera=z['world_from_camera'][i].copy())for i in receipt['retained_keyframe_indices']]
rows=[]
for obj in hypotheses['objects']:
 for hyp in obj['hypotheses']:
  parts=[(np.asarray(origin),np.asarray(size))for origin,size in hyp['parts']]
  xyz,area=model.quadrature(parts,obj['base_geometry']['rotation']);history=repair.classify_history(xyz,frames)
  free=history['strict_free'];measured=history['measured_surface']
  rows.append(dict(object_index=obj['object_index'],support_track=obj['support_track'],shape_id=hyp['shape_id'],shape_name=hyp['shape_name'],quadrature_points=len(xyz),quadrature_area_m2=float(area.sum()),strict_free_points=int(free.sum()),strict_free_point_fraction=float(free.mean()),strict_free_area_m2=float(area[free].sum()),strict_free_area_fraction=float(area[free].sum()/area.sum()),measured_surface_points=int(measured.sum()),measured_surface_area_m2=float(area[measured].sum())))
out=dict(status='complete_current_hypothesis_surface_audit_no_future_rays',basis='Frozen B8 per-primitive face-center quadrature, max spacing0.22m; primitive internal overlaps retained',history_keyframes=len(frames),rows=rows,future_pixel_first_hit_rate_evaluated=False,candidate_scores_evaluated=False,strict_free_occluder_skip_implemented=False,required_before_fitted_C='Future hypothesis intersections must use original B2 strict-free skip with cross-history measured-surface veto, then nearest acquired-mesh hit; foreground blocks cannot silently reappear. C template area must retain original strict-free exclusion.',sources={str(p):sha(p)for p in (Path(__file__),FOLDER/'receipt.json',FOLDER/'conditional_hypotheses.json',FOLDER/'acquired_frames.npz')})
p=FOLDER/'conditional_history_audit.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'path':str(p),'rows':rows},indent=2))
