"""Fixed hypothesis-own .06m quadrature; no source/reference samples."""
import hashlib,sys
from pathlib import Path
import numpy as np
import open3d as o3d
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from parts_audit import boundary_classification
import acquisition_forecast as f

def build_surface_basis(parts,world_from_local=None,orbital_poses=None,spacing_m=.06):
 if spacing_m!=.06:raise ValueError('frozen .06m quadrature required')
 transform=np.eye(4)if world_from_local is None else np.asarray(world_from_local,float)
 points=[];areas=[];normals=[];pids=[]
 for pi,(raw,size)in enumerate(parts):
  lo=np.asarray(raw,float);size=np.asarray(size,float)
  if np.any(size<=0)or not np.isfinite(np.r_[lo,size]).all():raise ValueError('positive finite primitive dimensions required')
  for axis in range(3):
   tang=[j for j in range(3)if j!=axis];counts=[max(1,int(np.ceil(size[j]/spacing_m)))for j in tang]
   for side in (0,1):
    for i in range(counts[0]):
     for j in range(counts[1]):
      p=lo.copy();p[axis]+=side*size[axis];p[tang[0]]+=(i+.5)*size[tang[0]]/counts[0];p[tang[1]]+=(j+.5)*size[tang[1]]/counts[1]
      n=np.zeros(3);n[axis]=side*2-1;points.append(p);normals.append(n);pids.append(pi);areas.append(size[tang[0]]*size[tang[1]]/(counts[0]*counts[1]))
 points=np.asarray(points);normals=np.asarray(normals);areas=np.asarray(areas);pids=np.asarray(pids)
 internal,duplicate,mult=boundary_classification(parts,points,normals,pids)
 mesh=o3d.geometry.TriangleMesh()
 for lo,size in parts:mesh+=o3d.geometry.TriangleMesh.create_box(*size).translate(lo)
 mesh.transform(transform);world=points@transform[:3,:3].T+transform[:3,3];scene=f.ray_scene(mesh)
 if orbital_poses is None:
  lo=np.min([o for o,s in parts],axis=0);hi=np.max([np.asarray(o)+s for o,s in parts],axis=0);center=(lo+hi)/2;radius=max(1.6,float(np.linalg.norm((hi-lo)[:2])/2+.8));orbital_poses=[]
  for angle in range(0,360,30):
   a=np.deg2rad(angle);eye=np.array([center[0]+radius*np.sin(a),center[1]-radius*np.cos(a),max(1.3,center[2])]);z=center-eye;z/=np.linalg.norm(z);x=np.cross(z,[0,0,1.]);x/=np.linalg.norm(x);y=np.cross(z,x);pose=np.eye(4);pose[:3,:3]=np.column_stack([x,y,z]);pose[:3,3]=eye;orbital_poses.append(transform@pose)
 visible=np.zeros(len(world),bool)
 for pose in orbital_poses:
  origin=np.asarray(pose)[:3,3];delta=world-origin;rr=np.column_stack([np.tile(origin,(len(world),1)),delta]).astype(np.float32);hit=scene.cast_rays(o3d.core.Tensor(rr),nthreads=1)['t_hit'].numpy()
  local=delta@np.asarray(pose)[:3,:3];z=local[:,2];inview=(z>.15)&(z<=8)&(abs(local[:,0])<z)&(abs(local[:,1])<z*.75)
  visible|=inview&(abs(hit-1)*np.linalg.norm(delta,axis=1)<=.001)
 weight=areas/mult*(~internal)*visible
 return dict(points=world,areas=weight,raw_areas=areas,internal=internal,visible=visible,multiplicity=mult,part_ids=pids,
  receipt=dict(spacing_m=.06,orbital_poses=len(orbital_poses),raw_points=len(points),positive_area_points=int((weight>0).sum()),raw_primitive_area_m2=float(areas.sum()),external_visible_area_m2=float(weight.sum()),internal_points=int(internal.sum()),coincident_external_points=int(duplicate.sum()),basis_sha256=f.digest(world),area_sha256=f.digest(weight),definition='Hypothesis-owned deterministic quadrature, local union-boundary centers, coincident outward area/multiplicity, own fixed 12 orbital first-hit visibility. Not source-reference samples or exact analytic surface union.'))
