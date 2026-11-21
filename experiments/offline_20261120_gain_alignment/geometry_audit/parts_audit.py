"""Read-only union-boundary diagnostic of frozen template quadrature.

Each object/particle/shape is inspected separately. Alternate hypotheses are
never unioned, clipped or normalized together. No source geometry is loaded.
"""
from pathlib import Path
import hashlib,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
B8=BASE/'dual_template_uncertainty_20261120';B9=BASE/'dual_template_followup_20261120'
EPS=1e-7

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def box(origin,dims):
 origin=np.asarray(origin,float);dims=np.asarray(dims,float);points=[];weights=[];normals=[]
 for normal in range(3):
  axes=[i for i in range(3)if i!=normal];counts=[max(1,int(np.ceil(dims[a]/.22)))for a in axes];area=dims[axes[0]]*dims[axes[1]]/(counts[0]*counts[1])
  for side in (0.,1.):
   for i in range(counts[0]):
    for j in range(counts[1]):
     p=origin.copy();p[normal]+=side*dims[normal];p[axes[0]]+=(i+.5)*dims[axes[0]]/counts[0];p[axes[1]]+=(j+.5)*dims[axes[1]]/counts[1]
     n=np.zeros(3);n[normal]=side*2-1;points.append(p);weights.append(area);normals.append(n)
 return np.array(points),np.array(weights),np.array(normals)
def quad(parts):
 xyz=[];weights=[];normals=[];ids=[]
 for i,(o,d)in enumerate(parts):
  p,w,n=box(o,d);xyz.extend(p);weights.extend(w);normals.extend(n);ids.extend([i]*len(w))
 return np.asarray(xyz),np.asarray(weights),np.asarray(normals),np.asarray(ids)
def parts_for(obj,particle,shape,target):
 base=obj['conditional_geometry'];R=np.asarray(base['rotation']);center=np.asarray(base['center'])@R
 width,height=np.asarray(base['dims'])[[0,2]];depth,lower=particle['depth_m'],particle['lower_m'];x=center[0]-width/2
 if shape==0:parts=[([x,lower,0.],[width,depth,height])]
 else:
  parts=[([x,lower,z['lower_m']],[width,depth,z['upper_m']-z['lower_m']])for z in base['observed_layers_audit']['final_intervals']]
  parts += [([x+dx,lower+dy,0.],[.08,.08,height])for dx in (0.,width-.08)for dy in (0.,depth-.08)]
  # B8's last board is one of four model-declared local faces. Identify it
  # exclusively by exact agreement with the archived template point basis.
  matches=[]
  for axis in (0,1):
   for high in (False,True):
    origin=np.array([x,lower,0.]);size=np.array([width,depth,height]);size[axis]=.08
    if high:origin[axis]+=[width,depth][axis]-.08
    candidate=parts+[(origin,size)];q=quad(candidate)
    if len(q[0])==len(target)and np.allclose(q[0]@R.T,target,atol=1e-12,rtol=0):matches.append(candidate)
  assert matches,'No exact declared board reproduces frozen points'
  parts=matches[0]
 q=quad(parts);np.testing.assert_allclose(q[0]@R.T,target,atol=1e-12,rtol=0)
 return parts,q

def boundary_classification(parts,points,normals,part_ids):
 # Inside after moving along an individual primitive's outward normal means
 # this sample is an internal union face. Equal outward faces are external
 # but are covered by multiple component surfaces.
 outward=points+EPS*normals;internal=np.zeros(len(points),bool);coincident=np.ones(len(points),int)
 for j,(origin,dims)in enumerate(parts):
  lo=np.asarray(origin,float);hi=lo+np.asarray(dims,float)
  other=part_ids!=j
  inside=np.all(outward>lo+EPS*.01,axis=1)&np.all(outward<hi-EPS*.01,axis=1)
  internal|=other&inside
  for axis in range(3):
   tang=[i for i in range(3)if i!=axis];onspan=np.all(points[:,tang]>=lo[tang]-EPS*.01,axis=1)&np.all(points[:,tang]<=hi[tang]+EPS*.01,axis=1)
   for side,sign in ((lo[axis],-1),(hi[axis],1)):
    coincident += (other&onspan&(np.abs(points[:,axis]-side)<EPS*.01)&(normals[:,axis]==sign))
 return internal,(coincident>1)&~internal,coincident

def masks(points,objectids,particleids,shapeids,objects):
 internal=np.zeros(len(points),bool);duplicate=np.zeros(len(points),bool);multiplicity=np.ones(len(points),int);areas=np.zeros(len(points));parts_count=0
 for oi,obj in enumerate(objects):
  for ki,particle in enumerate(obj['particles']):
   for hi in (0,1):
    ids=np.flatnonzero((objectids==oi)&(particleids==ki)&(shapeids==hi))
    if not len(ids):continue
    parts,(xyz,a,n,pid)=parts_for(obj,particle,hi,points[ids]);ii,dd,mm=boundary_classification(parts,xyz,n,pid)
    internal[ids]=ii;duplicate[ids]=dd;multiplicity[ids]=mm;areas[ids]=a;parts_count+=len(parts)
 return dict(internal=internal,coincident_external=duplicate,multiplicity=multiplicity,unweighted_area=areas,parts_count=parts_count)

def amounts(values,m):
 total=float(values.sum());internal=float(values[m['internal']].sum());dup=float(values[m['coincident_external']].sum())
 return dict(total_m2=total,internal_sample_m2=internal,internal_fraction=internal/total if total else 0.,coincident_external_sample_m2=dup,coincident_external_fraction=dup/total if total else 0.,
  duplicate_redundancy_center_quadrature_estimate_m2=float((values[m['coincident_external']]*(1-1/m['multiplicity'][m['coincident_external']])).sum()))
def inspect(label,source,points,oids,pids,sids,objects,weights,base,matrix,candidate_ids,mask=None):
 m=mask or masks(points,oids,pids,sids,objects);expected=np.zeros(len(points))
 for oi,o in enumerate(objects):
  ids=np.flatnonzero(oids==oi);joint=np.asarray(o['joint_probability_mass']);expected[ids]=m['unweighted_area'][ids]*joint[pids[ids],sids[ids]]
 np.testing.assert_allclose(expected,base,rtol=1e-12,atol=1e-14)
 assert np.all((weights==0)|np.isclose(weights,base,rtol=0,atol=0))
 rows=[dict(candidate_id=int(c),**amounts(matrix[i],m))for i,c in enumerate(candidate_ids)]
 return dict(label=label,source=source,points=len(points),raw=amounts(base,m),after_history=amounts(weights,m),candidate_rows=rows,geometry_basis_reproduced=True,primitive_area_weights_reproduced=True),m

def main():
 results=[]
 for layout,states in {'shelf_left':[24,54,109],'shelf_right':[24,53,110]}.items():
  for paid in states:
   folder=B9/'structure/replay_attempt_001'/f'{layout}_{paid}';r=read(folder/'result.json');mask=None
   for name in ('G','S'):
    arm=r['arms'][name];f=Path(arm['arrays']['path']);assert sha(f)==arm['arrays']['sha256'];a=dict(np.load(f,allow_pickle=False))
    row,mask=inspect(f'fixed66:{layout}:{paid}:{name}',ref(f),a['points'],a['object_ids'],a['particle_ids'],a['hypothesis_ids']%2,arm['model_audit']['objects'],a['weights'],a['base_weights'],a['route_union_gain_matrix'],a['candidate_ids'],mask)
    results.append(row)
 mech=B9/'semantic/mechanism_v2_meshfix/candidate_attempt_002'
 for layout in ('shelf_left','shelf_right'):
  mask=None
  for T in (1,16):
   for cue in ('correct','neutral','swapped'):
    f=mech/layout/f'T{T}_{cue}_predictions.npz';a=dict(np.load(f,allow_pickle=False));r=read(f.with_suffix('.json'))
    for name in ('G','S'):
     row,mask=inspect(f'mechanism8:{layout}:T{T}:{cue}:{name}',ref(f),a[name+'_points'],a[name+'_object_ids'],a[name+'_particle_ids'],a[name+'_shape_ids'],r['model_audits'][name]['objects'],a[name+'_weights'],a[name+'_base_weights'],a[name+'_route_union_gain_matrix'],a['candidate_ids'],mask)
     results.append(row)
 out=dict(status='complete_frozen_primitive_surface_audit',results=results,source_geometry_loaded=False,actor_executed=False,new_actions=0,source_files_modified=False,
  units='Quadrature- and probability-weighted template samples, not unique physical measured area or independent observations.',
  caveats=['Internal/duplicate classification is at archived face quadrature centers; finite patch overlap fractions are not analytically integrated.',
   'Alternative particles/shapes are mutually exclusive: masks are per individual hypothesis, never a union across alternatives.',
   'Reference sampling removes exact coincident triangles then tests first-hit visibility; it is not a complete mesh boolean union.',
   'Positive internal gains can arise when B2 repairs self-occlusion by removing intervening predicted surfaces contradicted by measured free evidence; eliminating all original-union internal faces would change this declared behavior.'],
  code=ref(Path(__file__)))
 write(HERE/'parts_audit.json',out)
 table=[dict(label=r['label'],**c)for r in results for c in r['candidate_rows']]
 with(HERE/'parts_candidates.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 print(json.dumps(dict(status=out['status'],model_conditions=len(results),candidate_rows=len(table),max_gain_internal_fraction=max(r['internal_fraction']for r in table),max_gain_coincident_fraction=max(r['coincident_external_fraction']for r in table),output=ref(HERE/'parts_audit.json'))))
if __name__=='__main__':main()
