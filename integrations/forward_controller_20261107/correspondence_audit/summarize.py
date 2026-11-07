#!/usr/bin/env python3
"""Summarize retained CSM associations; only estimates diagnostic Jacobian conditioning."""
import collections,json,math,os,pathlib,numpy as np
P=pathlib.Path(os.environ['AUDIT_OUTPUT_DIR']).resolve()
rows=collections.defaultdict(list)
for line in (P/'correspondence.tmp').read_text().splitlines():
 v=line.split()
 if not v or v[0]!='CORR':continue
 ref,i,j1,j2,typ=map(int,v[1:6]);angle,r,rr,res,dist,x,y,nx,ny=map(float,v[6:]);rows[ref].append(dict(i=i,j1=j1,j2=j2,type=typ,angle=angle,range=r,ref_range=rr,residual=res,nearest_dist=dist,x=x,y=y,nx=nx,ny=ny))

def stats(a):
 if not a:return None
 r=np.array([x['residual'] for x in a]);return {'n':len(a),'mean_abs_m':float(np.mean(abs(r))),'median_abs_m':float(np.median(abs(r))),'p90_abs_m':float(np.quantile(abs(r),.9)),'p99_abs_m':float(np.quantile(abs(r),.99)),'rms_m':float(np.sqrt(np.mean(r*r))),'sum_sq_m2':float(r@r)}
results={}
for ref,a in rows.items():
 j=np.array([[x['nx'],x['ny'],-x['nx']*x['y']+x['ny']*x['x']] for x in a]);H=j.T@j;e=np.linalg.eigvalsh(H)
 angle={};
 for lo in range(-180,180,45):
  segment=[x for x in a if lo<=math.degrees(x['angle'])<(lo+45)];angle[f'{lo:+d}..{lo+45:+d}']=stats(segment)
 distance={}
 for lo,hi in [(0,1),(1,2),(2,4),(4,6),(6,8),(8,12),(12,float('inf'))]:distance[f'{lo}..{hi}m']=stats([x for x in a if lo<=x['range']<hi])
 results[str(ref)]={'all':stats(a),'correspondence_types':dict(collections.Counter(str(x['type']) for x in a)),'angle_degrees':angle,'range_m':distance,'gauss_newton_jtj_unweighted':{'matrix':H.tolist(),'eigenvalues':e.tolist(),'condition_number':float(e[-1]/e[0]) if e[0]>0 else None,'description':'Unweighted point-to-line local Jacobian [nx,ny,-nx*y+ny*x] at reported final pose; x/y in reference frame, radians for rotation. Not solver Hessian or normalized condition; units/origin affect number.'}}
common=sorted(set(x['i'] for x in rows[96]) & set(x['i'] for x in rows[98]));byref={ref:{x['i']:x for x in a} for ref,a in rows.items()};common_sectors={}
for lo,hi in [(-180,45),(45,90),(90,135),(135,180)]:
 keys=[i for i in common if lo<=math.degrees(byref[96][i]['angle'])<hi]
 common_sectors[f'{lo:+d}..{hi:+d}']={'n':len(keys),'ref096':stats([byref[96][i] for i in keys]),'ref098':stats([byref[98][i] for i in keys])}
out={'common_target_ray_count':len(common),'common_target_ray_sectors':common_sectors,'scope':'retained post-sm_icp correspondence flags, fixed reference/input/parameters, residual evaluated at output pose; no refitting and no sample changes','limits':['Correspondences may originate at final iteration linearization rather than recomputed at returned output pose','Residual summaries unweighted and do not equal library error','No ground-truth match or physical surface identity','Unweighted JtJ approximates geometry only; not solver covariance'],'results':results}
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
for k,x in results.items():
 print(k,x['all'],'condition',x['gauss_newton_jtj_unweighted']['condition_number'])
 for typ in ('angle_degrees','range_m'):
  print(typ,[(key,round(v['sum_sq_m2'],3),v['n'],round(v['rms_m'],3)) for key,v in x[typ].items() if v])
