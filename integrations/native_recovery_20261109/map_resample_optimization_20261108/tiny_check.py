"""Tiny deterministic math only; no actual-grid run or performance benchmark."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import pathlib,json,hashlib,numpy as np
from reference_function import conservative_belief as old
from candidate import conservative_belief as new
D=pathlib.Path(__file__).resolve().parent;results=[]
for label,angle,translation,res,source,shape in [
 ('identity_occupied',0,[0,0],.1,np.array([[0,0,0,0],[0,100,0,0],[0,0,-1,0],[0,0,0,0]],np.int8),(2,2)),
 ('intermediate_unknown',0,[0,0],.1,np.array([[0,0,0,0],[0,1,0,0],[0,0,99,0],[0,0,0,0]],np.int16),(2,2)),
 ('positive_area_touch',0,[.2,0],.2,np.array([[100,0,0],[0,0,0]],np.int8),(2,3)),
 ('near_touch_low',0,[.2-1e-11,0],.2,np.array([[100,0,0],[0,-1,0]],np.int8),(2,3)),
 ('near_touch_high',0,[.2+1e-11,0],.2,np.array([[100,0,0],[0,-1,0]],np.int8),(2,3)),
 ('rotated47',47,[.12,-.04],.1,np.array([[0,100,0],[0,-1,0],[37,0,0],[0,0,0]],np.int8),(3,4)),
 ('rotated90',90,[.4,.2],.1,np.zeros((4,5),np.int8),(2,3)),
 ('outside',-21,[5,-2],.05,np.zeros((2,3),np.int8),(2,3)),
 ('unsigned',12,[.1,.2],.1,np.array([[0,100],[0,0]],np.uint8),(2,2))]:
 a=np.deg2rad(angle);T=np.eye(4);T[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]];T[:2,3]=translation;x=old(source,res,T,shape);y=new(source,res,T,shape);assert x.dtype==y.dtype==np.int8 and np.array_equal(x,y),label;results.append(dict(case=label,shape=list(shape),dtype=str(y.dtype),exact_equal=True,output=y.tolist()))
(D/'tiny_validation.json').write_text(json.dumps(dict(status='passed_tiny_math_only_not_full_equality_or_performance',cases=results,source_hashes={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['reference_function.py','candidate.py','tiny_check.py']}),indent=2)+'\n');print(json.dumps(dict(tiny_cases=len(results),exact_equal=True,actual_maps_evaluated=0,benchmarks_run=0)))
