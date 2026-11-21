import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import numpy as np,json,pathlib,hashlib
from reference_function import conservative_belief as old
from candidate_compact import conservative_belief as new
rng=np.random.default_rng(20261109);cases=[]
for i in range(200):
 shape=(int(rng.integers(1,18)),int(rng.integers(1,18)));a=rng.choice([-1,0,1,37,99,100],size=(int(rng.integers(1,30)),int(rng.integers(1,30)))).astype([np.int8,np.int16,np.int32][i%3]);res=[.025,.05,.1,.2,.3][i%5];angle=[0,np.pi/2,np.pi,-np.pi/2,rng.uniform(-np.pi,np.pi)][i%5];T=np.eye(4);T[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]];T[:2,3]=rng.uniform(-2,2,2)
 if i%4==0:T[0,3]=[0,.1,.2,-.2][i%4]+[-1e-11,0,1e-11][i%3]
 if i%11==0:T[0,0]+=1e-7
 x=old(a,res,T,shape);y=new(a,res,T,shape);assert x.dtype==y.dtype==np.int8 and np.array_equal(x,y),i
 cases.append(dict(case=i,shape=shape,source_shape=a.shape,resolution_m=res,dtype=str(a.dtype),exact_equal=True))
D=pathlib.Path(__file__).resolve().parent;(D/'compact_edge_validation.json').write_text(json.dumps(dict(seed=20261109,passed=len(cases),cases=cases,candidate_sha256=hashlib.sha256((D/'candidate_compact.py').read_bytes()).hexdigest()),indent=2)+'\n');print(len(cases))
