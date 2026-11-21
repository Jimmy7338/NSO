"""B10b proposed reward definition; no policy binding or geometry fitting.

Replace axial historical measured-surface consumption with acquired native
reconstruction support. Verified free rejection remains. Coarse/fine surface
quadrature is an explicit separate diagnostic switch, never a threshold fit.
"""
import numpy as np

def weights(base_weights,strict_free,native_keep):
 base=np.asarray(base_weights,float);free=np.asarray(strict_free);keep=np.asarray(native_keep)
 if base.ndim!=1 or not np.isfinite(base).all()or np.any(base<0):raise ValueError('finite nonnegative one-dimensional base weights required')
 if free.dtype!=bool or keep.dtype!=bool or free.shape!=base.shape or keep.shape!=base.shape:raise ValueError('aligned boolean free and native novelty masks required')
 return base*(~free&keep)

def box_quadrature(origin,dims,resolution_m=.22):
 origin=np.asarray(origin,float);dims=np.asarray(dims,float)
 if origin.shape!=(3,)or dims.shape!=(3,)or not np.isfinite(origin).all()or not np.isfinite(dims).all()or np.any(dims<=0):raise ValueError('finite origin and positive dimensions required')
 if not np.isfinite(resolution_m)or resolution_m<=0:raise ValueError('positive resolution required')
 points=[];areas=[]
 for normal in range(3):
  axes=[i for i in range(3)if i!=normal];counts=[max(1,int(np.ceil(dims[a]/resolution_m)))for a in axes]
  area=dims[axes[0]]*dims[axes[1]]/(counts[0]*counts[1])
  for side in (0.,1.):
   for i in range(counts[0]):
    for j in range(counts[1]):
     p=origin.copy();p[normal]+=side*dims[normal];p[axes[0]]+=(i+.5)*dims[axes[0]]/counts[0];p[axes[1]]+=(j+.5)*dims[axes[1]]/counts[1];points.append(p);areas.append(area)
 return np.asarray(points),np.asarray(areas)

def quadrature(parts,resolution_m=.22):
 values=[box_quadrature(o,d,resolution_m)for o,d in parts]
 return np.concatenate([v[0]for v in values]),np.concatenate([v[1]for v in values])
