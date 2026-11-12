import numpy as np

def conservative_belief(occupancy, raw_resolution_m, planning_from_grid, shape,
                        planning_resolution_m=.2):
    """Exact positive-area square overlap, with no unknown-to-free conversion.

    ROS source rows run along +y; destination rows are reversed, as in the
    original joint planner. Any overlapping occupied cell -> occupied. Free
    requires every overlapping source cell to be in-bounds and exactly zero.
    Unknown, out-of-map and intermediate probabilities otherwise stay unknown.
    No nearest-neighbour interpolation or morphological safety inflation.
    """
    source=np.asarray(occupancy);T=np.asarray(planning_from_grid,dtype=float)
    if (source.ndim!=2 or not np.issubdtype(source.dtype,np.integer) or
            np.any(source < -1) or np.any(source>100)):
        raise ValueError('raw ROS occupancy values required')
    if (len(shape)!=2 or any(type(n) is not int or n<1 for n in shape) or
            not np.isfinite(raw_resolution_m) or raw_resolution_m<=0 or
            not np.isclose(planning_resolution_m,.2,rtol=0,atol=1e-12)):
        raise ValueError('declared fixed shape and original 0.2 m joint grid required')
    if (T.shape!=(4,4) or not np.isfinite(T).all() or
            not np.allclose(T[3],[0,0,0,1]) or
            not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or
            not np.isclose(np.linalg.det(T[:3,:3]),1.,atol=1e-5) or
            not np.allclose(T[2,:2],0.,atol=1e-6) or not np.isclose(T[2,2],1.,atol=1e-6)):
        raise ValueError('actual planar rigid map/planning binding required')
    inv=np.linalg.inv(T);height,width=shape
    row,col=np.indices(shape)
    cx=(col.ravel()+.5)*planning_resolution_m
    cy=(height-row.ravel()-.5)*planning_resolution_m
    x=inv[0,0]*cx+inv[0,1]*cy+inv[0,3]
    y=inv[1,0]*cx+inv[1,1]*cy+inv[1,3]
    u=inv[:2,0];v=inv[:2,1];half=planning_resolution_m*.5;finehalf=raw_resolution_m*.5
    extent=half*(np.abs(u)+np.abs(v))
    bound_u=half+finehalf*np.abs(u).sum();bound_v=half+finehalf*np.abs(v).sum()
    ix=np.floor(x/raw_resolution_m).astype(np.int64)
    iy=np.floor(y/raw_resolution_m).astype(np.int64)
    free=np.ones(x.shape,bool);occupied=np.zeros(x.shape,bool);covered=np.zeros(x.shape,bool)
    radius=np.ceil((extent+finehalf)/raw_resolution_m).astype(int)+1
    epsilon=1e-10
    for dy in range(-int(radius[1]),int(radius[1])+1):
        sy=iy+dy;delta_y=(sy+.5)*raw_resolution_m-y
        for dx in range(-int(radius[0]),int(radius[0])+1):
            sx=ix+dx;delta_x=(sx+.5)*raw_resolution_m-x
            overlap=((np.abs(delta_x)<extent[0]+finehalf-epsilon)&
                     (np.abs(delta_y)<extent[1]+finehalf-epsilon)&
                     (np.abs(delta_x*u[0]+delta_y*u[1])<bound_u-epsilon)&
                     (np.abs(delta_x*v[0]+delta_y*v[1])<bound_v-epsilon))
            inside=(sx>=0)&(sx<source.shape[1])&(sy>=0)&(sy<source.shape[0])
            values=source[np.clip(sy,0,source.shape[0]-1),np.clip(sx,0,source.shape[1]-1)]
            free &= ~(overlap&(~inside|(values!=0)))
            occupied |= overlap&inside&(values==100)
            covered |= overlap
    belief=np.full(x.shape,-1,np.int8);belief[covered&free]=0;belief[occupied]=1
    return belief.reshape(shape)
