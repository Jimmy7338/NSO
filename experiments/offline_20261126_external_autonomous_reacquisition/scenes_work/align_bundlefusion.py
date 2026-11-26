"""Register the source camera frame to metric Z-up without modifying native geometry."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

from survey_bundlefusion import TRIANGLE, VERTEX


def header(path):
    with path.open("rb") as stream:
        lines=[]
        for line in stream:
            lines.append(line)
            if line==b"end_header\n":
                break
        offset=stream.tell()
    raw=b"".join(lines)
    v=int(re.search(rb"element vertex (\d+)",raw)[1])
    f=int(re.search(rb"element face (\d+)",raw)[1])
    if path.stat().st_size!=offset+v*VERTEX.itemsize+f*TRIANGLE.itemsize:
        raise ValueError("Unexpected PLY layout")
    return offset,v,f


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def align(native, output):
    offset,nv,nf=header(native)
    vertices=np.memmap(native,mode="r",dtype=VERTEX,offset=offset,shape=(nv,))
    faces=np.memmap(native,mode="r",dtype=TRIANGLE,offset=offset+nv*VERTEX.itemsize,shape=(nf,))
    rng=np.random.default_rng(20261126401)
    ids=np.sort(rng.choice(nf,size=min(nf,350000),replace=False))
    if not np.all(faces["arity"][ids]==3):
        raise ValueError("PLY non-triangle face at geometry check")
    triangles=vertices["coord"][faces["index"][ids]].astype(float)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    magnitude=np.linalg.norm(cross,axis=1)
    centers=triangles.mean(1)
    # BundleFusion identifies the first depth image's optical frame as world.
    # Positive optical Y points downward in the camera convention. The first
    # camera can have pitch/roll, so estimate vertical from broad parallel
    # surfaces below that camera before applying the original floor admission.
    candidate=(magnitude>1e-10) & (np.abs(cross[:,1])>=.75*magnitude) & (centers[:,1]>.35)
    weights=magnitude[candidate]/2
    normals=cross[candidate]/magnitude[candidate,None]
    normals *= np.where(normals[:,1]<0,1.,-1.)[:,None]
    coords=np.rint(normals[:,[0,2]]/.035).astype(int)
    keys,reverse=np.unique(coords,axis=0,return_inverse=True)
    i=int(np.argmax(np.bincount(reverse,weights=weights)))
    direction=np.average(normals[reverse==i],axis=0,weights=weights[reverse==i])
    direction/=np.linalg.norm(direction)
    coherent=normals @ direction >= np.cos(np.deg2rad(6.))
    elev=centers[candidate][coherent]@direction
    z_bin=np.floor(elev/.05).astype(int)
    z_keys,z_inverse=np.unique(z_bin,return_inverse=True)
    rank=int(np.argmax(np.bincount(z_inverse,weights=weights[coherent])))
    on_surface=np.abs(elev-(z_keys[rank]+.5)*.05)<=.05
    surface=np.flatnonzero(candidate)[coherent][on_surface]
    pts=triangles[surface].reshape(-1,3)
    measure=pts-pts.mean(axis=0)
    _,basis=np.linalg.eigh(measure.T@measure)
    up=basis[:,0]
    if up[1]>0:
        up=-up
    if up @ direction < np.cos(np.deg2rad(6.)):
        raise ValueError("Estimated vertical is inconsistent under two geometric estimates")
    east=np.array([1.,0.,0.])-up[0]*up
    east/=np.linalg.norm(east)
    axes=np.stack((east,np.cross(up,east),up))
    if np.linalg.det(axes)<.999999:
        raise ValueError("The rigid registration introduced a scale or mirror")
    shutil.copyfile(native,output)
    out=np.memmap(output,mode="r+",dtype=VERTEX,offset=offset,shape=(nv,))
    for first in range(0,nv,250_000):
        sl=slice(first,first+250_000)
        out["coord"][sl]=out["coord"][sl].astype(float) @ axes.T
    out.flush()
    residual=np.abs(measure@up)
    return {"record":"bundlefusion_native_gravity_normalization_v1","native_path":str(native),
            "native_asset_sha256":sha(native),"normalized_asset_sha256":sha(output),
            "reference":"https://pure.mpg.de/rest/items/item_2351932_3/component/file_2351931/arXiv%3A1604.01093.pdf",
            "rotation_from_native_xyz":axes.tolist(),"translation_from_native_xyz_m":[0,0,0],
            "inferred_native_up_unit":up.tolist(),"normal_mode_up_unit":direction.tolist(),
            "fit_triangles":int(len(surface)),"total_source_triangles_unchanged":nf,
            "raw_colors_and_face_connectivity_retained":True,
            "median_plane_fit_residual_m":float(np.median(residual)),"p95_plane_fit_residual_m":float(np.quantile(residual,.95)),
            "policy_calls":0}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    print(json.dumps(align(args.native,args.out),indent=2,allow_nan=False))
