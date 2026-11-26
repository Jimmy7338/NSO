"""Rotate publisher SceneNN RGB mesh from +Y up to +Z up without mesh edits."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


VERTEX=np.dtype([("pos","<f4",(3,)),("normal","<f4",(3,)),("label","<u4"),("rgb","u1",(3,))])


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def align(source,dest):
    with source.open("rb") as stream:
        header=[]
        for line in stream:
            header.append(line)
            if line.strip()==b"end_header":
                break
        offset=stream.tell()
    signature=b"".join(header)
    assert all(word in signature for word in [b"format binary_little_endian 1.0",b"property float nx",
                                               b"property uint label",b"property uchar blue",
                                               b"property list uchar int vertex_indices"])
    vertex=int(re.search(rb"element vertex (\d+)",signature)[1])
    faces=int(re.search(rb"element face (\d+)",signature)[1])
    assert source.stat().st_size==offset+vertex*VERTEX.itemsize+faces*13
    shutil.copyfile(source,dest)
    normalized=np.memmap(dest,mode="r+",dtype=VERTEX,offset=offset,shape=(vertex,))
    for name in ("pos","normal"):
        flip=normalized[name][:,2].copy()
        normalized[name][:,2]=normalized[name][:,1]
        normalized[name][:,1]=-flip
    normalized.flush()
    return {"native_publisher_asset_sha256":sha(source),"render_asset_sha256":sha(dest),
            "source_up_axis":"+Y (publisher scan geometry, upward depth-floor bin around Y=0)",
            "normalization_xyz_matrix":[[1,0,0],[0,0,-1],[0,1,0]],
            "vertex_and_face_count":[vertex,faces],"translation_m":[0,0,0],
            "triangle_connectivity_colors_and_all_annotations_retained":True,"actor_calls":0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native",type=Path)
    parser.add_argument("z_up",type=Path)
    args=parser.parse_args()
    print(json.dumps(align(args.native,args.z_up),indent=2)+"\n",end="")


if __name__=="__main__":
    main()
