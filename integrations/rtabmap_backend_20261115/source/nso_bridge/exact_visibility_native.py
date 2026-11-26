"""Exact original ordered-ray visibility counts via a local C++ batch kernel.

Caller supplies unchanged ray_templates offsets/valid. No ROS/SDK or policy.
"""
import ctypes
from pathlib import Path
import numpy as np
_LIBRARY = None

def _library():
    global _LIBRARY
    if _LIBRARY is None:
        path=Path(__file__).with_name('libexact_visibility_native.so')
        lib=ctypes.CDLL(str(path))
        u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
        i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        lib.nso_batch_visibility_counts.argtypes=[u8,u8,ctypes.c_int64,ctypes.c_int64,
            i64,ctypes.c_int64,i64,u8,ctypes.c_int64,ctypes.c_int64,i64]
        lib.nso_batch_visibility_counts.restype=ctypes.c_int
        _LIBRARY=lib
    return _LIBRARY

def _integers(value,name):
    array=np.asarray(value)
    if not np.issubdtype(array.dtype,np.integer):raise TypeError(name+' must be integer coordinates')
    return np.ascontiguousarray(array,dtype=np.int64)

def batch_visibility_counts(occupied,query,positions,offsets,valid):
    """Return int64[K] counts; positions/offsets use original ROW,COL order."""
    occupied=np.ascontiguousarray(occupied,dtype=np.bool_)
    query=np.ascontiguousarray(query,dtype=np.bool_)
    positions=_integers(positions,'positions');offsets=_integers(offsets,'offsets')
    valid=np.ascontiguousarray(valid,dtype=np.bool_)
    if occupied.ndim!=2 or min(occupied.shape)<=0 or query.shape!=occupied.shape:raise ValueError('equal nonempty occupied/query grids required')
    if positions.ndim!=2 or positions.shape[1]!=2:raise ValueError('positions must be Kx2')
    if offsets.ndim!=3 or offsets.shape[2]!=2 or min(offsets.shape[:2])<=0 or valid.shape!=offsets.shape[:2]:raise ValueError('original RaysxStepsx2 offsets and RaysxSteps valid required')
    output=np.empty(len(positions),dtype=np.int64)
    if len(output)==0:return output
    status=_library().nso_batch_visibility_counts(occupied.view(np.uint8),query.view(np.uint8),
        occupied.shape[0],occupied.shape[1],positions,len(positions),offsets,valid.view(np.uint8),
        offsets.shape[0],offsets.shape[1],output)
    if status:raise RuntimeError('exact native visibility count failed: '+str(status))
    return output
