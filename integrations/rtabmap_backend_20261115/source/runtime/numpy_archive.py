"""Write ordinary NumPy NPZ archives with fast lossless DEFLATE compression."""
import zipfile
import numpy as np


def savez_fast(file, **arrays):
    """Save keyword arrays in standard NPZ format, preserving NumPy conversion."""
    with zipfile.ZipFile(file, mode='w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name + '.npy', mode='w', force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=True)
