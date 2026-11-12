import numpy as np
from types import SimpleNamespace
from confidence_ring import decode
for endian in [False,True]:
 a=np.arange(8*12,dtype=np.float32).reshape(8,12)
 rows=[row.astype('>f4' if endian else '<f4').tobytes()+b'PAD!PAD!' for row in a]
 msg=SimpleNamespace(encoding='32FC1',height=8,width=12,step=56,is_bigendian=endian,data=b''.join(rows))
 assert np.array_equal(decode(msg),a[::4,::4])
 msg.data=msg.data[:-1]
 try:decode(msg)
 except ValueError:pass
 else:raise AssertionError('truncated payload accepted')
print('Little/big endian, padded row stride and truncated image checks passed')
