import sys,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from numpy_archive import savez_fast

class ArchiveTest(unittest.TestCase):
 def test_values_and_layout(self):
  bits=np.array([0x7fc00001,0x7fc12345,0xff800000],dtype=np.uint32)
  values={'nan_payload':bits.view(np.float32),'str':'hello 世界','scalar':2.25,'integer':123,'bool':True,'empty':np.empty((0,3),dtype='>f4'),'noncontiguous':np.arange(48,dtype='>i4').reshape(6,8)[::2,::2],'fortran':np.asfortranarray(np.arange(12).reshape(3,4)),'bytes':b'abc'}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.npz';savez_fast(p,**values)
   with zipfile.ZipFile(p) as z:
    self.assertEqual(set(z.namelist()),{k+'.npy' for k in values});self.assertTrue(all(x.compress_type==zipfile.ZIP_DEFLATED for x in z.infolist()))
   with np.load(p,allow_pickle=False) as z:
    self.assertEqual(set(z.files),set(values))
    for k,v in values.items():
     a=np.asanyarray(v);b=z[k];self.assertEqual(a.dtype,b.dtype,k);self.assertEqual(a.shape,b.shape,k);self.assertEqual(a.tobytes(),b.tobytes(),k)
 def test_file_object(self):
  import io
  x=io.BytesIO();savez_fast(x,a=[1,2]);x.seek(0)
  with np.load(x) as z:np.testing.assert_array_equal(z['a'],[1,2])
if __name__=='__main__':unittest.main()
