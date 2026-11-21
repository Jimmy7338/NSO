"""Passive bounded confidence evidence; never publishes or calls services."""
from collections import OrderedDict
import hashlib,io,json,threading,time
from pathlib import Path
import numpy as np

def decode(message):
    if message.encoding!='32FC1':raise ValueError('confidence requires 32FC1')
    h,w,step=int(message.height),int(message.width),int(message.step)
    if h<=0 or w<=0 or step<w*4 or len(message.data)<h*step:
        raise ValueError('invalid image shape, row step or truncated data')
    dtype=np.dtype('>f4' if message.is_bigendian else '<f4')
    return np.ndarray((h,w),dtype=dtype,buffer=message.data,strides=(step,4))[::4,::4].astype(np.float32,copy=True)

class ConfidenceRing:
    def __init__(self,rospy):
        from sensor_msgs.msg import Image
        self.lock=threading.Lock();self.frames=OrderedDict();self.errors=0
        self.sub=rospy.Subscriber('/zed2i/zed_node/confidence/confidence_map',Image,self._receive,queue_size=1,buff_size=4194304)
    def _receive(self,msg):
        try:
            a=decode(msg)
            # Bound canonical frame storage to 300 * 180 * 320 * 4 bytes.
            if a.shape!=(180,320):raise ValueError('expected actual 720x1280 confidence source')
            b=io.BytesIO();msg.serialize(b)
            meta=dict(stamp_ns=msg.header.stamp.to_nsec(),seq=msg.header.seq,frame_id=msg.header.frame_id,caller=(msg._connection_header or {}).get('callerid'),receipt_unix_ns=time.time_ns(),encoding=msg.encoding,is_bigendian=bool(msg.is_bigendian),step=int(msg.step),source_shape=[msg.height,msg.width],full_serialized_message_sha256=hashlib.sha256(b.getvalue()).hexdigest(),sampling='exact [::4,::4]; no confidence threshold or depth modification')
            with self.lock:
                self.frames[meta['stamp_ns']]=(a,meta);self.frames.move_to_end(meta['stamp_ns'])
                while len(self.frames)>300:self.frames.popitem(last=False)
        except Exception:
            with self.lock:self.errors+=1
    def save_for_observed(self,report,out):
        report=Path(report);out=Path(out);out.mkdir(parents=True,exist_ok=True)
        candidates=[];failures=[]
        for p in (report/'observed').glob('*.npz'):
            try:
                with np.load(p,allow_pickle=False) as z:d=json.loads(str(z['receipt_json'].item()))
                candidates.append((int(d['headers']['depth']['stamp_ns']),p,d['headers']['depth']))
            except Exception as e:failures.append(dict(path=str(p),error=str(e)))
        rows=[]
        for stamp,p,header in sorted(candidates)[-3:]:
            with self.lock:item=self.frames.get(stamp)
            row=dict(observed=str(p),depth_header=header,match='missing_exact_stamp')
            if item is not None:
                a,meta=item;target=out/(str(stamp)+'.confidence.npz')
                np.savez_compressed(target,confidence=a,provenance_json=json.dumps(dict(meta,observed=str(p),depth_header=header)))
                row.update(match='exact_original_header_stamp',path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),confidence_header=meta)
            rows.append(row)
        receipt=dict(scope='terminal-only passive confidence evidence; absent matches remain missing',observations=rows,read_errors=failures,decoder_errors=self.errors,max_frames=300,max_canonical_bytes=69120000)
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
    def close(self):self.sub.unregister()
