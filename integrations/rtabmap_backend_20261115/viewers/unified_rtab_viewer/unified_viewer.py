"""Read-only same-session RTAB display; never commands, TF or navigation."""
import argparse,copy,io,json,math,signal,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import urlopen
from urllib.parse import urlsplit
import numpy as np
from PIL import Image
FRAME='nso_loop_map'
LABEL='会话图优化彩色点云（全历史请求；非TSDF）'
CLOUD='/nso_unified_rtab/assembler/cloud_map'

def planar(t):
    q=t.rotation;p=t.translation
    values=[q.x,q.y,q.z,q.w,p.x,p.y,p.z]
    if not all(map(math.isfinite,values)) or abs(sum(x*x for x in values[:4])-1)>1e-5:raise ValueError('invalid transform')
    return [p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))]
def point(t,p):
    c,s=math.cos(t[2]),math.sin(t[2]);return [t[0]+c*p[0]-s*p[1],t[1]+s*p[0]+c*p[1]]
def age(stamp,now,limit):
    return dict(fresh=stamp is not None and 0<=now-stamp<=limit,age_s=None if stamp is None else now-stamp)
def project_semantics(source,correction,epoch_start):
    if source.get('frame')!='odom' or source.get('stamp',0)<epoch_start:raise ValueError('semantic frame/epoch mismatch')
    result=copy.deepcopy(source)
    for item in result.get('objects',[]):
        for p in [[item['x'],item['y']]]+item.get('polygon',[]):
            if len(p)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in p):raise ValueError('invalid semantic coordinates')
        item['x'],item['y']=point(correction,[item['x'],item['y']]);item['polygon']=[point(correction,p) for p in item.get('polygon',[])]
    result.update(frame=FRAME,source_frame='odom',semantic_scope='latest measured semantic packet projected with current graph correction; not historical optimized semantic fusion')
    return result

def cloud_message(m,maximum=50000):
    if m.header.frame_id!=FRAME:raise ValueError('cloud frame mismatch')
    fields={f.name:f for f in m.fields};end='>' if m.is_bigendian else '<'
    if any(fields[k].datatype!=7 for k in ('x','y','z')):raise ValueError('expected float32XYZ')
    names=['x','y','z'];formats=[end+'f4']*3;offsets=[fields[k].offset for k in names]
    color='rgb' if 'rgb' in fields else 'rgba' if 'rgba' in fields else None
    if color:names.append('color');formats.append(end+'u4');offsets.append(fields[color].offset)
    dtype=np.dtype(dict(names=names,formats=formats,offsets=offsets,itemsize=m.point_step))
    data=np.ndarray((m.height,m.width),dtype=dtype,buffer=m.data,strides=(m.row_step,m.point_step)).reshape(-1)
    stride=max(1,math.ceil(len(data)/maximum));sample=data[::stride]
    xyz=np.column_stack([sample[k] for k in ('x','y','z')]);valid=np.isfinite(xyz).all(axis=1);xyz=xyz[valid]
    if color:
        bits=sample['color'][valid];colors=np.column_stack(((bits>>16)&255,(bits>>8)&255,bits&255))/255.
    else:colors=np.tile([.65,.78,.85],(len(xyz),1))
    return dict(positions=xyz.ravel().astype(float).tolist(),colors=colors.ravel().tolist(),frame=FRAME,
        stamp=m.header.stamp.to_sec(),source_stamp_ns=str(m.header.stamp.to_nsec()),point_count=len(xyz),
        source_point_count=int(m.width*m.height),display_point_stride=stride,max_points=maximum,
        geometry_source='RTAB_optimized_graph_colored_cloud_not_TSDF',physical_accuracy_verified=False,
        preview=True,full_history_requested=True,label=LABEL,source_topic=CLOUD,
        sampling='uniform source-array stride for browser only; assembler history is not cropped by this display cap')

class Bundle:
    def __init__(self,epoch,start,semantic,history):
        self.epoch,self.start,self.semantic,self.history=epoch,start,Path(semantic),Path(history)
        self.lock=threading.RLock();self.map=None;self.png=None;self.graph=None;self.odom=None;self.cloud=None;self.error=[]
    def check(self,m,frame):
        if m.header.frame_id!=frame or m.header.stamp.to_sec()<self.start:raise ValueError('source frame/epoch differs')
    def grid(self,m):
        self.check(m,FRAME);q=m.info.origin.orientation
        if abs(q.x)+abs(q.y)+abs(q.z)>1e-8 or abs(q.w-1)>1e-8:raise ValueError('nonzero grid origin rotation unsupported')
        a=np.asarray(m.data,np.int16).reshape(m.info.height,m.info.width);gray=np.where(a<0,128,np.clip(245*(1-a/100.),8,245)).astype(np.uint8)
        stream=io.BytesIO();Image.fromarray(np.flipud(gray)).save(stream,format='PNG')
        with self.lock:
            self.png=stream.getvalue();self.map=dict(stamp=m.header.stamp.to_sec(),source_stamp_ns=str(m.header.stamp.to_nsec()),width=m.info.width,height=m.info.height,resolution=m.info.resolution,origin=[m.info.origin.position.x,m.info.origin.position.y],frame=FRAME,source_bundle='unified_current_session_RTAB',source_topic='/nso_loop/grid_map',source_label='同会话RTAB图优化地图',epoch_id=self.epoch)
    def graph_msg(self,m):
        self.check(m,FRAME)
        if len(m.posesId)!=len(m.poses):raise ValueError('graph lengths differ')
        trajectory=[[p.position.x,p.position.y] for node,p in sorted(zip(m.posesId,m.poses)) if node>0]
        with self.lock:self.graph=dict(stamp=m.header.stamp.to_sec(),source_stamp_ns=str(m.header.stamp.to_nsec()),correction=planar(m.mapToOdom),trajectory=trajectory,optimized_keyframe_count=len(trajectory))
    def odom_msg(self,m):
        self.check(m,'odom');p=m.pose.pose.position;q=m.pose.pose.orientation
        with self.lock:self.odom=dict(x=p.x,y=p.y,yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),stamp=m.header.stamp.to_sec())
    def snapshot(self,upstream):
        now=time.time()
        with self.lock:g=copy.deepcopy(self.graph);o=copy.deepcopy(self.odom);c=self.cloud;meta=copy.deepcopy(self.map)
        result={k:copy.deepcopy(upstream.get(k)) for k in ('camera','robot')};result.update(map=meta,pose=None,trajectory=[],errors=list(self.error),stamp=now)
        if g:
            result['trajectory']=g['trajectory'];result['trajectory_source']=dict(storage_frame=FRAME,display_frame=FRAME,kind='RTAB optimized keyframe poses',graph_source_stamp_ns=g['source_stamp_ns'],maximum_points=None)
            if o:
                x,y=point(g['correction'],[o['x'],o['y']]);result['pose']=dict(x=x,y=y,yaw=o['yaw']+g['correction'][2],frame=FRAME,stamp=o['stamp'])
            try:result['semantics']=project_semantics(json.loads(self.semantic.read_text()),g['correction'],self.start)
            except Exception as e:result['semantics']=dict(frame=FRAME,status='unavailable',objects=[],reason=str(e),stamp=None)
        else:result['semantics']=dict(frame=FRAME,status='waiting_for_graph',objects=[],stamp=None)
        history=None
        try:history=json.loads(self.history.read_text())
        except OSError:pass
        result['reconstruction']=dict(preview=True,label=LABEL,status='preview' if c else 'waiting',frame=FRAME,stamp=c['stamp']if c else None,point_count=c['point_count']if c else 0,reconstructed_point_count=c.get('source_point_count',0)if c else 0,voxel_m=.05,preview_voxel_m=.05,physical_accuracy_verified=False,full_history_requested=bool(history and history.get('full_current_session_requested')),history_load_receipt=history,geometry_source='RTAB graph colored cloud, not2cmTSDF',scope='All data returned by current-session global history service plus live updates; display subsampled; complete RGBD coverage must be checked in load receipt')
        result['freshness']={k:age((result.get(k)or{}).get('stamp'),now,10 if k in ('map','reconstruction','semantics')else 3) for k in ('camera','pose','map','reconstruction','semantics')}
        result['freshness']['graph']=age(g['stamp'] if g else None,now,3)
        for key in ('pose','semantics'):
            result['freshness'][key]['fresh'] &= result['freshness']['graph']['fresh']
        result['unified_source']=dict(epoch_id=self.epoch,frame=FRAME,graph_source_stamp_ns=g['source_stamp_ns']if g else None,cloud_source_stamp_ns=c['source_stamp_ns']if c else None,map_source_stamp_ns=meta.get('source_stamp_ns')if meta else None,exact_stamp_synchronization=False,graph_cloud_delta_s=c['stamp']-g['stamp']if c and g else None,physical_accuracy_verified=False,raw_TSDF_diagnostic_url='/api/raw-tsdf-cloud.json')
        return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epoch-id',required=True);ap.add_argument('--epoch-start-s',type=float,required=True);ap.add_argument('--semantic-state',required=True);ap.add_argument('--history-receipt',required=True);ap.add_argument('--host',default='192.168.31.216');ap.add_argument('--port',type=int,default=8767);a=ap.parse_args()
    import rospy
    from nav_msgs.msg import OccupancyGrid,Odometry
    from rtabmap_msgs.msg import MapGraph
    from sensor_msgs.msg import PointCloud2
    rospy.init_node('nso_unified_rtab_viewer',disable_signals=True)
    state=Bundle(a.epoch_id,a.epoch_start_s,a.semantic_state,a.history_receipt)
    def guarded(callback,caller):
        def call(m):
            try:
                if m._connection_header.get('callerid')!=caller:raise ValueError('unexpected source publisher')
                callback(m)
            except Exception as e:
                with state.lock:state.error=(state.error+[str(e)])[-4:]
        return call
    def cloud(m):
        state.check(m,FRAME);c=cloud_message(m)
        with state.lock:state.cloud=c
    subs=[rospy.Subscriber('/nso_loop/grid_map',OccupancyGrid,guarded(state.grid,'/nso_loop/rtabmap'),queue_size=1),rospy.Subscriber('/nso_loop/mapGraph',MapGraph,guarded(state.graph_msg,'/nso_loop/rtabmap'),queue_size=1),rospy.Subscriber('/nso_csm/odom',Odometry,guarded(state.odom_msg,'/nso_csm_odometry_bridge'),queue_size=1),rospy.Subscriber(CLOUD,PointCloud2,guarded(cloud,'/nso_unified_rtab/assembler'),queue_size=1,buff_size=64*1024*1024)]
    def fetch(route):
        with urlopen('http://127.0.0.1:8766'+route,timeout=2)as r:return r.read(),r.headers.get_content_type()
    assets=Path(__file__).parent/'assets'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=urlsplit(self.path).path
            try:
                if route=='/api/state':
                    try:body,_=fetch('/api/state');upstream=json.loads(body)
                    except Exception:upstream={}
                    payload=json.dumps(state.snapshot(upstream),allow_nan=False).encode();kind='application/json'
                elif route=='/api/map.png':
                    with state.lock:payload=state.png
                    kind='image/png'
                elif route=='/api/cloud.json':
                    with state.lock:c=state.cloud
                    payload=json.dumps(c,allow_nan=False).encode()if c else None;kind='application/json'
                elif route in ('/api/camera.jpg','/api/raw-tsdf-cloud.json'):
                    payload,kind=fetch('/api/cloud.json'if route.endswith('raw-tsdf-cloud.json')else route)
                elif route in ('/','/index.html','/dashboard.js','/dashboard.css'):
                    name='index.html'if route=='/'else route[1:];payload=(assets/name).read_bytes();kind={'index.html':'text/html; charset=utf-8','dashboard.js':'application/javascript','dashboard.css':'text/css'}[name]
                else:self.send_error(404);return
                if payload is None:self.send_error(503);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
            except(BrokenPipeError,ConnectionResetError):pass
            except Exception:self.send_error(503)
        def log_message(self,*unused):pass
    server=ThreadingHTTPServer((a.host,a.port),Handler)
    def stop(*unused):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        for sub in subs:sub.unregister()
        rospy.signal_shutdown('unified viewer stopped')
if __name__=='__main__':main()
