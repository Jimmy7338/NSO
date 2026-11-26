"""Optional subscriber + HTTP proxy. No publishers, commands, bag or cloud files."""
import argparse, copy, json, math, signal, struct, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
from urllib.parse import urlsplit

LABEL='回环校正重建预览（仅新增观测）'
FRAME='nso_loop_map'
TOPIC='/nso_optimized3d_preview/assembler/cloud_map'
MAX_POINTS=50000

def build_cloud(points, stamp, frame, voxel=.05, color_field=None):
    if frame != FRAME: raise ValueError('unexpected optimized preview frame: '+frame)
    seen=set();positions=[];colors=[];finite_count=0;truncated=False
    for row in points:
        x,y,z=map(float,row[:3])
        if not all(map(math.isfinite,(x,y,z))):continue
        finite_count+=1;key=(math.floor(x/voxel),math.floor(y/voxel),math.floor(z/voxel))
        if key in seen:continue
        if len(seen)>=MAX_POINTS:truncated=True;continue
        seen.add(key);positions.extend((x,y,z))
        if color_field:
            packed=struct.unpack('=I',struct.pack('=f',row[3]))[0] if isinstance(row[3],float) else int(row[3])
            colors.extend((((packed>>16)&255)/255.,((packed>>8)&255)/255.,(packed&255)/255.))
        else:colors.extend((.65,.78,.85))
    return dict(positions=positions,colors=colors,frame=frame,stamp=stamp,source_stamp_ns=str(round(stamp*1e9)),
        source_topic=TOPIC,geometry_source='RTABMap_optimized_graph_new_observations_preview',physical_accuracy_verified=False,
        preview=True,label=LABEL,point_count=len(seen),finite_source_points=finite_count,preview_voxel_m=voxel,
        max_points=MAX_POINTS,truncated=truncated,sampling='first endpoint per occupied preview voxel; first50000 voxels if capped')

class Latest:
    def __init__(self,voxel):
        self.lock=threading.Lock();self.voxel=voxel;self.cloud=None;self.last_received=None;self.error=None;self.last_processed=0.;self.source_resolution={}
    def snapshot(self):
        with self.lock:
            c=self.cloud
            return dict(preview=True,label=LABEL,status='waiting' if c is None else ('stale' if time.time()-c['stamp']>10 else 'preview'),
                frame=FRAME,stamp=c['stamp']if c else None,point_count=c['point_count']if c else 0,
                reconstructed_point_count=c['point_count']if c else 0,voxel_m=self.voxel,
                preview_voxel_m=self.voxel,source_resolution=self.source_resolution,source_topic=TOPIC,
                physical_accuracy_verified=False,latest_only_RAM=True,error=self.error,
                scope='Only assembler observations since module start; graph-corrected preview, not complete history or validated reconstruction accuracy')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--voxel-m',type=float,default=.05)
    ap.add_argument('--upstream',default='http://127.0.0.1:8766');args=ap.parse_args()
    if not math.isfinite(args.voxel_m)or args.voxel_m<=0:ap.error('positive finite voxel required')
    if args.upstream!='http://127.0.0.1:8766':ap.error('fixed existing local upstream only')
    import rospy
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2
    rospy.init_node('nso_optional_optimized3d_viewer',disable_signals=True,disable_rosout=True)
    latest=Latest(args.voxel_m)
    # Report actual available ROS parameters; never infer assembler voxel from UI default.
    for key in ['/nso_optimized3d_preview/assembler/Grid/CellSize','/nso_optimized3d_preview/assembler/cloud_output_voxelized','/nso_loop/rtabmap/Grid/CellSize']:
        latest.source_resolution[key]=rospy.get_param(key,None)
    def receive(m):
        if time.monotonic()-latest.last_processed<1:return
        latest.last_processed=time.monotonic()
        try:
            fields={f.name for f in m.fields};color='rgb'if'rgb'in fields else'rgba'if'rgba'in fields else None
            names=('x','y','z')+((color,)if color else())
            c=build_cloud(point_cloud2.read_points(m,field_names=names,skip_nans=False),m.header.stamp.to_sec(),m.header.frame_id,args.voxel_m,color)
            # Preserve original integer header stamp, not float-reconstructed nanoseconds.
            c['source_stamp_ns']=str(m.header.stamp.to_nsec())
            with latest.lock:latest.cloud=c;latest.last_received=time.time();latest.error=None
        except Exception as e:
            with latest.lock:latest.error=str(e)
    sub=rospy.Subscriber(TOPIC,PointCloud2,receive,queue_size=1,buff_size=16*1024*1024)
    assets=Path(__file__).resolve().parent/'assets'
    def fetch(route):
        with urlopen(args.upstream+route,timeout=2)as r:return r.read(),r.headers.get_content_type()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=urlsplit(self.path).path
            try:
                if route=='/api/cloud.json':
                    with latest.lock:c=latest.cloud
                    if c is None:self.send_error(503,'Waiting for optimized cloud');return
                    payload=json.dumps(c,allow_nan=False).encode();kind='application/json'
                elif route=='/api/state':
                    try:body,_=fetch(route);state=json.loads(body)
                    except Exception:state=dict(errors=['Upstream camera/2D state unavailable'])
                    state['reconstruction']=latest.snapshot();state['optimized_preview']=dict(label=LABEL,original_url='http://192.168.31.216:8766/',read_only=True)
                    state.setdefault('freshness',{})['reconstruction']=dict(fresh=state['reconstruction']['status']=='preview',age_s=time.time()-state['reconstruction']['stamp']if state['reconstruction']['stamp']else None)
                    payload=json.dumps(state,allow_nan=False).encode();kind='application/json'
                elif route in('/api/camera.jpg','/api/map.png'):payload,kind=fetch(route)
                elif route in('/','/index.html','/dashboard.js','/dashboard.css'):
                    name='index.html'if route=='/'else route[1:];payload=(assets/name).read_bytes();kind={'index.html':'text/html; charset=utf-8','dashboard.js':'application/javascript','dashboard.css':'text/css'}[name]
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
            except(BrokenPipeError,ConnectionResetError):pass
            except Exception:self.send_error(503)
        def log_message(self,*unused):pass
    server=ThreadingHTTPServer(('192.168.31.216',8767),Handler)
    def stop(*unused):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    print(json.dumps(dict(url='http://192.168.31.216:8767/',upstream=args.upstream,topic=TOPIC,frame=FRAME,label=LABEL,preview_voxel_m=args.voxel_m,source_resolution=latest.source_resolution,read_only=True,accuracy_verified=False)),flush=True)
    try:server.serve_forever(poll_interval=.2)
    finally:server.server_close();sub.unregister();rospy.signal_shutdown('optional viewer stopped')

if __name__=='__main__':main()
