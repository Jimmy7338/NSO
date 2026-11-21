#!/usr/bin/env python3
"""Read-only local camera/map/measured-TSDF display. Never publishes robot commands."""
import argparse
from collections import deque
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from mission import rgb_array
from reference_observation import source_depth_metres
from nso_ros_coordinator import matrix
from runtime_environment import bounded_environment

WORK=Path(__file__).resolve().parents[1]
ROOT=Path('/home/ouc/nso_robot')
SCIENCE=ROOT/'envs/science312/bin/python'


def encoded_image(array, kind):
    stream=io.BytesIO()
    Image.fromarray(array).save(stream,format=kind,**({'quality':80} if kind=='JPEG' else {}))
    return stream.getvalue()


def freshness(stamp, now, limit):
    """Display-only source age; missing, nonfinite and future timestamps are stale."""
    try:
        value = float(stamp)
        if not math.isfinite(value) or value <= 0 or not math.isfinite(now):
            return dict(fresh=False, age_s=None)
        age = now - value
        if age < 0:
            return dict(fresh=False, age_s=age)
        return dict(fresh=age <= limit, age_s=age)
    except (TypeError, ValueError, OverflowError):
        return dict(fresh=False, age_s=None)


class Display:
    def __init__(self, output, historical_seeds=False, no_self_mask=False):
        self.output=output;output.mkdir(parents=True,exist_ok=True)
        self.historical_seeds=historical_seeds
        self.no_self_mask=no_self_mask
        self.lock=threading.RLock();self.stop=threading.Event()
        self.camera_jpeg=self.map_png=None;self.latest_frame=None
        self.last_camera=0.;self.last_source_ns=-1;self.errors=deque(maxlen=4)
        self.trajectory=deque(maxlen=5000)
        self.state=dict(camera=None,map=None,pose=None,trajectory=[],
                        reconstruction=dict(status='starting',point_count=0,frames=0,physical_accuracy_verified=False),
                        robot=dict(status='控制程序未运行'),errors=[])
        self.tfbuf=None

    def camera(self, message):
        if time.monotonic()-self.last_camera<.25:return
        try:
            rgb=rgb_array(message)
            image=encoded_image(rgb,'JPEG')
            with self.lock:
                self.camera_jpeg=image
                self.state['camera']=dict(stamp=message.header.stamp.to_sec(),width=message.width,height=message.height)
                self.last_camera=time.monotonic()
        except Exception as exc:self.errors.append('camera: '+str(exc))

    def map(self,message):
        try:
            data=np.asarray(message.data,dtype=np.int16).reshape(message.info.height,message.info.width)
            gray=np.where(data<0,128,np.clip(245*(1-data/100.),8,245)).astype(np.uint8)
            image=encoded_image(np.flipud(gray),'PNG')
            with self.lock:
                self.map_png=image
                self.state['map']=dict(stamp=message.header.stamp.to_sec(),width=message.info.width,
                    height=message.info.height,resolution=message.info.resolution,
                    origin=[message.info.origin.position.x,message.info.origin.position.y],frame=message.header.frame_id)
        except Exception as exc:self.errors.append('map: '+str(exc))

    def synchronized(self,rgb,depth,info):
        with self.lock:self.latest_frame=(rgb,depth,info)

    def actuator(self,message):
        try:
            row=json.loads(message.data)
            with self.lock:self.state['robot']=dict(status='运行中' if row.get('armed') else '等待行驶',
                stamp=row.get('sampled_unix_s'),fault=row.get('fault'),control_active=True)
        except Exception:pass

    def seed_paths(self):
        # Historical odom poses are valid only for their original session.
        if not self.historical_seeds:return []
        paths=[]
        for name in ('pillar_reference_04','pillar_reference_05','pillar_reference_06','pillar_reference_07','pillar_reference_08'):
            log=ROOT/'reports/nso-hybrid-20261029'/name/'nso/decisions.jsonl'
            if not log.exists():continue
            for line in log.read_text().splitlines():
                row=json.loads(line)
                if row.get('request',{}).get('method')=='observe' and row.get('response',{}).get('ok'):
                    path=Path(row['request']['params']['path'])
                    if path not in paths:paths.append(path)
        return paths

    def reconstruct(self):
        import rospy
        directory=self.output/'reconstruction';directory.mkdir(exist_ok=True)
        log=(directory/'worker.log').open('ab')
        command=[str(SCIENCE),'-s','-B',str(WORK/'runtime/reconstruction_worker.py'),
                 '--output-directory',str(directory)]
        if not self.no_self_mask:command.append('--approved-self-mask')
        process=subprocess.Popen(command,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=bounded_environment())
        self.reconstruction_process=process
        def integrate(path):
            process.stdin.write(json.dumps(dict(method='integrate',path=str(path)))+'\n');process.stdin.flush()
            response=json.loads(process.stdout.readline())
            if not response.get('ok'):raise RuntimeError(response.get('error'))
            with self.lock:
                self.state['reconstruction']=dict(response,frames=response.get('integrated_frames',0))
            return response
        try:
            for path in self.seed_paths():
                if self.stop.is_set():break
                integrate(path)
                with np.load(path,allow_pickle=False) as data:
                    calibration=json.loads(str(data['calibration_json'].item()))
                    pose=data['world_from_camera']@np.linalg.inv(np.asarray(calibration['extrinsics']['base_from_camera']))
                    self.trajectory.append([float(pose[0,3]),float(pose[1,3])])
            while not self.stop.wait(2.):
                with self.lock:row=self.latest_frame
                if row is None:continue
                rgb,depth,info=row;ns=depth.header.stamp.to_nsec()
                if ns==self.last_source_ns:continue
                try:
                    tf=self.tfbuf.lookup_transform('odom',depth.header.frame_id,depth.header.stamp,rospy.Duration(.2))
                    path=self.output/'latest_display_frame.npz'
                    temporary=self.output/'latest_display_frame.tmp.npz'
                    np.savez(temporary,rgb=rgb_array(rgb),depth_m=source_depth_metres(depth),
                        intrinsic=np.asarray(info.K).reshape(3,3),world_from_camera=matrix(tf.transform),
                        stamp_ns=np.int64(ns),frame_id=str(ns),frame='odom',display_only=True)
                    os.replace(temporary,path)
                    integrate(path);self.last_source_ns=ns
                except Exception as exc:self.errors.append('reconstruction: '+str(exc))
        except Exception as exc:
            self.errors.append('reconstruction: '+str(exc))
            with self.lock:self.state['reconstruction']['status']='error'
        finally:
            try:
                if process.poll() is None:
                    process.stdin.write(json.dumps(dict(method='shutdown'))+'\n');process.stdin.flush()
                    process.wait(timeout=4.)
            except Exception:
                if process.poll() is None:process.terminate()
            log.close()

    def snapshot(self):
        import rospy
        try:
            pose_tf=self.tfbuf.lookup_transform('map','base_link',rospy.Time(0),rospy.Duration(.05))
            p=pose_tf.transform.translation;q=pose_tf.transform.rotation
            odom=self.tfbuf.lookup_transform('odom','base_link',rospy.Time(0),rospy.Duration(.05))
            op=odom.transform.translation
            if not self.trajectory or math.hypot(op.x-self.trajectory[-1][0],op.y-self.trajectory[-1][1])>.02:
                self.trajectory.append([op.x,op.y])
            transform=matrix(self.tfbuf.lookup_transform('map','odom',rospy.Time(0),rospy.Duration(.05)).transform)
            trajectory=np.asarray(list(self.trajectory),dtype=float)
            plotted=trajectory@transform[:2,:2].T+transform[:2,3]
            with self.lock:
                self.state['pose']=dict(x=p.x,y=p.y,yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)),
                    frame='map',stamp=pose_tf.header.stamp.to_sec())
                self.state['trajectory']=plotted.tolist()
        except Exception as exc:self.errors.append('pose: '+str(exc))
        with self.lock:
            result=json.loads(json.dumps(self.state))
        if result['robot'].get('stamp',0)<time.time()-2:result['robot']=dict(status='控制程序未运行',control_active=False)
        now=time.time()
        result['freshness']={name:freshness((result.get(name) or {}).get('stamp'),now,limit)
                             for name,limit in (('camera',3.),('pose',3.),('map',10.),('reconstruction',10.))}
        result['errors']=list(self.errors);result['stamp']=now
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--output',default=str(ROOT/'experiments/live_dashboard_20261029'))
    parser.add_argument('--historical-seeds',action='store_true',
                        help='explicitly load 20261029 observations; only valid with that original odom session')
    parser.add_argument('--no-self-mask',action='store_true',
                        help='omit the old IMU mask after camera relocation')
    args=parser.parse_args()
    import rospy,tf2_ros,message_filters
    from sensor_msgs.msg import Image as ROSImage,CameraInfo
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import String
    rospy.init_node('nso_live_dashboard',disable_signals=True)
    display=Display(Path(args.output),args.historical_seeds,args.no_self_mask);display.tfbuf=tf2_ros.Buffer(cache_time=rospy.Duration(30.))
    listener=tf2_ros.TransformListener(display.tfbuf)
    subscribers=[rospy.Subscriber('/zed2i/zed_node/left/image_rect_color',ROSImage,display.camera,queue_size=1,buff_size=8000000),
        rospy.Subscriber('/nso_visual/map',OccupancyGrid,display.map,queue_size=1),
        rospy.Subscriber('/nso_visual/actuator_state',String,display.actuator,queue_size=1)]
    filters=[message_filters.Subscriber(topic,kind,queue_size=2,buff_size=8000000) for topic,kind in
        [('/zed2i/zed_node/left/image_rect_color',ROSImage),('/nso_visual/depth_filtered',ROSImage),
         ('/zed2i/zed_node/depth/camera_info',CameraInfo)]]
    sync=message_filters.TimeSynchronizer(filters,10);sync.registerCallback(display.synchronized)
    worker=threading.Thread(target=display.reconstruct,daemon=True);worker.start()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route=self.path.split('?',1)[0];payload=None;kind='application/json'
            if route=='/api/state':payload=json.dumps(display.snapshot(),allow_nan=False).encode()
            elif route=='/api/camera.jpg':payload=display.camera_jpeg;kind='image/jpeg'
            elif route=='/api/map.png':payload=display.map_png;kind='image/png'
            elif route=='/api/cloud.json':
                path=display.output/'reconstruction/cloud.json'
                if path.exists():payload=path.read_bytes()
            elif route in ('/','/index.html','/dashboard.js','/dashboard.css'):
                path=WORK/'runtime/dashboard'/('index.html' if route=='/' else route.lstrip('/'))
                if path.is_file():payload=path.read_bytes();kind=mimetypes.guess_type(path.name)[0] or 'text/plain'
            if payload is None:self.send_error(503 if route.startswith('/api/') else 404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store');self.end_headers()
            try:self.wfile.write(payload)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    def stop(*_):
        display.stop.set();threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    print(json.dumps(dict(url='http://127.0.0.1:'+str(args.port),read_only=True)),flush=True)
    try:server.serve_forever(poll_interval=.2)
    finally:
        display.stop.set();server.server_close();worker.join(timeout=6.);rospy.signal_shutdown('dashboard ended')


if __name__=='__main__':main()
