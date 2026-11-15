#!/usr/bin/env python3
"""Record original camera frames in a separate process; never controls motion."""
import argparse,json,queue,signal,threading,time
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--fps',type=float,default=5.)
    args=parser.parse_args()
    if not .5<=args.fps<=10:raise ValueError('recording rate must be .5..10 Hz')
    import cv2,numpy as np,rospy
    from sensor_msgs.msg import Image
    from mission import rgb_array
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    frames=output/'frames';frames.mkdir()
    jobs=queue.Queue(maxsize=10);done=threading.Event()
    stats=dict(received=0,recorded=0,dropped=0,first_source_stamp_ns=None,last_source_stamp_ns=None,
               requested_fps=args.fps,source='/zed2i/zed_node/left/image_rect_color',
               physical_accuracy_claimed=False)
    errors=[]
    log=(output/'timestamps.jsonl').open('x')
    writer=None
    def work():
        nonlocal writer
        try:
            while not done.is_set() or not jobs.empty():
                try:msg=jobs.get(timeout=.1)
                except queue.Empty:continue
                a=cv2.cvtColor(rgb_array(msg),cv2.COLOR_RGB2BGR)
                a=cv2.resize(a,(640,360),interpolation=cv2.INTER_AREA)
                ns=msg.header.stamp.to_nsec()
                cv2.imwrite(str(frames/(str(ns)+'.jpg')),a,[cv2.IMWRITE_JPEG_QUALITY,88])
                if writer is None:
                    writer=cv2.VideoWriter(str(output/'camera.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),
                                           args.fps,(640,360))
                    if not writer.isOpened():raise RuntimeError('video encoder unavailable')
                writer.write(a)
                log.write(json.dumps(dict(frame=stats['recorded'],source_stamp_ns=ns,
                                          recorded_unix_s=time.time()))+'\n')
                stats['recorded']+=1
                if stats['first_source_stamp_ns'] is None:stats['first_source_stamp_ns']=ns
                stats['last_source_stamp_ns']=ns
        except Exception as exc:errors.append(str(exc));done.set()
        finally:
            if writer is not None:writer.release()
            log.flush();log.close()
    rospy.init_node('nso_camera_recorder',anonymous=True,disable_signals=True)
    signal.signal(signal.SIGINT,lambda *_:done.set());signal.signal(signal.SIGTERM,lambda *_:done.set())
    last=-1
    def receive(m):
        nonlocal last
        if getattr(m,'_connection_header',{}).get('callerid')!='/zed2i/zed_node':return
        stats['received']+=1
        ns=m.header.stamp.to_nsec()
        if ns<=last or (last>=0 and ns-last<int(1e9/args.fps*.98)):return
        last=ns
        try:jobs.put_nowait(m)
        except queue.Full:stats['dropped']+=1
    thread=threading.Thread(target=work);thread.start()
    sub=rospy.Subscriber(stats['source'],Image,receive,queue_size=1,buff_size=8000000)
    while not done.wait(.1) and not rospy.is_shutdown():pass
    sub.unregister();done.set();thread.join(timeout=5.)
    stats.update(errors=errors,worker_closed=not thread.is_alive(),video_sampling_disclosed=True)
    (output/'result.json').write_text(json.dumps(stats,indent=2)+'\n')
    rospy.signal_shutdown('recording ended')
if __name__=='__main__':main()

